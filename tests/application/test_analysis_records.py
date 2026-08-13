from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from flameox.analysis import FailureAnalysisResult, RecipeService
from flameox.application import (
    AcceleratorLaunchAnalysisRequest,
    AnalysisMaterializationService,
    ExecutionAnalysisRequest,
    FailureAnalysisRequest,
    HotspotAnalysisRequest,
    MaterializeAnalysisRequest,
    MemoryAnalysisRequest,
    PyTorchAnalysisRequest,
    ScalingAnalysisRequest,
)
from flameox.catalog import Catalog, Snapshot
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import RetentionIntentStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.serial]

_REQUEST_ADAPTER: TypeAdapter[MaterializeAnalysisRequest] = TypeAdapter(MaterializeAnalysisRequest)


@pytest.mark.parametrize(
    ("payload", "request_type"),
    [
        ({"recipe": "hotspots", "input_id": "run-1"}, HotspotAnalysisRequest),
        ({"recipe": "memory", "input_id": "run-1"}, MemoryAnalysisRequest),
        (
            {"recipe": "execution", "input_id": "run-1", "comparison_input_id": "run-2"},
            ExecutionAnalysisRequest,
        ),
        ({"recipe": "pytorch", "input_id": "run-1"}, PyTorchAnalysisRequest),
        (
            {"recipe": "accelerator_launches", "input_id": "run-1", "phase": "prefill"},
            AcceleratorLaunchAnalysisRequest,
        ),
        ({"recipe": "failures", "limit": 10}, FailureAnalysisRequest),
        ({"recipe": "scaling", "experiment_id": "experiment-1"}, ScalingAnalysisRequest),
    ],
)
def test_materialize_analysis_request_parses_recipe_specific_shape(
    payload: dict[str, object],
    request_type: type[ContractModel],
) -> None:
    assert isinstance(_REQUEST_ADAPTER.validate_python(payload), request_type)


@pytest.mark.parametrize(
    "payload",
    [
        {"input_id": "run-1"},
        {"recipe": "hotspots"},
        {"recipe": "hotspots", "input_id": "run-1", "phase": "prefill"},
        {"recipe": "memory", "input_id": "run-1", "comparison_input_id": "run-2"},
        {"recipe": "failures", "input_id": "run-1"},
        {"recipe": "scaling", "experiment_id": "experiment-1", "limit": 10},
    ],
)
def test_materialize_analysis_request_rejects_cross_recipe_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _REQUEST_ADAPTER.validate_python(payload)


def test_materialized_analysis_retry_reuses_exact_persisted_provenance(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_commit_id = workspace.corpus.read_head().commit_id
    request = FailureAnalysisRequest(
        recipe="failures",
        corpus_commit_id=source_commit_id,
    )
    service = AnalysisMaterializationService(workspace)

    first = service.record(request)
    second = service.record(request)

    assert second == first
    assert first.analysis.corpus_commit_id == source_commit_id
    assert "corpus_commit_id" not in first.analysis.parameters
    assert RetentionIntentStore(workspace).pending() == ()
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM analyses").fetchone() == (1,)


def test_materialized_analysis_failure_keeps_source_commit_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_commit_id = workspace.corpus.read_head().commit_id
    request = FailureAnalysisRequest(
        recipe="failures",
        corpus_commit_id=source_commit_id,
    )

    def fail_publication(*args: object, **kwargs: object) -> None:
        raise RuntimeError("publication failed")

    monkeypatch.setattr(GenerationPublisher, "publish_rows_idempotent", fail_publication)

    with pytest.raises(RuntimeError, match="publication failed"):
        AnalysisMaterializationService(workspace).record(request)

    pending = RetentionIntentStore(workspace).pending()
    assert len(pending) == 1
    assert pending[0].corpus_commit_id == source_commit_id


@pytest.mark.anyio
async def test_materialized_analysis_progress_failure_does_not_block_publication(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_commit_id = workspace.corpus.read_head().commit_id

    async def failed_notification(completed: float, total: float, message: str) -> None:
        raise RuntimeError("progress transport closed")

    result = await AnalysisMaterializationService(workspace).record_async(
        FailureAnalysisRequest(
            recipe="failures",
            corpus_commit_id=source_commit_id,
        ),
        progress=failed_notification,
    )

    with Catalog(workspace).open_snapshot(result.materialized_commit_id) as snapshot:
        assert snapshot.execute(
            "SELECT count(*) FROM analyses WHERE analysis_id = ?",
            (result.analysis.analysis_id,),
        ).fetchone() == (1,)


@pytest.mark.anyio
async def test_materialized_analysis_cancellation_interrupts_duckdb_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    original_head = workspace.corpus.read_head().commit_id
    query_started = threading.Event()
    query_released = threading.Event()
    interrupt_attempted = threading.Event()
    original_failures = RecipeService.failures
    original_interrupt = Snapshot.interrupt

    def tracked_interrupt(snapshot: Snapshot) -> None:
        interrupt_attempted.set()
        original_interrupt(snapshot)

    def slow_failures(
        service: RecipeService,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> FailureAnalysisResult:
        assert service.snapshot is not None
        query_started.set()
        assert query_released.wait(timeout=5)
        service.snapshot.execute("SELECT sum(sin(i)) FROM range(100000000000) values(i)").fetchall()
        return original_failures(
            service,
            limit=limit,
            corpus_commit_id=corpus_commit_id,
        )

    monkeypatch.setattr(RecipeService, "failures", slow_failures)
    monkeypatch.setattr(Snapshot, "interrupt", tracked_interrupt)
    task = asyncio.create_task(
        AnalysisMaterializationService(workspace).record_async(
            FailureAnalysisRequest(recipe="failures")
        )
    )
    assert await asyncio.to_thread(query_started.wait, 5)
    task.cancel()
    assert await asyncio.to_thread(interrupt_attempted.wait, 5)
    query_released.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert workspace.corpus.read_head().commit_id == original_head
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM analyses").fetchone() == (0,)
