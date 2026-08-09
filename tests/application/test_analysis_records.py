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
from flameox.models import ContractModel
from flameox.storage import Workspace

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
