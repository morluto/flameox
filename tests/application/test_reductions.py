from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from flameox.application import (
    BinaryChunkReductionPlan,
    BinaryChunkReductionRequest,
    ImportArtifactRequest,
    ImportService,
    NativeReductionPartitioner,
    PlanReductionRequest,
    ReductionAttemptSummary,
    ReductionDeterminism,
    ReductionLimits,
    ReductionPlan,
    ReductionResult,
    ReductionService,
    StructuredReductionPlan,
    StructuredReductionRequest,
)
from flameox.domain import ArtifactKind
from flameox.storage import Workspace


def _configure(project: Path, predicate_code: str) -> None:
    (project / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.predicate]
argv = ["python", "-c", {json.dumps(predicate_code)}]
cwd = "."
timeout_seconds = 30
"""
    )


def test_reduction_result_rejects_unknown_revalidation_classification() -> None:
    with pytest.raises(ValidationError):
        ReductionResult.model_validate(
            {
                "reduction_id": "reduction",
                "plan_id": "plan",
                "disposition": "inconclusive",
                "original_artifact_id": "sha256:" + "a" * 64,
                "predicate_definition_id": "predicate",
                "predicate_instance_id": "instance",
                "attempts": ReductionAttemptSummary(
                    attempted=0,
                    passed=0,
                    failed=0,
                    contradictory=0,
                    timed_out=0,
                ),
                "cleanup_complete": True,
                "partitioner": "text_lines",
                "final_revalidation_status": "unknown",
            }
        )


def test_reduction_request_partitioner_variants_encode_chunk_size_requirement() -> None:
    request: TypeAdapter[PlanReductionRequest] = TypeAdapter(PlanReductionRequest)
    common = {"original_artifact_id": "sha256:" + "0" * 64, "predicate_workload": "predicate"}

    binary = request.validate_python(
        {**common, "partitioner": "binary_chunks", "chunk_size": 4_096}
    )
    structured = request.validate_python({**common, "partitioner": "text_lines"})

    assert isinstance(binary, BinaryChunkReductionRequest)
    assert isinstance(structured, StructuredReductionRequest)

    with pytest.raises(ValidationError, match="chunk_size"):
        request.validate_python({**common, "partitioner": "binary_chunks"})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        request.validate_python({**common, "partitioner": "text_lines", "chunk_size": 4_096})


def test_reduction_plan_parser_preserves_schema_two_wire_variants() -> None:
    adapter: TypeAdapter[ReductionPlan] = TypeAdapter(ReductionPlan)
    common = {
        "schema_version": 2,
        "plan_id": "sha256:" + "2" * 64,
        "request_digest": "sha256:" + "2" * 64,
        "workspace_id": "workspace",
        "original_artifact_id": "sha256:" + "0" * 64,
        "predicate_workload": "predicate",
        "predicate_definition_id": "definition",
        "predicate_instance_id": "instance",
        "predicate_command": {"argv": ["python"], "cwd": "."},
        "predicate_parameters": {},
        "predicate_executable_digest": "sha256:" + "1" * 64,
        "limits": {},
        "expected_determinism": "deterministic",
        "created_at": "2026-01-01T00:00:00Z",
    }

    binary = adapter.validate_python(
        {**common, "partitioner": "binary_chunks", "chunk_size": 4_096}
    )
    structured = adapter.validate_python(
        {**common, "partitioner": "text_lines", "chunk_size": None}
    )

    assert isinstance(binary, BinaryChunkReductionPlan)
    assert isinstance(structured, StructuredReductionPlan)
    assert structured.expected_determinism is ReductionDeterminism.DETERMINISTIC

    with pytest.raises(ValidationError, match="request digest"):
        adapter.validate_python(
            {
                **common,
                "request_digest": "sha256:" + "3" * 64,
                "partitioner": "text_lines",
                "chunk_size": None,
            }
        )
    assert structured.model_dump(mode="json")["chunk_size"] is None

    with pytest.raises(ValidationError, match="chunk_size"):
        adapter.validate_python({**common, "partitioner": "binary_chunks"})
    with pytest.raises(ValidationError, match="Input should be None"):
        adapter.validate_python({**common, "partitioner": "text_lines", "chunk_size": 4_096})


def _original(workspace: Workspace, path: Path, content: str = "discard\nKEEP\ndiscard\n") -> str:
    path.write_text(content)
    return (
        ImportService(workspace)
        .import_artifact(ImportArtifactRequest(path=path, kind=ArtifactKind.COLLECTOR_METADATA))
        .artifact_id
    )


def _plan(workspace: Workspace, artifact_id: str) -> ReductionPlan:
    return ReductionService(workspace).plan(
        StructuredReductionRequest(
            original_artifact_id=artifact_id,
            partitioner=NativeReductionPartitioner.TEXT_LINES,
            predicate_workload="predicate",
            limits=ReductionLimits(predicate_repetitions=2),
            expected_determinism=ReductionDeterminism.REPEATED,
        )
    )


def test_reduction_plan_store_parses_the_persisted_variant(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "raise SystemExit(0)")
    artifact_id = _original(workspace, tmp_path / "binary.bin", "abcdefgh")
    service = ReductionService(workspace)

    planned = service.plan(
        BinaryChunkReductionRequest(
            original_artifact_id=artifact_id,
            partitioner=NativeReductionPartitioner.BINARY_CHUNKS,
            chunk_size=4,
            predicate_workload="predicate",
        )
    )
    stored = service.plans.read(planned.plan_id)

    assert isinstance(planned, BinaryChunkReductionPlan)
    assert isinstance(stored, BinaryChunkReductionPlan)
    assert stored.chunk_size == 4
    assert stored == planned


@pytest.mark.anyio
async def test_native_reduction_revalidates_final_candidate(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if 'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_text() else 1)",
    )
    service = ReductionService(workspace)
    plan = _plan(workspace, _original(workspace, tmp_path / "original.txt"))

    assert plan.schema_version == 2
    assert plan.engine == "native_ddmin"
    result = await service.execute(plan.plan_id)

    assert result.disposition == "succeeded"
    assert result.final_artifact_id is not None
    assert result.final_artifact_id != result.original_artifact_id
    assert result.attempts.attempted >= 2
    assert result.attempts.passed >= 1
    assert result.final_revalidation_status == "interesting"
    assert result.minimality == "one_minimal"
    assert result.cleanup_complete is True
    assert service.get(result.reduction_id) == result


@pytest.mark.anyio
async def test_native_reduction_publishes_empty_final_artifact(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "raise SystemExit(0)")
    original_id = _original(workspace, tmp_path / "original.txt", "remove-me\n")
    result = await ReductionService(workspace).execute(_plan(workspace, original_id).plan_id)

    assert result.disposition == "succeeded"
    assert result.final_unit_count == 0
    assert result.final_artifact_id is not None
    assert result.final_artifact_id != original_id
    final_path = ReductionService(workspace).artifacts.get(result.final_artifact_id).payload_path
    assert final_path.read_bytes() == b""


@pytest.mark.anyio
async def test_concurrent_native_execution_reconnects_to_one_result(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "raise SystemExit(0)")
    artifact_id = _original(workspace, tmp_path / "original.txt")
    plan = _plan(workspace, artifact_id)

    first, second = await asyncio.gather(
        ReductionService(workspace).execute(plan.plan_id),
        ReductionService(workspace).execute(plan.plan_id),
    )

    assert first == second


@pytest.mark.anyio
async def test_native_reduction_does_not_read_original_bytes_in_application_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if 'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_text() else 1)",
    )
    artifact_id = _original(workspace, tmp_path / "worker-original.txt")
    service = ReductionService(workspace)
    original_path = service.artifacts.get(artifact_id).payload_path
    original_read_bytes = Path.read_bytes

    def refuse_original_read(path: Path) -> bytes:
        if path == original_path:
            raise AssertionError("application process read the whole reduction artifact")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_original_read)

    result = await service.execute(_plan(workspace, artifact_id).plan_id)

    assert result.disposition == "succeeded"
    assert result.final_artifact_id is not None


@pytest.mark.anyio
async def test_waiter_retries_after_reduction_worker_owner_is_cancelled(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "raise SystemExit(0)")
    plan = _plan(workspace, _original(workspace, tmp_path / "retry-original.txt"))
    first = asyncio.create_task(ReductionService(workspace).execute(plan.plan_id))
    await asyncio.sleep(0.1)
    second = asyncio.create_task(ReductionService(workspace).execute(plan.plan_id))
    await asyncio.sleep(0.1)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    result = await asyncio.wait_for(second, timeout=8)

    assert result.disposition == "succeeded"


@pytest.mark.anyio
async def test_predicate_timeout_is_reported_without_killing_reduction_worker(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "import time; time.sleep(10)")
    artifact_id = _original(workspace, tmp_path / "timeout-original.txt")
    plan = ReductionService(workspace).plan(
        StructuredReductionRequest(
            original_artifact_id=artifact_id,
            partitioner=NativeReductionPartitioner.TEXT_LINES,
            predicate_workload="predicate",
            limits=ReductionLimits(
                max_attempts=2,
                wall_time_seconds=0.5,
                predicate_timeout_seconds=0.05,
            ),
        )
    )

    result = await ReductionService(workspace).execute(plan.plan_id)

    assert result.disposition == "inconclusive"
    assert result.attempts.attempted == 1
    assert result.attempts.timed_out == 1
