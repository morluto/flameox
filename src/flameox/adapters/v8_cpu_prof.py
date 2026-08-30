from __future__ import annotations

from pathlib import Path
from typing import Literal

from flameox.adapters.artifact_workers import IsolatedWorkerHarness
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    DomainError,
    ErrorCode,
    digest_model,
    missing_artifact_input,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace
from flameox.workers.v8_profiles_contract import (
    V8_PROFILE_WORKER,
    V8ProfileRequest,
    V8ProfileResult,
)


class V8CpuProfExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    node_count: int
    sample_count: int
    frame_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class V8CpuProfExtractor:
    name = "node-cpu-prof"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> V8CpuProfExtractionResult:
        registration = _registration(self.workspace, run_id, ArtifactKind.SAMPLE_PROFILE, "CPU")
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        response = _run_worker(
            self.workspace, artifact.payload_path, registration.artifact_id, "cpu"
        )
        measurement_rows = _frame_measurements(run_id, registration.artifact_id, response)
        published = self.publisher.publish_rows_idempotent(
            {
                "measurements": [
                    _measurement(
                        run_id,
                        registration.artifact_id,
                        "cpu.samples",
                        response.sample_count,
                        "count",
                    ),
                    _measurement(
                        run_id, registration.artifact_id, "cpu.nodes", response.node_count, "count"
                    ),
                ],
                "frames": list(response.frames),
                "frame_measurements": measurement_rows,
            },
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={"profile_kind": "cpu"},
        )
        return V8CpuProfExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            node_count=response.node_count,
            sample_count=response.sample_count,
            frame_count=len(response.frames),
            corpus_commit_id=published.commit.commit_id,
            limitations=response.limitations,
        )


def _registration(
    workspace: Workspace,
    run_id: str,
    kind: ArtifactKind,
    label: str,
) -> ArtifactRegistration:
    run = RunStore(workspace).read(run_id)
    adapter = "node-cpu-prof" if kind is ArtifactKind.SAMPLE_PROFILE else "node-heap-prof"
    registrations = [item for item in run.artifacts if item.kind is kind]
    if not registrations:
        raise missing_artifact_input(
            run_id=run_id,
            requirement=f"V8 {label} profile",
            artifact_kinds=(kind.value,),
            capture_adapters=(adapter,),
            import_producers=("auto",),
        )
    if len(registrations) != 1:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"The run must contain exactly one V8 {label} profile artifact.",
            run_id=run_id,
        )
    return registrations[0]


def _run_worker(
    workspace: Workspace,
    artifact_path: Path,
    artifact_id: str,
    profile_kind: Literal["cpu", "heap"],
) -> V8ProfileResult:
    maximum = workspace.config.storage.max_rows_per_generation
    response_budget = workspace.config.execution.max_output_bytes
    max_rows = min(maximum - 2, max(2, (response_budget - 64 * 1024) // 2_048))
    if maximum < 4 or max_rows < 2:
        raise DomainError(
            ErrorCode.QUERY_BUDGET_EXCEEDED,
            "V8 profile extraction requires room for bounded frame rows.",
        )
    return IsolatedWorkerHarness(workspace).run_typed_sync(
        V8_PROFILE_WORKER,
        V8ProfileRequest(
            profile_kind=profile_kind,
            artifact_path=str(artifact_path),
            artifact_id=artifact_id,
            project_root=str(workspace.project_root),
            max_nodes=min(100_000, max_rows // 2),
            max_samples=min(1_000_000, maximum),
            max_rows=max_rows,
        ),
    )


def _measurement(
    run_id: str,
    artifact_id: str,
    name: str,
    value: int,
    unit: str,
) -> dict[str, object]:
    return {
        "measurement_id": digest_model(
            {"run_id": run_id, "artifact_id": artifact_id, "name": name}
        ),
        "run_id": run_id,
        "artifact_id": artifact_id,
        "name": name,
        "value_int": value,
        "value_float": None,
        "unit": unit,
        "aggregation": "total",
        "scope": "process",
        "trial_id": None,
        "worker_id": None,
        "worker_run_index": None,
        "value_index": None,
        "loop_count": None,
        "is_warmup": False,
        "block_id": None,
        "variant_id": None,
        "order_in_block": None,
        "phase": None,
        "dimensions": {},
        "evidence_level": "observed",
    }


def _frame_measurements(
    run_id: str,
    artifact_id: str,
    response: V8ProfileResult,
) -> list[dict[str, object]]:
    return [
        {
            **dict(row),
            "run_id": run_id,
            "artifact_id": artifact_id,
            "thread_name": None,
            "process_name": None,
            "phase": None,
        }
        for row in response.frame_measurements
    ]
