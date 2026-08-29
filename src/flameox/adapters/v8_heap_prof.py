from __future__ import annotations

from flameox.adapters.v8_cpu_prof import (
    _frame_measurements,
    _measurement,
    _registration,
    _run_worker,
)
from flameox.domain import ArtifactKind
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, Workspace


class V8HeapProfExtractionResult(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    sample_count: int
    total_sampled_bytes: int
    frame_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class V8HeapProfExtractor:
    name = "node-heap-prof"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> V8HeapProfExtractionResult:
        registration = _registration(self.workspace, run_id, ArtifactKind.MEMORY_PROFILE, "heap")
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        response = _run_worker(
            self.workspace, artifact.payload_path, registration.artifact_id, "heap"
        )
        published = self.publisher.publish_rows_idempotent(
            {
                "measurements": [
                    _measurement(
                        run_id,
                        registration.artifact_id,
                        "memory.sampled_bytes",
                        response.total_sampled_bytes,
                        "bytes",
                    ),
                    _measurement(
                        run_id,
                        registration.artifact_id,
                        "memory.samples",
                        response.sample_count,
                        "count",
                    ),
                ],
                "frames": list(response.frames),
                "frame_measurements": _frame_measurements(
                    run_id, registration.artifact_id, response
                ),
            },
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={"profile_kind": "heap"},
        )
        return V8HeapProfExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            sample_count=response.sample_count,
            total_sampled_bytes=response.total_sampled_bytes,
            frame_count=len(response.frames),
            corpus_commit_id=published.commit.commit_id,
            limitations=response.limitations,
        )
