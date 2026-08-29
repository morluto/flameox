from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from packaging.version import InvalidVersion, Version

from flameox.action_graph import ActionId, tool_action
from flameox.adapters.artifact_workers import IsolatedWorkerHarness
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.domain import (
    ArtifactKind,
    CapabilityExtra,
    DomainError,
    ErrorCode,
    digest_model,
    missing_artifact_input,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace
from flameox.workers.memray_contract import MEMRAY_WORKER, MemrayWorkerRequest, MemrayWorkerResult


class MemrayExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    producer_version: str
    reader_version: str
    reader_environment_id: str
    extractor_profile: str
    peak_memory_bytes: int
    retained_end_bytes: int
    total_allocations: int
    frame_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class MemrayExtractor:
    name = "memray"
    version = "2"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)
        self.provider_runtimes = ProviderRuntimeManager(
            workspace.paths.records / "provider-runtimes"
        )

    def extract(
        self,
        run_id: str,
        *,
        cancel_check: Callable[[], None] | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
    ) -> MemrayExtractionResult:
        self._check_cancelled(cancel_check)
        run = RunStore(self.workspace).read(run_id)
        registrations = [item for item in run.artifacts if item.kind is ArtifactKind.MEMORY_PROFILE]
        if not registrations:
            raise missing_artifact_input(
                run_id=run_id,
                requirement="Memray memory-profile",
                artifact_kinds=(ArtifactKind.MEMORY_PROFILE.value,),
                capture_adapters=("memray",),
                import_producers=("memray",),
            )
        if len(registrations) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one Memray artifact.",
                run_id=run_id,
            )
        registration = registrations[0]
        producer_version = self._producer_version(
            registration.producer,
            registration.producer_version,
        )
        requirement = f"memray=={producer_version}"
        runtime = self.provider_runtimes.find_distribution(
            extra=CapabilityExtra.MEMORY,
            requirement=requirement,
        )
        if runtime is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "No verified Memray reader matches the artifact producer.",
                run_id=run_id,
                details={
                    "producer_version": producer_version,
                    "required_reader": requirement,
                },
                remediation=(
                    "Call start_capability_setup with adapters=['memray'] and "
                    f"memray_reader_version='{producer_version}', then retry extraction.",
                ),
                next_action=tool_action(
                    ActionId.START_CAPABILITY_SETUP,
                    adapters=["memray"],
                    idempotency_key=f"memray-reader-{producer_version}",
                    memray_reader_version=producer_version,
                ),
            )
        artifact = ArtifactStore(self.workspace).get(registration.artifact_id)
        harness = IsolatedWorkerHarness(self.workspace, python=runtime.python)
        worker_results: list[MemrayWorkerResult] = []
        expected_reader_version = runtime.receipt.distributions["memray"]

        def prepare(
            root: Path,
            generation_id: str,
            published_at: datetime,
        ) -> dict[str, Path]:
            if progress is not None:
                progress("reading_profile", 0, None)
            with (
                self.provider_runtimes.verified_use(runtime),
                harness.run_typed_sync_session(
                    MEMRAY_WORKER,
                    MemrayWorkerRequest(
                        artifact_path=str(artifact.payload_path),
                        run_id=run_id,
                        artifact_id=registration.artifact_id,
                        project_root=str(
                            Path(run.command.cwd)
                            if run.command is not None
                            else self.workspace.project_root
                        ),
                        generation_id=generation_id,
                        published_at=published_at,
                    ),
                ) as (result, job_root),
            ):
                if result.reader_version != expected_reader_version:
                    raise DomainError(
                        ErrorCode.ADAPTER_INCOMPATIBLE,
                        "The Memray worker reader does not match its runtime receipt.",
                        details={
                            "expected_reader_version": expected_reader_version,
                            "observed_reader_version": result.reader_version,
                        },
                    )
                staged = self._stage_worker_outputs(harness, result, job_root, root)
                worker_results.append(result)
            self._check_cancelled(cancel_check)
            return staged

        operation_digest = digest_model(
            {
                "artifact_id": registration.artifact_id,
                "producer_version": producer_version,
                "reader_environment_id": runtime.receipt.environment_id,
                "reader_version": expected_reader_version,
                "extractor_profile": MEMRAY_WORKER.implementation,
            }
        )
        try:
            published = self.publisher.publish_prepared_parquet(
                prepare,
                publisher=self.name,
                publisher_version=self.version,
                input_run_ids=(run_id,),
                input_artifact_ids=(registration.artifact_id,),
                operation_digest=operation_digest,
            )
        except DomainError as error:
            if error.code not in {ErrorCode.ADAPTER_INCOMPATIBLE, ErrorCode.ARTIFACT_PARSE_FAILED}:
                raise
            raise DomainError(
                error.code,
                error.message,
                run_id=run_id,
                details={
                    "producer_version": producer_version,
                    "reader_version": expected_reader_version,
                    "reader_environment_id": runtime.receipt.environment_id,
                },
                remediation=(
                    "Prepare or select a reader runtime qualified for this exact producer, "
                    "then retry extraction against the preserved native artifact.",
                ),
                next_action=tool_action(
                    ActionId.START_CAPABILITY_SETUP,
                    adapters=["memray"],
                    idempotency_key=f"memray-reader-{producer_version}",
                    memray_reader_version=producer_version,
                ),
            ) from error
        self._check_cancelled(cancel_check)
        if progress is not None:
            progress("publishing_evidence", 1, 1)
        worker = worker_results[-1]
        limitations = [
            "Frame aggregates expose bounded callers; complete stacks remain in Memray.",
            *runtime.receipt.limitations,
        ]
        if not worker.has_native_traces:
            limitations.append("The capture does not contain native stack traces.")
        return MemrayExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            producer_version=producer_version,
            reader_version=worker.reader_version,
            reader_environment_id=runtime.receipt.environment_id,
            extractor_profile=MEMRAY_WORKER.implementation,
            peak_memory_bytes=worker.peak_memory_bytes,
            retained_end_bytes=worker.retained_end_bytes,
            total_allocations=worker.total_allocations,
            frame_count=worker.frame_count,
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _stage_worker_outputs(
        harness: IsolatedWorkerHarness,
        result: MemrayWorkerResult,
        job_root: Path,
        staging_root: Path,
    ) -> dict[str, Path]:
        expected = {
            "measurements": "measurements.parquet",
            "frames": "frames.parquet",
            "frame_measurements": "frame_measurements.parquet",
        }
        actual_roles = [output.role for output in result.files]
        if len(set(actual_roles)) != len(actual_roles) or set(actual_roles) != set(expected):
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "The Memray worker returned an invalid evidence table set.",
            )
        staged: dict[str, Path] = {}
        for output in result.files:
            if (
                output.relative_path != expected[output.role]
                or output.media_type != "application/vnd.apache.parquet"
            ):
                raise DomainError(
                    ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                    "The Memray worker output does not match its table contract.",
                )
            source = harness.validate_output_file(job_root, output)
            destination = staging_root / source.name
            shutil.copyfile(source, destination)
            staged[output.role] = destination
        return staged

    @staticmethod
    def _producer_version(producer: str | None, version: str | None) -> str:
        if producer is None or producer.casefold() != "memray" or version is None:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "Memray extraction requires declared Memray producer identity and version.",
            )
        try:
            return str(Version(version))
        except InvalidVersion as error:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "The declared Memray producer version is invalid.",
                details={"producer_version": version},
            ) from error

    @staticmethod
    def _check_cancelled(cancel_check: Callable[[], None] | None) -> None:
        if cancel_check is not None:
            cancel_check()
