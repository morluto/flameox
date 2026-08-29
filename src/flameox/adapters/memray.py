from __future__ import annotations

import shutil
from collections.abc import Awaitable, Callable
from contextlib import suppress
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
from flameox.workers.memray_contract import (
    MEMRAY_EXTRACTOR_NAME,
    MEMRAY_EXTRACTOR_VERSION,
    MEMRAY_WORKER,
    MemrayExtractionCoverage,
    MemrayExtractionLimits,
    MemrayWorkerProgress,
    MemrayWorkerRequest,
    MemrayWorkerResult,
)

_WORKER_PROTOCOL_OVERHEAD_BYTES = 2 * 1024 * 1024


def memray_extraction_limits(workspace: Workspace) -> MemrayExtractionLimits:
    maximum_rows = workspace.config.storage.max_rows_per_generation
    return MemrayExtractionLimits(
        max_input_bytes=workspace.config.capture.max_artifact_bytes,
        max_provider_records=100_000,
        max_frames=max(1, min(250_000, maximum_rows // 3)),
        max_stack_depth=256,
        max_aggregate_rows=max(1, min(500_000, maximum_rows // 2)),
        max_output_bytes=min(
            workspace.config.storage.max_staging_bytes,
            workspace.config.capture.max_artifact_bytes,
        ),
        wall_time_seconds=workspace.config.capture.default_timeout_seconds,
        max_worker_memory_bytes=workspace.config.execution.max_memory_bytes,
    )


class _MemrayProgressReader:
    def __init__(
        self,
        harness: IsolatedWorkerHarness,
        emit: Callable[[str, int, int | None], Awaitable[None]] | None,
    ) -> None:
        self.harness = harness
        self.emit = emit
        self.last: MemrayWorkerProgress | None = None

    async def __call__(self, job_root: Path) -> None:
        if self.emit is None:
            return
        with suppress(DomainError, OSError, ValueError):
            payload = self.harness.read_staged_bytes(
                job_root,
                "progress.json",
                max_bytes=4_096,
            )
            if payload is None:
                return
            current = MemrayWorkerProgress.model_validate_json(payload)
            if current != self.last:
                self.last = current
                await self.emit(current.phase, current.records_seen, None)


class MemrayExtractionResult(ContractModel):
    run_id: str
    artifact_id: str
    producer_version: str
    reader_version: str
    reader_environment_id: str
    extractor_profile: str
    peak_memory_bytes: int
    retained_end_bytes: int
    allocation_operations: int | None
    total_allocated_bytes: int | None
    capture_records: int
    limits: MemrayExtractionLimits
    coverage: MemrayExtractionCoverage
    evidence_generation_id: str
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class MemrayExtractor:
    name = MEMRAY_EXTRACTOR_NAME
    version = MEMRAY_EXTRACTOR_VERSION

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)
        self.provider_runtimes = ProviderRuntimeManager(
            workspace.paths.records / "provider-runtimes"
        )

    async def extract(
        self,
        run_id: str,
        *,
        limits: MemrayExtractionLimits,
        progress: Callable[[str, int, int | None], Awaitable[None]] | None = None,
    ) -> MemrayExtractionResult:
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
        if artifact.payload_path.stat().st_size > limits.max_input_bytes:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Memray capture exceeds the extraction input-byte limit.",
                run_id=run_id,
                details={"limits": limits.model_dump(mode="json")},
                remediation=(
                    "Increase the workspace capture artifact limit or import a smaller profile.",
                ),
            )
        harness = IsolatedWorkerHarness(self.workspace, python=runtime.python)
        worker_results: list[MemrayWorkerResult] = []
        expected_reader_version = runtime.receipt.distributions["memray"]

        heartbeat = _MemrayProgressReader(harness, progress)

        async def prepare(
            root: Path,
            generation_id: str,
            published_at: datetime,
        ) -> dict[str, Path]:
            if progress is not None:
                await progress("reading_profile", 0, None)

            def consume(result: MemrayWorkerResult, job_root: Path) -> dict[str, Path]:
                if result.reader_version != expected_reader_version:
                    raise DomainError(
                        ErrorCode.ADAPTER_INCOMPATIBLE,
                        "The Memray worker reader does not match its runtime receipt.",
                        details={
                            "expected_reader_version": expected_reader_version,
                            "observed_reader_version": result.reader_version,
                        },
                    )
                self._validate_worker_coverage(result, limits)
                worker_results.append(result)
                return self._stage_worker_outputs(harness, result, job_root, root)

            with self.provider_runtimes.verified_use(runtime):
                return await harness.run_typed_session(
                    MEMRAY_WORKER,
                    MemrayWorkerRequest(
                        artifact_path=str(artifact.payload_path),
                        run_id=run_id,
                        artifact_id=registration.artifact_id,
                        workload_cwd=(
                            str(self.workspace.project_root / run.semantics.scope.workload_cwd)
                            if run.semantics.scope.workload_cwd is not None
                            else None
                        ),
                        project_root=str(self.workspace.project_root),
                        source_state_id=(
                            run.source_state_id
                            if run.semantics.scope.workload_cwd is not None
                            else None
                        ),
                        generation_id=generation_id,
                        published_at=published_at,
                        limits=limits,
                    ),
                    consume=consume,
                    timeout_seconds=limits.wall_time_seconds,
                    maximum_rss_bytes=limits.max_worker_memory_bytes,
                    maximum_writable_growth_bytes=(
                        limits.max_output_bytes + _WORKER_PROTOCOL_OVERHEAD_BYTES
                    ),
                    heartbeat=heartbeat,
                )

        operation_digest = digest_model(
            {
                "artifact_id": registration.artifact_id,
                "producer_version": producer_version,
                "reader_environment_id": runtime.receipt.environment_id,
                "reader_version": expected_reader_version,
                "extractor_profile": MEMRAY_WORKER.implementation,
                "limits": limits.model_dump(mode="json"),
            }
        )
        try:
            published = await self.publisher.publish_prepared_parquet(
                prepare,
                publisher=MEMRAY_EXTRACTOR_NAME,
                publisher_version=MEMRAY_EXTRACTOR_VERSION,
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
        if progress is not None:
            await progress("publishing_evidence", 1, 1)
        worker = worker_results[-1]
        limitations = [
            "Frame aggregates expose bounded callers; complete stacks remain in Memray.",
            *runtime.receipt.limitations,
        ]
        if not worker.has_native_traces:
            limitations.append("The capture does not contain native stack traces.")
        if worker.allocation_operations is None:
            limitations.append(
                "Memray structured allocation statistics are unavailable for this capture format; "
                "only the raw capture-record count is published."
            )
        if not worker.coverage.complete:
            limitations.append(
                "Memray normalized evidence reached an extraction limit; coverage reports the "
                "selected records and dropped contributions. Capture a narrower profile or raise "
                "applicable workspace budgets before starting a new extraction."
            )
        return MemrayExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            producer_version=producer_version,
            reader_version=worker.reader_version,
            reader_environment_id=runtime.receipt.environment_id,
            extractor_profile=MEMRAY_WORKER.implementation,
            peak_memory_bytes=worker.peak_memory_bytes,
            retained_end_bytes=worker.retained_end_bytes,
            allocation_operations=worker.allocation_operations,
            total_allocated_bytes=worker.total_allocated_bytes,
            capture_records=worker.capture_records,
            limits=limits,
            coverage=worker.coverage,
            evidence_generation_id=published.manifest.generation_id,
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _validate_worker_coverage(
        result: MemrayWorkerResult,
        limits: MemrayExtractionLimits,
    ) -> None:
        coverage = result.coverage
        if (
            coverage.frames_published > limits.max_frames
            or coverage.aggregate_rows_published > limits.max_aggregate_rows
            or coverage.output_bytes > limits.max_output_bytes
            or coverage.output_bytes != sum(output.byte_length for output in result.files)
        ):
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "The Memray worker coverage exceeds or contradicts its extraction limits.",
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
