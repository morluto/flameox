from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import Field

from flameox.analysis.inference_protocol import InferenceProtocolIdentity, ProfilerState
from flameox.application.evidence_query import EvidenceQueryService
from flameox.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
    source_state_row,
)
from flameox.application.imports import ImportArtifactRequest, ImportService
from flameox.application.inference import InferenceReplayService
from flameox.application.inference_providers import _loopback_http_url
from flameox.application.run_rows import run_row
from flameox.application.source import collect_partial_source_state
from flameox.application.workloads import WorkloadService
from flameox.atomic import atomic_write_bytes
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    CommandSpec,
    DomainError,
    EnvironmentRecord,
    ErrorCode,
    ExecutionStatus,
    ProcessResult,
    RunManifest,
    RunType,
    Sensitivity,
    SourceState,
    ValidationStatus,
    digest_model,
    new_id,
)
from flameox.domain.models import utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace


class InferenceProfilingPlan(ContractModel):
    schema_version: Literal[1] = 1
    plan_id: str
    server_name: str
    profiler: Literal["torch_profiler", "nsight_systems"]
    base_url: str
    server_argv: tuple[str, ...]
    server_cwd: Path
    environment_names: tuple[str, ...]
    environment_digest: str
    output_path: Path
    nsys_executable: Path | None = None
    configuration_id: str
    server_executable_digest: str | None = None
    server_version: str | None = None
    diagnostic_only: Literal[True] = True
    limitations: tuple[str, ...]


class InferenceProfilingResult(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    plan_id: str
    measurement_protocol_id: str
    measurement_run_id: str
    scenario_name: str
    profiler: Literal["torch_profiler", "nsight_systems"]
    benchmark_exit_code: int | None
    server_cleanup_complete: bool
    artifact_ids: tuple[str, ...]
    artifact_run_ids: tuple[str, ...]
    extracted_run_ids: tuple[str, ...] = ()
    coverage: Literal["complete", "partial", "unavailable"]
    limitations: tuple[str, ...]


class InferenceProfilingService:
    """Build bounded diagnostic capture plans for Flameox-managed vLLM servers."""

    def __init__(self, workspace: Workspace, *, broker: SubprocessBroker | None = None) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.broker = broker or SubprocessBroker()
        self.artifacts = ArtifactStore(workspace)
        self.runs = RunStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    def plan(
        self,
        server_name: str,
        *,
        profiler: Literal["torch_profiler", "nsight_systems"],
        nsys_executable: Path | None = None,
        expected_plan_id: str | None = None,
    ) -> InferenceProfilingPlan:
        project = self.workloads.load()
        try:
            server = project.inference_servers[server_name]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Inference server {server_name!r} is not declared.",
                remediation=("List or configure a managed inference server, then retry.",),
                details={"server": server_name, "next_tool": "list_inference_configurations"},
            ) from exc
        if server.mode != "managed" or server.workload is None:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Inference profiling requires a Flameox-managed server workload.",
            )
        workload = self.workloads.resolve(server.workload)
        replay = InferenceReplayService(self.workspace, broker=self.broker)
        server_executable_digest, server_version = replay._server_tool_identity(server)
        native_argv = workload.command.argv
        output_root = self.workspace.paths.staging / f"inference-profile-{server_name}" / new_id()
        environment = dict(workload.command.env_overrides)
        limitations: tuple[str, ...]
        if profiler == "torch_profiler":
            output_path = output_root / "torch"
            environment["VLLM_TORCH_PROFILER_DIR"] = str(output_path)
            argv = native_argv
            limitations = (
                "Diagnostic profile; measurements must come from a separate unprofiled run.",
            )
        else:
            if nsys_executable is None:
                discovered_nsys = shutil.which("nsys")
                nsys_executable = Path(discovered_nsys) if discovered_nsys is not None else None
            if (
                nsys_executable is None
                or not nsys_executable.is_file()
                or not os.access(nsys_executable, os.X_OK)
            ):
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "Nsight Systems profiling requires an installed nsys executable.",
                )
            output_path = output_root / "capture.nsys-rep"
            argv = (
                str(nsys_executable.resolve()),
                "profile",
                "--trace-fork-before-exec=true",
                "--cuda-graph-trace=node",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=repeat",
                "--output",
                str(output_path.with_suffix("")),
                *native_argv,
                "--profiler-config.profiler",
                "cuda",
            )
            limitations = (
                "Diagnostic profile; measurements must come from a separate unprofiled run.",
                "The native .nsys-rep must be exported with official nsys export "
                "before extraction.",
            )
        environment_names = set(environment)
        if profiler == "torch_profiler":
            environment_names.add("VLLM_TORCH_PROFILER_DIR")
        environment_digest = digest_model(workload.command.env_overrides)
        identity = {
            "server": server.model_dump(mode="json"),
            "profiler": profiler,
            "native_argv": native_argv,
            "cwd": workload.command.cwd,
            "environment_digest": environment_digest,
            "nsys_executable": str(nsys_executable.resolve()) if nsys_executable else None,
            "nsys_executable_digest": (
                replay._executable_digest(nsys_executable) if nsys_executable else None
            ),
            "configuration_id": digest_model(project.model_dump(mode="json")),
        }
        plan_id = digest_model(identity)
        if expected_plan_id is not None and expected_plan_id != plan_id:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The inference profiling plan no longer matches the reviewed plan identity.",
                remediation=("Plan the inference profile again and review the replacement.",),
                details={"expected_plan_id": expected_plan_id, "actual_plan_id": plan_id},
            )
        return InferenceProfilingPlan(
            plan_id=plan_id,
            server_name=server_name,
            profiler=profiler,
            base_url=server.base_url,
            server_argv=argv,
            server_cwd=Path(workload.command.cwd),
            environment_names=tuple(sorted(environment_names)),
            environment_digest=environment_digest,
            output_path=output_path,
            nsys_executable=nsys_executable,
            configuration_id=digest_model(project.model_dump(mode="json")),
            server_executable_digest=server_executable_digest,
            server_version=server_version,
            limitations=limitations,
        )

    async def capture(  # noqa: C901 - one lifecycle boundary owns every terminal transition
        self,
        plan: InferenceProfilingPlan,
        *,
        scenario_name: str,
        measurement_run_id: str,
        timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300,
    ) -> InferenceProfilingResult:
        """Capture one small diagnostic window under a single absolute deadline."""

        replay = InferenceReplayService(self.workspace, broker=self.broker)
        project = self.workloads.load()
        if digest_model(project.model_dump(mode="json")) != plan.configuration_id:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference configuration changed after this profiling plan was created.",
                remediation=("Plan the inference profile again, then retry capture.",),
            )
        server = project.inference_servers.get(plan.server_name)
        if server is None:
            raise DomainError(ErrorCode.REVISION_CONFLICT, "Planned inference server was removed.")
        assert server.workload is not None
        server_environment = dict(self.workloads.resolve(server.workload).command.env_overrides)
        if digest_model(server_environment) != plan.environment_digest:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Managed server environment identity changed after profiling was planned.",
                remediation=("Plan the inference profile again, then retry capture.",),
            )
        if plan.profiler == "torch_profiler":
            server_environment["VLLM_TORCH_PROFILER_DIR"] = str(plan.output_path)
        server_digest, server_version = replay._server_tool_identity(server)
        if server_digest != plan.server_executable_digest or server_version != plan.server_version:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Managed server executable identity changed after profiling was planned.",
                remediation=("Plan the inference profile again, then retry capture.",),
            )
        replay_plan = replay.plan(scenario_name, timeout_seconds=timeout_seconds)
        if replay_plan.server_name != plan.server_name or replay_plan.server_mode != "managed":
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Profiling scenario must target the planned managed server.",
            )
        if not replay_plan.tool_available:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The inference workload provider is unavailable for the profiling window.",
                remediation=replay_plan.tool_remediation,
            )
        environment = await replay._managed_environment(replay_plan)
        measurement_protocol = replay._protocol_identity(replay_plan, environment=environment)
        measurement_protocol_id = digest_model(measurement_protocol.model_dump(mode="json"))
        self._validate_measurement_run(measurement_run_id, measurement_protocol_id)
        diagnostic_protocol = measurement_protocol.model_copy(
            update={"profiler": ProfilerState(profiler=plan.profiler, attached=True)}
        )
        replay_output = Path(replay_plan.output_path or self.workspace.paths.staging)
        replay_output.parent.mkdir(parents=True, exist_ok=True)
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)
        deadline_at = replay_plan.deadline_at
        parsed = urlsplit(plan.base_url)
        assert parsed.hostname is not None
        port = parsed.port or 80

        async def readiness() -> bool:
            from flameox.application.inference_providers import probe_existing_vllm_server

            try:
                await asyncio.to_thread(
                    probe_existing_vllm_server,
                    plan.base_url,
                    timeout_seconds=0.5,
                )
            except DomainError:
                return False
            return True

        remaining = (deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            raise DomainError(ErrorCode.PROCESS_TIMEOUT, "Profiling startup deadline expired.")
        server_request = ExecutionRequest(
            argv=plan.server_argv,
            cwd=plan.server_cwd,
            environment_allowlist=self.workspace.config.execution.child_environment_allowlist,
            environment_overrides=server_environment,
            allowed_working_roots=(self.workspace.project_root, plan.output_path.parent),
            timeout_seconds=remaining,
            max_output_bytes=16 * 1024 * 1024,
        )
        run, source_state = self._start_run(
            plan,
            timeout_seconds=timeout_seconds,
            environment=environment,
            source_protocol_id=measurement_protocol_id,
            diagnostic_protocol=diagnostic_protocol,
            measurement_run_id=measurement_run_id,
        )
        try:
            lease = await self.broker.start_inference_server(
                server_request,
                host=parsed.hostname,
                port=port,
                readiness=readiness,
                absolute_deadline=time.monotonic() + remaining,
            )
            limitations = list(run.limitations)
            benchmark_exit_code: int | None = None
            benchmark_process = None
            control = VllmProfilerControlClient(plan.base_url)
            try:
                try:
                    control.timeout_seconds = min(
                        control.timeout_seconds,
                        max(0.01, (deadline_at - utc_now()).total_seconds()),
                    )
                    await asyncio.to_thread(control.start)
                    remaining = (deadline_at - utc_now()).total_seconds()
                    if remaining <= 0:
                        raise DomainError(ErrorCode.PROCESS_TIMEOUT, "Profiling deadline expired.")
                    outcome = await self.broker.run(
                        ExecutionRequest(
                            argv=replay_plan.argv,
                            cwd=self.workspace.project_root,
                            environment_allowlist=(
                                self.workspace.config.execution.child_environment_allowlist
                            ),
                            allowed_working_roots=(
                                self.workspace.project_root,
                                Path(
                                    replay_plan.output_path or self.workspace.paths.staging
                                ).parent,
                            ),
                            timeout_seconds=remaining,
                            max_output_bytes=16 * 1024 * 1024,
                        )
                    )
                    benchmark_process = outcome.process
                    benchmark_exit_code = outcome.process.exit_code
                    if outcome.process.exit_code != 0:
                        limitations.append(
                            f"Profile workload exited with status {outcome.process.exit_code}."
                        )
                except DomainError as error:
                    limitations.append(f"Profile window failed: {error.message}")
            finally:
                stop_remaining = (deadline_at - utc_now()).total_seconds()
                if stop_remaining <= 0:
                    limitations.append("No deadline remained for profiler flushing.")
                else:
                    control.timeout_seconds = min(5.0, stop_remaining)
                    try:
                        await asyncio.to_thread(control.stop)
                    except DomainError as error:
                        limitations.append(error.message)
                server_outcome = await asyncio.shield(lease.close())

            if server_outcome.stdout:
                atomic_write_bytes(plan.output_path.parent / "server.stdout", server_outcome.stdout)
            if server_outcome.stderr:
                atomic_write_bytes(plan.output_path.parent / "server.stderr", server_outcome.stderr)

            if server_outcome.process.cleanup_complete is not True:
                limitations.append("Managed server process cleanup was incomplete.")

            if plan.profiler == "nsight_systems" and plan.output_path.is_file():
                await self._export_nsight(plan, deadline_at, limitations)
            artifacts, runs, preservation_limitations = self._preserve(plan)
            limitations.extend(preservation_limitations)
            extracted_runs = await self._extract_preserved(plan, runs, deadline_at, limitations)
            if artifacts and not extracted_runs:
                limitations.append(
                    "No recognized profiler trace was extracted from preserved artifacts."
                )
            operational_limitations = limitations[len(run.limitations) :]
            coverage: Literal["complete", "partial", "unavailable"] = (
                "complete"
                if extracted_runs and not operational_limitations
                else ("partial" if artifacts else "unavailable")
            )
            finished = self._finish_run(
                run,
                environment,
                source_state,
                artifact_ids=artifacts,
                artifact_run_ids=runs,
                process=benchmark_process,
                coverage=coverage,
                limitations=tuple(limitations),
            )
            cleanup_limitation = self._cleanup_staging(
                plan, preservation_complete=not preservation_limitations
            )
            if cleanup_limitation is not None:
                limitations.append(cleanup_limitation)
                finished = self._append_limitation(
                    finished,
                    environment,
                    source_state,
                    artifacts,
                    cleanup_limitation,
                )
            return InferenceProfilingResult(
                run_id=finished.run_id,
                plan_id=plan.plan_id,
                measurement_protocol_id=measurement_protocol_id,
                measurement_run_id=measurement_run_id,
                scenario_name=scenario_name,
                profiler=plan.profiler,
                benchmark_exit_code=benchmark_exit_code,
                server_cleanup_complete=server_outcome.process.cleanup_complete is True,
                artifact_ids=artifacts,
                artifact_run_ids=runs,
                extracted_run_ids=extracted_runs,
                coverage=coverage,
                limitations=tuple(limitations),
            )
        except asyncio.CancelledError:
            artifacts, artifact_runs, _preservation_limitations = self._preserve(plan)
            self._finish_cancelled_run(run, environment, source_state, artifacts, artifact_runs)
            raise
        except DomainError as error:
            artifacts, artifact_runs, _preservation_limitations = self._preserve(plan)
            self._finish_failed_run(run, environment, source_state, error, artifacts, artifact_runs)
            error.run_id = run.run_id
            if artifacts:
                error.details = {
                    **error.details,
                    "partial_artifact_ids": artifacts,
                    "partial_artifact_run_ids": artifact_runs,
                }
            raise
        except Exception as cause:
            artifacts, artifact_runs, _preservation_limitations = self._preserve(plan)
            internal_error = DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Unexpected inference profiling failure.",
                run_id=run.run_id,
            )
            self._finish_failed_run(
                run, environment, source_state, internal_error, artifacts, artifact_runs
            )
            raise internal_error from cause

    def _start_run(
        self,
        plan: InferenceProfilingPlan,
        *,
        timeout_seconds: float,
        environment: EnvironmentRecord,
        source_protocol_id: str,
        diagnostic_protocol: InferenceProtocolIdentity,
        measurement_run_id: str,
    ) -> tuple[RunManifest, SourceState]:
        """Create the diagnostic run before server startup can mutate external state."""

        executable = Path(plan.server_argv[0])
        if not executable.is_absolute():
            executable = plan.server_cwd / executable
        source_state = collect_partial_source_state(
            self.workspace,
            executable=executable if executable.is_file() else None,
        )
        diagnostic_json = json.dumps(
            diagnostic_protocol.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        diagnostic_id = digest_model(diagnostic_protocol.model_dump(mode="json"))
        run = RunManifest(
            run_id=new_id(),
            run_type=RunType.EXECUTION,
            started_at=utc_now(),
            execution_status=ExecutionStatus.RUNNING,
            capture_status=CaptureStatus.RUNNING,
            validation_status=ValidationStatus.UNSUPPORTED,
            workload_definition_id=source_protocol_id,
            workload_instance_id=plan.plan_id,
            measurement_protocol_id=source_protocol_id,
            source_measurement_run_id=measurement_run_id,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            collector=plan.profiler,
            command=CommandSpec(
                argv=plan.server_argv,
                cwd=str(plan.server_cwd),
                env_overrides={name: "<redacted>" for name in plan.environment_names},
                timeout_seconds=timeout_seconds,
            ),
            inference_protocol_identity_id=diagnostic_id,
            inference_protocol_identity_json=diagnostic_json,
            limitations=(
                *plan.limitations,
                f"Diagnostic profile linked to measurement run {measurement_run_id}.",
            ),
        )
        self.runs.create(run)
        self._publish_run(run, environment, source_state, ())
        return run, source_state

    def _validate_measurement_run(
        self, measurement_run_id: str, measurement_protocol_id: str
    ) -> None:
        measurement_run = self.runs.read(measurement_run_id)
        if measurement_run.execution_status is not ExecutionStatus.SUCCEEDED:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run must have completed successfully.",
                run_id=measurement_run_id,
                details={"next_tool": "plan_inference_scenario"},
            )
        if (
            measurement_run.capture_status is not CaptureStatus.REGISTERED
            or not measurement_run.artifacts
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run has no preserved provider evidence.",
                run_id=measurement_run_id,
                remediation=("Run the unprofiled inference scenario successfully, then retry.",),
                details={"next_tool": "plan_inference_scenario"},
            )
        if (
            EvidenceQueryService(self.workspace)
            .measurements(run_id=measurement_run_id, limit=1)
            .total
            == 0
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run has no published measurements.",
                run_id=measurement_run_id,
                remediation=("Run the unprofiled inference scenario successfully, then retry.",),
                details={"next_tool": "plan_inference_scenario"},
            )
        if measurement_run.measurement_protocol_id != measurement_protocol_id:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run does not match the profiling workload protocol.",
                run_id=measurement_run_id,
                details={
                    "expected_measurement_protocol_id": measurement_protocol_id,
                    "actual_measurement_protocol_id": measurement_run.measurement_protocol_id,
                    "next_tool": "plan_inference_scenario",
                },
            )
        try:
            identity = InferenceProtocolIdentity.model_validate_json(
                measurement_run.inference_protocol_identity_json or ""
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked run has no valid inference protocol identity.",
                run_id=measurement_run_id,
                details={"next_tool": "plan_inference_scenario"},
            ) from exc
        if identity.profiler.attached:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run must be unprofiled.",
                run_id=measurement_run_id,
                details={"next_tool": "plan_inference_scenario"},
            )

    def _finish_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        *,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
        process: ProcessResult | None,
        coverage: Literal["complete", "partial", "unavailable"],
        limitations: tuple[str, ...],
    ) -> RunManifest:
        status = (
            ExecutionStatus.TIMED_OUT
            if process is not None and process.timed_out
            else ExecutionStatus.SUCCEEDED
            if process is not None and process.exit_code == 0 and coverage == "complete"
            else ExecutionStatus.FAILED
        )
        registrations = self._canonical_registrations(run.run_id, artifact_run_ids)
        finished = run.model_copy(
            update={
                "revision": 1,
                "finished_at": utc_now(),
                "execution_status": status,
                "capture_status": (
                    CaptureStatus.REGISTERED if artifact_ids else CaptureStatus.FAILED
                ),
                "process": process,
                "artifacts": registrations,
                "limitations": tuple(dict.fromkeys(limitations)),
            }
        )
        self.runs.append(finished, expected_revision=0)
        self._publish_run(finished, environment, source_state, artifact_ids)
        return finished

    def _finish_cancelled_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
    ) -> RunManifest:
        registrations = self._canonical_registrations(run.run_id, artifact_run_ids)
        finished = run.model_copy(
            update={
                "revision": 1,
                "finished_at": utc_now(),
                "execution_status": ExecutionStatus.CANCELLED,
                "capture_status": (
                    CaptureStatus.REGISTERED if artifact_ids else CaptureStatus.CANCELLED
                ),
                "artifacts": registrations,
                "limitations": tuple(
                    dict.fromkeys((*run.limitations, "Inference profiling was cancelled."))
                ),
            }
        )
        self.runs.append(finished, expected_revision=0)
        self._publish_run(finished, environment, source_state, artifact_ids)
        return finished

    def _finish_failed_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        error: DomainError,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
    ) -> RunManifest:
        registrations = self._canonical_registrations(run.run_id, artifact_run_ids)
        finished = run.model_copy(
            update={
                "revision": 1,
                "finished_at": utc_now(),
                "execution_status": (
                    ExecutionStatus.TIMED_OUT
                    if error.code is ErrorCode.PROCESS_TIMEOUT
                    else ExecutionStatus.FAILED
                ),
                "capture_status": (
                    CaptureStatus.REGISTERED if artifact_ids else CaptureStatus.FAILED
                ),
                "artifacts": registrations,
                "limitations": tuple(dict.fromkeys((*run.limitations, error.message))),
            }
        )
        self.runs.append(finished, expected_revision=0)
        self._publish_run(finished, environment, source_state, artifact_ids)
        return finished

    def _canonical_registrations(
        self, run_id: str, artifact_run_ids: tuple[str, ...]
    ) -> tuple[ArtifactRegistration, ...]:
        registrations: list[ArtifactRegistration] = []
        for artifact_run_id in artifact_run_ids:
            source = self.runs.read(artifact_run_id)
            registrations.extend(
                registration.model_copy(update={"registration_id": new_id(), "run_id": run_id})
                for registration in source.artifacts
            )
        return tuple(registrations)

    def _append_limitation(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        artifact_ids: tuple[str, ...],
        limitation: str,
    ) -> RunManifest:
        updated = run.model_copy(
            update={
                "revision": run.revision + 1,
                "limitations": tuple(dict.fromkeys((*run.limitations, limitation))),
            }
        )
        self.runs.append(updated, expected_revision=run.revision)
        self._publish_run(updated, environment, source_state, artifact_ids)
        return updated

    def _publish_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        artifact_ids: tuple[str, ...],
    ) -> None:
        self.publisher.publish_rows(
            {
                "runs": [run_row(run)],
                "artifact_registrations": [
                    artifact_registration_row(
                        registration,
                        byte_length=self.artifacts.get(
                            registration.artifact_id
                        ).content.byte_length,
                    )
                    for registration in run.artifacts
                ],
                "environments": [environment_row(environment)],
                "source_states": [source_state_row(source_state)],
            },
            publisher="flameox.inference_profiling",
            publisher_version="1",
            input_run_ids=tuple(
                dict.fromkeys(
                    (run.run_id,)
                    + (
                        (run.source_measurement_run_id,)
                        if run.source_measurement_run_id is not None
                        else ()
                    )
                )
            ),
            input_artifact_ids=artifact_ids,
        )

    async def _export_nsight(
        self,
        plan: InferenceProfilingPlan,
        deadline_at: datetime,
        limitations: list[str],
    ) -> None:
        if plan.nsys_executable is None:
            limitations.append("Nsight export executable identity is unavailable.")
            return
        remaining = (deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            limitations.append("No deadline remained for Nsight SQLite export.")
            return
        sqlite_path = plan.output_path.with_suffix(".sqlite")
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(
                        str(plan.nsys_executable),
                        "export",
                        "--type",
                        "sqlite",
                        "--output",
                        str(sqlite_path),
                        str(plan.output_path),
                    ),
                    cwd=self.workspace.project_root,
                    environment_allowlist=(
                        self.workspace.config.execution.child_environment_allowlist
                    ),
                    allowed_working_roots=(
                        self.workspace.project_root,
                        plan.output_path.parent,
                    ),
                    timeout_seconds=remaining,
                    max_output_bytes=4 * 1024 * 1024,
                )
            )
            if outcome.process.exit_code != 0:
                limitations.append(
                    f"Nsight SQLite export exited with status {outcome.process.exit_code}."
                )
        except DomainError as error:
            limitations.append(f"Nsight SQLite export failed: {error.message}")

    def _preserve(
        self, plan: InferenceProfilingPlan
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        root = plan.output_path if plan.output_path.is_dir() else plan.output_path.parent
        if not root.is_dir():
            return (), (), ()
        candidates, discovery_limitations = InferenceReplayService._bounded_output_candidates(root)
        artifacts: list[str] = []
        runs: list[str] = []
        limitations = list(discovery_limitations)
        importer = ImportService(self.workspace)
        for path in candidates:
            trace_file = path.name.endswith((".json", ".json.gz", ".pftrace", ".sqlite"))
            server_output = path.name.startswith("server.")
            try:
                imported = importer.import_artifact(
                    ImportArtifactRequest(
                        path=path,
                        kind=(
                            ArtifactKind.EXECUTION_TRACE
                            if trace_file
                            else ArtifactKind.PROCESS_OUTPUT
                            if server_output
                            else ArtifactKind.COLLECTOR_METADATA
                        ),
                        sensitivity=(
                            Sensitivity.SENSITIVE if server_output else Sensitivity.INTERNAL
                        ),
                        role=("inference_server_output" if server_output else "inference_profile"),
                        producer="nsys" if plan.profiler == "nsight_systems" else "torch_profiler",
                        allow_external_path=True,
                    )
                )
            except DomainError as error:
                limitations.append(
                    f"Profiler artifact {path.name!r} could not be preserved: {error.message}"
                )
                continue
            artifacts.append(imported.artifact_id)
            runs.append(imported.run.run_id)
        return tuple(artifacts), tuple(runs), tuple(limitations)

    async def _extract_preserved(
        self,
        plan: InferenceProfilingPlan,
        run_ids: tuple[str, ...],
        deadline_at: datetime,
        limitations: list[str],
    ) -> tuple[str, ...]:
        """Feed supported native traces through existing bounded extractors."""
        extracted: list[str] = []
        run_store = RunStore(self.workspace)
        for run_id in run_ids:
            run = run_store.read(run_id)
            traces = [item for item in run.artifacts if item.kind is ArtifactKind.EXECUTION_TRACE]
            if len(traces) != 1:
                continue
            name = traces[0].display_name
            remaining = (deadline_at - utc_now()).total_seconds()
            if remaining <= 0:
                limitations.append("No deadline remained for profile evidence extraction.")
                break
            try:
                if plan.profiler == "nsight_systems" and name.endswith(".sqlite"):
                    from flameox.adapters.nsight_systems import NsightSystemsExtractor

                    await asyncio.wait_for(
                        NsightSystemsExtractor(self.workspace, broker=self.broker).extract(run_id),
                        timeout=remaining,
                    )
                elif plan.profiler == "torch_profiler" and name.endswith(
                    (".json", ".json.gz", ".pftrace")
                ):
                    from flameox.adapters.perfetto import PerfettoExtractor

                    await asyncio.wait_for(
                        PerfettoExtractor(self.workspace, broker=self.broker).extract(run_id),
                        timeout=remaining,
                    )
                else:
                    continue
            except TimeoutError:
                limitations.append(f"Profile extraction deadline expired for {name}.")
            except DomainError as error:
                limitations.append(f"Profile extraction failed for {name}: {error.message}")
            else:
                extracted.append(run_id)
        return tuple(extracted)

    def _cleanup_staging(
        self, plan: InferenceProfilingPlan, *, preservation_complete: bool
    ) -> str | None:
        if not preservation_complete:
            return (
                "Profiler staging was retained because native artifact preservation was incomplete."
            )
        root = plan.output_path.parent.absolute()
        staging = self.workspace.paths.staging.resolve()
        if root.is_symlink():
            return "Profiler staging cleanup was refused because the operation path is a symlink."
        try:
            resolved_root = root.resolve(strict=True)
        except FileNotFoundError:
            return None
        if resolved_root == staging or not resolved_root.is_relative_to(staging):
            return "Profiler staging cleanup was refused because the path was not operation-owned."
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            return None
        except OSError:
            return "Profiler staging cleanup failed; immutable artifacts remain authoritative."
        return None


class VllmProfilerControlClient:
    """Bounded client for vLLM's start/stop profiler control endpoints."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 5.0) -> None:
        self.base_url = _loopback_http_url(base_url)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def start(self) -> None:
        self._post("/start_profile")

    def stop(self) -> None:
        self._post("/stop_profile")

    def _post(self, path: Literal["/start_profile", "/stop_profile"]) -> None:
        try:
            with urlopen(
                Request(f"{self.base_url}{path}", data=b"", method="POST"),
                timeout=self.timeout_seconds,
            ) as response:
                if not 200 <= response.status < 300:
                    raise OSError(f"profiler control returned HTTP {response.status}")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                f"vLLM profiler control failed at {path}.",
            ) from exc
