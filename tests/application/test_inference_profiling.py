from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
import pytest

from flameox.application.inference import InferenceReplayService
from flameox.application.inference_profiling import (
    InferenceProfilerControlClient,
    InferenceProfilingPlan,
    InferenceProfilingService,
    NsightSystemsProfilingPlan,
    SglangProfileOptions,
    SglangTorchProfilingPlan,
    VllmProfilerControlClient,
    VllmTorchProfilingPlan,
)
from flameox.application.inference_providers import (
    AvailableInferenceToolDiscovery,
    InferenceServerProvider,
    InferenceTool,
    InferenceToolDiscovery,
)
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ProcessResult,
    Sensitivity,
    new_id,
    process_termination_from_returncode,
)
from flameox.domain.models import ArtifactKind, ArtifactRegistration, utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ManagedSidecarLease,
    ManagedSidecarOutcome,
    ProcessContainment,
    SubprocessBroker,
)
from flameox.http_transport import BoundedHttpClient
from flameox.storage import RunStore, Workspace
from tests.support.execution import executable_binding

pytestmark = pytest.mark.unit


def _workspace(tmp_path: Path, *, mode: str = "managed") -> Workspace:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "serve.py").write_text("print('serve')")
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        '[workloads.serve]\nargv = ["python", "serve.py"]\n'
        '[inference_servers.local]\nprovider = "vllm"\n'
        f'mode = "{mode}"\n'
        + ('workload = "serve"\n' if mode == "managed" else "")
        + 'base_url = "http://127.0.0.1:8000"\nmodel = "model"\n'
        + (
            "[inference_scenarios.profile]\n"
            'server = "local"\nprovider = "aiperf"\n'
            'endpoint_type = "chat"\nstreaming = true\nnum_prompts = 1\n'
            if mode == "managed"
            else ""
        )
    )
    return workspace


class _FakeLease:
    def __init__(self, *, cleanup_complete: bool = True) -> None:
        self.cleanup_complete = cleanup_complete

    async def close(self) -> ManagedSidecarOutcome:
        now = datetime.now(UTC)
        return ManagedSidecarOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(0),
                cleanup_complete=self.cleanup_complete,
            ),
            stdout=b"",
            stderr=b"",
            containment=ProcessContainment.PROCESS_GROUP,
            process_observations=(),
            started_at=now,
            finished_at=now,
        )


class _ProfilingBroker(SubprocessBroker):
    def __init__(
        self,
        trace_path: Path,
        *,
        startup_error: DomainError | None = None,
        cancel_window: bool = False,
        cleanup_complete: bool = True,
    ) -> None:
        self.trace_path = trace_path
        self.startup_error = startup_error
        self.cancel_window = cancel_window
        self.cleanup_complete = cleanup_complete

    async def start_inference_server(
        self,
        request: ExecutionRequest,
        *,
        host: str,
        port: int,
        readiness: Callable[[], Awaitable[bool]],
        absolute_deadline: float,
    ) -> ManagedSidecarLease:
        del request, host, port, readiness, absolute_deadline
        if self.startup_error is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text('{"traceEvents": []}')
            raise self.startup_error
        return cast(
            ManagedSidecarLease,
            _FakeLease(cleanup_complete=self.cleanup_complete),
        )

    async def run(self, request: ExecutionRequest, **_kwargs: object) -> ExecutionOutcome:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text('{"traceEvents": []}')
        if self.cancel_window:
            raise asyncio.CancelledError
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(0),
                cleanup_complete=True,
                wall_time_ns=1_234,
            ),
            stdout=b"",
            stderr=b"",
            resolved_executable=Path(request.argv[0]),
            executable_binding=request.executable_binding,
            containment=ProcessContainment.PROCESS_GROUP,
        )


def _patch_capture_dependencies(monkeypatch: pytest.MonkeyPatch, executable: Path) -> None:
    from flameox.application import inference as inference_module

    def fake_discover(
        tool: InferenceTool,
        *,
        provider_runtime: object | None = None,
    ) -> InferenceToolDiscovery:
        del provider_runtime
        return AvailableInferenceToolDiscovery(
            tool=tool,
            executable=executable,
            executable_binding=executable_binding(executable),
            version="0.12.0",
            executable_digest="sha256:" + "a" * 64,
            available=True,
        )

    async def fake_environment(_self: object, _plan: object) -> object:
        from flameox.application.environment import collect_environment

        return collect_environment()

    async def fake_extract(
        _self: object,
        _plan: object,
        run_ids: tuple[str, ...],
        _deadline_at: object,
        _limitations: list[str],
    ) -> tuple[str, ...]:
        return run_ids

    monkeypatch.setattr(inference_module, "discover_inference_tool", fake_discover)
    monkeypatch.setattr(
        "flameox.application.inference.InferenceReplayService._managed_environment",
        fake_environment,
    )
    monkeypatch.setattr(InferenceProfilingService, "_extract_preserved", fake_extract)

    async def control_noop(_self: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(VllmProfilerControlClient, "start_async", control_noop)
    monkeypatch.setattr(VllmProfilerControlClient, "stop_async", control_noop)


async def _seed_measurement_run(workspace: Workspace) -> str:
    replay = InferenceReplayService(workspace)
    replay_plan = replay.plan("profile", timeout_seconds=5)
    environment = await replay._managed_environment(replay_plan)
    run, measurement_environment, source_state = replay._start_run(
        replay_plan, environment=environment
    )
    evidence_path = workspace.project_root / "measurement.json"
    evidence_path.write_text("{}")
    stored = replay.artifacts.import_path(
        evidence_path,
        allowed_roots=(workspace.project_root,),
        max_bytes=workspace.config.capture.max_artifact_bytes,
    )
    registration = ArtifactRegistration(
        registration_id=new_id(),
        run_id=run.run_id,
        artifact_id=stored.content.artifact_id,
        display_name=evidence_path.name,
        media_type="application/json",
        kind=ArtifactKind.INFERENCE_RESULT,
        role="inference_provider_output",
        producer="aiperf",
        sensitivity=Sensitivity.SENSITIVE,
    )
    finished = run.model_copy(
        update={
            "revision": 1,
            "finished_at": utc_now(),
            "execution_status": ExecutionStatus.SUCCEEDED,
            "capture_status": CaptureStatus.REGISTERED,
            "artifacts": (registration,),
        }
    )
    replay.projections.append_run(
        finished,
        expected_revision=0,
        environment=measurement_environment,
        source_state=source_state,
    )
    GenerationPublisher(workspace).publish_rows(
        {
            "measurements": [
                {
                    "measurement_id": new_id(),
                    "run_id": run.run_id,
                    "artifact_id": stored.content.artifact_id,
                    "name": "aiperf.request_throughput",
                    "value_int": None,
                    "value_float": 1.0,
                    "unit": "requests/s",
                    "aggregation": "rate",
                    "scope": "workload",
                    "trial_id": None,
                    "worker_id": None,
                    "worker_run_index": None,
                    "value_index": None,
                    "loop_count": None,
                    "is_warmup": False,
                    "block_id": None,
                    "variant_id": None,
                    "order_in_block": None,
                    "phase": "steady_state",
                    "dimensions": {},
                    "evidence_level": "derived",
                }
            ]
        },
        publisher="test.inference_measurement",
        publisher_version="1",
        input_run_ids=(run.run_id,),
        input_artifact_ids=(stored.content.artifact_id,),
    )
    return run.run_id


async def _authorized_profile(
    service: InferenceProfilingService,
    workspace: Workspace,
) -> tuple[InferenceProfilingPlan, str]:
    measurement_run_id = await _seed_measurement_run(workspace)
    plan = service.plan(
        "local",
        profiler="torch_profiler",
        scenario_name="profile",
        measurement_run_id=measurement_run_id,
        timeout_seconds=5,
    )
    return plan, measurement_run_id


def test_torch_profile_plan_uses_managed_workload_and_trace_directory(tmp_path: Path) -> None:
    service = InferenceProfilingService(_workspace(tmp_path))
    plan = service.plan("local", profiler="torch_profiler")
    second = service.plan("local", profiler="torch_profiler")

    assert isinstance(plan, VllmTorchProfilingPlan)
    assert plan.diagnostic_only is True
    assert "VLLM_TORCH_PROFILER_DIR" in plan.environment_names
    assert plan.environment_digest.startswith("sha256:")
    assert plan.server_argv[-1] == "serve.py"
    assert plan.output_path.parent != second.output_path.parent
    assert plan.plan_id == second.plan_id


def test_sglang_torch_plan_has_stable_identity_and_derived_profile_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace.initialize(tmp_path)
    launcher = tmp_path / "sglang-python"
    launcher.write_text("launcher")
    launcher.chmod(0o755)
    (tmp_path / "serve.py").write_text("print('serve')")
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        '[workloads.serve]\nargv = ["python", "serve.py"]\n'
        "[inference_servers.local]\n"
        'provider = "sglang"\n'
        f'benchmark_python = "{launcher}"\n'
        'mode = "managed"\nworkload = "serve"\n'
        'base_url = "http://127.0.0.1:8000"\nmodel = "model"\n'
        "[inference_scenarios.profile]\n"
        'server = "local"\nprovider = "sglang_bench"\n'
        "random_input_len = 4\nrandom_output_len = 2\n"
    )

    discovery = AvailableInferenceToolDiscovery(
        tool=InferenceTool.SGLANG,
        executable=launcher,
        executable_binding=executable_binding(launcher),
        available=True,
        version="0.5.16",
        executable_digest="sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        "flameox.application.inference_profiling.discover_sglang",
        lambda _launcher, *, broker: discovery,
    )

    service = InferenceProfilingService(workspace)
    first = service.plan("local", profiler="torch_profiler")
    second = service.plan("local", profiler="torch_profiler")

    assert isinstance(first, SglangTorchProfilingPlan)
    assert first.plan_id == second.plan_id
    assert first.sglang_profile_id == second.sglang_profile_id
    assert first.sglang_profile_id == f"flameox-{first.plan_id[7:31]}"
    assert first.benchmark_executable_digest == discovery.executable_digest


def test_nsight_plan_wraps_server_with_documented_cuda_capture_range(tmp_path: Path) -> None:
    nsys = tmp_path / "nsys"
    nsys.write_text("#!/bin/sh\n")
    nsys.chmod(0o755)

    plan = InferenceProfilingService(_workspace(tmp_path)).plan(
        "local", profiler="nsight_systems", nsys_executable=nsys
    )

    assert isinstance(plan, NsightSystemsProfilingPlan)
    assert plan.nsys_executable == nsys
    assert plan.server_argv[:3] == (str(nsys), "profile", "--trace-fork-before-exec=true")
    assert "--capture-range=cudaProfilerApi" in plan.server_argv
    assert "--resolve-symbols=false" in plan.server_argv
    assert plan.symbol_resolution == "disabled"
    assert plan.server_argv[-2:] == ("--profiler-config.profiler", "cuda")


@pytest.mark.anyio
async def test_nsight_capture_reports_noninteractive_phase_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    executable = tmp_path / "aiperf"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    nsys = tmp_path / "nsys"
    nsys.write_text("#!/bin/sh\n")
    nsys.chmod(0o755)
    _patch_capture_dependencies(monkeypatch, executable)
    measurement_run_id = await _seed_measurement_run(workspace)
    service = InferenceProfilingService(workspace)
    plan = service.plan(
        "local",
        profiler="nsight_systems",
        nsys_executable=nsys,
        scenario_name="profile",
        measurement_run_id=measurement_run_id,
        timeout_seconds=5,
    )
    service.broker = _ProfilingBroker(plan.output_path)

    result = await service.capture(plan.plan_token)

    assert result.symbol_resolution_status == "disabled"
    assert result.symbol_resolution_duration_ns is None
    assert result.workload_duration_ns == 1_234
    assert result.finalization_duration_ns >= 0
    assert result.export_duration_ns is not None
    assert result.export_duration_ns >= 0
    run = RunStore(workspace).read(result.run_id)
    assert run.semantics.configuration["symbol_resolution"] == "disabled"


def test_existing_local_server_cannot_be_profiled(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as caught:
        InferenceProfilingService(_workspace(tmp_path, mode="existing_local")).plan(
            "local", profiler="torch_profiler"
        )

    assert caught.value.code is ErrorCode.EXECUTION_REFUSED


def test_profiler_control_uses_only_start_and_stop_endpoints() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(204, stream=httpx.ByteStream(b""))

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as http_client:
        client = VllmProfilerControlClient(
            "http://127.0.0.1:8000",
            http_client=http_client,
        )
        client.start()
        client.stop()

    assert calls == [
        ("POST", "http://127.0.0.1:8000/start_profile"),
        ("POST", "http://127.0.0.1:8000/stop_profile"),
    ]


def test_sglang_profiler_control_posts_only_fixed_profile_payload(tmp_path: Path) -> None:
    payloads: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append((str(request.url), request.content))
        return httpx.Response(204, stream=httpx.ByteStream(b""))

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as http_client:
        client = InferenceProfilerControlClient(
            "http://127.0.0.1:8000",
            provider=InferenceServerProvider.SGLANG,
            http_client=http_client,
        )
        client.start(
            output_dir=tmp_path / "traces",
            profile_id="profile-id",
            options=SglangProfileOptions(),
        )
        client.stop()

    assert json.loads(payloads[0][1]) == {
        "output_dir": str(tmp_path / "traces"),
        "profile_id": "profile-id",
        "start_step": 5,
        "num_steps": 2,
        "activities": ["CPU", "GPU"],
        "profile_by_stage": True,
        "record_shapes": True,
        "with_stack": True,
    }
    assert payloads[1] == ("http://127.0.0.1:8000/stop_profile", b"")


def test_torch_profile_preserves_compressed_trace_for_perfetto(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    service = InferenceProfilingService(workspace)
    plan = service.plan("local", profiler="torch_profiler")
    plan.output_path.mkdir(parents=True, exist_ok=True)
    (plan.output_path / "worker.pt.trace.json.gz").write_bytes(b"trace")

    _artifacts, run_ids, _limitations = service._preserve(plan)

    run = RunStore(workspace).read(run_ids[0])
    assert run.artifacts[0].kind is ArtifactKind.EXECUTION_TRACE
    assert run.artifacts[0].producer == "torch_profiler"


@pytest.mark.anyio
async def test_torch_profile_feeds_preserved_trace_to_perfetto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    service = InferenceProfilingService(workspace)
    plan = service.plan("local", profiler="torch_profiler")
    plan.output_path.mkdir(parents=True, exist_ok=True)
    (plan.output_path / "worker.pt.trace.json.gz").write_bytes(b"trace")
    _artifacts, run_ids, _limitations = service._preserve(plan)
    extracted: list[str] = []

    async def fake_extract(_self: object, run_id: str) -> object:
        extracted.append(run_id)
        return object()

    monkeypatch.setattr("flameox.adapters.perfetto.PerfettoExtractor.extract", fake_extract)
    limitations: list[str] = []

    result = await service._extract_preserved(
        plan,
        run_ids,
        utc_now() + timedelta(seconds=5),
        limitations,
    )

    assert result == run_ids
    assert extracted == list(run_ids)
    assert limitations == []


@pytest.mark.anyio
async def test_nsight_profile_feeds_only_sqlite_export_to_existing_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    nsys = tmp_path / "nsys"
    nsys.write_text("#!/bin/sh\n")
    nsys.chmod(0o755)
    service = InferenceProfilingService(workspace)
    plan = service.plan("local", profiler="nsight_systems", nsys_executable=nsys)
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    plan.output_path.write_bytes(b"native")
    plan.output_path.with_suffix(".sqlite").write_bytes(b"SQLite format 3\0")
    _artifacts, run_ids, _limitations = service._preserve(plan)
    extracted: list[str] = []

    async def fake_extract(_self: object, run_id: str) -> object:
        extracted.append(run_id)
        return object()

    monkeypatch.setattr(
        "flameox.adapters.nsight_systems.NsightSystemsExtractor.extract",
        fake_extract,
    )
    limitations: list[str] = []

    result = await service._extract_preserved(
        plan,
        run_ids,
        utc_now() + timedelta(seconds=5),
        limitations,
    )

    assert result == tuple(extracted)
    assert len(extracted) == 1
    extracted_run = RunStore(workspace).read(extracted[0])
    assert extracted_run.artifacts[0].display_name.endswith(".sqlite")
    assert extracted_run.artifacts[0].producer == "nsys"
    assert limitations == []


@pytest.mark.anyio
async def test_capture_publishes_canonical_diagnostic_run_linked_to_measurement_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    executable = tmp_path / "aiperf"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    _patch_capture_dependencies(monkeypatch, executable)
    service = InferenceProfilingService(workspace)
    measurement_run_id = await _seed_measurement_run(workspace)
    plan = service.plan(
        "local",
        profiler="torch_profiler",
        scenario_name="profile",
        measurement_run_id=measurement_run_id,
        timeout_seconds=5,
    )
    service.broker = _ProfilingBroker(plan.output_path / "worker.pt.trace.json")

    forged = plan.validated_copy(update={"measurement_run_id": "forged"})
    assert forged.measurement_run_id != measurement_run_id
    result = await service.capture(plan.plan_token)

    run = RunStore(workspace).read(result.run_id)
    protocol = json.loads(run.inference_protocol_identity_json or "{}")
    assert result.measurement_protocol_id == run.measurement_protocol_id
    assert result.measurement_run_id == measurement_run_id
    assert run.source_measurement_run_id == measurement_run_id
    assert any(measurement_run_id in limitation for limitation in run.limitations)
    assert run.measurement_protocol_id != run.inference_protocol_identity_id
    assert run.workload_definition_id == result.measurement_protocol_id
    assert run.workload_instance_id == plan.plan_id
    assert protocol["profiler"] == {
        "attached": True,
        "profiler": "torch_profiler",
        "profiler_version": None,
    }
    assert run.execution_status is ExecutionStatus.SUCCEEDED
    assert run.capture_status is CaptureStatus.REGISTERED
    assert result.coverage == "complete"
    assert result.artifact_ids


@pytest.mark.anyio
@pytest.mark.parametrize("target", ("profile", "replay"))
async def test_capture_refuses_operation_directory_replaced_after_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    workspace = _workspace(tmp_path)
    executable = tmp_path / "aiperf"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    _patch_capture_dependencies(monkeypatch, executable)
    service = InferenceProfilingService(workspace)
    plan, _measurement_run_id = await _authorized_profile(service, workspace)
    assert plan.replay_plan is not None
    selected = plan.output_root if target == "profile" else plan.replay_plan.output_root
    operation_root = workspace.paths.staging.joinpath(*selected.parts())
    parked = operation_root.with_name(f"{operation_root.name}-parked")
    outside = tmp_path / f"outside-{target}"
    outside.mkdir()
    operation_root.rename(parked)
    operation_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(DomainError) as caught:
        await service.capture(plan.plan_token)

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert list(outside.iterdir()) == []


@pytest.mark.anyio
async def test_startup_failure_finalizes_run_and_preserves_partial_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    executable = tmp_path / "aiperf"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    _patch_capture_dependencies(monkeypatch, executable)
    service = InferenceProfilingService(workspace)
    plan, _measurement_run_id = await _authorized_profile(service, workspace)
    service.broker = _ProfilingBroker(
        plan.output_path / "worker.pt.trace.json",
        startup_error=DomainError(ErrorCode.PROCESS_FAILED, "server startup failed"),
    )

    with pytest.raises(DomainError, match="server startup failed") as caught:
        await service.capture(plan.plan_token)

    assert caught.value.run_id is not None
    run = RunStore(workspace).read(caught.value.run_id)
    assert run.finished_at is not None
    assert run.execution_status is ExecutionStatus.FAILED
    assert run.capture_status is CaptureStatus.REGISTERED
    assert caught.value.details["partial_artifact_ids"]


@pytest.mark.anyio
async def test_profiler_flush_failure_returns_partial_capture_and_failed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    executable = tmp_path / "aiperf"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    _patch_capture_dependencies(monkeypatch, executable)

    async def fail_stop(_self: object, **_kwargs: object) -> None:
        raise DomainError(ErrorCode.PROCESS_FAILED, "profiler flush failed")

    monkeypatch.setattr(VllmProfilerControlClient, "stop_async", fail_stop)
    service = InferenceProfilingService(workspace)
    plan, _measurement_run_id = await _authorized_profile(service, workspace)
    service.broker = _ProfilingBroker(plan.output_path / "worker.pt.trace.json")

    result = await service.capture(plan.plan_token)

    run = RunStore(workspace).read(result.run_id)
    assert result.coverage == "partial"
    assert result.artifact_ids
    assert "profiler flush failed" in result.limitations
    assert run.execution_status is ExecutionStatus.FAILED
    assert run.capture_status is CaptureStatus.REGISTERED


@pytest.mark.anyio
async def test_incomplete_server_cleanup_is_durable_and_downgrades_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    executable = tmp_path / "aiperf"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    _patch_capture_dependencies(monkeypatch, executable)
    service = InferenceProfilingService(workspace)
    plan, _measurement_run_id = await _authorized_profile(service, workspace)
    service.broker = _ProfilingBroker(
        plan.output_path / "worker.pt.trace.json", cleanup_complete=False
    )

    result = await service.capture(plan.plan_token)

    run = RunStore(workspace).read(result.run_id)
    assert result.server_cleanup_complete is False
    assert result.coverage == "partial"
    assert run.execution_status is ExecutionStatus.FAILED
    assert "Managed server process cleanup was incomplete." in run.limitations


@pytest.mark.anyio
async def test_cancelled_profile_window_finalizes_run_and_preserves_partial_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    executable = tmp_path / "aiperf"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    _patch_capture_dependencies(monkeypatch, executable)
    service = InferenceProfilingService(workspace)
    plan, _measurement_run_id = await _authorized_profile(service, workspace)
    service.broker = _ProfilingBroker(plan.output_path / "worker.pt.trace.json", cancel_window=True)

    with pytest.raises(asyncio.CancelledError):
        await service.capture(plan.plan_token)

    diagnostic_runs = [
        run for run in RunStore(workspace).list() if run.semantics.adapter == "torch_profiler"
    ]
    assert len(diagnostic_runs) == 1
    run = diagnostic_runs[0]
    assert run.finished_at is not None
    assert run.execution_status is ExecutionStatus.CANCELLED
    assert run.capture_status is CaptureStatus.REGISTERED
    assert "Inference profiling was cancelled." in run.limitations
