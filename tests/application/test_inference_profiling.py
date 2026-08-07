from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from flameox.application.inference import InferenceReplayService
from flameox.application.inference_profiling import (
    InferenceProfilingService,
    VllmProfilerControlClient,
)
from flameox.application.inference_providers import InferenceToolDiscovery
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ProcessResult,
    Sensitivity,
    new_id,
)
from flameox.domain.models import ArtifactKind, ArtifactRegistration, utc_now
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ManagedSidecarLease,
    ManagedSidecarOutcome,
    SubprocessBroker,
)
from flameox.storage import RunStore, Workspace


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
            process=ProcessResult(exit_code=0, cleanup_complete=self.cleanup_complete),
            stdout=b"",
            stderr=b"",
            containment="process_group",
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
            process=ProcessResult(exit_code=0, cleanup_complete=True),
            stdout=b"",
            stderr=b"",
            resolved_executable=Path(request.argv[0]),
            containment="process_group",
        )


def _patch_capture_dependencies(monkeypatch: pytest.MonkeyPatch, executable: Path) -> None:
    from flameox.application import inference as inference_module

    def fake_discover(tool: str) -> InferenceToolDiscovery:
        return InferenceToolDiscovery(
            tool=tool,  # type: ignore[arg-type]
            executable=executable,
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
    monkeypatch.setattr(VllmProfilerControlClient, "start", lambda _self: None)
    monkeypatch.setattr(VllmProfilerControlClient, "stop", lambda _self: None)


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
    replay.runs.append(finished, expected_revision=0)
    replay._publish_run(
        finished,
        measurement_environment,
        source_state,
        (stored.content.artifact_id,),
    )
    replay.publisher.publish_rows(
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


def test_torch_profile_plan_uses_managed_workload_and_trace_directory(tmp_path: Path) -> None:
    service = InferenceProfilingService(_workspace(tmp_path))
    plan = service.plan("local", profiler="torch_profiler")
    second = service.plan("local", profiler="torch_profiler")

    assert plan.diagnostic_only is True
    assert "VLLM_TORCH_PROFILER_DIR" in plan.environment_names
    assert plan.environment_digest.startswith("sha256:")
    assert plan.server_argv[-1] == "serve.py"
    assert plan.output_path.parent != second.output_path.parent
    assert plan.plan_id == second.plan_id


def test_nsight_plan_wraps_server_with_documented_cuda_capture_range(tmp_path: Path) -> None:
    nsys = tmp_path / "nsys"
    nsys.write_text("#!/bin/sh\n")
    nsys.chmod(0o755)

    plan = InferenceProfilingService(_workspace(tmp_path)).plan(
        "local", profiler="nsight_systems", nsys_executable=nsys
    )

    assert plan.server_argv[:3] == (str(nsys), "profile", "--trace-fork-before-exec=true")
    assert "--capture-range=cudaProfilerApi" in plan.server_argv
    assert plan.server_argv[-2:] == ("--profiler-config.profiler", "cuda")


def test_existing_local_server_cannot_be_profiled(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as caught:
        InferenceProfilingService(_workspace(tmp_path, mode="existing_local")).plan(
            "local", profiler="torch_profiler"
        )

    assert caught.value.code is ErrorCode.EXECUTION_REFUSED


def test_profiler_control_uses_only_start_and_stop_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        del timeout
        calls.append((request.method, request.full_url))  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr("flameox.application.inference_profiling.urlopen", fake_urlopen)
    client = VllmProfilerControlClient("http://127.0.0.1:8000")
    client.start()
    client.stop()

    assert calls == [
        ("POST", "http://127.0.0.1:8000/start_profile"),
        ("POST", "http://127.0.0.1:8000/stop_profile"),
    ]


def test_torch_profile_preserves_compressed_trace_for_perfetto(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    service = InferenceProfilingService(workspace)
    plan = service.plan("local", profiler="torch_profiler")
    plan.output_path.mkdir(parents=True)
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
    plan.output_path.mkdir(parents=True)
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
    plan.output_path.parent.mkdir(parents=True)
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
    plan = service.plan("local", profiler="torch_profiler")
    service.broker = _ProfilingBroker(plan.output_path / "worker.pt.trace.json")

    measurement_run_id = await _seed_measurement_run(workspace)

    result = await service.capture(
        plan,
        scenario_name="profile",
        measurement_run_id=measurement_run_id,
        timeout_seconds=5,
    )

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
    plan = service.plan("local", profiler="torch_profiler")
    service.broker = _ProfilingBroker(
        plan.output_path / "worker.pt.trace.json",
        startup_error=DomainError(ErrorCode.PROCESS_FAILED, "server startup failed"),
    )
    measurement_run_id = await _seed_measurement_run(workspace)

    with pytest.raises(DomainError, match="server startup failed") as caught:
        await service.capture(
            plan,
            scenario_name="profile",
            measurement_run_id=measurement_run_id,
            timeout_seconds=5,
        )

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

    def fail_stop(_self: object) -> None:
        raise DomainError(ErrorCode.PROCESS_FAILED, "profiler flush failed")

    monkeypatch.setattr(VllmProfilerControlClient, "stop", fail_stop)
    service = InferenceProfilingService(workspace)
    plan = service.plan("local", profiler="torch_profiler")
    service.broker = _ProfilingBroker(plan.output_path / "worker.pt.trace.json")
    measurement_run_id = await _seed_measurement_run(workspace)

    result = await service.capture(
        plan,
        scenario_name="profile",
        measurement_run_id=measurement_run_id,
        timeout_seconds=5,
    )

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
    plan = service.plan("local", profiler="torch_profiler")
    service.broker = _ProfilingBroker(
        plan.output_path / "worker.pt.trace.json", cleanup_complete=False
    )
    measurement_run_id = await _seed_measurement_run(workspace)

    result = await service.capture(
        plan,
        scenario_name="profile",
        measurement_run_id=measurement_run_id,
        timeout_seconds=5,
    )

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
    plan = service.plan("local", profiler="torch_profiler")
    service.broker = _ProfilingBroker(plan.output_path / "worker.pt.trace.json", cancel_window=True)
    measurement_run_id = await _seed_measurement_run(workspace)

    with pytest.raises(asyncio.CancelledError):
        await service.capture(
            plan,
            scenario_name="profile",
            measurement_run_id=measurement_run_id,
            timeout_seconds=5,
        )

    diagnostic_runs = [
        RunStore(workspace).read(path.name)
        for path in workspace.paths.runs.iterdir()
        if json.loads((path / "manifest.json").read_text()).get("collector") == "torch_profiler"
    ]
    assert len(diagnostic_runs) == 1
    run = diagnostic_runs[0]
    assert run.finished_at is not None
    assert run.execution_status is ExecutionStatus.CANCELLED
    assert run.capture_status is CaptureStatus.REGISTERED
    assert "Inference profiling was cancelled." in run.limitations
