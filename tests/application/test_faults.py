from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from urllib.request import urlopen

import pytest

from flameox.adapters.toxiproxy import ToxiproxyToolManager, ToxiproxyToolReceipt
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.faults import (
    FaultExperimentPlan,
    FaultExperimentService,
    _FailedSidecarAttempt,
    _SidecarAttemptCancelled,
    _SidecarAttemptError,
)
from flameox.application.records import (
    CreateInvestigationRequest,
    InvestigationService,
)
from flameox.application.workloads import FaultExperimentConfig, ProjectConfig
from flameox.domain import (
    DomainError,
    ErrorCode,
    ProcessResult,
    process_termination_from_returncode,
)
from flameox.execution import (
    ManagedSidecarOutcome,
    ProcessContainment,
    ProcessDiscoverySource,
    ProcessObservation,
    ProcessSnapshotPhase,
)
from flameox.storage import ArtifactStore, RunStore, Workspace

pytestmark = pytest.mark.integration


def test_fault_configuration_rejects_non_discriminating_transport_definitions() -> None:
    with pytest.raises(ValueError, match="endpoint_template"):
        FaultExperimentConfig.model_validate(
            {
                "workload": "client",
                "endpoint_parameter": "endpoint",
                "upstream_host": "127.0.0.1",
                "upstream_port": 8080,
                "endpoint_template": "http://{host}:{port}{",
                "scenarios": {"off": {"type": "proxy", "enabled": False}},
            }
        )
    with pytest.raises(ValueError, match="Input should be False"):
        FaultExperimentConfig.model_validate(
            {
                "workload": "client",
                "endpoint_parameter": "endpoint",
                "upstream_host": "127.0.0.1",
                "upstream_port": 8080,
                "endpoint_template": "http://{host}:{port}",
                "scenarios": {"same": {"type": "proxy", "enabled": True}},
            }
        )


def test_fault_configuration_hard_migrates_numeric_measurements() -> None:
    common = {
        "workload": "client",
        "endpoint_parameter": "endpoint",
        "upstream_host": "127.0.0.1",
        "upstream_port": 8080,
        "endpoint_template": "http://{host}:{port}",
        "scenarios": {"delay": {"type": "latency", "latency_ms": 10}},
    }
    configured = FaultExperimentConfig.model_validate(
        {**common, "measurement": {"source": "stdout_json"}}
    )
    assert configured.measurement is not None
    assert configured.measurement.source == "stdout_json"

    with pytest.raises(ValueError, match="primary_metric"):
        FaultExperimentConfig.model_validate({**common, "primary_metric": "fault.client_elapsed"})
    with pytest.raises(ValueError, match="polarity"):
        FaultExperimentConfig.model_validate(
            {**common, "measurement": {"source": "stdout_json", "polarity": "higher_is_better"}}
        )


def test_fault_configuration_rejects_unused_endpoint_parameter() -> None:
    with pytest.raises(ValueError, match="must be rendered"):
        ProjectConfig.model_validate(
            {
                "workloads": {
                    "client": {
                        "argv": ["python", "-c", "print(1)"],
                        "parameters": {"endpoint": ["unused"]},
                    }
                },
                "fault_experiments": {
                    "transport": {
                        "workload": "client",
                        "endpoint_parameter": "endpoint",
                        "upstream_host": "127.0.0.1",
                        "upstream_port": 8080,
                        "endpoint_template": "http://{host}:{port}",
                        "scenarios": {"off": {"type": "proxy", "enabled": False}},
                    }
                },
            }
        )


def test_fault_configuration_rejects_escaped_endpoint_parameter() -> None:
    with pytest.raises(ValueError, match="must be rendered"):
        ProjectConfig.model_validate(
            {
                "workloads": {
                    "client": {
                        "argv": ["python", "-c", "print('{{endpoint}}')"],
                        "parameters": {"endpoint": ["unused"]},
                    }
                },
                "fault_experiments": {
                    "transport": {
                        "workload": "client",
                        "endpoint_parameter": "endpoint",
                        "upstream_host": "127.0.0.1",
                        "upstream_port": 8080,
                        "endpoint_template": "http://{host}:{port}",
                        "scenarios": {"off": {"type": "proxy", "enabled": False}},
                    }
                },
            }
        )


class _ToolManager(ToxiproxyToolManager):
    def __init__(self, executable: Path) -> None:
        super().__init__(executable.parent)
        self.receipt = ToxiproxyToolReceipt(
            "2.12.0",
            "test.tar.gz",
            "a" * 64,
            executable,
            "b" * 64,
            "test-manifest",
        )
        self.stage_calls = 0

    def stage(self, **_kwargs: object) -> ToxiproxyToolReceipt:
        self.stage_calls += 1
        return self.receipt

    def staged_receipt(self) -> ToxiproxyToolReceipt:
        return self.receipt


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        payload = b"local-upstream\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class _FakeLease:
    def __init__(self) -> None:
        self.treatments: list[dict[str, object]] = []
        self.outcome: ManagedSidecarOutcome | None = None

    async def add_toxic_async(self, **kwargs: object) -> dict[str, object]:
        self.treatments.append(kwargs)
        return kwargs

    async def update_proxy_async(self, name: str, *, enabled: bool) -> dict[str, object]:
        treatment = {"proxy": name, "enabled": enabled}
        self.treatments.append(treatment)
        return treatment

    async def close(self) -> ManagedSidecarOutcome:
        now = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
        observation = ProcessObservation(
            pid=12345,
            create_time=1.0,
            discovery_source=ProcessDiscoverySource.ROOT,
            name="toxiproxy-server",
            status="running",
            snapshot_phase=ProcessSnapshotPhase.PRE_CLEANUP,
            alive_before_cleanup=True,
            cleanup_action="terminate",
            cleanup_outcome="True",
        )
        post = ProcessObservation.model_validate(
            {
                **observation.model_dump(mode="python"),
                "snapshot_phase": ProcessSnapshotPhase.POST_CLEANUP,
                "alive_before_cleanup": False,
            }
        )
        self.outcome = ManagedSidecarOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(0),
                cleanup_complete=True,
            ),
            stdout=b"proxy-log",
            stderr=b"",
            containment=ProcessContainment.PROCESS_GROUP,
            process_observations=(observation, post),
            started_at=now,
            finished_at=now,
        )
        return self.outcome


class _CancellingLease(_FakeLease):
    async def create_proxy_async(self, **_kwargs: object) -> dict[str, object]:
        raise asyncio.CancelledError


class _LoopbackProxyLease(_FakeLease):
    def __init__(self, upstream_port: int) -> None:
        super().__init__()
        self.upstream_port = upstream_port
        self.latency_seconds = 0.0
        self.reset_peer = False
        self.request_count = 0
        lease = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                lease.request_count += 1
                if lease.reset_peer:
                    self.close_connection = True
                    return
                if lease.latency_seconds:
                    time.sleep(lease.latency_seconds)
                with urlopen(
                    f"http://127.0.0.1:{lease.upstream_port}{self.path}",
                    timeout=2,
                ) as response:
                    payload = response.read()
                    self.send_response(response.status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def listen_port(self) -> int:
        return int(self.server.server_port)

    async def add_toxic_async(self, **kwargs: object) -> dict[str, object]:
        treatment = await super().add_toxic_async(**kwargs)
        if treatment["toxic_type"] == "latency":
            attributes = cast(dict[str, int], treatment["attributes"])
            self.latency_seconds = attributes["latency"] / 1_000
        if treatment["toxic_type"] == "reset_peer":
            self.reset_peer = True
        return treatment

    async def close(self) -> ManagedSidecarOutcome:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()
        return await super().close()


@pytest.mark.anyio
async def test_proxy_creation_cancellation_finalizes_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = FaultExperimentService(
        workspace,
        tool_manager=_ToolManager(Path("/bin/true")),
    )
    lease = _CancellingLease()

    async def start(*_args: object, **_kwargs: object) -> _CancellingLease:
        return lease

    monkeypatch.setattr(service.broker, "start_toxiproxy", start)
    plan = cast(
        FaultExperimentPlan,
        SimpleNamespace(
            tool_version="2.12.0",
            upstream_host="127.0.0.1",
            upstream_port=8080,
        ),
    )

    with pytest.raises(_SidecarAttemptCancelled) as captured:
        await service._start_sidecar(
            Path("/bin/true"),
            object(),
            plan,
            "proxy-name",
        )

    attempt = captured.value.attempts[-1]
    assert attempt.phase == "proxy_creation"
    assert attempt.outcome is not None
    assert attempt.outcome.stdout == b"proxy-log"


@pytest.mark.parametrize(
    ("failure_phase", "has_process_outcome"),
    [
        ("sidecar_readiness", True),
        ("sidecar_readiness", False),
        ("proxy_creation", True),
        ("workload_planning", True),
    ],
)
@pytest.mark.anyio
async def test_pre_capture_sidecar_failure_preserves_diagnostic_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: Literal["sidecar_readiness", "proxy_creation", "workload_planning"],
    has_process_outcome: bool,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """

[workloads.client]
argv = ["python", "-c", "print(1)", "{endpoint}"]
cwd = "."

[workloads.client.parameters]
endpoint = ["unused"]

[fault_experiments.transport]
workload = "client"
endpoint_parameter = "endpoint"
upstream_host = "127.0.0.1"
upstream_port = 8080
endpoint_template = "http://{host}:{port}"

[fault_experiments.transport.scenarios.delay]
type = "latency"
latency_ms = 10
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="why did the sidecar fail?")
    )
    service = FaultExperimentService(
        workspace,
        tool_manager=_ToolManager(Path("/bin/true")),
    )
    failed_outcome = await _FakeLease().close()
    failed_outcome = ManagedSidecarOutcome(
        process=failed_outcome.process,
        stdout=b"startup stdout",
        stderr=b"startup stderr",
        containment=failed_outcome.containment,
        process_observations=failed_outcome.process_observations,
        started_at=failed_outcome.started_at,
        finished_at=failed_outcome.finished_at,
    )

    async def fail_start(*_args: object, **_kwargs: object) -> None:
        raise _SidecarAttemptError(
            (
                _FailedSidecarAttempt(
                    cast(Literal["sidecar_readiness", "proxy_creation"], failure_phase),
                    DomainError(
                        ErrorCode.CAPABILITY_UNAVAILABLE,
                        "control API was not ready",
                        retryable=True,
                    ),
                    failed_outcome if has_process_outcome else None,
                ),
            ),
        )

    if failure_phase == "workload_planning":

        async def start_for_planning_failure(
            *_args: object,
            **_kwargs: object,
        ) -> tuple[_FakeLease, int, int, tuple[_FailedSidecarAttempt, ...]]:
            return _FakeLease(), 48000, 48001, ()

        async def fail_plan(*_args: object, **_kwargs: object) -> None:
            raise ValueError("token=supersecret /home/private/workload.py")

        monkeypatch.setattr(service, "_start_sidecar", start_for_planning_failure)
        monkeypatch.setattr(service.captures, "plan", fail_plan)
    else:
        monkeypatch.setattr(service, "_start_sidecar", fail_start)
    plan = await service.plan(
        experiment_name="transport",
        investigation_id=investigation.investigation_id,
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert all(trial.run_id is not None for trial in result.trials)
    assert set(result.trial_diagnostics) == {trial.trial_id for trial in result.trials}
    for trial in result.trials:
        assert trial.run_id is not None
        diagnostic = result.trial_diagnostics[trial.trial_id]
        assert diagnostic.phase == failure_phase
        if failure_phase == "workload_planning":
            assert diagnostic.error_code == "ValueError"
            assert diagnostic.retryable is False
            assert diagnostic.next_action is None
        else:
            assert diagnostic.error_code == ErrorCode.CAPABILITY_UNAVAILABLE
            assert diagnostic.retryable is True
            assert diagnostic.next_action is not None
        run = RunStore(workspace).read(trial.run_id)
        roles = {artifact.role for artifact in run.artifacts}
        if failure_phase == "workload_planning":
            assert "fault_configuration" in roles
        else:
            assert "fault_startup_attempt-01_diagnostic" in roles
            if has_process_outcome:
                assert "fault_startup_attempt-01_stdout" in roles
                assert "fault_startup_attempt-01_stderr" in roles
        payloads = [
            ArtifactStore(workspace).get(artifact.artifact_id).payload_path.read_bytes()
            for artifact in run.artifacts
        ]
        if failure_phase == "workload_planning":
            assert any(b"proxy-log" in payload for payload in payloads)
            assert any(
                b"workload capture plan could not be created" in payload for payload in payloads
            )
        else:
            if has_process_outcome:
                assert any(b"startup stdout" in payload for payload in payloads)
                assert any(b"startup stderr" in payload for payload in payloads)
        assert all(b"environment" not in payload.lower() for payload in payloads)
        assert all(b"supersecret" not in payload for payload in payloads)
        assert all(b"/home/private" not in payload for payload in payloads)


@pytest.mark.anyio
async def test_capture_cancellation_preserves_phase_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """

[workloads.client]
argv = ["python", "-c", "print(1)", "{endpoint}"]
cwd = "."

[workloads.client.parameters]
endpoint = ["unused"]

[fault_experiments.transport]
workload = "client"
endpoint_parameter = "endpoint"
upstream_host = "127.0.0.1"
upstream_port = 8080
endpoint_template = "http://{host}:{port}"

[fault_experiments.transport.scenarios.delay]
type = "latency"
latency_ms = 10
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="what happened to the cancelled fault trial?")
    )
    service = FaultExperimentService(workspace, tool_manager=_ToolManager(Path("/bin/true")))
    lease = _FakeLease()

    async def start_sidecar(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[_FakeLease, int, int, tuple[_FailedSidecarAttempt, ...]]:
        return lease, 48000, 48001, ()

    async def cancel_capture(*_args: object, **_kwargs: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(service, "_start_sidecar", start_sidecar)
    monkeypatch.setattr(service.captures, "execute", cancel_capture)
    plan = await service.plan(
        experiment_name="transport",
        investigation_id=investigation.investigation_id,
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.run(plan.plan_token)

    runs = RunStore(workspace).list()
    assert runs
    configuration = next(
        artifact
        for run in runs
        for artifact in run.artifacts
        if artifact.role == "fault_configuration"
    )
    payload = json.loads(
        ArtifactStore(workspace).get(configuration.artifact_id).payload_path.read_text()
    )
    assert payload["failure"] == {
        "error_code": ErrorCode.PROCESS_CANCELLED,
        "message": "The workload capture could not be completed.",
        "phase": "capture_execution",
        "retryable": True,
    }


@pytest.mark.anyio
async def test_fault_plan_records_workload_and_sidecar_containment_separately(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """

[workloads.client]
argv = ["python", "-c", "print(1)", "{endpoint}"]
cwd = "."

[workloads.client.parameters]
endpoint = ["unused"]

[fault_experiments.transport]
workload = "client"
endpoint_parameter = "endpoint"
upstream_host = "127.0.0.1"
upstream_port = 8080
endpoint_template = "http://{host}:{port}"

[fault_experiments.transport.scenarios.delay]
type = "latency"
latency_ms = 10
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="does transport latency change the outcome?")
    )

    tool_manager = _ToolManager(Path("/bin/true"))
    plan = await FaultExperimentService(workspace, tool_manager=tool_manager).plan(
        experiment_name="transport",
        investigation_id=investigation.investigation_id,
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )

    assert "request_digest" not in plan.model_dump(mode="json")
    assert plan.workload_containment == ExecutionPolicy.APPROVED_AGENT.value
    assert plan.containment == "managed_process_group"
    assert tool_manager.stage_calls == 0

    (tmp_path / "flameox.toml").write_text(
        (tmp_path / "flameox.toml").read_text().replace("print(1)", "print(2)")
    )
    with pytest.raises(DomainError) as error:
        await FaultExperimentService(
            workspace,
            tool_manager=_ToolManager(Path("/bin/true")),
        ).run(plan.plan_token)
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN


@pytest.mark.anyio
async def test_fault_run_preserves_proxy_config_process_evidence_and_endpoint_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    client_code = (
        "from urllib.request import urlopen; import sys; "
        "print(sys.argv[1]); print(urlopen(sys.argv[1], timeout=2).read().decode())"
    )
    (tmp_path / "flameox.toml").write_text(
        f"""

[workloads.client]
argv = ["python", "-c", {json.dumps(client_code)}, "{{endpoint}}"]
cwd = "."
timeout_seconds = 5

[workloads.client.parameters]
endpoint = ["unused"]

[fault_experiments.transport]
workload = "client"
endpoint_parameter = "endpoint"
upstream_host = "127.0.0.1"
upstream_port = {upstream.server_port}
endpoint_template = "http://{{host}}:{{port}}"
blocks = 1
repetitions = 1

[fault_experiments.transport.scenarios.delay]
type = "latency"
latency_ms = 10
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="does transport latency change the outcome?")
    )
    tool_manager = _ToolManager(Path("/bin/true"))
    captures = CaptureService(workspace)
    service = FaultExperimentService(
        workspace,
        captures=captures,
        tool_manager=tool_manager,
    )
    leases: list[_FakeLease] = []
    proxy_names: list[str] = []

    async def fake_start(
        executable: Path,
        receipt: object,
        plan: object,
        proxy_name: str,
    ) -> tuple[_FakeLease, int, int, tuple[_FailedSidecarAttempt, ...]]:
        proxy_names.append(proxy_name)
        lease = _FakeLease()
        leases.append(lease)
        recovered = await _FakeLease().close()
        failed_attempt = _FailedSidecarAttempt(
            "sidecar_readiness",
            DomainError(ErrorCode.CAPABILITY_UNAVAILABLE, "first attempt failed"),
            recovered,
        )
        return lease, 48000 + len(leases), upstream.server_port, (failed_attempt,)

    monkeypatch.setattr(service, "_start_sidecar", fake_start)
    try:
        plan = await service.plan(
            experiment_name="transport",
            investigation_id=investigation.investigation_id,
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        result = await service.run(plan.plan_token)
    finally:
        upstream.shutdown()
        thread.join(timeout=2)
        upstream.server_close()

    assert len(result.trials) == 2
    assert all(trial.run_id is not None for trial in result.trials)
    assert len(leases) == 2
    assert len(set(proxy_names)) == 2
    assert all(name.startswith("flameox-") and ":" not in name for name in proxy_names)
    assert all(len(name) <= 100 for name in proxy_names)
    assert result.block_treatment_orders == tuple(
        block.order for block in plan.experiment_plan.blocks
    )
    # Baseline must have no toxics; the latency treatment must have one.
    baseline_lease = next(lease for lease in leases if lease.treatments == [])
    treatment_lease = next(
        lease for lease in leases if lease.treatments and lease is not baseline_lease
    )
    assert treatment_lease.treatments[0]["toxic_type"] == "latency"
    assert all(trial.trial_id in result.trial_artifacts for trial in result.trials)
    assert result.trial_diagnostics == {}

    treatment_run = next(
        trial.run_id for trial in result.trials if trial.factors["scenario"] == "delay"
    )
    assert treatment_run is not None
    run = captures.runs.read(treatment_run)
    assert any(artifact.role == "fault_startup_attempt-01_diagnostic" for artifact in run.artifacts)
    config_registration = next(
        artifact for artifact in run.artifacts if artifact.role == "fault_configuration"
    )
    snapshot_registration = next(
        artifact for artifact in run.artifacts if artifact.role == "fault_process_observation"
    )
    config_payload = json.loads(
        ArtifactStore(workspace).get(config_registration.artifact_id).payload_path.read_text()
    )
    assert config_payload["tool"]["version"] == "2.12.0"
    assert config_payload["tool"]["sha256"] == "a" * 64
    assert config_payload["observed"]["admin_port"] in (48001, 48002)
    assert config_payload["observed"]["proxy_upstream"] == f"127.0.0.1:{upstream.server_port}"
    assert config_payload["observed"]["oracle"]["validation_status"] == "not_requested"
    endpoint = f"http://127.0.0.1:{upstream.server_port}"
    workload_payloads = (
        ArtifactStore(workspace).get(artifact.artifact_id).payload_path.read_bytes()
        for artifact in run.artifacts
        if artifact.role not in {"fault_configuration", "fault_process_observation"}
    )
    assert any(endpoint.encode() in payload for payload in workload_payloads)
    snapshot_payload = json.loads(
        ArtifactStore(workspace).get(snapshot_registration.artifact_id).payload_path.read_text()
    )
    assert snapshot_payload["evidence_status"] == "available"
    assert snapshot_payload["limitations"] == []
    assert {item["snapshot_phase"] for item in snapshot_payload["observations"]} == {
        "pre_cleanup",
        "post_cleanup",
    }
    assert all(
        not set(item).intersection({"cmdline", "environment", "cwd", "exe", "connections"})
        for item in snapshot_payload["observations"]
    )


@pytest.mark.anyio
async def test_fault_receipt_compares_latency_without_measuring_reset_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    client_code = (
        "from urllib.request import urlopen; import json, sys, time; "
        "started = time.monotonic_ns(); urlopen(sys.argv[1], timeout=2).read(); "
        "print(json.dumps({'elapsed_ns': time.monotonic_ns() - started, 'outcome': 'completed'}))"
    )
    (tmp_path / "flameox.toml").write_text(
        f"""

[workloads.client]
argv = ["python", "-c", {json.dumps(client_code)}, "{{endpoint}}"]
cwd = "."
timeout_seconds = 5

[workloads.client.parameters]
endpoint = ["unused"]

[fault_experiments.transport]
workload = "client"
endpoint_parameter = "endpoint"
upstream_host = "127.0.0.1"
upstream_port = {upstream.server_port}
endpoint_template = "http://{{host}}:{{port}}"
blocks = 2

[fault_experiments.transport.measurement]
source = "stdout_json"
practical_threshold = 0.01

[fault_experiments.transport.scenarios.delay]
type = "latency"
latency_ms = 40

[fault_experiments.transport.scenarios.reset]
type = "reset_peer"
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="how do transport faults affect client wall time?")
    )
    service = FaultExperimentService(workspace, tool_manager=_ToolManager(Path("/bin/true")))
    leases: list[_LoopbackProxyLease] = []

    async def fake_start(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[_LoopbackProxyLease, int, int, tuple[_FailedSidecarAttempt, ...]]:
        lease = _LoopbackProxyLease(upstream.server_port)
        leases.append(lease)
        return lease, 48000 + len(leases), lease.listen_port, ()

    monkeypatch.setattr(service, "_start_sidecar", fake_start)
    try:
        plan = await service.plan(
            experiment_name="transport",
            investigation_id=investigation.investigation_id,
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        result = await service.run(plan.plan_token)
    finally:
        upstream.shutdown()
        thread.join(timeout=2)
        upstream.server_close()

    assert result.experiment.primary_metric == "fault.client_elapsed"
    assert result.experiment.primary_metric_unit == "ns"
    assert len(result.run_sets) == 3
    assert len(result.comparisons) == 2
    delay = next(
        comparison
        for comparison in result.comparisons
        if comparison.candidate_run_set.selection["variant"] == "delay"
    ).comparison
    reset = next(
        comparison
        for comparison in result.comparisons
        if comparison.candidate_run_set.selection["variant"] == "reset"
    ).comparison
    assert delay.baseline_eligible_n == 2
    assert delay.candidate_eligible_n == 2
    assert delay.absolute_change is not None
    assert sum(lease.request_count for lease in leases if lease.latency_seconds) == 2
    assert delay.absolute_change.value > 20_000_000, {
        "baseline": delay.baseline_value,
        "candidate": delay.candidate_value,
        "change": delay.absolute_change,
    }
    assert reset.candidate_eligible_n == 0
    assert reset.candidate_value is None
    assert all(
        trial.outcome.value != "succeeded"
        for trial in result.trials
        if trial.factors["scenario"] == "reset"
    )
    assert result.measurement_recovery is not None
    assert result.measurement_recovery.suggested_action is not None
    assert result.measurement_recovery.suggested_action.value == "fault_experiment.plan"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("client_code", "error_code"),
    [
        ("print('not-json')", ErrorCode.ARTIFACT_PARSE_FAILED),
        ("pass", ErrorCode.ARTIFACT_NOT_FOUND),
    ],
)
async def test_fault_invalid_receipt_is_diagnostic_and_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client_code: str,
    error_code: ErrorCode,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    (tmp_path / "flameox.toml").write_text(
        f"""

[workloads.client]
argv = ["python", "-c", {json.dumps(client_code)}, "{{endpoint}}"]
cwd = "."

[workloads.client.parameters]
endpoint = ["unused"]

[fault_experiments.transport]
workload = "client"
endpoint_parameter = "endpoint"
upstream_host = "127.0.0.1"
upstream_port = 8080
endpoint_template = "http://{{host}}:{{port}}"

[fault_experiments.transport.measurement]
source = "stdout_json"

[fault_experiments.transport.scenarios.delay]
type = "latency"
latency_ms = 10
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="what if a fault workload emits no receipt?")
    )
    service = FaultExperimentService(workspace, tool_manager=_ToolManager(Path("/bin/true")))

    async def fake_start(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[_FakeLease, int, int, tuple[_FailedSidecarAttempt, ...]]:
        return _FakeLease(), 48000, 48001, ()

    monkeypatch.setattr(service, "_start_sidecar", fake_start)
    plan = await service.plan(
        experiment_name="transport",
        investigation_id=investigation.investigation_id,
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.run(plan.plan_token)

    assert len(result.trial_diagnostics) == 2
    assert {
        (diagnostic.phase, diagnostic.error_code)
        for diagnostic in result.trial_diagnostics.values()
    } == {("measurement_receipt", error_code)}
    assert len(result.comparisons) == 1
    comparison = result.comparisons[0].comparison
    assert comparison.baseline_eligible_n == 0
    assert comparison.candidate_eligible_n == 0
    assert comparison.absolute_change is None
    assert comparison.validity.value == "invalid"
    assert result.measurement_recovery is not None


@pytest.mark.anyio
async def test_fault_plan_randomizes_treatment_order_across_blocks(
    tmp_path: Path,
) -> None:
    """Regression: random_seed must permute the schedule so baseline is not always first.

    Issue #286: Fault experiments always ran baseline first, confounding
    treatment with execution order. The seed must now produce a different
    schedule than the fixed declaration order.
    """
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """

[workloads.client]
argv = ["python", "-c", "print(1)", "{endpoint}"]
cwd = "."

[workloads.client.parameters]
endpoint = ["unused"]

[fault_experiments.transport]
workload = "client"
endpoint_parameter = "endpoint"
upstream_host = "127.0.0.1"
upstream_port = 8080
endpoint_template = "http://{host}:{port}"
blocks = 2
repetitions = 1
random_seed = 42

[fault_experiments.transport.scenarios.delay]
type = "latency"
latency_ms = 10

[fault_experiments.transport.scenarios.timeout]
type = "timeout"
timeout_ms = 50
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="does transport latency change the outcome?")
    )
    tool_manager = _ToolManager(Path("/bin/true"))
    plan = await FaultExperimentService(workspace, tool_manager=tool_manager).plan(
        experiment_name="transport",
        investigation_id=investigation.investigation_id,
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    # Baseline must not be first in every block (the old confounding bug).
    first_treatments = [block.order[0] for block in plan.experiment_plan.blocks]
    assert not all(t == "baseline" for t in first_treatments), (
        "Baseline is always first — treatment is confounded with execution order."
    )

    # All declared treatments appear in each block (no drop).
    for block in plan.experiment_plan.blocks:
        assert set(block.order) == {"baseline", "delay", "timeout"}

    # Reproducibility: same seed + same block gives the same order.
    plan2 = await FaultExperimentService(
        workspace, tool_manager=_ToolManager(Path("/bin/true"))
    ).plan(
        experiment_name="transport",
        investigation_id=investigation.investigation_id,
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    assert [b.order for b in plan.experiment_plan.blocks] == [
        b.order for b in plan2.experiment_plan.blocks
    ]

    # Different seed gives a different schedule.
    (tmp_path / "flameox.toml").write_text(
        (tmp_path / "flameox.toml").read_text().replace("random_seed = 42", "random_seed = 999")
    )
    plan_diff = await FaultExperimentService(
        workspace, tool_manager=_ToolManager(Path("/bin/true"))
    ).plan(
        experiment_name="transport",
        investigation_id=investigation.investigation_id,
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    assert [b.order for b in plan_diff.experiment_plan.blocks] != [
        b.order for b in plan.experiment_plan.blocks
    ]
