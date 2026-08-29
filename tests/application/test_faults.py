from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from flameox.adapters.toxiproxy import ToxiproxyToolManager, ToxiproxyToolReceipt
from flameox.application import (
    CaptureService,
    CreateInvestigationRequest,
    ExecutionPolicy,
    FaultExperimentService,
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
from flameox.storage import ArtifactStore, Workspace

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


def test_fault_configuration_rejects_unused_endpoint_parameter() -> None:
    with pytest.raises(ValueError, match="must be rendered"):
        ProjectConfig.model_validate(
            {
                "schema_version": 1,
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
                "schema_version": 1,
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


@pytest.mark.anyio
async def test_fault_plan_records_workload_and_sidecar_containment_separately(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1

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
schema_version = 1

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
    ) -> tuple[_FakeLease, int, int]:
        proxy_names.append(proxy_name)
        lease = _FakeLease()
        leases.append(lease)
        return lease, 48000 + len(leases), upstream.server_port

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
    assert result.treatment_order is None
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

    treatment_run = next(
        trial.run_id for trial in result.trials if trial.factors["scenario"] == "delay"
    )
    assert treatment_run is not None
    run = captures.runs.read(treatment_run)
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
    assert {item["snapshot_phase"] for item in snapshot_payload} == {
        "pre_cleanup",
        "post_cleanup",
    }
    assert all(
        not set(item).intersection({"cmdline", "environment", "cwd", "exe", "connections"})
        for item in snapshot_payload
    )


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
schema_version = 1

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
