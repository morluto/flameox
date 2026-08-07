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
from flameox.domain import ProcessResult
from flameox.execution import ManagedSidecarOutcome, ProcessObservation
from flameox.storage import ArtifactStore, Workspace


class _ToolManager(ToxiproxyToolManager):
    def __init__(self, executable: Path) -> None:
        super().__init__(executable.parent)
        self.receipt = ToxiproxyToolReceipt("2.12.0", "test.tar.gz", "a" * 64, executable)

    def stage(self) -> ToxiproxyToolReceipt:
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

    def add_toxic(self, **kwargs: object) -> dict[str, object]:
        self.treatments.append(kwargs)
        return kwargs

    def update_proxy(self, name: str, *, enabled: bool) -> dict[str, object]:
        treatment = {"proxy": name, "enabled": enabled}
        self.treatments.append(treatment)
        return treatment

    async def close(self) -> ManagedSidecarOutcome:
        now = datetime.now(UTC)
        observation = ProcessObservation(
            pid=12345,
            create_time=1.0,
            discovery_source="root",
            name="toxiproxy-server",
            status="running",
            snapshot_phase="pre_cleanup",
            alive_before_cleanup=True,
            cleanup_action="terminate",
            cleanup_outcome="True",
        )
        post = observation.model_copy(
            update={
                "snapshot_phase": "post_cleanup",
                "alive_before_cleanup": False,
            }
        )
        self.outcome = ManagedSidecarOutcome(
            process=ProcessResult(exit_code=0, cleanup_complete=True),
            stdout=b"proxy-log",
            stderr=b"",
            containment="process_group",
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
argv = ["python", "-c", "print(1)"]
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

    plan = await FaultExperimentService(
        workspace,
        tool_manager=_ToolManager(Path("/bin/true")),
    ).plan(
        experiment_name="transport",
        investigation_id=investigation.investigation_id,
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )

    assert plan.workload_containment == ExecutionPolicy.APPROVED_AGENT.value
    assert plan.containment == "managed_process_group"


@pytest.mark.anyio
async def test_fault_run_preserves_proxy_config_process_evidence_and_endpoint_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
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

    async def fake_start(
        executable: Path,
        receipt: object,
        plan: object,
        proxy_name: str,
    ) -> tuple[_FakeLease, int, int]:
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
        result = await service.run(plan.plan_id)
    finally:
        upstream.shutdown()
        thread.join(timeout=2)
        upstream.server_close()

    assert len(result.trials) == 2
    assert all(trial.run_id is not None for trial in result.trials)
    assert len(leases) == 2
    assert leases[0].treatments == []
    assert leases[1].treatments[0]["toxic_type"] == "latency"
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
    assert config_payload["observed"]["admin_port"] == 48002
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
