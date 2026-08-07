from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters.toxiproxy import ToxiproxyToolManager, ToxiproxyToolReceipt
from flameox.application import (
    CreateInvestigationRequest,
    ExecutionPolicy,
    FaultExperimentService,
    InvestigationService,
)
from flameox.storage import Workspace


class _ToolManager(ToxiproxyToolManager):
    def __init__(self, executable: Path) -> None:
        super().__init__(executable.parent)
        self.receipt = ToxiproxyToolReceipt("2.12.0", "test.tar.gz", "a" * 64, executable)

    def stage(self) -> ToxiproxyToolReceipt:
        return self.receipt


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
