from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import (
    CapabilityList,
    CapabilityService,
    CaptureService,
    ExecutionPolicy,
    PreflightService,
    WorkloadService,
)
from flameox.domain import (
    CapabilityReport,
    CapabilityStatus,
    DomainError,
    ErrorCode,
)
from flameox.storage import Workspace
from tests.support.capture import write_workload


def test_workload_approval_is_bound_to_canonical_definition(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    service = WorkloadService(workspace)

    definition = service.definition("echo")
    assert definition.approved_definition_digest is None
    with pytest.raises(DomainError) as unapproved:
        service.resolve("echo", require_approval=True)

    approved = service.approve("echo")
    instance = service.resolve(
        "echo",
        {"message": "candidate"},
        require_approval=True,
    )
    write_workload(tmp_path, message="changed")

    assert unapproved.value.code is ErrorCode.EXECUTION_REFUSED
    assert approved.approved_definition_digest == approved.workload_definition_id
    assert instance.command.argv[-1] == "print('candidate')"
    assert service.definition("echo").approved_definition_digest is None


def test_declared_workflow_discovery_is_paginated_and_bound_to_configuration(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.alpha]
argv = ["python", "-c", "print('{size}')"]
[workloads.alpha.parameters]
size = [1, 2]
[workloads.beta]
argv = ["python", "-c", "print('beta')"]
[experiments.scaling]
workload = "alpha"
variants = ["1", "2"]
primary_metric = "duration"
polarity = "lower_is_better"
"""
    )
    service = WorkloadService(workspace)
    service.approve("alpha")

    first = service.list_declared(kind="workload", approval="any", limit=1)
    assert [item.name for item in first.workflows] == ["alpha"]
    assert first.workflows[0].approval == "approved"
    assert first.next_cursor is not None
    second = service.list_declared(
        kind="workload",
        approval="any",
        limit=1,
        cursor=first.next_cursor,
    )
    detail = service.get_declared(kind="experiment", name="scaling")

    assert [item.name for item in second.workflows] == ["beta"]
    assert second.next_cursor is None
    assert detail.allowed_parameters == {"size": (1, 2)}
    assert detail.variants == ("1", "2")

    (tmp_path / "flameox.toml").write_text(
        (tmp_path / "flameox.toml").read_text().replace("print('beta')", "print('changed')")
    )
    with pytest.raises(DomainError) as stale:
        service.list_declared(
            kind="workload",
            approval="any",
            limit=1,
            cursor=first.next_cursor,
        )
    assert stale.value.code is ErrorCode.STALE_CURSOR


@pytest.mark.anyio
async def test_preflight_distinguishes_required_optional_and_active_requirements(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["python", "-c", "print('ok')"]
[workloads.probe.requirements]
executables = ["python", "flameox-definitely-missing"]
python_distributions = ["flameox-missing-distribution"]
capabilities = ["missing.active.capability"]
optional = ["flameox-definitely-missing", "flameox-missing-distribution"]
active = ["missing.active.capability"]
allow_exploratory = true
"""
    )

    passive = await PreflightService(workspace).inspect("probe", mode="passive")
    active = await PreflightService(workspace).inspect("probe", mode="active")
    by_name = {item.requirement: item for item in passive.requirements}

    assert by_name["python"].status == "available"
    assert by_name["flameox-definitely-missing"].status == "absent"
    assert by_name["flameox-definitely-missing"].required is False
    assert by_name["flameox-missing-distribution"].status == "absent"
    assert by_name["missing.active.capability"].status == "unknown"
    assert by_name["missing.active.capability"].probe_kind == "active"
    assert passive.disposition == "exploratory"
    assert active.requirements[-1].status == "absent"
    assert active.preflight_id != passive.preflight_id


@pytest.mark.anyio
async def test_active_preflight_preserves_permission_denied_state(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["python", "-c", "print('ok')"]
[workloads.probe.requirements]
capabilities = ["permission.probe"]
active = ["permission.probe"]
"""
    )

    class PermissionCapabilities(CapabilityService):
        def list(self) -> CapabilityList:
            return CapabilityList(
                capabilities=(
                    CapabilityReport(
                        adapter="permission.probe",
                        status=CapabilityStatus.AVAILABLE,
                    ),
                )
            )

        async def probe(self, adapter: str, *, refresh: bool = False) -> CapabilityReport:
            assert adapter == "permission.probe"
            assert refresh
            return CapabilityReport(
                adapter=adapter,
                status=CapabilityStatus.PERMISSION_REQUIRED,
                permission_status="denied",
                remediation=("Grant the documented local permission.",),
                probe_kind="active",
            )

    result = await PreflightService(
        workspace,
        capabilities=PermissionCapabilities(workspace),
    ).inspect("probe", mode="active")

    assert result.disposition == "blocked"
    assert result.requirements[0].status == "permission_denied"
    assert result.requirements[0].probe_kind == "active"


@pytest.mark.anyio
async def test_required_preflight_failure_blocks_capture_planning(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["python", "-c", "print('ok')"]
[workloads.probe.requirements]
python_distributions = ["flameox-missing-distribution"]
"""
    )

    with pytest.raises(DomainError) as blocked:
        await CaptureService(workspace).plan(
            workload_name="probe",
            adapter="command",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    assert blocked.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    preflight = blocked.value.details["preflight"]
    assert isinstance(preflight, dict)
    assert preflight["disposition"] == "blocked"


@pytest.mark.anyio
async def test_capture_plan_binds_declared_writable_root_and_rejects_replacement(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    output = tmp_path / "target"
    output.mkdir()
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.build]
argv = ["python", "-c", "print('build')"]
writable_paths = ["target"]
"""
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="build",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert len(plan.writable_roots) == 1
    binding = plan.writable_roots[0]
    assert binding.target_path == str(output)
    assert binding.storage_path.endswith(f"/{plan.plan_id}/writable/0")
    output.rename(tmp_path / "target-old")
    output.mkdir()

    with pytest.raises(DomainError) as replaced:
        await service.execute(plan.plan_id)
    assert replaced.value.code is ErrorCode.INVALID_CAPTURE_PLAN
