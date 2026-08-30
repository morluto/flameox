from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from flameox.action_graph import ActionId, ToolAction
from flameox.application.capabilities import (
    CapabilityList,
    CapabilityService,
)
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.preflight import PreflightService
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.application.workloads import (
    DeclaredWorkflowKind,
    WorkloadService,
)
from flameox.domain import (
    CapabilityPermissionStatus,
    CapabilityReport,
    CapabilityStatus,
    DomainError,
    ErrorCode,
    ProbeKind,
    ProcessResult,
    process_termination_from_returncode,
)
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ProcessContainment,
    SubprocessBroker,
)
from flameox.storage import Workspace
from tests.support.capture import write_workload

pytestmark = pytest.mark.integration


class _NvccProbeBroker(SubprocessBroker):
    async def run(self, request: ExecutionRequest, **_: Any) -> ExecutionOutcome:
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(1),
                cleanup_complete=True,
            ),
            stdout=b"",
            stderr=b"fatal error: cuda_runtime.h: No such file or directory\n",
            resolved_executable=Path(request.argv[0]),
            executable_binding=request.executable_binding,
            containment=ProcessContainment.PROCESS_GROUP,
        )


class _NvccProbeFailureBroker(SubprocessBroker):
    async def run(self, request: ExecutionRequest, **_: Any) -> ExecutionOutcome:
        raise DomainError(
            ErrorCode.PROCESS_TIMEOUT,
            "The bounded CUDA probe exceeded its execution budget.",
            details={
                "process": {
                    "stdout": "",
                    "stderr": f"timed out while reading {request.argv[4]}",
                }
            },
        )


class _PathReportingNvccProbeBroker(SubprocessBroker):
    async def run(self, request: ExecutionRequest, **_: Any) -> ExecutionOutcome:
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(1),
                cleanup_complete=True,
            ),
            stdout=b"",
            stderr=(
                f"{request.argv[4]}: fatal error: cuda_runtime.h: No such file or directory\n"
            ).encode(),
            resolved_executable=Path(request.argv[0]),
            executable_binding=request.executable_binding,
            containment=ProcessContainment.PROCESS_GROUP,
        )


@pytest.mark.anyio
async def test_memray_plan_binds_the_profiled_workload_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    workload_python = tmp_path / "workload-env" / "bin" / "python"
    workload_python.parent.mkdir(parents=True)
    workload_python.write_text('#!/bin/sh\nprintf \'{"memray":"1.20.0"}\\n\'\n')
    workload_python.chmod(0o755)
    (tmp_path / "flameox.toml").write_text(
        f"""
[workloads.profile]
argv = ["{workload_python}", "workload.py", "--size", "4"]
"""
    )
    monkeypatch.setattr(
        ProviderRuntimeManager,
        "find_distribution",
        lambda *_args, **_kwargs: object(),
    )

    plan = await CaptureService(workspace).plan(
        workload_name="profile",
        adapter="memray",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert plan.collector_argv[:3] == (str(workload_python), "-m", "memray")
    assert plan.collector_argv[-3:] == ("workload.py", "--size", "4")
    assert plan.semantics.scope.workload_cwd == "."


@pytest.mark.anyio
async def test_memray_plan_requires_an_exact_reader_before_capture(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    workload_python = tmp_path / "workload-env" / "bin" / "python"
    workload_python.parent.mkdir(parents=True)
    workload_python.write_text('#!/bin/sh\nprintf \'{"memray":"1.20.0"}\\n\'\n')
    workload_python.chmod(0o755)
    (tmp_path / "flameox.toml").write_text(
        f'''[workloads.profile]
argv = ["{workload_python}", "workload.py"]
'''
    )

    with pytest.raises(DomainError) as raised:
        await CaptureService(workspace).plan(
            workload_name="profile",
            adapter="memray",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    assert raised.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert raised.value.details["producer_version"] == "1.20.0"
    assert raised.value.next_action == ToolAction(
        action=ActionId.START_CAPABILITY_SETUP,
        arguments={
            "adapters": ["memray"],
            "idempotency_key": "memray-reader-1.20.0",
            "memray_reader_version": "1.20.0",
        },
    )


def test_current_workload_definition_is_active_and_bound_to_plans(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    service = WorkloadService(workspace)

    definition = service.definition("echo")
    instance = service.resolve(
        "echo",
        {"message": "candidate"},
    )
    write_workload(tmp_path, message="changed")

    assert instance.command.argv[-1] == "print('candidate')"
    assert service.definition("echo").workload_definition_id != definition.workload_definition_id


def test_declared_workflow_discovery_is_paginated_and_bound_to_configuration(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.alpha]
argv = ["python", "-c", "print('{size}')"]
[workloads.alpha.parameters]
size = [1, 2]
[workloads.beta]
argv = ["python", "-c", "print('beta')"]
[experiments.scaling]
workload = "alpha"
treatment_factor = "size"
primary_metric = "duration"
polarity = "lower_is_better"
[experiments.scaling.factors]
size = [1, 2]
"""
    )
    service = WorkloadService(workspace)

    first = service.list_declared(kind=DeclaredWorkflowKind.WORKLOAD, limit=1)
    assert [item.name for item in first.workflows] == ["alpha"]
    assert first.next_cursor is not None
    second = service.list_declared(
        kind=DeclaredWorkflowKind.WORKLOAD,
        limit=1,
        cursor=first.next_cursor,
    )
    detail = service.get_declared(kind=DeclaredWorkflowKind.EXPERIMENT, name="scaling")

    assert [item.name for item in second.workflows] == ["beta"]
    assert second.next_cursor is None
    assert detail.allowed_parameters == {"size": (1, 2)}
    assert detail.factors == {"size": (1, 2)}

    (tmp_path / "flameox.toml").write_text(
        (tmp_path / "flameox.toml").read_text().replace("print('beta')", "print('changed')")
    )
    with pytest.raises(DomainError) as stale:
        service.list_declared(
            kind=DeclaredWorkflowKind.WORKLOAD,
            limit=1,
            cursor=first.next_cursor,
        )
    assert stale.value.code is ErrorCode.STALE_CURSOR


def test_declared_workflow_details_expose_requirements_and_adapter_options(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.probe]
argv = ["python", "-c", "print('ok')"]
[workloads.probe.requirements]
executables = ["python"]
capabilities = ["perf"]
optional = ["perf"]
active = ["perf"]
"""
    )

    service = WorkloadService(workspace)
    detail = service.get_declared(kind=DeclaredWorkflowKind.WORKLOAD, name="probe")
    inspection = service.inspect("probe")

    requirements = {item.name: item for item in detail.requirements}
    assert requirements["python"].kind == "executable"
    assert requirements["python"].required is True
    assert requirements["perf"].required is False
    assert requirements["perf"].probe_kind == "active"
    assert detail.adapter_option_total >= len(detail.adapter_options)
    assert detail.adapter_options_truncated is (
        detail.adapter_option_total > len(detail.adapter_options)
    )
    assert detail.validated_copy() == detail
    assert tuple(item.adapter for item in detail.adapter_options) == tuple(
        sorted(item.adapter for item in detail.adapter_options)
    )
    command = next(item for item in detail.adapter_options if item.adapter == "command")
    assert command.planning_disposition == "ready"
    assert inspection.command_template == ("python", "-c", "print('ok')")
    assert inspection.configuration_id == detail.configuration_id


@pytest.mark.anyio
async def test_preflight_distinguishes_required_optional_and_active_requirements(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
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

    passive = await PreflightService(workspace).inspect("probe", mode=ProbeKind.PASSIVE)
    active = await PreflightService(workspace).inspect("probe", mode=ProbeKind.ACTIVE)
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
                permission_status=CapabilityPermissionStatus.DENIED,
                remediation=("Grant the documented local permission.",),
                probe_kind=ProbeKind.ACTIVE,
            )

    result = await PreflightService(
        workspace,
        capabilities=PermissionCapabilities(workspace),
    ).inspect("probe", mode=ProbeKind.ACTIVE)

    assert result.disposition == "blocked"
    assert result.requirements[0].status == "permission_denied"
    assert result.requirements[0].probe_kind == "active"


@pytest.mark.anyio
async def test_active_nvcc_preflight_classifies_missing_cuda_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.gpu]
argv = ["python", "-c", "print('gpu')"]
[workloads.gpu.requirements]
executables = ["nvcc"]
"""
    )
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which",
        lambda _name, path=None: sys.executable,
    )

    result = await PreflightService(
        workspace,
        broker=_NvccProbeBroker(),
    ).inspect("gpu", mode=ProbeKind.ACTIVE)

    requirement = result.requirements[0]
    assert result.disposition == "blocked"
    assert requirement.status == "environment_blocked"
    assert "cuda_runtime.h" in requirement.limitations[0]
    assert "cuda_runtime.h" in requirement.remediation[0]
    assert "cuda_runtime.h" in requirement.evidence[1]


@pytest.mark.anyio
async def test_active_nvcc_probe_failure_is_not_classified_as_compile_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.gpu]
argv = ["python", "-c", "print('gpu')"]
[workloads.gpu.requirements]
executables = ["nvcc"]
"""
    )
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which",
        lambda _name, path=None: sys.executable,
    )

    result = await PreflightService(
        workspace,
        broker=_NvccProbeFailureBroker(),
    ).inspect("gpu", mode=ProbeKind.ACTIVE)

    requirement = result.requirements[0]
    assert requirement.status == "probe_failed"
    assert "Retry active preflight" in requirement.remediation[0]


@pytest.mark.anyio
async def test_cuda_preflight_id_ignores_ephemeral_probe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.gpu]
argv = ["python", "-c", "print('gpu')"]
[workloads.gpu.requirements]
executables = ["nvcc"]
"""
    )
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which",
        lambda _name, path=None: sys.executable,
    )
    service = PreflightService(workspace, broker=_PathReportingNvccProbeBroker())

    first = await service.inspect("gpu", mode=ProbeKind.ACTIVE)
    second = await service.inspect("gpu", mode=ProbeKind.ACTIVE)

    assert first.preflight_id == second.preflight_id
    assert first.requirements[0].evidence[1] == (
        "<cuda-preflight-root>/header_probe.cu: fatal error: cuda_runtime.h: "
        "No such file or directory"
    )


@pytest.mark.anyio
async def test_required_preflight_failure_blocks_capture_planning(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
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
async def test_capture_planning_defaults_to_bounded_active_preflight(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)

    plan = await CaptureService(workspace).plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert plan.preflight.mode == "active"
    assert plan.collector_executable_binding.invocation_path == Path(plan.collector_argv[0])
    assert plan.workload_instance.executable_binding is not None


@pytest.mark.anyio
async def test_invalid_capture_adapter_reports_bounded_recovery_choices(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)

    with pytest.raises(DomainError) as refused:
        await CaptureService(workspace).plan(
            workload_name="echo",
            adapter="not-a-capture-adapter",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    assert refused.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    choices = refused.value.details["allowed_adapters"]
    assert isinstance(choices, list)
    assert len(choices) <= 64
    assert choices == sorted(choices)
    assert isinstance(refused.value.next_action, ToolAction)
    assert refused.value.next_action.action is ActionId.GET_DECLARED_WORKFLOW
    assert refused.value.next_action.arguments == {"kind": "workload", "name": "echo"}


@pytest.mark.anyio
async def test_missing_workload_package_points_to_declared_environment(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    python = tmp_path / "workload-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nprintf '{\"torch\":null}\\n'\n")
    python.chmod(0o755)
    (tmp_path / "flameox.toml").write_text(
        f"""
[workloads.echo]
argv = ["{python}", "-c", "print('candidate')"]
"""
    )
    service = CaptureService(workspace)

    with pytest.raises(DomainError) as refused:
        await service.plan(
            workload_name="echo",
            adapter="torch.profiler",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    assert isinstance(refused.value.next_action, ToolAction)
    assert refused.value.next_action.action is ActionId.INSPECT_CAPABILITIES
    assert refused.value.details["setup_adapters"] == []
    assert "command" in refused.value.details["fallback_adapters"]
    assert any(str(python) in item for item in refused.value.remediation)


@pytest.mark.anyio
async def test_capture_plan_binds_declared_writable_root_and_rejects_replacement(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    output = tmp_path / "target"
    output.mkdir()
    (tmp_path / "flameox.toml").write_text(
        """
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
    assert binding.storage_path.endswith(f"/{plan.run_id}/writable/0")
    output.rename(tmp_path / "target-old")
    output.mkdir()

    with pytest.raises(DomainError) as replaced:
        await service.execute(plan.plan_token)
    assert replaced.value.code is ErrorCode.INVALID_CAPTURE_PLAN
