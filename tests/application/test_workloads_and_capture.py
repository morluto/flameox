from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from flameox.adapters import PerfettoExtractor, PyPerfExtractor
from flameox.application import (
    CapabilityList,
    CapabilityService,
    CaptureService,
    ExecutionIdentityService,
    ExecutionPolicy,
    PreflightService,
    RunDiscoveryService,
    RunFilter,
    WorkloadService,
)
from flameox.catalog import Catalog
from flameox.config import WorkspaceConfig
from flameox.domain import (
    CapabilityReport,
    CapabilityStatus,
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ExternalExecutionContext,
    ValidationStatus,
)
from flameox.storage import ArtifactStore, RunStore, Workspace


def write_workload(project: Path, *, message: str = "hello") -> None:
    (project / "flameox.toml").write_text(
        f"""
schema_version = 1

[workloads.echo]
argv = ["python", "-c", "print('{{message}}')"]
cwd = "."
timeout_seconds = 5

[workloads.echo.parameters]
message = ["{message}", "candidate"]
"""
    )


def disable_containment(workspace: Workspace) -> None:
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    assert isinstance(config, WorkspaceConfig)
    workspace.paths.config.write_text(config.to_toml())


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


@pytest.mark.anyio
async def test_capture_preserves_runtime_storage_policy_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.wait]
argv = ["python", "-c", "import time; time.sleep(10)"]
timeout_seconds = 20
"""
    )
    disable_containment(workspace)
    monkeypatch.setattr(
        "flameox.application.capture.StorageQuota.require_capacity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "flameox.execution.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=9_999, free=1),
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="wait",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(DomainError) as terminated:
        await service.execute(plan.plan_id)

    run = RunStore(workspace).read(plan.run_id)
    assert terminated.value.code is ErrorCode.STORAGE_QUOTA_EXCEEDED
    assert terminated.value.run_id == plan.run_id
    assert run.process is not None
    assert run.process.cancellation_cause == "storage_reserve_exceeded"
    assert run.process.cleanup_complete is True
    assert run.process.resources is not None
    assert run.process.resources.policy_termination == "storage_reserve_exceeded"


@pytest.mark.anyio
async def test_external_execution_context_is_bound_preserved_and_discoverable(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.fail]
argv = ["python", "-c", "raise SystemExit(3)"]
"""
    )
    disable_containment(workspace)
    context = ExternalExecutionContext(
        orchestrator="crabbox",
        provider="runpod",
        lease_id="lease-123",
        worker_id="worker-7",
        orchestration_run_id="validation-9",
        sensitivity="sensitive",
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="fail",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        external_context=context,
    )
    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.external_context == context
    discovered = RunDiscoveryService(workspace).list(
        filter=RunFilter(provider="runpod", lease_id="lease-123"),
        limit=10,
    )
    assert [item.run_id for item in discovered.runs] == [result.run.run_id]
    assert discovered.runs[0].worker_id == "worker-7"

    replacement = context.model_copy(update={"lease_id": "lease-replaced"})
    second = await service.plan(
        workload_name="fail",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        external_context=context,
    )
    await service.plans.issue(second.model_copy(update={"external_context": replacement}))
    with pytest.raises(DomainError) as tampered:
        await service.execute(second.plan_id)
    assert tampered.value.code is ErrorCode.INVALID_CAPTURE_PLAN

    with pytest.raises(ValueError):
        ExternalExecutionContext(
            orchestrator="crabbox",
            provider="runpod",
            lease_id="contains secret whitespace",
            worker_id="worker",
            orchestration_run_id="run",
        )


@pytest.mark.anyio
async def test_declared_module_and_native_file_identity_is_observed_and_revalidated(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "project_module.py").write_text("VALUE = 1\n")
    (tmp_path / "lib").mkdir()
    native = tmp_path / "lib" / "extension.so"
    native.write_bytes(b"candidate-a")
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.identity]
argv = ["python", "-c", "import project_module; print(project_module.VALUE)"]
[workloads.identity.identity]
python_modules = ["project_module"]
native_files = ["lib/extension.so"]
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="identity",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    assert plan.planned_execution_identity.quality == "partial"
    result = await service.execute(plan.plan_id)

    identity = result.run.execution_identity
    assert identity is not None
    assert identity.quality == "exact"
    module, library = identity.inputs
    assert module.requested == "project_module"
    assert module.resolved_path == str(tmp_path / "project_module.py")
    assert module.loaded_path == module.resolved_path
    assert module.content_digest is not None
    assert library.configured_path == str(native)
    assert library.resolved_path == str(native)
    assert library.content_digest is not None
    assert library.loaded_path is None

    first_digest = library.content_digest
    native.write_bytes(b"candidate-b")
    changed = ExecutionIdentityService(workspace).plan("identity")
    assert changed.inputs[1].content_digest != first_digest

    stale_plan = await service.plan(
        workload_name="identity",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    native.write_bytes(b"candidate-c")
    with pytest.raises(DomainError) as stale:
        await service.execute(stale_plan.plan_id)
    assert stale.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_missing_declared_native_identity_is_explicitly_partial(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.identity]
argv = ["python", "-c", "print('ok')"]
[workloads.identity.identity]
native_files = ["build/missing.so"]
"""
    )

    identity = ExecutionIdentityService(workspace).plan("identity")

    assert identity.quality == "partial"
    assert identity.missing_inputs == ("build/missing.so",)
    assert identity.inputs[0].status == "missing"


@pytest.mark.anyio
async def test_declared_missing_accelerator_downgrades_captured_environment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.cuda]
argv = ["python", "-c", "print('bounded')"]
[workloads.cuda.identity.environment]
required = ["cuda.driver", "cuda.runtime", "cuda.devices", "cuda.peer_topology"]
"""
    )
    disable_containment(workspace)
    real_which = shutil.which
    monkeypatch.setattr(
        "flameox.application.environment.shutil.which",
        lambda name: None if name == "nvidia-smi" else real_which(name),
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="cuda",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    with Catalog(workspace).open_snapshot() as snapshot:
        row = snapshot.execute(
            "SELECT identity_quality, missing_fields, fields_json "
            "FROM environments WHERE environment_id = ?",
            (result.run.environment_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "partial"
    assert set(row[1]) == {
        "cuda.driver",
        "cuda.runtime",
        "cuda.devices",
        "cuda.peer_topology",
    }
    assert '"status":"missing"' in row[2]


def test_writable_roots_reject_traversal_and_symlink_escape(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    for value in ("../outside", "escape", ".git", ".diagnostics/build"):
        (tmp_path / "flameox.toml").write_text(
            f"""
schema_version = 1
[workloads.build]
argv = ["python", "-c", "print('build')"]
writable_paths = [{json.dumps(value)}]
"""
        )
        with pytest.raises(DomainError) as refused:
            WorkloadService(workspace).writable_targets("build")
        assert refused.value.code is ErrorCode.EXECUTION_REFUSED


@pytest.mark.anyio
async def test_capture_plan_uses_minimal_bubblewrap_and_systemd_limits(
    tmp_path: Path,
) -> None:
    if shutil.which("bwrap") is None or shutil.which("systemd-run") is None:
        pytest.skip("Bubblewrap and systemd-run are required for active containment.")
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        "[workloads.echo]\n"
        'argv = ["python", "-c", "'
        "import os, pathlib; "
        "assert not pathlib.Path('.diagnostics/workspace.json').exists(); "
        "pathlib.Path(os.environ['FLAMEOX_OBSERVATIONS_PATH']).parent"
        ".joinpath('write-proof').write_text('ok'); "
        "print('contained')"
        '"]\n'
        'cwd = "."\n'
        "timeout_seconds = 5\n"
        "[workloads.echo.oracle]\n"
        'argv = ["python", "-c", "'
        "import pathlib; "
        "assert not pathlib.Path('.diagnostics/workspace.json').exists()"
        '"]\n'
    )

    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    argv = plan.collector_argv
    assert plan.containment == "active"
    assert plan.network_contained
    assert plan.systemd_scope_unit is not None
    assert "--property=KillMode=control-group" in argv
    assert f"--property=MemoryMax={workspace.config.execution.max_memory_bytes}" in argv
    assert f"--property=TasksMax={workspace.config.execution.max_processes}" in argv
    assert ("--ro-bind", "/", "/") not in tuple(zip(argv, argv[1:], argv[2:], strict=False))
    diagnostics_index = argv.index(str(workspace.paths.root.resolve()))
    assert argv[diagnostics_index - 1] == "--tmpfs"

    result = await service.execute(plan.plan_id)
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.validation_status is ValidationStatus.PASSED
    assert result.run.process is not None
    assert result.run.process.cleanup_complete
    assert not any((workspace.paths.staging / "captures").iterdir())


@pytest.mark.anyio
async def test_approved_cargo_build_uses_only_declared_writable_root(
    tmp_path: Path,
) -> None:
    if any(shutil.which(name) is None for name in ("bwrap", "systemd-run", "cargo")):
        pytest.skip("Cargo, bubblewrap, and systemd-run are required.")
    cargo = subprocess.run(
        ("rustup", "which", "cargo"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    toolchain_bin = str(Path(cargo).parent)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main() {}\n")
    (tmp_path / "target").mkdir()
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "flameox-proof"\nversion = "0.1.0"\nedition = "2024"\n'
    )
    subprocess.run(
        (cargo, "generate-lockfile", "--offline"),
        cwd=tmp_path,
        check=True,
        env={"PATH": toolchain_bin + ":/usr/bin:/bin"},
        capture_output=True,
    )
    for generated in (tmp_path / "target").iterdir():
        generated.unlink()
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.cargo_check]
argv = [{json.dumps(cargo)}, "check", "--offline"]
writable_paths = ["target"]
timeout_seconds = 30
[workloads.cargo_check.environment]
PATH = {json.dumps(toolchain_bin + ":/usr/bin:/bin")}
"""
    )
    workspace = Workspace.initialize(tmp_path)
    WorkloadService(workspace).approve("cargo_check")
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="cargo_check",
        adapter="command",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )
    binding = plan.writable_roots[0]

    result = await service.execute(plan.plan_id)

    stderr = next(
        (
            ArtifactStore(workspace).get(item.artifact_id).payload_path.read_text()
            for item in result.run.artifacts
            if item.role == "stderr"
        ),
        "",
    )
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED, stderr
    assert result.run.writable_roots == plan.writable_roots
    assert result.run.process is not None
    assert result.run.process.resources is not None
    assert result.run.process.resources.writable_root_growth_bytes[binding.storage_path] > 0
    assert not any((tmp_path / "target").iterdir())


@pytest.mark.anyio
async def test_capture_plan_reports_degraded_when_systemd_user_manager_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    service = CaptureService(workspace)
    real_which = shutil.which

    def available_executable(name: str) -> str | None:
        if name in {"bwrap", "systemd-run"}:
            return "/usr/bin/true"
        return real_which(name)

    async def unavailable_user_manager(_systemd_run: str) -> bool:
        return False

    monkeypatch.setattr("flameox.application.capture.shutil.which", available_executable)
    monkeypatch.setattr(service, "_systemd_user_scope_available", unavailable_user_manager)

    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert plan.containment == "degraded"
    assert plan.systemd_scope_unit is None


@pytest.mark.anyio
async def test_capture_plan_is_single_use_and_publishes_process_evidence(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        parameters={"message": "candidate"},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    assert result.run.source_state_id is not None
    assert result.run.artifacts
    payload = ArtifactStore(workspace).get(result.run.artifacts[0].artifact_id)
    assert payload.payload_path.read_text().strip() == "candidate"
    events = [
        json.loads(line)
        for line in workspace.paths.operation_log.read_text().splitlines()
        if json.loads(line)["operation"] == "capture.execute"
    ]
    assert events[-1]["phase"] == "Evidence publication complete"
    assert "candidate" not in workspace.paths.operation_log.read_text()
    with pytest.raises(DomainError) as replay:
        await service.execute(plan.plan_id)
    assert replay.value.code is ErrorCode.INVALID_CAPTURE_PLAN


@pytest.mark.anyio
async def test_measurement_protocol_identity_binds_collector_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    identity = {"digest": "first"}
    monkeypatch.setattr(
        service,
        "_executable_identity",
        lambda _executable: dict(identity),
    )

    first_plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    first = await service.execute(first_plan.plan_id)
    identity["digest"] = "second"
    second_plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    second = await service.execute(second_plan.plan_id)

    assert first.run.measurement_protocol_id != second.run.measurement_protocol_id


@pytest.mark.anyio
async def test_concurrent_captures_execute_while_publications_serialize(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    first, second = await asyncio.gather(
        service.plan(
            workload_name="echo",
            adapter="command",
            parameters={"message": "hello"},
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        ),
        service.plan(
            workload_name="echo",
            adapter="command",
            parameters={"message": "candidate"},
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        ),
    )

    results = await asyncio.gather(
        service.execute(first.plan_id),
        service.execute(second.plan_id),
    )

    assert {result.run.execution_status for result in results} == {ExecutionStatus.SUCCEEDED}
    assert len({result.run.run_id for result in results}) == 2
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(DISTINCT run_id) FROM runs").fetchone() == (2,)


@pytest.mark.anyio
async def test_capture_cancellation_leaves_terminal_run_revision(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.wait]
argv = ["python", "-c", "import time; time.sleep(10)"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    context = ExternalExecutionContext(
        orchestrator="crabbox",
        provider="runpod",
        lease_id="lease-cancelled",
        worker_id="worker-cancelled",
        orchestration_run_id="orchestration-cancelled",
        sensitivity="sensitive",
    )
    plan = await service.plan(
        workload_name="wait",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        external_context=context,
    )
    task = asyncio.create_task(service.execute(plan.plan_id))
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    runs = [path.name for path in workspace.paths.runs.iterdir() if path.is_dir()]
    assert len(runs) == 1
    terminal = RunStore(workspace).read(runs[0])
    assert terminal.execution_status is ExecutionStatus.CANCELLED
    assert terminal.capture_status is CaptureStatus.CANCELLED
    assert terminal.finished_at is not None
    assert terminal.external_context == context


@pytest.mark.anyio
async def test_artifact_registration_failure_is_terminal_and_cleans_staging(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    config = workspace.config.model_copy(
        update={
            "capture": workspace.config.capture.model_copy(update={"max_artifact_bytes": 1}),
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"}),
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(DomainError) as error:
        await service.execute(plan.plan_id)

    terminal = RunStore(workspace).read(plan.run_id)
    assert error.value.code is ErrorCode.ARTIFACT_TOO_LARGE
    assert terminal.execution_status is ExecutionStatus.FAILED
    assert terminal.capture_status is CaptureStatus.FAILED
    assert not any((workspace.paths.staging / "captures").iterdir())


@pytest.mark.anyio
async def test_publication_failure_is_terminal_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    original = cast(Any, service.publisher.publish_rows)

    def fail_evidence_publication(
        rows: dict[str, list[dict[str, object]]],
        *args: object,
        **kwargs: object,
    ) -> object:
        if "artifact_registrations" in rows:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "simulated publication failure",
            )
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(service.publisher, "publish_rows", fail_evidence_publication)

    with pytest.raises(DomainError, match="simulated publication failure"):
        await service.execute(plan.plan_id)

    terminal = RunStore(workspace).read(plan.run_id)
    assert terminal.execution_status is ExecutionStatus.FAILED
    assert terminal.capture_status is CaptureStatus.FAILED
    assert not any((workspace.paths.staging / "captures").iterdir())


@pytest.mark.parametrize("cancel_at_phase", (1, 2, 3, 4, 5, 6, 7))
@pytest.mark.anyio
async def test_capture_cancellation_at_awaited_phase_is_terminal(
    tmp_path: Path,
    cancel_at_phase: int,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    async def cancel(
        completed: float,
        _total: float,
        _message: str,
    ) -> None:
        if completed == cancel_at_phase:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await service.execute(plan.plan_id, progress=cancel)

    terminal = RunStore(workspace).read(plan.run_id)
    assert terminal.execution_status is ExecutionStatus.CANCELLED
    assert terminal.capture_status is CaptureStatus.CANCELLED


@pytest.mark.anyio
async def test_cancellation_during_atomic_publication_finishes_then_marks_run_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    started = threading.Event()
    release = threading.Event()
    original = cast(Any, service.publisher.publish_rows)

    def delayed(*args: object, **kwargs: object) -> object:
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.publisher, "publish_rows", delayed)
    task = asyncio.create_task(service.execute(plan.plan_id))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    terminal = RunStore(workspace).read(plan.run_id)
    assert terminal.execution_status is ExecutionStatus.CANCELLED
    assert terminal.capture_status is CaptureStatus.CANCELLED
    with Catalog(workspace).open_snapshot() as snapshot:
        latest = snapshot.execute(
            "SELECT execution_status FROM runs WHERE run_id = ? ORDER BY published_at DESC LIMIT 1",
            (plan.run_id,),
        ).fetchone()
    assert latest == ("cancelled",)


@pytest.mark.anyio
async def test_coverage_capture_uses_supported_launcher_and_registers_native_data(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "workload.py").write_text("value = sum(range(10))\nprint(value)\n")
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.script]
argv = ["python", "workload.py"]
cwd = "."
timeout_seconds = 10
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="script",
        adapter="coverage",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert any(item.kind.value == "execution_coverage" for item in result.run.artifacts)


@pytest.mark.anyio
async def test_py_spy_capture_round_trips_through_perfetto(
    tmp_path: Path,
) -> None:
    binaries = sorted(
        (Path.home() / ".local" / "share" / "perfetto" / "prebuilts").glob(
            "trace_processor_shell-*"
        )
    )
    if not binaries:
        pytest.skip("A local Trace Processor binary is not installed.")
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "busy.py").write_text(
        "import time\n"
        "total = 0\n"
        "deadline = time.monotonic() + 1.0\n"
        "while time.monotonic() < deadline:\n"
        "    total += sum(i * i for i in range(5000))\n"
        "print(total)\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.busy]
argv = ["python", "busy.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"}),
            "analysis": workspace.config.analysis.model_copy(
                update={"trace_processor_path": str(binaries[-1])}
            ),
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="busy",
        adapter="py-spy",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    captured = await service.execute(plan.plan_id)
    extracted = await PerfettoExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.process is not None
    expected_status = (
        ExecutionStatus.SUCCEEDED if captured.run.process.exit_code == 0 else ExecutionStatus.FAILED
    )
    assert captured.run.execution_status is expected_status
    assert extracted.slice_count > 0
    assert extracted.frame_count > 0


@pytest.mark.anyio
async def test_pyperf_capture_preserves_native_worker_hierarchy(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "scan.py").write_text(
        "values = list(range(2000))\n"
        "total = 0\n"
        "for value in reversed(values):\n"
        "    total += value\n"
        "assert total > 0\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.scan]
argv = ["python", "scan.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="scan",
        adapter="pyperf",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    captured = await service.execute(plan.plan_id)
    extracted = PyPerfExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.SUCCEEDED
    separator = captured.plan.collector_argv.index("--")
    assert Path(captured.plan.collector_argv[separator + 1]).name.startswith("python")
    assert captured.plan.collector_argv[separator + 2 :] == ("scan.py",)
    assert extracted.measurement_count == 9
    assert extracted.warmup_count >= 3
    primary = next(artifact for artifact in captured.run.artifacts if artifact.role == "primary")
    assert captured.plan.adapter_version is not None
    assert primary.producer == "pyperf"
    assert primary.producer_version == captured.plan.adapter_version
    assert extracted.limitations == ()


@pytest.mark.anyio
async def test_perf_capture_registers_native_profile_when_kernel_allows_sampling(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    if CapabilityService(workspace).get("perf").executable is None:
        pytest.skip("perf is not installed.")
    (tmp_path / "busy.py").write_text(
        "total = 0\nfor value in range(1000000):\n    total += value * value\nprint(total)\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.busy]
argv = ["python", "busy.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="busy",
        adapter="perf",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)
    if result.run.execution_status is not ExecutionStatus.SUCCEEDED:
        pytest.skip("The host kernel does not permit perf sampling for this process.")

    assert any(
        registration.kind.value == "sample_profile" and registration.display_name == "perf.data"
        for registration in result.run.artifacts
    )


@pytest.mark.anyio
async def test_torch_profiler_capture_registers_public_chrome_trace(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    if CapabilityService(workspace).get("torch.profiler").status is not CapabilityStatus.AVAILABLE:
        pytest.skip("PyTorch is not installed.")
    (tmp_path / "torch_workload.py").write_text(
        "import torch\n"
        "left = torch.ones((16, 16))\n"
        "right = torch.ones((16, 16))\n"
        "print(torch.mm(left, right).sum().item())\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.torch]
argv = ["python", "torch_workload.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="torch",
        adapter="torch.profiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    trace = next(
        registration
        for registration in result.run.artifacts
        if registration.kind.value == "execution_trace"
    )
    assert trace.display_name == "torch-trace.json"
