from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from flameox.adapters import NsightSystemsExtractor
from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.options import bind_adapter_options
from flameox.analysis import RecipeService
from flameox.application import CapabilityService, CaptureService, ExecutionPolicy
from flameox.domain import (
    ArtifactKind,
    CapabilityStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
)
from flameox.storage import Workspace
from tests.support.capture import disable_containment

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


def test_nsight_systems_invocation_binds_conservative_effective_options(tmp_path: Path) -> None:
    bound = bind_adapter_options(
        "nsight.systems",
        {
            "trace": ["cuda", "nvtx", "nccl"],
            "capture_range": "cudaProfilerApi",
            "capture_range_end": "stop-shutdown",
            "cuda_trace_scope": "process-tree",
            "include_pre_exec_fork_interval": True,
        },
        project_root=tmp_path,
    )

    invocation = build_capture_invocation(
        "nsight.systems",
        ("python", "train.py"),
        tmp_path / "capture",
        executable="/opt/nvidia/nsys",
        options=cast(dict[str, object], bound),
    )

    assert invocation.argv == (
        "/opt/nvidia/nsys",
        "profile",
        "--trace=cuda,nvtx,nccl",
        "--sample=none",
        "--cpuctxsw=none",
        "--resolve-symbols=false",
        "--export=sqlite",
        "--force-overwrite=true",
        "--trace-fork-before-exec=true",
        "--cuda-trace-scope=process-tree",
        "--cuda-graph-trace=graph",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--output",
        str(tmp_path / "capture" / "nsight-systems"),
        "python",
        "train.py",
    )
    assert invocation.artifact_kinds == (ArtifactKind.EXECUTION_TRACE,)


def test_nsight_systems_options_reject_inconsistent_capture_range(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as failure:
        bind_adapter_options(
            "nsight.systems",
            {"trace": ["nvtx"], "capture_range": "cudaProfilerApi"},
            project_root=tmp_path,
        )

    assert failure.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    with pytest.raises(DomainError) as missing_nvtx:
        bind_adapter_options(
            "nsight.systems",
            {"trace": ["nvtx"], "capture_range": "nvtx"},
            project_root=tmp_path,
        )
    assert "nvtx_capture" in missing_nvtx.value.details["validation_error"]


@pytest.mark.anyio
async def test_declared_workload_capture_preserves_native_and_extracts_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "nsight_systems" / "nsight-2025.5.2.sqlite"
    executable_directory = tmp_path / "bin"
    executable_directory.mkdir()
    _write_fake_nsys(executable_directory / "nsys", fixture)
    monkeypatch.setenv("PATH", f"{executable_directory}{os.pathsep}{os.environ['PATH']}")
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        "[workloads.probe]\n"
        "argv = ['/bin/true']\n"
        "cwd = '.'\n"
        "timeout_seconds = 5\n"
    )
    disable_containment(workspace)
    capabilities = CapabilityService(workspace)
    passive = capabilities.get("nsight.systems")
    active = await capabilities.probe("nsight.systems", refresh=True)
    assert passive.status is CapabilityStatus.AVAILABLE
    assert passive.executable == str((executable_directory / "nsys").resolve())
    assert passive.version is None
    assert active.version == "NVIDIA Nsight Systems version 2025.5.2.266-255236693005v0"
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="nsight.systems",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    captured = await service.execute(plan.plan_token)
    extracted = await NsightSystemsExtractor(workspace).extract(captured.run.run_id)
    analysis = RecipeService(workspace).accelerator_launches(captured.run.run_id)

    traces = [
        artifact
        for artifact in captured.run.artifacts
        if artifact.kind is ArtifactKind.EXECUTION_TRACE
    ]
    assert captured.run.execution_status is ExecutionStatus.SUCCEEDED
    assert {(item.display_name, item.role) for item in traces} == {
        ("nsight-systems.nsys-rep", "native_report"),
        ("nsight-systems.sqlite", "sqlite_export"),
    }
    assert all(item.producer == "nsight.systems" for item in traces)
    assert captured.run.semantics.scope.mode is not None
    assert captured.run.semantics.scope.mode.value == "none"
    assert captured.run.semantics.scope.process_scope is not None
    assert captured.run.semantics.scope.process_scope.value == "process-tree"
    assert captured.run.semantics.scope.bounds == {}
    assert captured.run.semantics.configuration["trace"] == ["cuda", "nvtx", "osrt"]
    assert extracted.artifact_id == next(
        item.artifact_id for item in traces if item.role == "sqlite_export"
    )
    assert extracted.kernel_event_count == 3
    assert analysis.regions[0].kernel_count == 3


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("exit_code", "valid_sqlite", "emit_native"),
    [(0, False, True), (7, True, True), (7, True, False)],
)
async def test_failed_nsight_outputs_remain_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    valid_sqlite: bool,
    emit_native: bool,
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "nsight_systems" / "nsight-2025.5.2.sqlite"
    executable_directory = tmp_path / "bin"
    executable_directory.mkdir()
    _write_fake_nsys(
        executable_directory / "nsys",
        fixture,
        exit_code=exit_code,
        valid_sqlite=valid_sqlite,
        emit_native=emit_native,
    )
    monkeypatch.setenv("PATH", f"{executable_directory}{os.pathsep}{os.environ['PATH']}")
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        "[workloads.probe]\n"
        "argv = ['/bin/true']\n"
        "cwd = '.'\n"
        "timeout_seconds = 5\n"
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="nsight.systems",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    captured = await service.execute(plan.plan_token)

    assert captured.run.execution_status is ExecutionStatus.FAILED
    trace_roles = {
        item.role
        for item in captured.run.artifacts
        if item.kind is ArtifactKind.EXECUTION_TRACE
    }
    assert trace_roles <= {"partial_native_report", "partial_sqlite_export"}
    assert "sqlite_export" not in trace_roles
    with pytest.raises(DomainError, match="structured export"):
        await NsightSystemsExtractor(workspace).extract(captured.run.run_id)


@pytest.mark.anyio
async def test_sqlite_registration_failure_retains_native_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "nsight_systems" / "nsight-2025.5.2.sqlite"
    executable_directory = tmp_path / "bin"
    executable_directory.mkdir()
    _write_fake_nsys(executable_directory / "nsys", fixture)
    monkeypatch.setenv("PATH", f"{executable_directory}{os.pathsep}{os.environ['PATH']}")
    workspace = Workspace.initialize(tmp_path)
    constrained = workspace.config.validated_copy(
        update={
            "capture": workspace.config.capture.validated_copy(
                update={"max_artifact_bytes": 1_024}
            )
        }
    )
    workspace.paths.config.write_text(constrained.to_toml())
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        "[workloads.probe]\n"
        "argv = ['/bin/true']\n"
        "cwd = '.'\n"
        "timeout_seconds = 5\n"
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="nsight.systems",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(DomainError) as failure:
        await service.execute(plan.plan_token)

    assert failure.value.run_id is not None
    run = service.runs.read(failure.value.run_id)
    assert any(item.role == "native_report" for item in run.artifacts)
    assert all(item.role != "sqlite_export" for item in run.artifacts)


def _write_fake_nsys(
    path: Path,
    fixture: Path,
    *,
    exit_code: int = 0,
    valid_sqlite: bool = True,
    emit_native: bool = True,
) -> None:
    sqlite_command = (
        f"cp {fixture} \"${{output}}.sqlite\"\n"
        if valid_sqlite
        else "printf 'not-sqlite' > \"${output}.sqlite\"\n"
    )
    native_command = (
        "printf 'native-report' > \"${output}.nsys-rep\"\n" if emit_native else ""
    )
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  printf 'NVIDIA Nsight Systems version 2025.5.2.266-255236693005v0\\n'\n"
        "  exit 0\n"
        "fi\n"
        "output=''\n"
        "previous=''\n"
        "for argument in \"$@\"; do\n"
        "  if [ \"$previous\" = \"--output\" ]; then output=$argument; fi\n"
        "  previous=$argument\n"
        "done\n"
        + native_command
        + sqlite_command
        + f"exit {exit_code}\n"
    )
    path.chmod(0o755)
