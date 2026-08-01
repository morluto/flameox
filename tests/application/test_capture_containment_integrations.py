from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from flameox.application import (
    CaptureService,
    ExecutionPolicy,
)
from flameox.domain import (
    ExecutionStatus,
    ValidationStatus,
)
from flameox.storage import ArtifactStore, Workspace


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_bwrap
@pytest.mark.requires_systemd
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
@pytest.mark.optional
@pytest.mark.requires_bwrap
@pytest.mark.requires_cargo
@pytest.mark.requires_systemd
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
