from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

import pytest

from flameox.application import CaptureService, ExecutionPolicy
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace
from tests.support.capture import disable_containment


def _install_fake_compute_sanitizer(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: str,
) -> None:
    executable = project / "compute-sanitizer"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        "  printf 'Version 2026.2.1.0 (build fixture)\\n'\n"
        "  exit 0\n"
        "fi\n"
        f"{body}\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{project}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.anyio
async def test_compute_sanitizer_rejects_suppression_changed_after_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "sanitizer-ran"
    _install_fake_compute_sanitizer(
        tmp_path,
        monkeypatch,
        body=f"touch {shlex.quote(str(marker))}",
    )
    suppression = tmp_path / "sanitizer.supp"
    suppression.write_text("# planned bytes\n")
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\n"
    )
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="compute-sanitizer",
        adapter_options={"suppression_file": "sanitizer.supp"},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    suppression.write_text("# bytes changed after authorization\n")

    with pytest.raises(DomainError) as changed:
        await service.execute(plan.plan_id)
    assert changed.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert not marker.exists()


@pytest.mark.anyio
async def test_compute_sanitizer_preserves_workload_suppression_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_argv = tmp_path / "captured-argv"
    _install_fake_compute_sanitizer(
        tmp_path,
        monkeypatch,
        body=(
            f"printf '%s\\n' \"$@\" > {shlex.quote(str(captured_argv))}\n"
            'while [ "$1" != "--save" ]; do shift; done\n'
            "shift\n"
            "printf '%s' '<ComputeSanitizerOutput/>' > \"$1\""
        ),
    )
    suppression = tmp_path / "sanitizer.supp"
    suppression.write_text("# authorized collector suppression\n")
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        "[workloads.probe]\n"
        "argv = ['/bin/true', '--suppressions', 'workload.supp']\n"
        "cwd = '.'\n"
    )
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="compute-sanitizer",
        adapter_options={"suppression_file": "sanitizer.supp"},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    await service.execute(plan.plan_id)

    argv = captured_argv.read_text().splitlines()
    suppression_indexes = [
        index for index, argument in enumerate(argv) if argument == "--suppressions"
    ]
    assert len(suppression_indexes) == 2
    assert argv[suppression_indexes[0] + 1] != str(suppression)
    assert argv[suppression_indexes[1] + 1] == "workload.supp"


@pytest.mark.anyio
async def test_compute_sanitizer_clean_capture_with_unknown_xml_is_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_compute_sanitizer(
        tmp_path,
        monkeypatch,
        body=(
            'while [ "$1" != "--save" ]; do shift; done\n'
            "shift\n"
            "printf '%s' "
            "'<ComputeSanitizerOutput><futureFinding/></ComputeSanitizerOutput>' "
            '> "$1"\n'
        ),
    )
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\n"
    )
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="compute-sanitizer",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_id)

    assert result.run.validation_status.value == "inconclusive"
    assert "Unknown Compute Sanitizer XML element: futureFinding." in result.run.limitations


@pytest.mark.anyio
async def test_compute_sanitizer_refuses_managed_capture_without_gpu_device_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\n"
    )
    workspace = Workspace.initialize(tmp_path)
    _install_fake_compute_sanitizer(tmp_path, monkeypatch, body="exit 0")
    service = CaptureService(workspace)

    with pytest.raises(DomainError, match="cannot access NVIDIA devices") as error:
        await service.plan(
            workload_name="probe",
            adapter="compute-sanitizer",
            execution_policy=ExecutionPolicy.APPROVED_AGENT,
        )

    assert error.value.code is ErrorCode.EXECUTION_REFUSED
