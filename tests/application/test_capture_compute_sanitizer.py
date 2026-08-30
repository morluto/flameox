from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

import pytest

from flameox.adapters.compute_sanitizer import ComputeSanitizerExtractor
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode
from flameox.storage import RunStore, Workspace
from tests.support.capture import disable_containment

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


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
    (tmp_path / "flameox.toml").write_text("[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\n")
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
        await service.execute(plan.plan_token)
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
        "[workloads.probe]\nargv = ['/bin/true', '--suppressions', 'workload.supp']\ncwd = '.'\n"
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

    await service.execute(plan.plan_token)

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
    (tmp_path / "flameox.toml").write_text("[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\n")
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="compute-sanitizer",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_token)

    assert result.run.validation_status.value == "inconclusive"
    assert "Unknown Compute Sanitizer XML element: futureFinding." in result.run.limitations


@pytest.mark.anyio
async def test_compute_sanitizer_same_clean_bytes_retain_mode_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_compute_sanitizer(
        tmp_path,
        monkeypatch,
        body=(
            'while [ "$1" != "--save" ]; do shift; done\n'
            "shift\n"
            "printf '%s' '<ComputeSanitizerOutput/>' > \"$1\""
        ),
    )
    (tmp_path / "flameox.toml").write_text("[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\n")
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)

    runs = []
    for tool in ("racecheck", "synccheck"):
        plan = await service.plan(
            workload_name="probe",
            adapter="compute-sanitizer",
            adapter_options={"tool": tool},
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        assert plan.adapter_options["launch_count"] == 1
        runs.append(await service.execute(plan.plan_token))

    first, second = runs
    assert first.run.artifacts[0].artifact_id == second.run.artifacts[0].artifact_id
    assert first.run.artifacts[0].producer_version is not None
    assert first.run.artifacts[0].producer_version.startswith("Version 2026.2.1")
    first_extraction = ComputeSanitizerExtractor(workspace).extract(first.run.run_id)
    second_extraction = ComputeSanitizerExtractor(workspace).extract(second.run.run_id)
    assert first_extraction.status == second_extraction.status == "clean"
    assert first_extraction.semantics.mode == "racecheck"
    assert second_extraction.semantics.mode == "synccheck"
    assert first_extraction.semantics.semantic_id != second_extraction.semantics.semantic_id
    assert (
        first_extraction.semantics.bounds
        == second_extraction.semantics.bounds
        == {
            "launch_count": 1,
            "launch_skip": 0,
        }
    )
    with Catalog(workspace).open_snapshot(second_extraction.corpus_commit_id) as snapshot:
        provenance = snapshot.execute(
            "SELECT run_id, value_json FROM observations "
            "WHERE kind = 'sanitizer.extraction' ORDER BY run_id"
        ).fetchall()
    assert {run_id for run_id, _ in provenance} == {first.run.run_id, second.run.run_id}
    assert any('"mode":"racecheck"' in value for _, value in provenance)
    assert any('"mode":"synccheck"' in value for _, value in provenance)


@pytest.mark.anyio
async def test_compute_sanitizer_timeout_returns_bounded_replan_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_compute_sanitizer(tmp_path, monkeypatch, body="sleep 10")
    (tmp_path / "flameox.toml").write_text(
        "[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\ntimeout_seconds = 0.1\n"
    )
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="compute-sanitizer",
        adapter_options={"launch_count": 0},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(DomainError) as failure:
        await service.execute(plan.plan_token)

    assert plan.adapter_options["launch_count"] == 0
    assert failure.value.code is ErrorCode.PROCESS_TIMEOUT
    assert failure.value.details["bounded_replan"]["effective_launch_count"] == 0
    assert failure.value.next_action is not None
    assert failure.value.next_action.kind == "tool"
    assert failure.value.next_action.arguments["compute_sanitizer_options"]["launch_count"] == 1
    assert failure.value.run_id is not None
    run = RunStore(workspace).read(failure.value.run_id)
    assert any("isolate a target-only workload" in limitation for limitation in run.limitations)
    assert any("kernel_name" in limitation for limitation in run.limitations)


@pytest.mark.anyio
async def test_compute_sanitizer_partial_timeout_returns_same_recovery_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_compute_sanitizer(
        tmp_path,
        monkeypatch,
        body=(
            'while [ "$1" != "--save" ]; do shift; done\n'
            "shift\n"
            "printf '%s' '<ComputeSanitizerOutput/>' > \"$1\"\n"
            "sleep 10"
        ),
    )
    (tmp_path / "flameox.toml").write_text(
        "[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\ntimeout_seconds = 0.1\n"
    )
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="probe",
        adapter="compute-sanitizer",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_token)

    assert result.run.execution_status.value == "timed_out"
    assert result.recovery is not None
    assert result.recovery.kind == "manual"
    assert "Do not repeat it unchanged" in result.recovery.instruction
    assert any("Do not repeat it unchanged" in item for item in result.run.limitations)


@pytest.mark.anyio
async def test_compute_sanitizer_refuses_managed_capture_without_gpu_device_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "flameox.toml").write_text("[workloads.probe]\nargv = ['/bin/true']\ncwd = '.'\n")
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
