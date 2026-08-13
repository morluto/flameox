from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.nsight_compute import find_ncu_report_interface
from flameox.adapters.options import bind_adapter_options
from flameox.domain import ArtifactKind, DomainError, ErrorCode

pytestmark = pytest.mark.unit


def test_strict_options_build_only_documented_bounded_arguments(tmp_path: Path) -> None:
    selected = bind_adapter_options(
        "nsight.compute",
        {
            "sections": ["LaunchStats", "SpeedOfLight"],
            "kernel_name": "vector_add",
            "launch_skip": 2,
            "launch_count": 3,
            "replay_mode": "application",
        },
        project_root=tmp_path,
    )
    invocation = build_capture_invocation(
        "nsight.compute",
        ("./workload",),
        tmp_path,
        executable="/opt/nvidia/ncu",
        options=cast(dict[str, object], selected),
        project_root=tmp_path,
    )

    assert invocation.artifact_kinds == (ArtifactKind.KERNEL_PROFILE,)
    assert invocation.argv == (
        "/opt/nvidia/ncu",
        "--export",
        str(tmp_path / "nsight-compute.ncu-rep"),
        "--force-overwrite",
        "--replay-mode",
        "application",
        "--launch-skip",
        "2",
        "--launch-count",
        "3",
        "--section",
        "LaunchStats",
        "--section",
        "SpeedOfLight",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        "vector_add",
        "./workload",
    )

    with pytest.raises(DomainError) as unknown:
        bind_adapter_options(
            "nsight.compute",
            {"set": "basic", "arbitrary_flags": ["--import"]},
            project_root=tmp_path,
        )
    assert unknown.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    with pytest.raises(DomainError):
        bind_adapter_options(
            "nsight.compute",
            {"set": None, "sections": ["regex:.*"]},
            project_root=tmp_path,
        )


def test_interface_next_to_selected_executable_wins_over_newer_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_root = tmp_path / "2025.1"
    executable = selected_root / "bin" / "ncu"
    executable.parent.mkdir(parents=True)
    executable.touch()
    selected_interface = selected_root / "extras" / "python" / "ncu_report.py"
    selected_interface.parent.mkdir(parents=True)
    selected_interface.touch()
    installation_root = tmp_path / "installations"
    matching_interface = installation_root / "2025.1" / "extras" / "python" / "ncu_report.py"
    matching_interface.parent.mkdir(parents=True)
    matching_interface.touch()
    newer_interface = installation_root / "2099.1" / "extras" / "python" / "ncu_report.py"
    newer_interface.parent.mkdir(parents=True)
    newer_interface.touch()
    monkeypatch.setattr(
        "flameox.adapters.nsight_compute._NCU_INSTALL_ROOTS",
        (installation_root,),
    )

    selected = find_ncu_report_interface(
        executable=executable,
        producer_version="Version 2025.1.0",
    )

    assert selected == selected_interface
    assert find_ncu_report_interface(producer_version="Version 2025.1.0") == matching_interface
