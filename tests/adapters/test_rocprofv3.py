from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.options import bind_adapter_options
from flameox.domain import ArtifactKind, DomainError, ErrorCode


def test_rocprofv3_invocation_uses_only_selected_pftrace_domains(tmp_path: Path) -> None:
    bound = bind_adapter_options(
        "rocprofv3",
        {
            "hip_trace": True,
            "kernel_trace": False,
            "memory_copy_trace": True,
            "memory_allocation_trace": True,
            "scratch_memory_trace": True,
            "marker_trace": True,
        },
        project_root=tmp_path,
    )

    invocation = build_capture_invocation(
        "rocprofv3",
        ("python", "workload.py"),
        tmp_path / "capture",
        executable="/opt/rocm/bin/rocprofv3",
        options=cast(dict[str, object], bound),
    )

    assert invocation.argv == (
        "/opt/rocm/bin/rocprofv3",
        "--output-format",
        "pftrace",
        "-o",
        "rocprofv3",
        "-d",
        str(tmp_path / "capture"),
        "--hip-trace",
        "--memory-copy-trace",
        "--memory-allocation-trace",
        "--scratch-memory-trace",
        "--marker-trace",
        "--",
        "python",
        "workload.py",
    )
    assert invocation.artifact_kinds == (ArtifactKind.EXECUTION_TRACE,)


def test_rocprofv3_options_reject_unknown_fields_and_empty_domain_set(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as unknown:
        bind_adapter_options(
            "rocprofv3",
            {"arbitrary_flags": "--sys-trace"},
            project_root=tmp_path,
        )
    assert unknown.value.code is ErrorCode.INVALID_CAPTURE_PLAN

    with pytest.raises(DomainError) as empty:
        bind_adapter_options(
            "rocprofv3",
            {
                "hip_trace": False,
                "kernel_trace": False,
                "memory_copy_trace": False,
                "memory_allocation_trace": False,
                "scratch_memory_trace": False,
                "marker_trace": False,
            },
            project_root=tmp_path,
        )
    assert empty.value.code is ErrorCode.INVALID_CAPTURE_PLAN
