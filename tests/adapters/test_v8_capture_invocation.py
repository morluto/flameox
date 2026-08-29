from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters.builtins import (
    build_capture_invocation,
    builtin_adapter,
    node_version_is_supported,
)
from flameox.domain import ArtifactKind, DomainError, ErrorCode

pytestmark = pytest.mark.unit


def test_node_cpu_prof_capture_injects_cpu_prof_flags(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "node-cpu-prof",
        ("node", "script.js", "--arg"),
        tmp_path,
        executable="/usr/bin/node",
    )

    assert invocation.artifact_kinds == (ArtifactKind.SAMPLE_PROFILE,)
    assert "--cpu-prof" in invocation.argv
    dir_flag = next(a for a in invocation.argv if a.startswith("--cpu-prof-dir="))
    assert dir_flag.startswith("--cpu-prof-dir=")
    assert invocation.argv[0] == "node"
    assert invocation.argv[-2:] == ("script.js", "--arg")


def test_node_cpu_prof_capture_preserves_workload_arguments(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "node-cpu-prof",
        ("node", "index.js", "--flag", "value"),
        tmp_path,
        executable="/usr/bin/node",
    )

    workload_part = invocation.argv[-3:]
    assert workload_part == ("index.js", "--flag", "value")


def test_node_cpu_prof_capture_rejects_empty_workload(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as failure:
        build_capture_invocation(
            "node-cpu-prof",
            (),
            tmp_path,
            executable="/usr/bin/node",
        )

    assert failure.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "Node.js" in failure.value.message


def test_node_cpu_prof_capture_rejects_non_node_workload(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as failure:
        build_capture_invocation(
            "node-cpu-prof",
            ("python", "script.py"),
            tmp_path,
            executable=None,
        )

    assert failure.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_node_heap_prof_capture_injects_heap_prof_flags(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "node-heap-prof",
        ("node", "script.js"),
        tmp_path,
        executable="/usr/bin/node",
    )

    assert invocation.artifact_kinds == (ArtifactKind.MEMORY_PROFILE,)
    assert "--heap-prof" in invocation.argv
    assert invocation.argv[0] == "node"
    assert invocation.argv[-1] == "script.js"


def test_node_heap_prof_capture_preserves_workload_arguments(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "node-heap-prof",
        ("node", "index.js", "--flag", "value"),
        tmp_path,
        executable="/usr/bin/node",
    )

    workload_part = invocation.argv[-3:]
    assert workload_part == ("index.js", "--flag", "value")


def test_node_heap_prof_capture_rejects_empty_workload(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as failure:
        build_capture_invocation(
            "node-heap-prof",
            (),
            tmp_path,
            executable="/usr/bin/node",
        )

    assert failure.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "Node.js" in failure.value.message


def test_node_v8_capture_uses_output_filename(tmp_path: Path) -> None:
    invocation_cpu = build_capture_invocation(
        "node-cpu-prof",
        ("node", "script.js"),
        tmp_path,
        executable="/usr/bin/node",
    )
    invocation_heap = build_capture_invocation(
        "node-heap-prof",
        ("node", "script.js"),
        tmp_path,
        executable="/usr/bin/node",
    )

    cpu_name_flag = next(a for a in invocation_cpu.argv if a.startswith("--cpu-prof-name="))
    assert "cpu.cpuprofile" in cpu_name_flag
    heap_name_flag = next(a for a in invocation_heap.argv if a.startswith("--heap-prof-name="))
    assert "heap.heapprofile" in heap_name_flag


def test_node_v8_adapters_preserve_failed_workload_profiles() -> None:
    cpu = builtin_adapter("node-cpu-prof")
    heap = builtin_adapter("node-heap-prof")
    assert cpu is not None and cpu.preserve_artifact_on_nonzero is True
    assert heap is not None and heap.preserve_artifact_on_nonzero is True


def test_node_v8_capture_uses_the_bound_workload_executable(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "node-cpu-prof",
        ("node", "script.js"),
        tmp_path,
        executable="/usr/bin/node",
        workload_executable="/opt/node/bin/node",
    )

    assert invocation.argv[0] == "/opt/node/bin/node"


@pytest.mark.parametrize(
    ("version", "supported"),
    [("v20.15.1", False), ("v20.16.0", True), ("v21.7.0", False), ("v22.4.0", True)],
)
def test_node_v8_version_floor(version: str, supported: bool) -> None:
    assert node_version_is_supported(version) is supported
