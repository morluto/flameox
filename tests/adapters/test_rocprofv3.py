from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from flameox.adapters import PerfettoExtractor
from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.options import bind_adapter_options
from flameox.analysis import RecipeService
from flameox.application import (
    CaptureService,
    ExecutionPolicy,
    ImportArtifactRequest,
    ImportService,
)
from flameox.domain import (
    ArtifactKind,
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
)
from flameox.storage import ArtifactStore, Workspace
from tests.support.capture import disable_containment
from tests.support.providers import require_trace_processor

_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "rocprofv3"


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


def test_rocprofv3_process_fixture_writes_expected_pftrace_path(tmp_path: Path) -> None:
    executable = _write_fake_rocprofv3(tmp_path / "rocprofv3", exit_code=0)
    output = tmp_path / "output"
    output.mkdir()
    invocation = build_capture_invocation(
        "rocprofv3",
        ("/bin/true",),
        output,
        executable=str(executable),
        options={"kernel_trace": True},
    )

    completed = subprocess.run(invocation.argv, check=False, capture_output=True)

    assert completed.returncode == 0
    assert (output / "rocprofv3_results.pftrace").read_bytes() == b"fixture-pftrace"
    assert completed.stdout == b"rocprof fixture stdout"
    assert completed.stderr == b"rocprof fixture stderr"


@pytest.mark.anyio
async def test_rocprofv3_capture_preserves_partial_pftrace_on_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_directory = tmp_path / "bin"
    executable_directory.mkdir()
    _write_fake_rocprofv3(executable_directory / "rocprofv3", exit_code=7)
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
        adapter="rocprofv3",
        adapter_options={"kernel_trace": True, "hip_trace": True},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    trace = next(
        registration
        for registration in result.run.artifacts
        if registration.kind is ArtifactKind.EXECUTION_TRACE
    )
    assert trace.producer == "rocprofv3"
    assert ArtifactStore(workspace).get(trace.artifact_id).payload_path.read_bytes() == (
        b"fixture-pftrace"
    )
    roles = {registration.role for registration in result.run.artifacts}
    assert {"primary", "stdout", "stderr"} <= roles
    assert any(detail.code == "nonzero_exit" for detail in result.run.limitation_details)


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_perfetto
async def test_project_owned_rocm_shaped_trace_reaches_accelerator_summary(
    tmp_path: Path,
) -> None:
    """Exercise the shared path without presenting this fixture as rocprofv3 output."""
    binary = require_trace_processor()
    trace = tmp_path / "project-owned-rocm-shaped-perfetto.json"
    shutil.copyfile(_FIXTURE_ROOT / trace.name, trace)
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "analysis": workspace.config.analysis.validated_copy(
                update={"trace_processor_path": str(binary)}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="project-owned-rocm-shaped-perfetto",
        )
    )

    extracted = await PerfettoExtractor(workspace).extract(imported.run.run_id)
    summary = RecipeService(workspace).accelerator_launches(
        imported.run.run_id,
        phase="decode",
    )

    assert extracted.slice_count == 3
    assert extracted.trace_event_count == 3
    assert summary.coverage == {
        "runtime_launches": True,
        "accelerator_kernels": True,
        "phase_annotations": True,
        "correlation_ids": True,
        "host_to_device_correlation": True,
        "stream_identity": True,
    }
    assert summary.total == 1
    region = summary.regions[0]
    assert region.direct_launch_count == 1
    assert region.kernel_count == 2
    assert region.kernel_duration_ns == 8_000
    assert region.correlated_kernel_count == 1
    assert region.idle_gap_total_ns == 3_000
    assert region.streams[0].device == "amd:0"
    assert region.streams[0].stream == "7"


def _write_fake_rocprofv3(path: Path, *, exit_code: int) -> Path:
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        "  printf 'rocprofv3 fixture 7.0\\n'\n"
        "  exit 0\n"
        "fi\n"
        "output_name=''\n"
        "output_directory=''\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        "    -o) output_name=$2; shift 2 ;;\n"
        "    -d) output_directory=$2; shift 2 ;;\n"
        "    --) break ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        "printf 'fixture-pftrace' > \"$output_directory/${output_name}_results.pftrace\"\n"
        "printf 'rocprof fixture stdout'\n"
        "printf 'rocprof fixture stderr' >&2\n"
        f"exit {exit_code}\n"
    )
    path.chmod(0o755)
    return path
