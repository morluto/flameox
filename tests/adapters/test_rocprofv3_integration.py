from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from flameox.adapters import PerfettoExtractor
from flameox.analysis import RecipeService
from flameox.application import (
    CaptureService,
    ExecutionPolicy,
    ImportArtifactRequest,
    ImportService,
)
from flameox.domain import ArtifactKind, CaptureStatus, ExecutionStatus
from flameox.storage import ArtifactStore, Workspace
from tests.support.capture import disable_containment
from tests.support.providers import require_trace_processor

_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "rocprofv3"


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

    result = await service.execute(plan.plan_token)

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
    config = workspace.config.model_copy(
        update={
            "analysis": workspace.config.analysis.model_copy(
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


def _write_fake_rocprofv3(path: Path, *, exit_code: int) -> None:
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
