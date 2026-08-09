from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from flameox.adapters.compute_sanitizer import ComputeSanitizerExtractor
from flameox.application import (
    CaptureService,
    ExecutionPolicy,
    ImportArtifactRequest,
    ImportService,
)
from flameox.domain import ArtifactKind, ExecutionStatus, ValidationStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_live_compute_sanitizer_clean_and_out_of_bounds_reports(tmp_path: Path) -> None:
    sanitizer = shutil.which("compute-sanitizer")
    nvcc = shutil.which("nvcc")
    if sanitizer is None or nvcc is None:
        pytest.skip("Compute Sanitizer live proof requires compute-sanitizer and nvcc")
    executable = tmp_path / "kernel-probe"
    subprocess.run(
        (
            nvcc,
            "-lineinfo",
            "-arch=sm_86",
            str(PROJECT_ROOT / "tests" / "fixtures" / "compute_sanitizer" / "kernel_probe.cu"),
            "-o",
            str(executable),
        ),
        check=True,
        timeout=60,
    )
    workspace = Workspace.initialize(tmp_path)

    def collect(name: str, *arguments: str) -> tuple[int, int]:
        report = tmp_path / f"{name}.xml"
        process = subprocess.run(
            (
                sanitizer,
                "--tool",
                "memcheck",
                "--xml",
                "--save",
                str(report),
                "--error-exitcode",
                "86",
                "--target-processes",
                "application-only",
                str(executable),
                *arguments,
            ),
            check=False,
            capture_output=True,
            timeout=60,
        )
        imported = ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=report,
                kind=ArtifactKind.SANITIZER_REPORT,
                producer="compute-sanitizer",
                producer_version="live",
            )
        )
        extracted = ComputeSanitizerExtractor(workspace).extract(imported.run.run_id)
        return process.returncode, extracted.finding_count

    assert collect("clean") == (0, 0)
    finding_exit, finding_count = collect("out-of-bounds", "oob")
    assert finding_exit == 86
    assert finding_count > 0


@pytest.mark.anyio
async def test_live_compute_sanitizer_wrapper_preserves_finding_as_validation(
    tmp_path: Path,
) -> None:
    sanitizer = shutil.which("compute-sanitizer")
    nvcc = shutil.which("nvcc")
    if sanitizer is None or nvcc is None:
        pytest.skip("Compute Sanitizer live proof requires compute-sanitizer and nvcc")
    executable = tmp_path / "kernel-probe"
    subprocess.run(
        (
            nvcc,
            "-lineinfo",
            "-arch=sm_86",
            str(PROJECT_ROOT / "tests" / "fixtures" / "compute_sanitizer" / "kernel_probe.cu"),
            "-o",
            str(executable),
        ),
        check=True,
        timeout=60,
    )
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n"
        "[workloads.out_of_bounds]\n"
        f"argv = [{str(executable)!r}, 'oob']\n"
        "cwd = '.'\n"
        "timeout_seconds = 30\n"
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="out_of_bounds",
        adapter="compute-sanitizer",
        adapter_options={"tool": "memcheck", "finding_exit_code": 86},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.validation_status is ValidationStatus.FAILED
    report = next(
        item for item in result.run.artifacts if item.kind is ArtifactKind.SANITIZER_REPORT
    )
    extracted = ComputeSanitizerExtractor(workspace).extract(result.run.run_id)
    assert extracted.artifact_id == report.artifact_id
    assert extracted.finding_count > 0
