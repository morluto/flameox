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


@pytest.fixture(scope="module")
def compute_sanitizer_probe(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, str, Path]:
    sanitizer = shutil.which("compute-sanitizer")
    nvcc = shutil.which("nvcc")
    assert sanitizer is not None
    assert nvcc is not None
    version_probe = subprocess.run(
        (sanitizer, "--version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    producer_version = next(
        line.strip()
        for line in (version_probe.stdout + version_probe.stderr).splitlines()
        if line.strip().casefold().startswith("version ")
    )
    executable = tmp_path_factory.mktemp("compute-sanitizer-probe") / "kernel-probe"
    subprocess.run(
        (
            nvcc,
            "-lineinfo",
            "-arch=native",
            str(PROJECT_ROOT / "tests" / "fixtures" / "compute_sanitizer" / "kernel_probe.cu"),
            "-o",
            str(executable),
        ),
        check=True,
        timeout=60,
    )
    return sanitizer, producer_version, executable


def test_live_compute_sanitizer_clean_report_extracts_without_findings(
    tmp_path: Path,
    compute_sanitizer_probe: tuple[str, str, Path],
) -> None:
    sanitizer, producer_version, executable = compute_sanitizer_probe
    workspace = Workspace.initialize(tmp_path)
    report = tmp_path / "clean.xml"
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
            producer_version=producer_version,
        )
    )

    extracted = ComputeSanitizerExtractor(workspace).extract(imported.run.run_id)

    assert process.returncode == 0
    assert extracted.finding_count == 0
    assert extracted.status == "clean"


@pytest.mark.anyio
async def test_live_compute_sanitizer_wrapper_preserves_finding_as_validation(
    tmp_path: Path,
    compute_sanitizer_probe: tuple[str, str, Path],
) -> None:
    _, _, executable = compute_sanitizer_probe
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
