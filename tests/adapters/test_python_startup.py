from __future__ import annotations

from pathlib import Path

import pyperf
import pytest

from flameox.adapters import PythonStartupExtractor
from flameox.adapters.python_startup import _group_imports
from flameox.application import ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    DomainError,
    ErrorCode,
    Sensitivity,
    new_id,
)
from flameox.startup_profile import PYTHON_STARTUP_PROFILE
from flameox.storage import ArtifactStore, RunStore, Workspace

pytestmark = pytest.mark.unit


def _write_wall_samples(path: Path, *, process_count: int = 5) -> None:
    runs = [
        pyperf.Run(
            [0.020 - index * 0.001],
            metadata={
                "name": PYTHON_STARTUP_PROFILE.benchmark_name,
                "unit": "second",
                "loops": 1,
                "command_max_rss": 12_000_000 + index * 100_000,
            },
            collect_metadata=False,
        )
        for index in range(process_count)
    ]
    pyperf.BenchmarkSuite([pyperf.Benchmark(runs)]).dump(str(path), replace=True)


def test_startup_profile_disables_calibration_loops_and_warmups(tmp_path: Path) -> None:
    argv = PYTHON_STARTUP_PROFILE.pyperf_argv(
        python="/runtime/python",
        output=tmp_path / "wall.json",
        timeout_seconds=7.5,
        workload=("/workload/python", "-m", "example"),
    )

    assert argv == (
        "/runtime/python",
        "-m",
        "pyperf",
        "command",
        "--output",
        str(tmp_path / "wall.json"),
        "--processes",
        "5",
        "--values",
        "1",
        "--loops",
        "1",
        "--warmups",
        "0",
        "--timeout",
        "8",
        "--copy-env",
        "--name",
        "flameox.python_startup.wall_time",
        "--",
        "/workload/python",
        "-m",
        "example",
    )


def _startup_run(
    workspace: Workspace,
    wall_path: Path,
    import_trace_path: Path,
) -> str:
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=wall_path,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
            role="startup_wall",
            producer="pyperf",
            producer_version=pyperf.__version__,
        )
    )
    stored_trace = ArtifactStore(workspace).import_path(
        import_trace_path,
        allowed_roots=(workspace.project_root,),
        max_bytes=workspace.config.capture.max_artifact_bytes,
    )
    trace_registration = ArtifactRegistration(
        registration_id=new_id(),
        run_id=imported.run.run_id,
        artifact_id=stored_trace.content.artifact_id,
        display_name=import_trace_path.name,
        media_type="text/plain",
        kind=ArtifactKind.PYTHON_STARTUP,
        role="import_trace",
        producer="cpython",
        sensitivity=Sensitivity.INTERNAL,
    )
    updated = imported.run.validated_copy(
        update={
            "revision": imported.run.revision + 1,
            "artifacts": (*imported.run.artifacts, trace_registration),
        }
    )
    RunStore(workspace).append(updated, expected_revision=imported.run.revision)
    return imported.run.run_id


def test_python_startup_projects_native_pyperf_and_raw_importtime(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    wall = tmp_path / PYTHON_STARTUP_PROFILE.wall_output_name
    trace = tmp_path / PYTHON_STARTUP_PROFILE.import_trace_output_name
    _write_wall_samples(wall)
    trace.write_text("import time: 80 | 160 | json\n")
    run_id = _startup_run(workspace, wall, trace)

    result = PythonStartupExtractor(workspace).extract(run_id)

    assert result.sample_count == 5
    assert result.package_count == 1
    assert result.measurement_count == 13
    assert result.peak_rss_backends == ("pyperf.command_max_rss",)
    assert "uncontrolled" in result.limitations[0]
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT name, value_int, unit, dimensions['package'], "
            "dimensions['cache_semantics'] FROM measurements "
            "ORDER BY worker_run_index, name"
        ).fetchall()
    assert (
        "python_startup.wall_time",
        20_000_000,
        "ns",
        None,
        "uncontrolled_initial",
    ) in rows
    assert (
        "python_startup.import.module_count",
        1,
        "count",
        "json",
        None,
    ) in rows
    assert (
        "python_startup.import.max_cumulative_time",
        160,
        "us",
        "json",
        None,
    ) in rows


@pytest.mark.parametrize("payload", (b"", b"{truncated"))
def test_python_startup_rejects_malformed_pyperf_artifact(
    tmp_path: Path,
    payload: bytes,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    wall = tmp_path / PYTHON_STARTUP_PROFILE.wall_output_name
    trace = tmp_path / PYTHON_STARTUP_PROFILE.import_trace_output_name
    wall.write_bytes(payload)
    trace.write_text("import time: 1 | 1 | module\n")
    run_id = _startup_run(workspace, wall, trace)

    with pytest.raises(DomainError) as error:
        PythonStartupExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_python_startup_rejects_a_non_closed_pyperf_profile(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    wall = tmp_path / PYTHON_STARTUP_PROFILE.wall_output_name
    trace = tmp_path / PYTHON_STARTUP_PROFILE.import_trace_output_name
    _write_wall_samples(wall, process_count=4)
    trace.write_text("import time: 1 | 1 | module\n")
    run_id = _startup_run(workspace, wall, trace)

    with pytest.raises(DomainError) as error:
        PythonStartupExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "closed Python startup profile" in error.value.message


def test_python_startup_requires_the_separate_import_trace(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    wall = tmp_path / PYTHON_STARTUP_PROFILE.wall_output_name
    _write_wall_samples(wall)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=wall,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
            role="startup_wall",
            producer="pyperf",
            producer_version=pyperf.__version__,
        )
    )

    with pytest.raises(DomainError) as error:
        PythonStartupExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_import_grouping_accumulates_duplicate_self_time_and_keeps_maximum() -> None:
    _, packages, ignored = _group_imports(
        "import time: 10 | 100 | package.module\nimport time: 20 | 80 | package.module\n"
    )

    assert ignored == 0
    assert packages == [
        {
            "package": "package",
            "module_count": 1,
            "self_us": 30,
            "max_cumulative_us": 100,
        }
    ]
