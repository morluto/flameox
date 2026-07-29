from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters import PythonStartupExtractor
from flameox.application import ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.collectors.python_startup import _group_imports
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace


def _write_startup(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "flameox.python-startup.v1",
                "started_at_ns": 100,
                "finished_at_ns": 500,
                "python_executable": "python",
                "python_version": "3.12",
                "workload_argv": ["python", "app.py"],
                "semantics": {
                    "interpreter": "fresh process per sample",
                    "initial_cache": "uncontrolled; no OS caches were dropped",
                    "later_cache": "warm process restart",
                    "wall_time": "uninstrumented workload execution",
                    "import_trace": "separate instrumented workload execution",
                },
                "samples": [
                    {
                        "index": 0,
                        "cache_semantics": "uncontrolled_initial",
                        "fresh_interpreter": True,
                        "duration_ns": 20_000_000,
                        "peak_rss_bytes": 12_000_000,
                        "peak_rss_backend": "wait4_ru_maxrss",
                        "exit_code": 0,
                    },
                    {
                        "index": 1,
                        "cache_semantics": "warm_process_restart",
                        "fresh_interpreter": True,
                        "duration_ns": 15_000_000,
                        "peak_rss_bytes": 12_500_000,
                        "peak_rss_backend": "wait4_ru_maxrss",
                        "exit_code": 0,
                    },
                ],
                "import_trace": {
                    "exit_code": 0,
                    "raw_importtime": "import time: 80 | 160 | json\n",
                    "packages": [
                        {
                            "package": "json",
                            "module_count": 4,
                            "self_us": 80,
                            "max_cumulative_us": 160,
                        }
                    ],
                    "unparsed_importtime_lines": 0,
                },
            }
        )
    )


def test_python_startup_extracts_samples_and_package_costs(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "startup.json"
    _write_startup(source)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.PYTHON_STARTUP)
    )

    result = PythonStartupExtractor(workspace).extract(imported.run.run_id)

    assert result.sample_count == 2
    assert result.package_count == 1
    assert result.measurement_count == 7
    assert result.peak_rss_backends == ("wait4_ru_maxrss",)
    assert "uncontrolled" in result.limitations[0]
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT name, value_int, unit, dimensions['package'] "
            "FROM measurements ORDER BY value_index, name"
        ).fetchall()
    assert ("python_startup.wall_time", 20_000_000, "ns", None) in rows
    assert ("python_startup.import.module_count", 4, "count", "json") in rows
    assert ("python_startup.import.max_cumulative_time", 160, "us", "json") in rows


@pytest.mark.parametrize("payload", ("", "{}", "{broken", "[]", "null"))
def test_python_startup_rejects_malformed_artifacts(tmp_path: Path, payload: str) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "startup.json"
    source.write_text(payload)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.PYTHON_STARTUP)
    )

    with pytest.raises(DomainError) as error:
        PythonStartupExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_python_startup_rejects_malformed_package_container(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "startup.json"
    _write_startup(source)
    payload = json.loads(source.read_text())
    payload["import_trace"]["packages"] = None
    source.write_text(json.dumps(payload))
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.PYTHON_STARTUP)
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
