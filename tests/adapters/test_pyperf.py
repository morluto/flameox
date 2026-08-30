from __future__ import annotations

from pathlib import Path

import pyperf
import pytest

from flameox.adapters.pyperf import PyPerfExtractor
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace

pytestmark = pytest.mark.unit


def write_suite(path: Path) -> None:
    first = pyperf.Run(
        [0.010, 0.011],
        warmups=[(2, 0.020)],
        metadata={"name": "gae", "unit": "second", "loops": 8},
        collect_metadata=False,
    )
    second = pyperf.Run(
        [0.009, 0.010],
        warmups=[(4, 0.015)],
        metadata={"name": "gae", "unit": "second", "loops": 8},
        collect_metadata=False,
    )
    pyperf.BenchmarkSuite([pyperf.Benchmark([first, second])]).dump(
        str(path),
        replace=True,
    )


def test_pyperf_extraction_preserves_worker_and_warmup_hierarchy(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "benchmark.json"
    write_suite(source)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=source,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
        )
    )

    result = PyPerfExtractor(workspace).extract(imported.run.run_id)

    assert result.measurement_count == 4
    assert result.warmup_count == 2
    assert result.limitations == ()
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT worker_id, worker_run_index, value_index, loop_count, "
            "is_warmup, value_int, unit FROM measurements "
            "ORDER BY worker_run_index, is_warmup DESC, value_index"
        ).fetchall()
    assert rows == [
        ("gae:0", 0, 0, 2, True, 20_000_000, "ns"),
        ("gae:0", 0, 0, 8, False, 10_000_000, "ns"),
        ("gae:0", 0, 1, 8, False, 11_000_000, "ns"),
        ("gae:1", 1, 0, 4, True, 15_000_000, "ns"),
        ("gae:1", 1, 0, 8, False, 9_000_000, "ns"),
        ("gae:1", 1, 1, 8, False, 10_000_000, "ns"),
    ]


@pytest.mark.parametrize("payload", (b"", b"{truncated"))
def test_pyperf_rejects_empty_and_malformed_artifacts(
    tmp_path: Path,
    payload: bytes,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "benchmark.json"
    source.write_bytes(payload)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=source,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
        )
    )

    with pytest.raises(DomainError) as error:
        PyPerfExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_pyperf_uses_the_reader_as_the_format_compatibility_boundary(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "benchmark.json"
    write_suite(source)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=source,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
            producer="pyperf",
            producer_version="999.0",
        )
    )

    result = PyPerfExtractor(workspace).extract(imported.run.run_id)

    assert result.measurement_count == 4
    assert result.limitations == ()
