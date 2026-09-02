from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flameox.providers.benchmarks import BenchmarkProvider
from flameox.repository import EvidenceRepository
from flameox.runtime_contracts import PathSource, RequestLimits
from flameox.stateless import AnalysisRuntime

pytestmark = pytest.mark.performance


def test_comparison_accumulates_one_thousand_members_linearly() -> None:
    row_sets = [
        [
            {
                "benchmark": "operation",
                "unit": "ns",
                "is_warmup": False,
                "value_int": index + 1,
                "value_float": None,
            }
        ]
        for index in range(1_000)
    ]

    started = time.monotonic()
    result = BenchmarkProvider._compare_row_sets(
        row_sets,
        {},
        max_rows=1_000,
        provider_id="fixture",
        provider_version="1",
    )

    assert result.rows_observed == 999
    assert result.complete is True
    assert time.monotonic() - started < 5


def test_query_pins_ten_thousand_manifest_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = EvidenceRepository(tmp_path, "performance-session")
    repository.initialize()
    evidence_root = tmp_path / ".flameox" / "evidence" / "sha256"
    created_at = datetime.now(UTC).isoformat()
    bodies = [
        {
            "evidence_kind": "analysis",
            "capability_id": "artifact.preview",
            "provider": {"id": "fixture", "version": "1"},
            "inputs": [],
            "episode": {"created_at": created_at},
            "coverage": {"rows_returned": 0, "rows_observed": 0, "complete": True},
            "limitations": [],
            "data_files": [],
            "artifacts": [],
            "sequence": index,
        }
        for index in range(10_000)
    ]
    inventory = [
        evidence_root / f"{index:064x}"[:2] / f"{index:064x}" / "manifest.json"
        for index in range(10_000)
    ]
    original_glob = Path.glob

    def pinned_glob(path: Path, pattern: str) -> Iterator[Path]:
        if path == evidence_root and pattern == "*/*/manifest.json":
            return iter(inventory)
        return original_glob(path, pattern)

    def read_manifest(
        _repository: EvidenceRepository,
        path: Path,
        expected: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del expected
        index = int(path.parent.name, 16)
        return {"format_version": "1", "evidence_id": path.parent.name, "body": bodies[index]}

    monkeypatch.setattr(Path, "glob", pinned_glob)
    monkeypatch.setattr(repository, "_validate_evidence", MethodType(read_manifest, repository))

    started = time.monotonic()
    result = repository.query(capability_id="missing", limit=50)

    assert result["evidence"] == []
    assert len(result["inventory_digest"]) == 64
    assert time.monotonic() - started < 5


@pytest.mark.process
def test_nsight_continuations_reuse_one_session_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template.parquet"
    pq.write_table(
        pa.table({"start_ns": list(range(100)), "kernel": ["cached"] * 100}),
        template,
    )
    counter = tmp_path / "exports.txt"
    executable = tmp_path / "bin" / "nsys"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, shutil, sys\n"
        "arguments = sys.argv[1:]\n"
        "output = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "destination = output.with_suffix('.parquetdir')\n"
        "destination.mkdir(parents=True, exist_ok=True)\n"
        f"shutil.copyfile(pathlib.Path({str(template)!r}), "
        "destination / 'CUDA_GPU_KERN_SUM.parquet')\n"
        f"with pathlib.Path({str(counter)!r}).open('a') as stream: stream.write('1\\n')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"native-nsight-report")

    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        continuation: str | None = None
        rows = 0
        while True:
            result = runtime.analyze(
                "gpu.launches",
                [PathSource(path=str(report))],
                {},
                limits=RequestLimits(max_rows=10),
                continuation=continuation,
            )
            rows += result["coverage"]["rows_returned"]
            continuation = result["continuation"]
            if continuation is None:
                break
    finally:
        runtime.close()

    assert rows == 100
    assert counter.read_text().splitlines() == ["1"]
