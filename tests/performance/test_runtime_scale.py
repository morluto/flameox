from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flameox.evidence_models import EvidenceManifest
from flameox.providers.benchmarks import BenchmarkProvider
from flameox.repository import EvidenceRepository
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource, RequestLimits

pytestmark = pytest.mark.performance


def test_comparison_preserves_one_thousand_member_identities() -> None:
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

    result = BenchmarkProvider._compare_row_sets(
        row_sets,
        {},
        max_rows=1_000,
        provider_id="fixture",
        provider_version="1",
    )

    assert result.rows_observed == 999
    assert result.complete is True
    rows = result.blocks[1]["rows"]
    assert {
        row["candidate_index"]: (
            row["baseline_index"],
            row["baseline_mean"],
            row["candidate_mean"],
            row["ratio"],
        )
        for row in rows
    } == {index: (0, 1.0, float(index + 1), float(index + 1)) for index in range(1, 1_000)}
    assert {row["benchmark"] for row in rows} == {"operation"}


def test_query_pins_ten_thousand_manifest_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "sample.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        manifest = EvidenceManifest.model_validate(runtime.read_evidence(preserved["evidence_id"]))
        repository = runtime.repository
    finally:
        runtime.close()
    evidence_root = tmp_path / ".flameox" / "evidence" / "sha256"
    inventory = [
        evidence_root / f"{index:064x}"[:2] / f"{index:064x}" / "manifest.json"
        for index in range(10_000)
    ]
    original_glob = Path.glob
    visited = 0

    def pinned_glob(path: Path, pattern: str) -> Iterator[Path]:
        if path == evidence_root and pattern == "*/*/manifest.json":
            return iter(inventory)
        return original_glob(path, pattern)

    def read_manifest(
        _repository: EvidenceRepository,
        bundle: Path,
        expected: dict[str, Any] | None = None,
    ) -> EvidenceManifest:
        nonlocal visited
        del expected
        visited += 1
        # Isolate inventory/query scaling from filesystem hashing and schema parsing.
        return manifest.model_copy(update={"evidence_id": bundle.name})

    monkeypatch.setattr(Path, "glob", pinned_glob)
    monkeypatch.setattr(repository, "_validate_evidence", MethodType(read_manifest, repository))

    result = repository.query(capability_id="missing", limit=50)

    assert result["evidence"] == []
    assert visited == 10_000
    assert len(result["inventory_digest"]) == 64
    first = repository.query(limit=50)
    second = repository.query(limit=50, cursor=first["continuation"])
    assert [item["evidence_id"] for item in first["evidence"]] == [
        f"{index:064x}" for index in range(50)
    ]
    assert [item["evidence_id"] for item in second["evidence"]] == [
        f"{index:064x}" for index in range(50, 100)
    ]
    assert first["inventory_digest"] == second["inventory_digest"] == result["inventory_digest"]


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
