from __future__ import annotations

import tracemalloc
from pathlib import Path

import pytest

from flameox.providers.xctrace import XctraceProvider
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource

pytestmark = [pytest.mark.performance, pytest.mark.integration]


@pytest.mark.serial
def test_xctrace_parser_memory_does_not_grow_with_completed_xml_siblings(tmp_path: Path) -> None:
    peaks: list[int] = []
    for count in (50_000, 200_000):
        artifact = tmp_path / f"toc-{count}.xml"
        artifact.write_text("<trace>" + '<run name="sample"/>' * count + "</trace>")
        tracemalloc.start()
        try:
            result = XctraceProvider.analyze(artifact, max_rows=1, provider_version="golden")
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert result.rows_observed == count + 1
        assert result.complete is False
        assert result.blocks[1]["rows"] == [
            {"element": "run", "attributes": {"name": "sample"}, "text": None}
        ]
    assert peaks[1] < peaks[0] * 2, peaks


def test_query_pages_one_thousand_real_immutable_manifests(tmp_path: Path) -> None:
    artifact = tmp_path / "input.txt"
    store = tmp_path / "evidence"
    expected: set[str] = set()
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        for index in range(1_000):
            artifact.write_text(f"measurement {index}\n")
            result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            expected.add(runtime.preserve_evidence(result["analysis_id"])["evidence_id"])
    finally:
        runtime.close()

    assert len(expected) == 1_000
    assert len(list((store / "evidence" / "sha256").glob("*/*/manifest.json"))) == 1_000
    reopened = AnalysisRuntime(evidence_directory=store)
    try:
        page = reopened.query_evidence(limit=200)
        actual = [item["evidence_id"] for item in page["evidence"]]
        inventory_digest = page["inventory_digest"]
        while page["continuation"] is not None:
            page = reopened.query_evidence(limit=200, cursor=page["continuation"])
            assert page["inventory_digest"] == inventory_digest
            actual.extend(item["evidence_id"] for item in page["evidence"])
        assert len(actual) == 1_000
        assert set(actual) == expected
        assert reopened.query_evidence(capability_id="missing")["evidence"] == []
    finally:
        reopened.close()
