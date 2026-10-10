from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import EvidenceSource, PathSource, RequestLimits, RuntimeFailure


@pytest.mark.parametrize("loop", [False, True])
def test_sarif_export_uses_explicit_source_root_and_preserves_containment(
    tmp_path: Path, loop: bool
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    bad_path = tmp_path / "outside.py"
    if loop:
        bad_path = root / "loop.py"
        bad_path.symlink_to("loop.py")
    report = tmp_path / "export.sarif"
    report.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "ruff", "version": "0.16.0"}},
                        "results": [
                            {
                                "ruleId": "PERF401",
                                "message": {"text": "Use a list comprehension"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": path.as_uri()},
                                            "region": {"startLine": 3},
                                        }
                                    }
                                ],
                            }
                            for path in (root / "work.py", bad_path)
                        ],
                    }
                ],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        result = runtime.analyze(
            "static.performance_candidates",
            [PathSource(path=str(report), format="sarif")],
            {"source_root": str(root)},
        )
        if loop:
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze(
                    "static.performance_candidates",
                    [PathSource(path=str(report))],
                    {"source_root": str(bad_path)},
                )
            assert failure.value.code == "INVALID_INPUT"
    finally:
        runtime.close()
    metrics = result["blocks"][0]["values"]
    assert metrics["normalized_count"] == 1
    assert metrics["invalid_count"] == 1
    assert result["blocks"][1]["rows"][0]["relative_path"] == "work.py"
    assert not (root / "work.py").exists()


@pytest.mark.golden
def test_sarif_candidates_are_scoped_to_project_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "report.sarif"
    artifact.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "scanner", "version": "1.2.3"}},
                        "results": [
                            {
                                "ruleId": "slow-loop",
                                "level": "warning",
                                "message": {"text": "Loop is a performance candidate"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "src/slow.py"},
                                            "region": {"startLine": 7},
                                        }
                                    }
                                ],
                            },
                            {
                                "ruleId": "ignored",
                                "message": {"text": "Excluded candidate"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "tests/test_slow.py"},
                                            "region": {"startLine": 3},
                                        }
                                    }
                                ],
                            },
                            {
                                "ruleId": "invalid-confidence",
                                "message": {"text": "Invalid native confidence"},
                                "properties": {"confidence": 10**400},
                                "locations": [
                                    {"physicalLocation": {"artifactLocation": {"uri": "src/a.py"}}}
                                ],
                            },
                        ],
                    }
                ],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "static.performance_candidates",
            [PathSource(path=str(artifact))],
            {"include_paths": ["src/*"]},
        )
    finally:
        runtime.close()

    assert result["provider"] == {"id": "sarif", "version": "2.1.0"}
    assert result["blocks"][0]["values"]["result_count"] == 3
    assert result["blocks"][0]["values"]["invalid_count"] == 1
    assert result["blocks"][0]["values"]["excluded_count"] == 1
    assert result["blocks"][1]["rows"][0]["relative_path"] == "src/slow.py"
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.golden
def test_truncated_sarif_never_reports_complete_coverage(tmp_path: Path) -> None:
    artifact = tmp_path / "truncated.sarif"
    payload = json.dumps(
        {
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {"driver": {"name": "scanner"}},
                    "results": [
                        {
                            "ruleId": "slow-loop",
                            "message": {"text": "candidate"},
                            "locations": [
                                {"physicalLocation": {"artifactLocation": {"uri": "src/slow.py"}}}
                            ],
                        }
                    ],
                }
            ],
        }
    )
    artifact.write_text(payload[:-1])
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "static.performance_candidates",
            [PathSource(path=str(artifact), format="sarif")],
            {},
        )
    finally:
        runtime.close()

    assert result["blocks"][1]["rows"]
    assert result["coverage"]["complete"] is False
    assert any("stopped before the document ended" in item for item in result["limitations"])


def test_sarif_keeps_literal_keys_distinct_from_candidate_locations(tmp_path: Path) -> None:
    candidate = {
        "message": {"text": "native candidate"},
        "partialFingerprints": {"hash.v1": "native-hash"},
        "locations": [{"physicalLocation": {"artifactLocation": {"uri": "code.py"}}}],
    }
    run = {
        "tool": {"driver": {"name": "scanner"}},
        "results": [candidate],
        "results.item": {**candidate, "message": {"text": "forged candidate"}},
    }
    artifact = tmp_path / "native.sarif"
    artifact.write_text(json.dumps({"version": "2.1.0", "runs": [run]}))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "static.performance_candidates",
            [PathSource(path=str(artifact))],
            {"source_root": str(tmp_path)},
        )
        assert result["blocks"][0]["values"]["result_count"] == 1
        row = result["blocks"][1]["rows"][0]
        assert row["message"] == "native candidate"
        assert row["provider_fingerprint"] == "native-hash"
    finally:
        runtime.close()


@pytest.mark.parametrize("runs", [None, {}, [42], [{"results": {}}], [{"results": [42]}], []])
def test_sarif_malformed_collections_never_claim_complete_empty_evidence(
    tmp_path: Path, runs: object
) -> None:
    artifact = tmp_path / "native.sarif"
    document: dict[str, object] = {"version": "2.1.0"}
    if runs is not None:
        document["runs"] = runs
    artifact.write_text(json.dumps(document))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "static.performance_candidates",
            [PathSource(path=str(artifact))],
            {"source_root": str(tmp_path)},
        )
        assert result["coverage"]["complete"] is (runs == [])
        if runs != []:
            assert result["limitations"]
    finally:
        runtime.close()


def test_sarif_preserved_next_page_retains_the_original_implicit_source_root(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "native.sarif"
    artifact.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "scanner"}},
                        "results": [
                            {
                                "message": {"text": name},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": (tmp_path / name).as_uri()}
                                        }
                                    }
                                ],
                            }
                            for name in ("first.py", "second.py")
                        ],
                    }
                ],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        first = runtime.analyze(
            "static.performance_candidates",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=1),
        )
        runtime.preserve_evidence(first["analysis_id"])
        handoff = runtime.next_analysis_request(first)
        assert handoff is not None
        assert handoff["options"]["source_root"] == str(tmp_path.resolve())
    finally:
        runtime.close()
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        second = reopened.analyze(
            handoff["capability_id"],
            [EvidenceSource.model_validate(item) for item in handoff["sources"]],
            handoff["options"],
            limits=RequestLimits.model_validate(handoff["limits"]),
            continuation=handoff["continuation"],
        )
        assert second["blocks"][1]["rows"][0]["relative_path"] == "second.py"
        assert second["coverage"]["complete"] is True
    finally:
        reopened.close()


def test_sarif_candidate_coordinates_resolve_within_each_native_run(tmp_path: Path) -> None:
    native = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "scanner"}},
                "results": [
                    {
                        "message": {"text": f"run {run} result {result}"},
                        "locations": [
                            {"physicalLocation": {"artifactLocation": {"uri": "work.py"}}}
                        ],
                    }
                    for result in range(count)
                ],
            }
            for run, count in enumerate((2, 0, 1, 2))
        ],
    }
    artifact = tmp_path / "native.sarif"
    artifact.write_text(json.dumps(native))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        analysis = runtime.analyze(
            "static.performance_candidates", [PathSource(path=str(artifact))], {}
        )
        rows = analysis["blocks"][1]["rows"]
        assert [(row["run_index"], row["result_index"]) for row in rows] == [
            (0, 0),
            (0, 1),
            (2, 0),
            (3, 0),
            (3, 1),
        ]
        for row in rows:
            result = native["runs"][row["run_index"]]["results"][row["result_index"]]
            assert row["message"] == result["message"]["text"]
    finally:
        runtime.close()


@pytest.mark.parametrize("number", ["1e3", "1e10000"])
def test_sarif_decimal_coordinates_respect_integer_conversion_bounds(
    tmp_path: Path, number: str
) -> None:
    artifact = tmp_path / "native.sarif"
    document = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "scanner"}},
                "invocations": [{"exitCode": "DECIMAL"}],
                "results": [
                    {
                        "message": {"text": "native candidate"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "work.py"},
                                    "region": {"startLine": "DECIMAL"},
                                }
                            }
                        ],
                    }
                ],
            }
        ],
    }
    artifact.write_text(json.dumps(document).replace('"DECIMAL"', number))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "static.performance_candidates", [PathSource(path=str(artifact))], {}
        )
        if number == "1e3":
            assert result["blocks"][1]["rows"][0]["start_line"] == 1000
            assert result["blocks"][0]["values"]["exit_status"] == 1000
        else:
            assert result["blocks"][1]["rows"] == []
            assert result["blocks"][0]["values"]["invalid_count"] == 1
    finally:
        runtime.close()
