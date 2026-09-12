from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    PathSource,
    RuntimeFailure,
)


def test_sarif_export_uses_explicit_source_root_and_preserves_containment(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
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
                            for path in (root / "work.py", tmp_path / "outside.py")
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
    finally:
        runtime.close()
    metrics = result["blocks"][0]["values"]
    assert metrics["normalized_count"] == 1
    assert metrics["invalid_count"] == 1
    assert result["blocks"][1]["rows"][0]["relative_path"] == "work.py"
    assert not (root / "work.py").exists()


@pytest.mark.parametrize("root", ["relative/project", "\x00"])
def test_sarif_source_root_rejects_ambiguous_paths(tmp_path: Path, root: str) -> None:
    runtime = AnalysisRuntime()
    try:
        with pytest.raises(ValidationError, match="source_root must be an absolute path"):
            runtime.analyze(
                "static.performance_candidates",
                [PathSource(path=str(tmp_path / "unused.sarif"))],
                {"source_root": root},
            )
    finally:
        runtime.close()


@pytest.mark.unit
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
    assert result["blocks"][0]["values"]["result_count"] == 2
    assert result["blocks"][0]["values"]["excluded_count"] == 1
    assert result["blocks"][1]["rows"][0]["relative_path"] == "src/slow.py"
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.unit
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


@pytest.mark.unit
def test_semantic_observations_reject_unknown_fields(tmp_path: Path) -> None:
    events = tmp_path / "observations.jsonl"
    events.write_text(
        json.dumps(
            {
                "name": "phase",
                "phase": None,
                "monotonic_ns": 1,
                "values": {},
                "unexpected": True,
            }
        )
        + "\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "failures.summary",
                [PathSource(path=str(events), format="observations")],
                {},
            )
    finally:
        runtime.close()

    assert failure.value.code == "DECODE_FAILURE"
