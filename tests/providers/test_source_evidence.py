from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.runtime_contracts import PathSource
from flameox.stateless import AnalysisRuntime


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
