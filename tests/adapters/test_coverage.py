from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from flameox.adapters import CoverageExtractor
from flameox.analysis import RecipeService
from flameox.application import ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace

_coverage = pytest.importorskip(
    "coverage", reason="optional provider unavailable: install coverage"
)
Coverage = _coverage.Coverage
CoverageData = _coverage.CoverageData
CoverageWarning = _coverage.exceptions.CoverageWarning


def test_coverage_extractor_uses_public_data_api_and_normalizes_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("x = 1\nif x:\n    x += 1\n")
    data_path = tmp_path / ".coverage"
    data = CoverageData(basename=str(data_path))
    data.set_context("test_case")
    data.add_arcs({str(source): [(1, 2), (2, 3)]})
    data.write()

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=data_path,
            kind=ArtifactKind.EXECUTION_COVERAGE,
        )
    )
    result = CoverageExtractor(workspace).extract(imported.run.run_id)

    assert result.arc_count == 2
    assert result.line_count == 3
    assert any("compatibility could not be verified" in item for item in result.limitations)
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT kind, file, line_from, line_to, context "
            "FROM observations ORDER BY kind, line_from, line_to"
        ).fetchall()
    assert all(row[1] == "module.py" for row in rows)
    assert ("branch_arc", "module.py", 1, 2, None) in rows
    assert ("line_hit", "module.py", 1, None, "test_case") in rows
    analysis = RecipeService(workspace).execution(imported.run.run_id, limit=2)
    assert analysis.returned == 2
    assert analysis.total == 5
    assert analysis.truncated is True


def test_coverage_extractor_accepts_empty_public_data_file(tmp_path: Path) -> None:
    data_path = tmp_path / ".coverage"
    coverage = Coverage(data_file=str(data_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoverageWarning)
        coverage.start()
        coverage.stop()
        coverage.save()
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=data_path,
            kind=ArtifactKind.EXECUTION_COVERAGE,
        )
    )

    result = CoverageExtractor(workspace).extract(imported.run.run_id)

    assert result.line_count == 0
    assert result.arc_count == 0


def test_coverage_extractor_rejects_truncated_data_file(tmp_path: Path) -> None:
    data_path = tmp_path / ".coverage"
    data_path.write_bytes(b"not sqlite")
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=data_path,
            kind=ArtifactKind.EXECUTION_COVERAGE,
        )
    )

    with pytest.raises(DomainError) as error:
        CoverageExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
