from __future__ import annotations

import warnings
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pytest

from flameox.adapters.coverage import CoverageExtractor
from flameox.analysis import RecipeService
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace

pytestmark = pytest.mark.integration

_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "coverage"
_SUPPORTED_FIXTURES = (
    pytest.param("coverage-7.14.sqlite", "7.14.2", id="coverage-7.14"),
    pytest.param("coverage-7.15.sqlite", "7.15.2", id="coverage-7.15"),
)


def test_coverage_extractor_uses_public_data_api_and_normalizes_paths(
    tmp_path: Path,
) -> None:
    coverage_module = cast(Any, import_module("coverage"))
    CoverageData = coverage_module.CoverageData
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
            producer="coverage",
            producer_version=version("coverage"),
        )
    )
    result = CoverageExtractor(workspace).extract(imported.run.run_id)

    assert result.arc_count == 2
    assert result.line_count == 3
    assert result.producer == "coverage"
    assert result.producer_version == version("coverage")
    assert result.reader_version == version("coverage")
    assert result.limitations == ()
    generation = workspace.corpus.read_generation(
        next(
            generation_id
            for generation_id in workspace.corpus.read_head().generation_ids
            if workspace.corpus.read_generation(generation_id).publisher == "coverage"
        )
    )
    assert generation.publisher_version == result.reader_version
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
    coverage_module = cast(Any, import_module("coverage"))
    Coverage = coverage_module.Coverage
    CoverageWarning = coverage_module.exceptions.CoverageWarning
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
            producer="coverage",
            producer_version=version("coverage"),
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
            producer="coverage",
            producer_version=version("coverage"),
        )
    )

    with pytest.raises(DomainError) as error:
        CoverageExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


@pytest.mark.parametrize("filename,producer_version", _SUPPORTED_FIXTURES)
def test_coverage_extractor_reads_provider_generated_minor_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    producer_version: str,
) -> None:
    workspace = Workspace.initialize(_FIXTURE_ROOT, workspace_root=tmp_path / "workspace")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=_FIXTURE_ROOT / filename,
            kind=ArtifactKind.EXECUTION_COVERAGE,
            producer="coverage",
            producer_version=producer_version,
        )
    )

    monkeypatch.chdir(_FIXTURE_ROOT)
    result = CoverageExtractor(workspace).extract(imported.run.run_id)

    assert result.producer == "coverage"
    assert result.producer_version == producer_version
    assert result.reader_version == version("coverage")
    assert result.line_count == 7
    assert result.arc_count == 9
    registration = next(
        item for item in imported.run.artifacts if item.kind is ArtifactKind.EXECUTION_COVERAGE
    )
    assert registration.producer == result.producer
    assert registration.producer_version == result.producer_version
    generation = next(
        workspace.corpus.read_generation(generation_id)
        for generation_id in workspace.corpus.read_head().generation_ids
        if workspace.corpus.read_generation(generation_id).publisher == "coverage"
    )
    assert generation.publisher == "coverage"
    assert generation.publisher_version == result.reader_version
    assert generation.input_artifact_ids == (registration.artifact_id,)


def test_coverage_extractor_rejects_unsupported_provider_before_parsing(tmp_path: Path) -> None:
    workspace = Workspace.initialize(_FIXTURE_ROOT, workspace_root=tmp_path / "workspace")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=_FIXTURE_ROOT / "coverage-7.13.sqlite",
            kind=ArtifactKind.EXECUTION_COVERAGE,
            producer="coverage",
            producer_version="7.13.4",
        )
    )

    with pytest.raises(DomainError) as error:
        CoverageExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ADAPTER_INCOMPATIBLE
    assert error.value.details == {
        "producer": "coverage",
        "producer_version": "7.13.4",
        "reader": "coverage",
        "reader_version": version("coverage"),
        "supported_requirement": "coverage<8,>=7.14",
    }
    assert error.value.remediation == (
        "Recapture with coverage>=7.14,<8 in the declared workload interpreter, then extract "
        "the new run; the native artifact remains preserved but is not normalized by this "
        "reader.",
    )


def test_coverage_extractor_requires_explicit_producer_identity(tmp_path: Path) -> None:
    workspace = Workspace.initialize(_FIXTURE_ROOT, workspace_root=tmp_path / "workspace")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=_FIXTURE_ROOT / "coverage-7.15.sqlite",
            kind=ArtifactKind.EXECUTION_COVERAGE,
        )
    )

    with pytest.raises(DomainError) as error:
        CoverageExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ADAPTER_INCOMPATIBLE
    assert error.value.details["producer"] == "flameox.import"
    assert error.value.details["producer_version"] is None
    assert "producer identity" in error.value.remediation[0]
