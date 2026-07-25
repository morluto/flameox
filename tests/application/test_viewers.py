from __future__ import annotations

from pathlib import Path

from flamo.application import (
    ImportArtifactRequest,
    ImportService,
    NativeViewerService,
)
from flamo.catalog import Catalog
from flamo.domain import ArtifactKind
from flamo.storage import Workspace


def test_benchmark_artifact_uses_pyperf_viewer(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    artifact_path = tmp_path / "benchmark.json"
    artifact_path.write_text("{}")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=artifact_path,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
        )
    )

    plan = NativeViewerService(workspace).plan(
        imported.run.artifacts[0].artifact_id
    )

    assert plan.viewer == "pyperf show"
    assert plan.argv[1] == "show"
