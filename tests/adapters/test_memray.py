from __future__ import annotations

from pathlib import Path

import memray

from flamo.adapters import MemrayExtractor
from flamo.analysis import RecipeService
from flamo.application import ImportArtifactRequest, ImportService
from flamo.catalog import Catalog
from flamo.domain import ArtifactKind
from flamo.storage import Workspace


def test_memray_extractor_preserves_native_capture_and_names_memory_concepts(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        retained = bytearray(100_000)
    assert len(retained) == 100_000

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
        )
    )
    result = MemrayExtractor(workspace).extract(imported.run.run_id)

    assert result.peak_memory_bytes >= 100_000
    assert result.retained_end_bytes >= 100_000
    assert result.total_allocations >= 1
    assert result.frame_count >= 1
    with Catalog(workspace).open_snapshot() as snapshot:
        measurements = snapshot.execute(
            "SELECT name, value_int, unit FROM measurements ORDER BY name"
        ).fetchall()
        frames = snapshot.execute(
            "SELECT metric, sum(self_value), sum(inclusive_value) "
            "FROM frame_measurements GROUP BY metric ORDER BY metric"
        ).fetchall()
    assert ("memory.peak", result.peak_memory_bytes, "bytes") in measurements
    assert ("memory.retained_end", result.retained_end_bytes, "bytes") in measurements
    assert {row[0] for row in frames} == {
        "memory.high_watermark",
        "memory.retained_end",
    }
    analysis = RecipeService(workspace).memory(imported.run.run_id)
    assert {item.name for item in analysis.measurements} == {
        "memory.peak",
        "memory.retained_end",
        "memory.total_allocations",
    }
    assert analysis.hotspots
