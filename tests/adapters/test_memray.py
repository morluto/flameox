from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from flameox.adapters import MemrayExtractor
from flameox.analysis import RecipeService
from flameox.application import ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace

pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.requires_memray]


def _memray_module() -> ModuleType:
    return cast(
        ModuleType,
        pytest.importorskip("memray", reason="optional provider unavailable: install memray"),
    )


def test_memray_extractor_preserves_native_capture_and_names_memory_concepts(
    tmp_path: Path,
) -> None:
    memray = _memray_module()
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
    progress: list[tuple[str, int, int | None]] = []
    result = MemrayExtractor(workspace).extract(
        imported.run.run_id,
        progress=lambda phase, completed, total: progress.append((phase, completed, total)),
    )

    assert result.peak_memory_bytes >= 100_000
    assert result.retained_end_bytes >= 100_000
    assert result.total_allocations >= 1
    assert result.frame_count >= 1
    assert any("compatibility could not be verified" in item for item in result.limitations)
    assert {phase for phase, _completed, _total in progress} == {
        "reading_high_watermark",
        "reading_retained_end",
        "aggregating_high_watermark",
        "aggregating_retained_end",
    }
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
    pinned_commit_id = workspace.corpus.read_head().commit_id
    GenerationPublisher(workspace).publish_rows(
        {
            "frames": [
                {
                    "frame_id": "later-frame",
                    "language": "python",
                    "function": "published_later",
                    "module": None,
                    "file": "later.py",
                    "line": 1,
                    "column": None,
                    "address": None,
                    "build_id": None,
                    "module_relative_address": None,
                    "inline_chain_id": None,
                    "source_state_id": None,
                    "artifact_id": imported.artifact_id,
                    "inlined": False,
                    "symbolization": "complete",
                }
            ],
            "frame_measurements": [
                {
                    "run_id": imported.run.run_id,
                    "artifact_id": imported.artifact_id,
                    "frame_id": "later-frame",
                    "metric": "memory.retained_end",
                    "self_value": 1,
                    "inclusive_value": 1,
                    "unit": "bytes",
                    "sample_count": 1,
                    "thread_name": None,
                    "process_name": None,
                    "phase": None,
                }
            ],
        },
        publisher="pinned-recipe-regression",
        publisher_version="1",
        input_run_ids=(imported.run.run_id,),
        input_artifact_ids=(imported.artifact_id,),
    )

    analysis = RecipeService(
        workspace,
        snapshot_handle=Catalog(workspace).pin(pinned_commit_id),
    ).memory(
        imported.run.run_id,
        corpus_commit_id=pinned_commit_id,
    )
    assert analysis.corpus_commit_id == pinned_commit_id
    assert {item.name for item in analysis.measurements} == {
        "memory.peak",
        "memory.retained_end",
        "memory.total_allocations",
    }
    assert analysis.hotspots
    assert all(item.frame_id != "later-frame" for item in analysis.hotspots)


@pytest.mark.parametrize("payload", (b"", b"truncated"))
def test_memray_rejects_empty_and_truncated_captures(
    tmp_path: Path,
    payload: bytes,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    capture = tmp_path / "memory.bin"
    capture.write_bytes(payload)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
        )
    )

    with pytest.raises(DomainError) as error:
        MemrayExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_memray_cancellation_before_publication_preserves_the_corpus_head(
    tmp_path: Path,
) -> None:
    memray = _memray_module()
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        retained = bytearray(100_000)
    assert retained

    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=capture, kind=ArtifactKind.MEMORY_PROFILE)
    )
    head_before = workspace.corpus.read_head().commit_id
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 4:
            raise DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled")

    with pytest.raises(DomainError) as cancelled:
        MemrayExtractor(workspace).extract(imported.run.run_id, cancel_check=cancel)

    assert cancelled.value.code is ErrorCode.PROCESS_CANCELLED
    assert workspace.corpus.read_head().commit_id == head_before
