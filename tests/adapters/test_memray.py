from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
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


def _extractor(
    workspace: Workspace,
    version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> MemrayExtractor:
    extractor = MemrayExtractor(workspace)
    runtime = SimpleNamespace(
        python=Path(sys.executable),
        receipt=SimpleNamespace(
            environment_id="sha256:" + "e" * 64,
            distributions={"memray": version},
            limitations=(),
        ),
    )
    monkeypatch.setattr(
        extractor.provider_runtimes,
        "find_distribution",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(
        extractor.provider_runtimes,
        "verified_use",
        lambda _runtime: nullcontext(runtime),
    )
    return extractor


def test_memray_extractor_preserves_native_capture_and_names_memory_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
            producer="memray",
            producer_version=memray.__version__,
        )
    )
    progress: list[tuple[str, int, int | None]] = []
    result = _extractor(workspace, memray.__version__, monkeypatch).extract(
        imported.run.run_id,
        progress=lambda phase, completed, total: progress.append((phase, completed, total)),
    )

    assert result.peak_memory_bytes >= 100_000
    assert result.retained_end_bytes >= 100_000
    assert result.total_allocations >= 1
    assert result.frame_count >= 1
    assert result.producer_version == memray.__version__
    assert result.reader_version == memray.__version__
    assert result.reader_environment_id == "sha256:" + "e" * 64
    assert {phase for phase, _completed, _total in progress} == {
        "reading_profile",
        "publishing_evidence",
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


def test_memray_extraction_rejects_a_reader_that_disagrees_with_its_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memray = _memray_module()
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        bytearray(1_000)
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version=memray.__version__,
        )
    )
    original_head = workspace.corpus.read_head().commit_id

    with pytest.raises(DomainError) as raised:
        _extractor(workspace, "0.0", monkeypatch).extract(imported.run.run_id)

    assert raised.value.code is ErrorCode.ADAPTER_INCOMPATIBLE
    assert raised.value.details["reader_version"] == "0.0"
    assert workspace.corpus.read_head().commit_id == original_head


@pytest.mark.parametrize("payload", (b"", b"truncated"))
def test_memray_rejects_empty_and_truncated_captures(
    tmp_path: Path,
    payload: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    capture = tmp_path / "memory.bin"
    capture.write_bytes(payload)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version="1.20.0",
        )
    )

    with pytest.raises(DomainError) as error:
        _extractor(workspace, "1.20.0", monkeypatch).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_memray_cancellation_before_publication_preserves_the_corpus_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memray = _memray_module()
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        retained = bytearray(100_000)
    assert retained

    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version=memray.__version__,
        )
    )
    head_before = workspace.corpus.read_head().commit_id
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled")

    with pytest.raises(DomainError) as cancelled:
        _extractor(workspace, memray.__version__, monkeypatch).extract(
            imported.run.run_id,
            cancel_check=cancel,
        )

    assert cancelled.value.code is ErrorCode.PROCESS_CANCELLED
    assert workspace.corpus.read_head().commit_id == head_before
