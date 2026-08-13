from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters import PerfettoExtractor
from flameox.analysis import RecipeService
from flameox.application import ImportArtifactRequest, ImportService
from flameox.domain import ArtifactKind
from flameox.storage import Workspace
from flameox.workers.perfetto_contract import (
    PerfettoExtractRequest,
    PerfettoExtractResult,
    PerfettoSliceRow,
)

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_perfetto_normalization_inherits_trace_visible_sdk_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text('{"traceEvents": []}')
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="torch.profiler",
        )
    )
    extractor = PerfettoExtractor(workspace)
    monkeypatch.setattr(extractor, "_trace_processor_path", lambda: trace)

    async def worker(_request: PerfettoExtractRequest) -> PerfettoExtractResult:
        def row(
            event_id: int,
            parent_id: int | None,
            name: str,
            category: str,
            start: int,
            duration: int,
        ) -> dict[str, object]:
            return {
                "id": event_id,
                "parent_id": parent_id,
                "name": name,
                "ts": start,
                "dur": duration,
                "track_id": 1,
                "category": category,
                "thread_name": "main",
                "process_name": "fixture",
                "filename": None,
                "line": None,
                "input_shapes": None,
                "allocation_bytes": None,
                "phase": None,
                "correlation_id": "41" if event_id > 1 else None,
                "device": "0" if event_id == 3 else None,
                "stream": "9" if event_id == 3 else None,
            }

        return PerfettoExtractResult(
            truncated=False,
            rows=tuple(
                PerfettoSliceRow.model_validate(item)
                for item in (
                    row(1, None, "flameox.phase:decode", "user_annotation", 0, 100),
                    row(2, 1, "cudaLaunchKernel", "cuda_runtime", 10, 2),
                    row(3, 1, "projection_kernel", "kernel", 20, 5),
                )
            ),
        )

    monkeypatch.setattr(extractor, "_run_worker", worker)

    extracted = await extractor.extract(imported.run.run_id)
    repeated = await extractor.extract(imported.run.run_id)
    result = RecipeService(workspace).accelerator_launches(
        imported.run.run_id,
        phase="decode",
    )

    assert repeated.corpus_commit_id == extracted.corpus_commit_id
    assert extracted.query_version == "flameox.perfetto.trace-events.v1"
    assert extracted.trace_processor_sha256.startswith("sha256:")
    assert result.regions[0].direct_launch_count == 1
    assert result.regions[0].kernel_count == 1
