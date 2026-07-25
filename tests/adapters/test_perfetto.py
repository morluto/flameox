from __future__ import annotations

import json
from pathlib import Path

import pytest

from flamo.adapters import PerfettoExtractor
from flamo.analysis import RecipeService
from flamo.application import DrilldownService, ImportArtifactRequest, ImportService
from flamo.catalog import Catalog
from flamo.domain import ArtifactKind, DomainError, ErrorCode
from flamo.storage import Workspace


def local_trace_processor() -> Path:
    candidates = sorted(
        (Path.home() / ".local" / "share" / "perfetto" / "prebuilts").glob(
            "trace_processor_shell-*"
        )
    )
    if not candidates:
        pytest.skip("A local Trace Processor binary is not installed.")
    return candidates[-1]


@pytest.mark.anyio
async def test_perfetto_extractor_uses_local_binary_and_curated_query(
    tmp_path: Path,
) -> None:
    binary = local_trace_processor()
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "reverse_scan",
                        "ph": "X",
                        "ts": 0,
                        "dur": 1000,
                        "pid": 1,
                        "tid": 1,
                        "args": {"filename": "scan.py", "line": 10},
                    },
                    {
                        "name": "accumulate",
                        "ph": "X",
                        "ts": 100,
                        "dur": 500,
                        "pid": 1,
                        "tid": 1,
                        "args": {"filename": "scan.py", "line": 11},
                    },
                    {
                        "name": "finalize",
                        "ph": "X",
                        "ts": 650,
                        "dur": 200,
                        "pid": 1,
                        "tid": 1,
                        "args": {"filename": "scan.py", "line": 12},
                    },
                ]
            }
        )
    )
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "analysis": workspace.config.analysis.model_copy(
                update={"trace_processor_path": str(binary)}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.EXECUTION_TRACE,
        )
    )

    result = await PerfettoExtractor(workspace).extract(imported.run.run_id)
    hotspots = RecipeService(workspace).hotspots(imported.run.run_id)
    hotspot_by_name = {item.function: item for item in hotspots.hotspots}
    drilldown = DrilldownService(workspace)
    callers = drilldown.callers(
        imported.run.run_id,
        hotspot_by_name["accumulate"].frame_id,
    )
    examples = drilldown.examples(
        imported.run.run_id,
        hotspot_by_name["reverse_scan"].frame_id,
    )
    window = await PerfettoExtractor(workspace).trace_window(
        imported.run.artifacts[0].artifact_id,
        start_ns=0,
        end_ns=2_000_000,
        limit=2,
    )
    first_callees = drilldown.callees(
        imported.run.run_id,
        hotspot_by_name["reverse_scan"].frame_id,
        limit=1,
    )
    assert first_callees.next_cursor is not None
    second_callees = drilldown.callees(
        imported.run.run_id,
        hotspot_by_name["reverse_scan"].frame_id,
        limit=1,
        cursor=first_callees.next_cursor,
    )

    assert result.slice_count == 3
    assert result.frame_count == 3
    assert result.call_edge_count == 2
    assert result.representative_stack_count == 2
    assert hotspot_by_name["reverse_scan"].inclusive_value == 1_000_000
    assert callers.frames[0].function == "reverse_scan"
    assert [frame.function for frame in examples.examples[0].frames] == [
        "reverse_scan",
        "accumulate",
    ]
    assert second_callees.returned == 1
    assert second_callees.next_cursor is None
    assert window.total == 3
    assert window.returned == 2
    assert window.truncated
    assert window.next_cursor is not None
    second_page = await PerfettoExtractor(workspace).trace_window(
        imported.run.artifacts[0].artifact_id,
        start_ns=0,
        end_ns=2_000_000,
        limit=2,
        cursor=window.next_cursor,
    )
    assert second_page.returned == 1
    assert second_page.next_cursor is None
    with pytest.raises(DomainError) as not_torch:
        RecipeService(workspace).pytorch(imported.run.run_id)
    assert not_torch.value.code is ErrorCode.COMPARISON_INVALID


@pytest.mark.anyio
async def test_perfetto_worker_maps_malformed_trace_to_domain_error(
    tmp_path: Path,
) -> None:
    binary = local_trace_processor()
    trace = tmp_path / "malformed.pftrace"
    trace.write_bytes(b"not a trace")
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "analysis": workspace.config.analysis.model_copy(
                update={"trace_processor_path": str(binary)}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=trace, kind=ArtifactKind.EXECUTION_TRACE)
    )

    with pytest.raises(DomainError) as failure:
        await PerfettoExtractor(workspace).extract(imported.run.run_id)

    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
