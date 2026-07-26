from __future__ import annotations

import json
from pathlib import Path

import pytest

from flamo.adapters import PerfettoExtractor
from flamo.analysis import RecipeService
from flamo.application import (
    AnalysisMaterializationService,
    ArtifactService,
    DrilldownService,
    EvidenceLookupService,
    ImportArtifactRequest,
    ImportService,
    MaterializeAnalysisRequest,
    NativeViewerService,
)
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
    materialized = AnalysisMaterializationService(workspace).record(
        MaterializeAnalysisRequest(
            recipe="hotspots",
            input_id=imported.run.run_id,
            limit=10,
        )
    )
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
    provenance = EvidenceLookupService(workspace).get(
        "analysis",
        materialized.analysis.analysis_id,
    )
    run_evidence = EvidenceLookupService(workspace).get(
        "run",
        imported.run.run_id,
    )
    artifact = ArtifactService(workspace).get(imported.artifact_id)
    viewer = NativeViewerService(workspace).plan(imported.artifact_id)

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
    assert provenance.data["corpus_commit_id"] == hotspots.corpus_commit_id
    assert run_evidence.data["run_id"] == imported.run.run_id
    assert artifact.registrations[0].run_id == imported.run.run_id
    assert viewer.viewer == "trace_processor_shell"
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


@pytest.mark.anyio
async def test_perfetto_preserves_recursive_multi_process_and_thread_dimensions(
    tmp_path: Path,
) -> None:
    binary = local_trace_processor()
    trace = tmp_path / "topology.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "recursive",
                        "ph": "X",
                        "ts": 0,
                        "dur": 100,
                        "pid": 1,
                        "tid": 1,
                        "args": {"filename": "scan.py", "line": 1},
                    },
                    {
                        "name": "recursive",
                        "ph": "X",
                        "ts": 10,
                        "dur": 50,
                        "pid": 1,
                        "tid": 1,
                        "args": {"filename": "scan.py", "line": 1},
                    },
                    {
                        "name": "thread_worker",
                        "ph": "X",
                        "ts": 0,
                        "dur": 80,
                        "pid": 1,
                        "tid": 2,
                        "args": {"filename": "scan.py", "line": 2},
                    },
                    {
                        "name": "process_worker",
                        "ph": "X",
                        "ts": 0,
                        "dur": 70,
                        "pid": 2,
                        "tid": 1,
                        "args": {"filename": "scan.py", "line": 3},
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
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.EXECUTION_TRACE,
        )
    )

    result = await PerfettoExtractor(workspace).extract(imported.run.run_id)

    with Catalog(workspace).open_snapshot() as snapshot:
        dimensions = snapshot.execute(
            "SELECT count(DISTINCT thread_name), count(DISTINCT process_name) "
            "FROM frame_measurements WHERE run_id = ?",
            (imported.run.run_id,),
        ).fetchone()
        recursive_edge = snapshot.execute(
            "SELECT count(*) FROM call_edges ce JOIN frames parent "
            "ON parent.frame_id = ce.parent_frame_id JOIN frames child "
            "ON child.frame_id = ce.child_frame_id "
            "WHERE parent.function = 'recursive' AND child.function = 'recursive'"
        ).fetchone()
    assert result.slice_count == 4
    assert dimensions is not None
    assert dimensions[0] >= 2
    assert dimensions[1] >= 2
    assert recursive_edge == (1,)


@pytest.mark.anyio
async def test_torch_profiler_trace_preserves_shapes_memory_phases_and_sync(
    tmp_path: Path,
) -> None:
    binary = local_trace_processor()
    trace = tmp_path / "torch-trace.json"
    events = [
        {
            "name": "aten::add",
            "cat": "cpu_op",
            "ph": "X",
            "ts": index * 10,
            "dur": 5,
            "pid": 1,
            "tid": 1,
            "args": {
                "Input Shapes": "[[4, 8]]",
                "Bytes": 4096,
                "phase": "warmup",
            },
        }
        for index in range(4)
    ]
    events.extend(
        [
            {
                "name": "cudaDeviceSynchronize",
                "cat": "cuda_runtime",
                "ph": "X",
                "ts": 50,
                "dur": 7,
                "pid": 1,
                "tid": 1,
                "args": {},
            },
            {
                "name": "Torch-Compiled Region",
                "cat": "cpu_op",
                "ph": "X",
                "ts": 60,
                "dur": 11,
                "pid": 1,
                "tid": 1,
                "args": {"phase": "compile"},
            },
        ]
    )
    trace.write_text(json.dumps({"traceEvents": events}))
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "analysis": workspace.config.analysis.model_copy(
                update={"trace_processor_path": str(binary)}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="torch.profiler",
        )
    )

    await PerfettoExtractor(workspace).extract(imported.run.run_id)
    result = RecipeService(workspace).pytorch(imported.run.run_id)

    by_name = {operator.operator: operator for operator in result.operators}
    assert by_name["aten::add"].input_shapes == ("[[4, 8]]",)
    assert by_name["aten::add"].allocation_bytes == 4 * 4096
    assert by_name["aten::add"].warmup is True
    assert result.coverage["input_shapes"]
    assert result.coverage["memory_allocations"]
    assert result.synchronization_time_ns == 7_000
    assert result.compilation_time_ns == 11_000
    assert result.repeated_small_operations[0].operator == "aten::add"


@pytest.mark.anyio
async def test_pytorch_analysis_keeps_same_named_frames_and_mixed_phases_separate(
    tmp_path: Path,
) -> None:
    binary = local_trace_processor()
    trace = tmp_path / "torch-frames.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "aten::add",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 0,
                        "dur": 5,
                        "pid": 1,
                        "tid": 1,
                        "args": {
                            "filename": "model.py",
                            "line": 10,
                            "Input Shapes": "[[2, 2]]",
                            "Bytes": 100,
                            "phase": "warmup",
                        },
                    },
                    {
                        "name": "aten::add",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 10,
                        "dur": 7,
                        "pid": 1,
                        "tid": 1,
                        "args": {
                            "filename": "model.py",
                            "line": 10,
                            "Input Shapes": "[[2, 2]]",
                            "Bytes": 100,
                            "phase": "steady_state",
                        },
                    },
                    {
                        "name": "aten::add",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 20,
                        "dur": 11,
                        "pid": 1,
                        "tid": 1,
                        "args": {
                            "filename": "model.py",
                            "line": 20,
                            "Input Shapes": "[[8, 8]]",
                            "Bytes": 700,
                            "phase": "steady_state",
                        },
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
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="torch.profiler",
        )
    )

    await PerfettoExtractor(workspace).extract(imported.run.run_id)
    result = RecipeService(workspace).pytorch(imported.run.run_id)

    by_shapes = {operator.input_shapes: operator for operator in result.operators}
    mixed = by_shapes[("[[2, 2]]",)]
    steady = by_shapes[("[[8, 8]]",)]
    assert mixed.allocation_bytes == 200
    assert mixed.warmup is None
    assert steady.allocation_bytes == 700
    assert steady.warmup is False
    assert result.warmup_time_ns == 5_000
