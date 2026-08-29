from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters import PerfettoExtractor
from flameox.analysis import RecipeService
from flameox.application import (
    AnalysisMaterializationService,
    ArtifactService,
    DrilldownService,
    EvidenceLookupService,
    HotspotAnalysisRequest,
    ImportArtifactRequest,
    ImportProfile,
    ImportService,
    NativeViewerService,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode, EvidenceReferenceType
from flameox.storage import Workspace
from tests.support.providers import require_trace_processor

pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.requires_perfetto]
_PYSPY_FIXTURE = Path(__file__).parents[1] / "fixtures" / "pyspy" / "chrometrace-0.4.2.json"


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_perfetto
async def test_perfetto_extractor_uses_local_binary_and_curated_query(
    tmp_path: Path,
) -> None:
    binary = require_trace_processor()
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "cat": "py-spy",
                    "ph": phase,
                    "ts": timestamp,
                    "pid": 1,
                    "tid": 1,
                    "args": {"filename": "scan.py", "line": line},
                }
                for name, phase, timestamp, line in (
                    ("reverse_scan", "B", 0, 10),
                    ("accumulate", "B", 100, 11),
                    ("accumulate", "E", 600, 11),
                    ("finalize", "B", 650, 12),
                    ("finalize", "E", 850, 12),
                    ("reverse_scan", "E", 1000, 10),
                )
            ]
        )
    )
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "analysis": workspace.config.analysis.validated_copy(
                update={"trace_processor_path": str(binary)}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.SAMPLE_PROFILE,
            producer="py-spy",
            producer_version="0.4.2",
            profile=ImportProfile.PYSPY_CHROMETRACE,
        )
    )

    result = await PerfettoExtractor(workspace).extract(imported.run.run_id)
    hotspots = RecipeService(workspace).hotspots(imported.run.run_id)
    materialized = AnalysisMaterializationService(workspace).record(
        HotspotAnalysisRequest(
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
        EvidenceReferenceType.ANALYSIS,
        materialized.analysis.analysis_id,
    )
    run_evidence = EvidenceLookupService(workspace).get(
        EvidenceReferenceType.RUN,
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
    with Catalog(workspace).open_snapshot(result.corpus_commit_id) as snapshot:
        languages = snapshot.execute(
            "SELECT DISTINCT f.language FROM frames f "
            "JOIN frame_measurements fm USING (frame_id) WHERE fm.run_id = ?",
            (imported.run.run_id,),
        ).fetchall()
    assert languages == [("Python",)]
    with pytest.raises(DomainError) as not_torch:
        RecipeService(workspace).pytorch(imported.run.run_id)
    assert not_torch.value.code is ErrorCode.COMPARISON_INVALID


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_perfetto
async def test_real_pyspy_import_profile_controls_language_attribution(tmp_path: Path) -> None:
    binary = require_trace_processor()
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "analysis": workspace.config.analysis.validated_copy(
                update={"trace_processor_path": str(binary)}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    Catalog(workspace).rebuild()
    service = ImportService(workspace)
    qualified = service.import_artifact(
        ImportArtifactRequest(
            path=_PYSPY_FIXTURE,
            kind=ArtifactKind.SAMPLE_PROFILE,
            producer="py-spy",
            producer_version="0.4.2",
            profile=ImportProfile.PYSPY_CHROMETRACE,
            allow_external_path=True,
        )
    )
    generic = service.import_artifact(
        ImportArtifactRequest(
            path=_PYSPY_FIXTURE,
            kind=ArtifactKind.SAMPLE_PROFILE,
            producer="py-spy",
            producer_version="0.4.2",
            allow_external_path=True,
        )
    )

    await PerfettoExtractor(workspace).extract(qualified.run.run_id)
    generic_result = await PerfettoExtractor(workspace).extract(generic.run.run_id)

    with Catalog(workspace).open_snapshot(generic_result.corpus_commit_id) as snapshot:
        languages = snapshot.execute(
            "SELECT fm.run_id, f.language FROM frames f "
            "JOIN frame_measurements fm USING (frame_id) "
            "WHERE fm.run_id IN (?, ?) ORDER BY fm.run_id",
            (qualified.run.run_id, generic.run.run_id),
        ).fetchall()
    language_by_run = {run_id: language for run_id, language in languages}
    assert language_by_run[qualified.run.run_id] == "Python"
    assert language_by_run[generic.run.run_id] is None
    assert any("qualify_artifact_import" in item for item in generic_result.limitations)


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_perfetto
async def test_perfetto_worker_maps_malformed_trace_to_domain_error(
    tmp_path: Path,
) -> None:
    binary = require_trace_processor()
    trace = tmp_path / "malformed.pftrace"
    trace.write_bytes(b"not a trace")
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "analysis": workspace.config.analysis.validated_copy(
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
@pytest.mark.optional
@pytest.mark.requires_perfetto
async def test_perfetto_preserves_recursive_multi_process_and_thread_dimensions(
    tmp_path: Path,
) -> None:
    binary = require_trace_processor()
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
    config = workspace.config.validated_copy(
        update={
            "analysis": workspace.config.analysis.validated_copy(
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
