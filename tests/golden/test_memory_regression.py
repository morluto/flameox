from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters import MemrayExtractor
from flameox.application import (
    AnalysisMaterializationService,
    CompareRunSetsRequest,
    ComparisonService,
    CreateInvestigationRequest,
    EvidenceInput,
    FindingService,
    FreezeRunSetRequest,
    ImportArtifactRequest,
    ImportService,
    InvestigationService,
    MaterializeAnalysisRequest,
    RecordFindingRequest,
    RecordHypothesisRequest,
    RunSetService,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, EvidenceLevel, FindingAssessment
from flameox.storage import Workspace

memray = pytest.importorskip("memray", reason="optional provider unavailable: install memray")


def _capture(path: Path, allocation_count: int) -> None:
    with memray.Tracker(str(path)):
        retained = [bytearray(4_096) for _ in range(allocation_count)]
        assert len(retained) == allocation_count


def test_retained_memory_regression_has_native_and_normalized_evidence(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    investigations = InvestigationService(workspace)
    investigation = investigations.create(
        CreateInvestigationRequest(
            question="Does the candidate retain more allocations at shutdown?"
        )
    )
    hypothesis = investigations.record_hypothesis(
        RecordHypothesisRequest(
            investigation_id=investigation.investigation_id,
            claim="The candidate retains a larger live allocation set.",
            prediction="Memray reports a larger retained-end byte count.",
            discriminating_condition="The candidate retained-end count exceeds baseline.",
        )
    )

    runs = {}
    extractions = {}
    for name, allocation_count in (("baseline", 8), ("candidate", 64)):
        capture_path = tmp_path / f"{name}.bin"
        _capture(capture_path, allocation_count)
        imported = ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=capture_path,
                kind=ArtifactKind.MEMORY_PROFILE,
                producer="memray",
            )
        )
        runs[name] = imported.run
        extractions[name] = MemrayExtractor(workspace).extract(imported.run.run_id)

    cohorts = RunSetService(workspace)
    baseline = cohorts.freeze(FreezeRunSetRequest(run_ids=(runs["baseline"].run_id,)))
    candidate = cohorts.freeze(FreezeRunSetRequest(run_ids=(runs["candidate"].run_id,)))
    comparison = ComparisonService(workspace).record(
        CompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="memory.retained_end",
            unit="bytes",
            polarity="lower_is_better",
            practical_threshold=0.05,
        )
    )
    memory_analysis = AnalysisMaterializationService(workspace).record(
        MaterializeAnalysisRequest(
            recipe="memory",
            input_id=runs["candidate"].run_id,
        )
    )
    finding = FindingService(workspace).record(
        RecordFindingRequest(
            kind="memory",
            title="Candidate retains allocations at capture end",
            claim="The candidate capture has non-zero retained-end allocations.",
            evidence_level=EvidenceLevel.DERIVED,
            confidence="high",
            assessment=FindingAssessment.SUPPORTED,
            evidence=(
                EvidenceInput(
                    ref_type="analysis",
                    ref_id=memory_analysis.analysis.analysis_id,
                    relation="supports",
                ),
            ),
        )
    )

    assert extractions["candidate"].retained_end_bytes > extractions["baseline"].retained_end_bytes
    assert comparison.analysis is not None
    assert hypothesis.hypothesis_id
    assert finding.finding.assessment is FindingAssessment.SUPPORTED
