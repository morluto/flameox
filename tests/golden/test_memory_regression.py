from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from flameox.adapters import MemrayExtractor
from flameox.adapters.memray import memray_extraction_limits
from flameox.application import (
    AnalysisMaterializationService,
    ComparisonService,
    CreateInvestigationRequest,
    EvidenceInput,
    FindingService,
    FreezeRunIdsRequest,
    ImportArtifactRequest,
    ImportService,
    InvestigationService,
    MeasurementCompareRunSetsRequest,
    MemoryAnalysisRequest,
    RecordFindingRequest,
    RecordHypothesisRequest,
    RunSetService,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    EvidenceLevel,
    EvidenceReferenceType,
    EvidenceRelation,
    FindingAssessment,
    FindingConfidence,
    MetricPolarity,
)
from flameox.storage import Workspace

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optional,
    pytest.mark.requires_memray,
    pytest.mark.serial,
]


def _capture(path: Path, allocation_count: int) -> str:
    memray = pytest.importorskip("memray", reason="optional provider unavailable: install memray")
    with memray.Tracker(str(path)):
        retained = [bytearray(4_096) for _ in range(allocation_count)]
        assert len(retained) == allocation_count
    return str(memray.__version__)


@pytest.mark.anyio
async def test_retained_memory_regression_has_native_and_normalized_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        producer_version = _capture(capture_path, allocation_count)
        imported = ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=capture_path,
                kind=ArtifactKind.MEMORY_PROFILE,
                producer="memray",
                producer_version=producer_version,
            )
        )
        runs[name] = imported.run
        extractor = MemrayExtractor(workspace)
        runtime = SimpleNamespace(
            python=Path(sys.executable),
            receipt=SimpleNamespace(
                environment_id="sha256:" + "e" * 64,
                distributions={"memray": producer_version},
                limitations=(),
            ),
        )
        monkeypatch.setattr(
            extractor.provider_runtimes,
            "find_distribution",
            lambda selected=runtime, **_kwargs: selected,
        )
        monkeypatch.setattr(
            extractor.provider_runtimes,
            "verified_use",
            lambda _runtime, selected=runtime: nullcontext(selected),
        )
        extractions[name] = await extractor.extract(
            imported.run.run_id, limits=memray_extraction_limits(workspace)
        )

    cohorts = RunSetService(workspace)
    baseline = cohorts.freeze(FreezeRunIdsRequest(run_ids=(runs["baseline"].run_id,)))
    candidate = cohorts.freeze(FreezeRunIdsRequest(run_ids=(runs["candidate"].run_id,)))
    comparison = ComparisonService(workspace).record(
        MeasurementCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="memory.retained_end",
            unit="bytes",
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )
    memory_analysis = AnalysisMaterializationService(workspace).record(
        MemoryAnalysisRequest(
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
            confidence=FindingConfidence.HIGH,
            assessment=FindingAssessment.SUPPORTED,
            evidence=(
                EvidenceInput(
                    ref_type=EvidenceReferenceType.ANALYSIS,
                    ref_id=memory_analysis.analysis.analysis_id,
                    relation=EvidenceRelation.SUPPORTS,
                ),
            ),
        )
    )

    assert extractions["candidate"].retained_end_bytes > extractions["baseline"].retained_end_bytes
    assert comparison.analysis is not None
    assert hypothesis.hypothesis_id
    assert finding.finding.assessment is FindingAssessment.SUPPORTED
