from __future__ import annotations

from pathlib import Path

import pyperf
import pytest

from flameox.adapters import PyPerfExtractor
from flameox.application import (
    CreateInvestigationRequest,
    EvidenceQueryService,
    ImportArtifactRequest,
    ImportService,
    InvestigationService,
)
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.analysis import run_row


def test_measurement_query_uses_bounded_snapshot_cursors(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    suite_path = tmp_path / "suite.json"
    run = pyperf.Run(
        [0.01, 0.02, 0.03],
        metadata={"name": "scan", "unit": "second", "loops": 1},
        collect_metadata=False,
    )
    pyperf.BenchmarkSuite([pyperf.Benchmark([run])]).dump(
        str(suite_path),
        replace=True,
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=suite_path,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
        )
    )
    PyPerfExtractor(workspace).extract(imported.run.run_id)
    service = EvidenceQueryService(workspace)

    first = service.measurements(
        run_id=imported.run.run_id,
        name_prefix="pyperf.",
        limit=2,
    )
    assert first.returned == 2
    assert first.total == 3
    assert first.schema_version == 2
    assert all(
        item.value is not None and item.value.kind == "integer" for item in first.measurements
    )
    assert first.next_cursor is not None
    second = service.measurements(
        run_id=imported.run.run_id,
        name_prefix="pyperf.",
        limit=2,
        cursor=first.next_cursor,
    )
    assert second.returned == 1
    assert {item.measurement_id for item in first.measurements}.isdisjoint(
        {item.measurement_id for item in second.measurements}
    )

    InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Advance corpus HEAD")
    )
    with pytest.raises(DomainError) as stale:
        service.measurements(cursor=first.next_cursor)
    assert stale.value.code is ErrorCode.STALE_CURSOR


def test_measurement_query_reports_filtered_warmups_as_empty(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("warmup-only")],
            "measurements": [
                {
                    "measurement_id": "warmup-measurement",
                    "run_id": "warmup-only",
                    "artifact_id": None,
                    "name": "benchmark.duration",
                    "value_int": None,
                    "value_float": 1.0,
                    "unit": "seconds",
                    "aggregation": "single",
                    "scope": "process",
                    "trial_id": None,
                    "worker_id": None,
                    "worker_run_index": 0,
                    "value_index": 0,
                    "loop_count": None,
                    "is_warmup": True,
                    "block_id": None,
                    "variant_id": None,
                    "order_in_block": None,
                    "phase": None,
                    "dimensions": {},
                    "evidence_level": "observed",
                }
            ],
        },
        publisher="warmup-fixture",
        publisher_version="1",
    )

    result = EvidenceQueryService(workspace).measurements(run_id="warmup-only")

    assert result.measurements == ()
    assert result.evidence.status == "empty"
    assert result.evidence.reason == "no_matching_measurements"
