from __future__ import annotations

from pathlib import Path

import pyperf
import pytest

from flamo.adapters import PyPerfExtractor
from flamo.application import (
    CreateInvestigationRequest,
    EvidenceQueryService,
    ImportArtifactRequest,
    ImportService,
    InvestigationService,
)
from flamo.domain import ArtifactKind, DomainError, ErrorCode
from flamo.storage import Workspace


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
