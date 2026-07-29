from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import (
    ArtifactService,
    CreateInvestigationRequest,
    FindingService,
    ImportArtifactRequest,
    ImportService,
    InvestigationService,
    RecordFindingRequest,
    RunDiscoveryService,
    RunFilter,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    EvidenceLevel,
    FindingAssessment,
    Sensitivity,
)
from flameox.storage import Workspace


def _import(workspace: Workspace, name: str) -> None:
    path = workspace.project_root / name
    path.write_text(f'{{"name": "{name}"}}')
    ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=path,
            kind=ArtifactKind.COLLECTOR_METADATA,
            sensitivity=Sensitivity.INTERNAL,
        )
    )


def test_run_and_artifact_pagination_are_snapshot_bound_and_filtered(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    _import(workspace, "one.json")
    _import(workspace, "two.json")

    runs = RunDiscoveryService(workspace)
    first = runs.list(
        filter=RunFilter(execution_status=("not_applicable",)),
        limit=1,
    )
    assert first.returned == 1
    assert first.next_cursor is not None
    assert first.coverage.filters_applied == ("execution_status",)
    matching_identity = first.runs[0].environment_id
    second = runs.list(
        filter=RunFilter(execution_status=("not_applicable",)),
        limit=1,
        cursor=first.next_cursor,
    )
    assert second.returned == 1
    assert second.runs[0].run_id != first.runs[0].run_id
    assert second.next_cursor is None
    assert (
        runs.list(
            filter=RunFilter(environment_id=matching_identity),
            limit=10,
        ).total
        == 2
    )
    assert (
        runs.list(
            filter=RunFilter(environment_id="sha256:" + "0" * 64),
            limit=10,
        ).total
        == 0
    )

    artifacts = ArtifactService(workspace)
    artifact_first = artifacts.list(limit=1)
    assert artifact_first.next_cursor is not None
    artifact_second = artifacts.list(limit=1, cursor=artifact_first.next_cursor)
    assert artifact_second.artifacts[0].artifact_id != artifact_first.artifacts[0].artifact_id

    _import(workspace, "three.json")
    with pytest.raises(DomainError) as stale:
        runs.list(
            filter=RunFilter(execution_status=("not_applicable",)),
            limit=1,
            cursor=first.next_cursor,
        )
    assert stale.value.code is ErrorCode.STALE_CURSOR
    with pytest.raises(DomainError) as changed_filter:
        runs.list(
            filter=RunFilter(validation_status=("not_requested",)),
            limit=1,
            cursor=first.next_cursor,
        )
    assert changed_filter.value.code is ErrorCode.STALE_CURSOR


def test_investigation_and_finding_lists_page_without_duplicates(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    investigations = InvestigationService(workspace)
    for question in ("first?", "second?"):
        investigations.create(CreateInvestigationRequest(question=question))

    first = investigations.list(limit=1)
    assert first.next_cursor is not None
    second = investigations.list(limit=1, cursor=first.next_cursor)
    assert first.investigations[0].investigation_id != second.investigations[0].investigation_id

    findings = FindingService(workspace)
    for title in ("first", "second"):
        findings.record(
            RecordFindingRequest(
                kind="runtime",
                title=title,
                claim=f"{title} claim",
                evidence_level=EvidenceLevel.INFERRED,
                confidence="unknown",
                assessment=FindingAssessment.UNASSESSED,
            )
        )
    finding_first = findings.list(limit=1)
    assert finding_first.next_cursor is not None
    finding_second = findings.list(limit=1, cursor=finding_first.next_cursor)
    assert finding_first.findings[0].finding_id != finding_second.findings[0].finding_id
