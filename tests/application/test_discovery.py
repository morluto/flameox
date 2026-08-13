from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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
    ExecutionStatus,
    FindingAssessment,
    FindingConfidence,
    Investigation,
    Sensitivity,
    ValidationStatus,
    new_id,
)
from flameox.storage import Workspace

pytestmark = pytest.mark.integration


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


def test_run_filter_uses_authoritative_lifecycle_vocabularies() -> None:
    parsed = RunFilter.model_validate(
        {
            "execution_status": ["planned"],
            "validation_status": ["running", "error", "cancelled"],
        }
    )

    assert parsed.execution_status == (ExecutionStatus.PLANNED,)
    assert parsed.validation_status == (
        ValidationStatus.RUNNING,
        ValidationStatus.ERROR,
        ValidationStatus.CANCELLED,
    )
    with pytest.raises(ValidationError):
        RunFilter.model_validate({"execution_status": ["pending"]})


def test_run_and_artifact_pagination_are_snapshot_bound_and_filtered(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    _import(workspace, "one.json")
    _import(workspace, "two.json")

    runs = RunDiscoveryService(workspace)
    first = runs.list(
        filter=RunFilter(execution_status=(ExecutionStatus.NOT_APPLICABLE,)),
        limit=1,
    )
    assert first.returned == 1
    assert first.truncated is True
    assert first.next_cursor is not None
    assert first.model_dump(mode="json")["returned"] == 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(first).model_validate({**first.model_dump(mode="python"), "returned": 0})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(first).model_validate({**first.model_dump(mode="python"), "truncated": False})
    assert first.coverage.filters_applied == ("execution_status",)
    coverage_payload = first.coverage.model_dump(mode="python")
    assert coverage_payload["population_complete"] is True
    assert type(first.coverage).model_validate(coverage_payload) == first.coverage
    with pytest.raises(ValidationError, match="population completeness"):
        type(first.coverage).model_validate(
            {**coverage_payload, "unavailable_facets": ["artifact_kinds"]}
        )
    matching_identity = first.runs[0].environment_id
    assert first.runs[0].artifact_kinds == ("collector_metadata",)
    second = runs.list(
        filter=RunFilter(execution_status=(ExecutionStatus.NOT_APPLICABLE,)),
        limit=1,
        cursor=first.next_cursor,
    )
    assert second.returned == 1
    assert second.truncated is False
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
    assert (
        runs.list(
            filter=RunFilter(validation_status=(ValidationStatus.UNSUPPORTED,)),
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
            filter=RunFilter(execution_status=(ExecutionStatus.NOT_APPLICABLE,)),
            limit=1,
            cursor=first.next_cursor,
        )
    assert stale.value.code is ErrorCode.STALE_CURSOR
    with pytest.raises(DomainError) as changed_filter:
        runs.list(
            filter=RunFilter(validation_status=(ValidationStatus.NOT_REQUESTED,)),
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
                confidence=FindingConfidence.UNKNOWN,
                assessment=FindingAssessment.UNASSESSED,
            )
        )
    finding_first = findings.list(limit=1)
    assert finding_first.next_cursor is not None
    finding_second = findings.list(limit=1, cursor=finding_first.next_cursor)
    assert finding_first.findings[0].finding_id != finding_second.findings[0].finding_id


def test_investigation_cursor_ignores_unpublished_control_record(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    investigations = InvestigationService(workspace)
    for question in ("first?", "second?"):
        investigations.create(CreateInvestigationRequest(question=question))

    first = investigations.list(limit=1)
    assert first.next_cursor is not None
    investigations.investigations.create(
        Investigation(
            investigation_id=new_id(),
            question="not published",
            project_root=str(workspace.project_root),
        )
    )

    second = investigations.list(limit=1, cursor=first.next_cursor)

    assert second.corpus_commit_id == first.corpus_commit_id
    assert second.investigations[0].investigation_id != first.investigations[0].investigation_id
    assert second.investigations[0].question == "first?"


def test_finding_list_ignores_revision_not_published_to_its_snapshot(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    findings = FindingService(workspace)
    published = findings.record(
        RecordFindingRequest(
            kind="runtime",
            title="published",
            claim="published claim",
            evidence_level=EvidenceLevel.INFERRED,
            confidence=FindingConfidence.UNKNOWN,
            assessment=FindingAssessment.UNASSESSED,
        )
    )
    findings.findings.append(
        published.finding.validated_copy(update={"revision": 2, "title": "not published"}),
        expected_revision=1,
    )

    result = findings.list(limit=10)

    assert result.corpus_commit_id == published.corpus_commit_id
    assert result.findings[0].revision == 1
    assert result.findings[0].title == "published"
