from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from flameox.application.records import (
    EvidenceInput,
    FindingService,
    RecordFindingRequest,
)
from flameox.application.run_rows import run_row
from flameox.application.static_analysis import (
    ImportStaticAnalysisRequest,
    StaticAnalysisImportResult,
    StaticAnalysisService,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    CaptureStatus,
    DomainError,
    ErrorCode,
    EvidenceLevel,
    EvidenceReferenceType,
    EvidenceRelation,
    ExecutionStatus,
    FindingAssessment,
    FindingConfidence,
    RunSemantics,
    ValidationStatus,
)
from flameox.domain.models import ExecutionRunManifest
from flameox.evidence import GenerationPublisher
from flameox.storage import ArtifactStore, RunStore, Workspace

pytestmark = pytest.mark.integration


def _result(
    uri: str,
    *,
    message: str = "Avoid this call",
    base: str | None = None,
) -> dict[str, object]:
    artifact_location: dict[str, object] = {"uri": uri}
    if base is not None:
        artifact_location["uriBaseId"] = base
    return {
        "ruleId": "example.rule",
        "level": "warning",
        "message": {"text": message},
        "partialFingerprints": {"primaryLocationLineHash": f"fingerprint-{message}"},
        "properties": {"confidence": 0.75, "provider_extension": {"opaque": True}},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": artifact_location,
                    "region": {"startLine": 4, "startColumn": 2, "endLine": 4, "endColumn": 8},
                }
            }
        ],
    }


def _write_sarif(
    path: Path,
    results: list[dict[str, object]],
    *,
    version: str = "2.1.0",
) -> bytes:
    payload = {
        "version": version,
        "runs": [
            {
                "tool": {"driver": {"name": "example-analyzer", "version": "4.2"}},
                "invocations": [{"exitCode": 0}],
                "results": results,
            }
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(encoded)
    return encoded


def _write_many_run_sarif(path: Path, count: int = 24) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "name": f"analyzer-{index}-{'x' * 5_000}",
                                "version": "v" * 5_000,
                            }
                        },
                        "results": [_result("app.py", message=f"run-{index}")],
                    }
                    for index in range(count)
                ],
            }
        )
    )


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=project,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _import(
    workspace: Workspace,
    report: Path,
    source_root: Path,
    *,
    include_paths: tuple[str, ...] = (),
    exclude_paths: tuple[str, ...] = (),
) -> StaticAnalysisImportResult:
    return StaticAnalysisService(workspace).import_sarif(
        ImportStaticAnalysisRequest(
            path=report,
            source_root=source_root,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
        )
    )


def test_static_analysis_import_preserves_sarif_and_projects_candidates(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "app.py").write_text("print('ok')\n")
    report = tmp_path / "analysis.sarif"
    source_result = _result("app.py", base="SRCROOT")
    locations = source_result["locations"]
    assert isinstance(locations, list)
    locations.append({"physicalLocation": {"artifactLocation": {"uri": "not-the-first.py"}}})
    source = _write_sarif(report, [source_result])
    workspace = Workspace.initialize(tmp_path)

    result = _import(workspace, report, source_root)

    run = RunStore(workspace).read(result.run_id)
    artifact = ArtifactStore(workspace).get(result.artifact_id)
    assert run.artifacts[0].kind is ArtifactKind.ANALYSIS_RESULT
    assert artifact.payload_path.read_bytes() == source
    assert result.coverage.model_dump() == {
        "result_count": 1,
        "normalized_count": 1,
        "excluded_count": 0,
        "invalid_count": 0,
        "omitted_count": 0,
    }
    assert result.first_page.candidates[0].model_dump() == {
        "candidate_id": result.first_page.candidates[0].candidate_id,
        "run_id": result.run_id,
        "artifact_id": result.artifact_id,
        "rule_id": "example.rule",
        "level": "warning",
        "message": "Avoid this call",
        "relative_path": "app.py",
        "start_line": 4,
        "start_column": 2,
        "end_line": 4,
        "end_column": 8,
        "provider_fingerprint": "fingerprint-Avoid this call",
        "provider_confidence": 0.75,
        "related_finding_ids": (),
        "related_findings_truncated": False,
    }
    assert run.semantics.configuration["source_root"] == "source"
    assert run.semantics.configuration["sarif_version"] == "2.1.0"
    assert run.semantics.configuration["analyzer"] == "example-analyzer"
    assert run.semantics.unavailable_fields == ("source_state",)
    assert "provider_extension" not in result.first_page.candidates[0].model_dump_json()
    with Catalog(workspace).open_snapshot(result.corpus_commit_id) as snapshot:
        assert snapshot.execute("SELECT count(*) FROM findings").fetchone() == (0,)


def test_static_analysis_bounds_run_semantics_without_dropping_source_findings(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "app.py").write_text("pass\n")
    report = tmp_path / "analysis.sarif"
    _write_many_run_sarif(report)
    workspace = Workspace.initialize(tmp_path)

    result = _import(workspace, report, source_root)

    run = RunStore(workspace).read(result.run_id)
    analyzers = cast(list[dict[str, object]], run.semantics.configuration["analyzers"])
    assert len(analyzers) == 16
    assert all(len(cast(str, analyzer["name"])) <= 256 for analyzer in analyzers)
    assert all(
        analyzer.get("version") is None
        or len(cast(str, analyzer["version"])) <= 256
        for analyzer in analyzers
    )
    assert result.coverage.normalized_count == 24
    assert any("provenance record(s) were omitted" in item for item in result.limitations)
    assert any("fields truncated" in item for item in result.limitations)
    assert len(run.semantics.model_dump_json()) < 12_000


def test_static_analysis_selects_one_deterministic_fingerprint_from_large_fanout(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "app.py").write_text("pass\n")
    report = tmp_path / "analysis.sarif"
    candidate = _result("app.py")
    candidate["partialFingerprints"] = {
        f"fingerprint-{index:05d}": f"value-{index:05d}"
        for index in reversed(range(20_000))
    }
    _write_sarif(report, [candidate])
    workspace = Workspace.initialize(tmp_path)

    result = _import(workspace, report, source_root)

    assert result.first_page.candidates[0].provider_fingerprint == "value-00000"


def test_static_analysis_coverage_counts_results_dropped_before_tool_identity(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "app.py").write_text("pass\n")
    report = tmp_path / "analysis.sarif"
    report.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "results": [
                            _result("app.py", message=f"candidate-{index}")
                            for index in range(1_002)
                        ],
                        "tool": {"driver": {"name": "late-analyzer", "version": "1"}},
                    }
                ],
            }
        )
    )
    workspace = Workspace.initialize(tmp_path)

    result = _import(workspace, report, source_root)
    coverage = result.coverage

    assert coverage.normalized_count == 1_000
    assert coverage.omitted_count == 2
    assert coverage.result_count == (
        coverage.normalized_count
        + coverage.excluded_count
        + coverage.invalid_count
        + coverage.omitted_count
    )
    assert any("streaming buffer" in limitation for limitation in result.limitations)


@pytest.mark.parametrize("selector", ["include_paths", "exclude_paths"])
def test_static_analysis_rejects_oversized_scope_selectors(
    tmp_path: Path,
    selector: str,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    report = tmp_path / "analysis.sarif"
    _write_sarif(report, [])
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError, match="at most 256 characters"):
        _import(workspace, report, source_root, **{selector: ("x" * 257,)})


def test_static_candidate_reaches_runtime_evidence_through_a_finding(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "app.py").write_text("pass\n")
    report = tmp_path / "analysis.sarif"
    _write_sarif(report, [_result("app.py")])
    workspace = Workspace.initialize(tmp_path)
    imported = _import(workspace, report, source_root)
    candidate_id = imported.first_page.candidates[0].candidate_id
    runtime_run = ExecutionRunManifest(
        run_id="measured-runtime-run",
        execution_status=ExecutionStatus.SUCCEEDED,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=ValidationStatus.PASSED,
        finished_at=datetime.now(UTC),
        environment_id="sha256:" + "1" * 64,
        semantics=RunSemantics(origin="capture", adapter="pyperf", adapter_version="2.9"),
    )
    RunStore(workspace).create(runtime_run)
    GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row(runtime_run)]},
        publisher="static-candidate-link-test",
        publisher_version="1",
        input_run_ids=(runtime_run.run_id,),
    )
    finding = FindingService(workspace).record(
        RecordFindingRequest(
            kind="performance",
            title="Measured candidate",
            claim="The runtime measurement supports this candidate.",
            evidence_level=EvidenceLevel.DERIVED,
            confidence=FindingConfidence.MEDIUM,
            assessment=FindingAssessment.SUPPORTED,
            evidence=(
                EvidenceInput(
                    ref_type=EvidenceReferenceType.STATIC_CANDIDATE,
                    ref_id=candidate_id,
                    relation=EvidenceRelation.CONTEXT,
                ),
                EvidenceInput(
                    ref_type=EvidenceReferenceType.RUN,
                    ref_id=runtime_run.run_id,
                    relation=EvidenceRelation.SUPPORTS,
                ),
            ),
        )
    )

    [candidate] = StaticAnalysisService(workspace).candidates(run_id=imported.run_id).candidates
    loaded = FindingService(workspace).get(finding.finding.finding_id)

    assert candidate.related_finding_ids == (finding.finding.finding_id,)
    assert candidate.related_findings_truncated is False
    assert {
        (reference.ref_type, reference.ref_id, reference.relation) for reference in loaded.evidence
    } == {
        (
            EvidenceReferenceType.STATIC_CANDIDATE,
            candidate_id,
            EvidenceRelation.CONTEXT,
        ),
        (EvidenceReferenceType.RUN, runtime_run.run_id, EvidenceRelation.SUPPORTS),
    }

    with pytest.raises(DomainError, match="can only provide context"):
        FindingService(workspace).record(
            RecordFindingRequest(
                kind="performance",
                title="Unmeasured candidate",
                claim="The static candidate proves itself.",
                evidence_level=EvidenceLevel.DERIVED,
                confidence=FindingConfidence.LOW,
                assessment=FindingAssessment.SUPPORTED,
                evidence=(
                    EvidenceInput(
                        ref_type=EvidenceReferenceType.STATIC_CANDIDATE,
                        ref_id=candidate_id,
                        relation=EvidenceRelation.SUPPORTS,
                    ),
                ),
            )
        )


def test_static_analysis_scope_excludes_workspace_and_honors_explicit_scope(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path, workspace_root=tmp_path / "evidence-state")
    for relative in (
        "app.py",
        "build/accepted.py",
        "build/rejected.py",
        "evidence-state/private.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n")
    report = tmp_path / "analysis.sarif"
    _write_sarif(
        report,
        [
            _result("app.py", message="app"),
            _result("build/accepted.py", message="accepted"),
            _result("build/rejected.py", message="rejected"),
            _result("evidence-state/private.py", message="workspace"),
        ],
    )

    default_result = _import(workspace, report, tmp_path)
    explicit_result = _import(
        workspace,
        report,
        tmp_path,
        include_paths=("build",),
        exclude_paths=("build/rejected.py",),
    )

    assert {candidate.message for candidate in default_result.first_page.candidates} == {"app"}
    assert {candidate.message for candidate in explicit_result.first_page.candidates} == {
        "accepted"
    }
    assert explicit_result.coverage.excluded_count == 3


def test_static_analysis_import_records_exact_repository_source_state(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "app.py").write_text("pass\n")
    report = tmp_path / "analysis.sarif"
    _write_sarif(report, [_result("app.py")])
    _git(tmp_path, "add", "app.py", "analysis.sarif")
    _git(tmp_path, "commit", "-m", "initial")
    workspace = Workspace.initialize(tmp_path)

    result = _import(workspace, report, tmp_path)

    run = RunStore(workspace).read(result.run_id)
    assert run.source_state_id is not None
    assert "source_state" not in run.semantics.unavailable_fields
    with Catalog(workspace).open_snapshot(result.corpus_commit_id) as snapshot:
        assert snapshot.execute(
            "SELECT identity_quality FROM source_states WHERE source_state_id = ?",
            (run.source_state_id,),
        ).fetchone() == ("clean",)


def test_static_analysis_rejects_escaped_locations_and_supports_srcroot(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (source_root / "safe.py").write_text("pass\n")
    (outside / "escaped.py").write_text("pass\n")
    os.symlink(outside, source_root / "linked")
    report = tmp_path / "analysis.sarif"
    _write_sarif(
        report,
        [
            _result("safe.py", message="safe", base="SRCROOT"),
            _result("../outside/escaped.py", message="traversal"),
            _result((outside / "escaped.py").as_uri(), message="absolute"),
            _result("linked/escaped.py", message="symlink"),
        ],
    )

    result = _import(workspace, report, source_root)

    assert [candidate.message for candidate in result.first_page.candidates] == ["safe"]
    assert result.coverage.invalid_count == 3
    assert any("traverses outside" in limitation for limitation in result.limitations)
    assert any(
        "outside the declared source root" in limitation for limitation in result.limitations
    )


def test_static_analysis_unsupported_sarif_stays_native_only(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    report = tmp_path / "analysis.sarif"
    source = _write_sarif(report, [_result("missing.py")], version="2.0.0")

    result = _import(workspace, report, source_root)

    assert ArtifactStore(workspace).get(result.artifact_id).payload_path.read_bytes() == source
    assert result.coverage.normalized_count == 0
    assert result.first_page.evidence.status == "unavailable"
    assert result.first_page.evidence.reason == "static_candidates_not_published"
    assert any("SARIF 2.1.0" in limitation for limitation in result.limitations)


def test_static_analysis_keeps_completed_candidates_from_a_truncated_sarif(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    report = tmp_path / "analysis.sarif"
    complete = _write_sarif(report, [_result("candidate.py")])
    source = complete[:-1]
    report.write_bytes(source)

    result = _import(workspace, report, source_root)

    assert ArtifactStore(workspace).get(result.artifact_id).payload_path.read_bytes() == source
    assert result.coverage.model_dump() == {
        "result_count": 1,
        "normalized_count": 1,
        "excluded_count": 0,
        "invalid_count": 0,
        "omitted_count": 0,
    }
    assert any(
        "stopped before the document ended" in limitation for limitation in result.limitations
    )


def test_static_analysis_candidate_cursor_is_snapshot_bound(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    report = tmp_path / "analysis.sarif"
    _write_sarif(
        report,
        [_result("first.py", message="first"), _result("second.py", message="second")],
    )
    service = StaticAnalysisService(workspace)
    imported = _import(workspace, report, source_root)

    first_page = service.candidates(run_id=imported.run_id, limit=1)
    assert first_page.next_cursor is not None
    second_page = service.candidates(
        run_id=imported.run_id,
        limit=1,
        cursor=first_page.next_cursor,
    )
    assert {item.message for item in first_page.candidates + second_page.candidates} == {
        "first",
        "second",
    }

    _write_sarif(report, [_result("third.py", message="third")])
    _import(workspace, report, source_root)

    with pytest.raises(DomainError) as stale:
        service.candidates(run_id=imported.run_id, limit=1, cursor=first_page.next_cursor)

    assert stale.value.code is ErrorCode.STALE_CURSOR
