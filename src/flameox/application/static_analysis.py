from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import cast

from pydantic import Field, JsonValue

from flameox.adapters.sarif import (
    DEFAULT_EXCLUDE_PATHS,
    SarifAnalyzer,
    SarifCandidate,
    SarifCoverage,
    parse_sarif,
)
from flameox.application.imports import ImportService
from flameox.application.source import collect_import_source_state
from flameox.catalog import Catalog, Snapshot
from flameox.domain import CursorNamespace, DomainError, ErrorCode, IdentityQuality, digest_model
from flameox.domain.models import ArtifactKind, Digest, RunSemantics, Sensitivity
from flameox.evidence import GenerationPublisher
from flameox.evidence_status import (
    EvidenceAvailability,
    available_availability,
    empty_availability,
    unavailable_availability,
)
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import RunStore, Workspace

_MAX_RELATED_FINDINGS = 16
_MAX_SELECTOR_LENGTH = 256
_MAX_PROJECTED_SOURCE_ROOT_LENGTH = 512


class ImportStaticAnalysisRequest(ContractModel):
    path: Path
    source_root: Path
    include_paths: tuple[str, ...] = Field(default=(), max_length=128)
    exclude_paths: tuple[str, ...] = Field(default=(), max_length=128)
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    allow_external_path: bool = False


class StaticCandidate(ContractModel):
    candidate_id: str
    run_id: str
    artifact_id: str
    rule_id: str | None
    level: str | None
    message: str
    relative_path: str
    start_line: int | None
    start_column: int | None
    end_line: int | None
    end_column: int | None
    provider_fingerprint: str | None
    provider_confidence: float | None
    related_finding_ids: tuple[str, ...] = Field(max_length=_MAX_RELATED_FINDINGS)
    related_findings_truncated: bool = False


class StaticAnalysisCoverage(ContractModel):
    result_count: int
    normalized_count: int
    excluded_count: int
    invalid_count: int
    omitted_count: int

    @classmethod
    def from_sarif(cls, coverage: SarifCoverage) -> StaticAnalysisCoverage:
        return cls(
            result_count=coverage.result_count,
            normalized_count=coverage.normalized_count,
            excluded_count=coverage.excluded_count,
            invalid_count=coverage.invalid_count,
            omitted_count=coverage.omitted_count,
        )


class StaticAnalyzerProjection(ContractModel):
    name: str = Field(min_length=1, max_length=256)
    version: str | None = Field(default=None, min_length=1, max_length=256)


class StaticAnalysisSemanticsProjection(ContractModel):
    """Small interpretation context; the linked run owns complete semantics."""

    semantic_id: Digest
    source_root: str = Field(min_length=1, max_length=_MAX_PROJECTED_SOURCE_ROOT_LENGTH)
    source_root_truncated: bool
    analyzers: tuple[StaticAnalyzerProjection, ...] = Field(max_length=16)


class StaticCandidateQueryResult(CursorPageContract):
    page_items_field = "candidates"

    corpus_commit_id: str
    candidates: tuple[StaticCandidate, ...]
    total: int
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class StaticAnalysisImportResult(ContractModel):
    run_id: str
    artifact_id: str
    corpus_commit_id: str
    coverage: StaticAnalysisCoverage
    limitations: tuple[str, ...]
    first_page: StaticCandidateQueryResult
    semantics: StaticAnalysisSemanticsProjection


class StaticAnalysisService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)

    def import_sarif(self, request: ImportStaticAnalysisRequest) -> StaticAnalysisImportResult:
        source_root = self._source_root(request.source_root)
        include_paths = _normalize_scope_paths(request.include_paths, "include")
        exclude_paths = _normalize_scope_paths(request.exclude_paths, "exclude")
        default_excludes = self._default_excludes(source_root)
        source_state = collect_import_source_state(self.workspace)
        imports = ImportService(self.workspace)
        with imports.snapshot_provider_document(
            request.path,
            allow_external_path=request.allow_external_path,
            max_bytes=self.workspace.config.capture.max_artifact_bytes,
        ) as snapshot:
            parsed = parse_sarif(
                snapshot.payload_path,
                source_root=source_root,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                default_exclude_paths=default_excludes,
                maximum_candidates=self.workspace.config.storage.max_rows_per_generation,
            )
            coverage = StaticAnalysisCoverage.from_sarif(parsed.coverage)
            limitations = list(parsed.limitations)
            if source_state.identity_quality is IdentityQuality.PARTIAL:
                limitations.append("Exact repository source state is unavailable for this import.")
            semantics = _run_semantics(
                source_root=source_root,
                project_root=self.workspace.project_root,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                default_excludes=default_excludes,
                analyzers=parsed.analyzers,
                exit_status=parsed.exit_status,
                coverage=coverage,
                source_state_exact=source_state.identity_quality is not IdentityQuality.PARTIAL,
                sarif_supported=parsed.supported,
            )
            imported = imports.import_snapshot(
                snapshot,
                kind=ArtifactKind.ANALYSIS_RESULT,
                sensitivity=request.sensitivity,
                display_name=request.path.name,
                media_type="application/sarif+json",
                role="primary",
                producer=_producer_name(parsed.analyzers),
                producer_version=_producer_version(parsed.analyzers),
                semantics=semantics,
                limitations=tuple(limitations),
                source_state=source_state,
            )

        if _sarif_supported(imported.run.semantics):
            rows = [
                _candidate_row(candidate, imported.run.run_id, imported.artifact_id)
                for candidate in parsed.candidates
            ]
            published = GenerationPublisher(self.workspace).publish_rows(
                {"static_candidates": rows},
                publisher="flameox.sarif",
                publisher_version="1",
                input_run_ids=(imported.run.run_id,),
                input_artifact_ids=(imported.artifact_id,),
            )
            corpus_commit_id = published.commit.commit_id
        else:
            corpus_commit_id = imported.corpus_commit_id
        first_page = self.candidates(
            run_id=imported.run.run_id,
            limit=self.workspace.config.analysis.default_row_limit,
        )
        return StaticAnalysisImportResult(
            run_id=imported.run.run_id,
            artifact_id=imported.artifact_id,
            corpus_commit_id=corpus_commit_id,
            coverage=coverage,
            limitations=tuple(limitations),
            first_page=first_page,
            semantics=_semantics_projection(
                imported.run.semantics,
                source_root=source_root,
                project_root=self.workspace.project_root,
                analyzers=parsed.analyzers,
            ),
        )

    def candidates(
        self,
        *,
        run_id: str,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> StaticCandidateQueryResult:
        run = self.runs.read(run_id)
        bounded = limit or self.workspace.config.analysis.default_row_limit
        if bounded < 1 or bounded > self.workspace.config.analysis.max_row_limit:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Limit must be between 1 and {self.workspace.config.analysis.max_row_limit}.",
            )
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model({"run_id": run_id})
        after = (
            cast(
                tuple[str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.STATIC_CANDIDATES,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )[0]
            if cursor is not None
            else None
        )
        where = "run_id = ?"
        parameters: list[object] = [run_id]
        if after is not None:
            where += " AND candidate_id > ?"
            parameters.append(after)
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            count = snapshot.execute(
                "SELECT count(*) FROM static_candidates WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            assert count is not None
            rows = snapshot.execute(
                "SELECT candidate_id, run_id, artifact_id, rule_id, level, message, relative_path, "
                "start_line, start_column, end_line, end_column, provider_fingerprint, "
                "provider_confidence FROM static_candidates WHERE "
                + where
                + " ORDER BY candidate_id LIMIT ?",
                (*parameters, bounded + 1),
            ).fetchall()
            selected_rows = rows[:bounded]
            related_findings = _related_findings(
                snapshot,
                tuple(cast(str, row[0]) for row in selected_rows),
            )
        candidates = tuple(
            StaticCandidate(
                candidate_id=row[0],
                run_id=row[1],
                artifact_id=row[2],
                rule_id=row[3],
                level=row[4],
                message=row[5],
                relative_path=row[6],
                start_line=row[7],
                start_column=row[8],
                end_line=row[9],
                end_column=row[10],
                provider_fingerprint=row[11],
                provider_confidence=row[12],
                related_finding_ids=related_findings[cast(str, row[0])][0],
                related_findings_truncated=related_findings[cast(str, row[0])][1],
            )
            for row in selected_rows
        )
        has_more = len(rows) > bounded
        return StaticCandidateQueryResult(
            corpus_commit_id=head.commit_id,
            candidates=candidates,
            total=int(count[0]),
            next_cursor=(
                self.workspace.cursors.issue(
                    namespace=CursorNamespace.STATIC_CANDIDATES,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                    position=(candidates[-1].candidate_id,),
                )
                if has_more and candidates
                else None
            ),
            evidence=(
                available_availability("static_candidates_present")
                if candidates
                else (
                    empty_availability("no_static_candidates")
                    if _sarif_supported(run.semantics)
                    else unavailable_availability("static_candidates_not_published")
                )
            ),
        )

    def _source_root(self, value: Path) -> Path:
        try:
            source_root = value.resolve(strict=True)
            source_root.relative_to(self.workspace.project_root)
        except (OSError, ValueError) as error:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "The static-analysis source root must be an existing project directory.",
            ) from error
        if not source_root.is_dir():
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "The static-analysis source root must be a directory.",
            )
        return source_root

    def _default_excludes(self, source_root: Path) -> tuple[str, ...]:
        defaults = list(DEFAULT_EXCLUDE_PATHS)
        try:
            workspace_relative = self.workspace.paths.root.resolve().relative_to(source_root)
        except ValueError:
            pass
        else:
            defaults.append(workspace_relative.as_posix())
        return tuple(dict.fromkeys(defaults))


def _related_findings(
    snapshot: Snapshot,
    candidate_ids: tuple[str, ...],
) -> dict[str, tuple[tuple[str, ...], bool]]:
    result: dict[str, tuple[tuple[str, ...], bool]] = {
        candidate_id: ((), False) for candidate_id in candidate_ids
    }
    if not candidate_ids:
        return result
    placeholders = ", ".join("?" for _ in candidate_ids)
    rows = snapshot.execute(
        f"""
        WITH current_findings AS (
            SELECT finding_id, revision, created_at
            FROM findings
            QUALIFY row_number() OVER (
                PARTITION BY finding_id ORDER BY revision DESC, published_at DESC
            ) = 1
        ), deduplicated_links AS (
            SELECT refs.ref_id AS candidate_id, refs.owner_id AS finding_id,
                   current_findings.created_at,
                   row_number() OVER (
                       PARTITION BY refs.owner_id, refs.owner_revision, refs.ref_id
                       ORDER BY refs.published_at DESC
                   ) AS duplicate_rank
            FROM evidence_refs AS refs
            JOIN current_findings
              ON current_findings.finding_id = refs.owner_id
             AND current_findings.revision = refs.owner_revision
            WHERE refs.owner_type = 'finding'
              AND refs.ref_type = 'static_candidate'
              AND refs.relation = 'context'
              AND refs.ref_id IN ({placeholders})
        ), ranked_links AS (
            SELECT candidate_id, finding_id,
                   row_number() OVER (
                       PARTITION BY candidate_id ORDER BY created_at DESC, finding_id
                   ) AS link_rank,
                   count(*) OVER (PARTITION BY candidate_id) AS total
            FROM deduplicated_links
            WHERE duplicate_rank = 1
        )
        SELECT candidate_id, finding_id, total
        FROM ranked_links
        WHERE link_rank <= ?
        ORDER BY candidate_id, link_rank
        """,
        (*candidate_ids, _MAX_RELATED_FINDINGS),
    ).fetchall()
    finding_ids: dict[str, list[str]] = {candidate_id: [] for candidate_id in candidate_ids}
    totals: dict[str, int] = {}
    for candidate_id, finding_id, total in rows:
        candidate = cast(str, candidate_id)
        finding_ids[candidate].append(cast(str, finding_id))
        totals[candidate] = cast(int, total)
    return {
        candidate_id: (
            tuple(finding_ids[candidate_id]),
            totals.get(candidate_id, 0) > len(finding_ids[candidate_id]),
        )
        for candidate_id in candidate_ids
    }


def _normalize_scope_paths(values: tuple[str, ...], kind: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not value or "\x00" in value or "\\" in value:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                f"Static-analysis {kind} paths must be non-empty normalized relative paths.",
            )
        if len(value) > _MAX_SELECTOR_LENGTH:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                f"Static-analysis {kind} paths must be at most {_MAX_SELECTOR_LENGTH} characters.",
            )
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                f"Static-analysis {kind} paths must stay within the declared source root.",
            )
        normalized.append(path.as_posix())
    if len(normalized) != len(set(normalized)):
        raise DomainError(
            ErrorCode.INVALID_ARGUMENTS,
            f"Static-analysis {kind} paths must be unique.",
        )
    return tuple(normalized)


def _run_semantics(
    *,
    source_root: Path,
    project_root: Path,
    include_paths: tuple[str, ...],
    exclude_paths: tuple[str, ...],
    default_excludes: tuple[str, ...],
    analyzers: tuple[SarifAnalyzer, ...],
    exit_status: int | None,
    coverage: StaticAnalysisCoverage,
    source_state_exact: bool,
    sarif_supported: bool,
) -> RunSemantics:
    source_root_path = source_root.relative_to(project_root).as_posix() or "."
    configuration: dict[str, JsonValue] = {
        "source_root": source_root_path,
        "include_paths": list(include_paths),
        "exclude_paths": list(exclude_paths),
        "effective_excludes": list(default_excludes),
        "parse_coverage": cast(JsonValue, coverage.model_dump(mode="json")),
    }
    if sarif_supported:
        configuration["sarif_version"] = "2.1.0"
    analyzer_values: list[dict[str, JsonValue]] = [
        {"name": analyzer.name, "version": analyzer.version} for analyzer in analyzers
    ]
    if len(analyzer_values) == 1:
        configuration["analyzer"] = analyzer_values[0]["name"]
        if analyzer_values[0]["version"] is not None:
            configuration["analyzer_version"] = analyzer_values[0]["version"]
    elif analyzer_values:
        configuration["analyzers"] = cast(JsonValue, analyzer_values)
    if exit_status is not None:
        configuration["analyzer_exit_status"] = exit_status
    unavailable = []
    if not analyzer_values:
        unavailable.append("analyzer")
    if not source_state_exact:
        unavailable.append("source_state")
    return RunSemantics(
        origin="import",
        adapter="sarif",
        configuration=configuration,
        unavailable_fields=tuple(unavailable),
    )


def _producer_name(analyzers: tuple[SarifAnalyzer, ...]) -> str | None:
    return analyzers[0].name if len(analyzers) == 1 else None


def _producer_version(analyzers: tuple[SarifAnalyzer, ...]) -> str | None:
    return analyzers[0].version if len(analyzers) == 1 else None


def _semantics_projection(
    semantics: RunSemantics,
    *,
    source_root: Path,
    project_root: Path,
    analyzers: tuple[SarifAnalyzer, ...],
) -> StaticAnalysisSemanticsProjection:
    relative_root = source_root.relative_to(project_root).as_posix() or "."
    return StaticAnalysisSemanticsProjection(
        semantic_id=semantics.semantic_id,
        source_root=relative_root[:_MAX_PROJECTED_SOURCE_ROOT_LENGTH],
        source_root_truncated=len(relative_root) > _MAX_PROJECTED_SOURCE_ROOT_LENGTH,
        analyzers=tuple(
            StaticAnalyzerProjection(name=analyzer.name, version=analyzer.version)
            for analyzer in analyzers
        ),
    )


def _candidate_row(candidate: SarifCandidate, run_id: str, artifact_id: str) -> dict[str, object]:
    candidate_id = digest_model(
        {
            "run_id": run_id,
            "artifact_id": artifact_id,
            "run_index": candidate.run_index,
            "result_index": candidate.result_index,
            "provider_fingerprint": candidate.provider_fingerprint,
        }
    )
    return {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "artifact_id": artifact_id,
        "rule_id": candidate.rule_id,
        "level": candidate.level,
        "message": candidate.message,
        "relative_path": candidate.relative_path,
        "start_line": candidate.start_line,
        "start_column": candidate.start_column,
        "end_line": candidate.end_line,
        "end_column": candidate.end_column,
        "provider_fingerprint": candidate.provider_fingerprint,
        "provider_confidence": candidate.provider_confidence,
    }


def _sarif_supported(semantics: RunSemantics) -> bool:
    return semantics.configuration.get("sarif_version") == "2.1.0"
