from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, cast

from pydantic import Field

from flameox.action_graph import ARTIFACT_PREVIEW_MAX_BYTES, ARTIFACT_PREVIEW_MAX_LINES
from flameox.catalog import Catalog, Snapshot
from flameox.domain import (
    ArtifactContent,
    ArtifactKind,
    CursorNamespace,
    DomainError,
    ErrorCode,
    EvidenceReferenceType,
    Sensitivity,
)
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import ArtifactStore, Workspace


class ArtifactRegistrationSummary(ContractModel):
    registration_id: str
    run_id: str
    display_name: str
    kind: ArtifactKind
    media_type: str
    sensitivity: Sensitivity
    role: str
    producer: str | None
    producer_version: str | None
    registered_at: datetime


class ArtifactReductionProvenance(ContractModel):
    reduction_id: str
    source_run_id: str
    source_registration_id: str
    original_artifact_id: str
    disposition: str
    role: str


class ArtifactMetadataResult(ContractModel):
    content: ArtifactContent
    resource_uri: str
    registrations: tuple[ArtifactRegistrationSummary, ...]
    total_registrations: int
    reduction_provenance: tuple[ArtifactReductionProvenance, ...]
    total_reductions: int
    reduction_provenance_next_cursor: str | None = None
    effective_sensitivity: Sensitivity


class ArtifactListItem(ContractModel):
    artifact_id: str
    byte_length: int
    effective_sensitivity: Sensitivity
    registration_count: int
    kinds: tuple[ArtifactKind, ...]


class ArtifactListResult(CursorPageContract):
    page_items_field = "artifacts"

    corpus_commit_id: str
    artifacts: tuple[ArtifactListItem, ...]
    total: int


class ArtifactReductionListResult(CursorPageContract):
    page_items_field = "reductions"

    corpus_commit_id: str
    artifact_id: str
    reductions: tuple[ArtifactReductionProvenance, ...]
    total: int


class ArtifactTextPreview(ContractModel):
    artifact_id: str
    kinds: tuple[ArtifactKind, ...]
    effective_sensitivity: Sensitivity
    encoding: str
    offset: Annotated[int, Field(ge=0)]
    returned_bytes: Annotated[int, Field(ge=0)]
    total_bytes: Annotated[int, Field(ge=0)]
    returned_lines: Annotated[int, Field(ge=0)]
    text: str
    truncated: bool
    next_offset: Annotated[int, Field(ge=0)] | None = None


@dataclass(frozen=True, slots=True)
class SnapshotArtifact:
    """An artifact whose registration and effective policy came from one snapshot."""

    metadata: ArtifactMetadataResult
    payload_path: Path


class ArtifactService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def get(self, artifact_id: str, *, limit: int = 100) -> ArtifactMetadataResult:
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin()) as snapshot:
            return self.get_at_snapshot(snapshot, artifact_id, limit=limit)

    def preview_text(
        self,
        artifact_id: str,
        *,
        offset: int,
        max_bytes: int,
        max_lines: int,
    ) -> ArtifactTextPreview:
        if (
            offset < 0
            or not 1 <= max_bytes <= ARTIFACT_PREVIEW_MAX_BYTES
            or not 1 <= max_lines <= ARTIFACT_PREVIEW_MAX_LINES
        ):
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Artifact preview requires a non-negative offset, "
                f"1-{ARTIFACT_PREVIEW_MAX_BYTES} bytes, and "
                f"1-{ARTIFACT_PREVIEW_MAX_LINES} lines.",
            )
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin()) as snapshot:
            artifact = self.resolve_at_snapshot(snapshot, artifact_id)
            kind_rows = snapshot.execute(
                "SELECT DISTINCT kind FROM artifact_registrations WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchall()
        kinds = {ArtifactKind(row[0]) for row in kind_rows}
        eligible = {ArtifactKind.PROCESS_OUTPUT, ArtifactKind.VALIDATION_OUTPUT}
        if not kinds.intersection(eligible):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Text preview is available only for process and validation output artifacts.",
                details={"artifact_id": artifact_id, "kinds": sorted(kind.value for kind in kinds)},
            )
        if artifact.metadata.effective_sensitivity is Sensitivity.SENSITIVE:
            raise DomainError(
                ErrorCode.SENSITIVE_ARTIFACT_REFUSED,
                "Sensitive artifact content cannot be previewed through this read-only surface.",
                details={"artifact_id": artifact_id},
            )
        total_bytes = artifact.metadata.content.byte_length
        if offset > total_bytes:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                f"Artifact preview offset {offset} exceeds its {total_bytes}-byte length.",
            )
        content, selected = ArtifactStore(self.workspace).read_range(
            artifact_id,
            offset=offset,
            max_bytes=max_bytes,
        )
        if content != artifact.metadata.content:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Artifact content identity changed after snapshot resolution.",
            )
        lines = selected.splitlines(keepends=True)
        if len(lines) > max_lines:
            selected = b"".join(lines[:max_lines])
        try:
            text = selected.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            if error.end == len(selected) and offset + len(selected) < total_bytes:
                if error.start == 0:
                    raise DomainError(
                        ErrorCode.QUERY_BUDGET_EXCEEDED,
                        "Artifact preview byte bound ends inside the next UTF-8 character.",
                        details={
                            "artifact_id": artifact_id,
                            "offset": offset,
                            "suggested_max_bytes": min(max_bytes + 4, ARTIFACT_PREVIEW_MAX_BYTES),
                        },
                    ) from error
                selected = selected[: error.start]
                text = selected.decode("utf-8", errors="strict")
            else:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "Artifact preview is not valid UTF-8 text.",
                    details={
                        "artifact_id": artifact_id,
                        "encoding": "utf-8",
                        "offset": offset + error.start,
                    },
                ) from error
        returned_bytes = len(selected)
        next_offset = offset + returned_bytes
        truncated = next_offset < total_bytes
        return ArtifactTextPreview(
            artifact_id=artifact_id,
            kinds=tuple(sorted(kinds, key=lambda kind: kind.value)),
            effective_sensitivity=artifact.metadata.effective_sensitivity,
            encoding="utf-8",
            offset=offset,
            returned_bytes=returned_bytes,
            total_bytes=total_bytes,
            returned_lines=len(selected.splitlines()),
            text=text,
            truncated=truncated,
            next_offset=next_offset if truncated else None,
        )

    def get_at_snapshot(
        self,
        snapshot: Snapshot,
        artifact_id: str,
        *,
        limit: int = 100,
    ) -> ArtifactMetadataResult:
        return self.resolve_at_snapshot(snapshot, artifact_id, limit=limit).metadata

    def resolve_at_snapshot(
        self,
        snapshot: Snapshot,
        artifact_id: str,
        *,
        limit: int = 100,
    ) -> SnapshotArtifact:
        rows = snapshot.execute(
            "SELECT registration_id, run_id, display_name, kind, media_type, "
            "sensitivity, role, producer, producer_version, registered_at "
            "FROM artifact_registrations WHERE artifact_id = ? "
            "ORDER BY registered_at DESC, registration_id LIMIT ?",
            (artifact_id, limit),
        ).fetchall()
        count_row = snapshot.execute(
            "SELECT count(*), max(CASE sensitivity WHEN 'sensitive' THEN 2 "
            "WHEN 'internal' THEN 1 ELSE 0 END) "
            "FROM artifact_registrations WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        assert count_row is not None
        if int(count_row[0]) == 0:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Artifact {artifact_id!r} is absent from the pinned corpus snapshot.",
                details={
                    "missing_entity": EvidenceReferenceType.ARTIFACT.value,
                    "corpus_commit_id": snapshot.handle.commit_id,
                },
            )
        stored = ArtifactStore(self.workspace).get(artifact_id)
        registrations = tuple(
            ArtifactRegistrationSummary(
                registration_id=row[0],
                run_id=row[1],
                display_name=row[2],
                kind=row[3],
                media_type=row[4],
                sensitivity=row[5],
                role=row[6],
                producer=row[7],
                producer_version=row[8],
                registered_at=row[9],
            )
            for row in rows
        )
        reduction_rows = snapshot.execute(
            "SELECT reduction_id, source_run_id, source_registration_id, "
            "original_artifact_id, disposition, final_artifact_id "
            "FROM reduction_results "
            "WHERE final_artifact_id = ? OR best_known_artifact_id = ? "
            "ORDER BY reduction_id LIMIT ?",
            (artifact_id, artifact_id, limit + 1),
        ).fetchall()
        reduction_count = snapshot.execute(
            "SELECT count(*) FROM reduction_results "
            "WHERE final_artifact_id = ? OR best_known_artifact_id = ?",
            (artifact_id, artifact_id),
        ).fetchone()
        assert reduction_count is not None
        metadata = ArtifactMetadataResult(
            content=stored.content,
            resource_uri=f"flameox://artifacts/{artifact_id}",
            registrations=registrations,
            total_registrations=int(count_row[0]),
            reduction_provenance=tuple(
                ArtifactReductionProvenance(
                    reduction_id=str(row[0]),
                    source_run_id=str(row[1]),
                    source_registration_id=str(row[2]),
                    original_artifact_id=str(row[3]),
                    disposition=str(row[4]),
                    role="final" if row[5] == artifact_id else "best_known",
                )
                for row in reduction_rows[:limit]
            ),
            total_reductions=int(reduction_count[0]),
            reduction_provenance_next_cursor=(
                self.workspace.cursors.issue(
                    namespace=CursorNamespace.ARTIFACT_REDUCTIONS,
                    snapshot_id=snapshot.handle.commit_id,
                    scope_digest=artifact_id,
                    position=(str(reduction_rows[limit - 1][0]),),
                )
                if len(reduction_rows) > limit
                else None
            ),
            effective_sensitivity={
                None: Sensitivity.NORMAL,
                0: Sensitivity.NORMAL,
                1: Sensitivity.INTERNAL,
                2: Sensitivity.SENSITIVE,
            }[count_row[1]],
        )
        return SnapshotArtifact(metadata=metadata, payload_path=stored.payload_path)

    def list_reductions(
        self,
        artifact_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> ArtifactReductionListResult:
        head = self.workspace.corpus.read_head()
        after = (
            cast(
                tuple[str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.ARTIFACT_REDUCTIONS,
                    snapshot_id=head.commit_id,
                    scope_digest=artifact_id,
                ),
            )[0]
            if cursor is not None
            else None
        )
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            registered = snapshot.execute(
                "SELECT 1 FROM artifact_registrations WHERE artifact_id = ? LIMIT 1",
                (artifact_id,),
            ).fetchone()
            if registered is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Artifact {artifact_id!r} is absent from the pinned corpus snapshot.",
                    details={
                        "missing_entity": EvidenceReferenceType.ARTIFACT.value,
                        "corpus_commit_id": snapshot.handle.commit_id,
                    },
                )
            count_row = snapshot.execute(
                "SELECT count(*) FROM reduction_results "
                "WHERE final_artifact_id = ? OR best_known_artifact_id = ?",
                (artifact_id, artifact_id),
            ).fetchone()
            assert count_row is not None
            predicate = "AND reduction_id > ? " if after is not None else ""
            parameters: tuple[object, ...] = (
                (artifact_id, artifact_id, after, limit + 1)
                if after is not None
                else (artifact_id, artifact_id, limit + 1)
            )
            rows = snapshot.execute(
                "SELECT reduction_id, source_run_id, source_registration_id, "
                "original_artifact_id, disposition, final_artifact_id "
                "FROM reduction_results "
                "WHERE (final_artifact_id = ? OR best_known_artifact_id = ?) "
                + predicate
                + "ORDER BY reduction_id LIMIT ?",
                parameters,
            ).fetchall()
            commit_id = snapshot.handle.commit_id
        has_more = len(rows) > limit
        reductions = tuple(
            ArtifactReductionProvenance(
                reduction_id=str(row[0]),
                source_run_id=str(row[1]),
                source_registration_id=str(row[2]),
                original_artifact_id=str(row[3]),
                disposition=str(row[4]),
                role="final" if row[5] == artifact_id else "best_known",
            )
            for row in rows[:limit]
        )
        return ArtifactReductionListResult(
            corpus_commit_id=commit_id,
            artifact_id=artifact_id,
            reductions=reductions,
            total=int(count_row[0]),
            next_cursor=(
                self.workspace.cursors.issue(
                    namespace=CursorNamespace.ARTIFACT_REDUCTIONS,
                    snapshot_id=commit_id,
                    scope_digest=artifact_id,
                    position=(reductions[-1].reduction_id,),
                )
                if has_more and reductions
                else None
            ),
        )

    def list(self, *, limit: int = 100, cursor: str | None = None) -> ArtifactListResult:
        head = self.workspace.corpus.read_head()
        after = (
            cast(
                tuple[str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.ARTIFACTS,
                    snapshot_id=head.commit_id,
                    scope_digest="all",
                ),
            )[0]
            if cursor is not None
            else None
        )
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            count_row = snapshot.execute(
                "SELECT count(DISTINCT artifact_id) FROM artifact_registrations"
            ).fetchone()
            assert count_row is not None
            rows = snapshot.execute(
                "SELECT artifact_id, max(byte_length), "
                "max(CASE sensitivity WHEN 'sensitive' THEN 2 "
                "WHEN 'internal' THEN 1 ELSE 0 END), count(*), "
                "list_sort(list_distinct(list(kind))) "
                "FROM artifact_registrations "
                + ("WHERE artifact_id > ? " if after is not None else "")
                + "GROUP BY artifact_id "
                "ORDER BY artifact_id LIMIT ?",
                ((after, limit + 1) if after is not None else (limit + 1,)),
            ).fetchall()
            commit_id = snapshot.commit.commit_id
        sensitivity = {
            0: Sensitivity.NORMAL,
            1: Sensitivity.INTERNAL,
            2: Sensitivity.SENSITIVE,
        }
        has_more = len(rows) > limit
        artifacts = tuple(
            ArtifactListItem(
                artifact_id=row[0],
                byte_length=row[1],
                effective_sensitivity=sensitivity[row[2]],
                registration_count=row[3],
                kinds=tuple(row[4]),
            )
            for row in rows[:limit]
        )
        total = int(count_row[0])
        return ArtifactListResult(
            corpus_commit_id=commit_id,
            artifacts=artifacts,
            total=total,
            next_cursor=(
                self.workspace.cursors.issue(
                    namespace=CursorNamespace.ARTIFACTS,
                    snapshot_id=head.commit_id,
                    scope_digest="all",
                    position=(artifacts[-1].artifact_id,),
                )
                if has_more and artifacts
                else None
            ),
        )
