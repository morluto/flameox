from __future__ import annotations

from datetime import datetime

from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactContent,
    ArtifactKind,
    CursorCodec,
    DomainError,
    ErrorCode,
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


class ArtifactMetadataResult(ContractModel):
    schema_version: int = 1
    content: ArtifactContent
    resource_uri: str
    registrations: tuple[ArtifactRegistrationSummary, ...]
    total_registrations: int
    effective_sensitivity: Sensitivity


class ArtifactListItem(ContractModel):
    artifact_id: str
    byte_length: int
    effective_sensitivity: Sensitivity
    registration_count: int
    kinds: tuple[ArtifactKind, ...]


class ArtifactListResult(CursorPageContract):
    page_items_field = "artifacts"

    schema_version: int = 1
    corpus_commit_id: str
    artifacts: tuple[ArtifactListItem, ...]
    total: int


class ArtifactService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def get(self, artifact_id: str, *, limit: int = 100) -> ArtifactMetadataResult:
        stored = ArtifactStore(self.workspace).get(artifact_id)
        with Catalog(self.workspace).open_snapshot() as snapshot:
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
        return ArtifactMetadataResult(
            content=stored.content,
            resource_uri=f"flameox://artifacts/{artifact_id}",
            registrations=registrations,
            total_registrations=int(count_row[0]),
            effective_sensitivity={
                None: Sensitivity.NORMAL,
                0: Sensitivity.NORMAL,
                1: Sensitivity.INTERNAL,
                2: Sensitivity.SENSITIVE,
            }[count_row[1]],
        )

    def list(self, *, limit: int = 100, cursor: str | None = None) -> ArtifactListResult:
        head = self.workspace.corpus.read_head()
        after = (
            CursorCodec.decode(
                cursor,
                namespace="artifacts",
                snapshot_id=head.commit_id,
                scope_digest="all",
            )[0]
            if cursor is not None
            else None
        )
        if after is not None and not isinstance(after, str):
            raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
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
                CursorCodec.encode(
                    namespace="artifacts",
                    snapshot_id=head.commit_id,
                    scope_digest="all",
                    position=(artifacts[-1].artifact_id,),
                )
                if has_more and artifacts
                else None
            ),
        )
