from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from flameox.catalog import Catalog
from flameox.domain import ArtifactContent, Sensitivity
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, Workspace


class ArtifactRegistrationSummary(ContractModel):
    registration_id: str
    run_id: str
    display_name: str
    kind: str
    media_type: str
    sensitivity: Sensitivity
    role: str
    producer: str | None
    producer_version: str | None
    registered_at: datetime


class ArtifactMetadataResult(ContractModel):
    schema_version: int = 1
    content: ArtifactContent
    local_handle: str
    registrations: tuple[ArtifactRegistrationSummary, ...]
    total_registrations: int
    effective_sensitivity: Sensitivity


class ArtifactListItem(ContractModel):
    artifact_id: str
    byte_length: int
    effective_sensitivity: Sensitivity
    registration_count: int
    kinds: tuple[str, ...]


class ArtifactListResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    artifacts: tuple[ArtifactListItem, ...]
    total: int
    returned: int
    truncated: bool


class ArtifactService:
    _SENSITIVITY_ORDER: ClassVar[dict[Sensitivity, int]] = {
        Sensitivity.NORMAL: 0,
        Sensitivity.INTERNAL: 1,
        Sensitivity.SENSITIVE: 2,
    }

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
                "SELECT count(*) FROM artifact_registrations WHERE artifact_id = ?",
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
            local_handle=str(stored.payload_path),
            registrations=registrations,
            total_registrations=int(count_row[0]),
            effective_sensitivity=max(
                (item.sensitivity for item in registrations),
                default=Sensitivity.NORMAL,
                key=self._SENSITIVITY_ORDER.__getitem__,
            ),
        )

    def list(self, *, limit: int = 100) -> ArtifactListResult:
        with Catalog(self.workspace).open_snapshot() as snapshot:
            count_row = snapshot.execute(
                "SELECT count(DISTINCT artifact_id) FROM artifact_registrations"
            ).fetchone()
            assert count_row is not None
            rows = snapshot.execute(
                "SELECT artifact_id, max(byte_length), "
                "max(CASE sensitivity WHEN 'sensitive' THEN 2 "
                "WHEN 'internal' THEN 1 ELSE 0 END), count(*), "
                "list_sort(list_distinct(list(kind))) "
                "FROM artifact_registrations GROUP BY artifact_id "
                "ORDER BY artifact_id LIMIT ?",
                (limit,),
            ).fetchall()
            commit_id = snapshot.commit.commit_id
        sensitivity = {
            0: Sensitivity.NORMAL,
            1: Sensitivity.INTERNAL,
            2: Sensitivity.SENSITIVE,
        }
        artifacts = tuple(
            ArtifactListItem(
                artifact_id=row[0],
                byte_length=row[1],
                effective_sensitivity=sensitivity[row[2]],
                registration_count=row[3],
                kinds=tuple(row[4]),
            )
            for row in rows
        )
        total = int(count_row[0])
        return ArtifactListResult(
            corpus_commit_id=commit_id,
            artifacts=artifacts,
            total=total,
            returned=len(artifacts),
            truncated=total > len(artifacts),
        )
