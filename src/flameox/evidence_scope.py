from __future__ import annotations

from flameox.catalog import Snapshot
from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel


class EvidenceScope(ContractModel):
    input_ids: tuple[str, ...]
    run_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    def predicate(
        self,
        *,
        run_column: str,
        artifact_column: str,
    ) -> tuple[str, tuple[object, ...]]:
        predicates: list[str] = []
        parameters: tuple[object, ...] = ()
        if self.run_ids:
            placeholders = ", ".join("?" for _ in self.run_ids)
            predicates.append(f"{run_column} IN ({placeholders})")
            parameters += self.run_ids
        if self.artifact_ids:
            placeholders = ", ".join("?" for _ in self.artifact_ids)
            predicates.append(f"{artifact_column} IN ({placeholders})")
            parameters += self.artifact_ids
        if not predicates:
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Analysis input has no scope.")
        return " OR ".join(f"({predicate})" for predicate in predicates), parameters


def resolve_evidence_scope(
    snapshot: Snapshot,
    input_ids: str | tuple[str, ...],
) -> EvidenceScope:
    values = (input_ids,) if isinstance(input_ids, str) else input_ids
    run_ids: set[str] = set()
    artifact_ids: set[str] = set()
    for input_id in values:
        if input_id.startswith("sha256:"):
            present = snapshot.execute(
                "SELECT 1 FROM artifact_registrations WHERE artifact_id = ? LIMIT 1",
                (input_id,),
            ).fetchone()
            if present is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Artifact is absent from the pinned corpus: {input_id}",
                )
            artifact_ids.add(input_id)
            continue
        present = snapshot.execute(
            "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1",
            (input_id,),
        ).fetchone()
        if present is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Run is absent from the pinned corpus: {input_id}",
            )
        run_ids.add(input_id)
        artifact_ids.update(
            str(row[0])
            for row in snapshot.execute(
                "SELECT DISTINCT artifact_id FROM artifact_registrations "
                "WHERE run_id = ? ORDER BY artifact_id",
                (input_id,),
            ).fetchall()
        )
    return EvidenceScope(
        input_ids=values,
        run_ids=tuple(sorted(run_ids)),
        artifact_ids=tuple(sorted(artifact_ids)),
    )
