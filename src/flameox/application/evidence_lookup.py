from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from enum import Enum
from typing import Any, ClassVar, Literal, cast

import duckdb
from pydantic import JsonValue

from flameox.application.artifacts import ArtifactService, SnapshotArtifact
from flameox.application.recoverable_move import validate_manifest_id
from flameox.catalog import Catalog, Snapshot, SnapshotHandle
from flameox.domain import (
    AnalysisRecord,
    DomainError,
    EnvironmentRecord,
    ErrorCode,
    EvidenceReference,
    EvidenceReferenceType,
    Finding,
    Investigation,
    RunManifest,
    SourceState,
)
from flameox.models import ContractModel
from flameox.storage import GenerationManifest, Workspace


class EvidenceLookupResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    ref_type: EvidenceReferenceType
    ref_id: str
    data: dict[str, JsonValue]


class EvidenceSession:
    """One retained corpus snapshot for every read in an application request."""

    _TABLES: ClassVar[dict[EvidenceReferenceType, tuple[str, str]]] = {
        EvidenceReferenceType.ANALYSIS: ("analyses", "analysis_id"),
        EvidenceReferenceType.COMPARISON: ("comparisons", "comparison_id"),
        EvidenceReferenceType.OBSERVATION: ("observations", "observation_id"),
        EvidenceReferenceType.RUN_SET: ("run_sets", "run_set_id"),
        EvidenceReferenceType.TRIAL: ("trials", "trial_id"),
    }

    def __init__(self, workspace: Workspace, snapshot: Snapshot) -> None:
        self._workspace = workspace
        self._snapshot = snapshot

    @property
    def commit_id(self) -> str:
        return self._snapshot.handle.commit_id

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> duckdb.DuckDBPyConnection:
        return self._snapshot.execute(sql, parameters)

    def run(self, run_id: str) -> RunManifest:
        return self._snapshot.run(run_id)

    def environment(self, environment_id: str) -> EnvironmentRecord:
        row = self.execute(
            "SELECT observed_at, identity_quality, fields_json, missing_fields "
            "FROM environments WHERE environment_id = ? "
            "ORDER BY published_at DESC LIMIT 1",
            (environment_id,),
        ).fetchone()
        if row is None:
            raise self._missing("environment", environment_id)
        try:
            return EnvironmentRecord.model_validate(
                {
                    "environment_id": environment_id,
                    "observed_at": row[0],
                    "identity_quality": row[1],
                    "fields": json.loads(str(row[2])),
                    "missing_fields": row[3],
                }
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                f"Environment evidence {environment_id!r} is invalid.",
            ) from exc

    def source_state(self, source_state_id: str) -> SourceState:
        row = self.execute(
            "SELECT identity_quality, repository_root, head_commit, diff_digest, "
            "executable_digest, build_id, fields_json, missing_fields "
            "FROM source_states WHERE source_state_id = ? "
            "ORDER BY published_at DESC LIMIT 1",
            (source_state_id,),
        ).fetchone()
        if row is None:
            raise self._missing("source state", source_state_id)
        try:
            return SourceState.model_validate(
                {
                    "source_state_id": source_state_id,
                    "identity_quality": row[0],
                    "repository_root": row[1],
                    "head_commit": row[2],
                    "diff_digest": row[3],
                    "executable_digest": row[4],
                    "build_id": row[5],
                    "fields": json.loads(str(row[6])),
                    "missing_fields": row[7],
                }
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                f"Source-state evidence {source_state_id!r} is invalid.",
            ) from exc

    def artifact(self, artifact_id: str, *, limit: int = 100) -> SnapshotArtifact:
        return ArtifactService(self._workspace).resolve_at_snapshot(
            self._snapshot,
            artifact_id,
            limit=limit,
        )

    def analysis(self, analysis_id: str) -> AnalysisRecord:
        row = self.execute(
            "SELECT analysis_id, recipe, recipe_version, parameters_json, "
            "parameters_digest, corpus_commit_id, input_generation_ids, input_run_ids, "
            "input_artifact_ids, result_digest, result_artifact_id, coverage_json, "
            "limitations, started_at, completed_at FROM analyses WHERE analysis_id = ? "
            "ORDER BY published_at DESC LIMIT 1",
            (analysis_id,),
        ).fetchone()
        if row is None:
            raise self._missing("analysis", analysis_id)
        try:
            return AnalysisRecord.model_validate(
                {
                    "analysis_id": row[0],
                    "recipe": row[1],
                    "recipe_version": row[2],
                    "parameters": json.loads(str(row[3])),
                    "parameters_digest": row[4],
                    "corpus_commit_id": row[5],
                    "input_generation_ids": row[6],
                    "input_run_ids": row[7],
                    "input_artifact_ids": row[8],
                    "result_digest": row[9],
                    "result_artifact_id": row[10],
                    "coverage": json.loads(str(row[11])),
                    "limitations": row[12],
                    "started_at": row[13],
                    "completed_at": row[14],
                }
            )
        except (TypeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                f"Analysis {analysis_id!r} is invalid in snapshot {self.commit_id!r}.",
            ) from exc

    def references(
        self,
        *,
        owner_type: Literal["analysis", "finding", "hypothesis"],
        owner_id: str,
        owner_revision: int | None = None,
    ) -> tuple[EvidenceReference, ...]:
        revisioned = owner_type in {"finding", "hypothesis"}
        if revisioned and owner_revision is None:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                f"Evidence references for revisioned {owner_type} {owner_id!r} require "
                "an exact owner revision.",
            )
        if not revisioned and owner_revision is not None:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                f"Evidence references for immutable {owner_type} {owner_id!r} cannot "
                "specify an owner revision.",
            )
        revision_predicate = (
            "owner_revision = ?" if owner_revision is not None else "owner_revision IS NULL"
        )
        parameters: tuple[object, ...] = (
            (owner_type, owner_id, owner_revision)
            if owner_revision is not None
            else (owner_type, owner_id)
        )
        rows = self.execute(
            "SELECT owner_type, owner_id, owner_revision, ref_type, ref_id, relation "
            "FROM evidence_refs WHERE owner_type = ? AND owner_id = ? AND "
            f"{revision_predicate} "
            "QUALIFY row_number() OVER (PARTITION BY owner_type, owner_id, owner_revision, "
            "ref_type, ref_id, relation ORDER BY published_at DESC) = 1 "
            "ORDER BY ref_type, ref_id, relation",
            parameters,
        ).fetchall()
        if revisioned and not rows:
            legacy = self.execute(
                "SELECT 1 FROM evidence_refs WHERE owner_type = ? AND owner_id = ? "
                "AND owner_revision IS NULL LIMIT 1",
                (owner_type, owner_id),
            ).fetchone()
            if legacy is not None:
                raise DomainError(
                    ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                    f"Evidence references for {owner_type} {owner_id!r} predate exact "
                    "owner-revision binding and are ambiguous.",
                    details={"owner_revision": owner_revision, "corpus_commit_id": self.commit_id},
                )
        try:
            return tuple(
                EvidenceReference.model_validate(
                    {
                        "owner_type": row[0],
                        "owner_id": row[1],
                        "owner_revision": row[2],
                        "ref_type": row[3],
                        "ref_id": row[4],
                        "relation": row[5],
                    }
                )
                for row in rows
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                f"Evidence references for {owner_type} {owner_id!r} are invalid in "
                f"snapshot {self.commit_id!r}.",
            ) from exc

    def finding(self, finding_id: str) -> Finding:
        row = self.execute(
            "SELECT finding_id, revision, created_at, kind, title, claim, "
            "evidence_level, confidence, assessment, lifecycle, limitations, "
            "next_experiments_json FROM findings WHERE finding_id = ? "
            "ORDER BY revision DESC, published_at DESC LIMIT 1",
            (finding_id,),
        ).fetchone()
        if row is None:
            raise self._missing("finding", finding_id)
        return self._finding_from_row(row, finding_id=finding_id)

    def list_investigations(
        self,
        *,
        offset: int,
        limit: int,
    ) -> tuple[tuple[Investigation, ...], int]:
        count_row = self.execute(
            "SELECT count(DISTINCT investigation_id) FROM investigations"
        ).fetchone()
        assert count_row is not None
        rows = self.execute(
            "SELECT investigation_id, question, symptom, project_root, status, "
            "parent_investigation_id, created_at FROM investigations "
            "QUALIFY row_number() OVER (PARTITION BY investigation_id "
            "ORDER BY published_at DESC) = 1 "
            "ORDER BY created_at DESC, investigation_id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        try:
            values = tuple(
                Investigation.model_validate(
                    {
                        "investigation_id": row[0],
                        "question": row[1],
                        "symptom": row[2],
                        "project_root": row[3],
                        "status": row[4],
                        "parent_investigation_id": row[5],
                        "created_at": row[6],
                    }
                )
                for row in rows
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                f"Investigation evidence is invalid in snapshot {self.commit_id!r}.",
            ) from exc
        return values, int(count_row[0])

    def list_findings(
        self,
        *,
        offset: int,
        limit: int,
    ) -> tuple[tuple[Finding, ...], int]:
        count_row = self.execute("SELECT count(DISTINCT finding_id) FROM findings").fetchone()
        assert count_row is not None
        rows = self.execute(
            "SELECT finding_id, revision, created_at, kind, title, claim, "
            "evidence_level, confidence, assessment, lifecycle, limitations, "
            "next_experiments_json FROM findings "
            "QUALIFY row_number() OVER (PARTITION BY finding_id "
            "ORDER BY revision DESC, published_at DESC) = 1 "
            "ORDER BY created_at DESC, finding_id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return (
            tuple(self._finding_from_row(row, finding_id=str(row[0])) for row in rows),
            int(count_row[0]),
        )

    def _finding_from_row(self, row: tuple[object, ...], *, finding_id: str) -> Finding:
        try:
            return Finding.model_validate(
                {
                    "finding_id": row[0],
                    "revision": row[1],
                    "created_at": row[2],
                    "kind": row[3],
                    "title": row[4],
                    "claim": row[5],
                    "evidence_level": row[6],
                    "confidence": row[7],
                    "assessment": row[8],
                    "lifecycle": row[9],
                    "limitations": row[10],
                    "next_experiments": json.loads(str(row[11])),
                }
            )
        except (TypeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                f"Finding {finding_id!r} is invalid in snapshot {self.commit_id!r}.",
            ) from exc

    def get(
        self,
        ref_type: EvidenceReferenceType,
        ref_id: str,
    ) -> EvidenceLookupResult:
        if ref_type is EvidenceReferenceType.RUN:
            data = self.run(ref_id).model_dump(mode="json")
        elif ref_type is EvidenceReferenceType.ARTIFACT:
            data = self.artifact(ref_id).metadata.model_dump(mode="json")
        elif ref_type is EvidenceReferenceType.GENERATION:
            data = self._generation(ref_id).model_dump(mode="json")
        else:
            table, identifier = self._TABLES[ref_type]
            connection = self.execute(
                f'SELECT * FROM "{table}" WHERE "{identifier}" = ? '
                "ORDER BY published_at DESC LIMIT 1",
                (ref_id,),
            )
            row = connection.fetchone()
            if row is None and ref_type is EvidenceReferenceType.COMPARISON:
                connection = self.execute(
                    "SELECT * FROM kernel_validation_comparisons "
                    "WHERE comparison_id = ? ORDER BY published_at DESC LIMIT 1",
                    (ref_id,),
                )
                row = connection.fetchone()
            columns = [item[0] for item in connection.description]
            if row is None:
                raise self._missing(ref_type.value, ref_id)
            data = {name: _json_value(value) for name, value in zip(columns, row, strict=True)}
        return EvidenceLookupResult(
            corpus_commit_id=self.commit_id,
            ref_type=ref_type,
            ref_id=ref_id,
            data=cast(dict[str, JsonValue], data),
        )

    def _generation(self, generation_id: str) -> GenerationManifest:
        validate_manifest_id(generation_id, kind="generation")
        relative = f"generations/{generation_id}/manifest.json"
        if relative not in self._snapshot.commit.generation_manifests:
            raise self._missing(
                EvidenceReferenceType.GENERATION.value,
                generation_id,
            )
        path = self._workspace.paths.root / relative
        try:
            manifest = GenerationManifest.model_validate_json(path.read_text())
        except (OSError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Generation {generation_id!r} is invalid in the pinned corpus snapshot.",
                details={"corpus_commit_id": self.commit_id},
            ) from exc
        if manifest.generation_id != generation_id:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The generation manifest identity does not match its committed path.",
                details={"corpus_commit_id": self.commit_id},
            )
        return manifest

    def _missing(
        self,
        entity: str,
        ref_id: str,
    ) -> DomainError:
        return DomainError(
            ErrorCode.WORKSPACE_INVALID,
            f"{entity} evidence {ref_id!r} is absent from snapshot {self.commit_id!r}.",
            details={"missing_entity": entity, "corpus_commit_id": self.commit_id},
            remediation=(
                f"Use a {entity} list or query operation at this snapshot to discover a valid ID.",
            ),
        )


class EvidenceLookupService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    @contextmanager
    def session(
        self,
        handle: SnapshotHandle | str | None = None,
    ) -> Iterator[EvidenceSession]:
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(handle) as snapshot:
            yield EvidenceSession(self.workspace, snapshot)

    def get(
        self,
        ref_type: EvidenceReferenceType,
        ref_id: str,
    ) -> EvidenceLookupResult:
        with self.session() as session:
            return session.get(ref_type, ref_id)


def _json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return cast(JsonValue, value.value)
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
