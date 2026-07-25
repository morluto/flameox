from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from flamo.catalog import Catalog
from flamo.domain import DomainError
from flamo.evidence.schemas import schema_for
from flamo.storage import ArtifactStore, GenerationManifest, RunStore, Workspace


class IntegrityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Literal["error", "warning"]
    code: str
    path: str | None = None
    message: str


class IntegrityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    level: Literal["quick", "full"]
    corpus_commit_id: str
    valid: bool
    checked_artifacts: int
    checked_generations: int
    checked_parquet_files: int
    issues: tuple[IntegrityIssue, ...]


class IntegrityService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def validate(self, *, full: bool = False) -> IntegrityResult:
        issues: list[IntegrityIssue] = []
        head = self.workspace.corpus.read_head()
        checked_generations = 0
        checked_parquet = 0
        for relative in head.generation_manifests:
            path = (self.workspace.paths.root / relative).resolve()
            try:
                path.relative_to(self.workspace.paths.root)
                manifest = GenerationManifest.model_validate_json(path.read_text())
                checked_generations += 1
            except (ValueError, OSError) as exc:
                issues.append(
                    IntegrityIssue(
                        severity="error",
                        code="INVALID_GENERATION_MANIFEST",
                        path=str(path),
                        message=str(exc),
                    )
                )
                continue
            for evidence_file in manifest.files:
                evidence_path = (self.workspace.paths.root / evidence_file.path).resolve()
                try:
                    evidence_path.relative_to(self.workspace.paths.root)
                    if not evidence_path.is_file():
                        raise FileNotFoundError(evidence_path)
                    metadata = pq.read_metadata(evidence_path)
                    if metadata.num_rows != evidence_file.row_count:
                        raise ValueError("Parquet row count differs from its manifest")
                    arrow_schema = pq.read_schema(evidence_path)
                    expected_schema = schema_for(evidence_file.table)
                    if (
                        not arrow_schema.equals(
                            expected_schema,
                            check_metadata=False,
                        )
                        or arrow_schema.metadata != expected_schema.metadata
                    ):
                        raise ValueError("Parquet schema differs from the schema registry")
                    if full and _sha256(evidence_path) != evidence_file.sha256:
                        raise ValueError("Parquet bytes differ from their manifest digest")
                    checked_parquet += 1
                except (OSError, ValueError) as exc:
                    issues.append(
                        IntegrityIssue(
                            severity="error",
                            code="INVALID_PARQUET",
                            path=str(evidence_path),
                            message=str(exc),
                        )
                    )

        checked_artifacts = 0
        for metadata_path in self.workspace.paths.artifacts.glob("*/*/artifact.json"):
            artifact_id = f"sha256:{metadata_path.parent.name}"
            try:
                stored = ArtifactStore(self.workspace).get(artifact_id)
                if not stored.payload_path.is_file():
                    raise FileNotFoundError(stored.payload_path)
                if full and _sha256(stored.payload_path) != stored.content.integrity.sha256:
                    raise ValueError("Artifact bytes differ from their content identity")
                checked_artifacts += 1
            except (DomainError, OSError, ValueError) as exc:
                issues.append(
                    IntegrityIssue(
                        severity="error",
                        code="INVALID_ARTIFACT",
                        path=str(metadata_path),
                        message=str(exc),
                    )
                )
        for projection in self.workspace.paths.runs.glob("*/manifest.json"):
            try:
                RunStore(self.workspace).read(projection.parent.name)
            except DomainError as exc:
                issues.append(
                    IntegrityIssue(
                        severity="error",
                        code="INVALID_RUN",
                        path=str(projection),
                        message=exc.message,
                    )
                )
        try:
            catalog = Catalog(self.workspace).status()
            if not catalog["fresh"]:
                issues.append(
                    IntegrityIssue(
                        severity="warning",
                        code="STALE_CATALOG",
                        message="The rebuildable catalog does not match corpus HEAD.",
                    )
                )
        except DomainError as exc:
            issues.append(
                IntegrityIssue(
                    severity="warning",
                    code="CATALOG_UNAVAILABLE",
                    message=exc.message,
                )
            )
        return IntegrityResult(
            level="full" if full else "quick",
            corpus_commit_id=head.commit_id,
            valid=not any(issue.severity == "error" for issue in issues),
            checked_artifacts=checked_artifacts,
            checked_generations=checked_generations,
            checked_parquet_files=checked_parquet,
            issues=tuple(issues),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
