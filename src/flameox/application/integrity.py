from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import ConfigDict, computed_field

from flameox.catalog import Catalog
from flameox.domain import DomainError
from flameox.evidence.schemas import schema_for
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, CorpusCommit, GenerationManifest, RunStore, Workspace


class IntegritySeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class IntegrityLevel(StrEnum):
    QUICK = "quick"
    FULL = "full"


class IntegrityIssue(ContractModel):
    severity: IntegritySeverity
    code: str
    path: str | None = None
    message: str


class IntegrityResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: int = 1
    level: IntegrityLevel
    corpus_commit_id: str
    checked_artifacts: int
    checked_generations: int
    checked_parquet_files: int
    issues: tuple[IntegrityIssue, ...]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def valid(self) -> bool:
        return not any(issue.severity is IntegritySeverity.ERROR for issue in self.issues)


class IntegrityService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def _check_manifests(
        self,
        head: CorpusCommit,
        level: IntegrityLevel,
    ) -> tuple[int, int, list[IntegrityIssue]]:
        issues: list[IntegrityIssue] = []
        checked_generations = 0
        checked_parquet = 0
        for relative in head.generation_manifests:
            path = (self.workspace.paths.root / relative).resolve()
            try:
                path.relative_to(self.workspace.paths.root)
                manifest = GenerationManifest.model_validate_json(path.read_text())
                checked_generations += 1
            except FileNotFoundError:
                issues.append(
                    IntegrityIssue(
                        severity=IntegritySeverity.ERROR,
                        code="MISSING_GENERATION_MANIFEST",
                        path=str(path),
                        message="Generation manifest is missing from the corpus.",
                    )
                )
                continue
            except OSError as exc:
                issues.append(
                    IntegrityIssue(
                        severity=IntegritySeverity.ERROR,
                        code="UNREADABLE_GENERATION_MANIFEST",
                        path=str(path),
                        message=str(exc),
                    )
                )
                continue
            except ValueError as exc:
                # ValidationError is a ValueError subclass and is the
                # intended target; a stray ValueError from elsewhere would
                # otherwise be misreported as a corrupt manifest.
                issues.append(
                    IntegrityIssue(
                        severity=IntegritySeverity.ERROR,
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
                    if (
                        level is IntegrityLevel.FULL
                        and _sha256(evidence_path) != evidence_file.sha256
                    ):
                        raise ValueError("Parquet bytes differ from their manifest digest")
                    checked_parquet += 1
                except (OSError, ValueError) as exc:
                    issues.append(
                        IntegrityIssue(
                            severity=IntegritySeverity.ERROR,
                            code="INVALID_PARQUET",
                            path=str(evidence_path),
                            message=str(exc),
                        )
                    )

        return checked_generations, checked_parquet, issues

    def validate(self, level: IntegrityLevel = IntegrityLevel.QUICK) -> IntegrityResult:
        issues: list[IntegrityIssue] = []
        head = self.workspace.corpus.read_head()
        checked_generations, checked_parquet, manifest_issues = self._check_manifests(head, level)
        issues.extend(manifest_issues)

        checked_artifacts = 0
        for metadata_path in self.workspace.paths.artifacts.glob("*/*/artifact.json"):
            artifact_id = f"sha256:{metadata_path.parent.name}"
            try:
                stored = ArtifactStore(self.workspace).get(artifact_id)
                if not stored.payload_path.is_file():
                    raise FileNotFoundError(stored.payload_path)
                if (
                    level is IntegrityLevel.FULL
                    and _sha256(stored.payload_path) != stored.content.integrity.sha256
                ):
                    raise ValueError("Artifact bytes differ from their content identity")
                checked_artifacts += 1
            except (DomainError, OSError, ValueError) as exc:
                issues.append(
                    IntegrityIssue(
                        severity=IntegritySeverity.ERROR,
                        code="INVALID_ARTIFACT",
                        path=str(metadata_path),
                        message=str(exc),
                    )
                )
        try:
            RunStore(self.workspace).list()
        except DomainError as exc:
            issues.append(
                IntegrityIssue(
                    severity=IntegritySeverity.ERROR,
                    code="INVALID_CONTROL_PLANE_RUN",
                    path=str(self.workspace.paths.control_plane),
                    message=exc.message,
                )
            )
        try:
            catalog = Catalog(self.workspace).status()
            if not catalog["fresh"]:
                issues.append(
                    IntegrityIssue(
                        severity=IntegritySeverity.WARNING,
                        code="STALE_CATALOG",
                        message="The rebuildable catalog does not match corpus HEAD.",
                    )
                )
        except DomainError as exc:
            issues.append(
                IntegrityIssue(
                    severity=IntegritySeverity.WARNING,
                    code="CATALOG_UNAVAILABLE",
                    message=exc.message,
                )
            )
        return IntegrityResult(
            level=level,
            corpus_commit_id=head.commit_id,
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
