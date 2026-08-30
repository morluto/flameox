from __future__ import annotations

from enum import StrEnum

import pyarrow.parquet as pq
from pydantic import ConfigDict, computed_field

from flameox.action_graph import ActionId, ToolAction, tool_action
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode
from flameox.evidence.schemas import schema_for
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, CorpusCommit, RunStore, Workspace
from flameox.storage.corpus import file_sha256


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
    next_action: ToolAction | None = None


class IntegrityResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

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
        for generation_id in head.generation_ids:
            path = self.workspace.corpus.generation_path(generation_id)
            try:
                manifest = self.workspace.corpus.read_generation(generation_id)
                checked_generations += 1
            except DomainError as exc:
                issues.append(
                    IntegrityIssue(
                        severity=IntegritySeverity.ERROR,
                        code=(
                            "MISSING_GENERATION_MANIFEST"
                            if exc.code is ErrorCode.WORKSPACE_INVALID
                            else "INVALID_GENERATION_MANIFEST"
                        ),
                        path=str(path),
                        message=exc.message,
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
                    if evidence_path.stat().st_size != evidence_file.byte_length:
                        raise ValueError("Parquet byte length differs from its manifest")
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
                        and file_sha256(evidence_path) != evidence_file.sha256
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
                if level is IntegrityLevel.FULL and file_sha256(
                    stored.payload_path
                ) != stored.content.artifact_id.removeprefix("sha256:"):
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
            Catalog(self.workspace).status()
        except DomainError as exc:
            issues.append(
                IntegrityIssue(
                    severity=IntegritySeverity.WARNING,
                    code="CATALOG_UNAVAILABLE",
                    message=exc.message,
                    next_action=tool_action(ActionId.REBUILD_CATALOG),
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
