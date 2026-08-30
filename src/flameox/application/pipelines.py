from __future__ import annotations

import codecs
import hashlib
import os
import stat
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, assert_never, cast

from pydantic import (
    Field,
    JsonValue,
    computed_field,
    model_validator,
)

from flameox.domain import (
    CursorNamespace,
    DomainError,
    ErrorCode,
    RunManifest,
    Sensitivity,
    digest_model,
)
from flameox.domain.compiler_build import compiler_identity_id, compiler_target_identity_id
from flameox.domain.models import utc_now
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import ArtifactStore, ControlRecordStore, RunStore, Workspace
from flameox.storage.control_plane import _serialize_control_payload


class PipelineStageStatus(StrEnum):
    AVAILABLE = "available"
    CACHED = "cached"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class PipelineStageDisposition(StrEnum):
    IDENTICAL = "identical"
    CHANGED = "changed"
    CONTENT_CHANGED = "content_changed"
    ADDED = "added"
    MISSING = "missing"
    REORDERED = "reordered"
    INCOMPATIBLE = "incompatible"
    UNINSPECTABLE = "uninspectable"


class PipelineCompatibility(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class PipelineIdentityQuality(StrEnum):
    MANAGED_EXACT = "managed_exact"
    MANAGED_PARTIAL = "managed_partial"
    LOCAL_DECLARED = "local_declared"
    IMPORTED_UNVERIFIED = "imported_unverified"


class PipelineExtractorProfile(StrEnum):
    TEXT_LINES_V1 = "text-lines-v1"


class _PipelineStageDeclaration(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    ordinal: Annotated[int, Field(ge=0, le=99)]
    predecessor: str | None = None
    format: Annotated[str, Field(min_length=1, max_length=100)]
    format_schema: Annotated[str, Field(min_length=1, max_length=100)]
    extractor_profile: PipelineExtractorProfile | None = None
    limitations: Annotated[tuple[str, ...], Field(max_length=20)] = ()


class RegisteredPipelineStageDeclaration(_PipelineStageDeclaration):
    status: Literal[PipelineStageStatus.AVAILABLE, PipelineStageStatus.CACHED]
    registration_id: str


class UnregisteredPipelineStageDeclaration(_PipelineStageDeclaration):
    status: Literal[
        PipelineStageStatus.SKIPPED,
        PipelineStageStatus.UNAVAILABLE,
        PipelineStageStatus.FAILED,
    ]
    registration_id: None = None
    extractor_profile: Literal[None] = None


type PipelineStageDeclaration = Annotated[
    RegisteredPipelineStageDeclaration | UnregisteredPipelineStageDeclaration,
    Field(discriminator="status"),
]


class RegisterPipelineRequest(ContractModel):
    run_id: str
    pipeline_name: Annotated[str, Field(min_length=1, max_length=100)]
    producer: Annotated[str, Field(min_length=1, max_length=100)]
    compiler_identity_id: str | None = None
    target_identity_id: str | None = None
    stages: Annotated[tuple[PipelineStageDeclaration, ...], Field(max_length=100)]
    limitations: Annotated[tuple[str, ...], Field(max_length=20)] = ()

    @model_validator(mode="after")
    def ordered_lineage(self) -> RegisterPipelineRequest:
        names = [stage.name for stage in self.stages]
        ordinals = [stage.ordinal for stage in self.stages]
        if len(names) != len(set(names)) or len(ordinals) != len(set(ordinals)):
            raise ValueError("pipeline stage names and ordinals must be unique")
        if ordinals != sorted(ordinals):
            raise ValueError("pipeline stages must be declared in ordinal order")
        seen: set[str] = set()
        for stage in self.stages:
            if stage.predecessor is not None and stage.predecessor not in seen:
                raise ValueError("stage predecessors must identify an earlier declared stage")
            seen.add(stage.name)
        if self.target_identity_id is not None and self.compiler_identity_id is None:
            raise ValueError("target identity requires a compiler identity")
        return self


class _PipelineStage(ContractModel):
    name: str
    ordinal: int
    predecessor: str | None
    format: str
    format_schema: str
    extractor_profile: PipelineExtractorProfile | None = None
    extractor: str | None
    extractor_version: str | None
    extraction_operation_id: str | None = None
    structural_summary: dict[str, JsonValue] | None
    elapsed_ns: int | None
    limitations: tuple[str, ...]


class RegisteredPipelineStage(_PipelineStage):
    status: Literal[PipelineStageStatus.AVAILABLE, PipelineStageStatus.CACHED]
    registration_id: str
    artifact_id: str
    artifact_length: int
    sensitivity: Sensitivity


class UnregisteredPipelineStage(_PipelineStage):
    status: Literal[
        PipelineStageStatus.SKIPPED,
        PipelineStageStatus.UNAVAILABLE,
        PipelineStageStatus.FAILED,
    ]
    registration_id: Literal[None] = None
    artifact_id: Literal[None] = None
    artifact_length: Literal[None] = None
    sensitivity: Literal[None] = None


type PipelineStage = Annotated[
    RegisteredPipelineStage | UnregisteredPipelineStage,
    Field(discriminator="status"),
]


class ArtifactPipeline(ContractModel):
    pipeline_id: str
    run_id: str
    pipeline_name: str
    producer: str
    identity_quality: PipelineIdentityQuality
    compiler_identity_id: str | None = None
    target_identity_id: str | None = None
    stages: tuple[PipelineStage, ...]
    limitations: Annotated[tuple[str, ...], Field(max_length=20)]
    created_at: datetime = Field(default_factory=utc_now)


class PipelineFilter(ContractModel):
    run_id: str | None = None
    pipeline_name: str | None = None
    producer: str | None = None
    source_artifact_id: str | None = None


class PipelineSummary(ContractModel):
    pipeline_id: str
    run_id: str
    pipeline_name: str
    producer: str
    identity_quality: PipelineIdentityQuality
    stage_count: int
    artifact_ids: Annotated[tuple[str, ...], Field(max_length=100)]
    limitation_count: int
    created_at: datetime


class PipelineListResult(CursorPageContract):
    page_items_field = "pipelines"

    corpus_commit_id: str
    pipelines: tuple[PipelineSummary, ...]
    total: int
    next_cursor: str | None = None


class PipelineDetail(ContractModel):
    pipeline: ArtifactPipeline
    compatible_pipeline_ids: Annotated[tuple[str, ...], Field(max_length=20)]
    compatible_pipeline_count: int
    candidates_truncated: bool


class _PipelineStageComparison(ContractModel):
    stage_name: str
    limitations: tuple[str, ...] = ()


class AddedPipelineStageComparison(_PipelineStageComparison):
    baseline_ordinal: Literal[None] = None
    candidate_ordinal: int
    disposition: Literal[PipelineStageDisposition.ADDED] = PipelineStageDisposition.ADDED
    baseline_artifact_id: Literal[None] = None
    candidate_artifact_id: str | None = None
    artifact_length_change: Literal[None] = None
    baseline_summary: Literal[None] = None
    candidate_summary: Literal[None] = None


class MissingPipelineStageComparison(_PipelineStageComparison):
    baseline_ordinal: int
    candidate_ordinal: Literal[None] = None
    disposition: Literal[PipelineStageDisposition.MISSING] = PipelineStageDisposition.MISSING
    baseline_artifact_id: str | None = None
    candidate_artifact_id: Literal[None] = None
    artifact_length_change: Literal[None] = None
    baseline_summary: Literal[None] = None
    candidate_summary: Literal[None] = None


type _PairedPipelineStageDisposition = Literal[
    PipelineStageDisposition.IDENTICAL,
    PipelineStageDisposition.CHANGED,
    PipelineStageDisposition.CONTENT_CHANGED,
    PipelineStageDisposition.REORDERED,
    PipelineStageDisposition.INCOMPATIBLE,
    PipelineStageDisposition.UNINSPECTABLE,
]


class PairedPipelineStageComparison(_PipelineStageComparison):
    baseline_ordinal: int
    candidate_ordinal: int
    disposition: _PairedPipelineStageDisposition
    baseline_artifact_id: str | None = None
    candidate_artifact_id: str | None = None
    artifact_length_change: int | None = None
    baseline_summary: dict[str, JsonValue] | None = None
    candidate_summary: dict[str, JsonValue] | None = None


type PipelineStageComparison = Annotated[
    AddedPipelineStageComparison | MissingPipelineStageComparison | PairedPipelineStageComparison,
    Field(discriminator="disposition"),
]


def _first_observed_divergent_stage(
    compatibility: PipelineCompatibility,
    stages: tuple[PipelineStageComparison, ...],
) -> str | None:
    if compatibility is not PipelineCompatibility.COMPATIBLE:
        return None
    for stage in stages:
        if stage.disposition is PipelineStageDisposition.IDENTICAL:
            continue
        if stage.disposition is PipelineStageDisposition.UNINSPECTABLE:
            return None
        return stage.stage_name
    return None


class PipelineComparison(ContractModel):
    baseline_pipeline_id: str
    candidate_pipeline_id: str
    compatibility: PipelineCompatibility
    identity_mismatches: tuple[str, ...]
    stages: tuple[PipelineStageComparison, ...]
    input_artifact_ids: tuple[str, ...]
    extractor_identities: tuple[str, ...]
    limitations: tuple[str, ...]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def first_observed_divergent_stage(self) -> str | None:
        return _first_observed_divergent_stage(self.compatibility, self.stages)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def comparison_id(self) -> str:
        return digest_model(self.content_without_identity())

    def content_without_identity(self) -> dict[str, JsonValue]:
        return {
            "baseline_pipeline_id": self.baseline_pipeline_id,
            "candidate_pipeline_id": self.candidate_pipeline_id,
            "compatibility": self.compatibility,
            "identity_mismatches": list(self.identity_mismatches),
            "stages": [stage.model_dump(mode="json") for stage in self.stages],
            "first_observed_divergent_stage": self.first_observed_divergent_stage,
            "input_artifact_ids": list(self.input_artifact_ids),
            "extractor_identities": list(self.extractor_identities),
            "limitations": list(self.limitations),
        }


def _pipeline_identity_compatibility(
    baseline: ArtifactPipeline,
    candidate: ArtifactPipeline,
    baseline_run: RunManifest,
    candidate_run: RunManifest,
) -> tuple[PipelineCompatibility, tuple[str, ...], tuple[str, ...]]:
    mismatches = [
        name
        for name in ("pipeline_name", "producer")
        if getattr(baseline, name) != getattr(candidate, name)
    ]
    unknown_identities: list[str] = []
    for name in ("compiler_identity_id", "target_identity_id"):
        baseline_value = getattr(baseline, name)
        candidate_value = getattr(candidate, name)
        if baseline_value is None or candidate_value is None:
            unknown_identities.append(name)
        elif baseline_value != candidate_value:
            mismatches.append(name)
    if not all(
        pipeline.identity_quality is PipelineIdentityQuality.MANAGED_EXACT
        for pipeline in (baseline, candidate)
    ):
        unknown_identities.append("identity_quality")
    for name in (
        "workload_definition_id",
        "workload_instance_id",
        "source_state_id",
    ):
        baseline_value = getattr(baseline_run, name)
        candidate_value = getattr(candidate_run, name)
        if baseline_value is None or candidate_value is None:
            unknown_identities.append(name)
        elif baseline_value != candidate_value:
            mismatches.append(name)
    if baseline_run.environment_id != candidate_run.environment_id:
        mismatches.append("environment_id")
    compatibility = PipelineCompatibility.COMPATIBLE
    if mismatches:
        compatibility = PipelineCompatibility.INCOMPATIBLE
    elif unknown_identities:
        compatibility = PipelineCompatibility.UNKNOWN
    return compatibility, tuple(mismatches), tuple(unknown_identities)


def _merge_pipeline_limitations(
    declared: tuple[str, ...],
    authority: tuple[str, ...],
) -> tuple[str, ...]:
    declared_unique = tuple(dict.fromkeys(declared))
    authority_unique = tuple(
        item for item in dict.fromkeys(authority) if item not in declared_unique
    )
    if len(declared_unique) + len(authority_unique) <= 20:
        return (*declared_unique, *authority_unique)
    available = max(0, 19 - len(authority_unique))
    summary = "Additional limitations remain available in the source pipeline document."
    return (*declared_unique[:available], summary, *authority_unique[:19])[:20]


class ArtifactPipelineService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.pipelines = ControlRecordStore(
            workspace,
            kind="artifact_pipelines",
            model=ArtifactPipeline,
            id_field="pipeline_id",
        )
        self.publisher = GenerationPublisher(workspace)

    @staticmethod
    def _summary(pipeline: ArtifactPipeline) -> PipelineSummary:
        return PipelineSummary(
            pipeline_id=pipeline.pipeline_id,
            run_id=pipeline.run_id,
            pipeline_name=pipeline.pipeline_name,
            producer=pipeline.producer,
            identity_quality=pipeline.identity_quality,
            stage_count=len(pipeline.stages),
            artifact_ids=tuple(
                dict.fromkeys(
                    stage.artifact_id for stage in pipeline.stages if stage.artifact_id is not None
                )
            ),
            limitation_count=len(pipeline.limitations),
            created_at=pipeline.created_at,
        )

    def list(
        self,
        *,
        filter: PipelineFilter,
        limit: int,
        cursor: str | None = None,
    ) -> PipelineListResult:
        if not 1 <= limit <= 1_000:
            raise DomainError(ErrorCode.INVALID_ARGUMENTS, "Pipeline list limit must be 1-1000.")
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model(filter)
        after: tuple[datetime, str] | None = None
        if cursor is not None:
            position = cast(
                tuple[str, str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.PIPELINES,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )
            try:
                after = (datetime.fromisoformat(position[0]), position[1])
            except ValueError as exc:
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.") from exc

        def selected(pipeline: ArtifactPipeline) -> bool:
            return all(
                value is None or getattr(pipeline, name) == value
                for name, value in (
                    ("run_id", filter.run_id),
                    ("pipeline_name", filter.pipeline_name),
                    ("producer", filter.producer),
                )
            ) and (
                filter.source_artifact_id is None
                or any(stage.artifact_id == filter.source_artifact_id for stage in pipeline.stages)
            )

        pipelines = sorted(
            (pipeline for pipeline in self.pipelines.list() if selected(pipeline)),
            key=lambda item: (item.created_at, item.pipeline_id),
            reverse=True,
        )
        total = len(pipelines)
        if after is not None:
            pipelines = [item for item in pipelines if (item.created_at, item.pipeline_id) < after]
        page = pipelines[: limit + 1]
        items = tuple(self._summary(item) for item in page[:limit])
        return PipelineListResult(
            corpus_commit_id=head.commit_id,
            pipelines=items,
            total=total,
            next_cursor=(
                self.workspace.cursors.issue(
                    namespace=CursorNamespace.PIPELINES,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                    position=(items[-1].created_at.isoformat(), items[-1].pipeline_id),
                )
                if len(page) > limit and items
                else None
            ),
        )

    def get(self, pipeline_id: str, *, candidate_limit: int = 20) -> PipelineDetail:
        if not 0 <= candidate_limit <= 20:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Pipeline candidate limit must be 0-20.",
            )
        pipeline = self.pipelines.read(pipeline_id)
        pipeline_run = self.runs.read(pipeline.run_id)
        compatible = tuple(
            item.pipeline_id
            for item in sorted(
                self.pipelines.list(),
                key=lambda item: (item.created_at, item.pipeline_id),
                reverse=True,
            )
            if item.pipeline_id != pipeline.pipeline_id
            and _pipeline_identity_compatibility(
                pipeline,
                item,
                pipeline_run,
                self.runs.read(item.run_id),
            )[0]
            is not PipelineCompatibility.INCOMPATIBLE
        )
        return PipelineDetail(
            pipeline=pipeline,
            compatible_pipeline_ids=compatible[:candidate_limit],
            compatible_pipeline_count=len(compatible),
            candidates_truncated=len(compatible) > candidate_limit,
        )

    def resolve_reference(self, reference: str) -> ArtifactPipeline:
        pipelines = self.pipelines.list()
        direct = next((item for item in pipelines if item.pipeline_id == reference), None)
        if direct is not None:
            return direct
        matches = tuple(item for item in pipelines if item.run_id == reference)
        if len(matches) == 1:
            return matches[0]
        raise DomainError(
            ErrorCode.INVALID_ARGUMENTS,
            "Pipeline reference must be a pipeline ID or a run with exactly one pipeline.",
            details={
                "reference": reference,
                "matching_pipeline_ids": tuple(item.pipeline_id for item in matches[:20]),
                "matching_pipeline_count": len(matches),
            },
        )

    def extend_with_registration(
        self,
        pipeline_id: str,
        registration_id: str,
        *,
        stage_name: str,
        format_schema: str,
    ) -> ArtifactPipeline:
        pipeline = self.pipelines.read(pipeline_id)
        run = self.runs.read(pipeline.run_id)
        registration = next(
            (item for item in run.artifacts if item.registration_id == registration_id),
            None,
        )
        if registration is None:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Pipeline extension registration is not attached to the pipeline run.",
            )
        if any(stage.name == stage_name for stage in pipeline.stages):
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                f"Pipeline already contains stage {stage_name!r}.",
            )
        if len(pipeline.stages) == 100:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Pipeline cannot be extended beyond 100 stages.",
            )
        declarations: list[PipelineStageDeclaration] = []
        for stage in pipeline.stages:
            if isinstance(stage, RegisteredPipelineStage):
                declarations.append(
                    RegisteredPipelineStageDeclaration(
                        name=stage.name,
                        ordinal=stage.ordinal,
                        predecessor=stage.predecessor,
                        status=stage.status,
                        registration_id=stage.registration_id,
                        format=stage.format,
                        format_schema=stage.format_schema,
                        extractor_profile=stage.extractor_profile,
                        limitations=stage.limitations,
                    )
                )
            else:
                declarations.append(
                    UnregisteredPipelineStageDeclaration(
                        name=stage.name,
                        ordinal=stage.ordinal,
                        predecessor=stage.predecessor,
                        status=stage.status,
                        format=stage.format,
                        format_schema=stage.format_schema,
                        limitations=stage.limitations,
                    )
                )
        last = pipeline.stages[-1] if pipeline.stages else None
        declarations.append(
            RegisteredPipelineStageDeclaration(
                name=stage_name,
                ordinal=last.ordinal + 1 if last is not None else 0,
                predecessor=last.name if last is not None else None,
                status=PipelineStageStatus.AVAILABLE,
                registration_id=registration.registration_id,
                format=registration.media_type,
                format_schema=format_schema,
            )
        )
        request = RegisterPipelineRequest(
            run_id=pipeline.run_id,
            pipeline_name=pipeline.pipeline_name,
            producer=pipeline.producer,
            compiler_identity_id=pipeline.compiler_identity_id,
            target_identity_id=pipeline.target_identity_id,
            stages=tuple(declarations),
            limitations=pipeline.limitations,
        )
        return self._register(
            request,
            identity_quality=pipeline.identity_quality,
            derived_from=pipeline,
        )

    def register(self, request: RegisterPipelineRequest) -> ArtifactPipeline:
        """Register locally declared pipeline identity without external provenance."""

        exact_claims = (request.compiler_identity_id, request.target_identity_id)
        if any(item is not None for item in exact_claims):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Generic pipeline registration cannot authenticate exact identity claims.",
            )
        return self._register(request, identity_quality=PipelineIdentityQuality.LOCAL_DECLARED)

    def register_imported(self, request: RegisterPipelineRequest) -> ArtifactPipeline:
        """Register imported native evidence without claiming managed run identity."""

        limitation = "Imported compiler evidence has no authoritative execution identity."
        return self._register(
            request,
            identity_quality=PipelineIdentityQuality.IMPORTED_UNVERIFIED,
            additional_limitations=(limitation,),
        )

    def register_managed(
        self,
        request: RegisterPipelineRequest,
    ) -> ArtifactPipeline:
        """Authenticate compiler qualification against authoritative run semantics."""

        run = self.runs.read(request.run_id)
        if run.semantics.origin != "capture":
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Managed compiler lineage requires a captured run.",
            )
        qualification = run.compiler_qualification
        expected_compiler = compiler_identity_id(qualification)
        try:
            expected_target = compiler_target_identity_id(qualification)
        except ValueError as error:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Managed compiler qualification is invalid in the authoritative run.",
            ) from error
        if qualification is not None and (
            qualification.compiler.adapter != run.semantics.adapter
            or qualification.compiler.version != run.semantics.adapter_version
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Managed compiler qualification contradicts run semantics.",
            )
        if (
            request.compiler_identity_id != expected_compiler
            or request.target_identity_id != expected_target
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Compiler qualification does not match the authoritative managed run.",
            )
        exact = expected_compiler is not None and expected_target is not None
        additional_limitations: tuple[str, ...] = ()
        if not exact:
            additional_limitations = (
                "Managed pipeline identity is partial because compiler or target qualification "
                "is unavailable in run semantics.",
            )
        return self._register(
            request,
            identity_quality=(
                PipelineIdentityQuality.MANAGED_EXACT
                if exact
                else PipelineIdentityQuality.MANAGED_PARTIAL
            ),
            additional_limitations=additional_limitations,
        )

    def _register(
        self,
        request: RegisterPipelineRequest,
        *,
        identity_quality: PipelineIdentityQuality,
        additional_limitations: tuple[str, ...] = (),
        derived_from: ArtifactPipeline | None = None,
    ) -> ArtifactPipeline:
        run = self.runs.read(request.run_id)
        registrations = {item.registration_id: item for item in run.artifacts}
        stages: list[PipelineStage] = []
        for declaration in request.stages:
            if isinstance(declaration, RegisteredPipelineStageDeclaration):
                registration = registrations.get(declaration.registration_id)
                if registration is None:
                    raise DomainError(
                        ErrorCode.WORKSPACE_INVALID,
                        f"Artifact registration {declaration.registration_id!r} is not on the run.",
                    )
                stored = self.artifacts.get(registration.artifact_id)
                content = stored.content
                (
                    extractor,
                    extractor_version,
                    extraction_operation_id,
                    structural_summary,
                ) = self._extract_structure(
                    stored.payload_path,
                    artifact_id=registration.artifact_id,
                    byte_length=content.byte_length,
                    profile=declaration.extractor_profile,
                    media_type=registration.media_type,
                )
                locally_declared = identity_quality is PipelineIdentityQuality.LOCAL_DECLARED
                stage: PipelineStage = RegisteredPipelineStage(
                    **declaration.model_dump(exclude={"format", "format_schema"}),
                    format=(registration.media_type if locally_declared else declaration.format),
                    format_schema=(
                        f"flameox.artifact-kind.{registration.kind.value}.v1"
                        if locally_declared
                        else declaration.format_schema
                    ),
                    artifact_id=registration.artifact_id,
                    artifact_length=content.byte_length,
                    sensitivity=registration.sensitivity,
                    extractor=extractor,
                    extractor_version=extractor_version,
                    extraction_operation_id=extraction_operation_id,
                    structural_summary=structural_summary,
                    elapsed_ns=None,
                )
            elif isinstance(declaration, UnregisteredPipelineStageDeclaration):
                stage = UnregisteredPipelineStage(
                    **declaration.model_dump(),
                    extractor=None,
                    extractor_version=None,
                    extraction_operation_id=None,
                    structural_summary=None,
                    elapsed_ns=None,
                )
            else:
                assert_never(declaration)
            stages.append(stage)
        locally_declared = (
            identity_quality is PipelineIdentityQuality.LOCAL_DECLARED and derived_from is None
        )
        registered_producers = {
            registrations[stage.registration_id].producer or "unknown"
            for stage in stages
            if isinstance(stage, RegisteredPipelineStage)
        }
        if locally_declared and len(registered_producers) == 1:
            producer = registered_producers.pop()
        elif locally_declared:
            producer = "unknown"
        else:
            producer = request.producer
        if derived_from is not None:
            producer = derived_from.producer
        limitations = _merge_pipeline_limitations(request.limitations, additional_limitations)
        identity = {
            "run_id": request.run_id,
            "pipeline_name": request.pipeline_name,
            "producer": producer,
            "compiler_identity_id": request.compiler_identity_id,
            "target_identity_id": request.target_identity_id,
            "identity_quality": identity_quality,
            "stages": [stage.model_dump(mode="json") for stage in stages],
            "limitations": list(limitations),
        }
        pipeline = ArtifactPipeline(
            pipeline_id=digest_model(identity),
            run_id=run.run_id,
            pipeline_name=request.pipeline_name,
            producer=producer,
            identity_quality=identity_quality,
            compiler_identity_id=request.compiler_identity_id,
            target_identity_id=request.target_identity_id,
            stages=tuple(stages),
            limitations=limitations,
        )
        try:
            created = self.pipelines.create(pipeline)
        except DomainError as error:
            if error.code is not ErrorCode.REVISION_CONFLICT:
                raise
            existing = self.pipelines.read(pipeline.pipeline_id)
            if existing.model_dump(exclude={"created_at"}) != pipeline.model_dump(
                exclude={"created_at"}
            ):
                raise
            return existing
        self.publisher.publish_rows(
            {
                "artifact_pipelines": [
                    {
                        "pipeline_id": created.pipeline_id,
                        "run_id": created.run_id,
                        "pipeline_name": created.pipeline_name,
                        "producer": created.producer,
                        "identity_quality": created.identity_quality,
                        "compiler_identity_id": created.compiler_identity_id,
                        "target_identity_id": created.target_identity_id,
                        "stages_json": _serialize_control_payload(
                            [stage.model_dump(mode="json") for stage in created.stages]
                        ),
                        "limitations": list(created.limitations),
                        "created_at": created.created_at,
                    }
                ]
            },
            publisher="flameox.artifact_pipelines",
            publisher_version="1",
            input_run_ids=(created.run_id,),
            input_artifact_ids=tuple(
                stage.artifact_id for stage in created.stages if stage.artifact_id is not None
            ),
        )
        return created

    @staticmethod
    def _extract_structure(
        payload_path: Path,
        *,
        artifact_id: str,
        byte_length: int,
        profile: PipelineExtractorProfile | None,
        media_type: str,
    ) -> tuple[str | None, str | None, str | None, dict[str, JsonValue] | None]:
        if profile is None:
            return None, None, None, None
        if profile is not PipelineExtractorProfile.TEXT_LINES_V1:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Unknown pipeline structural extraction profile.",
            )
        if not (media_type.startswith("text/") or media_type == "application/json"):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The text-lines structural profile requires a textual artifact.",
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(payload_path, flags)
        except OSError as error:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Pipeline extraction could not open the immutable artifact.",
            ) from error
        digest = hashlib.sha256()
        line_count = 0
        nonempty = False
        ends_with_newline = False
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != byte_length:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Pipeline extraction requires the exact immutable regular artifact.",
                )
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                nonempty = True
                ends_with_newline = chunk.endswith(b"\n")
                line_count += chunk.count(b"\n")
                digest.update(chunk)
                decoder.decode(chunk, final=False)
            decoder.decode(b"", final=True)
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or f"sha256:{digest.hexdigest()}" != artifact_id:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Pipeline artifact changed or failed digest verification during extraction.",
                    retryable=True,
                )
        except UnicodeDecodeError as error:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The text-lines structural profile requires valid UTF-8.",
            ) from error
        finally:
            os.close(descriptor)
        if nonempty and not ends_with_newline:
            line_count += 1
        extractor = "flameox.pipeline.text-lines"
        extractor_version = "1"
        operation_id = digest_model(
            {
                "artifact_id": artifact_id,
                "profile": profile,
                "extractor": extractor,
                "extractor_version": extractor_version,
            }
        )
        summary: dict[str, JsonValue] = {"line_count": line_count, "utf8_valid": True}
        _validate_bounded_json(summary)
        return extractor, extractor_version, operation_id, summary

    def compare(self, baseline_pipeline_id: str, candidate_pipeline_id: str) -> PipelineComparison:
        baseline = self.pipelines.read(baseline_pipeline_id)
        candidate = self.pipelines.read(candidate_pipeline_id)
        compatibility, mismatches, unknown_identities = _pipeline_identity_compatibility(
            baseline,
            candidate,
            self.runs.read(baseline.run_id),
            self.runs.read(candidate.run_id),
        )
        baseline_by_name = {stage.name: stage for stage in baseline.stages}
        candidate_by_name = {stage.name: stage for stage in candidate.stages}
        ordered_names = tuple(
            sorted(
                baseline_by_name.keys() | candidate_by_name.keys(),
                key=lambda name: (
                    min(
                        stage.ordinal
                        for stage in (baseline_by_name.get(name), candidate_by_name.get(name))
                        if stage is not None
                    ),
                    name,
                ),
            )
        )
        stages: list[PipelineStageComparison] = []
        limitations = [*baseline.limitations, *candidate.limitations]
        if unknown_identities:
            limitations.append(
                "Critical pipeline identity evidence is unavailable for: "
                f"{', '.join(unknown_identities)}."
            )
        if {"compiler_identity_id", "target_identity_id"}.intersection(unknown_identities):
            limitations.append(
                "Exact compiler-pipeline comparison is unavailable; re-capture each run with "
                "the Triton compiler listener and authoritative CUDA identity available."
            )
        for name in ordered_names:
            left = baseline_by_name.get(name)
            right = candidate_by_name.get(name)
            if left is None:
                added = candidate_by_name[name]
                stages.append(
                    AddedPipelineStageComparison(
                        stage_name=name,
                        candidate_ordinal=added.ordinal,
                        candidate_artifact_id=added.artifact_id,
                    )
                )
            elif right is None:
                stages.append(
                    MissingPipelineStageComparison(
                        stage_name=name,
                        baseline_ordinal=left.ordinal,
                        baseline_artifact_id=left.artifact_id,
                    )
                )
            else:
                stages.append(self._compare_paired_stage(name, left, right, compatibility))
        first_divergence = _first_observed_divergent_stage(compatibility, tuple(stages))
        if (
            compatibility is PipelineCompatibility.COMPATIBLE
            and first_divergence is None
            and any(stage.disposition is not PipelineStageDisposition.IDENTICAL for stage in stages)
        ):
            limitations.append(
                "The first divergent stage is unknown because stage ordering or "
                "inspection coverage is incomplete."
            )
        limitations.append("The first observed divergent stage is not a root-cause attribution.")
        input_artifacts = tuple(
            dict.fromkeys(
                stage.artifact_id
                for pipeline in (baseline, candidate)
                for stage in pipeline.stages
                if stage.artifact_id is not None
            )
        )
        extractors = tuple(
            sorted(
                {
                    (f"{stage.extractor}@{stage.extractor_version}:{stage.extraction_operation_id}")
                    for pipeline in (baseline, candidate)
                    for stage in pipeline.stages
                    if stage.extractor is not None and stage.extraction_operation_id is not None
                }
            )
        )
        comparison = PipelineComparison(
            baseline_pipeline_id=baseline.pipeline_id,
            candidate_pipeline_id=candidate.pipeline_id,
            compatibility=compatibility,
            identity_mismatches=mismatches,
            stages=tuple(stages),
            input_artifact_ids=input_artifacts,
            extractor_identities=extractors,
            limitations=tuple(dict.fromkeys(limitations)),
        )
        self.publisher.publish_rows_idempotent(
            {
                "pipeline_comparisons": [
                    {
                        "comparison_id": comparison.comparison_id,
                        "baseline_pipeline_id": comparison.baseline_pipeline_id,
                        "candidate_pipeline_id": comparison.candidate_pipeline_id,
                        "compatibility": comparison.compatibility,
                        "identity_mismatches": list(comparison.identity_mismatches),
                        "stages_json": _serialize_control_payload(
                            [stage.model_dump(mode="json") for stage in comparison.stages]
                        ),
                        "first_observed_divergent_stage": (
                            comparison.first_observed_divergent_stage
                        ),
                        "input_artifact_ids": list(comparison.input_artifact_ids),
                        "extractor_identities": list(comparison.extractor_identities),
                        "limitations": list(comparison.limitations),
                    }
                ]
            },
            publisher="flameox.pipeline_comparisons",
            publisher_version="1",
            input_run_ids=(baseline.run_id, candidate.run_id),
            input_artifact_ids=comparison.input_artifact_ids,
            operation_identity={"comparison_id": comparison.comparison_id},
            supersede_matching=False,
        )
        return comparison

    @staticmethod
    def _compare_paired_stage(
        name: str,
        left: PipelineStage,
        right: PipelineStage,
        compatibility: PipelineCompatibility,
    ) -> PairedPipelineStageComparison:
        disposition: _PairedPipelineStageDisposition
        stage_limitations = (*left.limitations, *right.limitations)
        incompatible = (
            compatibility is PipelineCompatibility.INCOMPATIBLE
            or left.format != right.format
            or left.format_schema != right.format_schema
            or left.extractor != right.extractor
            or left.extractor_version != right.extractor_version
        )
        short_circuit = left.artifact_id is not None and left.artifact_id == right.artifact_id
        if left.ordinal != right.ordinal or left.predecessor != right.predecessor:
            disposition = PipelineStageDisposition.REORDERED
        elif left.status not in {
            PipelineStageStatus.AVAILABLE,
            PipelineStageStatus.CACHED,
        } or right.status not in {
            PipelineStageStatus.AVAILABLE,
            PipelineStageStatus.CACHED,
        }:
            disposition = PipelineStageDisposition.UNINSPECTABLE
        elif incompatible:
            disposition = PipelineStageDisposition.INCOMPATIBLE
        elif short_circuit:
            disposition = PipelineStageDisposition.IDENTICAL
        elif left.structural_summary is None or right.structural_summary is None:
            disposition = PipelineStageDisposition.UNINSPECTABLE
            stage_limitations = (
                *stage_limitations,
                "Structural extraction coverage is incomplete.",
            )
        else:
            disposition = (
                PipelineStageDisposition.CONTENT_CHANGED
                if left.structural_summary == right.structural_summary
                else PipelineStageDisposition.CHANGED
            )
        return PairedPipelineStageComparison(
            stage_name=name,
            baseline_ordinal=left.ordinal,
            candidate_ordinal=right.ordinal,
            disposition=disposition,
            baseline_artifact_id=left.artifact_id,
            candidate_artifact_id=right.artifact_id,
            artifact_length_change=(
                right.artifact_length - left.artifact_length
                if left.artifact_length is not None and right.artifact_length is not None
                else None
            ),
            baseline_summary=left.structural_summary,
            candidate_summary=right.structural_summary,
            limitations=stage_limitations,
        )


def _validate_bounded_json(value: JsonValue) -> None:
    count = 0

    def visit(item: JsonValue, depth: int) -> None:
        nonlocal count
        count += 1
        if count > 500:
            raise ValueError("structural summaries are limited to 500 values")
        if depth > 8:
            raise ValueError("structural summaries are limited to 8 nested levels")
        if isinstance(item, str) and len(item) > 2_000:
            raise ValueError("structural summary strings are limited to 2000 characters")
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, dict):
            for key, child in item.items():
                if len(key) > 100:
                    raise ValueError("structural summary keys are limited to 100 characters")
                visit(child, depth + 1)

    visit(value, 0)
