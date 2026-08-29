from __future__ import annotations

import codecs
import hashlib
import os
import stat
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, assert_never, cast

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    ModelWrapValidatorHandler,
    computed_field,
    model_validator,
)

from flameox.domain import (
    CursorNamespace,
    DomainError,
    ErrorCode,
    Sensitivity,
    WorkloadInstance,
    canonical_json,
    digest_model,
)
from flameox.domain.models import utc_now
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import ArtifactStore, ControlRecordStore, RunStore, Workspace


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
    PRODUCER_DECLARED = "producer_declared"
    LEGACY_UNQUALIFIED = "legacy_unqualified"


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
    pipeline_schema: Annotated[str, Field(min_length=1, max_length=100)]
    producer: Annotated[str, Field(min_length=1, max_length=100)]
    producer_version: Annotated[str, Field(min_length=1, max_length=100)]
    workload_identity: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    device_identity: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    workload_definition_id: str | None = None
    workload_instance_id: str | None = None
    command_digest: str | None = None
    parameters_digest: str | None = None
    compiler_identity_id: str | None = None
    build_protocol_id: str | None = None
    target_identity_id: str | None = None
    stages: Annotated[tuple[PipelineStageDeclaration, ...], Field(min_length=1, max_length=100)]
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
        exact_claims = (
            self.workload_definition_id,
            self.workload_instance_id,
            self.command_digest,
            self.parameters_digest,
            self.compiler_identity_id,
            self.build_protocol_id,
        )
        if any(item is not None for item in exact_claims) and not all(
            item is not None for item in exact_claims
        ):
            raise ValueError("exact pipeline identity claims must be supplied together")
        if self.target_identity_id is not None and not all(
            item is not None for item in exact_claims
        ):
            raise ValueError("target identity requires complete exact pipeline identity claims")
        return self


class _PipelineStage(ContractModel):
    name: str
    ordinal: int
    predecessor: str | None
    format: str
    format_schema: str
    producer: str
    producer_version: str
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
    schema_version: Literal[1, 2] = 2
    pipeline_id: str
    run_id: str
    pipeline_name: str
    pipeline_schema: str
    producer: str
    producer_version: str
    workload_identity: str | None = None
    device_identity: str | None = None
    identity_quality: PipelineIdentityQuality = PipelineIdentityQuality.LEGACY_UNQUALIFIED
    workload_definition_id: str | None = None
    workload_instance_id: str | None = None
    command_digest: str | None = None
    parameters_digest: str | None = None
    compiler_identity_id: str | None = None
    build_protocol_id: str | None = None
    target_identity_id: str | None = None
    source_state_id: str | None
    environment_id: str
    stages: tuple[PipelineStage, ...]
    limitations: Annotated[tuple[str, ...], Field(max_length=20)]
    created_at: datetime = Field(default_factory=utc_now)


class PipelineFilter(ContractModel):
    run_id: str | None = None
    pipeline_name: str | None = None
    pipeline_schema: str | None = None
    producer: str | None = None
    source_artifact_id: str | None = None


class PipelineSummary(ContractModel):
    pipeline_id: str
    run_id: str
    pipeline_name: str
    pipeline_schema: str
    producer: str
    producer_version: str
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
    extraction_short_circuited: Literal[False] = False
    baseline_summary: Literal[None] = None
    candidate_summary: Literal[None] = None


class MissingPipelineStageComparison(_PipelineStageComparison):
    baseline_ordinal: int
    candidate_ordinal: Literal[None] = None
    disposition: Literal[PipelineStageDisposition.MISSING] = PipelineStageDisposition.MISSING
    baseline_artifact_id: str | None = None
    candidate_artifact_id: Literal[None] = None
    artifact_length_change: Literal[None] = None
    extraction_short_circuited: Literal[False] = False
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


def _advertise_short_circuit_projection(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties["extraction_short_circuited"] = {"type": "boolean", "readOnly": True}
    required = schema.setdefault("required", [])
    assert isinstance(required, list)
    if "extraction_short_circuited" not in required:
        required.append("extraction_short_circuited")


class PairedPipelineStageComparison(_PipelineStageComparison):
    model_config = ConfigDict(json_schema_extra=_advertise_short_circuit_projection)

    baseline_ordinal: int
    candidate_ordinal: int
    disposition: _PairedPipelineStageDisposition
    baseline_artifact_id: str | None = None
    candidate_artifact_id: str | None = None
    artifact_length_change: int | None = None
    baseline_summary: dict[str, JsonValue] | None = None
    candidate_summary: dict[str, JsonValue] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def extraction_short_circuited(self) -> bool:
        return (
            self.baseline_artifact_id is not None
            and self.baseline_artifact_id == self.candidate_artifact_id
        )


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
        if stage.disposition is PipelineStageDisposition.CHANGED:
            return stage.stage_name
        if stage.disposition is not PipelineStageDisposition.IDENTICAL:
            return None
    return None


def _advertise_pipeline_comparison_projections(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties.update(
        {
            "comparison_id": {"type": "string", "readOnly": True},
            "first_observed_divergent_stage": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "readOnly": True,
            },
            "result_digest": {"type": "string", "readOnly": True},
        }
    )
    required = schema.setdefault("required", [])
    assert isinstance(required, list)
    for name in ("comparison_id", "first_observed_divergent_stage", "result_digest"):
        if name not in required:
            required.append(name)


class PipelineComparison(ContractModel):
    model_config = ConfigDict(json_schema_extra=_advertise_pipeline_comparison_projections)

    schema_version: Literal[1] = 1
    baseline_pipeline_id: str
    candidate_pipeline_id: str
    compatibility: PipelineCompatibility
    identity_mismatches: tuple[str, ...]
    stages: tuple[PipelineStageComparison, ...]
    input_artifact_ids: tuple[str, ...]
    extractor_identities: tuple[str, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="wrap")
    @classmethod
    def parse_projection_fields(
        cls,
        value: Any,
        handler: ModelWrapValidatorHandler[PipelineComparison],
    ) -> PipelineComparison:
        if not isinstance(value, Mapping):
            return handler(value)
        payload = dict(value)
        supplied_comparison_id = payload.pop("comparison_id", None)
        supplied_divergence = payload.pop("first_observed_divergent_stage", None)
        divergence_was_supplied = "first_observed_divergent_stage" in value
        supplied_digest = payload.pop("result_digest", None)
        comparison = handler(payload)
        if (
            divergence_was_supplied
            and supplied_divergence != comparison.first_observed_divergent_stage
        ):
            raise ValueError("first divergent stage does not match the stage comparison")
        if supplied_digest is not None and supplied_digest != comparison.result_digest:
            raise ValueError("pipeline comparison digest does not match its content")
        if (
            supplied_comparison_id is not None
            and supplied_comparison_id != comparison.comparison_id
        ):
            raise ValueError("pipeline comparison identifier does not match its content")
        return comparison

    @computed_field  # type: ignore[prop-decorator]
    @property
    def first_observed_divergent_stage(self) -> str | None:
        return _first_observed_divergent_stage(self.compatibility, self.stages)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def result_digest(self) -> str:
        return digest_model(self.content_without_identity())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def comparison_id(self) -> str:
        return self.result_digest

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
) -> tuple[PipelineCompatibility, tuple[str, ...], tuple[str, ...]]:
    mismatches = [
        name
        for name in ("pipeline_name", "pipeline_schema", "producer")
        if getattr(baseline, name) != getattr(candidate, name)
    ]
    unknown_identities: list[str] = []
    if any(pipeline.producer_version.casefold() == "unknown" for pipeline in (baseline, candidate)):
        unknown_identities.append("producer_version")
    elif baseline.producer_version != candidate.producer_version:
        mismatches.append("producer_version")
    exact_names = (
        "workload_definition_id",
        "workload_instance_id",
        "command_digest",
        "parameters_digest",
        "compiler_identity_id",
        "build_protocol_id",
        "target_identity_id",
    )
    exact_qualities = {
        PipelineIdentityQuality.MANAGED_EXACT,
        PipelineIdentityQuality.MANAGED_PARTIAL,
        PipelineIdentityQuality.PRODUCER_DECLARED,
    }
    exact_identity_expected = any(
        pipeline.pipeline_schema.startswith("flameox.kernel-build.v")
        or pipeline.identity_quality in exact_qualities
        or any(getattr(pipeline, name) is not None for name in exact_names)
        for pipeline in (baseline, candidate)
    )
    if exact_identity_expected:
        for name in exact_names:
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
        if all(pipeline.workload_definition_id is None for pipeline in (baseline, candidate)):
            for name in ("workload_identity", "device_identity"):
                baseline_value = getattr(baseline, name)
                candidate_value = getattr(candidate, name)
                if (
                    baseline_value is not None
                    and candidate_value is not None
                    and baseline_value != candidate_value
                ):
                    mismatches.append(name)
    else:
        for name in ("workload_identity", "device_identity"):
            baseline_value = getattr(baseline, name)
            candidate_value = getattr(candidate, name)
            if baseline_value is None or candidate_value is None:
                unknown_identities.append(name)
            elif baseline_value != candidate_value:
                mismatches.append(name)
    if baseline.source_state_id != candidate.source_state_id:
        mismatches.append("source_state_id")
    if baseline.environment_id != candidate.environment_id:
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
        self.comparisons = ControlRecordStore(
            workspace,
            kind="pipeline_comparisons",
            model=PipelineComparison,
            id_field="comparison_id",
            output_only_fields={
                "stages": {"__all__": {"extraction_short_circuited"}},
            },
        )
        self.publisher = GenerationPublisher(workspace)

    @staticmethod
    def _summary(pipeline: ArtifactPipeline) -> PipelineSummary:
        return PipelineSummary(
            pipeline_id=pipeline.pipeline_id,
            run_id=pipeline.run_id,
            pipeline_name=pipeline.pipeline_name,
            pipeline_schema=pipeline.pipeline_schema,
            producer=pipeline.producer,
            producer_version=pipeline.producer_version,
            identity_quality=pipeline.identity_quality,
            stage_count=len(pipeline.stages),
            artifact_ids=tuple(
                dict.fromkeys(
                    stage.artifact_id
                    for stage in pipeline.stages
                    if stage.artifact_id is not None
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
                    ("pipeline_schema", filter.pipeline_schema),
                    ("producer", filter.producer),
                )
            ) and (
                filter.source_artifact_id is None
                or any(
                    stage.artifact_id == filter.source_artifact_id for stage in pipeline.stages
                )
            )

        pipelines = sorted(
            (pipeline for pipeline in self.pipelines.list() if selected(pipeline)),
            key=lambda item: (item.created_at, item.pipeline_id),
            reverse=True,
        )
        total = len(pipelines)
        if after is not None:
            pipelines = [
                item for item in pipelines if (item.created_at, item.pipeline_id) < after
            ]
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
        compatible = tuple(
            item.pipeline_id
            for item in sorted(
                self.pipelines.list(),
                key=lambda item: (item.created_at, item.pipeline_id),
                reverse=True,
            )
            if item.pipeline_id != pipeline.pipeline_id
            and _pipeline_identity_compatibility(pipeline, item)[0]
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
        last = pipeline.stages[-1]
        declarations.append(
            RegisteredPipelineStageDeclaration(
                name=stage_name,
                ordinal=last.ordinal + 1,
                predecessor=last.name,
                status=PipelineStageStatus.AVAILABLE,
                registration_id=registration.registration_id,
                format=registration.media_type,
                format_schema=format_schema,
            )
        )
        request = RegisterPipelineRequest(
            run_id=pipeline.run_id,
            pipeline_name=pipeline.pipeline_name,
            pipeline_schema=pipeline.pipeline_schema,
            producer=pipeline.producer,
            producer_version=pipeline.producer_version,
            workload_identity=pipeline.workload_identity,
            device_identity=pipeline.device_identity,
            workload_definition_id=pipeline.workload_definition_id,
            workload_instance_id=pipeline.workload_instance_id,
            command_digest=pipeline.command_digest,
            parameters_digest=pipeline.parameters_digest,
            compiler_identity_id=pipeline.compiler_identity_id,
            build_protocol_id=pipeline.build_protocol_id,
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

        exact_claims = (
            request.workload_definition_id,
            request.workload_instance_id,
            request.command_digest,
            request.parameters_digest,
            request.compiler_identity_id,
            request.build_protocol_id,
            request.target_identity_id,
        )
        if any(item is not None for item in exact_claims):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Generic pipeline registration cannot authenticate exact identity claims.",
            )
        return self._register(request, identity_quality=PipelineIdentityQuality.LOCAL_DECLARED)

    def register_imported(self, request: RegisterPipelineRequest) -> ArtifactPipeline:
        """Register unverified identity claims parsed from a provider document."""

        limitation = (
            "Imported pipeline identity is producer-declared and was not authenticated by flameox."
        )
        return self._register(
            request,
            identity_quality=PipelineIdentityQuality.PRODUCER_DECLARED,
            additional_limitations=(limitation,),
        )

    def register_managed(
        self,
        request: RegisterPipelineRequest,
        *,
        workload_instance: WorkloadInstance,
    ) -> ArtifactPipeline:
        """Authenticate exact build claims against the authoritative managed run."""

        run = self.runs.read(request.run_id)
        expected = {
            "workload_definition_id": workload_instance.workload_definition_id,
            "workload_instance_id": workload_instance.workload_instance_id,
            "command_digest": digest_model(workload_instance.command.model_dump(mode="json")),
            "parameters_digest": digest_model(workload_instance.parameters),
        }
        run_matches = (
            run.workload_definition_id == workload_instance.workload_definition_id
            and run.workload_instance_id == workload_instance.workload_instance_id
            and run.command == workload_instance.command
        )
        claims_match = all(getattr(request, name) == value for name, value in expected.items())
        protocol_complete = (
            request.compiler_identity_id is not None and request.build_protocol_id is not None
        )
        if not run_matches or not claims_match or not protocol_complete:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Kernel-build identity does not match the authoritative managed run.",
            )
        exact = (
            request.target_identity_id is not None
            and request.producer_version.casefold() != "unknown"
        )
        additional_limitations: tuple[str, ...] = ()
        if not exact:
            additional_limitations = (
                "Managed pipeline identity is partial because compiler version or target "
                "identity is unavailable.",
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
                    producer=registration.producer or "unknown",
                    producer_version=registration.producer_version or "unknown",
                    extractor=extractor,
                    extractor_version=extractor_version,
                    extraction_operation_id=extraction_operation_id,
                    structural_summary=structural_summary,
                    elapsed_ns=None,
                )
            elif isinstance(declaration, UnregisteredPipelineStageDeclaration):
                stage = UnregisteredPipelineStage(
                    **declaration.model_dump(),
                    producer=request.producer,
                    producer_version=request.producer_version,
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
            identity_quality is PipelineIdentityQuality.LOCAL_DECLARED
            and derived_from is None
        )
        registered_provenance = {
            (stage.producer, stage.producer_version)
            for stage in stages
            if isinstance(stage, RegisteredPipelineStage)
        }
        if locally_declared and len(registered_provenance) == 1:
            producer, producer_version = registered_provenance.pop()
        elif locally_declared:
            producer, producer_version = "unknown", "unknown"
        else:
            producer, producer_version = request.producer, request.producer_version
        workload_identity = None if locally_declared else request.workload_identity
        device_identity = None if locally_declared else request.device_identity
        if derived_from is not None:
            producer = derived_from.producer
            producer_version = derived_from.producer_version
            workload_identity = derived_from.workload_identity
            device_identity = derived_from.device_identity
        limitations = _merge_pipeline_limitations(request.limitations, additional_limitations)
        identity = {
            "run_id": request.run_id,
            "pipeline_name": request.pipeline_name,
            "pipeline_schema": request.pipeline_schema,
            "producer": producer,
            "producer_version": producer_version,
            "workload_identity": workload_identity,
            "device_identity": device_identity,
            "workload_definition_id": request.workload_definition_id,
            "workload_instance_id": request.workload_instance_id,
            "command_digest": request.command_digest,
            "parameters_digest": request.parameters_digest,
            "compiler_identity_id": request.compiler_identity_id,
            "build_protocol_id": request.build_protocol_id,
            "target_identity_id": request.target_identity_id,
            "identity_quality": identity_quality,
            "source_state_id": run.source_state_id,
            "environment_id": run.environment_id,
            "stages": [stage.model_dump(mode="json") for stage in stages],
            "limitations": list(limitations),
        }
        pipeline = ArtifactPipeline(
            pipeline_id=digest_model(identity),
            run_id=run.run_id,
            pipeline_name=request.pipeline_name,
            pipeline_schema=request.pipeline_schema,
            producer=producer,
            producer_version=producer_version,
            workload_identity=workload_identity,
            device_identity=device_identity,
            identity_quality=identity_quality,
            workload_definition_id=request.workload_definition_id,
            workload_instance_id=request.workload_instance_id,
            command_digest=request.command_digest,
            parameters_digest=request.parameters_digest,
            compiler_identity_id=request.compiler_identity_id,
            build_protocol_id=request.build_protocol_id,
            target_identity_id=request.target_identity_id,
            source_state_id=run.source_state_id,
            environment_id=run.environment_id,
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
                        "pipeline_schema": created.pipeline_schema,
                        "producer": created.producer,
                        "producer_version": created.producer_version,
                        "identity_quality": created.identity_quality,
                        "workload_identity": created.workload_identity,
                        "device_identity": created.device_identity,
                        "workload_definition_id": created.workload_definition_id,
                        "workload_instance_id": created.workload_instance_id,
                        "command_digest": created.command_digest,
                        "parameters_digest": created.parameters_digest,
                        "compiler_identity_id": created.compiler_identity_id,
                        "build_protocol_id": created.build_protocol_id,
                        "target_identity_id": created.target_identity_id,
                        "source_state_id": created.source_state_id,
                        "environment_id": created.environment_id,
                        "stages_json": canonical_json(
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
        )
        baseline_by_name = {stage.name: stage for stage in baseline.stages}
        candidate_by_name = {stage.name: stage for stage in candidate.stages}
        ordered_names = tuple(
            dict.fromkeys(
                [stage.name for stage in baseline.stages]
                + [stage.name for stage in candidate.stages]
            )
        )
        stages: list[PipelineStageComparison] = []
        limitations = [*baseline.limitations, *candidate.limitations]
        if unknown_identities:
            limitations.append(
                "Critical pipeline identity evidence is unavailable for: "
                f"{', '.join(unknown_identities)}."
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
        try:
            created = self.comparisons.create(comparison)
        except DomainError as error:
            if error.code is not ErrorCode.REVISION_CONFLICT:
                raise
            existing = self.comparisons.read(comparison.comparison_id)
            if existing != comparison:
                raise
            return existing
        self.publisher.publish_rows(
            {
                "pipeline_comparisons": [
                    {
                        "comparison_id": created.comparison_id,
                        "baseline_pipeline_id": created.baseline_pipeline_id,
                        "candidate_pipeline_id": created.candidate_pipeline_id,
                        "compatibility": created.compatibility,
                        "identity_mismatches": list(created.identity_mismatches),
                        "stages_json": canonical_json(
                            [stage.model_dump(mode="json") for stage in created.stages]
                        ),
                        "first_observed_divergent_stage": (created.first_observed_divergent_stage),
                        "input_artifact_ids": list(created.input_artifact_ids),
                        "extractor_identities": list(created.extractor_identities),
                        "limitations": list(created.limitations),
                        "result_digest": created.result_digest,
                    }
                ]
            },
            publisher="flameox.pipeline_comparisons",
            publisher_version="1",
            input_run_ids=(baseline.run_id, candidate.run_id),
            input_artifact_ids=created.input_artifact_ids,
        )
        return created

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
