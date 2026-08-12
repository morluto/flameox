from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, assert_never

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    ModelWrapValidatorHandler,
    computed_field,
    model_validator,
)

from flameox.domain import DomainError, ErrorCode, Sensitivity, canonical_json, digest_model
from flameox.domain.models import utc_now
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
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


class _PipelineStageDeclaration(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    ordinal: Annotated[int, Field(ge=0, le=99)]
    predecessor: str | None = None
    format: Annotated[str, Field(min_length=1, max_length=100)]
    format_schema: Annotated[str, Field(min_length=1, max_length=100)]
    extractor: str | None = None
    extractor_version: str | None = None
    structural_summary: dict[str, JsonValue] | None = None
    elapsed_ns: Annotated[int, Field(ge=0)] | None = None
    limitations: Annotated[tuple[str, ...], Field(max_length=20)] = ()

    @model_validator(mode="after")
    def complete_structural_summary(self) -> _PipelineStageDeclaration:
        supplied = (
            self.extractor is not None,
            self.extractor_version is not None,
            self.structural_summary is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("structural summaries require extractor name and version")
        if self.structural_summary is not None:
            _validate_bounded_json(self.structural_summary)
        return self


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
        return self


class _PipelineStage(ContractModel):
    name: str
    ordinal: int
    predecessor: str | None
    format: str
    format_schema: str
    producer: str
    producer_version: str
    extractor: str | None
    extractor_version: str | None
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
    schema_version: Literal[1] = 1
    pipeline_id: str
    run_id: str
    pipeline_name: str
    pipeline_schema: str
    producer: str
    producer_version: str
    workload_identity: str | None = None
    device_identity: str | None = None
    source_state_id: str | None
    environment_id: str
    stages: tuple[PipelineStage, ...]
    limitations: tuple[str, ...]
    created_at: datetime = Field(default_factory=utc_now)


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

    def register(self, request: RegisterPipelineRequest) -> ArtifactPipeline:
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
                content = self.artifacts.get(registration.artifact_id).content
                stage: PipelineStage = RegisteredPipelineStage(
                    **declaration.model_dump(),
                    artifact_id=registration.artifact_id,
                    artifact_length=content.byte_length,
                    sensitivity=registration.sensitivity,
                    producer=request.producer,
                    producer_version=request.producer_version,
                )
            elif isinstance(declaration, UnregisteredPipelineStageDeclaration):
                stage = UnregisteredPipelineStage(
                    **declaration.model_dump(),
                    producer=request.producer,
                    producer_version=request.producer_version,
                )
            else:
                assert_never(declaration)
            stages.append(stage)
        identity = {
            **request.model_dump(mode="json"),
            "source_state_id": run.source_state_id,
            "environment_id": run.environment_id,
            "stages": [stage.model_dump(mode="json") for stage in stages],
        }
        pipeline = ArtifactPipeline(
            pipeline_id=digest_model(identity),
            run_id=run.run_id,
            pipeline_name=request.pipeline_name,
            pipeline_schema=request.pipeline_schema,
            producer=request.producer,
            producer_version=request.producer_version,
            workload_identity=request.workload_identity,
            device_identity=request.device_identity,
            source_state_id=run.source_state_id,
            environment_id=run.environment_id,
            stages=tuple(stages),
            limitations=request.limitations,
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
                    f"{stage.extractor}@{stage.extractor_version}"
                    for pipeline in (baseline, candidate)
                    for stage in pipeline.stages
                    if stage.extractor is not None
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
