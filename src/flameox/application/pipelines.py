from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from flameox.domain import DomainError, ErrorCode, canonical_json, digest_model
from flameox.domain.models import utc_now
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, JsonRecordStore, RunStore, Workspace

StageDisposition = Literal[
    "identical",
    "changed",
    "content_changed",
    "added",
    "missing",
    "reordered",
    "incompatible",
    "uninspectable",
]


class PipelineStageDeclaration(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    ordinal: Annotated[int, Field(ge=0, le=99)]
    predecessor: str | None = None
    status: Literal["available", "cached", "skipped", "unavailable", "failed"]
    registration_id: str | None = None
    format: Annotated[str, Field(min_length=1, max_length=100)]
    format_schema: Annotated[str, Field(min_length=1, max_length=100)]
    extractor: str | None = None
    extractor_version: str | None = None
    structural_summary: dict[str, JsonValue] | None = None
    elapsed_ns: Annotated[int, Field(ge=0)] | None = None
    limitations: Annotated[tuple[str, ...], Field(max_length=20)] = ()

    @model_validator(mode="after")
    def complete_stage_contract(self) -> PipelineStageDeclaration:
        if (self.status in {"available", "cached"}) != (self.registration_id is not None):
            raise ValueError("available or cached stages require exactly one artifact registration")
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


class RegisterPipelineRequest(ContractModel):
    run_id: str
    pipeline_name: Annotated[str, Field(min_length=1, max_length=100)]
    pipeline_schema: Annotated[str, Field(min_length=1, max_length=100)]
    producer: Annotated[str, Field(min_length=1, max_length=100)]
    producer_version: Annotated[str, Field(min_length=1, max_length=100)]
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


class PipelineStage(ContractModel):
    name: str
    ordinal: int
    predecessor: str | None
    status: str
    registration_id: str | None
    artifact_id: str | None
    artifact_length: int | None
    sensitivity: str | None
    format: str
    format_schema: str
    producer: str
    producer_version: str
    extractor: str | None
    extractor_version: str | None
    structural_summary: dict[str, JsonValue] | None
    elapsed_ns: int | None
    limitations: tuple[str, ...]


class ArtifactPipeline(ContractModel):
    schema_version: Literal[1] = 1
    pipeline_id: str
    run_id: str
    pipeline_name: str
    pipeline_schema: str
    producer: str
    producer_version: str
    source_state_id: str | None
    environment_id: str
    stages: tuple[PipelineStage, ...]
    limitations: tuple[str, ...]
    created_at: datetime = Field(default_factory=utc_now)


class PipelineStageComparison(ContractModel):
    stage_name: str
    baseline_ordinal: int | None
    candidate_ordinal: int | None
    disposition: StageDisposition
    baseline_artifact_id: str | None = None
    candidate_artifact_id: str | None = None
    artifact_length_change: int | None = None
    extraction_short_circuited: bool = False
    baseline_summary: dict[str, JsonValue] | None = None
    candidate_summary: dict[str, JsonValue] | None = None
    limitations: tuple[str, ...] = ()


class PipelineComparison(ContractModel):
    schema_version: Literal[1] = 1
    comparison_id: str
    baseline_pipeline_id: str
    candidate_pipeline_id: str
    compatibility: Literal["compatible", "incompatible", "unknown"]
    identity_mismatches: tuple[str, ...]
    stages: tuple[PipelineStageComparison, ...]
    first_observed_divergent_stage: str | None
    input_artifact_ids: tuple[str, ...]
    extractor_identities: tuple[str, ...]
    limitations: tuple[str, ...]
    result_digest: str


class ArtifactPipelineService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.pipelines = JsonRecordStore(
            workspace,
            kind="artifact_pipelines",
            model=ArtifactPipeline,
            id_field="pipeline_id",
        )
        self.comparisons = JsonRecordStore(
            workspace,
            kind="pipeline_comparisons",
            model=PipelineComparison,
            id_field="comparison_id",
        )
        self.publisher = GenerationPublisher(workspace)

    def register(self, request: RegisterPipelineRequest) -> ArtifactPipeline:
        run = self.runs.read(request.run_id)
        registrations = {item.registration_id: item for item in run.artifacts}
        stages: list[PipelineStage] = []
        for declaration in request.stages:
            registration = (
                registrations.get(declaration.registration_id)
                if declaration.registration_id is not None
                else None
            )
            if declaration.registration_id is not None and registration is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Artifact registration {declaration.registration_id!r} is not on the run.",
                )
            content = (
                self.artifacts.get(registration.artifact_id).content
                if registration is not None
                else None
            )
            stages.append(
                PipelineStage(
                    **declaration.model_dump(),
                    artifact_id=registration.artifact_id if registration else None,
                    artifact_length=content.byte_length if content else None,
                    sensitivity=registration.sensitivity.value if registration else None,
                    producer=request.producer,
                    producer_version=request.producer_version,
                )
            )
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
                stage.artifact_id
                for stage in created.stages
                if stage.artifact_id is not None
            ),
        )
        return created

    def compare(self, baseline_pipeline_id: str, candidate_pipeline_id: str) -> PipelineComparison:
        baseline = self.pipelines.read(baseline_pipeline_id)
        candidate = self.pipelines.read(candidate_pipeline_id)
        mismatches = tuple(
            name
            for name in ("pipeline_name", "pipeline_schema", "producer", "producer_version")
            if getattr(baseline, name) != getattr(candidate, name)
        )
        if baseline.source_state_id != candidate.source_state_id:
            mismatches = (*mismatches, "source_state_id")
        if baseline.environment_id != candidate.environment_id:
            mismatches = (*mismatches, "environment_id")
        compatibility: Literal["compatible", "incompatible", "unknown"] = (
            "incompatible" if mismatches else "compatible"
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
        for name in ordered_names:
            left = baseline_by_name.get(name)
            right = candidate_by_name.get(name)
            stages.append(self._compare_stage(name, left, right, compatibility))
        first_divergence = None
        if compatibility == "compatible":
            for stage in stages:
                if stage.disposition == "changed":
                    first_divergence = stage.stage_name
                    break
                if stage.disposition not in {"identical"}:
                    limitations.append(
                        "The first divergent stage is unknown because stage ordering or "
                        "inspection coverage is incomplete."
                    )
                    break
        limitations.append(
            "The first observed divergent stage is not a root-cause attribution."
        )
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
        payload = {
            "baseline_pipeline_id": baseline.pipeline_id,
            "candidate_pipeline_id": candidate.pipeline_id,
            "compatibility": compatibility,
            "identity_mismatches": mismatches,
            "stages": [stage.model_dump(mode="json") for stage in stages],
            "first_observed_divergent_stage": first_divergence,
            "input_artifact_ids": input_artifacts,
            "extractor_identities": extractors,
            "limitations": tuple(dict.fromkeys(limitations)),
        }
        result_digest = digest_model(payload)
        comparison = PipelineComparison(
            comparison_id=result_digest,
            baseline_pipeline_id=baseline.pipeline_id,
            candidate_pipeline_id=candidate.pipeline_id,
            compatibility=compatibility,
            identity_mismatches=mismatches,
            stages=tuple(stages),
            first_observed_divergent_stage=first_divergence,
            input_artifact_ids=input_artifacts,
            extractor_identities=extractors,
            limitations=tuple(dict.fromkeys(limitations)),
            result_digest=result_digest,
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
                        "first_observed_divergent_stage": (
                            created.first_observed_divergent_stage
                        ),
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
    def _compare_stage(
        name: str,
        left: PipelineStage | None,
        right: PipelineStage | None,
        compatibility: str,
    ) -> PipelineStageComparison:
        if left is None:
            assert right is not None
            return PipelineStageComparison(
                stage_name=name,
                baseline_ordinal=None,
                candidate_ordinal=right.ordinal,
                disposition="added",
                candidate_artifact_id=right.artifact_id,
            )
        if right is None:
            return PipelineStageComparison(
                stage_name=name,
                baseline_ordinal=left.ordinal,
                candidate_ordinal=None,
                disposition="missing",
                baseline_artifact_id=left.artifact_id,
            )
        disposition: StageDisposition
        stage_limitations = (*left.limitations, *right.limitations)
        incompatible = (
            compatibility != "compatible"
            or left.format != right.format
            or left.format_schema != right.format_schema
            or left.extractor != right.extractor
            or left.extractor_version != right.extractor_version
        )
        short_circuit = left.artifact_id is not None and left.artifact_id == right.artifact_id
        if left.ordinal != right.ordinal or left.predecessor != right.predecessor:
            disposition = "reordered"
        elif left.status not in {"available", "cached"} or right.status not in {
            "available",
            "cached",
        }:
            disposition = "uninspectable"
        elif incompatible:
            disposition = "incompatible"
        elif short_circuit:
            disposition = "identical"
        elif left.structural_summary is None or right.structural_summary is None:
            disposition = "uninspectable"
            stage_limitations = (
                *stage_limitations,
                "Structural extraction coverage is incomplete.",
            )
        else:
            disposition = (
                "content_changed"
                if left.structural_summary == right.structural_summary
                else "changed"
            )
        return PipelineStageComparison(
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
            extraction_short_circuited=short_circuit,
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
