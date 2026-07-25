from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flamo.analysis import (
    ExecutionAnalysisResult,
    FailureAnalysisResult,
    HotspotResult,
    MemoryAnalysisResult,
    PyTorchAnalysisResult,
    RecipeService,
    ScalingAnalysisResult,
)
from flamo.application.analysis_rows import analysis_row
from flamo.catalog import Catalog
from flamo.domain import (
    AnalysisRecord,
    DomainError,
    EvidenceReference,
    digest_model,
    new_id,
)
from flamo.evidence import GenerationPublisher
from flamo.storage import ArtifactStore, GenerationManifest, RunStore, Workspace

AnalysisRecipe = Literal[
    "hotspots",
    "memory",
    "execution",
    "pytorch",
    "failures",
    "scaling",
]
AnalysisValue = (
    HotspotResult
    | MemoryAnalysisResult
    | ExecutionAnalysisResult
    | PyTorchAnalysisResult
    | FailureAnalysisResult
    | ScalingAnalysisResult
)


class MaterializeAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe: AnalysisRecipe
    input_id: str | None = None
    experiment_id: str | None = None
    limit: int | None = Field(default=None, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_scope(self) -> MaterializeAnalysisRequest:
        if self.recipe == "scaling":
            if self.experiment_id is None or self.input_id is not None:
                raise ValueError("scaling requires only experiment_id")
        elif self.recipe == "failures":
            if self.input_id is not None or self.experiment_id is not None:
                raise ValueError("failures uses the pinned corpus population")
        elif self.input_id is None or self.experiment_id is not None:
            raise ValueError(f"{self.recipe} requires only input_id")
        return self


class MaterializedAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    result: AnalysisValue
    analysis: AnalysisRecord
    evidence: tuple[EvidenceReference, ...]
    materialized_commit_id: str


class AnalysisMaterializationService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.recipes = RecipeService(workspace)
        self.publisher = GenerationPublisher(workspace)

    def record(
        self,
        request: MaterializeAnalysisRequest,
    ) -> MaterializedAnalysisResult:
        started = datetime.now(UTC)
        result = self._run(request)
        completed = datetime.now(UTC)
        run_ids, artifact_ids, generation_ids = self._inputs(request)
        parameters = request.model_dump(mode="json")
        coverage_value = getattr(result, "coverage", {})
        coverage = coverage_value if isinstance(coverage_value, dict) else {}
        limitations_value = getattr(result, "limitations", ())
        limitations = (
            tuple(str(value) for value in limitations_value)
            if isinstance(limitations_value, tuple)
            else ()
        )
        analysis = AnalysisRecord(
            analysis_id=new_id(),
            recipe=request.recipe,
            recipe_version="1",
            parameters=parameters,
            parameters_digest=digest_model(parameters),
            corpus_commit_id=result.corpus_commit_id,
            input_generation_ids=generation_ids,
            input_run_ids=run_ids,
            input_artifact_ids=artifact_ids,
            result_digest=digest_model(result.model_dump(mode="json")),
            coverage=coverage,
            limitations=limitations,
            started_at=started,
            completed_at=completed,
        )
        references = tuple(
            [
                EvidenceReference(
                    owner_type="analysis",
                    owner_id=analysis.analysis_id,
                    ref_type="run",
                    ref_id=run_id,
                    relation="context",
                )
                for run_id in run_ids
            ]
            + [
                EvidenceReference(
                    owner_type="analysis",
                    owner_id=analysis.analysis_id,
                    ref_type="artifact",
                    ref_id=artifact_id,
                    relation="context",
                )
                for artifact_id in artifact_ids
            ]
            + [
                EvidenceReference(
                    owner_type="analysis",
                    owner_id=analysis.analysis_id,
                    ref_type="generation",
                    ref_id=generation_id,
                    relation="context",
                )
                for generation_id in generation_ids
            ]
        )
        published = self.publisher.publish_rows(
            {
                "analyses": [analysis_row(analysis)],
                "evidence_refs": [
                    reference.model_dump(mode="python") for reference in references
                ],
            },
            publisher="flamo.analyses",
            publisher_version="1",
            input_run_ids=run_ids,
            input_artifact_ids=artifact_ids,
        )
        return MaterializedAnalysisResult(
            result=result,
            analysis=analysis,
            evidence=references,
            materialized_commit_id=published.commit.commit_id,
        )

    def _run(self, request: MaterializeAnalysisRequest) -> AnalysisValue:
        if request.recipe == "hotspots":
            assert request.input_id is not None
            return self.recipes.hotspots(request.input_id, limit=request.limit)
        if request.recipe == "memory":
            assert request.input_id is not None
            return self.recipes.memory(request.input_id, limit=request.limit)
        if request.recipe == "execution":
            assert request.input_id is not None
            return self.recipes.execution(request.input_id, limit=request.limit)
        if request.recipe == "pytorch":
            assert request.input_id is not None
            return self.recipes.pytorch(request.input_id, limit=request.limit)
        if request.recipe == "failures":
            return self.recipes.failures(limit=request.limit)
        assert request.experiment_id is not None
        return self.recipes.scaling(request.experiment_id)

    def _inputs(
        self,
        request: MaterializeAnalysisRequest,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        if request.input_id is not None:
            try:
                run = RunStore(self.workspace).read(request.input_id)
            except DomainError:
                artifact = ArtifactStore(self.workspace).get(request.input_id)
                return (), (artifact.content.artifact_id,), ()
            return (
                (run.run_id,),
                tuple(item.artifact_id for item in run.artifacts),
                (),
            )
        if request.experiment_id is not None:
            with Catalog(self.workspace).open_snapshot() as snapshot:
                rows = snapshot.execute(
                    "SELECT DISTINCT run_id FROM trials WHERE experiment_id = ? "
                    "ORDER BY run_id",
                    (request.experiment_id,),
                ).fetchall()
            return tuple(str(row[0]) for row in rows), (), ()
        head = self.workspace.corpus.read_head()
        generations = tuple(
            GenerationManifest.model_validate_json(
                (self.workspace.paths.root / path).read_text()
            ).generation_id
            for path in head.generation_manifests
        )
        return (), (), generations
