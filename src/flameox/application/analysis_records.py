from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from flameox.analysis import (
    AcceleratorLaunchAnalysisResult,
    ExecutionAnalysisResult,
    FailureAnalysisResult,
    HotspotResult,
    MemoryAnalysisResult,
    PyTorchAnalysisResult,
    RecipeService,
    ScalingAnalysisResult,
)
from flameox.application.analysis_provenance import (
    AnalysisProvenanceInput,
    build_analysis_provenance,
    context_references,
)
from flameox.application.async_work import run_atomic_thread
from flameox.catalog import Catalog, Snapshot
from flameox.domain import (
    AnalysisRecord,
    EvidenceReference,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.evidence_scope import resolve_evidence_scope
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, GenerationManifest, Workspace

AnalysisRecipe = Literal[
    "accelerator_launches",
    "hotspots",
    "memory",
    "execution",
    "pytorch",
    "failures",
    "scaling",
]
AnalysisValue = (
    AcceleratorLaunchAnalysisResult
    | HotspotResult
    | MemoryAnalysisResult
    | ExecutionAnalysisResult
    | PyTorchAnalysisResult
    | FailureAnalysisResult
    | ScalingAnalysisResult
)


class MaterializeAnalysisRequest(ContractModel):
    recipe: AnalysisRecipe
    input_id: str | None = None
    comparison_input_id: str | None = None
    experiment_id: str | None = None
    limit: int | None = Field(default=None, ge=1, le=1_000)
    phase: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_scope(self) -> MaterializeAnalysisRequest:
        if self.recipe == "scaling":
            if (
                self.experiment_id is None
                or self.input_id is not None
                or self.comparison_input_id is not None
            ):
                raise ValueError("scaling requires only experiment_id")
        elif self.recipe == "failures":
            if (
                self.input_id is not None
                or self.comparison_input_id is not None
                or self.experiment_id is not None
            ):
                raise ValueError("failures uses the pinned corpus population")
        elif self.input_id is None or self.experiment_id is not None:
            raise ValueError(f"{self.recipe} requires only input_id")
        elif self.comparison_input_id is not None and self.recipe not in {
            "execution",
            "accelerator_launches",
        }:
            raise ValueError(
                "comparison_input_id is supported only by execution and accelerator_launches"
            )
        if self.phase is not None and self.recipe != "accelerator_launches":
            raise ValueError("phase is supported only by accelerator_launches")
        return self


class MaterializedAnalysisResult(ContractModel):
    schema_version: int = 1
    result: AnalysisValue
    analysis: AnalysisRecord
    evidence: tuple[EvidenceReference, ...]
    materialized_commit_id: str


class AnalysisMaterializationService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)

    def record(
        self,
        request: MaterializeAnalysisRequest,
    ) -> MaterializedAnalysisResult:
        corpus_commit_id = self.workspace.corpus.read_head().commit_id
        started = datetime.now(UTC)
        with Catalog(self.workspace).open_snapshot(corpus_commit_id) as snapshot:
            result = self._run(
                request,
                recipes=RecipeService(self.workspace, snapshot=snapshot),
            )
            run_ids, artifact_ids, generation_ids = self._inputs(
                request,
                snapshot=snapshot,
            )
        completed = datetime.now(UTC)
        return self._publish(
            request,
            result=result,
            run_ids=run_ids,
            artifact_ids=artifact_ids,
            generation_ids=generation_ids,
            started=started,
            completed=completed,
        )

    async def record_async(
        self,
        request: MaterializeAnalysisRequest,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> MaterializedAnalysisResult:
        corpus_commit_id = self.workspace.corpus.read_head().commit_id
        started = datetime.now(UTC)
        if progress is not None:
            await progress(0, 3, "Analysis snapshot pinned")

        def prepare(
            snapshot: Snapshot,
        ) -> tuple[
            AnalysisValue,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ]:
            result = self._run(
                request,
                recipes=RecipeService(self.workspace, snapshot=snapshot),
            )
            run_ids, artifact_ids, generation_ids = self._inputs(
                request,
                snapshot=snapshot,
            )
            return result, run_ids, artifact_ids, generation_ids

        result, run_ids, artifact_ids, generation_ids = await Catalog(
            self.workspace
        ).run_interruptible(
            prepare,
            commit_id=corpus_commit_id,
            query_name=f"materialize.{request.recipe}",
        )
        if progress is not None:
            await progress(1, 3, "Analysis query complete")
        completed = datetime.now(UTC)
        if progress is not None:
            await progress(2, 3, "Publishing analysis provenance")
        materialized = await run_atomic_thread(
            lambda: self._publish(
                request,
                result=result,
                run_ids=run_ids,
                artifact_ids=artifact_ids,
                generation_ids=generation_ids,
                started=started,
                completed=completed,
            )
        )
        if progress is not None:
            await progress(3, 3, "Analysis publication complete")
        return materialized

    def _publish(
        self,
        request: MaterializeAnalysisRequest,
        *,
        result: AnalysisValue,
        run_ids: tuple[str, ...],
        artifact_ids: tuple[str, ...],
        generation_ids: tuple[str, ...],
        started: datetime,
        completed: datetime,
    ) -> MaterializedAnalysisResult:
        parameters = request.model_dump(mode="json")
        coverage_value = getattr(result, "coverage", {})
        coverage = coverage_value if isinstance(coverage_value, dict) else {}
        limitations_value = getattr(result, "limitations", ())
        limitations = (
            tuple(str(value) for value in limitations_value)
            if isinstance(limitations_value, tuple)
            else ()
        )
        provenance = build_analysis_provenance(
            AnalysisProvenanceInput(
                recipe=request.recipe,
                parameters=parameters,
                corpus_commit_id=result.corpus_commit_id,
                input_generation_ids=generation_ids,
                input_run_ids=run_ids,
                input_artifact_ids=artifact_ids,
                result_digest=digest_model(result.model_dump(mode="json")),
                coverage=coverage,
                limitations=limitations,
                started_at=started,
                completed_at=completed,
                references=context_references(
                    run_ids=run_ids,
                    artifact_ids=artifact_ids,
                    generation_ids=generation_ids,
                ),
            )
        )
        published = self.publisher.publish_rows(
            provenance.rows(),
            publisher="flameox.analyses",
            publisher_version="1",
            input_run_ids=run_ids,
            input_artifact_ids=artifact_ids,
        )
        return MaterializedAnalysisResult(
            result=result,
            analysis=provenance.analysis,
            evidence=provenance.evidence,
            materialized_commit_id=published.commit.commit_id,
        )

    def _run(
        self,
        request: MaterializeAnalysisRequest,
        *,
        recipes: RecipeService,
    ) -> AnalysisValue:
        if request.recipe == "hotspots":
            assert request.input_id is not None
            return recipes.hotspots(
                request.input_id,
                limit=request.limit,
            )
        if request.recipe == "memory":
            assert request.input_id is not None
            return recipes.memory(
                request.input_id,
                limit=request.limit,
            )
        if request.recipe == "execution":
            assert request.input_id is not None
            return recipes.execution(
                request.input_id,
                comparison_input_id=request.comparison_input_id,
                limit=request.limit,
            )
        if request.recipe == "pytorch":
            assert request.input_id is not None
            return recipes.pytorch(
                request.input_id,
                limit=request.limit,
            )
        if request.recipe == "accelerator_launches":
            assert request.input_id is not None
            return recipes.accelerator_launches(
                request.input_id,
                comparison_input_id=request.comparison_input_id,
                phase=request.phase,
                limit=request.limit,
            )
        if request.recipe == "failures":
            return recipes.failures(
                limit=request.limit,
            )
        assert request.experiment_id is not None
        return recipes.scaling(
            request.experiment_id,
        )

    def _inputs(
        self,
        request: MaterializeAnalysisRequest,
        *,
        snapshot: Snapshot,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        input_ids = tuple(
            value for value in (request.input_id, request.comparison_input_id) if value is not None
        )
        if input_ids:
            scope = resolve_evidence_scope(snapshot, input_ids)
            for artifact_id in (value for value in input_ids if value.startswith("sha256:")):
                ArtifactStore(self.workspace).get(artifact_id)
            return scope.run_ids, scope.artifact_ids, ()
        if request.experiment_id is not None:
            rows = snapshot.execute(
                "SELECT DISTINCT run_id FROM trials "
                "WHERE experiment_id = ? AND run_id IS NOT NULL ORDER BY run_id",
                (request.experiment_id,),
            ).fetchall()
            return tuple(str(row[0]) for row in rows), (), ()
        commit = self.workspace.corpus.read_commit(snapshot.commit.commit_id)
        generations = tuple(
            GenerationManifest.model_validate_json(
                (self.workspace.paths.root / path).read_text()
            ).generation_id
            for path in commit.generation_manifests
        )
        return (), (), generations
