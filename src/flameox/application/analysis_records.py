from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, Literal, assert_never

from pydantic import Field

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

AnalysisValue = (
    AcceleratorLaunchAnalysisResult
    | HotspotResult
    | MemoryAnalysisResult
    | ExecutionAnalysisResult
    | PyTorchAnalysisResult
    | FailureAnalysisResult
    | ScalingAnalysisResult
)


class _InputAnalysisRequest(ContractModel):
    input_id: str
    limit: int | None = Field(default=None, ge=1, le=1_000)


class HotspotAnalysisRequest(_InputAnalysisRequest):
    recipe: Literal["hotspots"]


class MemoryAnalysisRequest(_InputAnalysisRequest):
    recipe: Literal["memory"]


class ExecutionAnalysisRequest(_InputAnalysisRequest):
    recipe: Literal["execution"]
    comparison_input_id: str | None = None


class PyTorchAnalysisRequest(_InputAnalysisRequest):
    recipe: Literal["pytorch"]


class AcceleratorLaunchAnalysisRequest(_InputAnalysisRequest):
    recipe: Literal["accelerator_launches"]
    comparison_input_id: str | None = None
    phase: str | None = Field(default=None, min_length=1, max_length=200)


class FailureAnalysisRequest(ContractModel):
    recipe: Literal["failures"]
    limit: int | None = Field(default=None, ge=1, le=1_000)


class ScalingAnalysisRequest(ContractModel):
    recipe: Literal["scaling"]
    experiment_id: str


type MaterializeAnalysisRequest = Annotated[
    AcceleratorLaunchAnalysisRequest
    | HotspotAnalysisRequest
    | MemoryAnalysisRequest
    | ExecutionAnalysisRequest
    | PyTorchAnalysisRequest
    | FailureAnalysisRequest
    | ScalingAnalysisRequest,
    Field(discriminator="recipe"),
]


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
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(corpus_commit_id)) as snapshot:
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

        catalog = Catalog(self.workspace)
        result, run_ids, artifact_ids, generation_ids = await catalog.run_interruptible(
            prepare,
            handle=catalog.pin(corpus_commit_id),
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
        if isinstance(request, HotspotAnalysisRequest):
            return recipes.hotspots(
                request.input_id,
                limit=request.limit,
            )
        if isinstance(request, MemoryAnalysisRequest):
            return recipes.memory(
                request.input_id,
                limit=request.limit,
            )
        if isinstance(request, ExecutionAnalysisRequest):
            return recipes.execution(
                request.input_id,
                comparison_input_id=request.comparison_input_id,
                limit=request.limit,
            )
        if isinstance(request, PyTorchAnalysisRequest):
            return recipes.pytorch(
                request.input_id,
                limit=request.limit,
            )
        if isinstance(request, AcceleratorLaunchAnalysisRequest):
            return recipes.accelerator_launches(
                request.input_id,
                comparison_input_id=request.comparison_input_id,
                phase=request.phase,
                limit=request.limit,
            )
        if isinstance(request, FailureAnalysisRequest):
            return recipes.failures(
                limit=request.limit,
            )
        if isinstance(request, ScalingAnalysisRequest):
            return recipes.scaling(
                request.experiment_id,
            )
        assert_never(request)

    def _inputs(
        self,
        request: MaterializeAnalysisRequest,
        *,
        snapshot: Snapshot,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        if isinstance(request, ScalingAnalysisRequest):
            rows = snapshot.execute(
                "SELECT DISTINCT run_id FROM trials "
                "WHERE experiment_id = ? AND run_id IS NOT NULL ORDER BY run_id",
                (request.experiment_id,),
            ).fetchall()
            return tuple(str(row[0]) for row in rows), (), ()
        if isinstance(request, FailureAnalysisRequest):
            commit = self.workspace.corpus.read_commit(snapshot.commit.commit_id)
            generations = tuple(
                GenerationManifest.model_validate_json(
                    (self.workspace.paths.root / path).read_text()
                ).generation_id
                for path in commit.generation_manifests
            )
            return (), (), generations
        if not isinstance(request, _InputAnalysisRequest):
            assert_never(request)
        comparison_input_id = (
            request.comparison_input_id
            if isinstance(request, (ExecutionAnalysisRequest, AcceleratorLaunchAnalysisRequest))
            else None
        )
        input_ids = tuple(
            value for value in (request.input_id, comparison_input_id) if value is not None
        )
        scope = resolve_evidence_scope(snapshot, input_ids)
        for artifact_id in (value for value in input_ids if value.startswith("sha256:")):
            ArtifactStore(self.workspace).get(artifact_id)
        return scope.run_ids, scope.artifact_ids, ()
