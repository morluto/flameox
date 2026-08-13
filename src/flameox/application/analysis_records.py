from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal, assert_never

from pydantic import Field, JsonValue

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
    AnalysisProvenance,
    AnalysisProvenanceInput,
    build_analysis_provenance,
    context_references,
)
from flameox.application.async_work import run_atomic_thread
from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.application.progress import ProgressReporter
from flameox.catalog import Catalog, Snapshot
from flameox.domain import (
    AnalysisRecord,
    DomainError,
    ErrorCode,
    EvidenceReference,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.evidence_scope import resolve_evidence_scope
from flameox.models import ContractModel
from flameox.storage import (
    ArtifactStore,
    CompletedRetentionIntent,
    GenerationManifest,
    RetentionIntent,
    RetentionIntentStore,
    Workspace,
)

AnalysisValue = (
    AcceleratorLaunchAnalysisResult
    | HotspotResult
    | MemoryAnalysisResult
    | ExecutionAnalysisResult
    | PyTorchAnalysisResult
    | FailureAnalysisResult
    | ScalingAnalysisResult
)


class _AnalysisRequest(ContractModel):
    corpus_commit_id: str | None = None


class _InputAnalysisRequest(_AnalysisRequest):
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


class FailureAnalysisRequest(_AnalysisRequest):
    recipe: Literal["failures"]
    limit: int | None = Field(default=None, ge=1, le=1_000)


class ScalingAnalysisRequest(_AnalysisRequest):
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


@dataclass(frozen=True, slots=True)
class _PreparedAnalysis:
    result: AnalysisValue
    provenance: AnalysisProvenance
    retention: RetentionIntent
    operation_identity: dict[str, JsonValue]


class AnalysisMaterializationService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)
        self.retention = RetentionIntentStore(workspace)

    def record(
        self,
        request: MaterializeAnalysisRequest,
    ) -> MaterializedAnalysisResult:
        started = datetime.now(UTC)
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(request.corpus_commit_id)) as snapshot:
            prepared = self._prepare(request, snapshot=snapshot, started=started)
        return self._publish(prepared)

    async def record_async(
        self,
        request: MaterializeAnalysisRequest,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> MaterializedAnalysisResult:
        started = datetime.now(UTC)
        reporter = ProgressReporter(progress)
        await reporter.report(0, 3, "Analysis snapshot pinned")

        catalog = Catalog(self.workspace)
        prepared = await catalog.run_interruptible(
            lambda snapshot: self._prepare(request, snapshot=snapshot, started=started),
            handle=catalog.pin(request.corpus_commit_id),
            query_name=f"materialize.{request.recipe}",
        )
        await reporter.report(1, 3, "Analysis query complete")
        await reporter.report(2, 3, "Publishing analysis provenance")
        materialized = await run_atomic_thread(lambda: self._publish(prepared))
        await reporter.report(3, 3, "Analysis publication complete")
        return materialized

    def _prepare(
        self,
        request: MaterializeAnalysisRequest,
        *,
        snapshot: Snapshot,
        started: datetime,
    ) -> _PreparedAnalysis:
        result = self._run(
            request,
            recipes=RecipeService(self.workspace, snapshot=snapshot),
        )
        run_ids, artifact_ids, generation_ids = self._inputs(
            request,
            snapshot=snapshot,
        )
        completed = datetime.now(UTC)
        parameters = request.model_dump(mode="json", exclude={"corpus_commit_id"})
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
        operation_identity: dict[str, JsonValue] = {
            "kind": "materialize_analysis",
            "analysis_id": provenance.analysis.analysis_id,
            "corpus_commit_id": provenance.analysis.corpus_commit_id,
            "result_digest": provenance.analysis.result_digest,
        }
        retention = self.retention.acquire(
            corpus_commit_id=provenance.analysis.corpus_commit_id,
            owner_kind="analysis",
            owner_id=provenance.analysis.analysis_id,
            operation_digest=digest_model(operation_identity),
        )
        return _PreparedAnalysis(
            result=result,
            provenance=provenance,
            retention=retention,
            operation_identity=operation_identity,
        )

    def _publish(self, prepared: _PreparedAnalysis) -> MaterializedAnalysisResult:
        if isinstance(prepared.retention, CompletedRetentionIntent):
            materialized_commit_id = prepared.retention.materialized_commit_id
        else:
            analysis = prepared.provenance.analysis
            published = self.publisher.publish_rows_idempotent(
                prepared.provenance.rows(),
                publisher="flameox.analyses",
                publisher_version="1",
                input_run_ids=analysis.input_run_ids,
                input_artifact_ids=analysis.input_artifact_ids,
                operation_identity=prepared.operation_identity,
                supersede_matching=False,
            )
            completed = self.retention.complete(
                prepared.retention,
                materialized_commit_id=published.commit.commit_id,
            )
            materialized_commit_id = completed.materialized_commit_id
        provenance = self._persisted_provenance(
            materialized_commit_id=materialized_commit_id,
            expected=prepared.provenance,
        )
        return MaterializedAnalysisResult(
            result=prepared.result,
            analysis=provenance.analysis,
            evidence=provenance.evidence,
            materialized_commit_id=materialized_commit_id,
        )

    def _persisted_provenance(
        self,
        *,
        materialized_commit_id: str,
        expected: AnalysisProvenance,
    ) -> AnalysisProvenance:
        analysis_id = expected.analysis.analysis_id
        with EvidenceLookupService(self.workspace).session(materialized_commit_id) as session:
            persisted = AnalysisProvenance(
                analysis=session.analysis(analysis_id),
                evidence=session.references(owner_type="analysis", owner_id=analysis_id),
            )
        expected_identity = expected.analysis.model_dump(
            mode="json",
            exclude={"started_at", "completed_at"},
        )
        persisted_identity = persisted.analysis.model_dump(
            mode="json",
            exclude={"started_at", "completed_at"},
        )
        if persisted_identity != expected_identity or persisted.evidence != expected.evidence:
            mismatched_fields = tuple(
                sorted(
                    key
                    for key in expected_identity.keys() | persisted_identity.keys()
                    if expected_identity.get(key) != persisted_identity.get(key)
                )
            )
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The materialized analysis does not match the retained operation inputs"
                + (
                    f" (analysis fields: {', '.join(mismatched_fields)})."
                    if mismatched_fields
                    else " (evidence references differ)."
                ),
                details={
                    "analysis_id": analysis_id,
                    "materialized_commit_id": materialized_commit_id,
                    "mismatched_analysis_fields": list(mismatched_fields),
                    "evidence_mismatch": persisted.evidence != expected.evidence,
                },
            )
        return persisted

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
