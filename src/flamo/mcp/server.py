from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flamo.adapters import (
    CoverageExtractionResult,
    CoverageExtractor,
    MemrayExtractionResult,
    MemrayExtractor,
    ObservationExtractionResult,
    ObservationExtractor,
    PerfettoExtractionResult,
    PerfettoExtractor,
    PyPerfExtractionResult,
    PyPerfExtractor,
    TraceWindowResult,
)
from flamo.analysis import (
    ExecutionAnalysisResult,
    FailureAnalysisResult,
    HotspotResult,
    MemoryAnalysisResult,
    PyTorchAnalysisResult,
    RecipeService,
    ScalingAnalysisResult,
)
from flamo.application import (
    AnalysisMaterializationService,
    ArtifactListResult,
    ArtifactMetadataResult,
    ArtifactService,
    CallEdgeResult,
    CapabilityList,
    CapabilityService,
    CapturePlanRegistry,
    CaptureResult,
    CaptureService,
    CompareRunSetsRequest,
    ComparisonResult,
    ComparisonService,
    CreateInvestigationRequest,
    DrilldownService,
    EvidenceLookupResult,
    EvidenceLookupService,
    EvidenceQueryService,
    ExecutionPolicy,
    ExperimentPlan,
    ExperimentPlanRegistry,
    ExperimentRunResult,
    ExperimentService,
    FindingResult,
    FindingService,
    FreezeRunSetRequest,
    ImportArtifactRequest,
    ImportResult,
    ImportService,
    IntegrityResult,
    IntegrityService,
    InvestigationService,
    MaterializeAnalysisRequest,
    MaterializedAnalysisResult,
    MeasurementQueryResult,
    NativeViewerPlan,
    NativeViewerService,
    RecordFindingRequest,
    RecordHypothesisRequest,
    RunSetService,
    Scalar,
    StackExamplesResult,
    WorkspaceStatus,
    workspace_status,
)
from flamo.catalog import Catalog
from flamo.domain import (
    ArtifactKind,
    CapturePlan,
    DomainError,
    ErrorCode,
    Experiment,
    Finding,
    Hypothesis,
    Investigation,
    RunManifest,
    RunSet,
    Sensitivity,
)
from flamo.storage import RunStore, Workspace

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
ADDITIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
INITIALIZE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
EXECUTE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    retryable: bool
    details: dict[str, Any]
    remediation: list[str]
    run_id: str | None


class ToolPayload[T: BaseModel](BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    ok: bool
    result: T | None = None
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> ToolPayload[T]:
        if self.ok != (self.result is not None and self.error is None):
            raise ValueError("success requires a result and failure requires an error")
        if not self.ok and self.error is None:
            raise ValueError("failure requires an error")
        return self

    @classmethod
    def success(cls, result: T) -> ToolPayload[T]:
        return cls(ok=True, result=result)

    @classmethod
    def failure(cls, error: DomainError) -> ToolPayload[T]:
        return cls(ok=False, error=ErrorDetail.model_validate(error.to_detail()))


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    created_at: datetime
    run_type: str
    capture_status: str


class RunListResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
    runs: tuple[RunSummary, ...]
    total: int
    returned: int
    truncated: bool


class InvestigationListResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    investigations: tuple[Investigation, ...]
    total: int
    returned: int
    truncated: bool


class FindingListResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    findings: tuple[Finding, ...]
    total: int
    returned: int
    truncated: bool


@dataclass(slots=True)
class AppContext:
    project_root: Path
    workspace: Workspace | None
    capture_plans: CapturePlanRegistry
    experiment_plans: ExperimentPlanRegistry

    def require_workspace(self) -> Workspace:
        if self.workspace is None:
            raise DomainError(
                code=ErrorCode.WORKSPACE_NOT_FOUND,
                message="No .diagnostics workspace was found.",
                remediation=("Call initialize_workspace first.",),
            )
        return self.workspace

    def capture_service(self) -> CaptureService:
        return CaptureService(
            self.require_workspace(),
            plans=self.capture_plans,
        )

    def experiment_service(self) -> ExperimentService:
        return ExperimentService(
            self.require_workspace(),
            captures=self.capture_service(),
            plans=self.experiment_plans,
        )


def _success[T: BaseModel](result: T, summary: str) -> CallToolResult:
    payload = ToolPayload[T].success(result)
    return CallToolResult(
        content=[TextContent(type="text", text=summary)],
        structured_content=payload.model_dump(mode="json"),
    )


def _failure(error: DomainError) -> CallToolResult:
    payload = ToolPayload[BaseModel].failure(error)
    return CallToolResult(
        content=[TextContent(type="text", text=error.message)],
        structured_content=payload.model_dump(mode="json"),
        is_error=True,
    )


def _safe_project_path(project_root: Path, value: str) -> Path:
    candidate = PurePath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "\x00" in value:
        raise DomainError(
            code=ErrorCode.EXECUTION_REFUSED,
            message="MCP artifact paths must be relative to the fixed project root.",
        )
    return project_root / Path(candidate)


def create_server(
    project_root: Path,
    *,
    initialize: bool = False,
) -> MCPServer[AppContext]:
    project_root = project_root.resolve()
    lifespan_state: list[AppContext] = []

    @asynccontextmanager
    async def lifespan(_: MCPServer[AppContext]) -> AsyncIterator[AppContext]:
        workspace: Workspace | None
        if initialize:
            workspace = Workspace.initialize(project_root)
        else:
            try:
                workspace = Workspace.discover(
                    project_root,
                    explicit=project_root / ".diagnostics",
                )
            except DomainError:
                workspace = None
        state = AppContext(
            project_root=project_root,
            workspace=workspace,
            capture_plans=CapturePlanRegistry(
                max_parallel_captures=(
                    workspace.config.capture.max_parallel_captures if workspace is not None else 2
                )
            ),
            experiment_plans=ExperimentPlanRegistry(),
        )
        lifespan_state.append(state)
        try:
            yield state
        finally:
            lifespan_state.clear()

    server: MCPServer[AppContext] = MCPServer(
        "flamo",
        description="Query and collect local runtime evidence.",
        lifespan=lifespan,
    )

    @server.tool(annotations=INITIALIZE)
    async def initialize_workspace(
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[WorkspaceStatus]]:
        """Initialize Flamo in the server's fixed project root."""
        try:
            state = ctx.request_context.lifespan_context
            state.workspace = Workspace.initialize(state.project_root)
            state.capture_plans = CapturePlanRegistry(
                max_parallel_captures=(state.workspace.config.capture.max_parallel_captures)
            )
            state.experiment_plans = ExperimentPlanRegistry()
            result = workspace_status(state.workspace)
            return _success(result, f"Initialized workspace {result.workspace_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="workspace_status", annotations=READ_ONLY)
    async def workspace_status_tool(
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[WorkspaceStatus]]:
        """Return the current workspace and corpus status."""
        try:
            result = workspace_status(ctx.request_context.lifespan_context.require_workspace())
            return _success(result, f"Workspace is at {result.corpus_commit_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_capabilities", annotations=READ_ONLY)
    async def list_capabilities_tool(
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[CapabilityList]]:
        """Passively report installed collectors and analysis libraries."""
        result = CapabilityService(ctx.request_context.lifespan_context.workspace).list()
        return _success(
            result,
            f"Found {sum(item.status.value == 'available' for item in result.capabilities)} "
            "available capabilities.",
        )

    @server.tool(name="plan_capture", annotations=READ_ONLY)
    async def plan_capture_tool(
        workload_name: str,
        adapter: str,
        parameters: dict[str, Scalar],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[CapturePlan]]:
        """Plan an approved named workload without executing it."""
        try:
            result = await ctx.request_context.lifespan_context.capture_service().plan(
                workload_name=workload_name,
                adapter=adapter,
                parameters=parameters,
                execution_policy=ExecutionPolicy.APPROVED_AGENT,
            )
            return _success(
                result,
                f"Planned {adapter} capture with {result.containment} containment.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="execute_capture_plan", annotations=EXECUTE)
    async def execute_capture_plan_tool(
        plan_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[CaptureResult]]:
        """Consume one bound plan token and execute the approved capture."""
        try:
            await ctx.report_progress(0, 1, "Validating capture plan")
            result = await ctx.request_context.lifespan_context.capture_service().execute(plan_id)
            await ctx.report_progress(
                1,
                1,
                "Capture and evidence publication complete",
            )
            return _success(
                result,
                f"Capture run {result.run.run_id} is {result.run.execution_status.value}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="plan_experiment", annotations=READ_ONLY)
    async def plan_experiment_tool(
        experiment_name: str,
        investigation_id: str,
        adapter: str,
        parameters: dict[str, Scalar],
        ctx: Context[AppContext],
        hypothesis_id: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[ExperimentPlan]]:
        """Plan a predeclared randomized experiment without executing it."""
        try:
            result = await ctx.request_context.lifespan_context.experiment_service().plan(
                experiment_name=experiment_name,
                investigation_id=investigation_id,
                hypothesis_id=hypothesis_id,
                adapter=adapter,
                parameter_overrides=parameters,
                execution_policy=ExecutionPolicy.APPROVED_AGENT,
            )
            return _success(
                result,
                f"Planned {len(result.blocks)} randomized blocks "
                f"for {len(result.variants)} variants.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="run_experiment", annotations=EXECUTE)
    async def run_experiment_tool(
        plan_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ExperimentRunResult]]:
        """Consume one experiment plan and execute all declared trials."""
        try:
            await ctx.report_progress(0, 1, "Validating experiment plan")
            result = await ctx.request_context.lifespan_context.experiment_service().run(
                plan_id,
            )
            await ctx.report_progress(1, 1, "Experiment publication complete")
            return _success(
                result,
                f"Experiment {result.experiment.experiment_id} recorded "
                f"{len(result.trials)} attempted trials.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_experiment", annotations=READ_ONLY)
    async def get_experiment_tool(
        experiment_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[Experiment]]:
        """Return one immutable experiment protocol."""
        try:
            result = ExperimentService(
                ctx.request_context.lifespan_context.require_workspace()
            ).experiments.read(experiment_id)
            return _success(result, f"Loaded experiment {experiment_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="import_artifact", annotations=ADDITIVE)
    async def import_artifact_tool(
        path: str,
        kind: ArtifactKind,
        sensitivity: Sensitivity,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ImportResult]]:
        """Import a project-local artifact as a new immutable import run."""
        try:
            state = ctx.request_context.lifespan_context
            workspace = state.require_workspace()
            result = ImportService(workspace).import_artifact(
                ImportArtifactRequest(
                    path=_safe_project_path(state.project_root, path),
                    kind=kind,
                    sensitivity=sensitivity,
                    allow_external_path=False,
                )
            )
            return _success(
                result,
                f"Imported {result.artifact_id} in run {result.run.run_id}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def list_runs(
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[RunListResult]]:
        """List bounded run summaries from one pinned corpus snapshot."""
        try:
            catalog = Catalog(ctx.request_context.lifespan_context.require_workspace())
            with catalog.open_snapshot() as snapshot:
                count_row = snapshot.execute("SELECT count(*) FROM runs").fetchone()
                assert count_row is not None
                total = int(count_row[0])
                rows = snapshot.execute(
                    "SELECT run_id, created_at, run_type, capture_status "
                    "FROM runs ORDER BY created_at DESC, run_id LIMIT ?",
                    (limit,),
                ).fetchall()
                result = RunListResult(
                    corpus_commit_id=snapshot.commit.commit_id,
                    runs=tuple(
                        RunSummary(
                            run_id=row[0],
                            created_at=row[1],
                            run_type=row[2],
                            capture_status=row[3],
                        )
                        for row in rows
                    ),
                    total=total,
                    returned=len(rows),
                    truncated=total > len(rows),
                )
            return _success(result, f"Returned {result.returned} runs.")
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def get_run(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[RunManifest]]:
        """Return one run's current immutable projection."""
        try:
            result = RunStore(ctx.request_context.lifespan_context.require_workspace()).read(run_id)
            return _success(result, f"Run {run_id} is {result.capture_status.value}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_artifact", annotations=READ_ONLY)
    async def get_artifact_tool(
        artifact_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ArtifactMetadataResult]]:
        """Return bounded artifact metadata, never binary content."""
        try:
            result = ArtifactService(ctx.request_context.lifespan_context.require_workspace()).get(
                artifact_id
            )
            return _success(result, f"Artifact {artifact_id} is metadata-only.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_artifacts", annotations=READ_ONLY)
    async def list_artifacts_tool(
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ArtifactListResult]]:
        """List bounded artifact metadata from one pinned corpus snapshot."""
        try:
            result = ArtifactService(ctx.request_context.lifespan_context.require_workspace()).list(
                limit=limit
            )
            return _success(
                result,
                f"Returned {result.returned} of {result.total} artifacts.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="create_investigation", annotations=ADDITIVE)
    async def create_investigation_tool(
        request: CreateInvestigationRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[Investigation]]:
        """Create a durable diagnostic question."""
        try:
            result = InvestigationService(
                ctx.request_context.lifespan_context.require_workspace()
            ).create(request)
            return _success(result, f"Created investigation {result.investigation_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_investigations", annotations=READ_ONLY)
    async def list_investigations_tool(
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[InvestigationListResult]]:
        """List bounded current investigation projections."""
        try:
            values = InvestigationService(
                ctx.request_context.lifespan_context.require_workspace()
            ).investigations.list()
            selected = values[:limit]
            result = InvestigationListResult(
                investigations=selected,
                total=len(values),
                returned=len(selected),
                truncated=len(values) > len(selected),
            )
            return _success(
                result,
                f"Returned {result.returned} of {result.total} investigations.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_investigation", annotations=READ_ONLY)
    async def get_investigation_tool(
        investigation_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[Investigation]]:
        """Return one current investigation projection."""
        try:
            result = InvestigationService(
                ctx.request_context.lifespan_context.require_workspace()
            ).investigations.read(investigation_id)
            return _success(
                result,
                f"Investigation {investigation_id} is {result.status.value}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="record_hypothesis", annotations=ADDITIVE)
    async def record_hypothesis_tool(
        request: RecordHypothesisRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[Hypothesis]]:
        """Record or revise a falsifiable hypothesis."""
        try:
            result = InvestigationService(
                ctx.request_context.lifespan_context.require_workspace()
            ).record_hypothesis(request)
            return _success(
                result,
                f"Hypothesis {result.hypothesis_id} is revision {result.revision}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_hypothesis", annotations=READ_ONLY)
    async def get_hypothesis_tool(
        hypothesis_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[Hypothesis]]:
        """Return the current hypothesis revision."""
        try:
            result = InvestigationService(
                ctx.request_context.lifespan_context.require_workspace()
            ).hypotheses.read(hypothesis_id)
            return _success(
                result,
                f"Hypothesis {hypothesis_id} is revision {result.revision}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="record_finding", annotations=ADDITIVE)
    async def record_finding_tool(
        request: RecordFindingRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[FindingResult]]:
        """Record or revise an evidence-linked finding."""
        try:
            result = FindingService(
                ctx.request_context.lifespan_context.require_workspace()
            ).record(request)
            return _success(
                result,
                f"Finding {result.finding.finding_id} is {result.finding.assessment.value}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_finding", annotations=READ_ONLY)
    async def get_finding_tool(
        finding_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[Finding]]:
        """Return the current finding revision."""
        try:
            result = FindingService(
                ctx.request_context.lifespan_context.require_workspace()
            ).findings.read(finding_id)
            return _success(result, f"Finding {finding_id} is revision {result.revision}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_findings", annotations=READ_ONLY)
    async def list_findings_tool(
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[FindingListResult]]:
        """List bounded current finding projections."""
        try:
            values = FindingService(
                ctx.request_context.lifespan_context.require_workspace()
            ).findings.list()
            selected = values[:limit]
            result = FindingListResult(
                findings=selected,
                total=len(values),
                returned=len(selected),
                truncated=len(values) > len(selected),
            )
            return _success(
                result,
                f"Returned {result.returned} of {result.total} findings.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="freeze_run_set", annotations=ADDITIVE)
    async def freeze_run_set_tool(
        request: FreezeRunSetRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[RunSet]]:
        """Freeze a bounded cohort against one corpus snapshot."""
        try:
            result = RunSetService(ctx.request_context.lifespan_context.require_workspace()).freeze(
                request
            )
            return _success(
                result,
                f"Frozen {len(result.members)} runs as {result.run_set_id}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="compare_run_sets", annotations=READ_ONLY)
    async def compare_run_sets_tool(
        request: CompareRunSetsRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ComparisonResult]]:
        """Compare frozen cohorts with the declared paired estimand."""
        try:
            result = ComparisonService(
                ctx.request_context.lifespan_context.require_workspace()
            ).compare(request)
            return _success(
                result,
                f"Comparison is {result.comparison.decision.value} "
                f"({result.comparison.validity.value}).",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="record_comparison", annotations=ADDITIVE)
    async def record_comparison_tool(
        request: CompareRunSetsRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ComparisonResult]]:
        """Persist one comparison and its typed immutable evidence references."""
        try:
            result = ComparisonService(
                ctx.request_context.lifespan_context.require_workspace()
            ).record(request)
            return _success(
                result,
                f"Recorded comparison {result.comparison.comparison_id}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="record_analysis", annotations=ADDITIVE)
    async def record_analysis_tool(
        request: MaterializeAnalysisRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[MaterializedAnalysisResult]]:
        """Run and persist one curated analysis with typed provenance."""
        try:
            result = AnalysisMaterializationService(
                ctx.request_context.lifespan_context.require_workspace()
            ).record(request)
            return _success(
                result,
                f"Recorded {result.analysis.recipe} analysis "
                f"{result.analysis.analysis_id}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_hotspots", annotations=READ_ONLY)
    async def analyze_hotspots_tool(
        input_id: str,
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[HotspotResult]]:
        """Return bounded source-linked profile frame aggregates."""
        try:
            result = RecipeService(
                ctx.request_context.lifespan_context.require_workspace()
            ).hotspots(input_id, limit=limit)
            return _success(
                result,
                f"Returned {result.returned} of {result.total} hotspots.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_memory", annotations=READ_ONLY)
    async def analyze_memory_tool(
        input_id: str,
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[MemoryAnalysisResult]]:
        """Return explicit peak, retained-end, and allocation evidence."""
        try:
            result = RecipeService(ctx.request_context.lifespan_context.require_workspace()).memory(
                input_id, limit=limit
            )
            return _success(
                result,
                f"Returned {len(result.measurements)} memory measurements.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_execution", annotations=READ_ONLY)
    async def analyze_execution_tool(
        input_id: str,
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ExecutionAnalysisResult]]:
        """Return bounded coverage and semantic observations."""
        try:
            result = RecipeService(
                ctx.request_context.lifespan_context.require_workspace()
            ).execution(input_id, limit=limit)
            return _success(
                result,
                f"Returned {result.returned} of {result.total} observations.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_pytorch", annotations=READ_ONLY)
    async def analyze_pytorch_tool(
        input_id: str,
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[PyTorchAnalysisResult]]:
        """Summarize operators in an existing torch.profiler trace."""
        try:
            result = RecipeService(
                ctx.request_context.lifespan_context.require_workspace()
            ).pytorch(input_id, limit=limit)
            return _success(result, f"Returned {result.returned} operators.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_scaling", annotations=READ_ONLY)
    async def analyze_scaling_tool(
        experiment_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ScalingAnalysisResult]]:
        """Summarize an existing experiment without collecting missing trials."""
        try:
            result = RecipeService(
                ctx.request_context.lifespan_context.require_workspace()
            ).scaling(experiment_id)
            return _success(
                result,
                f"Found {result.complete_blocks} complete experiment blocks.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_failures", annotations=READ_ONLY)
    async def analyze_failures_tool(
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[FailureAnalysisResult]]:
        """Cluster terminal failures across the local run population."""
        try:
            result = RecipeService(
                ctx.request_context.lifespan_context.require_workspace()
            ).failures(limit=limit)
            return _success(result, f"Returned {result.returned} failure clusters.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_frame_callers", annotations=READ_ONLY)
    async def get_frame_callers_tool(
        input_id: str,
        frame_id: str,
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[CallEdgeResult]]:
        """Return bounded source-linked direct callers for a frame."""
        try:
            result = DrilldownService(
                ctx.request_context.lifespan_context.require_workspace()
            ).callers(input_id, frame_id, limit=limit, cursor=cursor)
            return _success(result, f"Returned {result.returned} callers.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_frame_callees", annotations=READ_ONLY)
    async def get_frame_callees_tool(
        input_id: str,
        frame_id: str,
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[CallEdgeResult]]:
        """Return bounded source-linked direct callees for a frame."""
        try:
            result = DrilldownService(
                ctx.request_context.lifespan_context.require_workspace()
            ).callees(input_id, frame_id, limit=limit, cursor=cursor)
            return _success(result, f"Returned {result.returned} callees.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_stack_examples", annotations=READ_ONLY)
    async def get_stack_examples_tool(
        input_id: str,
        frame_id: str,
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[StackExamplesResult]]:
        """Return bounded representative stacks containing a frame."""
        try:
            result = DrilldownService(
                ctx.request_context.lifespan_context.require_workspace()
            ).examples(input_id, frame_id, limit=limit, cursor=cursor)
            return _success(result, f"Returned {result.returned} stack examples.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_trace_window", annotations=READ_ONLY)
    async def get_trace_window_tool(
        artifact_id: str,
        start_ns: Annotated[int, Field(ge=0)],
        end_ns: Annotated[int, Field(gt=0)],
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[TraceWindowResult]]:
        """Return bounded trace slices overlapping a declared time window."""
        try:
            result = await PerfettoExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).trace_window(
                artifact_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
                cursor=cursor,
            )
            return _success(result, f"Returned {result.returned} trace events.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_native_viewer_plan", annotations=READ_ONLY)
    async def get_native_viewer_plan_tool(
        artifact_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[NativeViewerPlan]]:
        """Return, but never execute, the installed native viewer command."""
        try:
            result = NativeViewerService(
                ctx.request_context.lifespan_context.require_workspace()
            ).plan(artifact_id)
            return _success(result, f"Use {result.viewer} for this artifact.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="query_measurements", annotations=READ_ONLY)
    async def query_measurements_tool(
        ctx: Context[AppContext],
        run_id: str | None = None,
        artifact_id: str | None = None,
        name_prefix: str | None = None,
        include_warmups: bool = False,
        limit: Annotated[int, Field(ge=1, le=1_000)] = 100,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[MeasurementQueryResult]]:
        """Query normalized measurements through reviewed filters and cursors."""
        try:
            result = EvidenceQueryService(
                ctx.request_context.lifespan_context.require_workspace()
            ).measurements(
                run_id=run_id,
                artifact_id=artifact_id,
                name_prefix=name_prefix,
                include_warmups=include_warmups,
                limit=limit,
                cursor=cursor,
            )
            return _success(
                result,
                f"Returned {result.returned} of {result.total} measurements.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_evidence", annotations=READ_ONLY)
    async def get_evidence_tool(
        ref_type: Literal[
            "analysis",
            "artifact",
            "comparison",
            "generation",
            "observation",
            "run",
            "run_set",
            "trial",
        ],
        ref_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[EvidenceLookupResult]]:
        """Retrieve one typed immutable evidence reference."""
        try:
            result = EvidenceLookupService(
                ctx.request_context.lifespan_context.require_workspace()
            ).get(ref_type, ref_id)
            return _success(result, f"Retrieved {ref_type} evidence {ref_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="validate_workspace", annotations=READ_ONLY)
    async def validate_workspace_tool(
        full: bool,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[IntegrityResult]]:
        """Validate manifests and schemas; optionally hash every payload."""
        try:
            result = IntegrityService(
                ctx.request_context.lifespan_context.require_workspace()
            ).validate(full=full)
            return _success(
                result,
                f"Workspace validation {'passed' if result.valid else 'failed'}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=ADDITIVE)
    async def extract_pyperf(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[PyPerfExtractionResult]]:
        """Extract public pyperf run, warmup, loop, and value evidence."""
        try:
            result = PyPerfExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id)
            return _success(
                result,
                f"Extracted {result.measurement_count} measured values.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_coverage", annotations=ADDITIVE)
    async def extract_coverage_tool(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[CoverageExtractionResult]]:
        """Extract bounded execution-path evidence through coverage.py's public API."""
        try:
            result = CoverageExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id)
            return _success(
                result,
                f"Extracted {result.line_count} lines and {result.arc_count} arcs.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_memray", annotations=ADDITIVE)
    async def extract_memray_tool(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[MemrayExtractionResult]]:
        """Extract supported memory concepts through Memray's public FileReader."""
        try:
            result = MemrayExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id)
            return _success(
                result,
                f"Peak memory was {result.peak_memory_bytes} bytes; "
                f"{result.retained_end_bytes} bytes remained at end.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_perfetto", annotations=ADDITIVE)
    async def extract_perfetto_tool(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[PerfettoExtractionResult]]:
        """Run versioned curated queries through a configured local Trace Processor."""
        try:
            result = await PerfettoExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id)
            return _success(
                result,
                f"Extracted {result.slice_count} slices into "
                f"{result.frame_count} frame aggregates.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_observations", annotations=ADDITIVE)
    async def extract_observations_tool(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ObservationExtractionResult]]:
        """Extract bounded semantic observations emitted through flamo.sdk."""
        try:
            result = ObservationExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id)
            return _success(
                result,
                f"Extracted {result.observation_count} semantic observations.",
            )
        except DomainError as error:
            return _failure(error)

    @server.resource(
        "flamo://runs/{run_id}",
        mime_type="application/json",
        description="Bounded run manifest projection.",
    )
    async def run_resource(run_id: str) -> str:
        try:
            if not lifespan_state:
                raise DomainError(
                    ErrorCode.WORKSPACE_NOT_FOUND,
                    "The MCP server lifespan is not active.",
                )
            run = RunStore(lifespan_state[0].require_workspace()).read(run_id)
            return run.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flamo://artifacts/{artifact_id}",
        mime_type="application/json",
        description="Artifact metadata without binary content.",
    )
    async def artifact_resource(artifact_id: str) -> str:
        try:
            state = _active_state(lifespan_state)
            value = ArtifactService(state.require_workspace()).get(artifact_id)
            return value.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flamo://investigations/{investigation_id}",
        mime_type="application/json",
        description="Current investigation projection.",
    )
    async def investigation_resource(investigation_id: str) -> str:
        try:
            state = _active_state(lifespan_state)
            value = InvestigationService(state.require_workspace()).investigations.read(
                investigation_id
            )
            return value.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flamo://hypotheses/{hypothesis_id}",
        mime_type="application/json",
        description="Current hypothesis revision.",
    )
    async def hypothesis_resource(hypothesis_id: str) -> str:
        try:
            state = _active_state(lifespan_state)
            value = InvestigationService(state.require_workspace()).hypotheses.read(hypothesis_id)
            return value.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flamo://findings/{finding_id}",
        mime_type="application/json",
        description="Current finding revision.",
    )
    async def finding_resource(finding_id: str) -> str:
        try:
            state = _active_state(lifespan_state)
            value = FindingService(state.require_workspace()).findings.read(finding_id)
            return value.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flamo://experiments/{experiment_id}",
        mime_type="application/json",
        description="Immutable experiment protocol.",
    )
    async def experiment_resource(experiment_id: str) -> str:
        try:
            state = _active_state(lifespan_state)
            value = ExperimentService(state.require_workspace()).experiments.read(experiment_id)
            return value.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flamo://run-sets/{run_set_id}",
        mime_type="application/json",
        description="Immutable frozen run cohort.",
    )
    async def run_set_resource(run_set_id: str) -> str:
        try:
            state = _active_state(lifespan_state)
            value = RunSetService(state.require_workspace()).store.read(run_set_id)
            return value.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    return server


def _active_state(states: list[AppContext]) -> AppContext:
    if not states:
        raise DomainError(
            ErrorCode.WORKSPACE_NOT_FOUND,
            "The MCP server lifespan is not active.",
        )
    return states[0]


def run_server(project_root: Path, *, initialize: bool = False) -> None:
    create_server(project_root, initialize=initialize).run()
