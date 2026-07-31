from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal, cast

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import (
    CallToolResult,
    InputRequiredResult,
    ResourceLink,
    TextContent,
    Tool,
    ToolAnnotations,
)
from pydantic import BaseModel, Field, RootModel, ValidationError

from flameox.adapters import (
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
    PytestExtractionResult,
    PytestExtractor,
    PythonStartupExtractionResult,
    PythonStartupExtractor,
    TraceWindowResult,
)
from flameox.analysis import (
    ExecutionAnalysisResult,
    FailureAnalysisResult,
    HotspotResult,
    MemoryAnalysisResult,
    PyTorchAnalysisResult,
    RecipeService,
    ScalingAnalysisResult,
)
from flameox.application import (
    AnalysisMaterializationService,
    ArtifactListResult,
    ArtifactMetadataResult,
    ArtifactPipeline,
    ArtifactPipelineService,
    ArtifactService,
    CallEdgeResult,
    CapabilityList,
    CapabilityService,
    CapturePlanRegistry,
    CaptureService,
    CompareRunSetsRequest,
    ComparisonResult,
    ComparisonService,
    CreateInvestigationRequest,
    DeclaredWorkflowDetail,
    DeclaredWorkflowList,
    DetachedCaptureManager,
    DetachedCaptureStatus,
    DrilldownService,
    EvidenceLookupResult,
    EvidenceLookupService,
    EvidenceQueryService,
    EvidenceSummaryBundle,
    EvidenceSummaryRequest,
    EvidenceSummaryService,
    ExecutionPolicy,
    ExperimentPlan,
    ExperimentPlanRegistry,
    ExperimentService,
    ExperimentTrialCollection,
    FindingListResult,
    FindingResult,
    FindingService,
    FreezeRunSetRequest,
    ImportArtifactRequest,
    ImportService,
    IntegrityResult,
    IntegrityService,
    InvestigationListResult,
    InvestigationService,
    MaterializeAnalysisRequest,
    MeasurementQueryResult,
    NativeViewerPlan,
    NativeViewerService,
    PipelineComparison,
    PlanReductionRequest,
    RecordFindingRequest,
    RecordHypothesisRequest,
    ReductionPlan,
    ReductionResult,
    ReductionService,
    RegisterPipelineRequest,
    RunDiscoveryService,
    RunFilter,
    RunListResult,
    RunSetService,
    Scalar,
    StackExamplesResult,
    WorkloadService,
    WorkspaceStatus,
    workspace_status,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    CapturePlan,
    DomainError,
    ErrorCode,
    Experiment,
    ExternalExecutionContext,
    Finding,
    Hypothesis,
    Investigation,
    RunManifest,
    RunSet,
    Sensitivity,
    TrialOutcome,
)
from flameox.models import ContractModel
from flameox.storage import RunStore, Workspace

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
IDEMPOTENT_EXECUTE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=True,
)


class RecoveryAction(ContractModel):
    kind: Literal[
        "repeat_same_call",
        "wait_then_repeat",
        "replan_capture",
        "initialize_workspace",
        "inspect_capabilities",
        "discover_workflows",
        "discover_runs",
        "discover_artifacts",
        "manual",
    ]
    safe_to_repeat_same_call: bool
    retry_after_ms: int | None = Field(default=None, ge=0)
    next_tool: str | None = None


class ErrorDetail(ContractModel):
    code: str
    message: str
    retryable: bool
    details: dict[str, Any]
    remediation: list[str]
    run_id: str | None
    recovery: RecoveryAction


class SuccessPayload[T: BaseModel](ContractModel):
    schema_version: int = 1
    ok: Literal[True] = True
    result: T
    error: None = None


class FailurePayload(ContractModel):
    schema_version: int = 1
    ok: Literal[False] = False
    result: None = None
    error: ErrorDetail


class ToolPayload[T: BaseModel](
    RootModel[
        Annotated[
            SuccessPayload[T] | FailurePayload,
            Field(discriminator="ok"),
        ]
    ]
):
    """Advertised success/failure union matching the structured wire payload."""


class StrictMCPServer[LifespanResultT](MCPServer[LifespanResultT]):
    """Close generated argument schemas and enforce the same boundary at runtime."""

    async def list_tools(self) -> list[Tool]:
        tools = await super().list_tools()
        listed: list[Tool] = []
        for tool in tools:
            update: dict[str, Any] = {
                "input_schema": {
                    **tool.input_schema,
                    "additionalProperties": False,
                }
            }
            if tool.output_schema is not None:
                update["output_schema"] = {**tool.output_schema, "type": "object"}
            listed.append(tool.model_copy(update=update))
        return listed

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context[LifespanResultT, Any] | None = None,
    ) -> CallToolResult | InputRequiredResult:
        tool = next((item for item in await self.list_tools() if item.name == name), None)
        if tool is not None:
            allowed = set(tool.input_schema.get("properties", {}))
            unknown = sorted(set(arguments) - allowed)
            if unknown:
                return _invalid_arguments(
                    name,
                    tuple(
                        {
                            "field": field,
                            "message": "Unknown argument field.",
                            "type": "extra_forbidden",
                        }
                        for field in unknown
                    ),
                )
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as error:
            if isinstance(error.__cause__, ValidationError):
                return _invalid_arguments(name, _validation_fields(error.__cause__))
            raise


class CaptureReceipt(ContractModel):
    """Bounded execution receipt; follow resource_uri for the authoritative run."""

    schema_version: int = 1
    run_id: str
    execution_status: str
    validation_status: str
    source_state_id: str | None
    environment_id: str
    artifact_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    corpus_commit_id: str
    resource_uri: str


class ImportReceipt(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    corpus_commit_id: str
    run_resource_uri: str
    artifact_resource_uri: str


class ExperimentReceipt(ContractModel):
    schema_version: int = 1
    experiment_id: str
    attempted_trials: int
    run_set_ids: tuple[str, ...]
    comparison_id: str | None
    outcome_disposition: str | None = None
    outcome_method: str | None = None
    corpus_commit_id: str
    limitations: tuple[str, ...]
    resource_uri: str
    trial_collection_resource_uri: str
    first_failure_trial_id: str | None = None
    first_failure_resource_uri: str | None = None


class EvidenceReceipt(ContractModel):
    schema_version: int = 1
    ref_type: Literal["analysis", "comparison"]
    ref_id: str
    materialized_commit_id: str
    evidence_ref_ids: tuple[str, ...]
    resource_uri: str


@dataclass(slots=True)
class AppContext:
    project_root: Path
    workspace: Workspace | None
    capabilities: CapabilityService
    capture_plans: CapturePlanRegistry
    experiment_plans: ExperimentPlanRegistry
    detached_captures: DetachedCaptureManager | None = None

    def require_workspace(self) -> Workspace:
        if self.workspace is None:
            raise DomainError(
                code=ErrorCode.WORKSPACE_NOT_FOUND,
                message="No .diagnostics workspace was found.",
                remediation=(
                    "Verify the server's fixed project root is the intended checkout, "
                    "then call initialize_workspace.",
                ),
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

    def detached_service(self) -> DetachedCaptureManager:
        if self.detached_captures is None:
            workspace = self.require_workspace()
            self.detached_captures = DetachedCaptureManager(
                workspace,
                self.capture_service(),
            )
        return self.detached_captures


def _success[T: BaseModel](
    result: T,
    summary: str,
    *,
    resource_links: tuple[ResourceLink, ...] = (),
) -> CallToolResult:
    payload = SuccessPayload[T](result=result)
    return CallToolResult(
        content=[TextContent(type="text", text=summary), *resource_links],
        structured_content=payload.model_dump(mode="json"),
    )


def _failure(error: DomainError) -> CallToolResult:
    detail = ErrorDetail.model_validate(
        {
            **error.to_detail(),
            "recovery": _recovery_for(error).model_dump(mode="json"),
        }
    )
    payload = FailurePayload(error=detail)
    return CallToolResult(
        content=[TextContent(type="text", text=error.message)],
        structured_content=payload.model_dump(mode="json"),
        is_error=True,
    )


def _invalid_arguments(
    tool_name: str,
    fields: tuple[dict[str, str], ...],
) -> CallToolResult:
    field_summary = "; ".join(f"{item['field']}: {item['message']}" for item in fields)
    message = f"Invalid arguments for {tool_name}: {field_summary}"
    payload = FailurePayload(
        error=ErrorDetail(
            code="INVALID_ARGUMENTS",
            message=message,
            retryable=False,
            details={"fields": list(fields)},
            remediation=[f"Match the {tool_name} inputSchema and retry."],
            run_id=None,
            recovery=RecoveryAction(
                kind="manual",
                safe_to_repeat_same_call=False,
            ),
        )
    )
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structured_content=payload.model_dump(mode="json"),
        is_error=True,
    )


def _validation_fields(error: ValidationError) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "field": ".".join(str(part) for part in item["loc"]),
            "message": item["msg"],
            "type": item["type"],
        }
        for item in error.errors()
    )


def _recovery_for(error: DomainError) -> RecoveryAction:
    if error.code is ErrorCode.RUN_NOT_FOUND or error.details.get("missing_entity") == "run":
        return RecoveryAction(
            kind="discover_runs",
            safe_to_repeat_same_call=False,
            next_tool="list_runs",
        )
    if error.details.get("missing_entity") == "artifact":
        return RecoveryAction(
            kind="discover_artifacts",
            safe_to_repeat_same_call=False,
            next_tool="list_artifacts",
        )
    if error.details.get("next_tool") == "list_declared_workflows":
        return RecoveryAction(
            kind="discover_workflows",
            safe_to_repeat_same_call=False,
            next_tool="list_declared_workflows",
        )
    if error.code is ErrorCode.WORKSPACE_NOT_FOUND:
        return RecoveryAction(
            kind="initialize_workspace",
            safe_to_repeat_same_call=False,
            next_tool="initialize_workspace",
        )
    if error.code is ErrorCode.CAPABILITY_UNAVAILABLE:
        return RecoveryAction(
            kind="inspect_capabilities",
            safe_to_repeat_same_call=False,
            next_tool="list_capabilities",
        )
    if error.code in {ErrorCode.INVALID_CAPTURE_PLAN, ErrorCode.PROCESS_TIMEOUT}:
        return RecoveryAction(
            kind="replan_capture",
            safe_to_repeat_same_call=False,
            next_tool="plan_capture",
        )
    if error.code is ErrorCode.WRITE_LOCK_TIMEOUT:
        return RecoveryAction(
            kind="wait_then_repeat",
            safe_to_repeat_same_call=True,
            retry_after_ms=100,
        )
    if error.retryable:
        return RecoveryAction(
            kind="manual",
            safe_to_repeat_same_call=False,
        )
    return RecoveryAction(
        kind="manual",
        safe_to_repeat_same_call=False,
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
            capabilities=CapabilityService(workspace),
            capture_plans=CapturePlanRegistry(
                max_parallel_captures=(
                    workspace.config.capture.max_parallel_captures if workspace is not None else 2
                )
            ),
            experiment_plans=ExperimentPlanRegistry(),
        )
        if workspace is not None:
            state.detached_captures = DetachedCaptureManager(
                workspace,
                state.capture_service(),
            )
        lifespan_state.append(state)
        try:
            yield state
        finally:
            if state.detached_captures is not None:
                await state.detached_captures.shutdown()
            lifespan_state.clear()

    server: MCPServer[AppContext] = StrictMCPServer(
        "flameox",
        description="Query and collect local runtime evidence.",
        instructions=(
            "Use Flameox to collect, preserve, compare, and inspect local runtime evidence "
            "when source, environment, command, and artifact provenance must be reproducible. "
            "Do not use it to provision hosts, install dependencies, mutate source or GitHub, "
            "or prove static claims without runtime evidence. "
            "For a new project, call workspace_status first. If it returns "
            "WORKSPACE_NOT_FOUND, verify that the server's fixed project root is the intended "
            "checkout and, only when authorized, call initialize_workspace; then repeat "
            "workspace_status. Initialization writes .diagnostics. After initialization, use "
            "list_declared_workflows -> list_capabilities -> plan_capture -> "
            "execute_capture_plan for short work, or "
            "start_detached_capture for long work -> get_detached_capture -> get_run -> analyze. "
            "For existing evidence: list_runs and list_artifacts expose artifact_kinds; use "
            "extract_pyperf and query_measurements for benchmark_samples, "
            "extract_python_startup for Python startup/import evidence, extract_pytest for "
            "test phases, fixtures, workers, and failure latency, analyze_memory for "
            "memory profiles, analyze_execution for coverage, and the other analyze_* tools "
            "only for their documented artifact kinds. Then use get_evidence, record_analysis, "
            "or record_finding. Initialize a missing workspace only when authorized. A "
            "synchronous consumed capture plan is never retryable; detached starts are "
            "retryable only with the same idempotency key."
        ),
        lifespan=lifespan,
    )

    @server.tool(annotations=INITIALIZE)
    async def initialize_workspace(
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[WorkspaceStatus]]:
        """Initialize Flameox in the fixed project root after it has been verified."""
        try:
            state = ctx.request_context.lifespan_context
            if state.workspace is not None:
                result = workspace_status(state.workspace)
                return _success(result, f"Workspace is already initialized: {result.workspace_id}.")

            workspace = Workspace.initialize(state.project_root)
            capture_plans = CapturePlanRegistry(
                max_parallel_captures=workspace.config.capture.max_parallel_captures
            )
            detached_captures = DetachedCaptureManager(
                workspace,
                CaptureService(workspace, plans=capture_plans),
            )
            state.workspace = workspace
            state.capabilities = CapabilityService(workspace)
            state.capture_plans = capture_plans
            state.experiment_plans = ExperimentPlanRegistry()
            state.detached_captures = detached_captures
            result = workspace_status(workspace)
            return _success(result, f"Initialized workspace {result.workspace_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="workspace_status", annotations=READ_ONLY)
    async def workspace_status_tool(
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[WorkspaceStatus]]:
        """Return workspace status; on first use, follow WORKSPACE_NOT_FOUND recovery."""
        try:
            result = workspace_status(ctx.request_context.lifespan_context.require_workspace())
            return _success(result, f"Workspace is at {result.corpus_commit_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_capabilities", annotations=READ_ONLY)
    async def list_capabilities_tool(
        ctx: Context[AppContext],
        mode: Literal["passive", "active_cached", "active_refresh"] = "passive",
    ) -> Annotated[CallToolResult, ToolPayload[CapabilityList]]:
        """List capabilities; use passive, active_cached, or active_refresh mode explicitly."""
        service = ctx.request_context.lifespan_context.capabilities
        result = (
            service.list()
            if mode == "passive"
            else await service.list_active(refresh=mode == "active_refresh")
        )
        return _success(
            result,
            f"Found {sum(item.status.value == 'available' for item in result.capabilities)} of "
            f"{len(result.capabilities)} available capabilities.",
        )

    @server.tool(name="list_declared_workflows", annotations=READ_ONLY)
    async def list_declared_workflows_tool(
        kind: Literal["workload", "experiment"],
        ctx: Context[AppContext],
        approval: Literal["approved", "unapproved", "any"] = "any",
        limit: Annotated[int, Field(ge=1, le=100)] = 50,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[DeclaredWorkflowList]]:
        """Choose a declared workload or experiment before planning; this never approves or runs."""
        try:
            result = WorkloadService(
                ctx.request_context.lifespan_context.require_workspace()
            ).list_declared(
                kind=kind,
                approval=approval,
                limit=limit,
                cursor=cursor,
            )
            return _success(
                result,
                f"Returned {result.returned} declared {kind} definitions.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_declared_workflow", annotations=READ_ONLY)
    async def get_declared_workflow_tool(
        kind: Literal["workload", "experiment"],
        name: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[DeclaredWorkflowDetail]]:
        """Inspect allowed parameters and validation metadata, then call the matching plan tool."""
        try:
            result = WorkloadService(
                ctx.request_context.lifespan_context.require_workspace()
            ).get_declared(kind=kind, name=name)
            return _success(result, f"Loaded declared {kind} {name}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="plan_capture", annotations=READ_ONLY)
    async def plan_capture_tool(
        workload_name: Annotated[
            str,
            Field(
                min_length=1,
                description="Approved declared workload name from list_declared_workflows.",
            ),
        ],
        adapter: Annotated[
            str,
            Field(description="Adapter value reported by list_capabilities for this workload."),
        ],
        parameters: Annotated[
            dict[str, Scalar],
            Field(description="Declared workload parameters; inspect get_declared_workflow first."),
        ],
        ctx: Context[AppContext],
        preflight_mode: Literal["passive", "active"] = "passive",
        external_context: ExternalExecutionContext | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[CapturePlan]]:
        """After workflow and capability discovery, bind one approved capture without running it."""
        try:
            result = await ctx.request_context.lifespan_context.capture_service().plan(
                workload_name=workload_name,
                adapter=adapter,
                parameters=parameters,
                execution_policy=ExecutionPolicy.APPROVED_AGENT,
                preflight_mode=preflight_mode,
                external_context=external_context,
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
    ) -> Annotated[CallToolResult, ToolPayload[CaptureReceipt]]:
        """Run one approved plan with side effects; the token is single-use, then get_run."""
        try:
            await ctx.report_progress(0, 8, "Capture request accepted")

            async def report(
                completed: float,
                total: float,
                message: str,
            ) -> None:
                await ctx.report_progress(completed, total, message)

            result = await ctx.request_context.lifespan_context.capture_service().execute(
                plan_id,
                progress=report,
            )
            resource_uri = f"flameox://runs/{result.run.run_id}"
            receipt = CaptureReceipt(
                run_id=result.run.run_id,
                execution_status=result.run.execution_status.value,
                validation_status=result.run.validation_status.value,
                source_state_id=result.run.source_state_id,
                environment_id=result.run.environment_id,
                artifact_ids=tuple(item.artifact_id for item in result.run.artifacts),
                limitations=result.run.limitations,
                corpus_commit_id=result.corpus_commit_id,
                resource_uri=resource_uri,
            )
            return _success(
                receipt,
                f"Capture run {result.run.run_id} is {result.run.execution_status.value}.",
                resource_links=(
                    ResourceLink(
                        name=f"Run {result.run.run_id}",
                        uri=resource_uri,
                        description="Authoritative run manifest and provenance.",
                        mime_type="application/json",
                    ),
                ),
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="start_detached_capture", annotations=IDEMPOTENT_EXECUTE)
    async def start_detached_capture_tool(
        plan_id: str,
        idempotency_key: Annotated[
            str,
            Field(min_length=8, max_length=200, pattern=r"^[A-Za-z0-9._:/-]+$"),
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[DetachedCaptureStatus]]:
        """Start one approved plan once; reconnect by run_id without keeping this call open."""
        try:
            result = await ctx.request_context.lifespan_context.detached_service().start(
                plan_id,
                idempotency_key,
            )
            return _success(
                result,
                f"Detached capture {result.run_id} is {result.state}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def get_detached_capture(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[DetachedCaptureStatus]]:
        """Reconnect to bounded progress and lifecycle status for one detached run."""
        try:
            result = ctx.request_context.lifespan_context.detached_service().status(run_id)
            return _success(result, f"Detached capture {run_id} is {result.state}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=IDEMPOTENT_EXECUTE)
    async def cancel_detached_capture(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[DetachedCaptureStatus]]:
        """Cancel only the exact detached task owned by this server; repeated calls are safe."""
        try:
            result = await ctx.request_context.lifespan_context.detached_service().cancel(run_id)
            return _success(result, f"Detached capture {run_id} is {result.state}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="plan_experiment", annotations=READ_ONLY)
    async def plan_experiment_tool(
        experiment_name: Annotated[
            str,
            Field(min_length=1, description="Approved experiment from list_declared_workflows."),
        ],
        investigation_id: Annotated[
            str,
            Field(min_length=1, description="Investigation ID from create/list investigations."),
        ],
        adapter: Annotated[
            str,
            Field(description="Adapter capability reported by list_capabilities."),
        ],
        parameters: Annotated[
            dict[str, Scalar],
            Field(description="Declared parameters; inspect get_declared_workflow first."),
        ],
        ctx: Context[AppContext],
        hypothesis_id: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[ExperimentPlan]]:
        """After workflow and capability discovery, bind a declared experiment; then run it."""
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
    ) -> Annotated[CallToolResult, ToolPayload[ExperimentReceipt]]:
        """Execute all approved trials from one single-use plan, then inspect get_experiment."""
        try:

            async def report(
                completed: float,
                total: float,
                message: str,
            ) -> None:
                await ctx.report_progress(completed, total, message)

            result = await ctx.request_context.lifespan_context.experiment_service().run(
                plan_id,
                progress=report,
            )
            experiment_id = result.experiment.experiment_id
            resource_uri = f"flameox://experiments/{experiment_id}"
            attempted_trials = sum(trial.outcome != "unattempted" for trial in result.trials)
            trial_collection_uri = f"{resource_uri}/trials"
            first_failure_trial_id = (
                result.outcome.first_failure_trial_id
                if result.outcome is not None
                else next(
                    (
                        trial.trial_id
                        for trial in result.trials
                        if trial.outcome
                        not in {
                            TrialOutcome.SUCCEEDED,
                            TrialOutcome.UNSUPPORTED,
                            TrialOutcome.UNATTEMPTED,
                        }
                    ),
                    None,
                )
            )
            first_failure_uri = (
                f"{trial_collection_uri}/{first_failure_trial_id}"
                if first_failure_trial_id is not None
                else None
            )
            receipt = ExperimentReceipt(
                experiment_id=experiment_id,
                attempted_trials=attempted_trials,
                run_set_ids=tuple(item.run_set_id for item in result.run_sets),
                comparison_id=(
                    result.comparison.comparison.comparison_id
                    if result.comparison is not None
                    else None
                ),
                outcome_disposition=(
                    result.outcome.disposition if result.outcome is not None else None
                ),
                outcome_method=result.outcome.method if result.outcome is not None else None,
                corpus_commit_id=result.corpus_commit_id,
                limitations=result.limitations,
                resource_uri=resource_uri,
                first_failure_trial_id=first_failure_trial_id,
                first_failure_resource_uri=first_failure_uri,
                trial_collection_resource_uri=trial_collection_uri,
            )
            return _success(
                receipt,
                f"Experiment {experiment_id} recorded {attempted_trials} attempted trials.",
                resource_links=(
                    ResourceLink(
                        name=f"Experiment {experiment_id}",
                        uri=resource_uri,
                        description="Immutable experiment protocol.",
                        mime_type="application/json",
                    ),
                    ResourceLink(
                        name=f"Experiment {experiment_id} trials",
                        uri=trial_collection_uri,
                        description="Bounded immutable trial collection.",
                        mime_type="application/json",
                    ),
                    *(
                        (
                            ResourceLink(
                                name=f"First failing trial {first_failure_trial_id}",
                                uri=first_failure_uri,
                                description="First failing trial and structured oracle receipt.",
                                mime_type="application/json",
                            ),
                        )
                        if first_failure_uri is not None
                        else ()
                    ),
                ),
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def plan_reduction(
        request: PlanReductionRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ReductionPlan]]:
        """Bind immutable input and approved reducer/predicate identities before execution."""
        try:
            plan = ReductionService(ctx.request_context.lifespan_context.require_workspace()).plan(
                request
            )
            return _success(plan, f"Planned reduction {plan.plan_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=EXECUTE)
    async def execute_reduction(
        plan_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ReductionResult]]:
        """Execute one bound reducer lifecycle and independently revalidate its candidate."""
        try:
            result = await ReductionService(
                ctx.request_context.lifespan_context.require_workspace()
            ).execute(plan_id)
            return _success(
                result,
                f"Reduction {result.reduction_id} is {result.disposition}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def get_reduction(
        reduction_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ReductionResult]]:
        """Reconnect to one immutable terminal reduction result."""
        try:
            result = ReductionService(ctx.request_context.lifespan_context.require_workspace()).get(
                reduction_id
            )
            return _success(
                result,
                f"Reduction {result.reduction_id} is {result.disposition}.",
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

    @server.tool(name="list_experiment_trials", annotations=READ_ONLY)
    async def list_experiment_trials_tool(
        experiment_id: str,
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[ExperimentTrialCollection]]:
        """Return one bounded page of immutable trials for an experiment."""
        try:
            result = ExperimentService(
                ctx.request_context.lifespan_context.require_workspace()
            ).list_trials(experiment_id, limit=limit, cursor=cursor)
            return _success(result, f"Returned {result.returned} experiment trials.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="import_artifact", annotations=ADDITIVE)
    async def import_artifact_tool(
        path: Annotated[
            str,
            Field(
                description="Project-relative artifact path; absolute and parent paths are refused."
            ),
        ],
        kind: Annotated[
            ArtifactKind,
            Field(description="ArtifactKind enum value describing the imported format."),
        ],
        sensitivity: Annotated[
            Sensitivity,
            Field(description="Sensitivity classification: normal, internal, or sensitive."),
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ImportReceipt]]:
        """Import one project-local artifact as an immutable run; follow returned resource links."""
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
            run_uri = f"flameox://runs/{result.run.run_id}"
            artifact_uri = f"flameox://artifacts/{result.artifact_id}"
            receipt = ImportReceipt(
                run_id=result.run.run_id,
                artifact_id=result.artifact_id,
                corpus_commit_id=result.corpus_commit_id,
                run_resource_uri=run_uri,
                artifact_resource_uri=artifact_uri,
            )
            return _success(
                receipt,
                f"Imported {result.artifact_id} in run {result.run.run_id}.",
                resource_links=(
                    ResourceLink(
                        name=f"Run {result.run.run_id}",
                        uri=run_uri,
                        description="Authoritative import run manifest.",
                        mime_type="application/json",
                    ),
                    ResourceLink(
                        name=f"Artifact {result.artifact_id}",
                        uri=artifact_uri,
                        description="Imported artifact metadata.",
                        mime_type="application/json",
                    ),
                ),
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def list_runs(
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
        filter: RunFilter | None = None,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[RunListResult]]:
        """Discover a filtered run cohort; follow next_cursor without changing filters."""
        try:
            result = RunDiscoveryService(
                ctx.request_context.lifespan_context.require_workspace()
            ).list(filter=filter or RunFilter(), limit=limit, cursor=cursor)
            noun = "run" if result.returned == 1 else "runs"
            return _success(result, f"Returned {result.returned} {noun}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def get_run(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[RunManifest]]:
        """Hydrate a run selected by list_runs; use an analyze_* tool for bounded interpretation."""
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

    @server.tool(annotations=ADDITIVE)
    async def register_artifact_pipeline(
        request: RegisterPipelineRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[ArtifactPipeline]]:
        """Bind a bounded ordered pipeline to existing immutable run artifacts."""
        try:
            pipeline = ArtifactPipelineService(
                ctx.request_context.lifespan_context.require_workspace()
            ).register(request)
            return _success(pipeline, f"Registered artifact pipeline {pipeline.pipeline_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def compare_artifact_pipelines(
        baseline_pipeline_id: str,
        candidate_pipeline_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[PipelineComparison]]:
        """Compare compatible ordered stages without returning native artifact content."""
        try:
            comparison = ArtifactPipelineService(
                ctx.request_context.lifespan_context.require_workspace()
            ).compare(baseline_pipeline_id, candidate_pipeline_id)
            return _success(
                comparison,
                f"Compared artifact pipelines as {comparison.result_digest}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(annotations=READ_ONLY)
    async def summarize_evidence(
        request: EvidenceSummaryRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[EvidenceSummaryBundle]]:
        """Render one bounded canonical proof summary and its Markdown view."""
        try:
            result = EvidenceSummaryService(
                ctx.request_context.lifespan_context.require_workspace()
            ).summarize(request)
            return _success(
                result,
                f"Summarized evidence as {result.summary.summary_digest}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_artifacts", annotations=READ_ONLY)
    async def list_artifacts_tool(
        limit: Annotated[int, Field(ge=1, le=1_000)],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[ArtifactListResult]]:
        """List bounded artifact metadata from one pinned corpus snapshot."""
        try:
            result = ArtifactService(ctx.request_context.lifespan_context.require_workspace()).list(
                limit=limit,
                cursor=cursor,
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
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[InvestigationListResult]]:
        """List bounded current investigation projections."""
        try:
            result = InvestigationService(
                ctx.request_context.lifespan_context.require_workspace()
            ).list(limit=limit, cursor=cursor)
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
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[FindingListResult]]:
        """List bounded current finding projections."""
        try:
            result = FindingService(ctx.request_context.lifespan_context.require_workspace()).list(
                limit=limit, cursor=cursor
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
        """Preview compatible frozen cohorts without persistence; use record_comparison to save."""
        try:

            async def report(
                completed: float,
                total: float,
                message: str,
            ) -> None:
                await ctx.report_progress(completed, total, message)

            result = await ComparisonService(
                ctx.request_context.lifespan_context.require_workspace()
            ).compare_async(request, progress=report)
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
    ) -> Annotated[CallToolResult, ToolPayload[EvidenceReceipt]]:
        """Persist a reviewed comparison; use compare_run_sets for read-only preview."""
        try:

            async def report(
                completed: float,
                total: float,
                message: str,
            ) -> None:
                await ctx.report_progress(completed, total, message)

            result = await ComparisonService(
                ctx.request_context.lifespan_context.require_workspace()
            ).record_async(request, progress=report)
            comparison_id = result.comparison.comparison_id
            resource_uri = f"flameox://evidence/comparison/{comparison_id}"
            assert result.materialized_commit_id is not None
            receipt = EvidenceReceipt(
                ref_type="comparison",
                ref_id=comparison_id,
                materialized_commit_id=result.materialized_commit_id,
                evidence_ref_ids=tuple(item.ref_id for item in result.evidence),
                resource_uri=resource_uri,
            )
            return _success(
                receipt,
                f"Recorded comparison {comparison_id}.",
                resource_links=(
                    ResourceLink(
                        name=f"Comparison {comparison_id}",
                        uri=resource_uri,
                        description="Authoritative materialized comparison evidence.",
                        mime_type="application/json",
                    ),
                ),
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="record_analysis", annotations=ADDITIVE)
    async def record_analysis_tool(
        request: MaterializeAnalysisRequest,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[EvidenceReceipt]]:
        """Persist a curated analysis; use analyze_* first for read-only preview."""
        try:

            async def report(
                completed: float,
                total: float,
                message: str,
            ) -> None:
                await ctx.report_progress(completed, total, message)

            service = AnalysisMaterializationService(
                ctx.request_context.lifespan_context.require_workspace()
            )
            result = await service.record_async(request, progress=report)
            analysis_id = result.analysis.analysis_id
            resource_uri = f"flameox://evidence/analysis/{analysis_id}"
            receipt = EvidenceReceipt(
                ref_type="analysis",
                ref_id=analysis_id,
                materialized_commit_id=result.materialized_commit_id,
                evidence_ref_ids=tuple(item.ref_id for item in result.evidence),
                resource_uri=resource_uri,
            )
            return _success(
                receipt,
                f"Recorded {result.analysis.recipe} analysis {analysis_id}.",
                resource_links=(
                    ResourceLink(
                        name=f"Analysis {analysis_id}",
                        uri=resource_uri,
                        description="Authoritative materialized analysis evidence.",
                        mime_type="application/json",
                    ),
                ),
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_hotspots", annotations=READ_ONLY)
    async def analyze_hotspots_tool(
        run_or_artifact: Annotated[
            str,
            Field(
                min_length=1,
                description="Run or artifact ID; discover one with list_runs or list_artifacts.",
            ),
        ],
        limit: Annotated[
            int, Field(ge=1, le=1_000, description="Maximum hotspots to return (1-1000).")
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[HotspotResult]]:
        """Analyze sampled-profile runs or artifacts for bounded source-linked hotspots; use
        extract_pyperf/query_measurements for benchmark_samples instead."""
        try:
            workspace = ctx.request_context.lifespan_context.require_workspace()
            await ctx.report_progress(0, 2, "Hotspot snapshot pinned")
            result = await Catalog(workspace).run_interruptible(
                lambda snapshot: RecipeService(
                    workspace,
                    snapshot=snapshot,
                ).hotspots(run_or_artifact, limit=limit),
                query_name="analyze_hotspots",
            )
            await ctx.report_progress(1, 2, "Hotspot query complete")
            await ctx.report_progress(2, 2, "Hotspot result ready")
            return _success(
                result,
                f"Returned {result.returned} of {result.total} hotspots.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_memory", annotations=READ_ONLY)
    async def analyze_memory_tool(
        run_or_artifact: Annotated[
            str,
            Field(
                min_length=1,
                description="Run or artifact ID; discover one with list_runs or list_artifacts.",
            ),
        ],
        limit: Annotated[
            int,
            Field(ge=1, le=1_000, description="Maximum memory observations to return (1-1000)."),
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[MemoryAnalysisResult]]:
        """Analyze memory-profile runs or artifacts for peak, retained-end, and
        allocation evidence."""
        try:
            workspace = ctx.request_context.lifespan_context.require_workspace()
            await ctx.report_progress(0, 2, "Memory snapshot pinned")
            result = await Catalog(workspace).run_interruptible(
                lambda snapshot: RecipeService(
                    workspace,
                    snapshot=snapshot,
                ).memory(run_or_artifact, limit=limit),
                query_name="analyze_memory",
            )
            await ctx.report_progress(1, 2, "Memory query complete")
            await ctx.report_progress(2, 2, "Memory result ready")
            return _success(
                result,
                f"Returned {len(result.measurements)} memory measurements.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_execution", annotations=READ_ONLY)
    async def analyze_execution_tool(
        run_or_artifact: Annotated[
            str,
            Field(
                min_length=1,
                description=(
                    "Primary run or artifact ID; discover with list_runs or list_artifacts."
                ),
            ),
        ],
        limit: Annotated[
            int,
            Field(ge=1, le=1_000, description="Maximum execution observations to return (1-1000)."),
        ],
        ctx: Context[AppContext],
        comparison_run_or_artifact: Annotated[
            str | None,
            Field(description="Optional second run or artifact ID for a compatible comparison."),
        ] = None,
    ) -> Annotated[CallToolResult, ToolPayload[ExecutionAnalysisResult]]:
        """Inspect execution-coverage runs or a compatible pair read-only; use
        record_analysis to preserve it."""
        try:
            workspace = ctx.request_context.lifespan_context.require_workspace()
            await ctx.report_progress(0, 2, "Execution snapshot pinned")
            result = await Catalog(workspace).run_interruptible(
                lambda snapshot: RecipeService(
                    workspace,
                    snapshot=snapshot,
                ).execution(
                    run_or_artifact,
                    comparison_input_id=comparison_run_or_artifact,
                    limit=limit,
                ),
                query_name="analyze_execution",
            )
            await ctx.report_progress(1, 2, "Execution query complete")
            await ctx.report_progress(2, 2, "Execution result ready")
            return _success(
                result,
                f"Returned {result.returned} of {result.total} observations.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="analyze_pytorch", annotations=READ_ONLY)
    async def analyze_pytorch_tool(
        run_or_artifact: Annotated[
            str,
            Field(
                min_length=1,
                description="Run ID or artifact ID for an imported torch.profiler trace.",
            ),
        ],
        limit: Annotated[
            int, Field(ge=1, le=1_000, description="Maximum operators to return (1-1000).")
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[PyTorchAnalysisResult]]:
        """Summarize one imported torch.profiler run or artifact; other kinds are
        unsupported."""
        try:
            workspace = ctx.request_context.lifespan_context.require_workspace()
            await ctx.report_progress(0, 2, "PyTorch snapshot pinned")
            result = await Catalog(workspace).run_interruptible(
                lambda snapshot: RecipeService(
                    workspace,
                    snapshot=snapshot,
                ).pytorch(run_or_artifact, limit=limit),
                query_name="analyze_pytorch",
            )
            await ctx.report_progress(1, 2, "PyTorch query complete")
            await ctx.report_progress(2, 2, "PyTorch result ready")
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
            workspace = ctx.request_context.lifespan_context.require_workspace()
            await ctx.report_progress(0, 2, "Scaling snapshot pinned")
            result = await Catalog(workspace).run_interruptible(
                lambda snapshot: RecipeService(
                    workspace,
                    snapshot=snapshot,
                ).scaling(experiment_id),
                query_name="analyze_scaling",
            )
            await ctx.report_progress(1, 2, "Scaling query complete")
            await ctx.report_progress(2, 2, "Scaling result ready")
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
        filter: RunFilter | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[FailureAnalysisResult]]:
        """Analyze an explicit filtered failure cohort read-only after list_runs discovery."""
        try:
            workspace = ctx.request_context.lifespan_context.require_workspace()
            selected_filter = filter or RunFilter()
            await ctx.report_progress(0, 2, "Failure-population snapshot pinned")
            result = await Catalog(workspace).run_interruptible(
                lambda snapshot: RecipeService(
                    workspace,
                    snapshot=snapshot,
                ).failures(
                    limit=limit,
                    source_state_id=selected_filter.source_state_id,
                    environment_id=selected_filter.environment_id,
                    workload_definition_id=selected_filter.workload_definition_id,
                    execution_status=selected_filter.execution_status,
                    validation_status=selected_filter.validation_status,
                    created_after=selected_filter.created_after,
                    created_before=selected_filter.created_before,
                ),
                query_name="analyze_failures",
            )
            await ctx.report_progress(1, 2, "Failure-population query complete")
            await ctx.report_progress(2, 2, "Failure-population result ready")
            return _success(result, f"Returned {result.returned} failure clusters.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_frame_callers", annotations=READ_ONLY)
    async def get_frame_callers_tool(
        run_or_artifact: Annotated[
            str,
            Field(min_length=1, description="Run ID or artifact ID containing the frame."),
        ],
        frame_id: Annotated[
            str, Field(min_length=1, description="Frame ID returned by an analysis result.")
        ],
        limit: Annotated[
            int, Field(ge=1, le=1_000, description="Maximum caller edges to return (1-1000).")
        ],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[CallEdgeResult]]:
        """Return bounded source-linked direct callers for a frame."""
        try:
            result = DrilldownService(
                ctx.request_context.lifespan_context.require_workspace()
            ).callers(run_or_artifact, frame_id, limit=limit, cursor=cursor)
            return _success(result, f"Returned {result.returned} callers.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_frame_callees", annotations=READ_ONLY)
    async def get_frame_callees_tool(
        run_or_artifact: Annotated[
            str,
            Field(min_length=1, description="Run ID or artifact ID containing the frame."),
        ],
        frame_id: Annotated[
            str, Field(min_length=1, description="Frame ID returned by an analysis result.")
        ],
        limit: Annotated[
            int, Field(ge=1, le=1_000, description="Maximum callee edges to return (1-1000).")
        ],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[CallEdgeResult]]:
        """Return bounded source-linked direct callees for a frame."""
        try:
            result = DrilldownService(
                ctx.request_context.lifespan_context.require_workspace()
            ).callees(run_or_artifact, frame_id, limit=limit, cursor=cursor)
            return _success(result, f"Returned {result.returned} callees.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_stack_examples", annotations=READ_ONLY)
    async def get_stack_examples_tool(
        run_or_artifact: Annotated[
            str,
            Field(min_length=1, description="Run ID or artifact ID containing the frame."),
        ],
        frame_id: Annotated[
            str, Field(min_length=1, description="Frame ID returned by an analysis result.")
        ],
        limit: Annotated[
            int, Field(ge=1, le=1_000, description="Maximum stack examples to return (1-1000).")
        ],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[StackExamplesResult]]:
        """Return bounded representative stacks containing a frame."""
        try:
            result = DrilldownService(
                ctx.request_context.lifespan_context.require_workspace()
            ).examples(run_or_artifact, frame_id, limit=limit, cursor=cursor)
            return _success(result, f"Returned {result.returned} stack examples.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_trace_window", annotations=READ_ONLY)
    async def get_trace_window_tool(
        artifact_id: Annotated[
            str, Field(min_length=1, description="Trace artifact ID from list_artifacts.")
        ],
        start_ns: Annotated[int, Field(ge=0, description="Inclusive window start in nanoseconds.")],
        end_ns: Annotated[
            int,
            Field(
                gt=0,
                description="Exclusive window end in nanoseconds; greater than start_ns.",
            ),
        ],
        limit: Annotated[
            int, Field(ge=1, le=1_000, description="Maximum trace slices to return (1-1000).")
        ],
        ctx: Context[AppContext],
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[TraceWindowResult]]:
        """Return bounded trace slices overlapping a declared time window."""
        try:
            if end_ns <= start_ns:
                return _invalid_arguments(
                    "get_trace_window",
                    (
                        {
                            "field": "end_ns",
                            "message": "must be greater than start_ns",
                            "type": "greater_than",
                        },
                    ),
                )
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
        ref_id: Annotated[
            str, Field(min_length=1, description="ID of the reference, paired with ref_type.")
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[EvidenceLookupResult]]:
        """Resolve a known typed reference; pass ref_type and its ID separately after discovery."""
        try:
            result = EvidenceLookupService(
                ctx.request_context.lifespan_context.require_workspace()
            ).get(ref_type, ref_id)
            return _success(result, f"Retrieved {ref_type} evidence {ref_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="validate_workspace", annotations=READ_ONLY)
    async def validate_workspace_tool(
        ctx: Context[AppContext],
        mode: Literal["standard", "full"] = "standard",
    ) -> Annotated[CallToolResult, ToolPayload[IntegrityResult]]:
        """Validate manifests and schemas; optionally hash every payload."""
        try:
            result = IntegrityService(
                ctx.request_context.lifespan_context.require_workspace()
            ).validate(full=mode == "full")
            outcome = "passed" if result.valid else "failed"
            issue_suffix = f" with {len(result.issues)} reported issues" if result.issues else ""
            return _success(result, f"Workspace validation {outcome}{issue_suffix}.")
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

    @server.tool(name="extract_python_startup", annotations=ADDITIVE)
    async def extract_python_startup_tool(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[PythonStartupExtractionResult]]:
        """Extract repeated startup, peak RSS, and package-grouped import evidence."""
        try:
            result = PythonStartupExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id)
            return _success(
                result,
                f"Extracted {result.measurement_count} startup measurements.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_pytest", annotations=ADDITIVE)
    async def extract_pytest_tool(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[PytestExtractionResult]]:
        """Extract pytest phase, fixture, worker, outcome, and failure-latency evidence."""
        try:
            result = PytestExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id)
            return _success(
                result,
                f"Extracted {result.measurement_count} pytest measurements.",
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
        """Extract bounded semantic observations emitted through flameox.sdk."""
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
        "flameox://runs/{run_id}",
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
        "flameox://artifacts/{artifact_id}",
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
        "flameox://investigations/{investigation_id}",
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
        "flameox://hypotheses/{hypothesis_id}",
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
        "flameox://findings/{finding_id}",
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
        "flameox://experiments/{experiment_id}",
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
        "flameox://experiments/{experiment_id}/trials",
        mime_type="application/json",
        description="Bounded immutable trial collection for an experiment.",
    )
    async def experiment_trials_resource(experiment_id: str) -> str:
        try:
            state = _active_state(lifespan_state)
            value = ExperimentService(state.require_workspace()).list_trials(experiment_id)
            return value.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flameox://experiments/{experiment_id}/trials/{trial_id}",
        mime_type="application/json",
        description="One immutable trial and its structured oracle receipt.",
    )
    async def experiment_trial_resource(experiment_id: str, trial_id: str) -> str:
        try:
            state = _active_state(lifespan_state)
            value = ExperimentService(state.require_workspace()).get_trial(
                trial_id,
                experiment_id=experiment_id,
            )
            return value.model_dump_json(indent=2)
        except DomainError as error:
            return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flameox://run-sets/{run_set_id}",
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

    @server.resource(
        "flameox://evidence/{ref_type}/{ref_id}",
        mime_type="application/json",
        description="Authoritative persisted analysis or comparison evidence.",
    )
    async def evidence_resource(ref_type: str, ref_id: str) -> str:
        try:
            if ref_type not in {"analysis", "comparison"}:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Unsupported evidence resource type {ref_type!r}.",
                )
            state = _active_state(lifespan_state)
            value = EvidenceLookupService(state.require_workspace()).get(
                cast(Literal["analysis", "comparison"], ref_type),
                ref_id,
            )
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
