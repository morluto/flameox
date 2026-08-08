from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
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
from pydantic import BaseModel, Field, RootModel, StrictInt, ValidationError

from flameox import __version__
from flameox.adapters import (
    BenchmarkSamplesExtractionResult,
    BenchmarkSamplesExtractor,
    CoverageExtractionResult,
    CoverageExtractor,
    InferenceArtifactExtractor,
    InferenceExtractionResult,
    MemrayExtractionResult,
    MemrayExtractor,
    NsightSystemsExtractionResult,
    NsightSystemsExtractor,
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
    TorchProfilerCaptureOptions,
    TraceWindowResult,
)
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
from flameox.application import (
    AdapterPreparationResult,
    AnalysisMaterializationService,
    ArtifactListResult,
    ArtifactMetadataResult,
    ArtifactPipeline,
    ArtifactPipelineService,
    ArtifactService,
    CallEdgeResult,
    CapabilityList,
    CapabilityService,
    CapabilitySetupManager,
    CapturePlanRegistry,
    CaptureService,
    CompareRunSetsRequest,
    ComparisonResult,
    ComparisonService,
    ConfigureInferenceScenarioRequest,
    ConfigureInferenceServerRequest,
    ConfigureWorkloadRequest,
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
    FaultExperimentPlan,
    FaultExperimentResult,
    FaultExperimentService,
    FindingListResult,
    FindingResult,
    FindingService,
    FreezeRunSetRequest,
    ImportArtifactRequest,
    ImportService,
    InferenceConfigurationList,
    InferenceConfigurationResult,
    InferenceProfilingPlan,
    InferenceProfilingResult,
    InferenceProfilingService,
    InferenceReplayPlan,
    InferenceReplayResult,
    InferenceReplayService,
    InferenceRequestQueryResult,
    InferenceScenarioConfig,
    InferenceServerConfig,
    IntegrityResult,
    IntegrityService,
    InvestigationListResult,
    InvestigationService,
    LifecycleEvidenceService,
    LifecycleQueryResult,
    MaterializeAnalysisRequest,
    MeasurementQueryResult,
    NativeViewerPlan,
    NativeViewerService,
    OtlpExtractionResult,
    OtlpTraceService,
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
    WorkloadConfig,
    WorkloadConfigurationResult,
    WorkloadConfigurationStatus,
    WorkloadDependencyService,
    WorkloadDependencySetupResult,
    WorkloadIdentityConfig,
    WorkloadOracleConfig,
    WorkloadRequirementsConfig,
    WorkloadService,
    WorkspaceStatus,
    workspace_status,
)
from flameox.application.async_work import run_atomic_thread
from flameox.application.operations import OperationStatus
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
    LimitationDetail,
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
CONFIGURE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
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


class ConfigureWorkloadRecoveryContext(ContractModel):
    """Typed arguments for the safe structured configuration recovery."""

    kind: Literal["configure_workload"]
    operation: Literal["create"] = "create"
    config_path: Literal["flameox.toml"] = "flameox.toml"


class ManualConfigurationRecoveryContext(ContractModel):
    """Typed context for configuration that cannot be safely rewritten by MCP."""

    kind: Literal["manual_configuration"]
    config_path: Literal["flameox.toml"] = "flameox.toml"
    diagnostic: str = Field(max_length=500)
    verification_tool: Literal["workload_configuration_status"] = "workload_configuration_status"


class ExtractPerfettoRecoveryContext(ContractModel):
    kind: Literal["extract_perfetto"]
    run_id: str = Field(min_length=1)


type RecoveryContext = Annotated[
    ConfigureWorkloadRecoveryContext
    | ManualConfigurationRecoveryContext
    | ExtractPerfettoRecoveryContext,
    Field(discriminator="kind"),
]


class RecoveryAction(ContractModel):
    kind: Literal[
        "repeat_same_call",
        "wait_then_repeat",
        "replan_capture",
        "initialize_workspace",
        "configure_workload",
        "inspect_workload_configuration",
        "start_capability_setup",
        "prepare_adapter",
        "prepare_workload_dependencies",
        "inspect_capabilities",
        "discover_workflows",
        "discover_runs",
        "discover_artifacts",
        "import_artifact",
        "extract_perfetto",
        "configure_inference",
        "replan_inference",
        "manual",
    ]
    safe_to_repeat_same_call: bool
    retry_after_ms: int | None = Field(default=None, ge=0)
    next_tool: str | None = None
    context: RecoveryContext | None = None


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
    limitation_details: tuple[LimitationDetail, ...] = ()
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
    capture: CaptureService | None = None
    detached_captures: DetachedCaptureManager | None = None
    capability_setup: CapabilitySetupManager | None = None

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
        if self.capture is None:
            self.capture = CaptureService(
                self.require_workspace(),
                plans=self.capture_plans,
                capabilities=self.capabilities,
            )
        return self.capture

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

    def capability_setup_service(self) -> CapabilitySetupManager:
        if self.capability_setup is None:
            workspace = self.require_workspace()
            self.capability_setup = CapabilitySetupManager(workspace, self.capabilities)
        return self.capability_setup


def _success[T: BaseModel](
    result: T,
    summary: str,
    *,
    resource_links: tuple[ResourceLink, ...] = (),
) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=summary), *resource_links],
        # MCPServer's v2 converter validates this against ToolPayload[T] exactly once.
        # The application result is already a typed Pydantic model; constructing the
        # envelope here would validate the same payload a second time.
        structured_content={
            "schema_version": 1,
            "ok": True,
            "result": result.model_dump(mode="json"),
            "error": None,
        },
    )


def _failure(error: DomainError) -> CallToolResult:
    recovery = _recovery_for(error)
    visible_message = error.message
    if recovery.next_tool is not None:
        visible_message += f" Next tool: {recovery.next_tool}."
    if error.remediation:
        visible_message += f" {error.remediation[0]}"
    recovery_payload = recovery.model_dump(mode="json")
    if recovery.context is None:
        recovery_payload.pop("context", None)
    return CallToolResult(
        content=[TextContent(type="text", text=visible_message)],
        # As with success, the SDK output model is the single wire-boundary
        # validation point for this structured result.
        structured_content={
            "schema_version": 1,
            "ok": False,
            "result": None,
            "error": {
                **error.to_detail(),
                "recovery": recovery_payload,
            },
        },
        is_error=True,
    )


def _invalid_arguments(
    tool_name: str,
    fields: tuple[dict[str, str], ...],
) -> CallToolResult:
    field_summary = "; ".join(f"{item['field']}: {item['message']}" for item in fields)
    message = f"Invalid arguments for {tool_name}: {field_summary}"
    remediation = [f"Match the {tool_name} inputSchema and retry."]
    if tool_name == "import_artifact" and any(item["field"] == "kind" for item in fields):
        remediation.insert(
            0,
            "For a Chrome or Torch profiler trace use kind='execution_trace'; run "
            "extract_perfetto before analyze_pytorch.",
        )
    message = f"{message} {remediation[0]}"
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structured_content={
            "schema_version": 1,
            "ok": False,
            "result": None,
            "error": {
                "code": "INVALID_ARGUMENTS",
                "message": message,
                "retryable": False,
                "details": {"fields": list(fields)},
                "remediation": remediation,
                "run_id": None,
                "recovery": {
                    "kind": "manual",
                    "safe_to_repeat_same_call": False,
                    "retry_after_ms": None,
                    "next_tool": None,
                },
            },
        },
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
    if error.details.get("invalid_configuration") is True:
        if error.details.get("next_tool") == "configure_workload":
            return RecoveryAction(
                kind="configure_workload",
                safe_to_repeat_same_call=False,
                next_tool="configure_workload",
                context=ConfigureWorkloadRecoveryContext(kind="configure_workload"),
            )
        return RecoveryAction(
            kind="manual",
            safe_to_repeat_same_call=False,
            context=ManualConfigurationRecoveryContext(
                kind="manual_configuration",
                diagnostic=str(error.details.get("diagnostic", error.message))[:500],
            ),
        )
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
    if error.details.get("next_tool") == "configure_workload":
        return RecoveryAction(
            kind="configure_workload",
            safe_to_repeat_same_call=False,
            next_tool="configure_workload",
        )
    if error.details.get("next_tool") == "workload_configuration_status":
        return RecoveryAction(
            kind="inspect_workload_configuration",
            safe_to_repeat_same_call=False,
            next_tool="workload_configuration_status",
        )
    if error.details.get("next_tool") == "start_capability_setup":
        return RecoveryAction(
            kind="start_capability_setup",
            safe_to_repeat_same_call=True,
            next_tool="start_capability_setup",
        )
    if error.details.get("next_tool") == "prepare_adapter":
        return RecoveryAction(
            kind="prepare_adapter",
            safe_to_repeat_same_call=True,
            next_tool="prepare_adapter",
        )
    if error.details.get("next_tool") == "prepare_workload_dependencies":
        return RecoveryAction(
            kind="prepare_workload_dependencies",
            safe_to_repeat_same_call=True,
            next_tool="prepare_workload_dependencies",
        )
    if error.details.get("next_tool") == "get_declared_workflow":
        return RecoveryAction(
            kind="discover_workflows",
            safe_to_repeat_same_call=False,
            next_tool="get_declared_workflow",
        )
    if error.details.get("next_tool") == "plan_capture":
        return RecoveryAction(
            kind="replan_capture",
            safe_to_repeat_same_call=False,
            next_tool="plan_capture",
        )
    if error.details.get("next_tool") == "import_artifact":
        return RecoveryAction(
            kind="import_artifact",
            safe_to_repeat_same_call=False,
            next_tool="import_artifact",
        )
    if error.details.get("next_tool") == "extract_perfetto":
        run_id = error.details.get("run_id")
        context = (
            ExtractPerfettoRecoveryContext(kind="extract_perfetto", run_id=run_id)
            if isinstance(run_id, str)
            else None
        )
        return RecoveryAction(
            kind="extract_perfetto",
            safe_to_repeat_same_call=True,
            next_tool="extract_perfetto",
            context=context,
        )
    if error.details.get("next_tool") == "list_capabilities":
        return RecoveryAction(
            kind="inspect_capabilities",
            safe_to_repeat_same_call=False,
            next_tool="list_capabilities",
        )
    if error.details.get("next_tool") == "configure_inference_server":
        return RecoveryAction(
            kind="configure_inference",
            safe_to_repeat_same_call=False,
            next_tool="configure_inference_server",
        )
    if error.details.get("next_tool") == "list_inference_configurations":
        return RecoveryAction(
            kind="configure_inference",
            safe_to_repeat_same_call=False,
            next_tool="list_inference_configurations",
        )
    if error.details.get("next_tool") == "plan_inference_scenario":
        return RecoveryAction(
            kind="replan_inference",
            safe_to_repeat_same_call=False,
            next_tool="plan_inference_scenario",
        )
    if error.details.get("next_tool") == "manual":
        return RecoveryAction(
            kind="manual",
            safe_to_repeat_same_call=False,
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
            kind="repeat_same_call",
            safe_to_repeat_same_call=True,
            retry_after_ms=250,
        )
    return RecoveryAction(
        kind="manual",
        safe_to_repeat_same_call=False,
    )


def _safe_import_path(
    project_root: Path,
    value: str,
    source_root: Literal["project", "temp"],
) -> Path:
    root = (
        project_root.resolve()
        if source_root == "project"
        else Path(tempfile.gettempdir()).resolve()
    )
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if "\x00" in value:
        raise DomainError(
            code=ErrorCode.EXECUTION_REFUSED,
            message="MCP artifact paths cannot contain NUL bytes.",
            details={"next_tool": "import_artifact"},
            remediation=(f"Provide a regular file beneath source_root={source_root!r}.",),
        )
    try:
        candidate.parent.resolve().relative_to(root)
    except ValueError as exc:
        raise DomainError(
            code=ErrorCode.EXECUTION_REFUSED,
            message="MCP artifact path is outside the selected local import root.",
            details={"next_tool": "import_artifact", "source_root": source_root},
            remediation=(
                f"Use a path beneath source_root={source_root!r}; project and temporary roots "
                "are the only MCP import roots.",
            ),
        ) from exc
    return candidate


def create_server(
    project_root: Path,
    *,
    initialize: bool = False,
    workspace_root: Path | None = None,
) -> StrictMCPServer[AppContext]:
    project_root = project_root.resolve()
    if workspace_root is not None and "\x00" in str(workspace_root):
        raise DomainError(
            ErrorCode.WORKSPACE_INVALID,
            "The MCP workspace root cannot contain NUL bytes.",
            remediation=("Provide a valid local workspace directory path.",),
        )
    selected_workspace_root = (
        workspace_root.resolve() if workspace_root is not None else project_root / ".diagnostics"
    )
    lifespan_state: list[AppContext] = []

    @asynccontextmanager
    async def lifespan(_: MCPServer[AppContext]) -> AsyncIterator[AppContext]:
        workspace: Workspace | None
        if initialize:
            workspace = Workspace.initialize(
                project_root,
                workspace_root=selected_workspace_root,
            )
        else:
            try:
                workspace = Workspace.discover(
                    project_root,
                    explicit=selected_workspace_root,
                    project_root=project_root,
                )
            except DomainError:
                if workspace_root is not None:
                    raise
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
            state.capability_setup = CapabilitySetupManager(workspace, state.capabilities)
        lifespan_state.append(state)
        try:
            yield state
        finally:
            if state.detached_captures is not None:
                await state.detached_captures.shutdown()
            if state.capability_setup is not None:
                await state.capability_setup.shutdown()
            lifespan_state.clear()

    server = StrictMCPServer(
        "flameox",
        version=__version__,
        description="Query and collect local runtime evidence.",
        instructions=(
            "Use Flameox to collect, preserve, compare, and inspect local runtime evidence "
            "when source, environment, command, and artifact provenance must be reproducible. "
            "Do not provision hosts or install undeclared packages. If list_capabilities reports "
            "a managed setup action, call start_capability_setup; it installs only the declared "
            "FlameOx optional providers into the active managed runtime, verifies them, and "
            "does not execute a workload. Host containment is not required for the agent path: "
            "plan_capture(capture_mode='auto') followed by execute_capture_plan runs the "
            "declared workload directly and records the execution limitation. Use "
            "capture_mode='managed' only when the project explicitly requires containment. "
            "Do not mutate source or GitHub, "
            "or prove static claims without runtime evidence. "
            "For a new project, call workspace_status first. If it returns "
            "WORKSPACE_NOT_FOUND, call initialize_workspace for the server's fixed project root; "
            "then repeat "
            "workspace_status. Initialization writes .diagnostics. Then call "
            "workload_configuration_status. If flameox.toml is missing, call configure_workload "
            "with the validated argument array; it writes the canonical project definition but "
            "never executes it. After that, use list_declared_workflows (no arguments lists "
            "workloads) → "
            "get_declared_workflow → list_capabilities(adapter='<selected adapter>') → "
            "start_capability_setup (when the scoped result names setup_adapters) → "
            "prepare_adapter "
            "(for an unapproved installed third-party "
            "adapter) → prepare_workload_dependencies (when a named workload declares missing "
            "Python distributions) → list_capabilities(adapter='<selected adapter>') → "
            "plan_capture(preflight_mode='auto', "
            "capture_mode='auto') "
            "→ execute_capture_plan "
            "for short work, "
            "start_detached_capture for long work -> get_detached_capture -> get_run -> analyze. "
            "For existing evidence: list_runs and list_artifacts expose artifact_kinds; use "
            "extract_pyperf and query_measurements for benchmark_samples, "
            "extract_python_startup for Python startup/import evidence, extract_pytest for "
            "test phases, fixtures, workers, and failure latency, analyze_memory for "
            "memory profiles, analyze_execution for coverage, and the other analyze_* tools "
            "only for their documented artifact kinds. Imported Torch traces require "
            "extract_perfetto before analyze_pytorch; absent normalized rows are recovery, not "
            "an empty operator report. Then use get_evidence, record_analysis, "
            "or record_finding. Poll get_capability_setup after setup starts. A "
            "synchronous consumed capture plan is never retryable; detached starts are "
            "retryable only with the same idempotency key. For inference work, use "
            "configure_inference_server and configure_inference_scenario, then "
            "list_inference_configurations -> plan_inference_scenario -> "
            "run_inference_scenario with expected_plan_id -> list_inference_requests and "
            "query_measurements. Create an unprofiled measurement run before a diagnostic "
            "profile; plan_inference_profile -> run_inference_profile requires that run's "
            "measurement_run_id."
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

            workspace = Workspace.initialize(
                state.project_root,
                workspace_root=selected_workspace_root,
            )
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
            state.capability_setup = CapabilitySetupManager(workspace, state.capabilities)
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

    @server.tool(name="workload_configuration_status", annotations=READ_ONLY)
    async def workload_configuration_status_tool(
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[WorkloadConfigurationStatus]]:
        """Inspect flameox.toml without writing or executing anything.

        Use this after workspace initialization to decide whether to call configure_workload
        or list_declared_workflows. Invalid configuration is reported without replacement.
        """
        try:
            result = WorkloadService(
                ctx.request_context.lifespan_context.require_workspace()
            ).configuration_status()
            return _success(result, f"Workload configuration status: {result.status}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="configure_workload", annotations=CONFIGURE)
    async def configure_workload_tool(
        name: Annotated[
            str,
            Field(
                min_length=1,
                max_length=100,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
                description=(
                    "Stable named workload identifier used by later discovery and planning."
                ),
            ),
        ],
        operation: Literal["create", "replace"],
        argv: Annotated[
            tuple[str, ...],
            Field(
                min_length=1,
                max_length=1_024,
                description="Argument array; no shell parsing or command string is accepted.",
            ),
        ],
        ctx: Context[AppContext],
        cwd: Annotated[str, Field(min_length=1, max_length=4_096)] = ".",
        timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300,
        parameters: Annotated[
            dict[str, tuple[Scalar, ...]],
            Field(max_length=128),
        ]
        | None = None,
        environment: Annotated[dict[str, str], Field(max_length=128)] | None = None,
        oracle: WorkloadOracleConfig | None = None,
        requirements: WorkloadRequirementsConfig | None = None,
        writable_paths: Annotated[tuple[str, ...], Field(max_length=16)] = (),
        identity: WorkloadIdentityConfig | None = None,
        expected_configuration_id: Annotated[
            str,
            Field(pattern=r"^sha256:[0-9a-f]{64}$"),
        ]
        | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[WorkloadConfigurationResult]]:
        """Write one validated named workload without executing it.

        Writes only the project workload configuration, preserves existing workloads and
        experiments, and returns the next discovery step. It never executes the command.
        Use operation='replace' with the current configuration_id to update an existing workload.
        """
        try:
            request = ConfigureWorkloadRequest(
                name=name,
                operation=operation,
                config=WorkloadConfig(
                    argv=argv,
                    cwd=cwd,
                    timeout_seconds=timeout_seconds,
                    parameters=parameters or {},
                    environment=environment or {},
                    oracle=oracle,
                    requirements=requirements or WorkloadRequirementsConfig(),
                    writable_paths=writable_paths,
                    identity=identity or WorkloadIdentityConfig(),
                ),
                expected_configuration_id=expected_configuration_id,
            )
            result = WorkloadService(
                ctx.request_context.lifespan_context.require_workspace()
            ).configure(request)
            return _success(
                result,
                f"Workload {name!r} is {result.action}; call {result.next_tool} next.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="configure_inference_server", annotations=CONFIGURE)
    async def configure_inference_server_tool(
        name: Annotated[
            str,
            Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
        ],
        operation: Literal["create", "replace"],
        mode: Annotated[
            Literal["managed", "existing_local"],
            Field(description="managed requires workload; existing_local must use loopback."),
        ],
        model: Annotated[str, Field(min_length=1, max_length=500)],
        ctx: Context[AppContext],
        provider: Literal["vllm", "sglang"] = "vllm",
        benchmark_python: Annotated[
            str | None,
            Field(description="Absolute SGLang Python launcher; required only for sglang."),
        ] = None,
        workload: Annotated[
            str | None,
            Field(description="Declared workload that starts vLLM; required for managed mode."),
        ] = None,
        base_url: Annotated[
            str,
            Field(description="Loopback-only OpenAI-compatible server URL."),
        ] = "http://127.0.0.1:8000",
        model_revision: Annotated[str | None, Field(max_length=200)] = None,
        tokenizer: Annotated[str | None, Field(max_length=500)] = None,
        tokenizer_revision: Annotated[str | None, Field(max_length=200)] = None,
        quantization: Annotated[str | None, Field(max_length=100)] = None,
        expected_configuration_id: Annotated[
            str | None, Field(pattern=r"^sha256:[0-9a-f]{64}$")
        ] = None,
    ) -> Annotated[CallToolResult, ToolPayload[InferenceConfigurationResult]]:
        """Create or replace one validated vLLM server declaration without starting it."""
        try:
            result = WorkloadService(
                ctx.request_context.lifespan_context.require_workspace()
            ).configure_inference_server(
                ConfigureInferenceServerRequest(
                    name=name,
                    operation=operation,
                    expected_configuration_id=expected_configuration_id,
                    config=InferenceServerConfig(
                        provider=provider,
                        benchmark_python=benchmark_python,
                        mode=mode,
                        workload=workload,
                        base_url=base_url,
                        model=model,
                        model_revision=model_revision,
                        tokenizer=tokenizer,
                        tokenizer_revision=tokenizer_revision,
                        quantization=quantization,
                    ),
                )
            )
            return _success(result, f"Inference server {name!r} is {result.action}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="configure_inference_scenario", annotations=CONFIGURE)
    async def configure_inference_scenario_tool(
        name: Annotated[
            str,
            Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
        ],
        operation: Literal["create", "replace"],
        server_name: Annotated[str, Field(min_length=1, max_length=100)],
        provider: Literal["aiperf", "vllm_bench", "sglang_bench"],
        ctx: Context[AppContext],
        endpoint_type: Literal["chat", "completions"] = "chat",
        streaming: bool = True,
        trace_artifact_id: Annotated[
            str | None,
            Field(
                pattern=r"^sha256:[0-9a-f]{64}$",
                description="Mooncake JSONL artifact; supported only by the AIPerf provider.",
            ),
        ] = None,
        num_prompts: Annotated[int, Field(gt=0, le=10_000_000)] = 1,
        concurrency: Annotated[int | None, Field(gt=0, le=100_000)] = None,
        request_rate: Annotated[float | None, Field(gt=0, le=1_000_000)] = None,
        burstiness: Annotated[
            float | None,
            Field(gt=0, le=1_000_000, description="Requires request_rate when supplied."),
        ] = None,
        warmup_request_count: Annotated[int, Field(ge=0, le=1_000_000)] = 0,
        seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0,
        speedup_ratio: Annotated[float, Field(gt=0, le=100)] = 1.0,
        semantic_oracle_workload: Annotated[
            str | None,
            Field(
                max_length=100,
                description="Declared workload with an oracle contract, not an ordinary workload.",
            ),
        ] = None,
        random_input_len: Annotated[int | None, Field(gt=0, le=1_000_000)] = None,
        random_output_len: Annotated[int | None, Field(gt=0, le=1_000_000)] = None,
        random_range_ratio: Annotated[float | None, Field(gt=0, le=1)] = None,
        expected_configuration_id: Annotated[
            str | None, Field(pattern=r"^sha256:[0-9a-f]{64}$")
        ] = None,
    ) -> Annotated[CallToolResult, ToolPayload[InferenceConfigurationResult]]:
        """Create or replace one typed inference replay scenario without executing it."""
        try:
            result = WorkloadService(
                ctx.request_context.lifespan_context.require_workspace()
            ).configure_inference_scenario(
                ConfigureInferenceScenarioRequest(
                    name=name,
                    operation=operation,
                    expected_configuration_id=expected_configuration_id,
                    config=InferenceScenarioConfig(
                        server=server_name,
                        provider=provider,
                        endpoint_type=endpoint_type,
                        streaming=streaming,
                        trace_artifact_id=trace_artifact_id,
                        num_prompts=num_prompts,
                        concurrency=concurrency,
                        request_rate=request_rate,
                        burstiness=burstiness,
                        warmup_request_count=warmup_request_count,
                        seed=seed,
                        speedup_ratio=speedup_ratio,
                        semantic_oracle_workload=semantic_oracle_workload,
                        random_input_len=random_input_len,
                        random_output_len=random_output_len,
                        random_range_ratio=random_range_ratio,
                    ),
                )
            )
            return _success(result, f"Inference scenario {name!r} is {result.action}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_inference_configurations", annotations=READ_ONLY)
    async def list_inference_configurations_tool(
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[InferenceConfigurationList]]:
        """List declared inference servers and scenarios without probing or execution."""
        try:
            result = WorkloadService(
                ctx.request_context.lifespan_context.require_workspace()
            ).list_inference()
            return _success(
                result,
                f"Found {len(result.servers)} servers and {len(result.scenarios)} scenarios.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="plan_inference_scenario", annotations=READ_ONLY)
    async def plan_inference_scenario_tool(
        scenario_name: Annotated[str, Field(min_length=1, max_length=100)],
        ctx: Context[AppContext],
        timeout_seconds: Annotated[float | None, Field(gt=0, le=86_400)] = None,
    ) -> Annotated[CallToolResult, ToolPayload[InferenceReplayPlan]]:
        """Preflight a managed or existing-local server and construct a typed replay plan."""
        try:
            result = InferenceReplayService(
                ctx.request_context.lifespan_context.require_workspace()
            ).plan(scenario_name, timeout_seconds=timeout_seconds)
            return _success(
                result,
                f"Planned inference scenario {scenario_name!r}. Next tool: "
                "run_inference_scenario with the same scenario_name and "
                f"expected_plan_id={result.plan_id!r}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="run_inference_scenario", annotations=ADDITIVE)
    async def run_inference_scenario_tool(
        scenario_name: Annotated[str, Field(min_length=1, max_length=100)],
        ctx: Context[AppContext],
        timeout_seconds: Annotated[float | None, Field(gt=0, le=86_400)] = None,
        expected_plan_id: Annotated[str | None, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = None,
    ) -> Annotated[CallToolResult, ToolPayload[InferenceReplayResult]]:
        """Plan and execute one bounded replay against a managed or existing-local server."""
        try:
            service = InferenceReplayService(
                ctx.request_context.lifespan_context.require_workspace()
            )
            result = await service.run(
                service.plan(
                    scenario_name,
                    timeout_seconds=timeout_seconds,
                    expected_plan_id=expected_plan_id,
                )
            )
            return _success(result, f"Completed inference scenario {scenario_name!r}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_inference_requests", annotations=READ_ONLY)
    async def list_inference_requests_tool(
        run_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
        limit: Annotated[int | None, Field(ge=1, le=1_000)] = None,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[InferenceRequestQueryResult]]:
        """Page through bounded normalized inference requests without prompt or error text."""
        try:
            result = EvidenceQueryService(
                ctx.request_context.lifespan_context.require_workspace()
            ).inference_requests(run_id=run_id, limit=limit, cursor=cursor)
            return _success(result, f"Returned {result.returned} inference requests.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="plan_inference_profile", annotations=READ_ONLY)
    async def plan_inference_profile_tool(
        server_name: Annotated[str, Field(min_length=1, max_length=100)],
        profiler: Literal["torch_profiler", "nsight_systems"],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[InferenceProfilingPlan]]:
        """Build a diagnostic-only profile plan for one managed vLLM server."""
        try:
            result = InferenceProfilingService(
                ctx.request_context.lifespan_context.require_workspace()
            ).plan(
                server_name,
                profiler=profiler,
            )
            return _success(
                result,
                f"Planned {profiler} capture for {server_name!r}. Next tool: "
                "run_inference_profile with the same server_name and profiler, a compatible "
                "scenario_name, a successful unprofiled measurement_run_id, and "
                f"expected_plan_id={result.plan_id!r}.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="run_inference_profile", annotations=ADDITIVE)
    async def run_inference_profile_tool(
        server_name: Annotated[str, Field(min_length=1, max_length=100)],
        scenario_name: Annotated[str, Field(min_length=1, max_length=100)],
        profiler: Literal["torch_profiler", "nsight_systems"],
        measurement_run_id: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                description="Successful compatible unprofiled inference run to link.",
            ),
        ],
        ctx: Context[AppContext],
        timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300,
        expected_plan_id: Annotated[str | None, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = None,
    ) -> Annotated[CallToolResult, ToolPayload[InferenceProfilingResult]]:
        """Run one diagnostic-only profile window against a managed vLLM server."""
        try:
            service = InferenceProfilingService(
                ctx.request_context.lifespan_context.require_workspace()
            )
            plan = service.plan(
                server_name,
                profiler=profiler,
                expected_plan_id=expected_plan_id,
            )
            result = await service.capture(
                plan,
                scenario_name=scenario_name,
                measurement_run_id=measurement_run_id,
                timeout_seconds=timeout_seconds,
            )
            return _success(result, f"Completed {profiler} diagnostic capture.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_capabilities", annotations=READ_ONLY)
    async def list_capabilities_tool(
        ctx: Context[AppContext],
        mode: Literal["passive", "active_cached", "active_refresh"] = "passive",
        adapter: Annotated[
            str | None,
            Field(
                min_length=1,
                max_length=100,
                description=(
                    "Selected capture adapter whose setup recommendation should be returned. "
                    "Omit this for the read-only global inventory."
                ),
            ),
        ] = None,
    ) -> Annotated[CallToolResult, ToolPayload[CapabilityList]]:
        """List capabilities and setup actions scoped to a selected capture adapter.

        Omit adapter for a complete read-only inventory. In that mode, the per-capability setup
        fields and available_setup_adapters are informational only; select an adapter and call
        this tool again before mutating the managed environment. Managed setup never executes a
        workload.
        """
        service = ctx.request_context.lifespan_context.capabilities
        if mode == "passive":
            result = service.list() if adapter is None else service.list_for_adapter(adapter)
        else:
            result = await service.list_active(
                refresh=mode == "active_refresh",
                recommendation_adapter=adapter,
            )
        return _success(
            result,
            f"Found {sum(item.status.value == 'available' for item in result.capabilities)} of "
            f"{len(result.capabilities)} available capabilities. "
            + (
                "Call start_capability_setup for: " + ", ".join(result.setup_adapters) + "."
                if result.setup_adapters
                else (
                    "Call prepare_adapter for the reported adapter/distribution pairs."
                    if result.setup_third_party_adapters
                    else (
                        "Select an adapter and call list_capabilities(adapter=...) before setup."
                        if adapter is None
                        else "No managed capability setup is pending for this adapter."
                    )
                )
            ),
        )

    @server.tool(annotations=CONFIGURE)
    async def start_capability_setup(
        adapters: Annotated[
            tuple[
                Literal[
                    "coverage",
                    "memray",
                    "perfetto",
                    "py-spy",
                    "pytest",
                    "torch.profiler",
                    "toxiproxy",
                ],
                ...,
            ],
            Field(
                min_length=1,
                max_length=6,
                description="Managed adapters from list_capabilities to install or stage.",
            ),
        ],
        idempotency_key: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                description="Stable key for replaying this exact request.",
            ),
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[OperationStatus]]:
        """Start detached capability provisioning and return its durable operation ID.

        Use the same idempotency key to reconnect after a lost request. Poll
        get_capability_setup for named phases, item outcomes, and the terminal receipt;
        cancel_capability_setup requests cleanup of owned work.
        """
        try:
            result = await ctx.request_context.lifespan_context.capability_setup_service().start(
                tuple(adapters), idempotency_key
            )
            return _success(
                result,
                f"Started capability setup {result.operation_id} ({result.state}).",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_capability_setup", annotations=READ_ONLY)
    async def get_capability_setup(
        operation_id: Annotated[str, Field(min_length=4, max_length=100)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[OperationStatus]]:
        """Read durable capability setup state after the original request disappears."""
        try:
            result = await ctx.request_context.lifespan_context.capability_setup_service().status(
                operation_id
            )
            return _success(
                result,
                f"Capability setup {operation_id} is {result.state} ({result.phase}).",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="cancel_capability_setup", annotations=CONFIGURE)
    async def cancel_capability_setup(
        operation_id: Annotated[str, Field(min_length=4, max_length=100)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[OperationStatus]]:
        """Request cancellation and cleanup of a server-owned capability setup operation."""
        try:
            result = await ctx.request_context.lifespan_context.capability_setup_service().cancel(
                operation_id
            )
            return _success(result, f"Capability setup {operation_id} is {result.state}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="prepare_adapter", annotations=CONFIGURE)
    async def prepare_adapter_tool(
        adapter: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
                description="Third-party adapter name returned by list_capabilities.",
            ),
        ],
        distribution: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
                description="Exact installed distribution name reported for this adapter.",
            ),
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[AdapterPreparationResult]]:
        """Approve one installed third-party adapter by exact installed package identity.

        This records agent-created provenance under the workspace lock; it does not install a
        package, import plugin code, or execute a workload. Call list_capabilities again.
        """
        try:
            result = ctx.request_context.lifespan_context.capabilities.prepare_adapter(
                adapter,
                distribution,
            )
            return _success(
                result,
                f"Prepared adapter {adapter!r} from {distribution!r}; call list_capabilities next.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="prepare_workload_dependencies", annotations=CONFIGURE)
    async def prepare_workload_dependencies_tool(
        workload_name: Annotated[
            str,
            Field(
                min_length=1,
                max_length=100,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
                description=(
                    "Declared workload whose Python distribution requirements are installed."
                ),
            ),
        ],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[WorkloadDependencySetupResult]]:
        """Install declared workload Python distributions into the active managed runtime.

        Only requirements already present in the named workload's flameox.toml definition are
        installed. The tool never executes a workload. The result includes an active preflight and
        tells the agent whether to plan or inspect a remaining host capability.
        """
        try:
            result = await WorkloadDependencyService(
                ctx.request_context.lifespan_context.require_workspace(),
                broker=ctx.request_context.lifespan_context.capabilities.broker,
            ).prepare(workload_name)
            return _success(
                result,
                f"Prepared dependencies for {workload_name!r}; call "
                f"{result.next_tool or 'list_capabilities'} next.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="list_declared_workflows", annotations=READ_ONLY)
    async def list_declared_workflows_tool(
        ctx: Context[AppContext],
        kind: Literal["workload", "experiment", "fault_experiment"] = "workload",
        limit: Annotated[int, Field(ge=1, le=100)] = 50,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[DeclaredWorkflowList]]:
        """Discover declared workflows before planning; this never runs them.

        With no arguments, list workloads. Pass kind='experiment' or
        kind='fault_experiment' to list declared experiments.
        """
        try:
            result = WorkloadService(
                ctx.request_context.lifespan_context.require_workspace()
            ).list_declared(
                kind=kind,
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
        kind: Literal["workload", "experiment", "fault_experiment"],
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
                description="Current declared workload name from list_declared_workflows.",
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
        preflight_mode: Literal["auto", "passive", "active"] = "auto",
        capture_mode: Literal["auto", "managed", "trusted_local"] = "auto",
        external_context: ExternalExecutionContext | None = None,
        torch_profiler_options: TorchProfilerCaptureOptions | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[CapturePlan]]:
        """Bind one current capture without running it.

        The default auto mode runs the declared workload directly in the local environment and
        records that no enforced descendant containment was used. Use managed only when the
        project policy explicitly requires containment, and use trusted_local to request the
        same direct local execution explicitly. This tool never executes the workload.
        """
        try:
            execution_policy = (
                ExecutionPolicy.APPROVED_AGENT
                if capture_mode == "managed"
                else ExecutionPolicy.TRUSTED_LOCAL
            )
            result = await ctx.request_context.lifespan_context.capture_service().plan(
                workload_name=workload_name,
                adapter=adapter,
                parameters=parameters,
                execution_policy=execution_policy,
                preflight_mode=preflight_mode,
                external_context=external_context,
                adapter_options=(
                    torch_profiler_options.model_dump(mode="json")
                    if torch_profiler_options is not None
                    else None
                ),
            )
            return _success(
                result,
                f"Planned {adapter} capture with {result.containment} containment.",
            )
        except DomainError as error:
            if capture_mode == "managed" and error.code in {
                ErrorCode.CAPABILITY_UNAVAILABLE,
                ErrorCode.EXECUTION_REFUSED,
            }:
                error.details.update({"next_tool": "plan_capture", "capture_mode": "auto"})
                error.remediation = (
                    "Re-plan with capture_mode='auto' to run directly and record the missing "
                    "containment as an execution limitation.",
                    *error.remediation,
                )
            return _failure(error)

    @server.tool(name="execute_capture_plan", annotations=EXECUTE)
    async def execute_capture_plan_tool(
        plan_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[CaptureReceipt]]:
        """Run one current plan with side effects; the token is single-use, then get_run."""
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
                limitation_details=result.run.limitation_details,
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
        """Start one current plan once; reconnect by run_id without keeping this call open."""
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
            Field(min_length=1, description="Current experiment from list_declared_workflows."),
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
                execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
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
        """Execute all current trials from one single-use plan, then inspect get_experiment."""
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

    @server.tool(name="plan_fault_experiment", annotations=ADDITIVE)
    async def plan_fault_experiment_tool(
        experiment_name: Annotated[str, Field(min_length=1)],
        investigation_id: Annotated[str, Field(min_length=1)],
        parameters: dict[str, Scalar],
        ctx: Context[AppContext],
        hypothesis_id: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[FaultExperimentPlan]]:
        """Bind a declared loopback Toxiproxy experiment and its exact toxic scenarios."""
        try:
            result = await FaultExperimentService(
                ctx.request_context.lifespan_context.require_workspace()
            ).plan(
                experiment_name=experiment_name,
                investigation_id=investigation_id,
                hypothesis_id=hypothesis_id,
                parameter_overrides=parameters,
                execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
            )
            return _success(result, f"Planned fault experiment {result.plan_id}.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="run_fault_experiment", annotations=EXECUTE)
    async def run_fault_experiment_tool(
        plan_id: Annotated[str, Field(min_length=1)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[FaultExperimentResult]]:
        """Run every baseline and declared treatment through its managed loopback proxy."""
        try:

            async def report(completed: float, total: float, message: str) -> None:
                await ctx.report_progress(completed, total, message)

            service = FaultExperimentService(
                ctx.request_context.lifespan_context.require_workspace()
            )
            result = await service.run(plan_id, progress=report)
            return _success(
                result,
                f"Recorded {len(result.trials)} fault trials for "
                f"{result.experiment.experiment_id}.",
                resource_links=(
                    ResourceLink(
                        name=f"Fault experiment {result.experiment.experiment_id}",
                        uri=f"flameox://experiments/{result.experiment.experiment_id}",
                        description="Immutable fault experiment protocol and trial provenance.",
                        mime_type="application/json",
                    ),
                ),
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_fault_experiment", annotations=READ_ONLY)
    async def get_fault_experiment_tool(
        result_id: Annotated[str, Field(min_length=1)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[FaultExperimentResult]]:
        """Read one immutable completed fault experiment result."""
        try:
            result = FaultExperimentService(
                ctx.request_context.lifespan_context.require_workspace()
            ).show(result_id)
            return _success(result, f"Fault experiment result {result_id}.")
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
                description=(
                    "Artifact path relative to source_root, or an absolute path already inside "
                    "that bounded root."
                )
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
        media_type: Annotated[
            str | None,
            Field(
                max_length=200,
                description=(
                    "Explicit media type for formats whose encoding cannot be inferred, such as "
                    "application/x-protobuf OTLP traces."
                ),
            ),
        ] = None,
        source_root: Literal["project", "temp"] = "project",
        producer: Annotated[
            Literal[
                "auto",
                "torch.profiler",
                "perfetto",
                "py-spy",
                "memray",
                "coverage",
                "pyperf",
                "pytest",
                "aiperf",
                "vllm_bench",
                "mooncake",
            ],
            Field(
                description=(
                    "Producer identity. Use auto for common trace detection; use "
                    "torch.profiler when importing an ambiguous Torch trace, and declare the "
                    "maintained inference provider for imported replay/result artifacts."
                )
            ),
        ] = "auto",
        producer_version: Annotated[
            str | None,
            Field(description="Optional producer version, at most 100 characters.", max_length=100),
        ] = None,
    ) -> Annotated[CallToolResult, ToolPayload[ImportReceipt]]:
        """Import one project-local artifact and preserve producer identity.

        Chrome traces with Torch profiler markers are identified automatically. Use
        kind='execution_trace' for Chrome/Torch traces, then run extract_perfetto before
        analyze_pytorch. If detection is ambiguous, set producer='torch.profiler'.
        """
        try:
            state = ctx.request_context.lifespan_context
            workspace = state.require_workspace()
            request = ImportArtifactRequest(
                path=_safe_import_path(state.project_root, path, source_root),
                kind=kind,
                media_type=media_type,
                sensitivity=sensitivity,
                producer=None if producer == "auto" else producer,
                producer_version=producer_version,
                allow_external_path=source_root == "temp",
            )
            result = await run_atomic_thread(
                lambda: ImportService(workspace).import_artifact(request)
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
        """Return bounded metadata and an opaque resource URI, never a host path or bytes."""
        try:
            result = ArtifactService(ctx.request_context.lifespan_context.require_workspace()).get(
                artifact_id
            )
            return _success(
                result,
                f"Artifact {artifact_id} is metadata-only; read {result.resource_uri} for the "
                "canonical resource.",
                resource_links=(
                    ResourceLink(
                        name=f"Artifact {artifact_id}",
                        uri=result.resource_uri,
                        description="Opaque artifact metadata resource; no host path is exposed.",
                        mime_type="application/json",
                    ),
                ),
            )
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
        """Summarize normalized Perfetto evidence from a torch.profiler run or artifact.

        This read-only tool never extracts implicitly. If normalized rows are absent, follow
        the typed recovery result and call extract_perfetto for the exact run.
        """
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

    @server.tool(name="analyze_accelerator_launches", annotations=READ_ONLY)
    async def analyze_accelerator_launches_tool(
        run_or_artifact: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                description="Run or artifact with normalized Perfetto or Nsight trace events.",
            ),
        ],
        limit: Annotated[
            StrictInt,
            Field(ge=1, le=1_000, description="Maximum regions and kernel names to return."),
        ],
        ctx: Context[AppContext],
        comparison_run_or_artifact: Annotated[
            str | None,
            Field(min_length=1, max_length=200),
        ] = None,
        phase: Annotated[str | None, Field(min_length=1, max_length=200)] = None,
    ) -> Annotated[CallToolResult, ToolPayload[AcceleratorLaunchAnalysisResult]]:
        """Analyze observed runtime launches, graph launches, kernels, and idle gaps."""
        try:
            workspace = ctx.request_context.lifespan_context.require_workspace()
            await ctx.report_progress(0, 2, "Accelerator trace snapshot pinned")
            result = await Catalog(workspace).run_interruptible(
                lambda snapshot: RecipeService(
                    workspace,
                    snapshot=snapshot,
                ).accelerator_launches(
                    run_or_artifact,
                    comparison_input_id=comparison_run_or_artifact,
                    phase=phase,
                    limit=limit,
                ),
                query_name="analyze_accelerator_launches",
            )
            await ctx.report_progress(1, 2, "Accelerator launch query complete")
            await ctx.report_progress(2, 2, "Accelerator launch result ready")
            return _success(result, f"Returned {result.returned} launch regions.")
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

    @server.tool(name="extract_otlp_trace", annotations=ADDITIVE)
    async def extract_otlp_trace_tool(
        run_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
        artifact_id: Annotated[str | None, Field(min_length=1, max_length=200)] = None,
    ) -> Annotated[CallToolResult, ToolPayload[OtlpExtractionResult]]:
        """Normalize an explicitly registered OTLP file artifact into evidence tables."""
        try:
            result = await run_atomic_thread(
                lambda: OtlpTraceService(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract_otlp_trace(run_id, artifact_id)
            )
            return _success(result, f"Normalized {result.span_count} OTLP spans.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_operation_window", annotations=READ_ONLY)
    async def get_operation_window_tool(
        artifact_id: Annotated[str, Field(min_length=1, max_length=200)],
        start_ns: Annotated[int, Field(ge=0)],
        end_ns: Annotated[int, Field(gt=0)],
        ctx: Context[AppContext],
        trace_id: Annotated[str | None, Field(min_length=1, max_length=100)] = None,
        limit: Annotated[int | None, Field(ge=1, le=1_000)] = None,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[LifecycleQueryResult]]:
        """Return normalized OTLP spans overlapping a bounded time range."""
        try:
            result = await run_atomic_thread(
                lambda: LifecycleEvidenceService(
                    ctx.request_context.lifespan_context.require_workspace()
                ).get_operation_window(
                    artifact_id=artifact_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    trace_id=trace_id,
                    limit=limit,
                    cursor=cursor,
                )
            )
            return _success(result, f"Returned {result.returned} operation spans.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_operation_transitions", annotations=READ_ONLY)
    async def get_operation_transitions_tool(
        artifact_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
        trace_id: Annotated[str | None, Field(min_length=1, max_length=100)] = None,
        max_depth: Annotated[int, Field(ge=0, le=32)] = 8,
        limit: Annotated[int | None, Field(ge=1, le=1_000)] = None,
        cursor: str | None = None,
    ) -> Annotated[CallToolResult, ToolPayload[LifecycleQueryResult]]:
        """Return bounded parent/child transitions without interpreting causality."""
        try:
            result = await run_atomic_thread(
                lambda: LifecycleEvidenceService(
                    ctx.request_context.lifespan_context.require_workspace()
                ).get_operation_transitions(
                    artifact_id=artifact_id,
                    trace_id=trace_id,
                    max_depth=max_depth,
                    limit=limit,
                    cursor=cursor,
                )
            )
            return _success(result, f"Returned {result.returned} operation transitions.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="find_repeated_operation_sequences", annotations=READ_ONLY)
    async def find_repeated_operation_sequences_tool(
        artifact_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
        minimum_repetitions: Annotated[int, Field(ge=2, le=100)] = 2,
        limit: Annotated[int | None, Field(ge=1, le=1_000)] = None,
        cursor: Annotated[str | None, Field(max_length=4_096)] = None,
    ) -> Annotated[CallToolResult, ToolPayload[LifecycleQueryResult]]:
        """Return repeated span signatures as bounded derived evidence."""
        try:
            result = await run_atomic_thread(
                lambda: LifecycleEvidenceService(
                    ctx.request_context.lifespan_context.require_workspace()
                ).find_repeated_operation_sequences(
                    artifact_id=artifact_id,
                    minimum_repetitions=minimum_repetitions,
                    limit=limit,
                    cursor=cursor,
                )
            )
            return _success(result, f"Returned {result.returned} repeated operation spans.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_lifecycle_gaps", annotations=READ_ONLY)
    async def get_lifecycle_gaps_tool(
        artifact_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
        limit: Annotated[int | None, Field(ge=1, le=1_000)] = None,
    ) -> Annotated[CallToolResult, ToolPayload[LifecycleQueryResult]]:
        """Return explicit timestamp, identity, and missing-parent evidence."""
        try:
            result = await run_atomic_thread(
                lambda: LifecycleEvidenceService(
                    ctx.request_context.lifespan_context.require_workspace()
                ).get_lifecycle_gaps(artifact_id=artifact_id, limit=limit)
            )
            return _success(result, f"Returned {result.returned} lifecycle gaps.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="get_process_snapshot", annotations=READ_ONLY)
    async def get_process_snapshot_tool(
        run_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
        phase: Annotated[str | None, Field(min_length=1, max_length=50)] = None,
        limit: Annotated[int | None, Field(ge=1, le=1_000)] = None,
    ) -> Annotated[CallToolResult, ToolPayload[LifecycleQueryResult]]:
        """Return bounded privacy-limited process observations for one run."""
        try:
            result = await run_atomic_thread(
                lambda: LifecycleEvidenceService(
                    ctx.request_context.lifespan_context.require_workspace()
                ).get_process_snapshot(run_id=run_id, phase=phase, limit=limit)
            )
            return _success(result, f"Returned {result.returned} process observations.")
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
            result = await run_atomic_thread(
                lambda: IntegrityService(
                    ctx.request_context.lifespan_context.require_workspace()
                ).validate(full=mode == "full")
            )
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
            result = await run_atomic_thread(
                lambda: PyPerfExtractor(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract(run_id)
            )
            return _success(
                result,
                f"Extracted {result.measurement_count} measured values.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_benchmark_samples", annotations=ADDITIVE)
    async def extract_benchmark_samples_tool(
        run_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[BenchmarkSamplesExtractionResult]]:
        """Extract raw accelerator benchmark samples with explicit timing semantics."""
        try:
            result = await run_atomic_thread(
                lambda: BenchmarkSamplesExtractor(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract(run_id)
            )
            return _success(
                result,
                f"Extracted {result.measurement_count} measured values and "
                f"{result.warmup_count} warmups.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_inference_trace", annotations=ADDITIVE)
    async def extract_inference_trace_tool(
        run_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[InferenceExtractionResult]]:
        """Extract bounded prompt-free Mooncake request schedule evidence."""
        try:
            result = await run_atomic_thread(
                lambda: InferenceArtifactExtractor(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract_trace(run_id)
            )
            return _success(result, f"Extracted {result.request_count} inference requests.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_inference_result", annotations=ADDITIVE)
    async def extract_inference_result_tool(
        run_id: Annotated[str, Field(min_length=1, max_length=200)],
        provider: Literal["aiperf", "vllm_bench", "sglang_bench"],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[InferenceExtractionResult]]:
        """Extract prompt-free AIPerf requests or vLLM aggregate measurements."""
        try:
            result = await run_atomic_thread(
                lambda: (
                    InferenceArtifactExtractor(
                        ctx.request_context.lifespan_context.require_workspace()
                    ).extract_aiperf_result(run_id)
                    if provider == "aiperf"
                    else InferenceArtifactExtractor(
                        ctx.request_context.lifespan_context.require_workspace()
                    ).extract_sglang_result(run_id)
                    if provider == "sglang_bench"
                    else InferenceArtifactExtractor(
                        ctx.request_context.lifespan_context.require_workspace()
                    ).extract_vllm_result(run_id)
                )
            )
            count = result.request_count or result.measurement_count
            return _success(result, f"Extracted {count} inference evidence rows.")
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_python_startup", annotations=ADDITIVE)
    async def extract_python_startup_tool(
        run_id: str,
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[PythonStartupExtractionResult]]:
        """Extract repeated startup, peak RSS, and package-grouped import evidence."""
        try:
            result = await run_atomic_thread(
                lambda: PythonStartupExtractor(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract(run_id)
            )
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
            result = await run_atomic_thread(
                lambda: PytestExtractor(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract(run_id)
            )
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
            result = await run_atomic_thread(
                lambda: CoverageExtractor(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract(run_id)
            )
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
            result = await run_atomic_thread(
                lambda: MemrayExtractor(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract(run_id)
            )
            return _success(
                result,
                f"Peak memory was {result.peak_memory_bytes} bytes; "
                f"{result.retained_end_bytes} bytes remained at end.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_perfetto", annotations=ADDITIVE)
    async def extract_perfetto_tool(
        run_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
        artifact_id: Annotated[str | None, Field(min_length=1, max_length=200)] = None,
    ) -> Annotated[CallToolResult, ToolPayload[PerfettoExtractionResult]]:
        """Run versioned curated queries through a configured local Trace Processor."""
        try:
            result = await PerfettoExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id, artifact_id=artifact_id)
            return _success(
                result,
                f"Extracted {result.slice_count} slices into "
                f"{result.frame_count} frame aggregates.",
            )
        except DomainError as error:
            return _failure(error)

    @server.tool(name="extract_nsight_systems", annotations=ADDITIVE)
    async def extract_nsight_systems_tool(
        run_id: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context[AppContext],
    ) -> Annotated[CallToolResult, ToolPayload[NsightSystemsExtractionResult]]:
        """Extract curated evidence from an imported official Nsight Systems SQLite export."""
        try:
            result = await NsightSystemsExtractor(
                ctx.request_context.lifespan_context.require_workspace()
            ).extract(run_id)
            return _success(
                result,
                f"Extracted {result.event_count} Nsight Systems events.",
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
            result = await run_atomic_thread(
                lambda: ObservationExtractor(
                    ctx.request_context.lifespan_context.require_workspace()
                ).extract(run_id)
            )
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


def run_server(
    project_root: Path,
    *,
    initialize: bool = False,
    workspace_root: Path | None = None,
) -> None:
    create_server(
        project_root,
        initialize=initialize,
        workspace_root=workspace_root,
    ).run()
