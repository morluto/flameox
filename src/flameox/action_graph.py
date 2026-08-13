from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, JsonValue, StrictInt, ValidationError, model_validator

from flameox.models import ContractModel


class ActionId(StrEnum):
    """Stable product-level identities for executable agent workflow edges."""

    INITIALIZE_WORKSPACE = "workspace.initialize"
    INSPECT_WORKLOAD_CONFIGURATION = "workload.configuration.inspect"
    CONFIGURE_WORKLOAD = "workload.configure"
    INSPECT_CAPABILITIES = "capabilities.inspect"
    START_CAPABILITY_SETUP = "capabilities.setup.start"
    GET_CAPABILITY_SETUP = "capabilities.setup.status"
    PREPARE_ADAPTER = "adapter.prepare"
    PREPARE_WORKLOAD_DEPENDENCIES = "workload.dependencies.prepare"
    LIST_DECLARED_WORKFLOWS = "workflow.list"
    GET_DECLARED_WORKFLOW = "workflow.get"
    PLAN_CAPTURE = "capture.plan"
    START_DETACHED_CAPTURE = "capture.detached.start"
    GET_DETACHED_CAPTURE = "capture.detached.status"
    LIST_RUNS = "run.list"
    LIST_ARTIFACTS = "artifact.list"
    IMPORT_ARTIFACT = "artifact.import"
    IMPORT_XCTRACE = "artifact.import.xctrace"
    EXTRACT_PERFETTO = "artifact.extract.perfetto"
    EXTRACT_MEMRAY = "artifact.extract.memray"
    EXTRACT_NSIGHT_SYSTEMS = "artifact.extract.nsight_systems"
    CONFIGURE_INFERENCE_SERVER = "inference.server.configure"
    LIST_INFERENCE_CONFIGURATIONS = "inference.configuration.list"
    PLAN_INFERENCE_SCENARIO = "inference.scenario.plan"
    EXECUTE_REDUCTION = "reduction.execute"
    GET_REDUCTION = "reduction.status"


class ToolName(StrEnum):
    INITIALIZE_WORKSPACE = "initialize_workspace"
    WORKLOAD_CONFIGURATION_STATUS = "workload_configuration_status"
    CONFIGURE_WORKLOAD = "configure_workload"
    LIST_CAPABILITIES = "list_capabilities"
    START_CAPABILITY_SETUP = "start_capability_setup"
    GET_CAPABILITY_SETUP = "get_capability_setup"
    PREPARE_ADAPTER = "prepare_adapter"
    PREPARE_WORKLOAD_DEPENDENCIES = "prepare_workload_dependencies"
    LIST_DECLARED_WORKFLOWS = "list_declared_workflows"
    GET_DECLARED_WORKFLOW = "get_declared_workflow"
    PLAN_CAPTURE = "plan_capture"
    START_DETACHED_CAPTURE = "start_detached_capture"
    GET_DETACHED_CAPTURE = "get_detached_capture"
    LIST_RUNS = "list_runs"
    LIST_ARTIFACTS = "list_artifacts"
    IMPORT_ARTIFACT = "import_artifact"
    IMPORT_XCTRACE = "import_xctrace"
    EXTRACT_PERFETTO = "extract_perfetto"
    EXTRACT_MEMRAY = "extract_memray"
    EXTRACT_NSIGHT_SYSTEMS = "extract_nsight_systems"
    CONFIGURE_INFERENCE_SERVER = "configure_inference_server"
    LIST_INFERENCE_CONFIGURATIONS = "list_inference_configurations"
    PLAN_INFERENCE_SCENARIO = "plan_inference_scenario"
    EXECUTE_REDUCTION = "execute_reduction"
    GET_REDUCTION = "get_reduction"


class ActionLifecycle(StrEnum):
    READ = "read"
    CONFIGURE = "configure"
    EXECUTE = "execute"
    START = "start"
    STATUS = "status"


@dataclass(frozen=True, slots=True)
class ActionAnnotations:
    read_only: bool
    destructive: bool
    idempotent: bool
    open_world: bool


READ_ONLY_ACTION = ActionAnnotations(True, False, True, False)
CONFIGURE_ACTION = ActionAnnotations(False, False, True, False)
ADDITIVE_ACTION = ActionAnnotations(False, False, False, False)
EXECUTE_ACTION = ActionAnnotations(False, True, False, True)
IDEMPOTENT_EXECUTE_ACTION = ActionAnnotations(False, True, True, True)


class _EmptyArguments(ContractModel):
    pass


class _ConfigureWorkloadArguments(ContractModel):
    name: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    operation: Literal["create", "replace"]
    argv: Annotated[tuple[str, ...], Field(min_length=1, max_length=1_024)]
    cwd: str = Field(default=".", min_length=1, max_length=4_096)
    timeout_seconds: float = Field(default=300.0, gt=0, le=86_400)
    parameters: dict[str, tuple[JsonValue, ...]] | None = Field(default=None, max_length=128)
    environment: dict[str, str] | None = Field(default=None, max_length=128)
    oracle: dict[str, JsonValue] | None = None
    requirements: dict[str, JsonValue] | None = None
    writable_paths: tuple[str, ...] = Field(default=(), max_length=16)
    identity: dict[str, JsonValue] | None = None
    expected_configuration_id: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )


class _ListCapabilitiesArguments(ContractModel):
    mode: Literal["passive", "active_cached", "active_refresh"] = "passive"
    adapter: str | None = Field(default=None, min_length=1, max_length=100)


class _StartCapabilitySetupArguments(ContractModel):
    adapters: Annotated[tuple[str, ...], Field(min_length=1, max_length=6)]
    idempotency_key: str = Field(min_length=1, max_length=200)


class _OperationIdArguments(ContractModel):
    operation_id: str = Field(min_length=4, max_length=100)


class _PrepareAdapterArguments(ContractModel):
    adapter: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    distribution: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )


class _WorkloadNameArguments(ContractModel):
    workload_name: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )


class _ListDeclaredWorkflowsArguments(ContractModel):
    kind: Literal["workload", "experiment", "fault_experiment"] = "workload"
    limit: StrictInt = Field(default=50, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=4_096)


class _GetDeclaredWorkflowArguments(ContractModel):
    kind: Literal["workload", "experiment", "fault_experiment"]
    name: str = Field(min_length=1, max_length=100)


class _PlanCaptureArguments(ContractModel):
    workload_name: str = Field(min_length=1, max_length=100)
    adapter: str = Field(min_length=1, max_length=100)
    parameters: dict[str, JsonValue] = Field(max_length=128)
    preflight_mode: Literal["passive", "active", "auto"] = "auto"
    capture_mode: Literal["auto", "managed", "trusted_local"] = "auto"
    external_context: dict[str, JsonValue] | None = None
    compute_sanitizer_options: dict[str, JsonValue] | None = None
    torch_profiler_options: dict[str, JsonValue] | None = None


class _StartDetachedCaptureArguments(ContractModel):
    plan_token: str = Field(min_length=1, max_length=200)
    idempotency_key: str = Field(
        min_length=8,
        max_length=200,
        pattern=r"^[A-Za-z0-9._:/-]+$",
    )


class _RunIdArguments(ContractModel):
    run_id: str = Field(min_length=1, max_length=200)


class _ListRunsArguments(ContractModel):
    limit: StrictInt = Field(default=50, ge=1, le=1_000)
    filter: dict[str, JsonValue] | None = None
    cursor: str | None = Field(default=None, max_length=4_096)


class _ListArtifactsArguments(ContractModel):
    limit: StrictInt = Field(default=50, ge=1, le=1_000)
    cursor: str | None = Field(default=None, max_length=4_096)


class _ImportArtifactArguments(ContractModel):
    path: str = Field(min_length=1, max_length=4_096)
    kind: str = Field(min_length=1, max_length=100)
    sensitivity: Literal["normal", "internal", "sensitive"]
    media_type: str | None = Field(default=None, max_length=200)
    source_root: Literal["project", "temp"] = "project"
    producer: str = Field(default="auto", min_length=1, max_length=100)
    producer_version: str | None = Field(default=None, max_length=100)


class _ImportXctraceArguments(ContractModel):
    path: str = Field(min_length=1, max_length=4_096)
    source_root: Literal["project", "temp"] = "project"
    max_export_bytes: StrictInt = Field(default=4 * 1024 * 1024, ge=1, le=16 * 1024 * 1024)


class _ExtractPerfettoArguments(_RunIdArguments):
    artifact_id: str | None = Field(default=None, min_length=1, max_length=200)


class _ConfigureInferenceServerArguments(ContractModel):
    name: str = Field(min_length=1, max_length=100)
    operation: Literal["create", "replace"]
    mode: Literal["managed", "existing_local"]
    model: str = Field(min_length=1, max_length=500)
    provider: str = Field(default="vllm", min_length=1, max_length=100)
    benchmark_python: str | None = None
    workload: str | None = Field(default=None, max_length=100)
    base_url: str = Field(default="http://127.0.0.1:8000", max_length=2_048)
    model_revision: str | None = Field(default=None, max_length=200)
    tokenizer: str | None = Field(default=None, max_length=500)
    tokenizer_revision: str | None = Field(default=None, max_length=200)
    quantization: str | None = Field(default=None, max_length=100)
    expected_configuration_id: str | None = None


class _PlanInferenceScenarioArguments(ContractModel):
    scenario_name: str = Field(min_length=1, max_length=100)
    timeout_seconds: float | None = Field(default=None, gt=0, le=86_400)


class _ReductionPlanArguments(ContractModel):
    plan_id: str = Field(min_length=1, max_length=200)


class _ReductionIdArguments(ContractModel):
    reduction_id: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    action: ActionId
    tool_name: ToolName
    input_model: type[ContractModel]
    annotations: ActionAnnotations
    lifecycle: ActionLifecycle

    def validate_arguments(self, arguments: object) -> dict[str, Any]:
        validated = self.input_model.model_validate(arguments)
        return validated.model_dump(mode="json", exclude_none=True)


def _descriptor(
    action: ActionId,
    tool_name: ToolName,
    input_model: type[ContractModel],
    annotations: ActionAnnotations,
    lifecycle: ActionLifecycle,
) -> ActionDescriptor:
    return ActionDescriptor(action, tool_name, input_model, annotations, lifecycle)


ACTION_REGISTRY = MappingProxyType(
    {
        descriptor.action: descriptor
        for descriptor in (
            _descriptor(
                ActionId.INITIALIZE_WORKSPACE,
                ToolName.INITIALIZE_WORKSPACE,
                _EmptyArguments,
                CONFIGURE_ACTION,
                ActionLifecycle.CONFIGURE,
            ),
            _descriptor(
                ActionId.INSPECT_WORKLOAD_CONFIGURATION,
                ToolName.WORKLOAD_CONFIGURATION_STATUS,
                _EmptyArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.CONFIGURE_WORKLOAD,
                ToolName.CONFIGURE_WORKLOAD,
                _ConfigureWorkloadArguments,
                CONFIGURE_ACTION,
                ActionLifecycle.CONFIGURE,
            ),
            _descriptor(
                ActionId.INSPECT_CAPABILITIES,
                ToolName.LIST_CAPABILITIES,
                _ListCapabilitiesArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.START_CAPABILITY_SETUP,
                ToolName.START_CAPABILITY_SETUP,
                _StartCapabilitySetupArguments,
                CONFIGURE_ACTION,
                ActionLifecycle.START,
            ),
            _descriptor(
                ActionId.GET_CAPABILITY_SETUP,
                ToolName.GET_CAPABILITY_SETUP,
                _OperationIdArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.STATUS,
            ),
            _descriptor(
                ActionId.PREPARE_ADAPTER,
                ToolName.PREPARE_ADAPTER,
                _PrepareAdapterArguments,
                CONFIGURE_ACTION,
                ActionLifecycle.CONFIGURE,
            ),
            _descriptor(
                ActionId.PREPARE_WORKLOAD_DEPENDENCIES,
                ToolName.PREPARE_WORKLOAD_DEPENDENCIES,
                _WorkloadNameArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.LIST_DECLARED_WORKFLOWS,
                ToolName.LIST_DECLARED_WORKFLOWS,
                _ListDeclaredWorkflowsArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.GET_DECLARED_WORKFLOW,
                ToolName.GET_DECLARED_WORKFLOW,
                _GetDeclaredWorkflowArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.PLAN_CAPTURE,
                ToolName.PLAN_CAPTURE,
                _PlanCaptureArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.START_DETACHED_CAPTURE,
                ToolName.START_DETACHED_CAPTURE,
                _StartDetachedCaptureArguments,
                IDEMPOTENT_EXECUTE_ACTION,
                ActionLifecycle.START,
            ),
            _descriptor(
                ActionId.GET_DETACHED_CAPTURE,
                ToolName.GET_DETACHED_CAPTURE,
                _RunIdArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.STATUS,
            ),
            _descriptor(
                ActionId.LIST_RUNS,
                ToolName.LIST_RUNS,
                _ListRunsArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.LIST_ARTIFACTS,
                ToolName.LIST_ARTIFACTS,
                _ListArtifactsArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.IMPORT_ARTIFACT,
                ToolName.IMPORT_ARTIFACT,
                _ImportArtifactArguments,
                ADDITIVE_ACTION,
                ActionLifecycle.CONFIGURE,
            ),
            _descriptor(
                ActionId.IMPORT_XCTRACE,
                ToolName.IMPORT_XCTRACE,
                _ImportXctraceArguments,
                ADDITIVE_ACTION,
                ActionLifecycle.CONFIGURE,
            ),
            _descriptor(
                ActionId.EXTRACT_PERFETTO,
                ToolName.EXTRACT_PERFETTO,
                _ExtractPerfettoArguments,
                ADDITIVE_ACTION,
                ActionLifecycle.EXECUTE,
            ),
            _descriptor(
                ActionId.EXTRACT_MEMRAY,
                ToolName.EXTRACT_MEMRAY,
                _RunIdArguments,
                ADDITIVE_ACTION,
                ActionLifecycle.EXECUTE,
            ),
            _descriptor(
                ActionId.EXTRACT_NSIGHT_SYSTEMS,
                ToolName.EXTRACT_NSIGHT_SYSTEMS,
                _RunIdArguments,
                ADDITIVE_ACTION,
                ActionLifecycle.EXECUTE,
            ),
            _descriptor(
                ActionId.CONFIGURE_INFERENCE_SERVER,
                ToolName.CONFIGURE_INFERENCE_SERVER,
                _ConfigureInferenceServerArguments,
                CONFIGURE_ACTION,
                ActionLifecycle.CONFIGURE,
            ),
            _descriptor(
                ActionId.LIST_INFERENCE_CONFIGURATIONS,
                ToolName.LIST_INFERENCE_CONFIGURATIONS,
                _EmptyArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.PLAN_INFERENCE_SCENARIO,
                ToolName.PLAN_INFERENCE_SCENARIO,
                _PlanInferenceScenarioArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.READ,
            ),
            _descriptor(
                ActionId.EXECUTE_REDUCTION,
                ToolName.EXECUTE_REDUCTION,
                _ReductionPlanArguments,
                EXECUTE_ACTION,
                ActionLifecycle.EXECUTE,
            ),
            _descriptor(
                ActionId.GET_REDUCTION,
                ToolName.GET_REDUCTION,
                _ReductionIdArguments,
                READ_ONLY_ACTION,
                ActionLifecycle.STATUS,
            ),
        )
    }
)

ACTION_BY_TOOL_NAME = MappingProxyType(
    {descriptor.tool_name: descriptor.action for descriptor in ACTION_REGISTRY.values()}
)


class ToolAction(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    kind: Literal["tool"] = "tool"
    action: ActionId
    # Values are produced only by an action descriptor's strict input model.
    # Keeping the carrier untyped avoids recursively embedding every action's
    # JSON value graph into every MCP output schema.
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def arguments_match_registered_action(self) -> ToolAction:
        ACTION_REGISTRY[self.action].validate_arguments(self.arguments)
        return self

    @property
    def tool_name(self) -> ToolName:
        return ACTION_REGISTRY[self.action].tool_name


class ManualAction(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    kind: Literal["manual"] = "manual"
    instruction: str = Field(min_length=1, max_length=500)
    suggested_action: ActionId | None = None
    missing_arguments: tuple[str, ...] = ()


class ExternalAction(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    kind: Literal["external"] = "external"
    instruction: str = Field(min_length=1, max_length=500)
    system: str = Field(min_length=1, max_length=100)


type NextAction = Annotated[ToolAction | ManualAction | ExternalAction, Field(discriminator="kind")]


def tool_action(action: ActionId, /, **arguments: JsonValue) -> ToolAction:
    validated = ACTION_REGISTRY[action].validate_arguments(arguments)
    return ToolAction(action=action, arguments=validated)


def manual_action(
    instruction: str,
    *,
    suggested_action: ActionId | None = None,
    missing_arguments: tuple[str, ...] = (),
) -> ManualAction:
    return ManualAction(
        instruction=instruction,
        suggested_action=suggested_action,
        missing_arguments=missing_arguments,
    )


def next_action_for_tool(
    tool_name: ToolName | str,
    *,
    context: Mapping[str, object] = MappingProxyType({}),
    instruction: str,
) -> NextAction:
    """Resolve legacy recovery context into an executable or explicitly manual edge."""

    try:
        normalized_tool = ToolName(tool_name)
    except ValueError as error:
        raise ValueError(f"Unregistered workflow tool {tool_name!r}.") from error
    return next_action_for_action(
        ACTION_BY_TOOL_NAME[normalized_tool],
        context=context,
        instruction=instruction,
    )


def next_action_for_action(
    action: ActionId,
    *,
    context: Mapping[str, object] = MappingProxyType({}),
    instruction: str,
) -> NextAction:
    descriptor = ACTION_REGISTRY[action]
    arguments = {
        name: context[name] for name in descriptor.input_model.model_fields if name in context
    }
    try:
        validated = descriptor.validate_arguments(arguments)
    except ValidationError as error:
        missing = tuple(
            str(item["loc"][0])
            for item in error.errors(include_url=False)
            if item["type"] == "missing" and item["loc"]
        )
        return manual_action(
            instruction,
            suggested_action=action,
            missing_arguments=missing,
        )
    return ToolAction(action=action, arguments=validated)
