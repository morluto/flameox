"""Typed MCP results and recovery states."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, GetJsonSchemaHandler, JsonValue, RootModel
from pydantic.json_schema import JsonSchemaValue

from flameox.evidence_models import CaptureExecution
from flameox.runtime_contracts import (
    LOWERCASE_SHA256_PATTERN,
    AnalysisResult,
    Coverage,
    ProviderIdentity,
    StrictModel,
)


class CallToolAction(StrictModel):
    kind: Literal["call_tool"]
    tool: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    then_retry: str | None = None
    message: str


class ReconnectAction(StrictModel):
    kind: Literal["reconnect_mcp"]
    message: str
    necessity: Literal["required", "conditional"]
    launcher: dict[str, JsonValue] | None = None


class AdjustRequestAction(StrictModel):
    kind: Literal["adjust_request"]
    field_path: list[str | int] | None = None
    message: str


class ConfigureEnvironmentAction(StrictModel):
    kind: Literal["configure_environment"]
    environment: dict[str, str] = Field(default_factory=dict)
    message: str


class WaitAndRetryAction(StrictModel):
    kind: Literal["wait_and_retry"]
    retry_after_ms: int | None = Field(default=None, ge=0)
    message: str


class OperatorAction(StrictModel):
    kind: Literal["operator_action"]
    message: str


class PreserveThenAnalyzeAction(StrictModel):
    kind: Literal["preserve_then_analyze"]
    preserve_arguments: dict[str, JsonValue]
    message: str


NextAction = (
    CallToolAction
    | AdjustRequestAction
    | ReconnectAction
    | ConfigureEnvironmentAction
    | WaitAndRetryAction
    | OperatorAction
    | PreserveThenAnalyzeAction
)

type FailureCode = str


class ToolFailureEnvelope(StrictModel):
    status: Literal["failed"] = "failed"
    code: FailureCode = Field(description="Stable machine-readable failure code.")
    message: str = Field(description="Human-readable failure and recovery guidance.")
    retryable: bool = False
    field_path: list[str | int] | None = None
    accepted_values: list[str] | None = None
    next_action: NextAction | None = Field(default=None, discriminator="kind")
    retry_after_ms: int | None = Field(default=None, ge=0)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class RecoverableEnvelope(StrictModel):
    status: Literal["retryable", "unavailable"]
    code: str
    message: str
    retryable: bool
    next_action: NextAction | None = Field(default=None, discriminator="kind")
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ToolCallEnvelope(StrictModel):
    """Executable next call without recursively embedding another tool schema."""

    tool: Literal["analyze", "query_evidence"]
    arguments: dict[str, JsonValue]


class EvidenceReferenceEnvelope(StrictModel):
    evidence_id: str = Field(pattern=LOWERCASE_SHA256_PATTERN)
    uri: str
    artifact_count: int = Field(ge=0)


class AnalysisEnvelope(AnalysisResult):
    status: Literal["complete"] = "complete"
    capture: None = None
    analysis_failure: None = None
    preserved: EvidenceReferenceEnvelope | None = None
    next_page: ToolCallEnvelope | None = None


class CaptureOutcomeEnvelope(StrictModel):
    status: Literal["succeeded", "failed"]
    execution_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)


class CaptureDetailsEnvelope(StrictModel):
    status: Literal["complete"]
    workload_status: Literal["succeeded", "failed", "unknown"]
    mode: Literal["single", "experiment"]
    requested_capability_id: str
    executions: list[CaptureExecution]
    outcome: CaptureOutcomeEnvelope


class CaptureAnalysisEnvelope(AnalysisResult):
    status: Literal["complete", "partial"]
    capture: CaptureDetailsEnvelope  # type: ignore[assignment]
    preserved: EvidenceReferenceEnvelope | None = None
    next_page: ToolCallEnvelope | None = None
    next_action: NextAction | None = Field(default=None, discriminator="kind")


class ExternalRequirementEnvelope(StrictModel):
    provider_id: str
    guidance: str


class PreparationStatusEnvelope(StrictModel):
    status: Literal["prepared", "not_applicable"]


class LauncherEnvelope(StrictModel):
    command: str
    args: list[str]


class PreparationEnvelope(StrictModel):
    status: Literal["complete"] = "complete"
    requested_providers: list[str]
    prepared_managed_providers: list[str]
    external_requirements: list[ExternalRequirementEnvelope]
    preparation: PreparationStatusEnvelope
    launcher: LauncherEnvelope
    next_action: ReconnectAction | None
    activation_status: Literal["ready", "restart_required", "unknown", "not_applicable"]
    workload_requirements: list[ExternalRequirementEnvelope]


class PreservationEnvelope(EvidenceReferenceEnvelope):
    status: Literal["complete"] = "complete"
    next_page: ToolCallEnvelope | None = None


class RescueEnvironmentEnvelope(StrictModel):
    FLAMEOX_DATA_DIR: str


class RescueActionEnvelope(StrictModel):
    kind: Literal["restart_reconnect"]
    environment: RescueEnvironmentEnvelope
    message: str


class RescueEnvelope(PreservationEnvelope):
    rescue_destination: str
    next_action: RescueActionEnvelope


class EvidenceSummaryEnvelope(StrictModel):
    evidence_id: str = Field(pattern=LOWERCASE_SHA256_PATTERN)
    uri: str
    evidence_kind: str
    capability_id: str
    provider: ProviderIdentity
    created_at: str
    coverage: Coverage
    limitations: list[str]


class QueryEnvelope(StrictModel):
    status: Literal["complete"] = "complete"
    inventory_status: Literal["absent", "empty", "available"]
    match_status: Literal["matched", "no_matches"]
    inventory_size: int = Field(ge=0)
    evidence: list[EvidenceSummaryEnvelope]
    inventory_digest: str
    next_page: ToolCallEnvelope | None = None


class CaptureProviderSummary(StrictModel):
    id: str
    artifact_formats: list[str]
    option_schema: dict[str, JsonValue]


class CapabilityListRecord(StrictModel):
    capability_id: str
    summary: str
    accepted_formats: list[str]
    minimum_sources: int = Field(ge=1)
    maximum_sources: int = Field(ge=1)
    capture_providers: list[str]
    capture_supported: bool
    experiment_supported: bool


class CapabilityDetail(StrictModel):
    capability_id: str
    summary: str
    accepted_formats: list[str]
    minimum_sources: int = Field(ge=1)
    maximum_sources: int = Field(ge=1)
    capture_supported: bool
    experiment_supported: bool
    capture_providers: list[CaptureProviderSummary]
    analysis_option_schema: dict[str, JsonValue]
    analysis_example: dict[str, JsonValue]
    capture_example: dict[str, JsonValue] | None = None
    limitations: list[str] = Field(default_factory=list)
    routing_exclusions: list[str] = Field(default_factory=list)


class CapabilityListEnvelope(StrictModel):
    status: Literal["complete"] = "complete"
    mode: Literal["list"]
    capabilities: list[CapabilityListRecord]


class CapabilityGetEnvelope(StrictModel):
    status: Literal["complete"] = "complete"
    mode: Literal["get"]
    capabilities: list[CapabilityDetail]


class _ObjectOutcome(RootModel[Any]):
    """Keep MCP's required object root while validating result variants."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        schema["type"] = "object"
        return schema


class PreparationOutcome(_ObjectOutcome):
    root: PreparationEnvelope | RecoverableEnvelope | ToolFailureEnvelope


class AnalysisOutcome(_ObjectOutcome):
    root: AnalysisEnvelope | RecoverableEnvelope | ToolFailureEnvelope


class CaptureOutcome(_ObjectOutcome):
    root: CaptureAnalysisEnvelope | RecoverableEnvelope | ToolFailureEnvelope


class PreservationOutcome(_ObjectOutcome):
    root: PreservationEnvelope | RecoverableEnvelope | ToolFailureEnvelope


class RescueOutcome(_ObjectOutcome):
    root: RescueEnvelope | RecoverableEnvelope | ToolFailureEnvelope


class QueryOutcome(_ObjectOutcome):
    root: QueryEnvelope | RecoverableEnvelope | ToolFailureEnvelope


class CapabilityInspectionOutcome(_ObjectOutcome):
    root: CapabilityListEnvelope | CapabilityGetEnvelope | ToolFailureEnvelope
