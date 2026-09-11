"""Strict MCP result schemas projected from runtime-owned evidence contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, GetJsonSchemaHandler, JsonValue, RootModel
from pydantic.json_schema import JsonSchemaValue

from flameox.mcp.capability_tools import AnalysisRequest
from flameox.runtime_contracts import (
    LOWERCASE_SHA256_PATTERN,
    MAX_ROWS,
    AnalysisResult,
    StrictModel,
)


class ToolFailureEnvelope(StrictModel):
    code: str = Field(description="Stable machine-readable failure code.")
    message: str = Field(description="Human-readable failure and recovery guidance.")
    details: dict[str, JsonValue] = Field(description="Failure-specific structured context.")


class EvidenceReferenceEnvelope(StrictModel):
    evidence_id: str = Field(
        description="Content-addressed immutable evidence identifier.",
        pattern=LOWERCASE_SHA256_PATTERN,
    )
    uri: str = Field(description="Opaque MCP resource URI for the preserved manifest projection.")
    artifact_count: int = Field(description="Native artifacts preserved with the manifest.", ge=0)


class AnalyzeArgumentsEnvelope(StrictModel):
    request: AnalysisRequest = Field(description="Complete analyze request; submit unchanged.")
    page_size: int = Field(
        description="Maximum evidence rows returned on this page.", ge=1, le=MAX_ROWS
    )


class NextPageEnvelope(StrictModel):
    tool: Literal["analyze"] = Field(description="Exact MCP tool to call.")
    arguments: AnalyzeArgumentsEnvelope = Field(description="Complete MCP tool arguments.")


class AnalysisEnvelope(AnalysisResult):
    preserved: EvidenceReferenceEnvelope | None = Field(
        default=None,
        description="Immutable evidence reference when the operation preserved this result.",
    )
    next_page: NextPageEnvelope | None = Field(
        default=None,
        description="Exact analyze invocation for the next page, or null when unavailable.",
    )


class ExternalRequirementEnvelope(StrictModel):
    provider_id: str = Field(description="Host provider that Flameox cannot install.")
    guidance: str = Field(description="Host installation or access requirement.")


class PreparationStatusEnvelope(StrictModel):
    status: Literal["prepared", "not_applicable"] = Field(
        description="Whether Flameox prepared a managed provider environment."
    )


class LauncherEnvelope(StrictModel):
    command: str = Field(description="Executable for the prepared MCP launcher.")
    args: list[str] = Field(description="Arguments for the prepared MCP launcher.")


class ReconnectActionEnvelope(StrictModel):
    kind: Literal["reconnect_mcp"]
    message: str = Field(description="Bounded reconnection and preservation guidance.")
    necessity: Literal["required", "conditional"] = Field(
        description="Whether the active server is known to require replacement."
    )


class PreparationEnvelope(StrictModel):
    requested_providers: list[str] = Field(description="Complete requested provider set.")
    prepared_managed_providers: list[str] = Field(
        description="Requested Python providers included in the prepared environment."
    )
    external_requirements: list[ExternalRequirementEnvelope] = Field(
        description="Host tools, drivers, devices, or permissions still required."
    )
    preparation: PreparationStatusEnvelope = Field(
        description="Managed environment preparation status."
    )
    launcher: LauncherEnvelope = Field(description="Version-pinned launcher to configure.")
    next_action: ReconnectActionEnvelope | None = Field(
        description="Required or conditional reconnection, or null when already active."
    )
    activation_status: Literal["ready", "restart_required", "unknown", "not_applicable"] = Field(
        description="Whether the prepared dependency identity is active in this process."
    )
    workload_requirements: list[ExternalRequirementEnvelope] = Field(
        description="Requirements to verify in the exact workload interpreter."
    )


class PreservationEnvelope(EvidenceReferenceEnvelope):
    next_page: NextPageEnvelope | None = Field(
        default=None,
        description="Refreshed evidence-backed next-page call after live scratch is released.",
    )


class RescueEnvironmentEnvelope(StrictModel):
    FLAMEOX_DATA_DIR: str = Field(
        description="Alternate evidence store to configure for the restarted Flameox process."
    )


class RescueActionEnvelope(StrictModel):
    kind: Literal["restart_reconnect"]
    environment: RescueEnvironmentEnvelope = Field(
        description="Environment required by the restarted server."
    )
    message: str = Field(description="Bounded operator guidance for opening rescued evidence.")


class RescueEnvelope(PreservationEnvelope):
    rescue_destination: str = Field(description="Absolute directory containing rescued evidence.")
    next_action: RescueActionEnvelope = Field(
        description="Required restart or reconnection handoff."
    )


class QueryArgumentsEnvelope(StrictModel):
    evidence_kind: str | None = Field(default=None, description="Unchanged evidence-kind filter.")
    capability_id: str | None = Field(default=None, description="Unchanged capability filter.")
    provider_id: str | None = Field(default=None, description="Unchanged provider filter.")
    input_sha256: str | None = Field(
        default=None,
        description="Unchanged contributing-input digest filter.",
        pattern=LOWERCASE_SHA256_PATTERN,
    )
    created_after: datetime | None = Field(
        default=None, description="Unchanged inclusive lower creation-time bound."
    )
    created_before: datetime | None = Field(
        default=None, description="Unchanged inclusive upper creation-time bound."
    )
    page_size: int = Field(description="Unchanged page size.", ge=1, le=200)
    cursor: str = Field(description="Opaque cursor for the next inventory page.")


class QueryNextPageEnvelope(StrictModel):
    tool: Literal["query_evidence"] = Field(description="Exact MCP tool to call.")
    arguments: QueryArgumentsEnvelope = Field(description="Complete MCP tool arguments.")


class QueryEnvelope(StrictModel):
    evidence: list[dict[str, JsonValue]] = Field(
        description="Matching immutable evidence summaries."
    )
    inventory_digest: str = Field(description="Digest of the query's inventory snapshot.")
    next_page: QueryNextPageEnvelope | None = Field(
        default=None,
        description="Exact query_evidence call for the next page, or null when complete.",
    )


class _ObjectOutcome(RootModel[Any]):
    """Keep MCP's required object root while validating success and failure variants."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        schema["type"] = "object"
        return schema


class PreparationOutcome(_ObjectOutcome):
    root: PreparationEnvelope | ToolFailureEnvelope


class AnalysisOutcome(_ObjectOutcome):
    root: AnalysisEnvelope | ToolFailureEnvelope


class PreservationOutcome(_ObjectOutcome):
    root: PreservationEnvelope | ToolFailureEnvelope


class RescueOutcome(_ObjectOutcome):
    root: RescueEnvelope | ToolFailureEnvelope


class QueryOutcome(_ObjectOutcome):
    root: QueryEnvelope | ToolFailureEnvelope
