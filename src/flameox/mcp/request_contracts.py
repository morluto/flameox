"""Compact MCP request envelopes over runtime-owned capability contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    JsonValue,
    RootModel,
    WithJsonSchema,
)
from pydantic.json_schema import JsonSchemaValue

from flameox.providers.availability import MANAGED_PROVIDER_EXTRAS, SYSTEM_PROVIDER_GUIDANCE
from flameox.runtime_contracts import (
    CAPABILITIES,
    CAPTURE_PROVIDER_CONTRACTS,
    LOWERCASE_SHA256_PATTERN,
    MAX_ROWS,
    DirectTarget,
    EvidenceSource,
    ExperimentDesign,
    PathSource,
    StrictModel,
)

CAPABILITY_IDS = tuple(item.id for item in CAPABILITIES)
ARTIFACT_FORMATS = tuple(
    sorted({item for capability in CAPABILITIES for item in capability.formats})
)
CAPTURE_PROVIDER_IDS = tuple(CAPTURE_PROVIDER_CONTRACTS)
PREPARABLE_PROVIDER_IDS = tuple(sorted(MANAGED_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))


def _closed(value: str, values: tuple[str, ...], label: str) -> str:
    if value not in values:
        raise ValueError(f"unknown {label}; accepted values: {', '.join(values)}")
    return value


def _capability_id(value: str) -> str:
    return _closed(value, CAPABILITY_IDS, "capability_id")


def _artifact_format(value: str) -> str:
    return _closed(value, ARTIFACT_FORMATS, "artifact format")


def _capture_provider_id(value: str) -> str:
    return _closed(value, CAPTURE_PROVIDER_IDS, "capture provider")


def _preparable_provider_id(value: str) -> str:
    return _closed(value, PREPARABLE_PROVIDER_IDS, "preparable provider")


CapabilityId = Annotated[
    str,
    AfterValidator(_capability_id),
    WithJsonSchema({"type": "string", "enum": list(CAPABILITY_IDS)}),
]
ArtifactFormat = Annotated[
    str,
    AfterValidator(_artifact_format),
    WithJsonSchema({"type": "string", "enum": list(ARTIFACT_FORMATS)}),
]
CaptureProviderId = Annotated[
    str,
    AfterValidator(_capture_provider_id),
    WithJsonSchema({"type": "string", "enum": list(CAPTURE_PROVIDER_IDS)}),
]
PreparableProviderId = Annotated[
    str,
    AfterValidator(_preparable_provider_id),
    WithJsonSchema({"type": "string", "enum": list(PREPARABLE_PROVIDER_IDS)}),
]


class McpPathSource(PathSource):
    format: ArtifactFormat | None = Field(
        default=None,
        description="Explicit native artifact format. Omission requires unambiguous detection.",
    )


def _normalize_mcp_source_kind(value: Any) -> Any:
    if isinstance(value, Mapping) and "kind" not in value:
        value = dict(value)
        value["kind"] = "evidence" if "evidence_id" in value else "path"
    return value


McpSource = Annotated[
    McpPathSource | EvidenceSource,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_mcp_source_kind),
]


class AnalysisRequest(StrictModel):
    """Stable MCP envelope; capability option details are discovered separately."""

    capability_id: CapabilityId = Field(description="Capability selected for this analysis.")
    sources: list[McpSource] = Field(
        description="Native artifact paths or preserved evidence sources.",
        min_length=1,
        max_length=32,
    )
    options: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Capability-specific options from inspect_capabilities.",
    )
    continuation: str | None = Field(
        default=None, description="Opaque continuation returned by the preceding page."
    )


class CaptureProviderRequest(StrictModel):
    kind: CaptureProviderId = Field(description="Capture provider selected for this request.")
    options: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Provider-specific options from inspect_capabilities.",
    )


class CaptureRequest(StrictModel):
    capability_id: CapabilityId = Field(description="Capability to run over captured artifacts.")
    target: DirectTarget = Field(description="Explicit process target executed without a shell.")
    provider: CaptureProviderRequest
    options: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Capability-specific analysis options from inspect_capabilities.",
    )
    experiment: ExperimentDesign | None = Field(
        default=None, description="Optional randomized paired experiment design."
    )
    preserve: bool = Field(
        default=False, description="Preserve native artifacts and the result as immutable evidence."
    )


class PrepareProvidersArguments(StrictModel):
    provider_ids: list[PreparableProviderId] = Field(
        description="Provider IDs to prepare together in one environment.",
        min_length=1,
        max_length=16,
    )
    timeout_seconds: int = Field(default=1_800, ge=1, le=3_600)


class AnalyzeArguments(StrictModel):
    request: AnalysisRequest
    page_size: int = Field(default=100, ge=1, le=MAX_ROWS)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": "/tmp/results.json"}],
                    }
                },
                {
                    "request": {
                        "capability_id": "static.performance_candidates",
                        "sources": [
                            {"kind": "path", "path": "/tmp/report.sarif", "format": "sarif"}
                        ],
                    }
                },
            ]
        }
    )


class CaptureArguments(StrictModel):
    request: CaptureRequest
    page_size: int = Field(default=100, ge=1, le=MAX_ROWS)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {"argv": ["python", "-c", "print('ok')"], "cwd": "/tmp"},
                        "provider": {"kind": "direct"},
                    }
                }
            ]
        }
    )


class PreserveArguments(StrictModel):
    analysis_id: str = Field(pattern=LOWERCASE_SHA256_PATTERN)


class RescueArguments(PreserveArguments):
    destination: str = Field(
        min_length=1,
        max_length=4096,
        description="Agent-selected absolute path for a distinct new evidence directory.",
    )


class QueryArguments(StrictModel):
    evidence_kind: str | None = None
    capability_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
        description="Current or historical capability ID stored in immutable evidence.",
    )
    provider_id: str | None = None
    input_sha256: str | None = Field(default=None, pattern=LOWERCASE_SHA256_PATTERN)
    created_after: datetime | None = None
    created_before: datetime | None = None
    page_size: int = Field(default=50, ge=1, le=200)
    cursor: str | None = None


class InspectCapabilitiesList(StrictModel):
    mode: Literal["list"]
    artifact_format: ArtifactFormat | None = Field(
        default=None, description="Filter to capabilities that consume this artifact format."
    )
    capture_supported: bool | None = Field(
        default=None, description="Filter by whether compatible capture providers exist."
    )


class InspectCapabilitiesGet(StrictModel):
    mode: Literal["get"]
    capability_id: CapabilityId


InspectCapabilitiesRequest = Annotated[
    InspectCapabilitiesList | InspectCapabilitiesGet,
    Field(discriminator="mode"),
]


class InspectCapabilitiesArguments(RootModel[InspectCapabilitiesRequest]):
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        schema["type"] = "object"
        schema["examples"] = [
            {"mode": "list", "artifact_format": "sarif", "capture_supported": False},
            {"mode": "get", "capability_id": "static.performance_candidates"},
        ]
        return schema
