"""Declarative MCP tool contracts and JSON Schema projection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mcp.server import ServerRequestContext
from mcp_types import CallToolResult, Tool, ToolAnnotations
from pydantic import BaseModel, RootModel

from flameox.mcp.descriptions import TOOL_DESCRIPTIONS
from flameox.mcp.request_contracts import (
    AnalyzeArguments,
    CaptureArguments,
    InspectCapabilitiesArguments,
    PrepareProvidersArguments,
    PreserveArguments,
    QueryArguments,
    RescueArguments,
)
from flameox.mcp.result_contracts import (
    AnalysisOutcome,
    CapabilityInspectionOutcome,
    CaptureOutcome,
    PreparationOutcome,
    PreservationOutcome,
    QueryOutcome,
    RescueOutcome,
)
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import Capability, compatible_capture_providers

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
CAPTURE = ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=True
)
PRESERVE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
PREPARE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
)


@dataclass(frozen=True, slots=True)
class ToolContract[InputT: BaseModel, OutputT: RootModel[Any]]:
    name: str
    description: str
    input_model: type[InputT]
    output_model: type[OutputT]
    annotations: ToolAnnotations
    handler: Callable[[InputT, ServerRequestContext[AnalysisRuntime]], Awaitable[CallToolResult]]

    def project(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(mode="validation"),
            output_schema=self.output_model.model_json_schema(mode="serialization"),
            annotations=self.annotations,
        )


ToolDispatcher = Callable[
    [str, BaseModel, ServerRequestContext[AnalysisRuntime]], Awaitable[CallToolResult]
]

TOOL_SPECS: tuple[tuple[str, type[BaseModel], type[RootModel[Any]], ToolAnnotations], ...] = (
    ("inspect_capabilities", InspectCapabilitiesArguments, CapabilityInspectionOutcome, READ_ONLY),
    ("prepare_providers", PrepareProvidersArguments, PreparationOutcome, PREPARE),
    ("analyze", AnalyzeArguments, AnalysisOutcome, READ_ONLY),
    ("capture_and_analyze", CaptureArguments, CaptureOutcome, CAPTURE),
    ("preserve_evidence", PreserveArguments, PreservationOutcome, PRESERVE),
    ("rescue_evidence", RescueArguments, RescueOutcome, PRESERVE),
    ("query_evidence", QueryArguments, QueryOutcome, READ_ONLY),
)


def bind_tool_contracts(
    dispatcher: ToolDispatcher,
) -> tuple[ToolContract[BaseModel, RootModel[Any]], ...]:
    def bind(
        name: str,
    ) -> Callable[[BaseModel, ServerRequestContext[AnalysisRuntime]], Awaitable[CallToolResult]]:
        async def handler(
            request: BaseModel, ctx: ServerRequestContext[AnalysisRuntime]
        ) -> CallToolResult:
            return await dispatcher(name, request, ctx)

        return handler

    return tuple(
        ToolContract(
            name=name,
            description=TOOL_DESCRIPTIONS[name],
            input_model=input_model,
            output_model=output_model,
            annotations=annotations,
            handler=bind(name),
        )
        for name, input_model, output_model, annotations in TOOL_SPECS
    )


def capability_descriptor(capability: Capability) -> dict[str, Any]:
    """Project one shared registry entry into compact CLI/MCP discovery."""

    providers = compatible_capture_providers(capability)
    return {
        "capability_id": capability.id,
        "summary": capability.summary,
        "accepted_formats": list(capability.formats),
        "minimum_sources": capability.minimum_sources,
        "maximum_sources": capability.maximum_sources,
        "capture_supported": bool(providers),
        "experiment_supported": bool(providers) and capability.maximum_sources > 1,
        "capture_providers": [provider.id for provider in providers],
    }


def capability_detail(capability: Capability) -> dict[str, Any]:
    """Add selected capability schemas, examples, and routing constraints."""

    providers = compatible_capture_providers(capability)
    format_name = capability.formats[0]
    suffix = "sarif" if format_name == "sarif" else format_name.replace("-", ".")
    analysis_example = {
        "request": {
            "capability_id": capability.id,
            "sources": [
                {
                    "kind": "path",
                    "path": f"/absolute/path/artifact.{suffix}",
                    "format": format_name,
                }
            ],
        }
    }
    capture_example = (
        {
            "request": {
                "capability_id": capability.id,
                "target": {"argv": ["python", "workload.py"], "cwd": "/absolute/workdir"},
                "provider": {"kind": providers[0].id},
            }
        }
        if providers
        else None
    )
    exclusions = ["analyze reads existing artifacts and never executes a workload."]
    if capability.id == "static.performance_candidates":
        exclusions.append("Consumes SARIF and does not scan source files.")
    if not providers:
        exclusions.append("No capture provider produces this capability's accepted artifacts.")
    return capability_descriptor(capability) | {
        "capture_providers": [
            {
                "id": provider.id,
                "artifact_formats": [artifact.format for artifact in provider.artifacts],
                "option_schema": provider.argument_model.model_json_schema(),
            }
            for provider in providers
        ],
        "analysis_option_schema": capability.model.model_json_schema(),
        "analysis_example": analysis_example,
        "capture_example": capture_example,
        "limitations": [capability.limitation],
        "routing_exclusions": exclusions,
    }
