"""Declarative MCP tool contracts and JSON Schema projection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from jsonschema import Draft202012Validator
from mcp.server import ServerRequestContext
from mcp_types import CallToolResult, Tool, ToolAnnotations
from pydantic import BaseModel, RootModel

from flameox.mcp.catalog import ANALYSIS_TOOLS, CAPTURE_TOOLS
from flameox.mcp.descriptions import TOOL_DESCRIPTIONS, analysis_description, capture_description
from flameox.mcp.request_contracts import (
    PrepareProvidersArguments,
    PreserveArguments,
    QueryArguments,
    RescueArguments,
    analysis_arguments,
    analysis_example,
    capture_arguments,
    capture_example,
)
from flameox.mcp.result_contracts import (
    AnalysisOutcome,
    CaptureOutcome,
    PreparationOutcome,
    PreservationOutcome,
    QueryOutcome,
    RescueOutcome,
)
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import CAPABILITIES, Capability, compatible_capture_providers

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
class ToolContract:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[RootModel[Any]]
    annotations: ToolAnnotations
    handler: Callable[[BaseModel, ServerRequestContext[AnalysisRuntime]], Awaitable[CallToolResult]]
    input_schema: dict[str, Any] = field(init=False, repr=False, compare=False)
    input_validator: Draft202012Validator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        schema = self.input_model.model_json_schema(mode="validation")
        Draft202012Validator.check_schema(schema)
        object.__setattr__(self, "input_schema", schema)
        object.__setattr__(self, "input_validator", Draft202012Validator(schema))

    def project(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            input_schema=deepcopy(self.input_schema),
            output_schema=self.output_model.model_json_schema(mode="serialization"),
            annotations=self.annotations,
        )


ToolDispatcher = Callable[
    [str, BaseModel, ServerRequestContext[AnalysisRuntime]], Awaitable[CallToolResult]
]

TOOL_SPECS: tuple[tuple[str, type[BaseModel], type[RootModel[Any]], ToolAnnotations], ...] = (
    ("prepare_providers", PrepareProvidersArguments, PreparationOutcome, PREPARE),
    ("preserve_evidence", PreserveArguments, PreservationOutcome, PRESERVE),
    ("rescue_evidence", RescueArguments, RescueOutcome, PRESERVE),
    ("query_evidence", QueryArguments, QueryOutcome, READ_ONLY),
)


def bind_tool_contracts(
    dispatcher: ToolDispatcher,
) -> tuple[ToolContract, ...]:
    lifecycle = tuple(
        ToolContract(
            name=name,
            description=TOOL_DESCRIPTIONS[name],
            input_model=input_model,
            output_model=output_model,
            annotations=annotations,
            handler=partial(dispatcher, name),
        )
        for name, input_model, output_model, annotations in TOOL_SPECS
    )
    analysis = tuple(
        ToolContract(
            name=ANALYSIS_TOOLS[capability.id],
            description=analysis_description(capability),
            input_model=analysis_arguments(capability),
            output_model=AnalysisOutcome,
            annotations=READ_ONLY,
            handler=partial(dispatcher, ANALYSIS_TOOLS[capability.id]),
        )
        for capability in CAPABILITIES
    )
    capture = tuple(
        ToolContract(
            name=CAPTURE_TOOLS[capability.id],
            description=capture_description(capability),
            input_model=capture_arguments(capability),
            output_model=CaptureOutcome,
            annotations=CAPTURE,
            handler=partial(dispatcher, CAPTURE_TOOLS[capability.id]),
        )
        for capability in CAPABILITIES
        if capability.id in CAPTURE_TOOLS
    )
    return (*analysis, *capture, *lifecycle)


def capability_descriptor(capability: Capability) -> dict[str, Any]:
    """Project one shared registry entry into compact CLI/MCP discovery."""

    providers = compatible_capture_providers(capability)
    return {
        "capability_id": capability.id,
        "analysis_tool": ANALYSIS_TOOLS[capability.id],
        "capture_tool": CAPTURE_TOOLS.get(capability.id),
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
    option_schema = capability.model.model_json_schema()
    analysis_call = {
        "tool": ANALYSIS_TOOLS[capability.id],
        "arguments": analysis_example(capability),
    }
    capture_call = (
        {"tool": CAPTURE_TOOLS[capability.id], "arguments": capture_example(capability)}
        if providers
        else None
    )
    exclusions = ["Analysis tools read existing artifacts and never execute a workload."]
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
        "analysis_option_schema": option_schema,
        "analysis_example": analysis_call,
        "capture_example": capture_call,
        "limitations": [capability.limitation],
        "routing_exclusions": exclusions,
    }
