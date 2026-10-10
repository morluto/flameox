"""Declarative MCP tool contracts and JSON Schema projection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from jsonschema import Draft202012Validator
from mcp_types import Tool, ToolAnnotations
from pydantic import BaseModel, RootModel

from flameox.mcp.descriptions import TOOL_DESCRIPTIONS, analysis_description, capture_description
from flameox.mcp.request_contracts import (
    InspectEvidenceArguments,
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
    EvidenceInspectionOutcome,
    PreparationOutcome,
    PreservationOutcome,
    QueryOutcome,
    RescueOutcome,
)
from flameox.runtime_contracts import OPERATIONS, Operation, compatible_capture_providers

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


@cache
def _output_schema(model: type[RootModel[Any]]) -> dict[str, Any]:
    return model.model_json_schema(mode="serialization")


@dataclass(frozen=True, slots=True)
class ToolContract:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[RootModel[Any]]
    annotations: ToolAnnotations
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
            output_schema=deepcopy(_output_schema(self.output_model)),
            annotations=self.annotations.model_copy(deep=True),
        )


TOOL_SPECS: tuple[tuple[str, type[BaseModel], type[RootModel[Any]], ToolAnnotations], ...] = (
    ("prepare_providers", PrepareProvidersArguments, PreparationOutcome, PREPARE),
    ("preserve_evidence", PreserveArguments, PreservationOutcome, PRESERVE),
    ("rescue_evidence", RescueArguments, RescueOutcome, PRESERVE),
    ("query_evidence", QueryArguments, QueryOutcome, READ_ONLY),
    ("inspect_evidence", InspectEvidenceArguments, EvidenceInspectionOutcome, READ_ONLY),
)


def tool_contracts() -> tuple[ToolContract, ...]:
    lifecycle = tuple(
        ToolContract(
            name=name,
            description=TOOL_DESCRIPTIONS[name],
            input_model=input_model,
            output_model=output_model,
            annotations=annotations,
        )
        for name, input_model, output_model, annotations in TOOL_SPECS
    )
    analysis = tuple(
        ToolContract(
            name=spec.name,
            description=analysis_description(spec),
            input_model=analysis_arguments(spec),
            output_model=AnalysisOutcome,
            annotations=READ_ONLY,
        )
        for spec in OPERATIONS
    )
    capture = tuple(
        ToolContract(
            name=spec.capture_name,
            description=capture_description(spec),
            input_model=capture_arguments(spec),
            output_model=CaptureOutcome,
            annotations=CAPTURE,
        )
        for spec in OPERATIONS
        if compatible_capture_providers(spec)
    )
    return (*analysis, *capture, *lifecycle)


def operation_descriptor(spec: Operation) -> dict[str, Any]:
    """Project one shared registry entry into compact CLI/MCP discovery."""

    providers = compatible_capture_providers(spec)
    return {
        "operation": spec.name,
        "capture_tool": spec.capture_name if providers else None,
        "summary": spec.summary,
        "accepted_formats": list(spec.formats),
        "minimum_sources": spec.minimum_sources,
        "maximum_sources": spec.maximum_sources,
        "capture_supported": bool(providers),
        "experiment_supported": bool(providers) and spec.maximum_sources > 1,
        "capture_providers": [provider.id for provider in providers],
    }


def operation_detail(spec: Operation) -> dict[str, Any]:
    """Add selected operation schemas, examples, and routing constraints."""

    providers = compatible_capture_providers(spec)
    option_schema = spec.model.model_json_schema()
    analysis_call = {
        "tool": spec.name,
        "arguments": analysis_example(spec),
    }
    capture_call = (
        {"tool": spec.capture_name, "arguments": capture_example(spec)} if providers else None
    )
    exclusions = ["Analysis tools read existing artifacts and never execute a workload."]
    if spec.name == "inspect_performance_candidates":
        exclusions.append("Consumes SARIF and does not scan source files.")
    if not providers:
        exclusions.append("No capture provider produces this operation's accepted artifacts.")
    return operation_descriptor(spec) | {
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
        "limitations": [spec.limitation],
        "routing_exclusions": exclusions,
    }
