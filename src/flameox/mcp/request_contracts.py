"""Typed direct-tool arguments projected from runtime-owned contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import GenericAlias
from typing import Annotated, Any, Literal, cast

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, create_model

from flameox.providers.availability import MANAGED_PROVIDER_EXTRAS, SYSTEM_PROVIDER_GUIDANCE
from flameox.runtime_contracts import (
    LOWERCASE_SHA256_PATTERN,
    MAX_ROWS,
    Capability,
    DirectTarget,
    EvidenceSource,
    ExperimentDesign,
    PathSource,
    StrictModel,
    compatible_capture_providers,
)

PREPARABLE_PROVIDER_IDS = tuple(sorted(MANAGED_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))


def literal_type(values: tuple[str, ...]) -> Any:
    """Construct the same Literal used by validation and JSON Schema."""
    return Literal.__getitem__(values)


def _preparable_provider(value: str) -> str:
    if value not in PREPARABLE_PROVIDER_IDS:
        raise ValueError(f"unknown provider; accepted values: {', '.join(PREPARABLE_PROVIDER_IDS)}")
    return value


PreparableProviderId = Annotated[
    str,
    AfterValidator(_preparable_provider),
    Field(json_schema_extra={"enum": PREPARABLE_PROVIDER_IDS}),
]


def _normalize_mcp_source_kind(value: Any) -> Any:
    if isinstance(value, Mapping) and "kind" not in value:
        value = dict(value)
        value["kind"] = "evidence" if "evidence_id" in value else "path"
    return value


class McpEvidenceSource(EvidenceSource):
    kind: Literal["evidence"] = "evidence"


McpSource = Annotated[
    PathSource | McpEvidenceSource,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_mcp_source_kind),
]


class AnalysisArguments(StrictModel):
    sources: list[McpSource] = Field(
        description="Ordered native paths or preserved evidence sources."
    )
    continuation: str | None = Field(
        default=None, description="Opaque continuation from next_page; copy its complete call."
    )
    page_size: int = Field(default=100, ge=1, le=MAX_ROWS, description="Evidence rows per page.")


class CaptureArguments(StrictModel):
    target: DirectTarget = Field(
        description="Explicit argv and absolute cwd for workload execution."
    )
    provider: StrictModel = Field(description="Collector and its typed settings.")
    preserve: bool = Field(
        default=False, description="Publish native artifacts as immutable evidence."
    )
    page_size: int = Field(default=100, ge=1, le=MAX_ROWS, description="Evidence rows per page.")


def analysis_arguments(capability: Capability) -> type[AnalysisArguments]:
    """Expose capability options as fields while retaining their validators."""
    path_model = create_model(
        _model_name(capability.id, "PathSource"),
        __base__=PathSource,
        format=(
            literal_type(capability.formats) | None,
            Field(default=None, description="Native format; omit for unambiguous detection."),
        ),
    )
    source_type = cast(Any, Annotated)[
        path_model | McpEvidenceSource,
        Field(discriminator="kind"),
        BeforeValidator(_normalize_mcp_source_kind),
    ]
    model = create_model(
        _model_name(capability.id, "Analysis"),
        __base__=cast(tuple[type[BaseModel], ...], (capability.model, AnalysisArguments)),
        __config__=ConfigDict(json_schema_extra={"examples": [analysis_example(capability)]}),
        sources=(
            GenericAlias(list, source_type),
            Field(
                description=f"Ordered artifacts in these formats: {', '.join(capability.formats)}.",
                min_length=capability.minimum_sources,
                max_length=capability.maximum_sources,
            ),
        ),
    )

    return cast(type[AnalysisArguments], model)


def capture_arguments(capability: Capability) -> type[CaptureArguments]:
    """Advertise only compatible collectors, including their exact typed fields."""
    providers: Any = None
    for contract in compatible_capture_providers(capability):
        provider_model = create_model(
            _model_name(contract.id, "Collector"),
            __base__=contract.argument_model,
            kind=(literal_type((contract.id,)), Field(description="Selected capture collector.")),
        )
        providers = provider_model if providers is None else providers | provider_model
    fields: dict[str, Any] = {
        "provider": (providers, Field(discriminator="kind", description="Compatible collector.")),
    }
    if capability.maximum_sources > 1:
        fields["experiment"] = (
            ExperimentDesign | None,
            Field(
                default=None, description="Randomized paired experiment; omit for one execution."
            ),
        )
    model = create_model(
        _model_name(capability.id, "Capture"),
        __base__=cast(tuple[type[BaseModel], ...], (capability.model, CaptureArguments)),
        __config__=ConfigDict(json_schema_extra={"examples": [capture_example(capability)]}),
        **fields,
    )

    return cast(type[CaptureArguments], model)


def option_example(capability: Capability) -> dict[str, Any]:
    example = dict(capability.model.model_json_schema().get("examples", [{}])[0])
    capability.model.model_validate(example)
    return example


def analysis_example(capability: Capability) -> dict[str, Any]:
    format_name = capability.formats[0]
    return {
        **option_example(capability),
        "sources": [
            {"path": f"/absolute/path/artifact-{index + 1}.{format_name}", "format": format_name}
            for index in range(capability.minimum_sources)
        ],
    }


def capture_example(capability: Capability) -> dict[str, Any]:
    provider = compatible_capture_providers(capability)[0]
    executable = "node" if provider.id.startswith("node-") else "python"
    workload = "workload.js" if executable == "node" else "workload.py"
    return {
        **option_example(capability),
        "target": {"argv": [executable, workload], "cwd": "/absolute/workdir"},
        "provider": {"kind": provider.id},
    }


def _model_name(identity: str, suffix: str) -> str:
    return "".join(word.title() for word in identity.replace("-", ".").split(".")) + suffix


class PrepareProvidersArguments(StrictModel):
    provider_ids: list[PreparableProviderId] = Field(
        description="Provider IDs to prepare together in one environment.",
        min_length=1,
        max_length=16,
    )
    timeout_seconds: int = Field(default=1_800, ge=1, le=3_600)


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
