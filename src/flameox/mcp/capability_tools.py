"""Generate the typed MCP request models exposed for runtime capabilities."""

from __future__ import annotations

from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import Field, create_model
from pydantic.fields import FieldInfo

from flameox.runtime_contracts import (
    CAPABILITIES,
    CAPTURE_PROVIDER_CONTRACTS,
    Capability,
    CaptureProviderContract,
    DirectTarget,
    ExperimentDesign,
    Source,
    StrictModel,
    compatible_capture_providers,
)


def _provider_model(contract: CaptureProviderContract) -> type[StrictModel]:
    return create_model(
        f"{contract.id.title().replace('-', '')}Provider",
        __base__=StrictModel,
        kind=(
            Literal[contract.id],
            Field(description=f"Use the {contract.id} capture provider."),
        ),
        options=(
            contract.argument_model,
            Field(
                default_factory=contract.argument_model,
                description=f"Provider settings; produces {contract.artifact_description}.",
            ),
        ),
    )


PROVIDER_MODELS = {
    provider_id: _provider_model(contract)
    for provider_id, contract in CAPTURE_PROVIDER_CONTRACTS.items()
}


def capture_provider_type(capability: Capability) -> Any | None:
    providers = compatible_capture_providers(capability)
    if not providers:
        return None
    provider_models = tuple(PROVIDER_MODELS[contract.id] for contract in providers)
    provider_type: Any = provider_models[0]
    if len(provider_models) > 1:
        provider_type = Annotated[
            reduce(or_, provider_models),
            Field(discriminator="kind"),
        ]
    return provider_type


def _model_name(prefix: str, capability: Capability) -> str:
    stem = "".join(part.title() for part in capability.id.replace(".", "_").split("_"))
    return f"{prefix}{stem}Request"


def _options_field(capability: Capability) -> FieldInfo:
    if any(field.is_required() for field in capability.model.model_fields.values()):
        return cast(FieldInfo, Field(description=f"Options for {capability.summary.lower()}"))
    return cast(
        FieldInfo,
        Field(
            default_factory=capability.model,
            description=f"Options for {capability.summary.lower()}",
        ),
    )


def _analysis_request_model(capability: Capability) -> type[StrictModel]:
    return create_model(
        _model_name("Analyze", capability),
        __base__=StrictModel,
        __doc__=(
            f"{capability.summary} Accepted formats: {', '.join(capability.formats)}. "
            f"Limitation: {capability.limitation}"
        ),
        capability_id=(
            Literal[capability.id],
            Field(description="Evidence capability selected for this request."),
        ),
        sources=(
            list[Source],
            Field(
                description="Native paths or preserved evidence artifacts to analyze.",
                min_length=capability.minimum_sources,
                max_length=capability.maximum_sources,
            ),
        ),
        options=(capability.model, _options_field(capability)),
        continuation=(
            str | None,
            Field(
                default=None,
                description="Opaque token from next_page; pass the complete request unchanged.",
            ),
        ),
    )


def _capture_request_model(capability: Capability, provider_type: Any) -> type[StrictModel]:
    experiment_field: dict[str, Any] = {}
    if capability.maximum_sources > 1:
        experiment_field["experiment"] = (
            ExperimentDesign | None,
            Field(
                default=None,
                description="Optional paired experiment design; omit to execute the target once.",
            ),
        )
    return create_model(
        _model_name("Capture", capability),
        __base__=StrictModel,
        __doc__=(
            f"Execute a typed target and {capability.summary.lower()} "
            "The workload may have external side effects."
        ),
        capability_id=(
            Literal[capability.id],
            Field(description="Evidence capability selected for this request."),
        ),
        target=(DirectTarget, Field(description="Process target to execute and capture.")),
        provider=(
            provider_type,
            Field(description="Compatible capture provider and its typed settings."),
        ),
        options=(capability.model, _options_field(capability)),
        preserve=(
            bool,
            Field(
                default=False,
                description="Preserve the result and native artifacts as immutable evidence.",
            ),
        ),
        **experiment_field,
    )


class AnalysisRequestBase(StrictModel):
    capability_id: str
    sources: list[Source]
    options: StrictModel
    continuation: str | None = None


class CaptureRequestBase(StrictModel):
    capability_id: str
    target: DirectTarget
    provider: Any
    experiment: ExperimentDesign | None = None
    options: StrictModel
    preserve: bool = False


ANALYSIS_REQUEST_MODELS = tuple(_analysis_request_model(item) for item in CAPABILITIES)

CAPTURE_REQUEST_MODELS = tuple(
    _capture_request_model(item, provider_type)
    for item in CAPABILITIES
    if (provider_type := capture_provider_type(item)) is not None
)
if TYPE_CHECKING:
    AnalysisRequest = AnalysisRequestBase
    CaptureRequest = CaptureRequestBase
else:
    AnalysisRequest = Annotated[
        reduce(or_, ANALYSIS_REQUEST_MODELS),
        Field(discriminator="capability_id"),
    ]
    CaptureRequest = Annotated[
        reduce(or_, CAPTURE_REQUEST_MODELS),
        Field(discriminator="capability_id"),
    ]
