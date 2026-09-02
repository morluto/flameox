"""Generate the typed MCP request models exposed for runtime capabilities."""

from __future__ import annotations

from functools import reduce
from operator import or_
from typing import Annotated, Any, Literal

from pydantic import Field, create_model

from flameox.runtime_contracts import (
    CAPTURE_PROVIDER_CONTRACTS,
    Capability,
    CaptureProviderContract,
    ExperimentDesign,
    StrictModel,
    compatible_capture_providers,
)


class SingleExecution(StrictModel):
    kind: Literal["single"] = Field(
        default="single", description="Execute the target once without a paired experiment."
    )


class ExperimentExecution(StrictModel):
    kind: Literal["experiment"] = Field(description="Execute a randomized paired experiment.")
    design: ExperimentDesign = Field(description="Paired experiment design and decision rule.")


Execution = Annotated[SingleExecution | ExperimentExecution, Field(discriminator="kind")]


def tool_stem(capability: Capability) -> str:
    """Return the stable MCP name fragment for a capability."""

    if capability.id == "artifact.preview":
        return "artifact"
    return capability.id.replace(".", "_")


def analysis_tool_name(capability: Capability) -> str:
    if capability.id == "artifact.preview":
        return "preview_artifact"
    return f"analyze_{tool_stem(capability)}"


def capture_tool_name(capability: Capability) -> str:
    if capability.id == "artifact.preview":
        return "capture_process_output"
    return f"capture_{tool_stem(capability)}"


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
