"""Small typed facts shared by Nsight Compute extraction and analysis."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from flameox.models import ContractModel


class NsightComputeReportLocation(ContractModel):
    """One provider-report location, relative to its immutable report artifact."""

    range_index: Annotated[int, Field(ge=0)]
    action_index: Annotated[int, Field(ge=0)]
    action_name: str = Field(min_length=1, max_length=500)


class NsightComputeRuleMessage(ContractModel):
    title: str | None = Field(default=None, max_length=2_000)
    message: str = Field(min_length=1, max_length=8_000)
    provider_type: str | None = Field(default=None, max_length=100)


type NsightComputeSpeedupMeaning = Literal[
    "global_runtime_reduction",
    "local_hardware_efficiency_increase",
    "unknown",
]


class NsightComputeSpeedupEstimation(ContractModel):
    """A provider-reported guided-analysis estimate with its documented meaning."""

    estimated_speedup: Annotated[float, Field(ge=0)]
    meaning: NsightComputeSpeedupMeaning
    provider_type: str | None = Field(default=None, max_length=100)


class NsightComputeFocusMetric(ContractModel):
    name: str = Field(min_length=1, max_length=500)
    value: float | None = None
    severity: str | None = Field(default=None, max_length=100)
    info: str | None = Field(default=None, max_length=2_000)


class NsightComputeProviderRuleFact(ContractModel):
    """Bounded guided-analysis facts from the official report interface."""

    location: NsightComputeReportLocation
    rule_identifier: str = Field(min_length=1, max_length=500)
    section_identifier: str = Field(min_length=1, max_length=500)
    rule_message: NsightComputeRuleMessage | None = None
    speedup_estimation: NsightComputeSpeedupEstimation | None = None
    focus_metrics: tuple[NsightComputeFocusMetric, ...] = Field(default=(), max_length=32)
