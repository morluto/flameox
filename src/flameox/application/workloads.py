from __future__ import annotations

import os
import re
import shutil
import tomllib
from collections.abc import Mapping
from ipaddress import ip_address
from pathlib import Path, PurePath
from string import Formatter
from typing import Annotated, Literal, cast
from urllib.parse import urlsplit

import tomlkit
from pydantic import (
    Discriminator,
    Field,
    JsonValue,
    Tag,
    TypeAdapter,
    field_validator,
    model_validator,
)
from tomlkit.exceptions import ParseError
from tomlkit.items import Table

from flameox.adapters.builtins import BUILTIN_ADAPTERS
from flameox.adapters.registry import AdapterRegistry
from flameox.application.capabilities import CapabilityService
from flameox.application.inference_providers import _loopback_http_url
from flameox.atomic import atomic_write_text
from flameox.domain import (
    CapabilityReport,
    CapabilityStatus,
    CommandSpec,
    CursorCodec,
    DomainError,
    ErrorCode,
    OracleStrength,
    WorkloadDefinition,
    WorkloadInstance,
    digest_model,
)
from flameox.domain.models import Digest
from flameox.models import ContractModel
from flameox.storage import Workspace

Scalar = str | int | float | bool

RUNTIME_RESOURCE_METRICS = frozenset(
    {
        "runtime_resource.peak_rss_bytes",
        "runtime_resource.minimum_free_bytes",
        "runtime_resource.staging_growth_bytes",
    }
)


class WorkloadOracleConfig(ContractModel):
    strength: OracleStrength = OracleStrength.EXECUTION_CHECK
    argv: Annotated[tuple[str, ...], Field(max_length=1_024)] = ()
    receipt_schema: Literal["flameox.oracle-receipt.v1"] | None = None


class WorkloadRequirementsConfig(ContractModel):
    executables: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    python_distributions: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    capabilities: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    optional: Annotated[tuple[str, ...], Field(max_length=192)] = ()
    active: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    allow_exploratory: bool = False

    @model_validator(mode="after")
    def references_are_unique_and_declared(self) -> WorkloadRequirementsConfig:
        declared = (*self.executables, *self.python_distributions, *self.capabilities)
        if len(set(declared)) != len(declared):
            raise ValueError("workload requirements must be unique")
        if not set(self.optional).issubset(declared):
            raise ValueError("optional requirements must also be declared")
        if not set(self.active).issubset(self.capabilities):
            raise ValueError("active requirements must be declared capabilities")
        return self


AcceleratorIdentityRequirement = Literal[
    "cuda.driver",
    "cuda.runtime",
    "cuda.devices",
    "cuda.peer_topology",
]


class WorkloadEnvironmentIdentityConfig(ContractModel):
    required: Annotated[tuple[AcceleratorIdentityRequirement, ...], Field(max_length=4)] = ()

    @model_validator(mode="after")
    def requirements_are_unique(self) -> WorkloadEnvironmentIdentityConfig:
        if len(set(self.required)) != len(self.required):
            raise ValueError("environment identity requirements must be unique")
        return self


class WorkloadIdentityConfig(ContractModel):
    python_modules: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    native_files: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    environment: WorkloadEnvironmentIdentityConfig = Field(
        default_factory=WorkloadEnvironmentIdentityConfig
    )


class WorkloadConfig(ContractModel):
    argv: Annotated[tuple[str, ...], Field(min_length=1, max_length=1_024)]
    cwd: str = "."
    timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300
    parameters: dict[str, tuple[Scalar, ...]] = Field(
        default_factory=dict,
        max_length=128,
    )
    environment: dict[str, str] = Field(default_factory=dict, max_length=128)
    oracle: WorkloadOracleConfig | None = None
    requirements: WorkloadRequirementsConfig = Field(default_factory=WorkloadRequirementsConfig)
    writable_paths: Annotated[tuple[str, ...], Field(max_length=16)] = ()
    identity: WorkloadIdentityConfig = Field(default_factory=WorkloadIdentityConfig)

    @model_validator(mode="after")
    def validate_templates(self) -> WorkloadConfig:
        fields: set[str] = set()
        for value in (*self.argv, self.cwd, *self.environment.values()):
            fields.update(_template_fields(value))
        if self.oracle is not None:
            for value in self.oracle.argv:
                fields.update(_template_fields(value))
        undeclared = fields - set(self.parameters)
        if undeclared:
            raise ValueError(
                "template fields are not declared parameters: " + ", ".join(sorted(undeclared))
            )
        return self


class _CommonExperimentConfig(ContractModel):
    workload: str
    design: Literal[
        "randomized_complete_blocks",
        "randomized",
        "fixed_order",
    ] = "randomized_complete_blocks"
    blocks: Annotated[int, Field(gt=0, le=1_000)] = 1
    max_trials: Annotated[int, Field(gt=0, le=100_000)] = 10_000
    primary_metric: str = "categorical_outcome"
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"] = "neutral"
    estimand: str = "median_paired_log_ratio"
    practical_threshold: Annotated[float, Field(ge=0)] = 0
    confidence_level: Annotated[float, Field(gt=0, lt=1)] = 0.95
    random_seed: Annotated[int, Field(ge=0)] = 0

    @field_validator("primary_metric")
    @classmethod
    def runtime_resource_metric_is_known(cls, value: str) -> str:
        if value.startswith("runtime_resource.") and value not in RUNTIME_RESOURCE_METRICS:
            raise ValueError(
                "runtime-resource primary_metric must be one of "
                "runtime_resource.peak_rss_bytes, runtime_resource.minimum_free_bytes, "
                "or runtime_resource.staging_growth_bytes"
            )
        return value


class _FactorExperimentConfig(_CommonExperimentConfig):
    # Empty legacy fields remain in the wire model so existing config digests
    # and schema-1 round trips retain their shape.
    variants: Annotated[tuple[Scalar, ...], Field(max_length=0)] = ()
    factors: dict[str, Annotated[tuple[Scalar, ...], Field(min_length=1, max_length=32)]] = Field(
        min_length=1,
        max_length=8,
    )
    exclude: Annotated[tuple[dict[str, Scalar], ...], Field(max_length=1_000)] = ()
    treatment_factor: str
    scaling_parameter: Literal[None] = None
    scaling_values: Annotated[tuple[Scalar, ...], Field(max_length=0)] = ()

    @model_validator(mode="after")
    def treatment_is_declared(self) -> _FactorExperimentConfig:
        if self.treatment_factor not in self.factors:
            raise ValueError("factor experiments require a declared treatment_factor")
        return self


class _CartesianFactorExperimentConfig(_FactorExperimentConfig):
    combination_policy: Literal["cartesian"] = "cartesian"
    combinations: Annotated[tuple[dict[str, Scalar], ...], Field(max_length=0)] = ()


class _ExplicitFactorExperimentConfig(_FactorExperimentConfig):
    combination_policy: Literal["explicit"]
    combinations: Annotated[
        tuple[dict[str, Scalar], ...],
        Field(min_length=1, max_length=10_000),
    ]


class _LegacyExperimentConfig(_CommonExperimentConfig):
    # Legacy variants remain a supported schema-1 input. Parsing gives them a
    # distinct type before planning projects them into the factor model.
    variants: Annotated[tuple[str, ...], Field(min_length=1, max_length=16)]
    factors: Annotated[dict[str, tuple[Scalar, ...]], Field(max_length=0)] = Field(
        default_factory=dict
    )
    combination_policy: Literal["cartesian"] = "cartesian"
    combinations: Annotated[tuple[dict[str, Scalar], ...], Field(max_length=0)] = ()
    exclude: Annotated[tuple[dict[str, Scalar], ...], Field(max_length=0)] = ()
    treatment_factor: Literal[None] = None

    @field_validator("variants")
    @classmethod
    def variants_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("experiment variants must be unique")
        return value


class _UnscaledLegacyExperimentConfig(_LegacyExperimentConfig):
    scaling_parameter: Literal[None] = None
    scaling_values: Annotated[tuple[Scalar, ...], Field(max_length=0)] = ()


class _ScaledLegacyExperimentConfig(_LegacyExperimentConfig):
    scaling_parameter: str
    scaling_values: Annotated[tuple[Scalar, ...], Field(min_length=1, max_length=1_000)]

    @field_validator("scaling_values")
    @classmethod
    def scaling_values_are_unique(cls, value: tuple[Scalar, ...]) -> tuple[Scalar, ...]:
        if len(set(value)) != len(value):
            raise ValueError("experiment scaling values must be unique")
        return value


class _PerformanceExperimentConfig(_CommonExperimentConfig):
    analysis: Literal["performance"] = "performance"
    outcome_goal: Literal[None] = None
    minimum_attempts: Literal[None] = None
    maximum_attempts: Literal[None] = None


class _OutcomeExperimentConfig(_CommonExperimentConfig):
    analysis: Literal["outcome"]
    outcome_goal: Literal["equivalence", "absence_of_failure", "bounded_rate"]
    minimum_attempts: Annotated[int, Field(gt=0, le=1_000)] | None = None
    maximum_attempts: Annotated[int, Field(gt=0, le=1_000)] | None = None

    @model_validator(mode="after")
    def fixed_blocks_fit_attempt_bounds(self) -> _OutcomeExperimentConfig:
        minimum = self.minimum_attempts or self.blocks
        maximum = self.maximum_attempts or self.blocks
        if minimum > self.blocks or maximum < self.blocks or minimum > maximum:
            raise ValueError("fixed blocks must lie within declared attempt bounds")
        return self


class _CartesianFactorPerformanceExperimentConfig(
    _CartesianFactorExperimentConfig,
    _PerformanceExperimentConfig,
):
    pass


class _CartesianFactorOutcomeExperimentConfig(
    _CartesianFactorExperimentConfig,
    _OutcomeExperimentConfig,
):
    pass


class _ExplicitFactorPerformanceExperimentConfig(
    _ExplicitFactorExperimentConfig,
    _PerformanceExperimentConfig,
):
    pass


class _ExplicitFactorOutcomeExperimentConfig(
    _ExplicitFactorExperimentConfig,
    _OutcomeExperimentConfig,
):
    pass


class _UnscaledLegacyPerformanceExperimentConfig(
    _UnscaledLegacyExperimentConfig,
    _PerformanceExperimentConfig,
):
    pass


class _UnscaledLegacyOutcomeExperimentConfig(
    _UnscaledLegacyExperimentConfig,
    _OutcomeExperimentConfig,
):
    pass


class _ScaledLegacyPerformanceExperimentConfig(
    _ScaledLegacyExperimentConfig,
    _PerformanceExperimentConfig,
):
    pass


class _ScaledLegacyOutcomeExperimentConfig(
    _ScaledLegacyExperimentConfig,
    _OutcomeExperimentConfig,
):
    pass


def _experiment_config_kind(value: object) -> str:
    if isinstance(value, Mapping):
        analysis = value.get("analysis", "performance")
        if value.get("factors"):
            shape = f"factor_{value.get('combination_policy', 'cartesian')}"
        elif value.get("scaling_parameter") is not None or value.get("scaling_values"):
            shape = "legacy_scaled"
        else:
            shape = "legacy_unscaled"
        return f"{shape}_{analysis}"

    analysis = "outcome" if isinstance(value, _OutcomeExperimentConfig) else "performance"
    if isinstance(value, _ExplicitFactorExperimentConfig):
        shape = "factor_explicit"
    elif isinstance(value, _CartesianFactorExperimentConfig):
        shape = "factor_cartesian"
    elif isinstance(value, _ScaledLegacyExperimentConfig):
        shape = "legacy_scaled"
    else:
        shape = "legacy_unscaled"
    return f"{shape}_{analysis}"


type ExperimentConfig = Annotated[
    Annotated[_CartesianFactorPerformanceExperimentConfig, Tag("factor_cartesian_performance")]
    | Annotated[_CartesianFactorOutcomeExperimentConfig, Tag("factor_cartesian_outcome")]
    | Annotated[_ExplicitFactorPerformanceExperimentConfig, Tag("factor_explicit_performance")]
    | Annotated[_ExplicitFactorOutcomeExperimentConfig, Tag("factor_explicit_outcome")]
    | Annotated[_UnscaledLegacyPerformanceExperimentConfig, Tag("legacy_unscaled_performance")]
    | Annotated[_UnscaledLegacyOutcomeExperimentConfig, Tag("legacy_unscaled_outcome")]
    | Annotated[_ScaledLegacyPerformanceExperimentConfig, Tag("legacy_scaled_performance")]
    | Annotated[_ScaledLegacyOutcomeExperimentConfig, Tag("legacy_scaled_outcome")],
    Discriminator(_experiment_config_kind),
]

_EXPERIMENT_CONFIG_ADAPTER: TypeAdapter[ExperimentConfig] = TypeAdapter(ExperimentConfig)


def parse_experiment_config(value: object) -> ExperimentConfig:
    """Parse a flat schema-1 experiment into its single legal configuration case."""

    return _EXPERIMENT_CONFIG_ADAPTER.validate_python(value)


class LatencyFault(ContractModel):
    type: Literal["latency"]
    latency_ms: int = Field(gt=0, le=3_600_000)
    jitter_ms: int = Field(default=0, ge=0, le=3_600_000)
    stream: Literal["upstream", "downstream"] = "downstream"
    toxicity: float = Field(default=1.0, ge=0, le=1)


class TimeoutFault(ContractModel):
    type: Literal["timeout"]
    timeout_ms: int = Field(gt=0, le=3_600_000)
    stream: Literal["upstream", "downstream"] = "downstream"
    toxicity: float = Field(default=1.0, ge=0, le=1)


class ResetPeerFault(ContractModel):
    type: Literal["reset_peer"]
    stream: Literal["upstream", "downstream"] = "downstream"
    toxicity: float = Field(default=1.0, ge=0, le=1)


class BandwidthFault(ContractModel):
    type: Literal["bandwidth"]
    bandwidth_limit: int = Field(gt=0, le=10_000_000_000)
    stream: Literal["upstream", "downstream"] = "downstream"
    toxicity: float = Field(default=1.0, ge=0, le=1)


class SlicerFault(ContractModel):
    type: Literal["slicer"]
    average_size: int = Field(gt=0, le=10_000_000)
    size_variation: int = Field(default=0, ge=0, le=10_000_000)
    delay_ms: int = Field(default=0, ge=0, le=3_600_000)
    stream: Literal["upstream", "downstream"] = "downstream"
    toxicity: float = Field(default=1.0, ge=0, le=1)


class LimitDataFault(ContractModel):
    type: Literal["limit_data"]
    bytes: int = Field(gt=0, le=10_000_000_000)
    stream: Literal["upstream", "downstream"] = "downstream"
    toxicity: float = Field(default=1.0, ge=0, le=1)


class SlowCloseFault(ContractModel):
    type: Literal["slow_close"]
    delay_ms: int = Field(gt=0, le=3_600_000)
    stream: Literal["upstream", "downstream"] = "downstream"
    toxicity: float = Field(default=1.0, ge=0, le=1)


class ProxyFault(ContractModel):
    type: Literal["proxy"]
    enabled: bool = True


FaultScenario = Annotated[
    LatencyFault
    | TimeoutFault
    | ResetPeerFault
    | BandwidthFault
    | SlicerFault
    | LimitDataFault
    | SlowCloseFault
    | ProxyFault,
    Field(discriminator="type"),
]


class FaultExperimentConfig(ContractModel):
    workload: str
    endpoint_parameter: str
    upstream_host: str
    upstream_port: int = Field(gt=0, le=65_535)
    endpoint_template: str
    scenarios: dict[str, FaultScenario] = Field(min_length=1, max_length=64)
    blocks: int = Field(default=1, gt=0, le=1_000)
    repetitions: int = Field(default=1, gt=0, le=1_000)
    primary_metric: str = "categorical_outcome"
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"] = "neutral"
    estimand: str = "median_paired_log_ratio"
    practical_threshold: float = Field(default=0, ge=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    random_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_transport_scope(self) -> FaultExperimentConfig:
        if not ip_address(self.upstream_host).is_loopback:
            raise ValueError("fault experiment upstream_host must be a loopback IP literal")
        try:
            parsed = tuple(Formatter().parse(self.endpoint_template))
            fields = {field for _, field, _, _ in parsed if field is not None}
            if fields != {"host", "port"} or any(
                field not in {None, "host", "port"} or format_spec or conversion
                for _, field, format_spec, conversion in parsed
            ):
                raise ValueError
            self.endpoint_template.format(host="127.0.0.1", port=1)
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError("endpoint_template must contain only {host} and {port}") from error
        if "baseline" in self.scenarios:
            raise ValueError("baseline is reserved for the synthetic passthrough treatment")
        if self.blocks * self.repetitions * (len(self.scenarios) + 1) > 100_000:
            raise ValueError("fault experiment schedule exceeds the 100000-trial bound")
        for scenario in self.scenarios.values():
            if isinstance(scenario, ProxyFault) and scenario.enabled:
                raise ValueError("proxy treatment is non-discriminating when enabled")
            if (
                hasattr(scenario, "jitter_ms")
                and getattr(scenario, "toxicity", 1.0) == 1
                and getattr(scenario, "jitter_ms", 0) != 0
            ):
                raise ValueError("deterministic latency faults require zero jitter")
            if getattr(scenario, "toxicity", 1.0) < 1 and self.repetitions < 2:
                raise ValueError("stochastic fault scenarios require repeated trials")
        return self


class _CommonInferenceServerConfig(ContractModel):
    base_url: str = "http://127.0.0.1:8000"
    model: Annotated[str, Field(min_length=1, max_length=500)]
    model_revision: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    tokenizer: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    tokenizer_revision: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    quantization: Annotated[str, Field(min_length=1, max_length=100)] | None = None

    @field_validator("base_url")
    @classmethod
    def base_url_is_loopback_http(cls, value: str) -> str:
        _loopback_http_url(value)
        return value


class _VllmInferenceServerConfig(_CommonInferenceServerConfig):
    provider: Literal["vllm"] = "vllm"
    benchmark_python: Literal[None] = None


class _SglangInferenceServerConfig(_CommonInferenceServerConfig):
    provider: Literal["sglang"]
    benchmark_python: str

    @field_validator("benchmark_python")
    @classmethod
    def benchmark_launcher_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError(
                "sglang inference servers require an absolute benchmark_python launcher"
            )
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_is_root(cls, value: str) -> str:
        if urlsplit(value).path not in ("", "/"):
            raise ValueError("sglang inference servers require a root base_url in v1")
        return value


class _ManagedInferenceServerConfig(_CommonInferenceServerConfig):
    mode: Literal["managed"]
    workload: str

    @field_validator("base_url")
    @classmethod
    def base_url_uses_ip_literal(cls, value: str) -> str:
        # Managed servers are bind-probed by the broker, which deliberately
        # accepts an IP literal so it can prove ownership of the endpoint.
        hostname = urlsplit(value).hostname
        if hostname is not None and hostname.lower() == "localhost":
            raise ValueError("managed inference servers require an IP-literal loopback base_url")
        return value


class _ExistingLocalInferenceServerConfig(_CommonInferenceServerConfig):
    mode: Literal["existing_local"]
    workload: Literal[None] = None


class _ManagedVllmInferenceServerConfig(
    _ManagedInferenceServerConfig,
    _VllmInferenceServerConfig,
):
    pass


class _ExistingLocalVllmInferenceServerConfig(
    _ExistingLocalInferenceServerConfig,
    _VllmInferenceServerConfig,
):
    pass


class _ManagedSglangInferenceServerConfig(
    _ManagedInferenceServerConfig,
    _SglangInferenceServerConfig,
):
    pass


class _ExistingLocalSglangInferenceServerConfig(
    _ExistingLocalInferenceServerConfig,
    _SglangInferenceServerConfig,
):
    pass


def _inference_server_config_kind(value: object) -> str:
    if isinstance(value, Mapping):
        return f"{value.get('provider', 'vllm')}_{value.get('mode')}"
    provider = "sglang" if isinstance(value, _SglangInferenceServerConfig) else "vllm"
    mode = "managed" if isinstance(value, _ManagedInferenceServerConfig) else "existing_local"
    return f"{provider}_{mode}"


type InferenceServerConfig = Annotated[
    Annotated[_ManagedVllmInferenceServerConfig, Tag("vllm_managed")]
    | Annotated[_ExistingLocalVllmInferenceServerConfig, Tag("vllm_existing_local")]
    | Annotated[_ManagedSglangInferenceServerConfig, Tag("sglang_managed")]
    | Annotated[_ExistingLocalSglangInferenceServerConfig, Tag("sglang_existing_local")],
    Discriminator(_inference_server_config_kind),
]

_INFERENCE_SERVER_CONFIG_ADAPTER: TypeAdapter[InferenceServerConfig] = TypeAdapter(
    InferenceServerConfig
)


def parse_inference_server_config(value: object) -> InferenceServerConfig:
    """Parse a flat server declaration into one provider/lifecycle case."""

    return _INFERENCE_SERVER_CONFIG_ADAPTER.validate_python(value)


class _CommonInferenceScenarioConfig(ContractModel):
    """Fields shared by every maintained inference replay provider.

    Trace input, repetition, and timing-scale parameters are bounded and
    forward-compatible with the maintained AIPerf and vLLM bench providers.
    """

    server: str
    endpoint_type: Literal["chat", "completions"] = "chat"
    num_prompts: Annotated[int, Field(gt=0, le=10_000_000)] = 1
    concurrency: Annotated[int, Field(gt=0, le=100_000)] | None = None
    request_rate: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    warmup_request_count: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0
    semantic_oracle_workload: str | None = None


class _AIPerfInferenceScenarioConfig(_CommonInferenceScenarioConfig):
    provider: Literal["aiperf"]
    streaming: bool = True
    trace_artifact_id: Digest | None = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    speedup_ratio: Annotated[float, Field(gt=0, le=100)] = 1.0
    random_input_len: Literal[None] = None
    random_output_len: Literal[None] = None
    random_range_ratio: Literal[None] = None

    @model_validator(mode="after")
    def trace_options_are_consistent(self) -> _AIPerfInferenceScenarioConfig:
        if self.trace_artifact_id is None and self.speedup_ratio != 1.0:
            raise ValueError("speedup_ratio requires an aiperf trace_artifact_id")
        if self.burstiness is not None and self.request_rate is None:
            raise ValueError("burstiness requires request_rate")
        return self


class _VllmBenchInferenceScenarioConfig(_CommonInferenceScenarioConfig):
    provider: Literal["vllm_bench"]
    streaming: Literal[True] = True
    trace_artifact_id: Literal[None] = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    speedup_ratio: Annotated[float, Field(ge=1, le=1)] = 1.0
    random_input_len: Literal[None] = None
    random_output_len: Literal[None] = None
    random_range_ratio: Literal[None] = None

    @model_validator(mode="after")
    def burstiness_has_request_rate(self) -> _VllmBenchInferenceScenarioConfig:
        if self.burstiness is not None and self.request_rate is None:
            raise ValueError("burstiness requires request_rate")
        return self


class _SglangBenchInferenceScenarioConfig(_CommonInferenceScenarioConfig):
    provider: Literal["sglang_bench"]
    streaming: Literal[True] = True
    trace_artifact_id: Literal[None] = None
    burstiness: Literal[None] = None
    speedup_ratio: Annotated[float, Field(ge=1, le=1)] = 1.0
    random_input_len: Annotated[int, Field(gt=0, le=1_000_000)]
    random_output_len: Annotated[int, Field(gt=0, le=1_000_000)]
    random_range_ratio: Annotated[float, Field(gt=0, le=1)] | None = None


type InferenceScenarioConfig = Annotated[
    _AIPerfInferenceScenarioConfig
    | _VllmBenchInferenceScenarioConfig
    | _SglangBenchInferenceScenarioConfig,
    Field(discriminator="provider"),
]

_INFERENCE_SCENARIO_CONFIG_ADAPTER: TypeAdapter[InferenceScenarioConfig] = TypeAdapter(
    InferenceScenarioConfig
)


def parse_inference_scenario_config(value: object) -> InferenceScenarioConfig:
    """Parse a scenario declaration into one maintained provider case."""

    return _INFERENCE_SCENARIO_CONFIG_ADAPTER.validate_python(value)


class ProjectConfig(ContractModel):
    schema_version: Literal[1] = 1
    workloads: dict[str, WorkloadConfig] = Field(default_factory=dict, max_length=1_000)
    experiments: dict[str, ExperimentConfig] = Field(
        default_factory=dict,
        max_length=1_000,
    )
    fault_experiments: dict[str, FaultExperimentConfig] = Field(
        default_factory=dict,
        max_length=1_000,
    )
    inference_servers: dict[str, InferenceServerConfig] = Field(
        default_factory=dict,
        max_length=1_000,
    )
    inference_scenarios: dict[str, InferenceScenarioConfig] = Field(
        default_factory=dict,
        max_length=1_000,
    )

    @model_validator(mode="after")
    def experiments_reference_workloads(self) -> ProjectConfig:
        missing = sorted(
            {
                experiment.workload
                for experiment in self.experiments.values()
                if experiment.workload not in self.workloads
            }
        )
        missing_faults = sorted(
            {
                experiment.workload
                for experiment in self.fault_experiments.values()
                if experiment.workload not in self.workloads
            }
        )
        if missing_faults:
            missing.extend(missing_faults)
        if missing:
            raise ValueError("experiments reference unknown workloads: " + ", ".join(missing))
        for experiment in self.fault_experiments.values():
            if experiment.endpoint_parameter not in self.workloads[experiment.workload].parameters:
                raise ValueError(
                    "fault experiments must inject a declared workload parameter: "
                    + experiment.endpoint_parameter
                )
            workload = self.workloads[experiment.workload]
            template_values = (*workload.argv, workload.cwd, *workload.environment.values())
            if not any(
                experiment.endpoint_parameter in _template_fields(value)
                for value in template_values
            ):
                raise ValueError(
                    "fault experiment endpoint_parameter must be rendered by argv, cwd, "
                    "or environment"
                )
        return self

    @model_validator(mode="after")
    def inference_references_are_valid(self) -> ProjectConfig:
        missing_server_workloads = sorted(
            {
                server.workload
                for server in self.inference_servers.values()
                if isinstance(server, _ManagedInferenceServerConfig)
                and server.workload not in self.workloads
            }
        )
        if missing_server_workloads:
            raise ValueError(
                "inference servers reference unknown workloads: "
                + ", ".join(missing_server_workloads)
            )
        missing_servers = sorted(
            {
                scenario.server
                for scenario in self.inference_scenarios.values()
                if scenario.server not in self.inference_servers
            }
        )
        if missing_servers:
            raise ValueError(
                "inference scenarios reference unknown servers: " + ", ".join(missing_servers)
            )
        for scenario in self.inference_scenarios.values():
            server = self.inference_servers[scenario.server]
            if scenario.provider == "sglang_bench" and server.provider != "sglang":
                raise ValueError("sglang_bench scenarios require an sglang inference server")
            if scenario.provider != "sglang_bench" and server.provider == "sglang":
                raise ValueError("sglang inference servers require an sglang_bench scenario")
        missing_oracles = sorted(
            {
                scenario.semantic_oracle_workload
                for scenario in self.inference_scenarios.values()
                if scenario.semantic_oracle_workload is not None
                and scenario.semantic_oracle_workload not in self.workloads
            }
        )
        if missing_oracles:
            raise ValueError(
                "inference scenarios reference unknown oracle workloads: "
                + ", ".join(missing_oracles)
            )
        invalid_oracle_names: set[str] = set()
        for scenario in self.inference_scenarios.values():
            name = scenario.semantic_oracle_workload
            if name is None or name not in self.workloads:
                continue
            oracle = self.workloads[name].oracle
            if (
                oracle is None
                or oracle.strength is not OracleStrength.CONTRACT_CHECK
                or oracle.receipt_schema is None
            ):
                invalid_oracle_names.add(name)
        invalid_oracles = sorted(invalid_oracle_names)
        if invalid_oracles:
            raise ValueError(
                "inference oracle workloads must declare a contract-check receipt oracle: "
                + ", ".join(invalid_oracles)
            )
        return self


class ConfigureWorkloadRequest(ContractModel):
    name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        ),
    ]
    operation: Literal["create", "replace"]
    config: WorkloadConfig
    expected_configuration_id: str | None = None


class WorkloadConfigurationStatus(ContractModel):
    schema_version: Literal[1] = 1
    status: Literal["missing", "valid", "invalid"]
    config_path: Literal["flameox.toml"] = "flameox.toml"
    configuration_id: str | None = None
    workload_names: Annotated[
        tuple[Annotated[str, Field(max_length=100)], ...],
        Field(max_length=1_000),
    ] = ()
    diagnostics: Annotated[
        tuple[Annotated[str, Field(max_length=512)], ...],
        Field(max_length=8),
    ] = ()
    next_tool: Literal["configure_workload", "list_declared_workflows"] | None = None


class WorkloadConfigurationResult(ContractModel):
    schema_version: Literal[1] = 1
    action: Literal["created", "updated", "unchanged"]
    name: str
    configuration_id: str
    workload_definition_id: str
    configuration_source: Literal["agent"] = "agent"
    changed_paths: Annotated[
        tuple[Annotated[str, Field(max_length=200)], ...],
        Field(max_length=8),
    ]
    next_tool: Literal["list_declared_workflows"] = "list_declared_workflows"


class ConfigureInferenceServerRequest(ContractModel):
    name: Annotated[
        str,
        Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    operation: Literal["create", "replace"]
    config: InferenceServerConfig
    expected_configuration_id: Digest | None = None


class ConfigureInferenceScenarioRequest(ContractModel):
    name: Annotated[
        str,
        Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    operation: Literal["create", "replace"]
    config: InferenceScenarioConfig
    expected_configuration_id: Digest | None = None


class InferenceConfigurationResult(ContractModel):
    schema_version: Literal[1] = 1
    kind: Literal["server", "scenario"]
    action: Literal["created", "updated", "unchanged"]
    name: str
    configuration_id: Digest
    definition_id: Digest
    changed_paths: tuple[Literal["flameox.toml"], ...]


class InferenceConfigurationList(ContractModel):
    schema_version: Literal[1] = 1
    configuration_id: Digest
    servers: dict[str, InferenceServerConfig]
    scenarios: dict[str, InferenceScenarioConfig]


class ResolvedOracle(ContractModel):
    strength: OracleStrength
    command: CommandSpec
    receipt_schema: Literal["flameox.oracle-receipt.v1"] | None = None


class DeclaredWorkflowSummary(ContractModel):
    kind: Literal["workload", "experiment", "fault_experiment"]
    name: str
    definition_id: str
    parameter_names: tuple[str, ...] = ()
    oracle_strength: OracleStrength | None = None
    timeout_seconds: float | None = None


class DeclaredWorkflowList(ContractModel):
    schema_version: int = 1
    configuration_id: str
    workflows: tuple[DeclaredWorkflowSummary, ...]
    returned: int
    truncated: bool
    next_cursor: str | None


class DeclaredWorkflowDetail(ContractModel):
    schema_version: int = 1
    configuration_id: str
    summary: DeclaredWorkflowSummary
    allowed_parameters: dict[str, tuple[Scalar, ...]]
    workload_name: str | None = None
    factors: dict[str, tuple[Scalar, ...]] = Field(default_factory=dict)
    design: str | None = None
    blocks: int | None = None
    primary_metric: str | None = None
    polarity: str | None = None
    estimand: str | None = None
    validation_spec_id: str | None = None
    requirements: tuple[DeclaredWorkflowRequirement, ...] = ()
    adapter_options: tuple[AdapterOption, ...] = ()
    adapter_option_total: int = 0
    adapter_options_total: int = 0
    adapter_options_truncated: bool = False


class DeclaredWorkflowRequirement(ContractModel):
    name: str
    kind: Literal["executable", "python_distribution", "capability"]
    required: bool
    optional: bool
    probe_kind: Literal["passive", "active"]


class AdapterOption(ContractModel):
    adapter: str
    status: CapabilityStatus
    capability_status: CapabilityStatus
    planning_disposition: Literal[
        "ready",
        "active_probe_required",
        "unavailable",
        "unsupported",
        "degraded",
    ]
    required_preflight_mode: Literal["passive", "active"]
    permission_status: str | None = None
    supported_modes: tuple[str, ...] = ()
    supported_formats: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()


class WorkloadInspection(WorkloadDefinition):
    configuration_id: str
    requirements: tuple[DeclaredWorkflowRequirement, ...] = ()
    adapter_options: tuple[AdapterOption, ...] = ()
    adapter_option_total: int = 0
    adapter_options_total: int = 0
    adapter_options_truncated: bool = False


_TEMPLATE_FIELD = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _template_fields(value: str) -> set[str]:
    """Return only the explicit scalar placeholders in a workload value.

    Workload arguments are ordinary argv strings, so braces that are not an
    exact ``{parameter}`` token remain literal. Doubled braces retain the
    formatter-compatible escape for existing declarations.
    """
    return set(_TEMPLATE_FIELD.findall(value))


def _render(value: str, parameters: dict[str, Scalar]) -> str:
    fields = _template_fields(value)
    missing = fields - set(parameters)
    if missing:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"Missing workload parameter {sorted(missing)[0]!r}.",
        )

    rendered: list[str] = []
    index = 0
    while index < len(value):
        if value.startswith("{{", index):
            rendered.append("{")
            index += 2
            continue
        if value.startswith("}}", index):
            rendered.append("}")
            index += 2
            continue
        match = _TEMPLATE_FIELD.match(value, index)
        if match is not None:
            rendered.append(str(parameters[match.group(1)]))
            index = match.end()
            continue
        rendered.append(value[index])
        index += 1
    return "".join(rendered)


class WorkloadService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.project_config_path = workspace.project_root / "flameox.toml"

    def load(self) -> ProjectConfig:
        try:
            with self.project_config_path.open("rb") as stream:
                return ProjectConfig.model_validate(tomllib.load(stream))
        except FileNotFoundError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Project workload configuration is missing: flameox.toml",
                remediation=(
                    "Call configure_workload with a named command definition, then retry "
                    "discovery.",
                ),
                details={"next_tool": "configure_workload", "config_path": "flameox.toml"},
            ) from exc
        except (tomllib.TOMLDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Project workload configuration is invalid: {exc}",
                remediation=(
                    "Use configure_workload with a complete named definition; FlameOx will "
                    "replace the file only if the resulting project validates.",
                ),
                details={
                    "config_path": "flameox.toml",
                    "invalid_configuration": True,
                    "next_tool": "configure_workload",
                },
            ) from exc

    def configuration_status(self) -> WorkloadConfigurationStatus:
        if not self.project_config_path.exists():
            return WorkloadConfigurationStatus(
                status="missing",
                diagnostics=("No named workload configuration exists yet.",),
                next_tool="configure_workload",
            )
        try:
            project = self.load()
        except DomainError as error:
            return WorkloadConfigurationStatus(
                status="invalid",
                diagnostics=(error.message[:512],),
                next_tool="configure_workload",
            )
        has_workloads = bool(project.workloads)
        return WorkloadConfigurationStatus(
            status="valid",
            configuration_id=digest_model(project),
            workload_names=tuple(sorted(project.workloads)),
            diagnostics=(("No named workloads are declared yet.",) if not has_workloads else ()),
            next_tool=("list_declared_workflows" if has_workloads else "configure_workload"),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.load().workloads))

    def list_inference(self) -> InferenceConfigurationList:
        project = ProjectConfig() if not self.project_config_path.exists() else self.load()
        return InferenceConfigurationList(
            configuration_id=digest_model(project),
            servers=dict(sorted(project.inference_servers.items())),
            scenarios=dict(sorted(project.inference_scenarios.items())),
        )

    def list_declared(
        self,
        *,
        kind: Literal["workload", "experiment", "fault_experiment"],
        limit: int,
        cursor: str | None = None,
    ) -> DeclaredWorkflowList:
        project = self.load()
        configuration_id = digest_model(project)
        query_id = digest_model({"kind": kind})
        offset = 0
        if cursor is not None:
            position = CursorCodec.decode(
                cursor,
                namespace="declared_workflows",
                snapshot_id=configuration_id,
                scope_digest=query_id,
            )
            try:
                offset = int(position[0])
            except (IndexError, TypeError, ValueError) as exc:
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.") from exc

        names = sorted(
            project.workloads
            if kind == "workload"
            else (project.experiments if kind == "experiment" else project.fault_experiments)
        )
        summaries = tuple(self._workflow_summary(project, kind, name) for name in names)
        selected = summaries[offset : offset + limit]
        next_offset = offset + len(selected)
        next_cursor = (
            CursorCodec.encode(
                namespace="declared_workflows",
                snapshot_id=configuration_id,
                scope_digest=query_id,
                position=(next_offset,),
            )
            if next_offset < len(summaries)
            else None
        )
        return DeclaredWorkflowList(
            configuration_id=configuration_id,
            workflows=selected,
            returned=len(selected),
            truncated=next_cursor is not None,
            next_cursor=next_cursor,
        )

    def get_declared(
        self,
        *,
        kind: Literal["workload", "experiment", "fault_experiment"],
        name: str,
    ) -> DeclaredWorkflowDetail:
        project = self.load()
        configuration_id = digest_model(project)
        summary = self._workflow_summary(project, kind, name)
        workload_name = (
            name
            if kind == "workload"
            else (
                project.experiments[name].workload
                if kind == "experiment"
                else project.fault_experiments[name].workload
            )
        )
        workload_config = project.workloads[workload_name]
        requirements, adapter_options, option_total, options_truncated = self._inspection_fields(
            workload_config
        )
        if kind == "workload":
            config = project.workloads[name]
            definition = self.definition(name)
            return DeclaredWorkflowDetail(
                configuration_id=configuration_id,
                summary=summary,
                allowed_parameters=config.parameters,
                validation_spec_id=definition.validation_spec_id,
                requirements=requirements,
                adapter_options=adapter_options,
                adapter_option_total=option_total,
                adapter_options_total=option_total,
                adapter_options_truncated=options_truncated,
            )
        if kind == "fault_experiment":
            fault = project.fault_experiments[name]
            return DeclaredWorkflowDetail(
                configuration_id=configuration_id,
                summary=summary,
                allowed_parameters=workload_config.parameters,
                workload_name=fault.workload,
                blocks=fault.blocks,
                primary_metric=fault.primary_metric,
                polarity=fault.polarity,
                estimand=fault.estimand,
                validation_spec_id=self.definition(fault.workload).validation_spec_id,
                requirements=requirements,
                adapter_options=adapter_options,
                adapter_option_total=option_total,
                adapter_options_total=option_total,
                adapter_options_truncated=options_truncated,
            )
        experiment = project.experiments[name]
        workload = project.workloads[experiment.workload]
        return DeclaredWorkflowDetail(
            configuration_id=configuration_id,
            summary=summary,
            allowed_parameters=workload.parameters,
            workload_name=experiment.workload,
            factors=experiment.factors,
            design=experiment.design,
            blocks=experiment.blocks,
            primary_metric=experiment.primary_metric,
            polarity=experiment.polarity,
            estimand=experiment.estimand,
            validation_spec_id=self.definition(experiment.workload).validation_spec_id,
            requirements=requirements,
            adapter_options=adapter_options,
            adapter_option_total=option_total,
            adapter_options_total=option_total,
            adapter_options_truncated=options_truncated,
        )

    def inspect(self, name: str) -> WorkloadInspection:
        project = self.load()
        definition = self.definition(name)
        config = project.workloads[name]
        requirements, options, total, truncated = self._inspection_fields(config)
        return WorkloadInspection(
            **definition.model_dump(mode="python"),
            configuration_id=digest_model(project),
            requirements=requirements,
            adapter_options=options,
            adapter_option_total=total,
            adapter_options_total=total,
            adapter_options_truncated=truncated,
        )

    def _inspection_fields(
        self,
        config: WorkloadConfig,
    ) -> tuple[
        tuple[DeclaredWorkflowRequirement, ...],
        tuple[AdapterOption, ...],
        int,
        bool,
    ]:
        requirement_groups: tuple[
            tuple[Literal["executable", "python_distribution", "capability"], tuple[str, ...]],
            ...,
        ] = (
            ("executable", config.requirements.executables),
            ("python_distribution", config.requirements.python_distributions),
            ("capability", config.requirements.capabilities),
        )
        requirements = tuple(
            DeclaredWorkflowRequirement(
                name=name,
                kind=kind,
                required=name not in config.requirements.optional,
                optional=name in config.requirements.optional,
                probe_kind=("active" if name in config.requirements.active else "passive"),
            )
            for kind, values in requirement_groups
            for name in values
        )
        approved_third_party = {
            item.adapter
            for item in AdapterRegistry(self.workspace).discover().adapters
            if item.approved
        }
        reports = [
            item
            for item in CapabilityService(self.workspace).list().capabilities
            if (
                bool(BUILTIN_ADAPTERS.get(item.adapter, None))
                and bool(BUILTIN_ADAPTERS[item.adapter].artifact_kinds)
            )
            or item.adapter in approved_third_party
        ]
        reports.sort(key=lambda item: item.adapter)
        options = tuple(self._adapter_option(item) for item in reports[:64])
        return requirements, options, len(reports), len(reports) > len(options)

    @staticmethod
    def _adapter_option(capability: CapabilityReport) -> AdapterOption:
        permission_sensitive = capability.permission_status in {
            "unknown_until_active_probe",
            "not_exercised",
        }
        required_mode: Literal["passive", "active"] = (
            "active" if permission_sensitive else "passive"
        )
        if permission_sensitive and capability.status is CapabilityStatus.AVAILABLE:
            disposition: Literal[
                "ready",
                "active_probe_required",
                "unavailable",
                "unsupported",
                "degraded",
            ] = "active_probe_required"
        elif capability.status is CapabilityStatus.UNAVAILABLE:
            disposition = "unavailable"
        elif capability.status is CapabilityStatus.UNSUPPORTED_PLATFORM:
            disposition = "unsupported"
        elif capability.status is CapabilityStatus.DEGRADED:
            disposition = "degraded"
        else:
            disposition = "ready"
        return AdapterOption(
            adapter=capability.adapter,
            status=capability.status,
            capability_status=capability.status,
            planning_disposition=disposition,
            required_preflight_mode=required_mode,
            permission_status=capability.permission_status,
            supported_modes=capability.supported_modes,
            supported_formats=capability.supported_formats,
            features=capability.features,
            limitations=capability.limitations,
            remediation=capability.remediation,
        )

    def _workflow_summary(
        self,
        project: ProjectConfig,
        kind: Literal["workload", "experiment", "fault_experiment"],
        name: str,
    ) -> DeclaredWorkflowSummary:
        try:
            if kind == "workload":
                workload_config = project.workloads[name]
                definition = self.definition(name)
                return DeclaredWorkflowSummary(
                    kind=kind,
                    name=name,
                    definition_id=definition.workload_definition_id,
                    parameter_names=tuple(sorted(workload_config.parameters)),
                    oracle_strength=(
                        workload_config.oracle.strength
                        if workload_config.oracle is not None
                        else None
                    ),
                    timeout_seconds=workload_config.timeout_seconds,
                )
            if kind == "fault_experiment":
                fault_config = project.fault_experiments[name]
                workload_config = project.workloads[fault_config.workload]
                return DeclaredWorkflowSummary(
                    kind=kind,
                    name=name,
                    definition_id=digest_model(fault_config),
                    parameter_names=tuple(sorted(workload_config.parameters)),
                    oracle_strength=(
                        workload_config.oracle.strength
                        if workload_config.oracle is not None
                        else None
                    ),
                    timeout_seconds=workload_config.timeout_seconds,
                )
            experiment_config = project.experiments[name]
            return DeclaredWorkflowSummary(
                kind=kind,
                name=name,
                definition_id=digest_model(experiment_config),
                parameter_names=tuple(
                    sorted(project.workloads[experiment_config.workload].parameters)
                ),
            )
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unknown declared {kind} {name!r}.",
                remediation=(f"Call list_declared_workflows with kind={kind!r}.",),
                details={"next_tool": "list_declared_workflows", "kind": kind},
            ) from exc

    def definition(self, name: str) -> WorkloadDefinition:
        config = self._selected(name)
        content = self._definition_content(name, config)
        definition_id = digest_model(content)
        return WorkloadDefinition(
            workload_definition_id=definition_id,
            name=name,
            command_template=config.argv,
            parameter_names=tuple(sorted(config.parameters)),
            validation_spec_id=(
                digest_model(config.oracle.model_dump(mode="json"))
                if config.oracle is not None
                else None
            ),
        )

    def configure(self, request: ConfigureWorkloadRequest) -> WorkloadConfigurationResult:
        name = request.name
        config = request.config
        with self.workspace.write_locked():
            existing_text = ""
            current_id: str | None = None
            recovered_invalid = False
            recovered_text: str | None = None
            if self.project_config_path.exists():
                existing_text = self.project_config_path.read_text(encoding="utf-8")
                try:
                    project = self.load()
                    current_id = digest_model(project)
                except DomainError as error:
                    if request.operation != "create":
                        raise
                    try:
                        recovered_text = self._render_project_config(existing_text, name, config)
                        project = ProjectConfig.model_validate(tomllib.loads(recovered_text))
                    except (ParseError, tomllib.TOMLDecodeError, ValueError) as candidate_error:
                        raise DomainError(
                            ErrorCode.WORKSPACE_INVALID,
                            "The existing flameox.toml is invalid and the proposed workload "
                            "does not produce a valid project configuration.",
                            remediation=(
                                "Repair flameox.toml manually, then call "
                                "workload_configuration_status again.",
                            ),
                            details={
                                "config_path": "flameox.toml",
                                "invalid_configuration": True,
                                "next_tool": "manual",
                                "diagnostic": str(candidate_error)[:500],
                            },
                        ) from error
                    recovered_invalid = True
            else:
                project = ProjectConfig()

            existing = project.workloads.get(name)
            if request.operation == "create":
                if existing is not None and existing != config:
                    raise DomainError(
                        ErrorCode.EXECUTION_REFUSED,
                        f"Workload {name!r} already exists; use operation='replace'.",
                        remediation=(
                            "Retry with operation='replace' and the current configuration_id.",
                        ),
                        details={
                            "configuration_id": current_id,
                            "next_tool": "workload_configuration_status",
                        },
                    )
                action: Literal["created", "updated", "unchanged"] = (
                    "created"
                    if recovered_invalid
                    else ("unchanged" if existing is not None else "created")
                )
            else:
                if current_id is None or existing is None:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        f"Cannot replace workload {name!r} because it is not declared.",
                        remediation=("Retry with operation='create' for a new workload.",),
                        details={"next_tool": "configure_workload"},
                    )
                if request.expected_configuration_id != current_id:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "Workload configuration changed before replacement.",
                        remediation=(
                            "Refresh workload_configuration_status and retry with its "
                            "configuration_id.",
                        ),
                        details={
                            "configuration_id": current_id,
                            "next_tool": "workload_configuration_status",
                        },
                    )
                action = "updated"

            updated = ProjectConfig.model_validate(
                {
                    **project.model_dump(mode="python"),
                    "workloads": {**project.workloads, name: config},
                }
            )
            definition_id = digest_model(self._definition_content(name, config))
            changed_paths: list[str] = []
            if action != "unchanged":
                rendered = (
                    recovered_text
                    if recovered_invalid
                    else self._render_project_config(existing_text, name, config)
                )
                assert rendered is not None
                mode = (
                    self.project_config_path.stat().st_mode & 0o777
                    if self.project_config_path.exists()
                    else 0o644
                )
                atomic_write_text(self.project_config_path, rendered, mode=mode)
                changed_paths.append("flameox.toml")

        return WorkloadConfigurationResult(
            action=action,
            name=name,
            configuration_id=digest_model(updated),
            workload_definition_id=definition_id,
            configuration_source="agent",
            changed_paths=tuple(changed_paths),
        )

    def configure_inference_server(
        self, request: ConfigureInferenceServerRequest
    ) -> InferenceConfigurationResult:
        return self._configure_inference_entry(
            kind="server",
            name=request.name,
            operation=request.operation,
            config=request.config,
            expected_configuration_id=request.expected_configuration_id,
        )

    def configure_inference_scenario(
        self, request: ConfigureInferenceScenarioRequest
    ) -> InferenceConfigurationResult:
        return self._configure_inference_entry(
            kind="scenario",
            name=request.name,
            operation=request.operation,
            config=request.config,
            expected_configuration_id=request.expected_configuration_id,
        )

    def _configure_inference_entry(
        self,
        *,
        kind: Literal["server", "scenario"],
        name: str,
        operation: Literal["create", "replace"],
        config: InferenceServerConfig | InferenceScenarioConfig,
        expected_configuration_id: Digest | None,
    ) -> InferenceConfigurationResult:
        section: Literal["inference_servers", "inference_scenarios"] = (
            "inference_servers" if kind == "server" else "inference_scenarios"
        )
        with self.workspace.write_locked():
            text = (
                self.project_config_path.read_text(encoding="utf-8")
                if self.project_config_path.exists()
                else ""
            )
            project = self.load() if text else ProjectConfig()
            current_id = digest_model(project)
            entries = project.inference_servers if kind == "server" else project.inference_scenarios
            existing = entries.get(name)
            if operation == "create" and existing is not None and existing != config:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Inference {kind} {name!r} already exists; use operation='replace'.",
                    details={"configuration_id": current_id},
                )
            if operation == "replace":
                if existing is None:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        f"Cannot replace inference {kind} {name!r} because it is not declared.",
                    )
                if expected_configuration_id != current_id:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "Inference configuration changed before replacement.",
                        details={"configuration_id": current_id},
                    )
            action: Literal["created", "updated", "unchanged"] = (
                "unchanged"
                if existing == config
                else ("created" if existing is None else "updated")
            )
            updated_values = {**entries, name: config}
            updated = ProjectConfig.model_validate(
                {
                    **project.model_dump(mode="python"),
                    section: updated_values,
                }
            )
            changed_paths: tuple[Literal["flameox.toml"], ...] = ()
            if action != "unchanged":
                rendered = self._render_named_config(text, section, name, config)
                mode = (
                    self.project_config_path.stat().st_mode & 0o777
                    if self.project_config_path.exists()
                    else 0o644
                )
                atomic_write_text(self.project_config_path, rendered, mode=mode)
                changed_paths = ("flameox.toml",)
        return InferenceConfigurationResult(
            kind=kind,
            action=action,
            name=name,
            configuration_id=digest_model(updated),
            definition_id=digest_model(config),
            changed_paths=changed_paths,
        )

    def resolve(
        self,
        name: str,
        overrides: dict[str, Scalar] | None = None,
        *,
        dynamic_parameters: tuple[str, ...] = (),
    ) -> WorkloadInstance:
        config = self._selected(name)
        definition = self.definition(name)
        selected = self._parameters(config, overrides or {}, dynamic_parameters=dynamic_parameters)
        cwd = (self.workspace.project_root / _render(config.cwd, selected)).resolve()
        try:
            cwd.relative_to(self.workspace.project_root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Resolved workload directory leaves the project root.",
            ) from exc
        rendered_argv = tuple(_render(value, selected) for value in config.argv)
        command = CommandSpec(
            argv=(
                str(self._resolve_executable(rendered_argv[0], cwd)),
                *rendered_argv[1:],
            ),
            cwd=str(cwd),
            env_overrides={
                name: _render(value, selected) for name, value in config.environment.items()
            },
            timeout_seconds=config.timeout_seconds,
        )
        json_parameters = {name: cast(JsonValue, value) for name, value in selected.items()}
        content: dict[str, JsonValue] = {
            "workload_definition_id": definition.workload_definition_id,
            "command": command.model_dump(mode="json"),
            "parameters": json_parameters,
        }
        return WorkloadInstance(
            workload_instance_id=digest_model(content),
            workload_definition_id=definition.workload_definition_id,
            command=command,
            parameters=json_parameters,
        )

    def resolve_oracle(
        self,
        name: str,
        overrides: dict[str, Scalar] | None = None,
        *,
        dynamic_parameters: tuple[str, ...] = (),
    ) -> ResolvedOracle | None:
        config = self._selected(name)
        if config.oracle is None:
            return None
        if not config.oracle.argv:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Workload {name!r} declares an oracle without argv.",
            )
        selected = self._parameters(config, overrides or {}, dynamic_parameters=dynamic_parameters)
        cwd = (self.workspace.project_root / _render(config.cwd, selected)).resolve()
        rendered = tuple(_render(value, selected) for value in config.oracle.argv)
        return ResolvedOracle(
            strength=config.oracle.strength,
            receipt_schema=config.oracle.receipt_schema,
            command=CommandSpec(
                argv=(
                    str(self._resolve_executable(rendered[0], cwd)),
                    *rendered[1:],
                ),
                cwd=str(cwd),
                timeout_seconds=config.timeout_seconds,
            ),
        )

    def writable_targets(self, name: str) -> tuple[tuple[Path, str], ...]:
        config = self._selected(name)
        project_root = self.workspace.project_root.resolve()
        resolved: list[tuple[Path, str]] = []
        for value in config.writable_paths:
            relative = PurePath(value)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or "\x00" in value
                or relative.parts[0] in {".git", ".diagnostics", "flameox.toml"}
            ):
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Writable path {value!r} is not an allowed project build-output root.",
                )
            target = (project_root / Path(relative)).resolve(strict=True)
            try:
                target.relative_to(project_root)
            except ValueError as exc:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Writable path {value!r} escapes the project root.",
                ) from exc
            if not target.is_dir():
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Writable path {value!r} must be a pre-existing directory.",
                )
            stat = target.stat()
            identity = digest_model(
                {
                    "path": str(target),
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "mode": stat.st_mode,
                }
            )
            resolved.append((target, identity))
        if len({item[0] for item in resolved}) != len(resolved):
            raise DomainError(ErrorCode.EXECUTION_REFUSED, "Writable paths must be unique.")
        return tuple(resolved)

    def _resolve_executable(self, value: str, cwd: Path) -> Path:
        if os.sep in value or (os.altsep is not None and os.altsep in value):
            candidate = Path(value)
            candidate = candidate if candidate.is_absolute() else cwd / candidate
            resolved = candidate.parent.resolve() / candidate.name
        else:
            located = shutil.which(value)
            if located is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"Workload executable {value!r} is unavailable.",
                    details={
                        "next_tool": "get_declared_workflow",
                        "missing_executable": value,
                        "requirement_kind": "workload_executable",
                    },
                    remediation=(
                        f"Install executable {value!r} in the local environment or configure a "
                        "named workload using an available executable, then retry planning.",
                    ),
                )
            resolved = Path(located).absolute()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Workload executable is not runnable: {resolved}",
            )
        return resolved

    def _selected(self, name: str) -> WorkloadConfig:
        try:
            return self.load().workloads[name]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unknown workload {name!r}.",
            ) from exc

    def _parameters(
        self,
        config: WorkloadConfig,
        overrides: dict[str, Scalar],
        *,
        dynamic_parameters: tuple[str, ...] = (),
    ) -> dict[str, Scalar]:
        if len(set(dynamic_parameters)) != len(dynamic_parameters):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Dynamic workload parameters must be unique.",
            )
        unknown_dynamic = set(dynamic_parameters) - set(config.parameters)
        if unknown_dynamic:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Dynamic workload parameters must be declared by the workload.",
                details={"parameters": sorted(unknown_dynamic)},
            )
        unexpected = set(overrides) - set(config.parameters)
        if unexpected:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Unknown workload parameters.",
                details={"parameters": sorted(unexpected)},
            )
        selected: dict[str, Scalar] = {}
        for name, choices in config.parameters.items():
            if not choices:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Workload parameter {name!r} has no allowed values.",
                )
            if name in dynamic_parameters and name not in overrides:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    f"Dynamic workload parameter {name!r} must be supplied.",
                )
            value = overrides.get(name, choices[0])
            if name not in dynamic_parameters and value not in choices:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    f"Value for {name!r} is outside the declared choices.",
                    details={"allowed": list(choices), "received": value},
                )
            selected[name] = value
        return selected

    def _definition_content(
        self,
        name: str,
        config: WorkloadConfig,
    ) -> dict[str, JsonValue]:
        return {
            "name": name,
            "definition": config.model_dump(mode="json"),
        }

    def _render_project_config(self, text: str, name: str, config: WorkloadConfig) -> str:
        document = tomlkit.parse(text) if text else tomlkit.document()
        if "schema_version" not in document:
            document["schema_version"] = 1
        workloads = document.get("workloads")
        if workloads is None:
            workloads = tomlkit.table()
            document["workloads"] = workloads
        values = config.model_dump(mode="python", exclude_none=True, exclude_defaults=True)
        existing = workloads.get(name)
        if isinstance(existing, Table):
            for key in list(existing):
                if key not in values:
                    del existing[key]
            for key, value in values.items():
                existing[key] = tomlkit.item(value)
        else:
            workloads[name] = tomlkit.item(values)
        return tomlkit.dumps(document)

    @staticmethod
    def _render_named_config(
        text: str,
        section: Literal["inference_servers", "inference_scenarios"],
        name: str,
        config: InferenceServerConfig | InferenceScenarioConfig,
    ) -> str:
        document = tomlkit.parse(text) if text else tomlkit.document()
        if "schema_version" not in document:
            document["schema_version"] = 1
        group = document.get(section)
        if group is None:
            group = tomlkit.table()
            document[section] = group
        values = config.model_dump(mode="python", exclude_none=True, exclude_defaults=True)
        existing = group.get(name)
        if isinstance(existing, Table):
            for key in list(existing):
                if key not in values:
                    del existing[key]
            for key, value in values.items():
                existing[key] = tomlkit.item(value)
        else:
            group[name] = tomlkit.item(values)
        return tomlkit.dumps(document)
