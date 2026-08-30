from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path, PurePath
from string import Formatter
from typing import Annotated, Literal, cast
from urllib.parse import urlsplit

import tomlkit
from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    Tag,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)
from tomlkit.exceptions import ParseError
from tomlkit.items import Table

from flameox.action_graph import (
    ActionId,
    ManualAction,
    NextAction,
    ToolAction,
    manual_action,
    tool_action,
)
from flameox.adapters.builtins import BUILTIN_ADAPTERS
from flameox.adapters.registry import AdapterRegistry
from flameox.application.capabilities import CapabilityService
from flameox.application.inference_providers import (
    InferenceEndpointType,
    InferenceScenarioProvider,
    InferenceServerMode,
    InferenceServerProvider,
)
from flameox.application.runtime_resources import RUNTIME_RESOURCE_METRICS
from flameox.atomic import atomic_write_text
from flameox.command_binding import ExecutableResolver
from flameox.domain import (
    CapabilityPermissionStatus,
    CapabilityReport,
    CapabilityStatus,
    CommandSpec,
    CursorNamespace,
    DomainError,
    ErrorCode,
    ExperimentOutcomeGoal,
    MeasurementSeriesSelector,
    MetricPolarity,
    MetricValueDomain,
    MetricZeroPolicy,
    OracleStrength,
    ProbeKind,
    RequirementKind,
    WorkloadDefinition,
    WorkloadExecutionProtocol,
    WorkloadInstance,
    digest_model,
)
from flameox.domain.executables import (
    ExecutableResolutionRequest,
    ExecutableTrustPolicy,
    ResolvedExecutable,
)
from flameox.domain.models import Digest
from flameox.http_transport import validate_loopback_base_url
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import Workspace

Scalar = str | int | float | bool
_DECLARATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def scalar_identity(value: Scalar) -> tuple[str, object]:
    """Return a typed identity key distinguishing bool/int/float/str by exact JSON type.

    Python's numeric tower treats ``True == 1`` and ``1 == 1.0`` as equal, and
    ``hash(True) == hash(1)``. Configuration and evidence protocols must not
    treat those as the same scalar. This helper returns a ``(type_tag, value)``
    tuple where the type tag distinguishes the four scalar JSON kinds so that
    ``scalar_identity(True) != scalar_identity(1)`` and
    ``scalar_identity(1) != scalar_identity(1.0)``.
    """
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float", value)
    return ("string", value)


def scalar_equal(left: Scalar, right: Scalar) -> bool:
    """Return True only when both scalars share the exact JSON type and value."""
    return scalar_identity(left) == scalar_identity(right)


def scalar_contains(value: Scalar, choices: tuple[Scalar, ...] | list[Scalar]) -> bool:
    """Return True only when ``value`` is present in ``choices`` by exact scalar identity."""
    identity = scalar_identity(value)
    return any(scalar_identity(choice) == identity for choice in choices)


def scalar_identity_set(
    values: tuple[Scalar, ...] | list[Scalar],
) -> set[tuple[str, object]]:
    """Return the set of scalar identities for ``values`` without Python-equality collisions."""
    return {scalar_identity(value) for value in values}


def scalar_subset(subset_values: list[Scalar], superset_values: list[Scalar]) -> bool:
    """Return True only when every identity in ``subset_values`` is in ``superset_values``."""
    superset = scalar_identity_set(superset_values)
    return all(identity in superset for identity in (scalar_identity(v) for v in subset_values))


class ConfigurationOperation(StrEnum):
    CREATE = "create"
    REPLACE = "replace"


class ConfigurationAction(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class DeclaredWorkflowKind(StrEnum):
    WORKLOAD = "workload"
    EXPERIMENT = "experiment"
    FAULT_EXPERIMENT = "fault_experiment"


class InferenceConfigurationKind(StrEnum):
    SERVER = "server"
    SCENARIO = "scenario"


class AdapterPlanningDisposition(StrEnum):
    READY = "ready"
    ACTIVE_PROBE_REQUIRED = "active_probe_required"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    DEGRADED = "degraded"


class WorkloadOracleConfig(ContractModel):
    strength: OracleStrength = OracleStrength.EXECUTION_CHECK
    argv: Annotated[tuple[str, ...], Field(min_length=1, max_length=1_024)]
    receipt_schema: Literal["flameox.oracle-receipt.v1"] | None = None

    @model_validator(mode="after")
    def cross_treatment_requires_receipt(self) -> WorkloadOracleConfig:
        if (
            self.strength is OracleStrength.CROSS_TREATMENT_EQUIVALENCE
            and self.receipt_schema is None
        ):
            raise ValueError("cross-treatment equivalence requires a typed oracle receipt")
        return self


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
    "metal.devices",
    "metal.support",
    "metal.unified_memory",
    "macos.build",
]


class WorkloadEnvironmentIdentityConfig(ContractModel):
    required: Annotated[tuple[AcceleratorIdentityRequirement, ...], Field(max_length=4)] = ()

    @model_validator(mode="after")
    def requirements_are_unique(self) -> WorkloadEnvironmentIdentityConfig:
        if len(set(self.required)) != len(self.required):
            raise ValueError("environment identity requirements must be unique")
        providers = {
            "metal" if item.startswith("metal.") or item == "macos.build" else "cuda"
            for item in self.required
        }
        if len(providers) > 1:
            raise ValueError("one workload cannot mix CUDA and Metal identity requirements")
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
    executable_policy: ExecutableTrustPolicy = ExecutableTrustPolicy.TRUSTED_HOST_TOOL
    oracle: WorkloadOracleConfig | None = None
    requirements: WorkloadRequirementsConfig = Field(default_factory=WorkloadRequirementsConfig)
    writable_paths: Annotated[tuple[str, ...], Field(max_length=16)] = ()
    identity: WorkloadIdentityConfig = Field(default_factory=WorkloadIdentityConfig)
    execution_protocol: WorkloadExecutionProtocol | None = None

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
    polarity: MetricPolarity = MetricPolarity.NEUTRAL
    estimand: str = "median_paired_log_ratio"
    practical_threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.95
    random_seed: Annotated[int, Field(ge=0)] = 0

    @field_validator("primary_metric")
    @classmethod
    def runtime_resource_metric_is_known(cls, value: str) -> str:
        if value.startswith("runtime_resource.") and value not in RUNTIME_RESOURCE_METRICS:
            raise ValueError(
                "runtime-resource primary_metric must be one of "
                + ", ".join(sorted(RUNTIME_RESOURCE_METRICS))
            )
        return value


class _FactorExperimentConfig(_CommonExperimentConfig):
    factors: dict[str, Annotated[tuple[Scalar, ...], Field(min_length=1, max_length=32)]] = Field(
        min_length=1,
        max_length=8,
    )
    exclude: Annotated[tuple[dict[str, Scalar], ...], Field(max_length=1_000)] = ()
    treatment_factor: str
    baseline_value: Scalar | None = None

    @model_validator(mode="after")
    def treatment_is_declared(self) -> _FactorExperimentConfig:
        if self.treatment_factor not in self.factors:
            raise ValueError("factor experiments require a declared treatment_factor")
        if self.baseline_value is not None:
            allowed = self.factors[self.treatment_factor]
            if not scalar_contains(self.baseline_value, allowed):
                raise ValueError(
                    "baseline_value must be one of the declared treatment_factor values"
                )
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


class _PerformanceExperimentConfig(_CommonExperimentConfig):
    analysis: Literal["performance"] = "performance"
    primary_metric_unit: str | None = None
    measurement_series: MeasurementSeriesSelector | None = None
    value_domain: Literal[MetricValueDomain.STRICTLY_POSITIVE] = MetricValueDomain.STRICTLY_POSITIVE
    zero_policy: Literal[MetricZeroPolicy.REJECT] = MetricZeroPolicy.REJECT
    outcome_goal: Literal[None] = None
    minimum_attempts: Literal[None] = None
    maximum_attempts: Literal[None] = None


class _OutcomeExperimentConfig(_CommonExperimentConfig):
    analysis: Literal["outcome"]
    outcome_goal: Literal[ExperimentOutcomeGoal.ABSENCE_OF_FAILURE]
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


def _experiment_config_kind(value: object) -> str:
    if isinstance(value, Mapping):
        analysis = value.get("analysis", "performance")
        shape = f"factor_{value.get('combination_policy', 'cartesian')}"
        return f"{shape}_{analysis}"

    analysis = "outcome" if isinstance(value, _OutcomeExperimentConfig) else "performance"
    if isinstance(value, _ExplicitFactorExperimentConfig):
        shape = "factor_explicit"
    else:
        shape = "factor_cartesian"
    return f"{shape}_{analysis}"


type ExperimentConfig = Annotated[
    Annotated[_CartesianFactorPerformanceExperimentConfig, Tag("factor_cartesian_performance")]
    | Annotated[_CartesianFactorOutcomeExperimentConfig, Tag("factor_cartesian_outcome")]
    | Annotated[_ExplicitFactorPerformanceExperimentConfig, Tag("factor_explicit_performance")]
    | Annotated[_ExplicitFactorOutcomeExperimentConfig, Tag("factor_explicit_outcome")],
    Discriminator(_experiment_config_kind),
]

_EXPERIMENT_CONFIG_ADAPTER: TypeAdapter[ExperimentConfig] = TypeAdapter(ExperimentConfig)


def parse_experiment_config(value: object) -> ExperimentConfig:
    """Parse a flat factor experiment into its single legal configuration case."""

    return _EXPERIMENT_CONFIG_ADAPTER.validate_python(value)


class _ToxicFault(ContractModel):
    stream: Literal["upstream", "downstream"] = "downstream"
    toxicity: float = Field(default=1.0, ge=0, le=1)


class LatencyFault(_ToxicFault):
    type: Literal["latency"]
    latency_ms: int = Field(gt=0, le=3_600_000)
    jitter_ms: int = Field(default=0, ge=0, le=3_600_000)


class TimeoutFault(_ToxicFault):
    type: Literal["timeout"]
    timeout_ms: int = Field(gt=0, le=3_600_000)


class ResetPeerFault(_ToxicFault):
    type: Literal["reset_peer"]


class BandwidthFault(_ToxicFault):
    type: Literal["bandwidth"]
    bandwidth_limit: int = Field(gt=0, le=10_000_000_000)


class SlicerFault(_ToxicFault):
    type: Literal["slicer"]
    average_size: int = Field(gt=0, le=10_000_000)
    size_variation: int = Field(default=0, ge=0, le=10_000_000)
    delay_ms: int = Field(default=0, ge=0, le=3_600_000)


class LimitDataFault(_ToxicFault):
    type: Literal["limit_data"]
    bytes: int = Field(gt=0, le=10_000_000_000)


class SlowCloseFault(_ToxicFault):
    type: Literal["slow_close"]
    delay_ms: int = Field(gt=0, le=3_600_000)


class ProxyFault(ContractModel):
    type: Literal["proxy"]
    enabled: Literal[False]


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


class FaultJsonMeasurement(ContractModel):
    """Compare the workload's strict elapsed-time JSON receipt on stdout."""

    source: Literal["stdout_json"]
    practical_threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.95


class FaultExperimentConfig(ContractModel):
    workload: str
    endpoint_parameter: str
    upstream_host: str
    upstream_port: int = Field(gt=0, le=65_535)
    endpoint_template: str
    scenarios: dict[str, FaultScenario] = Field(min_length=1, max_length=64)
    blocks: int = Field(default=1, gt=0, le=1_000)
    repetitions: int = Field(default=1, gt=0, le=1_000)
    measurement: FaultJsonMeasurement | None = None
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
            if isinstance(scenario, ProxyFault):
                continue
            if (
                isinstance(scenario, LatencyFault)
                and scenario.toxicity == 1
                and scenario.jitter_ms != 0
            ):
                raise ValueError("deterministic latency faults require zero jitter")
            if scenario.toxicity < 1 and self.repetitions < 2:
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
        validate_loopback_base_url(value)
        return value


class _VllmInferenceServerConfig(_CommonInferenceServerConfig):
    provider: Literal[InferenceServerProvider.VLLM] = InferenceServerProvider.VLLM
    benchmark_python: Literal[None] = None


class _SglangInferenceServerConfig(_CommonInferenceServerConfig):
    provider: Literal[InferenceServerProvider.SGLANG]
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
            raise ValueError("sglang inference servers require a root base_url")
        return value


class _ManagedInferenceServerConfig(_CommonInferenceServerConfig):
    mode: Literal[InferenceServerMode.MANAGED]
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
    mode: Literal[InferenceServerMode.EXISTING_LOCAL]
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
    """Fields shared by every maintained inference benchmark provider.

    Trace input, repetition, and timing-scale parameters are bounded and
    forward-compatible with the maintained AIPerf and vLLM bench providers.
    """

    server: str
    endpoint_type: InferenceEndpointType = InferenceEndpointType.CHAT
    num_prompts: Annotated[int, Field(gt=0, le=10_000_000)] = 1
    concurrency: Annotated[int, Field(gt=0, le=100_000)] | None = None
    request_rate: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    warmup_request_count: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0
    semantic_oracle_workload: str | None = None


class _AIPerfInferenceScenarioConfig(_CommonInferenceScenarioConfig):
    provider: Literal[InferenceScenarioProvider.AIPERF]
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
    provider: Literal[InferenceScenarioProvider.VLLM_BENCH]
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
    provider: Literal[InferenceScenarioProvider.SGLANG_BENCH]
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

    @field_validator("inference_servers", "inference_scenarios")
    @classmethod
    def inference_names_are_safe_identifiers(cls, value: dict[str, object]) -> dict[str, object]:
        invalid = sorted(name for name in value if _DECLARATION_NAME.fullmatch(name) is None)
        if invalid:
            raise ValueError(
                "inference declaration names must match "
                "[A-Za-z0-9][A-Za-z0-9._-]{0,99}: " + ", ".join(invalid)
            )
        return value

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
            if (
                scenario.provider is InferenceScenarioProvider.SGLANG_BENCH
                and server.provider is not InferenceServerProvider.SGLANG
            ):
                raise ValueError("sglang_bench scenarios require an sglang inference server")
            if (
                scenario.provider is not InferenceScenarioProvider.SGLANG_BENCH
                and server.provider is InferenceServerProvider.SGLANG
            ):
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
    operation: ConfigurationOperation
    config: WorkloadConfig
    expected_configuration_id: str | None = None


class _WorkloadConfigurationStatus(ContractModel):
    config_path: Literal["flameox.toml"] = "flameox.toml"


def _configure_workload_action() -> ManualAction:
    return manual_action(
        "Supply a complete named workload definition before continuing.",
        suggested_action=ActionId.CONFIGURE_WORKLOAD,
        missing_arguments=("name", "operation", "argv"),
    )


def _list_workloads_action() -> ToolAction:
    return tool_action(ActionId.LIST_DECLARED_WORKFLOWS)


def _configure_request_action(
    request: ConfigureWorkloadRequest,
    *,
    operation: ConfigurationOperation,
    expected_configuration_id: str | None,
) -> ToolAction:
    config = request.config.model_dump(mode="json")
    return tool_action(
        ActionId.CONFIGURE_WORKLOAD,
        name=request.name,
        operation=operation.value,
        argv=config["argv"],
        cwd=config["cwd"],
        timeout_seconds=config["timeout_seconds"],
        parameters=config["parameters"],
        environment=config["environment"],
        oracle=config["oracle"],
        requirements=config["requirements"],
        writable_paths=config["writable_paths"],
        identity=config["identity"],
        execution_protocol=config["execution_protocol"],
        expected_configuration_id=expected_configuration_id,
    )


class _UnavailableWorkloadConfigurationStatus(_WorkloadConfigurationStatus):
    configuration_id: Literal[None] = None
    workload_names: tuple[()] = ()
    diagnostics: Annotated[
        tuple[Annotated[str, Field(max_length=512)], ...],
        Field(min_length=1, max_length=8),
    ]
    next_action: ManualAction = Field(default_factory=_configure_workload_action)


class MissingWorkloadConfigurationStatus(_UnavailableWorkloadConfigurationStatus):
    status: Literal["missing"] = "missing"


class InvalidWorkloadConfigurationStatus(_UnavailableWorkloadConfigurationStatus):
    status: Literal["invalid"] = "invalid"


class ValidWorkloadConfigurationStatus(_WorkloadConfigurationStatus):
    status: Literal["valid"] = "valid"
    configuration_id: Digest
    workload_names: Annotated[
        tuple[Annotated[str, Field(max_length=100)], ...],
        Field(max_length=1_000),
    ] = ()
    diagnostics: Annotated[
        tuple[Annotated[str, Field(max_length=512)], ...],
        Field(max_length=8),
    ] = ()
    next_action: NextAction

    @model_validator(mode="after")
    def recovery_matches_declared_workloads(self) -> ValidWorkloadConfigurationStatus:
        expected_diagnostics = (
            () if self.workload_names else ("No named workloads are declared yet.",)
        )
        expected_next_action: NextAction = (
            _list_workloads_action() if self.workload_names else _configure_workload_action()
        )
        if self.diagnostics != expected_diagnostics or self.next_action != expected_next_action:
            raise ValueError("valid configuration recovery must match declared workloads")
        return self


type WorkloadConfigurationStatus = Annotated[
    MissingWorkloadConfigurationStatus
    | InvalidWorkloadConfigurationStatus
    | ValidWorkloadConfigurationStatus,
    Field(discriminator="status"),
]


class WorkloadConfigurationResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    action: ConfigurationAction
    name: str
    configuration_id: str
    workload_definition_id: str
    next_action: ToolAction = Field(default_factory=_list_workloads_action)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def changed_paths(self) -> tuple[Literal["flameox.toml"], ...]:
        return () if self.action is ConfigurationAction.UNCHANGED else ("flameox.toml",)


class ConfigureInferenceServerRequest(ContractModel):
    name: Annotated[
        str,
        Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    operation: ConfigurationOperation
    config: InferenceServerConfig
    expected_configuration_id: Digest | None = None


class ConfigureInferenceScenarioRequest(ContractModel):
    name: Annotated[
        str,
        Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    operation: ConfigurationOperation
    config: InferenceScenarioConfig
    expected_configuration_id: Digest | None = None


class InferenceConfigurationResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    kind: InferenceConfigurationKind
    action: ConfigurationAction
    name: str
    configuration_id: Digest
    definition_id: Digest

    @computed_field  # type: ignore[prop-decorator]
    @property
    def changed_paths(self) -> tuple[Literal["flameox.toml"], ...]:
        return () if self.action is ConfigurationAction.UNCHANGED else ("flameox.toml",)


class InferenceConfigurationList(ContractModel):
    configuration_id: Digest
    servers: dict[str, InferenceServerConfig]
    scenarios: dict[str, InferenceScenarioConfig]


class ResolvedOracle(ContractModel):
    strength: OracleStrength
    command: CommandSpec
    executable_binding: ResolvedExecutable
    receipt_schema: Literal["flameox.oracle-receipt.v1"] | None = None


class DeclaredWorkflowSummary(ContractModel):
    kind: DeclaredWorkflowKind
    name: str
    definition_id: str
    parameter_names: tuple[str, ...] = ()
    oracle_strength: OracleStrength | None = None
    timeout_seconds: float | None = None


class DeclaredWorkflowList(CursorPageContract):
    page_items_field = "workflows"

    configuration_id: str
    workflows: tuple[DeclaredWorkflowSummary, ...]


def _require_complete_adapter_option_count(options: tuple[AdapterOption, ...], total: int) -> None:
    if total < len(options):
        raise ValueError("adapter option total cannot be smaller than the returned options")


class DeclaredWorkflowDetail(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    configuration_id: str
    summary: DeclaredWorkflowSummary
    allowed_parameters: dict[str, tuple[Scalar, ...]]
    workload_name: str | None = None
    factors: dict[str, tuple[Scalar, ...]] = Field(default_factory=dict)
    design: str | None = None
    blocks: int | None = None
    primary_metric: str | None = None
    polarity: MetricPolarity | None = None
    estimand: str | None = None
    validation_spec_id: str | None = None
    execution_protocol: WorkloadExecutionProtocol | None = None
    requirements: tuple[DeclaredWorkflowRequirement, ...] = ()
    adapter_options: tuple[AdapterOption, ...] = ()
    adapter_option_total: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def adapter_option_count_is_coherent(self) -> DeclaredWorkflowDetail:
        _require_complete_adapter_option_count(self.adapter_options, self.adapter_option_total)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def adapter_options_truncated(self) -> bool:
        return self.adapter_option_total > len(self.adapter_options)


class DeclaredWorkflowRequirement(ContractModel):
    name: str
    kind: RequirementKind
    probe_kind: ProbeKind
    required: bool


class AdapterOption(ContractModel):
    adapter: str
    status: CapabilityStatus
    planning_disposition: AdapterPlanningDisposition
    required_preflight_mode: ProbeKind
    permission_status: CapabilityPermissionStatus | None = None
    supported_modes: tuple[str, ...] = ()
    supported_formats: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()


class WorkloadInspection(WorkloadDefinition):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    configuration_id: str
    requirements: tuple[DeclaredWorkflowRequirement, ...] = ()
    adapter_options: tuple[AdapterOption, ...] = ()
    adapter_option_total: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def adapter_option_count_is_coherent(self) -> WorkloadInspection:
        _require_complete_adapter_option_count(self.adapter_options, self.adapter_option_total)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def adapter_options_truncated(self) -> bool:
        return self.adapter_option_total > len(self.adapter_options)


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
                details={"config_path": "flameox.toml"},
                next_action=_configure_workload_action(),
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
                },
                next_action=_configure_workload_action(),
            ) from exc

    def configuration_status(self) -> WorkloadConfigurationStatus:
        if not self.project_config_path.exists():
            return MissingWorkloadConfigurationStatus(
                diagnostics=("No named workload configuration exists yet.",),
            )
        try:
            project = self.load()
        except DomainError as error:
            return InvalidWorkloadConfigurationStatus(
                diagnostics=(error.message[:512],),
            )
        has_workloads = bool(project.workloads)
        return ValidWorkloadConfigurationStatus(
            configuration_id=digest_model(project),
            workload_names=tuple(sorted(project.workloads)),
            diagnostics=(("No named workloads are declared yet.",) if not has_workloads else ()),
            next_action=(
                _list_workloads_action() if has_workloads else _configure_workload_action()
            ),
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
        kind: DeclaredWorkflowKind,
        limit: int,
        cursor: str | None = None,
    ) -> DeclaredWorkflowList:
        project = self.load()
        configuration_id = digest_model(project)
        query_id = digest_model({"kind": kind})
        offset = 0
        if cursor is not None:
            position = cast(
                tuple[int],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.DECLARED_WORKFLOWS,
                    snapshot_id=configuration_id,
                    scope_digest=query_id,
                ),
            )
            offset = position[0]

        names = sorted(
            project.workloads
            if kind is DeclaredWorkflowKind.WORKLOAD
            else (
                project.experiments
                if kind is DeclaredWorkflowKind.EXPERIMENT
                else project.fault_experiments
            )
        )
        summaries = tuple(self._workflow_summary(project, kind, name) for name in names)
        selected = summaries[offset : offset + limit]
        next_offset = offset + len(selected)
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.DECLARED_WORKFLOWS,
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
            next_cursor=next_cursor,
        )

    def get_declared(
        self,
        *,
        kind: DeclaredWorkflowKind,
        name: str,
    ) -> DeclaredWorkflowDetail:
        project = self.load()
        configuration_id = digest_model(project)
        summary = self._workflow_summary(project, kind, name)
        workload_name = (
            name
            if kind is DeclaredWorkflowKind.WORKLOAD
            else (
                project.experiments[name].workload
                if kind is DeclaredWorkflowKind.EXPERIMENT
                else project.fault_experiments[name].workload
            )
        )
        workload_config = project.workloads[workload_name]
        workload_definition = self.definition(workload_name)
        requirements, adapter_options, option_total = self._inspection_fields(workload_config)
        if kind is DeclaredWorkflowKind.WORKLOAD:
            config = project.workloads[name]
            return DeclaredWorkflowDetail(
                configuration_id=configuration_id,
                summary=summary,
                allowed_parameters=config.parameters,
                validation_spec_id=workload_definition.validation_spec_id,
                execution_protocol=workload_definition.execution_protocol,
                requirements=requirements,
                adapter_options=adapter_options,
                adapter_option_total=option_total,
            )
        if kind is DeclaredWorkflowKind.FAULT_EXPERIMENT:
            fault = project.fault_experiments[name]
            measurement = fault.measurement
            return DeclaredWorkflowDetail(
                configuration_id=configuration_id,
                summary=summary,
                allowed_parameters=workload_config.parameters,
                workload_name=fault.workload,
                blocks=fault.blocks,
                primary_metric=(
                    "fault.client_elapsed" if measurement is not None else "categorical_outcome"
                ),
                polarity=(
                    MetricPolarity.LOWER_IS_BETTER
                    if measurement is not None
                    else MetricPolarity.NEUTRAL
                ),
                estimand=(
                    "median_paired_log_ratio" if measurement is not None else "categorical_outcome"
                ),
                validation_spec_id=workload_definition.validation_spec_id,
                execution_protocol=workload_definition.execution_protocol,
                requirements=requirements,
                adapter_options=adapter_options,
                adapter_option_total=option_total,
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
            validation_spec_id=workload_definition.validation_spec_id,
            execution_protocol=workload_definition.execution_protocol,
            requirements=requirements,
            adapter_options=adapter_options,
            adapter_option_total=option_total,
        )

    def inspect(self, name: str) -> WorkloadInspection:
        project = self.load()
        definition = self.definition(name)
        config = project.workloads[name]
        requirements, options, total = self._inspection_fields(config)
        return WorkloadInspection(
            **definition.model_dump(mode="python"),
            configuration_id=digest_model(project),
            requirements=requirements,
            adapter_options=options,
            adapter_option_total=total,
        )

    def _inspection_fields(
        self,
        config: WorkloadConfig,
    ) -> tuple[
        tuple[DeclaredWorkflowRequirement, ...],
        tuple[AdapterOption, ...],
        int,
    ]:
        requirement_groups: tuple[
            tuple[RequirementKind, tuple[str, ...]],
            ...,
        ] = (
            (RequirementKind.EXECUTABLE, config.requirements.executables),
            (RequirementKind.PYTHON_DISTRIBUTION, config.requirements.python_distributions),
            (RequirementKind.CAPABILITY, config.requirements.capabilities),
        )
        requirements = tuple(
            DeclaredWorkflowRequirement(
                name=name,
                kind=kind,
                required=name not in config.requirements.optional,
                probe_kind=(
                    ProbeKind.ACTIVE if name in config.requirements.active else ProbeKind.PASSIVE
                ),
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
        return requirements, options, len(reports)

    @staticmethod
    def _adapter_option(capability: CapabilityReport) -> AdapterOption:
        permission_sensitive = capability.permission_status in {
            CapabilityPermissionStatus.UNKNOWN_UNTIL_ACTIVE_PROBE,
            CapabilityPermissionStatus.NOT_EXERCISED,
        }
        required_mode = ProbeKind.ACTIVE if permission_sensitive else ProbeKind.PASSIVE
        if permission_sensitive and capability.status is CapabilityStatus.AVAILABLE:
            disposition = AdapterPlanningDisposition.ACTIVE_PROBE_REQUIRED
        elif capability.status is CapabilityStatus.UNAVAILABLE:
            disposition = AdapterPlanningDisposition.UNAVAILABLE
        elif capability.status is CapabilityStatus.UNSUPPORTED_PLATFORM:
            disposition = AdapterPlanningDisposition.UNSUPPORTED
        elif capability.status is CapabilityStatus.DEGRADED:
            disposition = AdapterPlanningDisposition.DEGRADED
        else:
            disposition = AdapterPlanningDisposition.READY
        return AdapterOption(
            adapter=capability.adapter,
            status=capability.status,
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
        kind: DeclaredWorkflowKind,
        name: str,
    ) -> DeclaredWorkflowSummary:
        try:
            if kind is DeclaredWorkflowKind.WORKLOAD:
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
            if kind is DeclaredWorkflowKind.FAULT_EXPERIMENT:
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
                details={"kind": kind},
                next_action=tool_action(
                    ActionId.LIST_DECLARED_WORKFLOWS,
                    kind=kind.value,
                ),
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
            execution_protocol=config.execution_protocol,
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
                    if request.operation is not ConfigurationOperation.CREATE:
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
                                "diagnostic": str(candidate_error)[:500],
                            },
                            next_action=manual_action(
                                "Repair flameox.toml manually, then verify its status.",
                                suggested_action=ActionId.INSPECT_WORKLOAD_CONFIGURATION,
                            ),
                        ) from error
                    recovered_invalid = True
            else:
                project = ProjectConfig()

            existing = project.workloads.get(name)
            if request.operation is ConfigurationOperation.CREATE:
                if existing is not None and existing != config:
                    raise DomainError(
                        ErrorCode.EXECUTION_REFUSED,
                        f"Workload {name!r} already exists; use operation='replace'.",
                        remediation=(
                            "Retry with operation='replace' and the current configuration_id.",
                        ),
                        details={
                            "configuration_id": current_id,
                        },
                        next_action=tool_action(ActionId.INSPECT_WORKLOAD_CONFIGURATION),
                    )
                action = (
                    ConfigurationAction.CREATED
                    if recovered_invalid
                    else (
                        ConfigurationAction.UNCHANGED
                        if existing is not None
                        else ConfigurationAction.CREATED
                    )
                )
            else:
                if current_id is None or existing is None:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        f"Cannot replace workload {name!r} because it is not declared.",
                        remediation=("Retry with operation='create' for a new workload.",),
                        next_action=_configure_request_action(
                            request,
                            operation=ConfigurationOperation.CREATE,
                            expected_configuration_id=None,
                        ),
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
                        },
                        next_action=tool_action(ActionId.INSPECT_WORKLOAD_CONFIGURATION),
                    )
                action = ConfigurationAction.UPDATED

            updated = ProjectConfig.model_validate(
                {
                    **project.model_dump(mode="python"),
                    "workloads": {**project.workloads, name: config},
                }
            )
            definition_id = digest_model(self._definition_content(name, config))
            if action is not ConfigurationAction.UNCHANGED:
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

        return WorkloadConfigurationResult(
            action=action,
            name=name,
            configuration_id=digest_model(updated),
            workload_definition_id=definition_id,
        )

    def configure_inference_server(
        self, request: ConfigureInferenceServerRequest
    ) -> InferenceConfigurationResult:
        return self._configure_inference_entry(
            kind=InferenceConfigurationKind.SERVER,
            name=request.name,
            operation=request.operation,
            config=request.config,
            expected_configuration_id=request.expected_configuration_id,
        )

    def configure_inference_scenario(
        self, request: ConfigureInferenceScenarioRequest
    ) -> InferenceConfigurationResult:
        return self._configure_inference_entry(
            kind=InferenceConfigurationKind.SCENARIO,
            name=request.name,
            operation=request.operation,
            config=request.config,
            expected_configuration_id=request.expected_configuration_id,
        )

    def _configure_inference_entry(
        self,
        *,
        kind: InferenceConfigurationKind,
        name: str,
        operation: ConfigurationOperation,
        config: InferenceServerConfig | InferenceScenarioConfig,
        expected_configuration_id: Digest | None,
    ) -> InferenceConfigurationResult:
        section: Literal["inference_servers", "inference_scenarios"] = (
            "inference_servers"
            if kind is InferenceConfigurationKind.SERVER
            else "inference_scenarios"
        )
        with self.workspace.write_locked():
            text = (
                self.project_config_path.read_text(encoding="utf-8")
                if self.project_config_path.exists()
                else ""
            )
            project = self.load() if text else ProjectConfig()
            current_id = digest_model(project)
            entries = (
                project.inference_servers
                if kind is InferenceConfigurationKind.SERVER
                else project.inference_scenarios
            )
            existing = entries.get(name)
            if (
                operation is ConfigurationOperation.CREATE
                and existing is not None
                and existing != config
            ):
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Inference {kind} {name!r} already exists; use operation='replace'.",
                    details={"configuration_id": current_id},
                )
            if operation is ConfigurationOperation.REPLACE:
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
            action = (
                ConfigurationAction.UNCHANGED
                if existing == config
                else (
                    ConfigurationAction.CREATED if existing is None else ConfigurationAction.UPDATED
                )
            )
            updated_values = {**entries, name: config}
            updated = ProjectConfig.model_validate(
                {
                    **project.model_dump(mode="python"),
                    section: updated_values,
                }
            )
            if action is not ConfigurationAction.UNCHANGED:
                rendered = self._render_named_config(text, section, name, config)
                mode = (
                    self.project_config_path.stat().st_mode & 0o777
                    if self.project_config_path.exists()
                    else 0o644
                )
                atomic_write_text(self.project_config_path, rendered, mode=mode)
        return InferenceConfigurationResult(
            kind=kind,
            action=action,
            name=name,
            configuration_id=digest_model(updated),
            definition_id=digest_model(config),
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
        environment = {name: _render(value, selected) for name, value in config.environment.items()}
        binding = self._bind_executable(
            rendered_argv[0],
            cwd,
            environment,
            config.executable_policy,
            workload_name=name,
        )
        command = CommandSpec(
            argv=(
                str(binding.invocation_path),
                *rendered_argv[1:],
            ),
            cwd=str(cwd),
            env_overrides=environment,
            timeout_seconds=config.timeout_seconds,
        )
        json_parameters = {name: cast(JsonValue, value) for name, value in selected.items()}
        content: dict[str, JsonValue] = {
            "workload_definition_id": definition.workload_definition_id,
            "command": command.model_dump(mode="json"),
            "executable_binding": binding.model_dump(mode="json"),
            "parameters": json_parameters,
        }
        return WorkloadInstance(
            workload_instance_id=digest_model(content),
            workload_definition_id=definition.workload_definition_id,
            command=command,
            executable_binding=binding,
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
        selected = self._parameters(config, overrides or {}, dynamic_parameters=dynamic_parameters)
        cwd = (self.workspace.project_root / _render(config.cwd, selected)).resolve()
        rendered = tuple(_render(value, selected) for value in config.oracle.argv)
        environment = {name: _render(value, selected) for name, value in config.environment.items()}
        binding = self._bind_executable(
            rendered[0],
            cwd,
            environment,
            config.executable_policy,
            workload_name=name,
        )
        return ResolvedOracle(
            strength=config.oracle.strength,
            receipt_schema=config.oracle.receipt_schema,
            command=CommandSpec(
                argv=(
                    str(binding.invocation_path),
                    *rendered[1:],
                ),
                cwd=str(cwd),
                timeout_seconds=config.timeout_seconds,
            ),
            executable_binding=binding,
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

    def _bind_executable(
        self,
        value: str,
        cwd: Path,
        environment: dict[str, str],
        policy: ExecutableTrustPolicy,
        *,
        workload_name: str,
    ) -> ResolvedExecutable:
        effective_environment = dict(environment)
        if "PATH" not in effective_environment and "PATH" in os.environ:
            effective_environment["PATH"] = os.environ["PATH"]
        try:
            return ExecutableResolver().resolve(
                ExecutableResolutionRequest(
                    token=value,
                    cwd=cwd,
                    environment=effective_environment,
                    policy=policy,
                    allowed_roots=(self.workspace.project_root,),
                )
            )
        except DomainError as error:
            if error.code is not ErrorCode.CAPABILITY_UNAVAILABLE:
                raise
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Workload executable {value!r} is unavailable.",
                details={
                    "missing_executable": value,
                    "requirement_kind": "workload_executable",
                },
                remediation=(
                    f"Install executable {value!r} in the workload environment or configure a "
                    "named workload using an available executable, then retry planning.",
                ),
                next_action=tool_action(
                    ActionId.GET_DECLARED_WORKFLOW,
                    kind="workload",
                    name=workload_name,
                ),
            ) from error

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
            if name not in dynamic_parameters and not scalar_contains(value, choices):
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
