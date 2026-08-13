from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from flameox.domain.models import ComparisonValidity, OracleStatus
from flameox.models import ContractModel

# ---------------------------------------------------------------------------
# Inference replay protocol identity
# ---------------------------------------------------------------------------
#
# Two inference replay runs are only comparable when they share the same
# protocol identity: the same trace input, schedule, model, server, hardware,
# profiler, and oracle. This module defines that typed identity and a pure
# comparator that returns the exact mismatched fields and exploratory reasons
# without touching the existing comparison service, CLI/MCP, execution, or
# provider/parser modules.
#
# Optional facets can be genuinely inapplicable (for example, a full trace has
# no replay-window bounds and an unprofiled run has no profiler version).
# Confirmatory completeness is therefore checked explicitly below instead of
# treating every pair of ``None`` values as missing evidence. A value declared
# on only one side remains an exact mismatch.

_Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
_Identifier = Annotated[str, StringConstraints(min_length=1, max_length=200)]
_NonEmptyStr = Annotated[str, StringConstraints(min_length=1, max_length=500)]


class TraceIdentity(ContractModel):
    """Identity of the replayed inference trace input."""

    format: Literal["mooncake", "aiperf", "vllm", "sglang.bench_serving", "custom"] | None = None
    producer: _Identifier
    producer_version: _NonEmptyStr | None = None
    artifact_digest: _Digest | None = None
    window_start_ms: Annotated[int, Field(ge=0)] | None = None
    window_end_ms: Annotated[int, Field(ge=0)] | None = None
    request_count: Annotated[int, Field(ge=0)] | None = None


class ScheduleIdentity(ContractModel):
    """Replay timing and concurrency schedule."""

    preserve_timing: bool
    time_scale: Annotated[float, Field(gt=0)] = 1.0
    max_concurrency: Annotated[int, Field(ge=1)] | None = None
    request_rate: Annotated[float, Field(gt=0)] | None = None
    burstiness: Annotated[float, Field(gt=0)] | None = None
    duration_seconds: Annotated[float, Field(gt=0)] | None = None
    warmup_request_count: Annotated[int, Field(ge=0)] = 0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0


class ModelIdentity(ContractModel):
    """Model and tokenizer identity for the replayed workload."""

    model_id: _Identifier
    model_revision: _NonEmptyStr | None = None
    tokenizer_id: _Identifier | None = None
    tokenizer_revision: _NonEmptyStr | None = None
    trust_remote_code: bool = False
    dtype: Literal["auto", "float16", "bfloat16", "float32", "float8", "other"] = "auto"
    quantization: _NonEmptyStr | None = None


class ServerConfigIdentity(ContractModel):
    """vLLM / serving server configuration identity."""

    backend: Literal["vllm", "sglang", "openai-chat", "custom"]
    endpoint: _NonEmptyStr | None = None
    kv_transfer_config: dict[str, str] = Field(default_factory=dict)
    cache_backend: Literal["none", "mooncake", "lmcache", "vllm_paged", "custom"] = "vllm_paged"
    tensor_parallel_size: Annotated[int, Field(ge=1)] | None = None
    gpu_memory_utilization: Annotated[float, Field(ge=0, le=1)] | None = None
    max_model_len: Annotated[int, Field(gt=0)] | None = None
    managed_server_command_digest: _Digest | None = None
    server_executable_digest: _Digest | None = None
    server_version: _NonEmptyStr | None = None


class HardwareIdentity(ContractModel):
    """Hardware facet for the replay run."""

    accelerator_kind: Literal["cuda", "hip", "xpu", "mps", "cpu", "unknown"] = "unknown"
    accelerator_count: Annotated[int, Field(ge=0)] | None = None
    accelerator_model: _NonEmptyStr | None = None
    driver_version: _NonEmptyStr | None = None
    runtime_version: _NonEmptyStr | None = None
    topology_digest: _Digest | None = None


class _ProfilerState(ContractModel):
    """Profiler attachment state during the replay."""

    profiler_version: _NonEmptyStr | None = None


class ProfilerKind(StrEnum):
    NONE = "none"
    NSIGHT_SYSTEMS = "nsight_systems"
    TORCH_PROFILER = "torch_profiler"
    PERFETTO = "perfetto"
    CUSTOM = "custom"


class ProfilerState(_ProfilerState):
    profiler: ProfilerKind = ProfilerKind.NONE
    attached: Literal[False] = False


class AttachedProfilerState(_ProfilerState):
    profiler: Literal[
        ProfilerKind.NSIGHT_SYSTEMS,
        ProfilerKind.TORCH_PROFILER,
        ProfilerKind.PERFETTO,
        ProfilerKind.CUSTOM,
    ]
    attached: Literal[True] = True


type InferenceProfilerState = Annotated[
    ProfilerState | AttachedProfilerState,
    Field(discriminator="attached"),
]


class OracleIdentity(ContractModel):
    """Semantic oracle identity used for replay correctness assessment."""

    kind: Literal["none", "execution_check", "contract_check", "cross_treatment_equivalence"]
    estimand: _NonEmptyStr | None = None
    tolerance_absolute: Annotated[float, Field(ge=0)] | None = None
    tolerance_relative: Annotated[float, Field(ge=0)] | None = None
    command_digest: _Digest | None = None


class OracleResult(ContractModel):
    """The observed oracle outcome for one replay run."""

    status: OracleStatus
    reason: _Identifier
    absolute_error: Annotated[float, Field(ge=0)] | None = None
    relative_error: Annotated[float, Field(ge=0)] | None = None


class InferenceProtocolIdentity(ContractModel):
    """The complete typed protocol identity of one inference replay run."""

    schema_version: Literal[1] = 1
    provider: _Identifier
    provider_version: _NonEmptyStr | None = None
    provider_executable_digest: _Digest | None = None
    provider_environment_id: _Digest | None = None
    provider_python_digest: _Digest | None = None
    trace: TraceIdentity
    schedule: ScheduleIdentity
    model: ModelIdentity
    server: ServerConfigIdentity
    hardware: HardwareIdentity
    profiler: InferenceProfilerState
    oracle: OracleIdentity
    oracle_result: OracleResult | None = None


# ---------------------------------------------------------------------------
# Compatibility comparison
# ---------------------------------------------------------------------------


class ProtocolMismatch(ContractModel):
    """One exact field-level mismatch between two protocol identities."""

    field: _Identifier
    baseline: str
    candidate: str


class ExploratoryReason(ContractModel):
    """A reason a comparison is exploratory rather than valid or invalid."""

    field: _Identifier
    reason: _NonEmptyStr


class InferenceProtocolComparison(ContractModel):
    """The result of comparing two inference replay protocol identities."""

    schema_version: Literal[1] = 1
    validity: ComparisonValidity
    mismatches: tuple[ProtocolMismatch, ...] = ()
    exploratory_reasons: tuple[ExploratoryReason, ...] = ()

    @property
    def is_comparable(self) -> bool:
        return self.validity is not ComparisonValidity.INVALID


# Each entry maps a dotted field path to a pair of getter lambdas returning the
# comparable value. ``None`` means the facet was not declared; a non-``None``
# mismatch invalidates the comparison, while two ``None`` values make it
# exploratory.
_FacetGetter = tuple[
    str,
    tuple[
        Callable[[InferenceProtocolIdentity], object],
        Callable[[InferenceProtocolIdentity], object],
    ],
]


def _facets() -> tuple[_FacetGetter, ...]:
    return (
        ("provider", (lambda p: p.provider, lambda p: p.provider)),
        ("provider_version", (lambda p: p.provider_version, lambda p: p.provider_version)),
        (
            "provider_executable_digest",
            (
                lambda p: p.provider_executable_digest,
                lambda p: p.provider_executable_digest,
            ),
        ),
        (
            "provider_environment_id",
            (lambda p: p.provider_environment_id, lambda p: p.provider_environment_id),
        ),
        (
            "provider_python_digest",
            (lambda p: p.provider_python_digest, lambda p: p.provider_python_digest),
        ),
        ("trace.format", (lambda p: p.trace.format, lambda p: p.trace.format)),
        ("trace.producer", (lambda p: p.trace.producer, lambda p: p.trace.producer)),
        (
            "trace.producer_version",
            (lambda p: p.trace.producer_version, lambda p: p.trace.producer_version),
        ),
        (
            "trace.artifact_digest",
            (lambda p: p.trace.artifact_digest, lambda p: p.trace.artifact_digest),
        ),
        (
            "trace.window_start_ms",
            (lambda p: p.trace.window_start_ms, lambda p: p.trace.window_start_ms),
        ),
        (
            "trace.window_end_ms",
            (lambda p: p.trace.window_end_ms, lambda p: p.trace.window_end_ms),
        ),
        (
            "trace.request_count",
            (lambda p: p.trace.request_count, lambda p: p.trace.request_count),
        ),
        (
            "schedule.preserve_timing",
            (lambda p: p.schedule.preserve_timing, lambda p: p.schedule.preserve_timing),
        ),
        (
            "schedule.time_scale",
            (lambda p: p.schedule.time_scale, lambda p: p.schedule.time_scale),
        ),
        (
            "schedule.max_concurrency",
            (lambda p: p.schedule.max_concurrency, lambda p: p.schedule.max_concurrency),
        ),
        (
            "schedule.request_rate",
            (lambda p: p.schedule.request_rate, lambda p: p.schedule.request_rate),
        ),
        (
            "schedule.burstiness",
            (lambda p: p.schedule.burstiness, lambda p: p.schedule.burstiness),
        ),
        (
            "schedule.duration_seconds",
            (lambda p: p.schedule.duration_seconds, lambda p: p.schedule.duration_seconds),
        ),
        (
            "schedule.warmup_request_count",
            (
                lambda p: p.schedule.warmup_request_count,
                lambda p: p.schedule.warmup_request_count,
            ),
        ),
        ("schedule.seed", (lambda p: p.schedule.seed, lambda p: p.schedule.seed)),
        ("model.model_id", (lambda p: p.model.model_id, lambda p: p.model.model_id)),
        (
            "model.model_revision",
            (lambda p: p.model.model_revision, lambda p: p.model.model_revision),
        ),
        (
            "model.tokenizer_id",
            (lambda p: p.model.tokenizer_id, lambda p: p.model.tokenizer_id),
        ),
        (
            "model.tokenizer_revision",
            (lambda p: p.model.tokenizer_revision, lambda p: p.model.tokenizer_revision),
        ),
        (
            "model.trust_remote_code",
            (lambda p: p.model.trust_remote_code, lambda p: p.model.trust_remote_code),
        ),
        ("model.dtype", (lambda p: p.model.dtype, lambda p: p.model.dtype)),
        (
            "model.quantization",
            (lambda p: p.model.quantization, lambda p: p.model.quantization),
        ),
        ("server.backend", (lambda p: p.server.backend, lambda p: p.server.backend)),
        ("server.endpoint", (lambda p: p.server.endpoint, lambda p: p.server.endpoint)),
        (
            "server.kv_transfer_config",
            (lambda p: p.server.kv_transfer_config, lambda p: p.server.kv_transfer_config),
        ),
        (
            "server.cache_backend",
            (lambda p: p.server.cache_backend, lambda p: p.server.cache_backend),
        ),
        (
            "server.tensor_parallel_size",
            (lambda p: p.server.tensor_parallel_size, lambda p: p.server.tensor_parallel_size),
        ),
        (
            "server.gpu_memory_utilization",
            (
                lambda p: p.server.gpu_memory_utilization,
                lambda p: p.server.gpu_memory_utilization,
            ),
        ),
        (
            "server.max_model_len",
            (lambda p: p.server.max_model_len, lambda p: p.server.max_model_len),
        ),
        (
            "server.managed_server_command_digest",
            (
                lambda p: p.server.managed_server_command_digest,
                lambda p: p.server.managed_server_command_digest,
            ),
        ),
        (
            "server.server_executable_digest",
            (
                lambda p: p.server.server_executable_digest,
                lambda p: p.server.server_executable_digest,
            ),
        ),
        (
            "server.server_version",
            (lambda p: p.server.server_version, lambda p: p.server.server_version),
        ),
        (
            "hardware.accelerator_kind",
            (lambda p: p.hardware.accelerator_kind, lambda p: p.hardware.accelerator_kind),
        ),
        (
            "hardware.accelerator_count",
            (lambda p: p.hardware.accelerator_count, lambda p: p.hardware.accelerator_count),
        ),
        (
            "hardware.accelerator_model",
            (lambda p: p.hardware.accelerator_model, lambda p: p.hardware.accelerator_model),
        ),
        (
            "hardware.driver_version",
            (lambda p: p.hardware.driver_version, lambda p: p.hardware.driver_version),
        ),
        (
            "hardware.runtime_version",
            (lambda p: p.hardware.runtime_version, lambda p: p.hardware.runtime_version),
        ),
        (
            "hardware.topology_digest",
            (lambda p: p.hardware.topology_digest, lambda p: p.hardware.topology_digest),
        ),
        (
            "profiler.profiler",
            (lambda p: p.profiler.profiler, lambda p: p.profiler.profiler),
        ),
        (
            "profiler.profiler_version",
            (lambda p: p.profiler.profiler_version, lambda p: p.profiler.profiler_version),
        ),
        (
            "profiler.attached",
            (lambda p: p.profiler.attached, lambda p: p.profiler.attached),
        ),
        ("oracle.kind", (lambda p: p.oracle.kind, lambda p: p.oracle.kind)),
        (
            "oracle.estimand",
            (lambda p: p.oracle.estimand, lambda p: p.oracle.estimand),
        ),
        (
            "oracle.tolerance_absolute",
            (lambda p: p.oracle.tolerance_absolute, lambda p: p.oracle.tolerance_absolute),
        ),
        (
            "oracle.tolerance_relative",
            (lambda p: p.oracle.tolerance_relative, lambda p: p.oracle.tolerance_relative),
        ),
        (
            "oracle.command_digest",
            (lambda p: p.oracle.command_digest, lambda p: p.oracle.command_digest),
        ),
        (
            "oracle_result.status",
            (
                lambda p: p.oracle_result.status if p.oracle_result is not None else None,
                lambda p: p.oracle_result.status if p.oracle_result is not None else None,
            ),
        ),
    )


def _normalize(value: object) -> str:
    """Return a display string for a protocol facet value.

    For dict values, use canonical JSON serialization (sorted keys) so that
    keys/values containing commas or equals signs cannot produce colliding
    normal forms. Plain scalar values use ``str()`` as before.
    """
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _identity_missing_fields(protocol: InferenceProtocolIdentity) -> dict[str, str]:
    missing: dict[str, str] = {}
    if protocol.provider_version is None:
        missing["provider_version"] = "exact provider version is unavailable"
    if protocol.provider_executable_digest is None:
        missing["provider_executable_digest"] = "provider executable digest is unavailable"
    if protocol.trace.artifact_digest is None:
        missing["trace.artifact_digest"] = "source trace digest is unavailable"
    if protocol.model.model_revision is None:
        missing["model.model_revision"] = "model revision is unavailable"
    if protocol.model.tokenizer_id is None:
        missing["model.tokenizer_id"] = "tokenizer identity is unavailable"
    if protocol.model.tokenizer_revision is None:
        missing["model.tokenizer_revision"] = "tokenizer revision is unavailable"
    if protocol.model.quantization is None:
        missing["model.quantization"] = "effective model quantization is unavailable"
    if protocol.server.managed_server_command_digest is None:
        missing["server.managed_server_command_digest"] = (
            "managed server and cache configuration provenance is unavailable"
        )
    if protocol.server.server_executable_digest is None:
        missing["server.server_executable_digest"] = "managed server executable is unavailable"
    if protocol.server.server_version is None:
        missing["server.server_version"] = "exact managed server version is unavailable"
    return missing


def _hardware_missing_fields(hardware: HardwareIdentity) -> dict[str, str]:
    missing: dict[str, str] = {}

    if hardware.accelerator_kind == "unknown":
        missing["hardware.accelerator_kind"] = "hardware class is unavailable"
    if hardware.accelerator_count is None:
        missing["hardware.accelerator_count"] = "accelerator count is unavailable"
    elif hardware.accelerator_kind != "cpu" and hardware.accelerator_count < 1:
        missing["hardware.accelerator_count"] = "accelerator count is not credible"
    if hardware.accelerator_kind not in {"cpu", "unknown"}:
        for field, value, reason in (
            ("accelerator_model", hardware.accelerator_model, "accelerator model is unavailable"),
            ("driver_version", hardware.driver_version, "accelerator driver is unavailable"),
            ("runtime_version", hardware.runtime_version, "accelerator runtime is unavailable"),
            ("topology_digest", hardware.topology_digest, "accelerator topology is unavailable"),
        ):
            if value is None:
                missing[f"hardware.{field}"] = reason
    return missing


def _diagnostic_missing_fields(protocol: InferenceProtocolIdentity) -> dict[str, str]:
    missing: dict[str, str] = {}

    profiler = protocol.profiler
    if profiler.attached and profiler.profiler_version is None:
        missing["profiler.profiler_version"] = "attached profiler version is unavailable"

    oracle = protocol.oracle
    if oracle.kind != "contract_check":
        missing["oracle.kind"] = (
            "a per-run contract oracle is required; cross-treatment equivalence must observe "
            "both treatments"
        )
    if oracle.command_digest is None:
        missing["oracle.command_digest"] = "semantic oracle command identity is unavailable"
    if protocol.oracle_result is None:
        missing["oracle_result.status"] = "semantic oracle result is unavailable"
    elif protocol.oracle_result.status != "pass":
        missing["oracle_result.status"] = (
            f"semantic oracle did not pass ({protocol.oracle_result.status})"
        )
    return missing


def _required_missing_fields(protocol: InferenceProtocolIdentity) -> dict[str, str]:
    """Return context-aware gaps that prevent a confirmatory comparison."""
    return {
        **_identity_missing_fields(protocol),
        **_hardware_missing_fields(protocol.hardware),
        **_diagnostic_missing_fields(protocol),
    }


def compare_inference_protocols(
    baseline: InferenceProtocolIdentity,
    candidate: InferenceProtocolIdentity,
) -> InferenceProtocolComparison:
    """Compare two inference replay protocol identities.

    Returns exact field mismatches and context-aware confirmatory-evidence
    gaps. Optional fields that are absent on both sides are inapplicable, not
    automatically exploratory. Missing values on only one side are mismatches.
    """
    mismatches: list[ProtocolMismatch] = []
    exploratory: list[ExploratoryReason] = []
    for field, (get_baseline, get_candidate) in _facets():
        left = get_baseline(baseline)
        right = get_candidate(candidate)
        if left is None and right is None:
            continue
        left_norm = _normalize(left)
        right_norm = _normalize(right)
        if left_norm != right_norm:
            mismatches.append(
                ProtocolMismatch(field=field, baseline=left_norm, candidate=right_norm)
            )

    mismatched_fields = {item.field for item in mismatches}
    baseline_missing = _required_missing_fields(baseline)
    candidate_missing = _required_missing_fields(candidate)
    for field in sorted(set(baseline_missing) | set(candidate_missing)):
        if field in mismatched_fields:
            continue
        left_reason = baseline_missing.get(field)
        right_reason = candidate_missing.get(field)
        if left_reason is not None and right_reason is not None:
            reason = (
                f"required for confirmatory comparison on both sides: {left_reason}"
                if left_reason == right_reason
                else "required for confirmatory comparison on both sides: "
                f"baseline {left_reason}; candidate {right_reason}"
            )
        elif left_reason is not None:
            reason = f"required for confirmatory comparison on baseline: {left_reason}"
        else:
            reason = f"required for confirmatory comparison on candidate: {right_reason}"
        exploratory.append(ExploratoryReason(field=field, reason=reason))

    if mismatches:
        validity = ComparisonValidity.INVALID
    elif exploratory:
        validity = ComparisonValidity.EXPLORATORY
    else:
        validity = ComparisonValidity.VALID

    return InferenceProtocolComparison(
        validity=validity,
        mismatches=tuple(mismatches),
        exploratory_reasons=tuple(exploratory),
    )
