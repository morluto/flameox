"""Typed command construction and passive probes for maintained inference tools.

The providers own request generation and their native report formats.  This
module intentionally owns only the small, stable boundary Flameox needs:
allowlisted argv construction, executable discovery, and non-invasive server
readiness checks.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from flameox.action_graph import manual_action
from flameox.application.provider_runtime import ProviderRuntime
from flameox.command_binding import ExecutableResolver
from flameox.domain import DomainError, ErrorCode
from flameox.domain.executables import ResolvedExecutable
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.http_transport import (
    BoundedHttpClient,
    BoundedHttpError,
    BoundedHttpResponse,
    HttpMethod,
    LoopbackHttpRequest,
    validate_loopback_base_url,
)
from flameox.models import ContractModel

_MAX_PROBE_RESPONSE_BYTES = 1024 * 1024


class InferenceServerMode(StrEnum):
    MANAGED = "managed"
    EXISTING_LOCAL = "existing_local"


class InferenceServerProvider(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"


class InferenceScenarioProvider(StrEnum):
    AIPERF = "aiperf"
    VLLM_BENCH = "vllm_bench"
    SGLANG_BENCH = "sglang_bench"


class InferenceEndpointType(StrEnum):
    CHAT = "chat"
    COMPLETIONS = "completions"


class InferenceTool(StrEnum):
    AIPERF = "aiperf"
    VLLM = "vllm"
    SGLANG = "sglang"


def _loopback_http_url(value: str) -> str:
    return validate_loopback_base_url(value)


class AIPerfProfileRequest(ContractModel):
    executable: Path
    base_url: str
    model: Annotated[str, Field(min_length=1, max_length=500)]
    tokenizer: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    endpoint_type: InferenceEndpointType = InferenceEndpointType.CHAT
    streaming: bool = True
    trace_path: Path | None = None
    output_dir: Path
    fixed_schedule: bool = True
    speedup_ratio: Annotated[float, Field(gt=0, le=100)] = 1.0
    concurrency: Annotated[int, Field(gt=0, le=100_000)] | None = None
    request_rate: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    warmup_request_count: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0
    request_count: Annotated[int, Field(gt=0, le=10_000_000)] | None = None

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        return _loopback_http_url(value)

    @model_validator(mode="after")
    def replay_schedule_is_unambiguous(self) -> AIPerfProfileRequest:
        if self.trace_path is None and self.fixed_schedule:
            raise ValueError("fixed_schedule requires a Mooncake trace")
        if self.burstiness is not None and self.request_rate is None:
            raise ValueError("burstiness requires request_rate")
        return self

    def argv(self) -> tuple[str, ...]:
        values = [
            str(self.executable),
            "profile",
            "--url",
            self.base_url,
            "--model",
            self.model,
            "--endpoint-type",
            self.endpoint_type,
            "--output-artifact-dir",
            str(self.output_dir),
            "--export-level",
            "records",
        ]
        if self.tokenizer is not None:
            values.extend(("--tokenizer", self.tokenizer))
        if self.streaming:
            values.append("--streaming")
        if self.trace_path is not None:
            values.extend(
                ("--input-file", str(self.trace_path), "--custom-dataset-type", "mooncake_trace")
            )
            if self.fixed_schedule:
                values.append("--fixed-schedule")
            if self.speedup_ratio != 1.0:
                values.extend(("--synthesis-speedup-ratio", str(self.speedup_ratio)))
        if self.concurrency is not None:
            values.extend(("--concurrency", str(self.concurrency)))
        if self.request_rate is not None:
            values.extend(("--request-rate", str(self.request_rate)))
        if self.burstiness is not None:
            values.extend(("--arrival-pattern", "gamma", "--vllm-burstiness", str(self.burstiness)))
        if self.warmup_request_count:
            values.extend(("--warmup-request-count", str(self.warmup_request_count)))
        values.extend(("--random-seed", str(self.seed)))
        if self.request_count is not None:
            values.extend(("--request-count", str(self.request_count)))
        return tuple(values)


class VllmBenchServeRequest(ContractModel):
    executable: Path
    base_url: str
    model: Annotated[str, Field(min_length=1, max_length=500)]
    result_path: Path
    endpoint_type: InferenceEndpointType = InferenceEndpointType.CHAT
    streaming: Literal[True] = True
    num_prompts: Annotated[int, Field(gt=0, le=10_000_000)]
    request_rate: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    max_concurrency: Annotated[int, Field(gt=0, le=100_000)] | None = None
    warmup_request_count: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        return _loopback_http_url(value)

    @model_validator(mode="after")
    def burstiness_has_request_rate(self) -> VllmBenchServeRequest:
        if self.burstiness is not None and self.request_rate is None:
            raise ValueError("burstiness requires request_rate")
        return self

    def argv(self) -> tuple[str, ...]:
        values = [
            str(self.executable),
            "bench",
            "serve",
            "--base-url",
            self.base_url,
            "--model",
            self.model,
            "--backend",
            "openai-chat" if self.endpoint_type is InferenceEndpointType.CHAT else "openai",
            "--endpoint",
            (
                "/v1/chat/completions"
                if self.endpoint_type is InferenceEndpointType.CHAT
                else "/v1/completions"
            ),
            "--num-prompts",
            str(self.num_prompts),
            "--seed",
            str(self.seed),
            "--save-result",
            "--result-dir",
            str(self.result_path.parent),
            "--result-filename",
            self.result_path.name,
        ]
        if self.request_rate is not None:
            values.extend(("--request-rate", str(self.request_rate)))
        if self.burstiness is not None:
            values.extend(("--burstiness", str(self.burstiness)))
        if self.max_concurrency is not None:
            values.extend(("--max-concurrency", str(self.max_concurrency)))
        values.extend(("--num-warmups", str(self.warmup_request_count)))
        return tuple(values)


class SglangBenchServingRequest(ContractModel):
    """Bounded SGLang 0.5.16 ``bench_serving`` random-workload invocation."""

    benchmark_python: Path
    base_url: str
    model: Annotated[str, Field(min_length=1, max_length=500)]
    tokenizer: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    result_path: Path
    endpoint_type: InferenceEndpointType = InferenceEndpointType.CHAT
    streaming: Literal[True] = True
    num_prompts: Annotated[int, Field(gt=0, le=10_000_000)]
    random_input_len: Annotated[int, Field(gt=0, le=1_000_000)]
    random_output_len: Annotated[int, Field(gt=0, le=1_000_000)]
    random_range_ratio: Annotated[float, Field(gt=0, le=1)] = 1.0
    request_rate: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    max_concurrency: Annotated[int, Field(gt=0, le=100_000)] | None = None
    warmup_request_count: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        value = _loopback_http_url(value)
        if urlsplit(value).path not in ("", "/"):
            raise ValueError("sglang bench_serving requires a root base_url in v1")
        return value

    @field_validator("benchmark_python")
    @classmethod
    def absolute_benchmark_python(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("benchmark_python must be absolute")
        return value

    def argv(self) -> tuple[str, ...]:
        parsed = urlsplit(self.base_url)
        assert parsed.hostname is not None
        values = [
            str(self.benchmark_python),
            "-m",
            "sglang.bench_serving",
            "--backend",
            (
                "sglang-oai-chat"
                if self.endpoint_type is InferenceEndpointType.CHAT
                else "sglang-oai"
            ),
            "--host",
            parsed.hostname,
            "--port",
            str(parsed.port or 80),
            "--model",
            self.model,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(self.random_input_len),
            "--random-output-len",
            str(self.random_output_len),
            "--random-range-ratio",
            str(self.random_range_ratio),
            "--num-prompts",
            str(self.num_prompts),
            "--seed",
            str(self.seed),
            "--output-file",
            str(self.result_path),
        ]
        if self.tokenizer is not None:
            values.extend(("--tokenizer", self.tokenizer))
        if self.request_rate is not None:
            values.extend(("--request-rate", str(self.request_rate)))
        if self.max_concurrency is not None:
            values.extend(("--max-concurrency", str(self.max_concurrency)))
        values.extend(("--warmup-requests", str(self.warmup_request_count)))
        return tuple(values)


class _InferenceToolDiscovery(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    tool: InferenceTool
    version: str | None = None
    executable_digest: str | None = None
    provider_environment_id: str | None = None
    provider_python: Path | None = None
    provider_python_sha256: str | None = None
    available: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def compatible(self) -> bool:
        return self.available


class AvailableInferenceToolDiscovery(_InferenceToolDiscovery):
    executable: Path
    executable_binding: ResolvedExecutable
    available: Literal[True] = True
    compatibility_reason: Literal[None] = None
    remediation: tuple[()] = ()


class UnavailableInferenceToolDiscovery(_InferenceToolDiscovery):
    executable: Path | None = None
    available: Literal[False] = False
    compatibility_reason: str | None = None
    remediation: tuple[str, ...] = ()


type InferenceToolDiscovery = Annotated[
    AvailableInferenceToolDiscovery | UnavailableInferenceToolDiscovery,
    Field(discriminator="available"),
]

_INFERENCE_TOOL_DISCOVERY_ADAPTER: TypeAdapter[InferenceToolDiscovery] = TypeAdapter(
    InferenceToolDiscovery
)


def parse_inference_tool_discovery(value: object) -> InferenceToolDiscovery:
    """Parse a probe result into an available or unavailable tool case."""

    return _INFERENCE_TOOL_DISCOVERY_ADAPTER.validate_python(value)


def discover_sglang(
    benchmark_python: Path, *, broker: SubprocessBroker | None = None
) -> InferenceToolDiscovery:
    """Discover SGLang through its declared launcher, never Flameox's PATH."""
    executable = benchmark_python.resolve()
    binding = (
        ExecutableResolver().resolve_host_tool(str(executable), cwd=executable.parent)
        if executable.is_file() and os.access(executable, os.X_OK)
        else None
    )
    digest = binding.identity.sha256 if binding is not None else _digest_executable(executable)
    tool_version: str | None = None
    if binding is not None:
        try:
            outcome = (broker or SubprocessBroker()).run_sync(
                ExecutionRequest(
                    argv=(
                        str(executable),
                        "-c",
                        "from importlib.metadata import version; print(version('sglang'))",
                    ),
                    executable_binding=binding,
                    cwd=executable.parent,
                    environment_allowlist=("PATH",),
                    allowed_working_roots=(executable.parent,),
                    timeout_seconds=5,
                    max_output_bytes=1024,
                )
            )
            if outcome.process.exit_code == 0:
                candidate = outcome.stdout.decode("utf-8", errors="strict").strip()
                tool_version = candidate if candidate else None
        except (DomainError, OSError, UnicodeDecodeError):
            pass
    compatible = tool_version == "0.5.16"
    if binding is not None and compatible:
        return AvailableInferenceToolDiscovery(
            tool=InferenceTool.SGLANG,
            executable=executable,
            executable_binding=binding,
            version=tool_version,
            executable_digest=digest,
        )
    return UnavailableInferenceToolDiscovery(
        tool=InferenceTool.SGLANG,
        executable=executable if executable.is_file() else None,
        version=tool_version,
        executable_digest=digest,
        compatibility_reason="SGLang 0.5.16 is required in benchmark_python.",
        remediation=(
            "Install sglang==0.5.16 in the declared benchmark_python runtime; "
            "Flameox does not install it.",
        ),
    )


def _digest_executable(executable: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with executable.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return f"sha256:{digest.hexdigest()}"


def discover_inference_tool(
    tool: Literal[InferenceTool.AIPERF, InferenceTool.VLLM],
    *,
    provider_runtime: ProviderRuntime | None = None,
) -> InferenceToolDiscovery:
    if tool is InferenceTool.AIPERF and provider_runtime is None:
        return UnavailableInferenceToolDiscovery(
            tool=tool,
            compatibility_reason="No verified AIPerf provider environment is available.",
            remediation=(
                "Call start_capability_setup with adapters=['aiperf'], wait for completion, "
                "then plan the inference scenario again.",
            ),
        )
    if provider_runtime is not None:
        candidate = provider_runtime.executable
        binding = (
            ExecutableResolver().resolve_host_tool(str(candidate), cwd=provider_runtime.root)
            if candidate is not None
            else None
        )
    else:
        scripts_dir = Path(sys.executable).resolve().parent
        environment = dict(os.environ)
        environment["PATH"] = os.pathsep.join((str(scripts_dir), environment.get("PATH", "")))
        binding = ExecutableResolver().resolve_host_tool(str(tool), environment=environment)
    executable = binding.canonical_target if binding is not None else None
    tool_version: str | None = None
    executable_digest: str | None = None
    if executable is not None and binding is not None:
        executable_digest = binding.identity.sha256
        if provider_runtime is not None:
            tool_version = provider_runtime.receipt.distributions.get(str(tool))
        else:
            try:
                tool_version = version(tool)
            except PackageNotFoundError:
                tool_version = None
    compatible = executable is not None
    compatibility_reason: str | None = None
    if executable is not None and tool is InferenceTool.AIPERF:
        compatible = tool_version is not None and tool_version.split(".")[:2] == ["0", "12"]
        if not compatible:
            compatibility_reason = (
                "AIPerf version is unknown or outside Flameox's supported >=0.12,<0.13 range."
            )
    if executable is not None and binding is not None and compatible:
        return AvailableInferenceToolDiscovery(
            tool=tool,
            executable=executable,
            executable_binding=binding,
            version=tool_version,
            executable_digest=executable_digest,
            provider_environment_id=(
                provider_runtime.receipt.environment_id if provider_runtime is not None else None
            ),
            provider_python=provider_runtime.python if provider_runtime is not None else None,
            provider_python_sha256=(
                provider_runtime.receipt.python_sha256 if provider_runtime is not None else None
            ),
        )
    return UnavailableInferenceToolDiscovery(
        tool=tool,
        executable=executable,
        version=tool_version,
        executable_digest=executable_digest,
        compatibility_reason=compatibility_reason,
        remediation=(
            (
                "Call start_capability_setup with adapters=['aiperf'] to create a compatible "
                "provider environment, then retry.",
            )
            if tool is InferenceTool.AIPERF
            else ("Install vLLM in the target runtime; Flameox does not install it.",)
        ),
    )


class ExistingServerProbe(ContractModel):
    base_url: str
    health_ready: bool
    model_ids: tuple[str, ...]


class _ModelRecord(ContractModel):
    model_config = ConfigDict(extra="ignore")

    id: Annotated[str, Field(min_length=1, max_length=500)]


class _ModelsResponse(ContractModel):
    model_config = ConfigDict(extra="ignore")

    data: tuple[_ModelRecord, ...]


def probe_existing_vllm_server(
    base_url: str,
    *,
    timeout_seconds: float = 2.0,
    http_client: BoundedHttpClient | None = None,
) -> ExistingServerProbe:
    """Probe only documented read endpoints; never send an inference request."""

    normalized = _loopback_http_url(base_url)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    deadline = time.monotonic() + timeout_seconds
    client = http_client or BoundedHttpClient()
    owns_client = http_client is None
    try:
        health = client.request_loopback(_probe_request(normalized, "/health", deadline))
        models = client.request_loopback(
            _probe_request(normalized, "/v1/models", deadline, maximum=_MAX_PROBE_RESPONSE_BYTES)
        )
    except BoundedHttpError as exc:
        raise _probe_failure(normalized, "health_and_models") from exc
    finally:
        if owns_client:
            client.close()
    return _parse_probe(normalized, health, models)


async def probe_existing_vllm_server_async(
    base_url: str,
    *,
    timeout_seconds: float = 2.0,
    http_client: BoundedHttpClient | None = None,
) -> ExistingServerProbe:
    """Asynchronously probe the same bounded read-only readiness contract."""

    normalized = _loopback_http_url(base_url)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    deadline = time.monotonic() + timeout_seconds
    client = http_client or BoundedHttpClient()
    owns_client = http_client is None
    try:
        health = await client.request_loopback_async(
            _probe_request(normalized, "/health", deadline)
        )
        models = await client.request_loopback_async(
            _probe_request(normalized, "/v1/models", deadline, maximum=_MAX_PROBE_RESPONSE_BYTES)
        )
    except BoundedHttpError as exc:
        raise _probe_failure(normalized, "health_and_models") from exc
    finally:
        if owns_client:
            await client.aclose()
    return _parse_probe(normalized, health, models)


def _probe_request(
    base_url: str,
    path: Literal["/health", "/v1/models"],
    deadline: float,
    *,
    maximum: int = 64 * 1024,
) -> LoopbackHttpRequest:
    return LoopbackHttpRequest(
        base_url=base_url,
        method=HttpMethod.GET,
        path=path,
        deadline_monotonic=deadline,
        max_response_bytes=maximum,
    )


def _parse_probe(
    normalized: str,
    health: BoundedHttpResponse,
    models: BoundedHttpResponse,
) -> ExistingServerProbe:
    try:
        payload = _ModelsResponse.model_validate(models.json())
    except (BoundedHttpError, ValidationError) as exc:
        raise _probe_failure(normalized, "models_schema") from exc
    return ExistingServerProbe(
        base_url=normalized,
        health_ready=200 <= health.status_code < 300,
        model_ids=tuple(record.id for record in payload.data),
    )


def _probe_failure(base_url: str, stage: str) -> DomainError:
    if stage == "models_schema":
        return DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "The inference server returned an unsupported /v1/models response.",
            remediation=("Inspect the declared server's OpenAI-compatible models endpoint.",),
            details={"base_url": base_url, "probe_stage": stage},
            next_action=manual_action(
                "Repair the declared server's OpenAI-compatible models endpoint before retrying."
            ),
        )
    return DomainError(
        ErrorCode.CAPABILITY_UNAVAILABLE,
        "The loopback inference server did not satisfy passive readiness probes.",
        remediation=("Start or inspect the declared local server, then retry planning.",),
        details={"base_url": base_url, "probe_stage": stage},
        next_action=manual_action(
            "Start or inspect the declared loopback inference server before retrying."
        ),
    )
