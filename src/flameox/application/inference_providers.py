"""Typed command construction and passive probes for maintained inference tools.

The providers own request generation and their native report formats.  This
module intentionally owns only the small, stable boundary Flameox needs:
allowlisted argv construction, executable discovery, and non-invasive server
readiness checks.
"""

from __future__ import annotations

import os
import sys
import time
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Literal, NoReturn
from urllib.parse import urlsplit

from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from flameox.action_graph import ActionId, NextAction, manual_action, tool_action
from flameox.application.provider_runtime import ProviderRuntime
from flameox.command_binding import ExecutableResolver
from flameox.domain import DomainError, ErrorCode, process_exit_code
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
        return validate_loopback_base_url(value)

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
        return validate_loopback_base_url(value)

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
    """Bounded SGLang ``benchmark.serving`` random-workload invocation."""

    executable: Path
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
        value = validate_loopback_base_url(value)
        if urlsplit(value).path not in ("", "/"):
            raise ValueError("sglang benchmark serving requires a root base_url")
        return value

    @field_validator("executable")
    @classmethod
    def absolute_executable(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("executable must be absolute")
        return value

    def argv(self) -> tuple[str, ...]:
        parsed = urlsplit(self.base_url)
        assert parsed.hostname is not None
        values = [
            str(self.executable),
            "-m",
            "sglang.benchmark.serving",
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


class QualifiedInferenceTool(ContractModel):
    """One executable admitted for a reviewed inference command."""

    tool: InferenceTool
    executable_binding: ResolvedExecutable
    version: str | None = None
    benchmark_capabilities: Annotated[tuple[str, ...], Field(max_length=32)] = ()
    provider_environment_id: str | None = None


_SGLANG_BENCHMARK_MODULE = "sglang.benchmark.serving"
_SGLANG_REQUIRED_OPTIONS = (
    "--backend",
    "--host",
    "--port",
    "--model",
    "--dataset-name",
    "--random-input-len",
    "--random-output-len",
    "--random-range-ratio",
    "--num-prompts",
    "--seed",
    "--output-file",
    "--warmup-requests",
)


def _capability_unavailable(
    tool: InferenceTool,
    message: str,
    *,
    remediation: tuple[str, ...],
    next_action: NextAction,
    details: dict[str, object] | None = None,
) -> NoReturn:
    raise DomainError(
        ErrorCode.CAPABILITY_UNAVAILABLE,
        message,
        details={"tool": tool.value, **(details or {})},
        remediation=remediation,
        next_action=next_action,
    )


def discover_sglang(
    benchmark_python: Path, *, broker: SubprocessBroker | None = None
) -> QualifiedInferenceTool:
    """Qualify the declared launcher for SGLang's current serving benchmark."""
    executable = benchmark_python.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        _capability_unavailable(
            InferenceTool.SGLANG,
            "The declared SGLang benchmark launcher is missing or not executable.",
            details={"benchmark_python": str(benchmark_python)},
            remediation=(
                "Set benchmark_python to an executable Python runtime with SGLang installed, "
                "then plan the scenario again.",
            ),
            next_action=manual_action(
                "Update the declared SGLang server with an absolute executable benchmark_python.",
                suggested_action=ActionId.CONFIGURE_INFERENCE_SERVER,
                missing_arguments=("name", "operation", "benchmark_python"),
            ),
        )
    binding = ExecutableResolver().resolve_host_tool(str(executable), cwd=executable.parent)
    if binding is None:
        _capability_unavailable(
            InferenceTool.SGLANG,
            "The declared SGLang benchmark launcher could not be qualified.",
            details={"benchmark_python": str(executable)},
            remediation=(
                "Set benchmark_python to an executable trusted by the configured workspace policy, "
                "then plan the scenario again.",
            ),
            next_action=manual_action(
                "Update the declared SGLang server launcher and retry planning.",
                suggested_action=ActionId.CONFIGURE_INFERENCE_SERVER,
                missing_arguments=("name", "operation", "benchmark_python"),
            ),
        )
    try:
        outcome = (broker or SubprocessBroker()).run_sync(
            ExecutionRequest(
                argv=(str(executable), "-m", _SGLANG_BENCHMARK_MODULE, "--help"),
                executable_binding=binding,
                cwd=executable.parent,
                environment_allowlist=("PATH",),
                allowed_working_roots=(executable.parent,),
                timeout_seconds=10,
                max_output_bytes=128 * 1024,
            )
        )
        help_text = outcome.stdout.decode("utf-8", errors="strict")
    except (DomainError, OSError, UnicodeDecodeError):
        _capability_unavailable(
            InferenceTool.SGLANG,
            "The declared launcher could not inspect sglang.benchmark.serving.",
            details={"benchmark_python": str(executable)},
            remediation=(
                "Install an SGLang runtime that provides sglang.benchmark.serving at "
                "benchmark_python, then plan the scenario again.",
            ),
            next_action=manual_action(
                "Repair the declared SGLang benchmark launcher and retry planning.",
                suggested_action=ActionId.CONFIGURE_INFERENCE_SERVER,
                missing_arguments=("name", "operation", "benchmark_python"),
            ),
        )
    supported_options = all(option in help_text for option in _SGLANG_REQUIRED_OPTIONS)
    if process_exit_code(outcome.process.termination) != 0 or not supported_options:
        _capability_unavailable(
            InferenceTool.SGLANG,
            "The declared launcher does not provide Flameox's supported "
            "sglang.benchmark.serving interface.",
            details={
                "benchmark_python": str(executable),
                "module": _SGLANG_BENCHMARK_MODULE,
                "missing_options": tuple(
                    option for option in _SGLANG_REQUIRED_OPTIONS if option not in help_text
                ),
            },
            remediation=(
                "Point benchmark_python at an SGLang runtime that supports the canonical "
                "sglang.benchmark.serving benchmark interface, then plan again.",
            ),
            next_action=manual_action(
                "Update the declared SGLang benchmark launcher and retry planning.",
                suggested_action=ActionId.CONFIGURE_INFERENCE_SERVER,
                missing_arguments=("name", "operation", "benchmark_python"),
            ),
        )
    tool_version: str | None = None
    try:
        version_outcome = (broker or SubprocessBroker()).run_sync(
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
        if process_exit_code(version_outcome.process.termination) == 0:
            candidate = version_outcome.stdout.decode("utf-8", errors="strict").strip()
            tool_version = candidate or None
    except (DomainError, OSError, UnicodeDecodeError):
        pass
    return QualifiedInferenceTool(
        tool=InferenceTool.SGLANG,
        executable_binding=binding,
        version=tool_version,
        benchmark_capabilities=_SGLANG_REQUIRED_OPTIONS,
    )


def discover_inference_tool(
    tool: Literal[InferenceTool.AIPERF, InferenceTool.VLLM],
    *,
    provider_runtime: ProviderRuntime | None = None,
) -> QualifiedInferenceTool:
    if tool is InferenceTool.AIPERF and provider_runtime is None:
        _capability_unavailable(
            tool,
            "No verified AIPerf provider environment is available.",
            remediation=(
                "Call start_capability_setup with adapters=['aiperf'], wait for completion, "
                "then plan the inference scenario again.",
            ),
            next_action=tool_action(
                ActionId.START_CAPABILITY_SETUP,
                adapters=["aiperf"],
                idempotency_key="inference-aiperf",
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
    tool_version: str | None = None
    if binding is not None:
        if provider_runtime is not None:
            tool_version = provider_runtime.receipt.distributions.get(str(tool))
        else:
            try:
                tool_version = version(tool)
            except PackageNotFoundError:
                tool_version = None
    if binding is None:
        _capability_unavailable(
            tool,
            f"No executable for inference tool {tool.value!r} is available.",
            remediation=(
                "Call start_capability_setup with adapters=['aiperf'] to create a qualified "
                "provider environment, then plan again.",
            )
            if tool is InferenceTool.AIPERF
            else ("Install vLLM in the target runtime; Flameox does not install it.",),
            next_action=(
                tool_action(
                    ActionId.START_CAPABILITY_SETUP,
                    adapters=["aiperf"],
                    idempotency_key="inference-aiperf",
                )
                if tool is InferenceTool.AIPERF
                else manual_action(
                    "Install vLLM in the target runtime and retry planning.",
                    suggested_action=ActionId.INSPECT_CAPABILITIES,
                )
            ),
        )
    if tool is InferenceTool.AIPERF and (
        tool_version is None or tool_version.split(".")[:2] != ["0", "12"]
    ):
        _capability_unavailable(
            tool,
            "AIPerf is outside Flameox's supported >=0.12,<0.13 range.",
            details={"version": tool_version},
            remediation=(
                "Call start_capability_setup with adapters=['aiperf'] to create a compatible "
                "provider environment, then plan again.",
            ),
            next_action=tool_action(
                ActionId.START_CAPABILITY_SETUP,
                adapters=["aiperf"],
                idempotency_key="inference-aiperf",
            ),
        )
    return QualifiedInferenceTool(
        tool=tool,
        executable_binding=binding,
        version=tool_version,
        provider_environment_id=(
            provider_runtime.receipt.environment_id if provider_runtime is not None else None
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

    normalized = validate_loopback_base_url(base_url)
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

    normalized = validate_loopback_base_url(base_url)
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
