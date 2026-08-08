"""Typed command construction and passive probes for maintained inference tools.

The providers own request generation and their native report formats.  This
module intentionally owns only the small, stable boundary Flameox needs:
allowlisted argv construction, executable discovery, and non-invasive server
readiness checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.response import addinfourl

from pydantic import Field, field_validator, model_validator

from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel

_MAX_PROBE_RESPONSE_BYTES = 1024 * 1024


class _RejectRedirects(HTTPRedirectHandler):
    """Keep passive loopback probes on the declared endpoint."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def _open_probe(request: Request, timeout: float) -> addinfourl:
    return cast(addinfourl, build_opener(_RejectRedirects()).open(request, timeout=timeout))


def _loopback_http_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("server URL must be an unauthenticated http URL")
    try:
        loopback = ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.lower() == "localhost"
    if not loopback or parsed.query or parsed.fragment:
        raise ValueError("server URL must target a loopback address without query or fragment")
    return value.rstrip("/")


class AIPerfProfileRequest(ContractModel):
    executable: Path
    base_url: str
    model: Annotated[str, Field(min_length=1, max_length=500)]
    tokenizer: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    endpoint_type: Literal["chat", "completions"] = "chat"
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
    endpoint_type: Literal["chat", "completions"] = "chat"
    streaming: bool = True
    num_prompts: Annotated[int, Field(gt=0, le=10_000_000)]
    request_rate: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    max_concurrency: Annotated[int, Field(gt=0, le=100_000)] | None = None
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        return _loopback_http_url(value)

    @model_validator(mode="after")
    def require_supported_streaming_mode(self) -> VllmBenchServeRequest:
        if not self.streaming:
            raise ValueError("vllm bench serve non-streaming response mode is unsupported in v1")
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
            "openai-chat" if self.endpoint_type == "chat" else "openai",
            "--endpoint",
            "/v1/chat/completions" if self.endpoint_type == "chat" else "/v1/completions",
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
        return tuple(values)


class InferenceToolDiscovery(ContractModel):
    tool: Literal["aiperf", "vllm"]
    executable: Path | None
    available: bool
    version: str | None = None
    executable_digest: str | None = None
    compatible: bool = False
    compatibility_reason: str | None = None
    remediation: tuple[str, ...] = ()


def discover_inference_tool(tool: Literal["aiperf", "vllm"]) -> InferenceToolDiscovery:
    located = shutil.which(tool)
    if located is None:
        scripts_dir = Path(sys.executable).resolve().parent
        for name in (tool, f"{tool}.exe"):
            sibling = scripts_dir / name
            if sibling.is_file() and os.access(sibling, os.X_OK):
                located = str(sibling)
                break
    executable = Path(located).resolve() if located is not None else None
    tool_version: str | None = None
    executable_digest: str | None = None
    if executable is not None:
        digest = hashlib.sha256()
        try:
            with executable.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            executable_digest = f"sha256:{digest.hexdigest()}"
        except OSError:
            executable_digest = None
        try:
            tool_version = version(tool)
        except PackageNotFoundError:
            tool_version = None
    compatible = executable is not None
    compatibility_reason: str | None = None
    if executable is not None and tool == "aiperf":
        compatible = tool_version is not None and tool_version.split(".")[:2] == ["0", "12"]
        if not compatible:
            compatibility_reason = (
                "AIPerf version is unknown or outside Flameox's supported >=0.12,<0.13 range."
            )
    return InferenceToolDiscovery(
        tool=tool,
        executable=executable,
        available=executable is not None and compatible,
        version=tool_version,
        executable_digest=executable_digest,
        compatible=compatible,
        compatibility_reason=compatibility_reason,
        remediation=(
            ("Install a compatible AIPerf >=0.12,<0.13 in the Flameox runtime, then retry.",)
            if tool == "aiperf" and (executable is None or not compatible)
            else (
                ("Install vLLM in the target runtime; Flameox does not install it.",)
                if executable is None
                else ()
            )
        ),
    )


class ExistingServerProbe(ContractModel):
    base_url: str
    health_ready: bool
    model_ids: tuple[str, ...]


def probe_existing_vllm_server(
    base_url: str, *, timeout_seconds: float = 2.0
) -> ExistingServerProbe:
    """Probe only documented read endpoints; never send an inference request."""

    normalized = _loopback_http_url(base_url)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("passive inference probe deadline expired")
        return value

    try:
        with _open_probe(
            Request(f"{normalized}/health", method="GET"), timeout=remaining()
        ) as response:
            health_ready = response.status is not None and 200 <= response.status < 300
        with _open_probe(
            Request(f"{normalized}/v1/models", method="GET"), timeout=remaining()
        ) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(min(64 * 1024, _MAX_PROBE_RESPONSE_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_PROBE_RESPONSE_BYTES:
                    raise ValueError("/v1/models response exceeded the passive probe limit")
                chunks.append(chunk)
                remaining()
            payload = json.loads(b"".join(chunks))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "The loopback inference server did not satisfy passive readiness probes.",
            remediation=("Start or inspect the declared local server, then retry planning.",),
            details={
                "base_url": normalized,
                "probe_stage": "health_and_models",
                "next_tool": "manual",
            },
        ) from exc
    records = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "The inference server returned an unsupported /v1/models response.",
            remediation=("Inspect the declared server's OpenAI-compatible models endpoint.",),
            details={"base_url": normalized, "probe_stage": "models_schema", "next_tool": "manual"},
        )
    model_ids = tuple(
        item["id"] for item in records if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    return ExistingServerProbe(base_url=normalized, health_ready=health_ready, model_ids=model_ids)
