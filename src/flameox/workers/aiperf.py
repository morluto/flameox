from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
from typing import cast

from aiperf.common.models import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
    MetricRecordInfo,
    MetricValue,
)

from flameox.canonical import sha256_id
from flameox.workers.aiperf_contract import (
    AIPERF_WORKER,
    AIPerfErrorCategory,
    AIPerfOutcome,
    AIPerfProjectionRow,
    AIPerfWorkerRequest,
    AIPerfWorkerResult,
)
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerFailureKind,
    WorkerOutputFile,
    run_typed_worker,
)

_ERROR_TYPES = {
    "AuthenticationError": "authentication",
    "CancelledError": "cancelled",
    "ConnectionError": "connection",
    "InvalidRequestError": "invalid_request",
    "NotFoundError": "not_found",
    "PermissionError": "permission_denied",
    "RateLimitError": "rate_limited",
    "RequestCancellationError": "cancelled",
    "ServerError": "server_error",
    "TimeoutError": "timeout",
    "UnavailableError": "unavailable",
}
_TIME_FACTORS = {"ns": 1.0, "us": 1_000.0, "µs": 1_000.0, "ms": 1_000_000.0, "s": 1e9}


def _duration_ns(metric: MetricValue | None) -> int | None:
    if metric is None or isinstance(metric.value, bool):
        return None
    value = float(metric.value)
    factor = _TIME_FACTORS.get(metric.unit)
    if factor is None or not math.isfinite(value) or value < 0:
        return None
    return round(value * factor)


def _token_count(metrics: dict[str, MetricValue], name: str) -> int:
    metric = metrics.get(name)
    if metric is None or metric.unit != "tokens" or isinstance(metric.value, bool):
        raise ValueError(f"{name} must be reported in tokens")
    value = float(metric.value)
    rounded = round(value)
    if not math.isfinite(value) or value < 0 or value != rounded:
        raise ValueError(f"{name} must be a non-negative integer")
    return rounded


def _error_category(error_type: str | None, code: int | None) -> AIPerfErrorCategory:
    if code == 401:
        return "authentication"
    if code == 403:
        return "permission_denied"
    if code == 404:
        return "not_found"
    if code == 408:
        return "timeout"
    if code == 429:
        return "rate_limited"
    if code is not None and 500 <= code <= 599:
        return "server_error"
    return cast(AIPerfErrorCategory, _ERROR_TYPES.get(error_type or "", "provider_error"))


def _projection(record: MetricRecordInfo, line_index: int) -> AIPerfProjectionRow:
    metadata = record.metadata
    input_tokens = _token_count(record.metrics, "input_sequence_length")
    output_tokens = _token_count(record.metrics, "output_sequence_length")
    latency_ns = _duration_ns(record.metrics.get("request_latency"))
    ttft_ns = _duration_ns(record.metrics.get("time_to_first_token"))
    tpot_ns = (
        round((latency_ns - ttft_ns) / (output_tokens - 1))
        if latency_ns is not None
        and ttft_ns is not None
        and latency_ns >= ttft_ns
        and output_tokens > 1
        else None
    )
    source_request_id = (
        f"{metadata.conversation_id}:{metadata.turn_index}"
        if metadata.conversation_id is not None and metadata.turn_index is not None
        else str(metadata.session_num)
    )
    error_type = None
    error_code = None
    if record.error is not None:
        error_type = _error_category(record.error.type, record.error.code)
        error_code = str(record.error.code) if record.error.code is not None else None
    outcome: AIPerfOutcome = (
        "cancelled"
        if metadata.was_cancelled
        else ("failed" if record.error is not None else "succeeded")
    )
    return AIPerfProjectionRow(
        line_index=line_index,
        source_request_id=source_request_id,
        provider_request_id=metadata.x_request_id,
        conversation_id=metadata.conversation_id,
        turn_index=metadata.turn_index,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        scheduled_ns=metadata.credit_issued_ns,
        observed_started_ns=metadata.request_start_ns,
        ttft_ns=ttft_ns,
        latency_ns=latency_ns,
        tpot_ns=tpot_ns,
        mean_itl_ns=_duration_ns(record.metrics.get("inter_token_latency")),
        outcome=outcome,
        error_type=error_type,
        error_code=error_code,
    )


def _handle(request: AIPerfWorkerRequest, job_root: Path) -> AIPerfWorkerResult:
    source = Path(request.artifact_path)
    output = job_root / "projection.jsonl"
    temporary = output.with_suffix(".tmp")
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    truncated = False
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            line_index = 0
            while raw := input_stream.readline(request.max_line_bytes + 1):
                if not raw.strip():
                    line_index += 1
                    continue
                if row_count >= request.max_rows:
                    truncated = True
                    break
                if len(raw) > request.max_line_bytes:
                    raise ValueError(f"record line {line_index} exceeds the byte limit")
                record = MetricRecordInfo.model_validate_json(raw)
                encoded = _projection(record, line_index).model_dump_json().encode() + b"\n"
                output_stream.write(encoded)
                digest.update(encoded)
                byte_count += len(encoded)
                row_count += 1
                line_index += 1
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = AIPerfWorkerResult(
        output=WorkerOutputFile(
            role="request_projection",
            relative_path="projection.jsonl",
            media_type="application/x-ndjson",
            byte_length=byte_count,
            sha256=sha256_id(digest.hexdigest()),
        ),
        row_count=row_count,
        truncated=truncated,
        aiperf_version=importlib.metadata.version("aiperf"),
    )
    return result


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=AIPERF_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="The AIPerf export violates its native record schema",
            caught=(OSError, ValueError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
