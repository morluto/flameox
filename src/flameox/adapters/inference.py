from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Literal

import ijson
from ijson import IncompleteJSONError, JSONError
from pydantic import (
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from flameox.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    EvidenceLevel,
    RunType,
    digest_model,
)
from flameox.evidence import (
    CancelledInferenceRequestOutcome,
    FailedInferenceRequestOutcome,
    GenerationPublisher,
    InferenceRequestItem,
    ReportedInferenceRequestOutcome,
    SucceededInferenceRequestOutcome,
    inference_request_outcome_columns,
)
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace

# ---------------------------------------------------------------------------
# Mooncake streaming JSONL request-trace validation
# ---------------------------------------------------------------------------
#
# The Mooncake request trace (see kvcache-ai/Mooncake ``mooncake_trace.jsonl``
# and kobe0938/mooncake-trace-replayer) is a JSON Lines stream where each line
# is one request with four observed fields:
#
#   {"timestamp": <int ms>, "input_length": <int>, "output_length": <int>,
#    "hash_ids": [<int>, ...]}
#
# The parser validates the stream incrementally and yields one typed row per
# line. It never reads the whole file into memory: rows are produced as lines
# are consumed so a truncated or oversized trace fails fast with an explicit
# limitation rather than OOM. The caller bounds consumption via ``max_rows``.

_REQUIRED_FIELDS = frozenset({"timestamp", "input_length", "output_length"})
_SAFE_ERROR_CATEGORIES = {
    "authentication",
    "cancelled",
    "connection",
    "invalid_request",
    "not_found",
    "permission_denied",
    "rate_limited",
    "server_error",
    "timeout",
    "unavailable",
}


def _safe_error_category(value: Any) -> str:
    """Reduce provider-owned error text to a fixed, non-sensitive category."""
    if not isinstance(value, str):
        return "provider_error"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _SAFE_ERROR_CATEGORIES:
        return normalized
    checks = (
        (("timeout", "timed_out", "deadline"), "timeout"),
        (("cancel",), "cancelled"),
        (("rate_limit", "throttl"), "rate_limited"),
        (("unauthor", "authenticat"), "authentication"),
        (("forbidden", "permission"), "permission_denied"),
        (("connect", "network"), "connection"),
        (("validation", "invalid", "bad_request"), "invalid_request"),
        (("not_found",), "not_found"),
        (("unavailable",), "unavailable"),
        (("server", "internal"), "server_error"),
    )
    for needles, category in checks:
        if any(needle in normalized for needle in needles):
            return category
    return "provider_error"


def _safe_error_code(value: Any) -> str | None:
    """Keep bounded numeric status codes; collapse all provider strings."""
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 999:
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isascii() and stripped.isdecimal() and len(stripped) <= 3:
            numeric = int(stripped)
            if 0 <= numeric <= 999:
                return str(numeric)
        return _safe_error_category(stripped)
    return None


class MooncakeRequestRow(ContractModel):
    """One validated, typed request extracted from a Mooncake trace line."""

    schema_version: Literal[1] = 1
    request_id: str
    line_index: Annotated[int, Field(ge=0)]
    timestamp_ms: Annotated[int, Field(ge=0)]
    input_length: Annotated[int, Field(ge=0)]
    output_length: Annotated[int, Field(ge=0)]
    prefix_hash_count: Annotated[int, Field(ge=0)]
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVED


class MooncakeTraceSummary(ContractModel):
    """Bounded summary of a parsed Mooncake trace stream."""

    schema_version: int = 1
    request_count: int
    prefix_hash_count: int
    max_input_length: int
    max_output_length: int
    timestamp_span_ms: int
    limitations: tuple[str, ...] = ()


class MooncakeTraceParser:
    """Streaming validator and normalizer for Mooncake request-trace JSONL.

    ``iter_rows`` yields validated :class:`MooncakeRequestRow` objects one line
    at a time without loading the entire file. ``parse`` consumes up to
    ``max_rows`` lines and returns a bounded summary plus the materialized rows.
    """

    max_line_bytes = 64 * 1024

    def __init__(self, max_rows: int = 1_000_000) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.max_rows = max_rows
        self.truncated = False

    def iter_rows(self, path: Path) -> Iterator[MooncakeRequestRow]:
        """Yield validated rows from a Mooncake trace JSONL file streamingly."""
        try:
            with path.open("rb") as stream:
                index = 0
                while raw := stream.readline(self.max_line_bytes + 1):
                    if not raw.strip():
                        index += 1
                        continue
                    if len(raw) > self.max_line_bytes:
                        raise ValueError(
                            f"trace line {index} exceeds the {self.max_line_bytes}-byte limit"
                        )
                    yield self._row_from_line(raw, index)
                    index += 1
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The Mooncake trace artifact is not a valid JSONL request stream.",
            ) from exc

    def parse(self, path: Path) -> tuple[MooncakeTraceSummary, list[MooncakeRequestRow]]:
        """Consume up to ``max_rows`` lines and return a bounded summary."""
        rows: list[MooncakeRequestRow] = []
        limitations: list[str] = []
        first_timestamp: int | None = None
        last_timestamp: int | None = None
        iterator = iter(self.iter_rows(path))
        while len(rows) < self.max_rows:
            try:
                row = next(iterator)
            except StopIteration:
                break
            if first_timestamp is None:
                first_timestamp = row.timestamp_ms
            if last_timestamp is not None and row.timestamp_ms < last_timestamp:
                limitations.append(
                    f"Request {row.line_index} timestamp regressed below the prior line."
                )
            last_timestamp = row.timestamp_ms
            rows.append(row)
        if len(rows) == self.max_rows:
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                limitations.append(f"Trace truncated at {self.max_rows} requests.")
        if not rows:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The Mooncake trace artifact contains no request lines.",
            )
        if first_timestamp is not None and first_timestamp != 0:
            limitations.append("The first request timestamp is not zero milliseconds.")
        timestamps = [row.timestamp_ms for row in rows]
        return (
            MooncakeTraceSummary(
                request_count=len(rows),
                prefix_hash_count=sum(row.prefix_hash_count for row in rows),
                max_input_length=max(row.input_length for row in rows),
                max_output_length=max(row.output_length for row in rows),
                timestamp_span_ms=max(timestamps) - min(timestamps),
                limitations=tuple(dict.fromkeys(limitations)),
            ),
            rows,
        )

    def _row_from_line(self, raw: bytes, index: int) -> MooncakeRequestRow:
        entry = json.loads(raw)
        timestamp, input_length, output_length, hash_ids = self._validate_entry(entry, index)
        identity = {
            "line_index": index,
            "timestamp_ms": timestamp,
            "input_length": input_length,
            "output_length": output_length,
            "hash_ids": hash_ids,
        }
        return MooncakeRequestRow(
            request_id=digest_model(identity),
            line_index=index,
            timestamp_ms=timestamp,
            input_length=input_length,
            output_length=output_length,
            prefix_hash_count=len(hash_ids),
        )

    @staticmethod
    def _validate_entry(entry: Any, index: int) -> tuple[int, int, int, list[int]]:
        if not isinstance(entry, dict):
            raise ValueError(f"trace line {index} is not a JSON object")
        missing = _REQUIRED_FIELDS.difference(entry)
        if missing:
            raise ValueError(f"trace line {index} is missing required fields: {sorted(missing)}")
        timestamp = entry["timestamp"]
        input_length = entry["input_length"]
        output_length = entry["output_length"]
        hash_ids = entry.get("hash_ids", [])
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            raise ValueError(f"trace line {index} timestamp must be a non-negative int")
        if not isinstance(input_length, int) or isinstance(input_length, bool) or input_length < 0:
            raise ValueError(f"trace line {index} input_length must be a non-negative int")
        if (
            not isinstance(output_length, int)
            or isinstance(output_length, bool)
            or output_length < 0
        ):
            raise ValueError(f"trace line {index} output_length must be a non-negative int")
        if not isinstance(hash_ids, list):
            raise ValueError(f"trace line {index} hash_ids must be a list")
        for value in hash_ids:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"trace line {index} hash_ids must be non-negative ints")
        return timestamp, input_length, output_length, hash_ids


class AIPerfRequestRow(ContractModel):
    """One AIPerf request whose export always reports a concrete outcome."""

    source_request_id: str
    provider_request_id: str | None
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    scheduled_ns: Annotated[int, Field(ge=0)] | None
    observed_started_ns: Annotated[int, Field(ge=0)]
    ttft_ns: Annotated[int, Field(ge=0)] | None
    latency_ns: Annotated[int, Field(ge=0)] | None
    tpot_ns: Annotated[int, Field(ge=0)] | None
    mean_itl_ns: Annotated[int, Field(ge=0)] | None
    outcome: ReportedInferenceRequestOutcome = Field(exclude=True)
    queue_ns: None = None
    prefill_ns: None = None
    decode_ns: None = None
    cache_hit: None = None
    prefix_hash_count: None = None
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVED
    line_index: Annotated[int, Field(ge=0)]

    @property
    def success(self) -> bool:
        success = inference_request_outcome_columns(self.outcome).success
        assert success is not None
        return success

    @property
    def cancelled(self) -> bool:
        cancelled = inference_request_outcome_columns(self.outcome).cancelled
        assert cancelled is not None
        return cancelled

    @property
    def error_type(self) -> str | None:
        return inference_request_outcome_columns(self.outcome).error_type

    @property
    def error_code(self) -> str | None:
        return inference_request_outcome_columns(self.outcome).error_code

    def evidence_columns(self) -> dict[str, Any]:
        columns = self.model_dump(mode="python", exclude={"line_index"})
        columns.update(
            {
                "success": self.success,
                "cancelled": self.cancelled,
                "error_type": self.error_type,
                "error_code": self.error_code,
            }
        )
        return columns


class AIPerfRecordParser:
    """Stream AIPerf 0.12 record exports without retaining provider payloads."""

    max_line_bytes = 1024 * 1024

    def __init__(self, max_rows: int = 1_000_000) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.max_rows = max_rows

    def iter_rows(
        self, path: Path, *, inputs_index: AIPerfInputsIndex | None = None
    ) -> Iterator[AIPerfRequestRow]:
        self.truncated = False
        self._inputs_index = inputs_index
        self._corr_matched = 0
        self._corr_missing_session = 0
        self._corr_turn_out_of_range = 0
        self._corr_no_id = 0
        record_count = 0
        try:
            with path.open("rb") as stream:
                line_index = 0
                while raw := stream.readline(self.max_line_bytes + 1):
                    if not raw.strip():
                        line_index += 1
                        continue
                    if record_count >= self.max_rows:
                        self.truncated = True
                        break
                    if len(raw) > self.max_line_bytes:
                        raise ValueError(f"record line {line_index} exceeds the byte limit")
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError(f"record line {line_index} is not an object")
                    if inputs_index is not None:
                        self._correlate(payload, inputs_index)
                    yield self._normalize(payload, line_index)
                    record_count += 1
                    line_index += 1
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The AIPerf profile export violates the supported 0.12 record schema.",
            ) from exc

    def _correlate(self, payload: dict[str, Any], index: AIPerfInputsIndex) -> None:
        """Track correlation status against the inputs index without modifying the row.

        Called before ``_normalize`` so the raw ``metadata`` is available for the
        ``conversation_id`` / ``turn_index`` lookup. Counts are stored on the
        parser instance and read by :meth:`correlation_summary`.
        """
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            self._corr_no_id += 1
            return
        conversation_id = metadata.get("conversation_id")
        turn_index = metadata.get("turn_index")
        if (
            not isinstance(conversation_id, str)
            or not isinstance(turn_index, int)
            or isinstance(turn_index, bool)
        ):
            self._corr_no_id += 1
            return
        if not index.has_session(conversation_id):
            self._corr_missing_session += 1
            return
        if not index.has_turn(conversation_id, turn_index):
            self._corr_turn_out_of_range += 1
            return
        self._corr_matched += 1

    def correlation_summary(self, inputs_index: AIPerfInputsIndex) -> AIPerfCorrelationSummary:
        """Build a typed correlation summary from the counts accumulated during iteration."""
        limitations: list[str] = []
        if self._corr_missing_session > 0:
            limitations.append(
                f"{self._corr_missing_session} requests had no matching session in inputs.json."
            )
        if self._corr_turn_out_of_range > 0:
            limitations.append(
                f"{self._corr_turn_out_of_range} requests had a turn_index outside the "
                "inputs.json payload range."
            )
        if self._corr_no_id > 0:
            limitations.append(
                f"{self._corr_no_id} requests lacked conversation_id and could not be correlated."
            )
        return AIPerfCorrelationSummary(
            inputs_session_count=inputs_index.session_count,
            matched_count=self._corr_matched,
            missing_session_count=self._corr_missing_session,
            turn_out_of_range_count=self._corr_turn_out_of_range,
            no_correlation_id_count=self._corr_no_id,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _metric(metrics: Any, name: str) -> tuple[float, str] | None:
        if not isinstance(metrics, dict):
            return None
        item = metrics.get(name)
        if not isinstance(item, dict):
            return None
        value, unit = item.get("value"), item.get("unit")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isinstance(unit, str)
        ):
            return None
        if not math.isfinite(float(value)):
            return None
        return float(value), unit

    @staticmethod
    def _duration_ns(metric: tuple[float, str] | None) -> int | None:
        if metric is None:
            return None
        value, unit = metric
        factors = {"ns": 1.0, "us": 1_000.0, "µs": 1_000.0, "ms": 1_000_000.0, "s": 1e9}
        factor = factors.get(unit)
        return round(value * factor) if factor is not None and value >= 0 else None

    @classmethod
    def _normalize(cls, payload: dict[str, Any], line_index: int) -> AIPerfRequestRow:
        metadata, metrics = payload.get("metadata"), payload.get("metrics")
        if not isinstance(metadata, dict) or not isinstance(metrics, dict):
            raise ValueError("metadata and metrics must be objects")

        def integer(name: str, *, required: bool = False) -> int | None:
            value = metadata.get(name)
            if value is None and not required:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"metadata.{name} must be a non-negative integer")
            return value

        session_num = integer("session_num", required=True)
        request_start_ns = integer("request_start_ns", required=True)
        assert session_num is not None
        assert request_start_ns is not None
        input_metric = cls._metric(metrics, "input_sequence_length")
        output_metric = cls._metric(metrics, "output_sequence_length")
        if input_metric is None or output_metric is None:
            raise ValueError("token count metrics are required")
        input_tokens, output_tokens = round(input_metric[0]), round(output_metric[0])
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        conversation_id = metadata.get("conversation_id")
        turn_index = integer("turn_index")
        source_request_id = (
            f"{conversation_id}:{turn_index}"
            if isinstance(conversation_id, str) and turn_index is not None
            else str(session_num)
        )
        error = payload.get("error")
        if error is not None and not isinstance(error, dict):
            raise ValueError("error must be an object or null")
        cancelled = metadata.get("was_cancelled", False)
        if not isinstance(cancelled, bool):
            raise ValueError("metadata.was_cancelled must be a boolean")
        latency_ns = cls._duration_ns(cls._metric(metrics, "request_latency"))
        ttft_ns = cls._duration_ns(cls._metric(metrics, "time_to_first_token"))
        tpot_ns = (
            round((latency_ns - ttft_ns) / (output_tokens - 1))
            if latency_ns is not None
            and ttft_ns is not None
            and latency_ns >= ttft_ns
            and output_tokens > 1
            else None
        )
        error_type = _safe_error_category(error.get("type")) if isinstance(error, dict) else None
        error_code = _safe_error_code(error.get("code")) if isinstance(error, dict) else None
        outcome: ReportedInferenceRequestOutcome
        if cancelled:
            outcome = CancelledInferenceRequestOutcome(
                error_type=error_type,
                error_code=error_code,
            )
        elif error is not None:
            outcome = FailedInferenceRequestOutcome(
                error_type=error_type,
                error_code=error_code,
            )
        else:
            outcome = SucceededInferenceRequestOutcome()
        return AIPerfRequestRow(
            source_request_id=source_request_id,
            provider_request_id=(
                metadata.get("x_request_id")
                if isinstance(metadata.get("x_request_id"), str)
                else None
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            scheduled_ns=integer("credit_issued_ns"),
            observed_started_ns=request_start_ns,
            ttft_ns=ttft_ns,
            latency_ns=latency_ns,
            tpot_ns=tpot_ns,
            mean_itl_ns=cls._duration_ns(cls._metric(metrics, "inter_token_latency")),
            outcome=outcome,
            line_index=line_index,
        )


# ---------------------------------------------------------------------------
# AIPerf inputs.json correlation
# ---------------------------------------------------------------------------
#
# AIPerf's ``inputs.json`` is the complete input dataset with formatted payloads
# for each request (see ai-dynamo/aiperf ``working-with-profile-export-files``).
# Its structure is::
#
#   {"data": [{"session_id": "<uuid>", "payloads": [<turn-0>, <turn-1>, ...]}]}
#
# Each ``profile_export.jsonl`` record carries ``metadata.conversation_id`` and
# ``metadata.turn_index`` that map to ``session_id`` and the ``payloads`` array
# index. The correlation index retains only ``session_id -> turn_count`` — never
# prompt text, tool definitions, or request bodies.
#
# Parsing uses low-level ``ijson.parse`` events so that no session dict or
# payload object is ever materialized. The event stream is processed in a single
# pass: ``session_id`` is captured from ``data.item.session_id`` string events,
# turn counts are derived by counting ``data.item.payloads.item`` start events,
# and all nested payload content events (prompts, tool definitions, etc.) are
# discarded without being built into Python objects. Peak memory is bounded by
# the fixed ijson chunk buffer plus the largest single string value that ijson
# accumulates, not by the total document or session size.


class AIPerfInputsIndex:
    """Bounded correlation index for AIPerf ``inputs.json``, retaining no payloads.

    The index maps ``session_id`` to the number of turns (payloads) declared for
    that session. The file is parsed via low-level ``ijson.parse`` events so no
    session dict or payload object is ever materialized; only ``session_id``
    strings and turn counts are retained.
    """

    max_input_bytes = 256 * 1024 * 1024
    max_sessions = 100_000
    max_turns_per_session = 10_000
    max_nesting_depth = 64
    max_session_id_length = 256
    stream_buffer_bytes = 65_536

    def __init__(self, session_turn_counts: dict[str, int]) -> None:
        self.session_turn_counts = dict(session_turn_counts)

    @classmethod
    def from_path(cls, path: Path) -> AIPerfInputsIndex:
        """Stream ``inputs.json`` via ``ijson.parse`` events and build a correlation index.

        Processes the binary file as a stream of ``(prefix, event, value)``
        tuples without materializing any session dict or payload object.
        ``session_id`` is captured from ``data.item.session_id`` string events;
        turn counts are derived by counting ``data.item.payloads.item`` start
        events. All nested payload content is discarded. Bounds: file size,
        nesting depth, session ID length, session count, turns per session.
        """
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The AIPerf inputs artifact is not accessible.",
            ) from exc
        if size > cls.max_input_bytes:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"The AIPerf inputs artifact exceeds the {cls.max_input_bytes}-byte limit.",
            )
        index: dict[str, int] = {}
        try:
            with path.open("rb") as stream:
                saw_data_array = cls._stream_events(stream, index)
                if not saw_data_array:
                    raise DomainError(
                        ErrorCode.ARTIFACT_PARSE_FAILED,
                        "The AIPerf inputs artifact must be an object with a data array.",
                    )
        except (OSError, UnicodeDecodeError, IncompleteJSONError, JSONError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The AIPerf inputs artifact is not a valid JSON document.",
            ) from exc
        return cls(index)

    @classmethod
    def _stream_events(cls, stream: Any, index: dict[str, int]) -> bool:
        """Process ``ijson.parse`` events in a single pass, populating ``index``.

        Returns ``True`` if a top-level ``data`` array was seen. Raises
        :class:`DomainError` on any structural violation or bound breach.
        """
        parser = ijson.parse(stream, buf_size=cls.stream_buffer_bytes)
        depth = 0
        saw_data_array = False
        in_session = False
        current_session_id: str | None = None
        saw_session_id = False
        saw_payloads = False
        in_payloads = False
        current_turn_count = 0
        for prefix, event, value in parser:
            depth = cls._track_depth(event, depth)
            saw_data_array = cls._track_top_level(prefix, event, saw_data_array)
            cls._reject_non_object_data_item(prefix, event)
            if prefix == "data.item" and event == "start_map":
                cls._check_session_limit(index)
                in_session = True
                current_session_id = None
                saw_session_id = False
                saw_payloads = False
                current_turn_count = 0
            current_session_id, saw_session_id = cls._capture_session_id(
                prefix, event, value, current_session_id, saw_session_id
            )
            saw_payloads, in_payloads = cls._track_payloads(
                prefix, event, saw_payloads, in_payloads
            )
            current_turn_count = cls._count_payload_item(
                prefix, event, in_payloads, current_turn_count, current_session_id
            )
            if prefix == "data.item" and event == "end_map" and in_session:
                cls._store_session(
                    index,
                    current_session_id,
                    saw_session_id,
                    saw_payloads,
                    current_turn_count,
                )
                in_session = False
        return saw_data_array

    @classmethod
    def _track_depth(cls, event: str, depth: int) -> int:
        if event in ("start_map", "start_array"):
            depth += 1
            if depth > cls.max_nesting_depth:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"The AIPerf inputs artifact exceeds the "
                    f"{cls.max_nesting_depth}-depth nesting limit.",
                )
        elif event in ("end_map", "end_array"):
            depth -= 1
        return depth

    @classmethod
    def _track_top_level(cls, prefix: str, event: str, saw_data_array: bool) -> bool:
        if prefix == "data" and event == "start_array":
            return True
        return saw_data_array

    @classmethod
    def _reject_non_object_data_item(cls, prefix: str, event: str) -> None:
        if prefix == "data.item" and event not in ("start_map", "end_map", "map_key"):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Each AIPerf inputs data entry must be an object.",
            )

    @classmethod
    def _check_session_limit(cls, index: dict[str, int]) -> None:
        if len(index) >= cls.max_sessions:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"The AIPerf inputs artifact exceeds the {cls.max_sessions}-session limit.",
            )

    @classmethod
    def _capture_session_id(
        cls,
        prefix: str,
        event: str,
        value: Any,
        current_session_id: str | None,
        saw_session_id: bool,
    ) -> tuple[str | None, bool]:
        if prefix != "data.item.session_id":
            return current_session_id, saw_session_id
        if saw_session_id:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Each AIPerf inputs data entry must have exactly one session_id.",
            )
        if event != "string":
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Each AIPerf inputs data entry must have a string session_id.",
            )
        if not isinstance(value, str) or not value:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Each AIPerf inputs data entry must have a non-empty session_id.",
            )
        if len(value) > cls.max_session_id_length:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"Session_id length exceeds the {cls.max_session_id_length}-character limit.",
            )
        return value, True

    @classmethod
    def _track_payloads(
        cls, prefix: str, event: str, saw_payloads: bool, in_payloads: bool
    ) -> tuple[bool, bool]:
        if prefix == "data.item.payloads":
            if event not in ("start_array", "end_array"):
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "Each AIPerf inputs data entry must have a payloads array.",
                )
            if event == "start_array":
                return True, True
            return saw_payloads, False
        return saw_payloads, in_payloads

    @classmethod
    def _count_payload_item(
        cls,
        prefix: str,
        event: str,
        in_payloads: bool,
        current_turn_count: int,
        current_session_id: str | None,
    ) -> int:
        if not (
            in_payloads
            and prefix == "data.item.payloads.item"
            and event not in ("end_map", "end_array", "map_key")
        ):
            return current_turn_count
        current_turn_count += 1
        if current_turn_count > cls.max_turns_per_session:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"Session {current_session_id!r} has more than {cls.max_turns_per_session} turns.",
            )
        return current_turn_count

    @classmethod
    def _store_session(
        cls,
        index: dict[str, int],
        current_session_id: str | None,
        saw_session_id: bool,
        saw_payloads: bool,
        current_turn_count: int,
    ) -> None:
        if not saw_session_id or current_session_id is None:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Each AIPerf inputs data entry must have a non-empty session_id.",
            )
        if not saw_payloads:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Each AIPerf inputs data entry must have a payloads array.",
            )
        if current_session_id in index:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"Duplicate session_id {current_session_id!r} in AIPerf inputs.",
            )
        index[current_session_id] = current_turn_count

    def has_session(self, session_id: str) -> bool:
        return session_id in self.session_turn_counts

    def has_turn(self, session_id: str, turn_index: int) -> bool:
        count = self.session_turn_counts.get(session_id)
        return count is not None and 0 <= turn_index < count

    @property
    def session_count(self) -> int:
        return len(self.session_turn_counts)


class AIPerfCorrelationSummary(ContractModel):
    """Typed summary of correlating ``profile_export`` records against ``inputs.json``."""

    schema_version: Literal[1] = 1
    inputs_session_count: int
    matched_count: int
    missing_session_count: int
    turn_out_of_range_count: int
    no_correlation_id_count: int
    limitations: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# vLLM aggregate benchmark-result JSON normalization
# ---------------------------------------------------------------------------
#
def _percentile_label(percentile: int | float) -> str:
    """Return a canonical label for a percentile rank, preserving fractional precision.

    ``int()`` truncation made p99.1 and p99.9 both produce ``p99``, losing
    the exact percentile identity in the metric name.
    """
    if float(percentile).is_integer():
        return str(int(percentile))
    return str(float(percentile))


# vLLM's ``benchmark_serving.BenchmarkMetrics`` dataclass is serialized by the
# Mooncake replayer (and other vLLM benchmark scripts) as a JSON object whose
# percentile fields are lists of ``[percentile, value_ms]`` pairs. The parser
# normalizes the aggregate metrics into bounded, typed measurement rows without
# preserving raw prompt text, error strings, or request payloads.

_VllmPercentile = tuple[StrictInt | StrictFloat, StrictInt | StrictFloat]


class VllmAggregateMetrics(ContractModel):
    """The ``BenchmarkMetrics`` dataclass shape serialized by vLLM scripts."""

    model_config = ConfigDict(extra="ignore")

    completed: Annotated[int, Field(ge=0)]
    total_input: Annotated[int, Field(ge=0)]
    total_output: Annotated[int, Field(ge=0)]
    request_throughput: Annotated[float, Field(ge=0)]
    request_goodput: Annotated[float, Field(ge=0)] | None = None
    output_throughput: Annotated[float, Field(ge=0)]
    total_token_throughput: Annotated[float, Field(ge=0)]
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: tuple[_VllmPercentile, ...] = ()
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: tuple[_VllmPercentile, ...] = ()
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: tuple[_VllmPercentile, ...] = ()
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: tuple[_VllmPercentile, ...] = ()

    @field_validator(
        "request_throughput",
        "request_goodput",
        "output_throughput",
        "total_token_throughput",
        mode="before",
    )
    @classmethod
    def finite_throughput(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("throughput metrics must be JSON numbers")
        if not math.isfinite(float(value)):
            raise ValueError("throughput metrics must be finite")
        return value

    @field_validator(
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
        "std_e2el_ms",
    )
    @classmethod
    def finite_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency metrics must be finite")
        return value

    @model_validator(mode="after")
    def non_negative_latency(self) -> VllmAggregateMetrics:
        for name in (
            "mean_ttft_ms",
            "median_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "mean_e2el_ms",
            "median_e2el_ms",
            "std_ttft_ms",
            "std_tpot_ms",
            "std_itl_ms",
            "std_e2el_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        return self

    @field_validator(
        "percentiles_ttft_ms",
        "percentiles_tpot_ms",
        "percentiles_itl_ms",
        "percentiles_e2el_ms",
    )
    @classmethod
    def valid_percentiles(cls, values: tuple[_VllmPercentile, ...]) -> tuple[_VllmPercentile, ...]:
        for percentile, latency_ms in values:
            if not 0 <= float(percentile) <= 100 or not math.isfinite(float(percentile)):
                raise ValueError("percentile ranks must be finite values from 0 through 100")
            if latency_ms < 0 or not math.isfinite(float(latency_ms)):
                raise ValueError("percentile latency values must be finite and non-negative")
        return values


class VllmResultDocument(ContractModel):
    """The wrapper emitted by the Mooncake replayer around vLLM metrics.

    The replayer stores ``{"metrics": <BenchmarkMetrics.asdict>, ...counts}``.
    Only the bounded aggregate metrics are normalized; raw request payloads,
    error text, and server endpoints are deliberately dropped.
    """

    model_config = ConfigDict(extra="ignore")

    metrics: VllmAggregateMetrics
    successful_requests: Annotated[int, Field(ge=0)]
    failed_requests: Annotated[int, Field(ge=0)]
    total_requests: Annotated[int, Field(ge=0)]
    actual_duration: Annotated[float, Field(ge=0)]
    time_scale: Annotated[float, Field(gt=0)] = 1.0

    @field_validator("actual_duration", "time_scale")
    @classmethod
    def finite_duration_and_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("duration and time_scale must be finite")
        return value

    @model_validator(mode="after")
    def totals_match(self) -> VllmResultDocument:
        if self.successful_requests + self.failed_requests != self.total_requests:
            raise ValueError("successful plus failed requests must equal total requests")
        if self.successful_requests != self.metrics.completed:
            raise ValueError("successful_requests must equal metrics.completed")
        return self


class VllmMeasurementRow(ContractModel):
    """One normalized measurement derived from a vLLM aggregate result."""

    schema_version: Literal[1] = 1
    measurement_id: str
    name: str
    value_float: float
    unit: str
    aggregation: str
    dimensions: dict[str, str]
    evidence_level: EvidenceLevel = EvidenceLevel.DERIVED


class VllmResultParser:
    """Validate a vLLM aggregate benchmark-result JSON document and normalize it.

    The parser reads the document once, validates it against the bounded
    :class:`VllmResultDocument` schema, and produces typed measurement rows.
    Raw prompt text, error strings, and server endpoints are never preserved.
    """

    max_document_bytes = 16 * 1024 * 1024

    def parse(self, path: Path) -> tuple[VllmResultDocument, list[VllmMeasurementRow]]:
        document = self._load(path)
        return document, self._measurement_rows(document)

    def parse_payload(
        self, payload: dict[str, Any]
    ) -> tuple[VllmResultDocument, list[VllmMeasurementRow]]:
        try:
            document = VllmResultDocument.model_validate(self._normalize_document(payload))
        except (ValidationError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The vLLM benchmark-result document violates the normalized schema.",
            ) from exc
        return document, self._measurement_rows(document)

    @staticmethod
    def _load(path: Path) -> VllmResultDocument:
        try:
            if path.stat().st_size > VllmResultParser.max_document_bytes:
                raise ValueError("vLLM benchmark result exceeds the document-size limit")
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a valid vLLM benchmark-result JSON document.",
            ) from exc
        try:
            if not isinstance(payload, dict):
                raise ValueError("vLLM benchmark result must be a JSON object")
            return VllmResultDocument.model_validate(VllmResultParser._normalize_document(payload))
        except (ValidationError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The vLLM benchmark-result document violates the normalized schema.",
            ) from exc

    @staticmethod
    def _normalize_document(payload: dict[str, Any]) -> dict[str, Any]:
        if "metrics" in payload:
            return payload
        completed = payload.get("completed")
        total_requests = payload.get("num_prompts", completed)
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or not isinstance(total_requests, int)
            or isinstance(total_requests, bool)
        ):
            return payload
        metrics = dict(payload)
        if "total_input" not in metrics and "total_input_tokens" in metrics:
            metrics["total_input"] = metrics["total_input_tokens"]
        if "total_output" not in metrics and "total_output_tokens" in metrics:
            metrics["total_output"] = metrics["total_output_tokens"]
        return {
            "metrics": metrics,
            "successful_requests": completed,
            "failed_requests": max(0, total_requests - completed),
            "total_requests": total_requests,
            "actual_duration": payload["duration"],
            "time_scale": 1.0,
        }

    @staticmethod
    def _measurement_rows(document: VllmResultDocument) -> list[VllmMeasurementRow]:
        metrics = document.metrics
        rows: list[VllmMeasurementRow] = []
        base_dimensions = {
            "producer": "vllm.benchmark_serving",
            "time_scale": str(document.time_scale),
            "successful_requests": str(document.successful_requests),
        }

        def add(
            name: str,
            value: float,
            *,
            unit: str,
            aggregation: str,
            extra: dict[str, str] | None = None,
        ) -> None:
            dimensions = dict(base_dimensions)
            if extra:
                dimensions.update(extra)
            identity = {
                "name": name,
                "aggregation": aggregation,
                "value": value,
                "unit": unit,
            }
            rows.append(
                VllmMeasurementRow(
                    measurement_id=digest_model(identity),
                    name=name,
                    value_float=value,
                    unit=unit,
                    aggregation=aggregation,
                    dimensions=dimensions,
                )
            )

        add(
            "vllm.request_throughput",
            metrics.request_throughput,
            unit="requests/sec",
            aggregation="aggregate",
        )
        if metrics.request_goodput is not None:
            add(
                "vllm.request_goodput",
                metrics.request_goodput,
                unit="requests/sec",
                aggregation="aggregate",
            )
        add(
            "vllm.output_throughput",
            metrics.output_throughput,
            unit="tokens/sec",
            aggregation="aggregate",
        )
        add(
            "vllm.total_token_throughput",
            metrics.total_token_throughput,
            unit="tokens/sec",
            aggregation="aggregate",
        )
        add(
            "vllm.total_input_tokens",
            float(metrics.total_input),
            unit="count",
            aggregation="aggregate",
        )
        add(
            "vllm.total_output_tokens",
            float(metrics.total_output),
            unit="count",
            aggregation="aggregate",
        )
        add(
            "vllm.completed_requests",
            float(document.successful_requests),
            unit="requests",
            aggregation="count",
        )
        add(
            "vllm.failed_requests",
            float(document.failed_requests),
            unit="requests",
            aggregation="count",
        )
        add(
            "vllm.total_requests",
            float(document.total_requests),
            unit="requests",
            aggregation="count",
        )
        add(
            "vllm.duration_seconds",
            document.actual_duration,
            unit="s",
            aggregation="aggregate",
        )
        for metric, label in (
            ("ttft", "time_to_first_token"),
            ("tpot", "time_per_output_token"),
            ("itl", "inter_token_latency"),
            ("e2el", "end_to_end_latency"),
        ):
            mean = getattr(metrics, f"mean_{metric}_ms")
            median = getattr(metrics, f"median_{metric}_ms")
            std = getattr(metrics, f"std_{metric}_ms")
            add(
                f"vllm.{label}.mean_ms", mean, unit="ms", aggregation="mean", extra={"stat": "mean"}
            )
            add(
                f"vllm.{label}.median_ms",
                median,
                unit="ms",
                aggregation="median",
                extra={"stat": "median"},
            )
            add(f"vllm.{label}.std_ms", std, unit="ms", aggregation="std", extra={"stat": "std"})
            for percentile, value in getattr(metrics, f"percentiles_{metric}_ms"):
                add(
                    f"vllm.{label}.p{_percentile_label(percentile)}_ms",
                    float(value),
                    unit="ms",
                    aggregation="percentile",
                    extra={"stat": "percentile", "percentile": str(float(percentile))},
                )
        return rows


class SglangResultDocument(ContractModel):
    """Safe aggregate subset of SGLang 0.5.16 ``bench_serving`` JSONL output."""

    model_config = ConfigDict(extra="ignore")

    duration: Annotated[float, Field(ge=0)]
    completed: Annotated[int, Field(ge=0)]
    total_input_tokens: Annotated[int, Field(ge=0)]
    total_output_tokens: Annotated[int, Field(ge=0)]
    num_prompts: Annotated[int, Field(gt=0)] | None = None
    request_throughput: Annotated[float, Field(ge=0)] | None = None
    input_throughput: Annotated[float, Field(ge=0)] | None = None
    output_throughput: Annotated[float, Field(ge=0)] | None = None
    total_token_throughput: Annotated[float, Field(ge=0)] | None = None
    accept_length: Annotated[float, Field(ge=0)] | None = None
    concurrency: Annotated[float, Field(ge=0)] | None = None
    mean_e2e_latency_ms: Annotated[float, Field(ge=0)] | None = None
    median_e2e_latency_ms: Annotated[float, Field(ge=0)] | None = None
    std_e2e_latency_ms: Annotated[float, Field(ge=0)] | None = None
    p90_e2e_latency_ms: Annotated[float, Field(ge=0)] | None = None
    p95_e2e_latency_ms: Annotated[float, Field(ge=0)] | None = None
    p99_e2e_latency_ms: Annotated[float, Field(ge=0)] | None = None
    mean_ttft_ms: Annotated[float, Field(ge=0)] | None = None
    median_ttft_ms: Annotated[float, Field(ge=0)] | None = None
    std_ttft_ms: Annotated[float, Field(ge=0)] | None = None
    p90_ttft_ms: Annotated[float, Field(ge=0)] | None = None
    p95_ttft_ms: Annotated[float, Field(ge=0)] | None = None
    p99_ttft_ms: Annotated[float, Field(ge=0)] | None = None
    mean_tpot_ms: Annotated[float, Field(ge=0)] | None = None
    median_tpot_ms: Annotated[float, Field(ge=0)] | None = None
    std_tpot_ms: Annotated[float, Field(ge=0)] | None = None
    p90_tpot_ms: Annotated[float, Field(ge=0)] | None = None
    p95_tpot_ms: Annotated[float, Field(ge=0)] | None = None
    p99_tpot_ms: Annotated[float, Field(ge=0)] | None = None
    mean_itl_ms: Annotated[float, Field(ge=0)] | None = None
    median_itl_ms: Annotated[float, Field(ge=0)] | None = None
    std_itl_ms: Annotated[float, Field(ge=0)] | None = None
    p90_itl_ms: Annotated[float, Field(ge=0)] | None = None
    p95_itl_ms: Annotated[float, Field(ge=0)] | None = None
    p99_itl_ms: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_sensitive_details(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("SGLang result must be a JSON object")
        sensitive = {"input_lens", "output_lens", "ttfts", "itls", "generated_texts", "errors"}
        if sensitive & value.keys() or any(isinstance(item, list) for item in value.values()):
            raise ValueError("SGLang detailed output is not accepted")
        return value

    @field_validator("*")
    @classmethod
    def finite_numbers(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("SGLang metrics must be JSON numbers")
        if not math.isfinite(float(value)):
            raise ValueError("SGLang metrics must be finite")
        return value

    @model_validator(mode="after")
    def counts_are_consistent(self) -> SglangResultDocument:
        if self.num_prompts is not None and self.completed > self.num_prompts:
            raise ValueError("SGLang completed count exceeds num_prompts")
        return self


class SglangResultParser:
    """Parse exactly one bounded, aggregate-only SGLang 0.5.16 JSONL record.

    SGLang's optional detailed output carries prompts, generations and error
    text.  It is intentionally rejected before any normalization occurs.
    """

    max_document_bytes = 1024 * 1024
    _measurement_fields = tuple(
        name for name in SglangResultDocument.model_fields if name not in {"num_prompts"}
    )

    def parse(self, path: Path) -> tuple[SglangResultDocument, list[VllmMeasurementRow]]:
        try:
            if path.stat().st_size > self.max_document_bytes:
                raise ValueError("SGLang result exceeds size bound")
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError("SGLang output must contain exactly one non-empty JSONL record")
            payload = json.loads(lines[0])
            document = SglangResultDocument.model_validate(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a valid aggregate SGLang JSONL result.",
            ) from exc
        rows: list[VllmMeasurementRow] = []
        for key in self._measurement_fields:
            value = getattr(document, key)
            if value is None:
                continue
            value = float(value)
            unit = (
                "ms"
                if key.endswith("_ms")
                else "tokens/sec"
                if key.endswith("throughput")
                else "s"
                if key == "duration"
                else "count"
            )
            dimensions = {
                "producer": "sglang.bench_serving",
                "completed": str(document.completed),
            }
            aggregation = "aggregate"
            if key.startswith("mean_"):
                aggregation = "mean"
                dimensions["stat"] = "mean"
            elif key.startswith("median_"):
                aggregation = "median"
                dimensions["stat"] = "median"
            elif key.startswith("std_"):
                aggregation = "std"
                dimensions["stat"] = "std"
            elif key.startswith("p") and "_" in key:
                percentile = key.split("_", 1)[0][1:]
                if percentile.isdigit():
                    aggregation = "percentile"
                    dimensions.update({"stat": "percentile", "percentile": percentile})
            rows.append(
                VllmMeasurementRow(
                    measurement_id=digest_model({"name": f"sglang.{key}", "value": value}),
                    name=f"sglang.{key}",
                    value_float=value,
                    unit=unit,
                    aggregation=aggregation,
                    dimensions=dimensions,
                )
            )
        return document, rows


class InferenceExtractionResult(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    artifact_id: str
    evidence_run_id: str
    inputs_artifact_id: str | None = None
    request_count: int = 0
    measurement_count: int = 0
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class InferenceArtifactExtractor:
    """Publish bounded inference trace rows and vLLM aggregate measurements."""

    name = "flameox.inference"
    version = "2"
    max_request_rows = 100_000

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    def _request_row_limit(self) -> int:
        """Bound materialized inference rows independently of storage quotas."""
        return min(
            self.max_request_rows,
            self.workspace.config.storage.max_rows_per_generation,
        )

    def extract_trace(
        self,
        run_id: str,
        *,
        evidence_run_id: str | None = None,
    ) -> InferenceExtractionResult:
        """Extract Mooncake trace evidence, optionally under a target evidence run.

        ``evidence_run_id`` defaults to ``run_id`` (the import run), preserving
        the original behavior. When set to a distinct canonical run, evidence
        rows are published under that run while provenance retains both the
        artifact run and artifact id.
        """
        registration = self._registration(run_id, ArtifactKind.INFERENCE_REQUEST_TRACE)
        stored = self.artifacts.get(registration.artifact_id)
        target_run = evidence_run_id or run_id
        parser = MooncakeTraceParser(max_rows=self._request_row_limit())
        summary, requests = parser.parse(stored.payload_path)
        rows = [
            InferenceRequestItem.model_validate(
                {
                    "request_id": row.request_id,
                    "run_id": target_run,
                    "artifact_id": registration.artifact_id,
                    "source_request_id": str(row.line_index),
                    "provider_request_id": None,
                    "input_tokens": row.input_length,
                    "output_tokens": row.output_length,
                    "scheduled_ns": row.timestamp_ms * 1_000_000,
                    "observed_started_ns": None,
                    "ttft_ns": None,
                    "latency_ns": None,
                    "tpot_ns": None,
                    "mean_itl_ns": None,
                    "success": None,
                    "cancelled": None,
                    "error_type": None,
                    "error_code": None,
                    "queue_ns": None,
                    "prefill_ns": None,
                    "decode_ns": None,
                    "cache_hit": None,
                    "prefix_hash_count": row.prefix_hash_count,
                    "evidence_level": "observed",
                }
            ).model_dump(mode="python")
            for row in requests
        ]
        published = self.publisher.publish_rows_idempotent(
            {"inference_requests": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=tuple(dict.fromkeys((run_id, target_run))),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "kind": "mooncake",
                "parser_version": self.version,
                "evidence_run_id": target_run,
            },
        )
        return InferenceExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            evidence_run_id=target_run,
            request_count=len(rows),
            corpus_commit_id=published.commit.commit_id,
            limitations=summary.limitations,
        )

    def extract_vllm_result(
        self,
        run_id: str,
        *,
        evidence_run_id: str | None = None,
    ) -> InferenceExtractionResult:
        """Extract vLLM aggregate measurements, optionally under a target evidence run."""
        registration = self._registration(run_id, ArtifactKind.INFERENCE_RESULT)
        stored = self.artifacts.get(registration.artifact_id)
        target_run = evidence_run_id or run_id
        _document, measurements = VllmResultParser().parse(stored.payload_path)
        rows = [
            {
                "measurement_id": digest_model(
                    {
                        "run_id": target_run,
                        "artifact_id": registration.artifact_id,
                        "row": row.model_dump(mode="json"),
                    }
                ),
                "run_id": target_run,
                "artifact_id": registration.artifact_id,
                "name": row.name,
                "value_int": None,
                "value_float": row.value_float,
                "unit": row.unit,
                "aggregation": row.aggregation,
                "scope": "workload",
                "trial_id": None,
                "worker_id": None,
                "worker_run_index": None,
                "value_index": None,
                "loop_count": None,
                "is_warmup": False,
                "block_id": None,
                "variant_id": None,
                "order_in_block": None,
                "phase": "steady_state",
                "dimensions": row.dimensions,
                "evidence_level": row.evidence_level,
            }
            for row in measurements
        ]
        published = self.publisher.publish_rows_idempotent(
            {"measurements": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=tuple(dict.fromkeys((run_id, target_run))),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "kind": "vllm_bench",
                "parser_version": self.version,
                "evidence_run_id": target_run,
            },
        )
        return InferenceExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            evidence_run_id=target_run,
            measurement_count=len(rows),
            corpus_commit_id=published.commit.commit_id,
        )

    def extract_sglang_result(
        self, run_id: str, *, evidence_run_id: str | None = None
    ) -> InferenceExtractionResult:
        """Publish scalar-only measurements from a preserved SGLang JSONL result."""
        registration = self._registration(run_id, ArtifactKind.INFERENCE_RESULT)
        stored = self.artifacts.get(registration.artifact_id)
        target_run = evidence_run_id or run_id
        _document, measurements = SglangResultParser().parse(stored.payload_path)
        rows = [
            {
                "measurement_id": digest_model(
                    {
                        "run_id": target_run,
                        "artifact_id": registration.artifact_id,
                        "row": row.model_dump(mode="json"),
                    }
                ),
                "run_id": target_run,
                "artifact_id": registration.artifact_id,
                "name": row.name,
                "value_int": None,
                "value_float": row.value_float,
                "unit": row.unit,
                "aggregation": row.aggregation,
                "scope": "workload",
                "trial_id": None,
                "worker_id": None,
                "worker_run_index": None,
                "value_index": None,
                "loop_count": None,
                "is_warmup": False,
                "block_id": None,
                "variant_id": None,
                "order_in_block": None,
                "phase": "steady_state",
                "dimensions": row.dimensions,
                "evidence_level": row.evidence_level,
            }
            for row in measurements
        ]
        published = self.publisher.publish_rows_idempotent(
            {"measurements": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=tuple(dict.fromkeys((run_id, target_run))),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "kind": "sglang_bench_serving",
                "parser_version": self.version,
                "evidence_run_id": target_run,
            },
        )
        return InferenceExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            evidence_run_id=target_run,
            measurement_count=len(rows),
            corpus_commit_id=published.commit.commit_id,
        )

    def extract_aiperf_result(
        self,
        run_id: str,
        *,
        evidence_run_id: str | None = None,
        inputs_run_id: str | None = None,
        inputs_artifact_id: str | None = None,
    ) -> InferenceExtractionResult:
        """Normalize AIPerf's published record JSONL with optional inputs.json correlation.

        Evidence rows are published under ``evidence_run_id`` (defaulting to
        ``run_id``). The ``inputs.json`` correlation index is resolved from
        ``inputs_artifact_id`` if provided, otherwise from ``inputs_run_id`` if
        provided, otherwise by auto-detecting an ``INFERENCE_REQUEST_TRACE``
        artifact in the same run as the result artifact. Provenance retains both
        the result run and the inputs run/artifact when correlation is used.
        """
        registration = self._aiperf_result_registration(run_id)
        stored = self.artifacts.get(registration.artifact_id)
        target_run = evidence_run_id or run_id
        inputs_index, inputs_artifact = self._resolve_inputs_index(
            run_id,
            inputs_run_id=inputs_run_id,
            inputs_artifact_id=inputs_artifact_id,
        )
        parser = AIPerfRecordParser(max_rows=self._request_row_limit())
        rows: list[dict[str, Any]] = []
        for record in parser.iter_rows(stored.payload_path, inputs_index=inputs_index):
            line_index = record.line_index
            rows.append(
                InferenceRequestItem.model_validate(
                    {
                        "request_id": digest_model(
                            {
                                "run_id": target_run,
                                "artifact_id": registration.artifact_id,
                                "line_index": line_index,
                                "source_request_id": record.source_request_id,
                            }
                        ),
                        "run_id": target_run,
                        "artifact_id": registration.artifact_id,
                        **record.evidence_columns(),
                    }
                ).model_dump(mode="python")
            )
        if not rows:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The AIPerf profile export contains no request records.",
                run_id=run_id,
            )
        input_run_ids: tuple[str, ...]
        input_artifact_ids: tuple[str, ...]
        result_source_run_id = self._run_id_for_artifact(registration.artifact_id)
        if inputs_index is not None and inputs_artifact is not None:
            inputs_source_run_id = self._run_id_for_artifact(inputs_artifact[1])
            input_run_ids = tuple(
                dict.fromkeys((result_source_run_id, inputs_source_run_id, target_run))
            )
            input_artifact_ids = (registration.artifact_id, inputs_artifact[1])
        else:
            input_run_ids = tuple(dict.fromkeys((result_source_run_id, target_run)))
            input_artifact_ids = (registration.artifact_id,)
        measurement_rows = self._aiperf_measurement_rows(
            rows,
            run_id=target_run,
            artifact_id=registration.artifact_id,
        )
        published = self.publisher.publish_rows_idempotent(
            {"inference_requests": rows, "measurements": measurement_rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=input_run_ids,
            input_artifact_ids=input_artifact_ids,
            operation_identity={
                "kind": "aiperf_0.12",
                "parser_version": self.version,
                "evidence_run_id": target_run,
            },
        )
        limitations: list[str] = []
        if parser.truncated:
            limitations.append(f"AIPerf request evidence truncated at {parser.max_rows} rows.")
        if inputs_index is not None:
            limitations.extend(parser.correlation_summary(inputs_index).limitations)
        return InferenceExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            evidence_run_id=target_run,
            inputs_artifact_id=inputs_artifact[1] if inputs_artifact is not None else None,
            request_count=len(rows),
            measurement_count=len(measurement_rows),
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _aiperf_measurement_rows(
        requests: list[dict[str, Any]],
        *,
        run_id: str,
        artifact_id: str,
    ) -> list[dict[str, Any]]:
        """Derive bounded trial-level aggregates from prompt-free request evidence."""
        successful = [row for row in requests if row["success"] is True]
        metrics: list[tuple[str, float, str, str, dict[str, str]]] = [
            ("aiperf.requests", float(len(requests)), "requests", "count", {}),
            (
                "aiperf.successful_requests",
                float(len(successful)),
                "requests",
                "count",
                {},
            ),
            (
                "aiperf.input_tokens",
                float(sum(int(row["input_tokens"]) for row in successful)),
                "tokens",
                "sum",
                {},
            ),
            (
                "aiperf.output_tokens",
                float(sum(int(row["output_tokens"]) for row in successful)),
                "tokens",
                "sum",
                {},
            ),
        ]
        ttft = sorted(int(row["ttft_ns"]) for row in successful if row["ttft_ns"] is not None)
        latency = sorted(
            int(row["latency_ns"]) for row in successful if row["latency_ns"] is not None
        )
        if ttft:
            metrics.append(
                (
                    "aiperf.ttft.median_ms",
                    float(median(ttft)) / 1_000_000,
                    "ms",
                    "median",
                    {"stat": "median"},
                )
            )
        if latency:
            metrics.append(
                (
                    "aiperf.end_to_end_latency.p95_ms",
                    float(InferenceArtifactExtractor._nearest_rank(latency, 0.95)) / 1_000_000,
                    "ms",
                    "percentile",
                    {"stat": "percentile", "percentile": "95"},
                )
            )
        starts = [
            int(row["observed_started_ns"])
            for row in successful
            if row["observed_started_ns"] is not None and row["latency_ns"] is not None
        ]
        ends = [
            int(row["observed_started_ns"]) + int(row["latency_ns"])
            for row in successful
            if row["observed_started_ns"] is not None and row["latency_ns"] is not None
        ]
        if starts and ends and max(ends) > min(starts):
            metrics.append(
                (
                    "aiperf.request_throughput",
                    len(ends) * 1_000_000_000 / (max(ends) - min(starts)),
                    "requests/s",
                    "rate",
                    {},
                )
            )
        return [
            {
                "measurement_id": digest_model(
                    {
                        "run_id": run_id,
                        "artifact_id": artifact_id,
                        "name": name,
                        "value": value,
                        "unit": unit,
                        "aggregation": aggregation,
                    }
                ),
                "run_id": run_id,
                "artifact_id": artifact_id,
                "name": name,
                "value_int": None,
                "value_float": value,
                "unit": unit,
                "aggregation": aggregation,
                "scope": "workload",
                "trial_id": None,
                "worker_id": None,
                "worker_run_index": None,
                "value_index": None,
                "loop_count": None,
                "is_warmup": False,
                "block_id": None,
                "variant_id": None,
                "order_in_block": None,
                "phase": "steady_state",
                "dimensions": {"producer": "aiperf", **dimensions},
                "evidence_level": "derived",
            }
            for name, value, unit, aggregation, dimensions in metrics
        ]

    @staticmethod
    def _nearest_rank(values: list[int], quantile: float) -> int:
        index = max(0, math.ceil(quantile * len(values)) - 1)
        return values[index]

    def _resolve_inputs_index(
        self,
        run_id: str,
        *,
        inputs_run_id: str | None,
        inputs_artifact_id: str | None,
    ) -> tuple[AIPerfInputsIndex | None, tuple[str, str] | None]:
        """Resolve the inputs.json correlation index and its provenance.

        Returns ``(index, (inputs_run_id, inputs_artifact_id))`` or ``(None, None)``
        when no inputs artifact is available. Resolution order:
        1. ``inputs_artifact_id`` — load directly from the artifact store.
        2. ``inputs_run_id`` — auto-detect the trace artifact in that run.
        3. Same-run auto-detection in ``run_id`` (convenience default).
        """
        if inputs_artifact_id is not None:
            stored = self.artifacts.get(inputs_artifact_id)
            source_run = inputs_run_id or self._run_id_for_artifact(inputs_artifact_id)
            return (
                AIPerfInputsIndex.from_path(stored.payload_path),
                (source_run, inputs_artifact_id),
            )
        if inputs_run_id is not None:
            return self._inputs_index_from_run(inputs_run_id)
        return self._inputs_index_from_run(run_id)

    def _inputs_index_from_run(
        self, run_id: str
    ) -> tuple[AIPerfInputsIndex | None, tuple[str, str] | None]:
        """Auto-detect an ``INFERENCE_REQUEST_TRACE`` artifact in a run."""
        run = self.runs.read(run_id)
        matches = [
            item for item in run.artifacts if item.kind is ArtifactKind.INFERENCE_REQUEST_TRACE
        ]
        if not matches:
            return None, None
        if len(matches) > 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run contains multiple inference_request_trace artifacts; "
                "AIPerf correlation requires at most one inputs.json.",
                run_id=run_id,
            )
        artifact_id = matches[0].artifact_id
        stored = self.artifacts.get(artifact_id)
        return AIPerfInputsIndex.from_path(stored.payload_path), (run_id, artifact_id)

    def _run_id_for_artifact(self, artifact_id: str) -> str:
        """Recover the run_id that owns an artifact by scanning run manifests."""
        fallback: str | None = None
        for run_id in self._list_run_ids():
            run = self.runs.read(run_id)
            if any(item.artifact_id == artifact_id for item in run.artifacts):
                if run.run_type is RunType.IMPORT:
                    return run_id
                fallback = fallback or run_id
        if fallback is not None:
            return fallback
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"No run declares artifact {artifact_id!r}.",
        )

    def _list_run_ids(self) -> list[str]:
        runs_root = self.workspace.paths.runs
        if not runs_root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in runs_root.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    def _registration(self, run_id: str, kind: ArtifactKind) -> Any:
        run = self.runs.read(run_id)
        matches = [item for item in run.artifacts if item.kind is kind]
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"The run must contain exactly one {kind.value} artifact.",
                run_id=run_id,
            )
        return matches[0]

    def _aiperf_result_registration(self, run_id: str) -> Any:
        run = self.runs.read(run_id)
        matches = [item for item in run.artifacts if item.kind is ArtifactKind.INFERENCE_RESULT]
        profile_exports = [item for item in matches if item.display_name == "profile_export.jsonl"]
        if len(profile_exports) == 1:
            return profile_exports[0]
        if len(matches) == 1:
            return matches[0]
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            "The run must identify exactly one AIPerf profile_export.jsonl artifact.",
            run_id=run_id,
        )
