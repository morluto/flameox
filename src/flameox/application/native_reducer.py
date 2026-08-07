from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Literal, cast

from pydantic import Field

from flameox.models import ContractModel

NativePredicateClassification = Literal["interesting", "not_interesting", "unresolved"]


class NativeReductionLimits(ContractModel):
    max_attempts: int = Field(default=1_000, ge=1, le=100_000)
    wall_time_seconds: float = Field(default=900, gt=0, le=86_400)
    repetitions: int = Field(default=1, ge=1, le=20)


class NativeReductionAttempt(ContractModel):
    attempt_id: str
    candidate_digest: str
    candidate_size_bytes: int = Field(ge=0)
    requested_unit_ids: tuple[str, ...] = ()
    dependency_added_unit_ids: tuple[str, ...] = ()
    removed_unit_ids: tuple[str, ...] = ()
    predicate_outcomes: tuple[NativePredicateClassification, ...] = ()
    classification: NativePredicateClassification
    cache_status: Literal["miss", "hit"]
    duration_ms: float = Field(ge=0)
    became_best: bool = False
    failure: str | None = None


class NativeReductionResult(ContractModel):
    disposition: Literal["succeeded", "unchanged", "inconclusive", "original_not_interesting"]
    original_digest: str
    final_digest: str
    original_unit_count: int = Field(ge=0)
    final_unit_count: int = Field(ge=0)
    attempts: tuple[NativeReductionAttempt, ...] = ()
    minimality: Literal["one_minimal", "not_claimed", "partitioner_incompatible"]
    budget_exhausted: bool = False
    final_revalidation: NativePredicateClassification
    limitations: tuple[str, ...] = ()
    # These fields are execution handoff data, not part of the persisted result contract.
    final_payload: bytes | None = Field(default=None, exclude=True)
    accepted_best_payloads: tuple[bytes, ...] = Field(default_factory=tuple, exclude=True)


@dataclass(frozen=True, slots=True)
class _Unit:
    unit_id: str
    payload: bytes
    dependencies: tuple[str, ...] = ()


class NativeDdminReducer:
    """Deterministic sequential ddmin over explicitly supported input formats."""

    def __init__(
        self,
        partitioner: Literal[
            "text_lines",
            "binary_chunks",
            "json_top_level",
            "jsonl_records",
            "otlp_spans",
            "chrome_trace_events",
        ],
        *,
        chunk_size: int | None = None,
        limits: NativeReductionLimits | None = None,
    ) -> None:
        if partitioner == "binary_chunks" and chunk_size is None:
            raise ValueError("binary_chunks requires chunk_size")
        if chunk_size is not None and not 1 <= chunk_size <= 16 * 1024 * 1024:
            raise ValueError("chunk_size must be between 1 and 16777216")
        self.partitioner = partitioner
        self.chunk_size = chunk_size
        self.limits = limits or NativeReductionLimits()

    def reduce(  # noqa: C901 - deterministic ddmin control flow is intentionally explicit
        self,
        original: bytes,
        predicate: Callable[[bytes], NativePredicateClassification],
        *,
        failure_detail: Callable[[], str | None] | None = None,
    ) -> NativeReductionResult:
        original_digest = _digest(original)
        try:
            units, rebuild = self._partition(original)
        except (ImportError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            return NativeReductionResult(
                disposition="inconclusive",
                original_digest=original_digest,
                final_digest=original_digest,
                original_unit_count=0,
                final_unit_count=0,
                minimality="partitioner_incompatible",
                final_revalidation="unresolved",
                limitations=(f"partitioner_incompatible:{error}",),
            )
        cache: dict[str, NativePredicateClassification] = {}
        attempts: list[NativeReductionAttempt] = []
        accepted_best_payloads: list[bytes] = []
        started = time.monotonic()

        def classify(
            payload: bytes, requested: tuple[str, ...], removed: tuple[str, ...]
        ) -> NativePredicateClassification:
            digest = _digest(payload)
            cache_status: Literal["hit", "miss"] = "hit" if digest in cache else "miss"
            candidate_started = time.monotonic()
            failure: str | None = None
            if digest in cache:
                outcome = cache[digest]
                outcomes: tuple[NativePredicateClassification, ...] = (outcome,)
            else:
                outcomes_list_values: list[NativePredicateClassification] = []
                failures: list[str] = []
                for _ in range(self.limits.repetitions):
                    outcomes_list_values.append(predicate(payload))
                    if failure_detail is not None and (detail := failure_detail()) is not None:
                        failures.append(detail)
                outcomes_list = tuple(outcomes_list_values)
                outcome = _collapse(outcomes_list)
                outcomes = outcomes_list
                cache[digest] = outcome
                failure = failures[0] if failures else None
            attempts.append(
                NativeReductionAttempt(
                    attempt_id=f"attempt-{len(attempts):08d}",
                    candidate_digest=digest,
                    candidate_size_bytes=len(payload),
                    requested_unit_ids=requested,
                    removed_unit_ids=removed,
                    predicate_outcomes=outcomes,
                    classification=outcome,
                    cache_status=cache_status,
                    duration_ms=(time.monotonic() - candidate_started) * 1_000,
                    failure=failure,
                )
            )
            return outcome

        initial = classify(original, (), ())
        if initial != "interesting":
            return self._result(
                "inconclusive" if initial == "unresolved" else "original_not_interesting",
                original,
                original,
                units,
                len(units),
                attempts,
                "not_claimed",
                initial,
            )

        full_selection = rebuild(units)
        if full_selection != original:
            normalized = classify(full_selection, ("__full_selection__",), ())
            if normalized != "interesting":
                return self._result(
                    "inconclusive",
                    original,
                    original,
                    units,
                    len(units),
                    attempts,
                    "partitioner_incompatible",
                    normalized,
                    limitations=(
                        "The partitioner's normalized full-selection artifact did not preserve "
                        "the predicate.",
                    ),
                    accepted_best_payloads=accepted_best_payloads,
                )

        current = list(units)
        granularity = 2
        budget_exhausted = False
        while current:
            if (
                len(attempts) >= self.limits.max_attempts
                or time.monotonic() - started >= self.limits.wall_time_seconds
            ):
                budget_exhausted = True
                break
            subsets = _split(current, granularity)
            changed = False
            candidates = (*subsets, *(subsets if len(subsets) > 1 else ()))
            for subset_index, subset in enumerate(candidates):
                if (
                    len(attempts) >= self.limits.max_attempts
                    or time.monotonic() - started >= self.limits.wall_time_seconds
                ):
                    budget_exhausted = True
                    break
                if subset_index < len(subsets):
                    candidate_before_dependencies = [unit for unit in current if unit not in subset]
                else:
                    candidate_before_dependencies = list(subset)
                candidate_units = _with_dependencies(candidate_before_dependencies, current)
                requested = tuple(unit.unit_id for unit in subset)
                dependency_added = tuple(
                    unit.unit_id
                    for unit in candidate_units
                    if unit.unit_id
                    not in {candidate.unit_id for candidate in candidate_before_dependencies}
                )
                if len(candidate_units) >= len(current):
                    candidate_payload = rebuild(candidate_units)
                    candidate_digest = _digest(candidate_payload)
                    attempts.append(
                        NativeReductionAttempt(
                            attempt_id=f"attempt-{len(attempts):08d}",
                            candidate_digest=candidate_digest,
                            candidate_size_bytes=len(candidate_payload),
                            requested_unit_ids=requested,
                            dependency_added_unit_ids=dependency_added,
                            removed_unit_ids=(),
                            classification="not_interesting",
                            cache_status="hit" if candidate_digest in cache else "miss",
                            duration_ms=0,
                            failure="dependency_closure_no_op",
                        )
                    )
                    continue
                payload = rebuild(candidate_units)
                removed = tuple(unit.unit_id for unit in current if unit not in candidate_units)
                outcome = classify(payload, requested, removed)
                attempts[-1] = attempts[-1].model_copy(
                    update={"dependency_added_unit_ids": dependency_added}
                )
                if outcome == "interesting":
                    current = candidate_units
                    accepted_best_payloads.append(payload)
                    attempts[-1] = attempts[-1].model_copy(update={"became_best": True})
                    granularity = max(granularity - 1, 2)
                    changed = True
                    break
                if outcome == "unresolved":
                    return self._result(
                        "inconclusive",
                        original,
                        rebuild(current),
                        units,
                        len(current),
                        attempts,
                        "not_claimed",
                        outcome,
                        budget_exhausted=budget_exhausted,
                        accepted_best_payloads=accepted_best_payloads,
                    )
            if budget_exhausted:
                break
            if not changed:
                if granularity >= len(current):
                    break
                granularity = min(len(current), granularity * 2)
        final = rebuild(current)
        final_classification = _collapse(predicate(final) for _ in range(self.limits.repetitions))
        if final_classification != "interesting":
            return self._result(
                "inconclusive",
                original,
                final,
                units,
                len(current),
                attempts,
                "not_claimed",
                final_classification,
                budget_exhausted=budget_exhausted,
                accepted_best_payloads=accepted_best_payloads,
            )
        disposition: Literal["succeeded", "unchanged"] = (
            "unchanged" if _digest(final) == original_digest else "succeeded"
        )
        return self._result(
            disposition,
            original,
            final,
            units,
            len(current),
            attempts,
            "not_claimed" if budget_exhausted else "one_minimal",
            final_classification,
            budget_exhausted=budget_exhausted,
            accepted_best_payloads=accepted_best_payloads,
        )

    def _result(
        self,
        disposition: Literal["succeeded", "unchanged", "inconclusive", "original_not_interesting"],
        original: bytes,
        final: bytes,
        original_units: list[_Unit],
        final_unit_count: int,
        attempts: list[NativeReductionAttempt],
        minimality: Literal["one_minimal", "not_claimed", "partitioner_incompatible"],
        final_revalidation: NativePredicateClassification,
        *,
        budget_exhausted: bool = False,
        limitations: tuple[str, ...] = (),
        accepted_best_payloads: list[bytes] | None = None,
    ) -> NativeReductionResult:
        return NativeReductionResult(
            disposition=disposition,
            original_digest=_digest(original),
            final_digest=_digest(final),
            original_unit_count=len(original_units),
            final_unit_count=final_unit_count,
            attempts=tuple(attempts),
            minimality=minimality,
            budget_exhausted=budget_exhausted,
            final_revalidation=final_revalidation,
            limitations=limitations,
            final_payload=final,
            accepted_best_payloads=tuple(accepted_best_payloads or ()),
        )

    def _partition(self, original: bytes) -> tuple[list[_Unit], Callable[[list[_Unit]], bytes]]:
        if self.partitioner == "text_lines":
            units = [
                _Unit(f"line-{index:08d}", line)
                for index, line in enumerate(original.splitlines(keepends=True))
            ]
            return units, lambda selected: b"".join(unit.payload for unit in selected)
        if self.partitioner == "binary_chunks":
            assert self.chunk_size is not None
            units = [
                _Unit(f"chunk-{index:08d}", original[offset : offset + self.chunk_size])
                for index, offset in enumerate(range(0, len(original), self.chunk_size))
            ]
            return units, lambda selected: b"".join(unit.payload for unit in selected)
        if self.partitioner == "jsonl_records":
            lines = original.splitlines(keepends=True)
            for line in lines:
                if line.strip():
                    json.loads(line)
            units = [_Unit(f"record-{index:08d}", line) for index, line in enumerate(lines)]
            return units, lambda selected: b"".join(unit.payload for unit in selected)
        if self.partitioner in {"json_top_level", "chrome_trace_events"}:
            value = json.loads(original)
            if self.partitioner == "chrome_trace_events":
                if not isinstance(value, dict) or not isinstance(value.get("traceEvents"), list):
                    raise ValueError("Chrome trace must contain a traceEvents array")
                values = value["traceEvents"]
                begin_by_key: dict[tuple[object, object, object], str] = {}
                dependencies: dict[str, tuple[str, ...]] = {}
                units = [
                    _Unit(f"event-{index:08d}", _json_bytes(item))
                    for index, item in enumerate(values)
                ]
                for index, item in enumerate(values):
                    if not isinstance(item, dict):
                        continue
                    phase = item.get("ph")
                    event_id = item.get("id")
                    if event_id is None:
                        continue
                    key = (item.get("pid"), item.get("tid"), event_id)
                    unit_id = units[index].unit_id
                    if phase in {"b", "S"}:
                        begin_by_key[key] = unit_id
                    elif phase in {"e", "F"} and key in begin_by_key:
                        dependencies[unit_id] = (begin_by_key[key],)
                units = [
                    replace(unit, dependencies=dependencies.get(unit.unit_id, ())) for unit in units
                ]
                return units, lambda selected: _rebuild_chrome(value, selected)
            if isinstance(value, list):
                units = [
                    _Unit(f"item-{index:08d}", _json_bytes(item))
                    for index, item in enumerate(value)
                ]
                return units, lambda selected: _json_bytes(
                    [json.loads(unit.payload) for unit in selected]
                )
            if isinstance(value, dict):
                units = [
                    _Unit(f"member-{key}", _json_bytes({key: item})) for key, item in value.items()
                ]
                return units, lambda selected: _json_bytes(
                    {
                        key: json.loads(unit.payload)[key]
                        for unit in selected
                        for key in json.loads(unit.payload)
                    }
                )
            raise ValueError("JSON top level must be an array or object")
        if self.partitioner == "otlp_spans":
            if original.lstrip().startswith((b"{", b"[")):
                return self._partition_json_otlp(original)
            return self._partition_binary_otlp(original)
        raise ValueError(f"unsupported partitioner {self.partitioner}")

    @staticmethod
    def _partition_json_otlp(original: bytes) -> tuple[list[_Unit], Callable[[list[_Unit]], bytes]]:
        value = json.loads(original)
        resources = value.get("resourceSpans") if isinstance(value, dict) else None
        if not isinstance(resources, list):
            raise ValueError("OTLP JSON must contain resourceSpans")
        units: list[_Unit] = []
        for resource_index, resource in enumerate(resources):
            for scope_index, scope in enumerate(resource.get("scopeSpans", [])):
                for span_index, span in enumerate(scope.get("spans", [])):
                    units.append(
                        _Unit(
                            f"span-{resource_index:04d}-{scope_index:04d}-{span_index:08d}",
                            _json_bytes(span),
                        )
                    )

        def rebuild(selected: list[_Unit]) -> bytes:
            wanted = {unit.unit_id for unit in selected}
            rebuilt = json.loads(original)
            for resource_index, resource in enumerate(rebuilt["resourceSpans"]):
                for scope_index, scope in enumerate(resource.get("scopeSpans", [])):
                    scope["spans"] = [
                        span
                        for span_index, span in enumerate(scope.get("spans", []))
                        if f"span-{resource_index:04d}-{scope_index:04d}-{span_index:08d}" in wanted
                    ]
            return _json_bytes(rebuilt)

        return units, rebuild

    @staticmethod
    def _partition_binary_otlp(
        original: bytes,
    ) -> tuple[list[_Unit], Callable[[list[_Unit]], bytes]]:
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )

        request = ExportTraceServiceRequest()
        request.ParseFromString(original)
        units: list[_Unit] = []
        for resource_index, resource in enumerate(request.resource_spans):
            for scope_index, scope in enumerate(resource.scope_spans):
                for span_index, span in enumerate(scope.spans):
                    units.append(
                        _Unit(
                            f"span-{resource_index:04d}-{scope_index:04d}-{span_index:08d}",
                            span.SerializeToString(deterministic=True),
                        )
                    )

        def rebuild(selected: list[_Unit]) -> bytes:
            wanted = {unit.unit_id for unit in selected}
            rebuilt = ExportTraceServiceRequest()
            rebuilt.CopyFrom(request)
            for resource_index, resource in enumerate(rebuilt.resource_spans):
                for scope_index, scope in enumerate(resource.scope_spans):
                    kept = [
                        span
                        for span_index, span in enumerate(scope.spans)
                        if f"span-{resource_index:04d}-{scope_index:04d}-{span_index:08d}" in wanted
                    ]
                    del scope.spans[:]
                    for span in kept:
                        scope.spans.add().CopyFrom(span)
            return cast(bytes, rebuilt.SerializeToString(deterministic=True))

        return units, rebuild


def _collapse(outcomes: Iterable[NativePredicateClassification]) -> NativePredicateClassification:
    values = tuple(outcomes)
    if not values or "unresolved" in values or len(set(values)) != 1:
        return "unresolved"
    return values[0]


def _split(units: list[_Unit], count: int) -> list[list[_Unit]]:
    size = max(1, (len(units) + count - 1) // count)
    return [units[index : index + size] for index in range(0, len(units), size)]


def _with_dependencies(selected: list[_Unit], universe: list[_Unit]) -> list[_Unit]:
    by_id = {unit.unit_id: unit for unit in universe}
    result = list(selected)
    present = {unit.unit_id for unit in result}
    changed = True
    while changed:
        changed = False
        for unit in tuple(result):
            for dependency in unit.dependencies:
                if dependency not in present and dependency in by_id:
                    result.append(by_id[dependency])
                    present.add(dependency)
                    changed = True
    return [unit for unit in universe if unit.unit_id in present]


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _rebuild_chrome(original: dict[str, object], selected: list[_Unit]) -> bytes:
    rebuilt = dict(original)
    rebuilt["traceEvents"] = [json.loads(unit.payload) for unit in selected]
    return _json_bytes(rebuilt)
