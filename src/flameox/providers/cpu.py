from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from flameox.filesystem import open_binary, read_regular_bytes
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

_MAX_PROFILE_BYTES = 64 * 1024 * 1024
_MAX_FRAMES = 100_000
_MAX_SAMPLES = 1_000_000
_MAX_LINE_BYTES = 256 * 1024
_SPEEDSCOPE_TIME_SCALES = {
    "nanoseconds": 1e-9,
    "microseconds": 1e-6,
    "milliseconds": 1e-3,
    "seconds": 1.0,
}


class CpuProfileProvider:
    """Bounded readers for py-spy Speedscope and collapsed perf stacks."""

    def analyze(
        self,
        operation: str,
        path: Path,
        format_name: str,
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> ProviderAnalysis | None:
        if format_name == "py-spy":
            return self._speedscope(operation, path, arguments, max_rows=max_rows)
        if format_name == "perf" and operation == "rank_cpu_hotspots":
            return self._collapsed(path, max_rows=max_rows)
        return None

    @staticmethod
    def _speedscope(
        operation: str, path: Path, arguments: Mapping[str, Any], *, max_rows: int
    ) -> ProviderAnalysis:
        normalized_frames, profiles = _read_speedscope(path)
        callers = (
            _SampledCallers(normalized_frames, arguments)
            if operation == "inspect_cpu_callers"
            else None
        )
        self_weights: defaultdict[int, int | float] = defaultdict(int)
        inclusive_weights: defaultdict[int, int | float] = defaultdict(int)
        sample_count = 0
        unresolved_sample_count = 0
        weight_units: set[str] = set()
        for profile_index, profile_value in enumerate(profiles):
            profile = _object(profile_value, "Speedscope sampled profile")
            if profile.get("type") != "sampled":
                raise ProviderFailure(
                    "UNSUPPORTED_FORMAT", "Only sampled Speedscope profiles are supported"
                )
            samples = profile.get("samples")
            weights = profile.get("weights")
            if not isinstance(samples, list) or (
                weights is not None and not isinstance(weights, list)
            ):
                raise ProviderFailure("DECODE_FAILURE", "Speedscope samples are invalid")
            if weights is not None and len(weights) != len(samples):
                raise ProviderFailure("DECODE_FAILURE", "Speedscope weights do not match samples")
            weight_unit, weight_scale = _speedscope_weight_unit(profile.get("unit", "none"))
            weight_units.add(weight_unit)
            if len(weight_units) > 1:
                raise ProviderFailure(
                    "UNSUPPORTED_FORMAT",
                    "Speedscope CPU profiles cannot mix time weights with sample counts",
                )
            sample_count += len(samples)
            if sample_count > _MAX_SAMPLES:
                raise ProviderFailure("LIMIT_EXCEEDED", "Speedscope sample count exceeds the limit")
            for sample_index, stack_value in enumerate(samples):
                if not isinstance(stack_value, list):
                    raise ProviderFailure("DECODE_FAILURE", "Speedscope stack is invalid")
                try:
                    weight = (
                        1 if weights is None else _number(weights[sample_index], "sample weight")
                    ) * weight_scale
                except OverflowError as error:
                    raise ProviderFailure(
                        "DECODE_FAILURE", "Speedscope weight is invalid"
                    ) from error
                if isinstance(weight, float) and not math.isfinite(weight):
                    raise ProviderFailure("DECODE_FAILURE", "Speedscope weight is invalid")
                if not stack_value:
                    unresolved_sample_count += 1
                    continue
                stack = [_frame_index(value, len(normalized_frames)) for value in stack_value]
                self_weights[stack[-1]] = _sum_weights(self_weights[stack[-1]], weight)
                for frame_index in set(stack):
                    inclusive_weights[frame_index] = _sum_weights(
                        inclusive_weights[frame_index], weight
                    )
                if callers is not None:
                    callers.add(profile_index, stack, weight)
        rows = [
            {
                **normalized_frames[index],
                "self_weight": self_weights[index],
                "inclusive_weight": inclusive_weights[index],
                "unit": next(iter(weight_units)),
            }
            for index in inclusive_weights
        ]
        rows.sort(key=lambda row: (-row["self_weight"], str(row["function"])))
        if not rows:
            raise ProviderFailure(
                "DECODE_FAILURE", "Speedscope profile contains no resolved samples"
            )
        limitations = [
            "Sample weights rank observed CPU stacks and do not prove causal optimization impact."
        ]
        if unresolved_sample_count:
            limitations.append(
                f"{unresolved_sample_count} of {sample_count} samples had no resolved Python "
                "frames and were excluded from frame aggregation."
            )
        metrics = {
            "frame_count": len(normalized_frames),
            "sample_count": sample_count,
            "weight_unit": next(iter(weight_units)),
        }
        if unresolved_sample_count:
            metrics["unresolved_sample_count"] = unresolved_sample_count
        if callers is not None:
            rows = callers.rows(next(iter(weight_units)))
            metrics["edge_count"] = len(rows)
            limitations.append(
                "Each edge is counted once per sampled stack, including recursive self-edges; "
                "sample counts and weights are not function invocation counts. "
                "Profiles remain separate."
            )
        return ProviderAnalysis(
            provider_id="py-spy-speedscope",
            provider_version="speedscope-1",
            blocks=[
                {
                    "type": "metrics",
                    "values": metrics,
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows),
            complete=len(rows) <= max_rows,
            limitations=limitations,
        )

    @staticmethod
    def _collapsed(path: Path, *, max_rows: int) -> ProviderAnalysis:
        aggregates: defaultdict[str, int] = defaultdict(int)
        sample_count = 0
        try:
            with open_binary(path) as stream:
                while raw := stream.readline(_MAX_LINE_BYTES + 1):
                    if len(raw) > _MAX_LINE_BYTES:
                        raise ProviderFailure(
                            "LIMIT_EXCEEDED", "Collapsed perf stack line is too large"
                        )
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    stack_text, separator, count_text = line.rpartition(" ")
                    if not separator or not count_text.isdigit() or not stack_text:
                        raise ProviderFailure("DECODE_FAILURE", "Collapsed perf stack is invalid")
                    try:
                        count = int(count_text)
                    except ValueError as error:
                        raise ProviderFailure(
                            "DECODE_FAILURE", "Collapsed perf sample count is invalid"
                        ) from error
                    frames = [frame for frame in stack_text.split(";") if frame]
                    if not frames:
                        raise ProviderFailure(
                            "DECODE_FAILURE", "Collapsed perf stack has no frames"
                        )
                    sample_count += count
                    aggregates[frames[-1]] += count
                    if len(aggregates) > _MAX_FRAMES:
                        raise ProviderFailure(
                            "LIMIT_EXCEEDED", "Collapsed perf frame count exceeds the limit"
                        )
        except UnicodeDecodeError as error:
            raise ProviderFailure(
                "DECODE_FAILURE", "Collapsed perf stacks must be UTF-8"
            ) from error
        rows = [
            {"function": function, "self_samples": count, "unit": "samples"}
            for function, count in sorted(aggregates.items(), key=lambda item: (-item[1], item[0]))
        ]
        return ProviderAnalysis(
            provider_id="perf-collapsed",
            provider_version="collapsed-stacks-v1",
            blocks=[
                {
                    "type": "metrics",
                    "values": {"frame_count": len(rows), "sample_count": sample_count},
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows),
            complete=len(rows) <= max_rows,
            limitations=[
                "Input must be collapsed perf stacks, not raw perf.data or perf script output."
            ],
        )


class _SampledCallers:
    def __init__(self, frames: list[dict[str, Any]], arguments: Mapping[str, Any]) -> None:
        self.frames = frames
        self.direction = arguments.get("direction", "both")
        function = arguments.get("function")
        self.matches = {
            index
            for index, frame in enumerate(frames)
            if function is None
            or function.casefold()
            in f"{frame['file']}:{frame['line']}:{frame['function']}".casefold()
        }
        self.weights: defaultdict[tuple[int, int, int], int | float] = defaultdict(int)
        self.samples: defaultdict[tuple[int, int, int], int] = defaultdict(int)

    def add(self, profile: int, stack: list[int], weight: int | float) -> None:
        for caller, callee in set(pairwise(stack)):
            if self.direction == "callers":
                matches = callee in self.matches
            elif self.direction == "callees":
                matches = caller in self.matches
            else:
                matches = caller in self.matches or callee in self.matches
            if not matches:
                continue
            key = (profile, caller, callee)
            if key not in self.weights and len(self.weights) >= 100_000:
                raise ProviderFailure(
                    "LIMIT_EXCEEDED", "Speedscope caller edge count exceeds the limit"
                )
            self.weights[key] = _sum_weights(self.weights[key], weight)
            self.samples[key] += 1

    def rows(self, unit: str) -> list[dict[str, Any]]:
        return [
            {
                "profile_index": profile,
                **{
                    f"caller_{field}": self.frames[caller][field]
                    for field in ("frame_index", "function", "file", "line", "column")
                },
                **{
                    f"callee_{field}": self.frames[callee][field]
                    for field in ("frame_index", "function", "file", "line", "column")
                },
                "sample_count": self.samples[(profile, caller, callee)],
                "weight": weight,
                "unit": unit,
            }
            for (profile, caller, callee), weight in sorted(
                self.weights.items(), key=lambda item: (-item[1], item[0])
            )
        ]


def _read_speedscope(path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    if path.stat().st_size > _MAX_PROFILE_BYTES:
        raise ProviderFailure("LIMIT_EXCEEDED", "py-spy profile exceeds 64 MiB")
    try:
        document = json.loads(read_regular_bytes(path, max_bytes=_MAX_PROFILE_BYTES))
    except (OSError, ValueError, RecursionError) as error:
        raise ProviderFailure("DECODE_FAILURE", "py-spy Speedscope profile is invalid") from error
    root = _object(document, "Speedscope profile")
    shared = _object(root.get("shared"), "Speedscope shared data")
    frames = shared.get("frames")
    profiles = root.get("profiles")
    if not isinstance(frames, list) or len(frames) > _MAX_FRAMES:
        raise ProviderFailure("LIMIT_EXCEEDED", "Speedscope frame count is invalid")
    if not isinstance(profiles, list) or not profiles:
        raise ProviderFailure("DECODE_FAILURE", "Speedscope profiles are missing")
    return [_frame(value, index) for index, value in enumerate(frames)], profiles


def _object(value: object, subject: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProviderFailure("DECODE_FAILURE", f"{subject} must be an object")
    return cast(dict[str, Any], value)


def _speedscope_weight_unit(value: object) -> tuple[str, int | float]:
    if value == "bytes":
        raise ProviderFailure(
            "UNSUPPORTED_FORMAT",
            "Byte-weighted Speedscope profiles are not CPU hotspot evidence",
        )
    if value == "none":
        return "samples", 1
    if isinstance(value, str) and value in _SPEEDSCOPE_TIME_SCALES:
        return "seconds", _SPEEDSCOPE_TIME_SCALES[value]
    raise ProviderFailure("DECODE_FAILURE", "Speedscope weight unit is invalid")


def _frame(value: object, index: int) -> dict[str, Any]:
    frame = _object(value, "Speedscope frame")
    name = frame.get("name")
    if not isinstance(name, str) or not name or len(name) > 2_000:
        raise ProviderFailure("DECODE_FAILURE", "Speedscope frame name is invalid")
    return {
        "frame_index": index,
        "function": name,
        "file": frame.get("file") if isinstance(frame.get("file"), str) else None,
        "line": frame.get("line") if isinstance(frame.get("line"), int) else None,
        "column": frame.get("col") if isinstance(frame.get("col"), int) else None,
    }


def _frame_index(value: object, frame_count: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < frame_count:
        raise ProviderFailure("DECODE_FAILURE", "Speedscope frame reference is invalid")
    return value


def _number(value: object, subject: str) -> int | float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or value < 0
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise ProviderFailure("DECODE_FAILURE", f"{subject} is invalid")
    return value


def _sum_weights(left: int | float, right: int | float) -> int | float:
    try:
        result = left + right
    except OverflowError as error:
        raise ProviderFailure("LIMIT_EXCEEDED", "Speedscope weight total is too large") from error
    if isinstance(result, float) and not math.isfinite(result):
        raise ProviderFailure("LIMIT_EXCEEDED", "Speedscope weight total is too large")
    return result
