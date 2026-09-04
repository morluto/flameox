from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

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

    def analyze(self, path: Path, format_name: str, *, max_rows: int) -> ProviderAnalysis | None:
        if format_name == "py-spy":
            return self._speedscope(path, max_rows=max_rows)
        if format_name == "perf":
            return self._collapsed(path, max_rows=max_rows)
        return None

    @staticmethod
    def _speedscope(path: Path, *, max_rows: int) -> ProviderAnalysis:
        if path.stat().st_size > _MAX_PROFILE_BYTES:
            raise ProviderFailure("LIMIT_EXCEEDED", "py-spy profile exceeds 64 MiB")
        try:
            document = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise ProviderFailure(
                "DECODE_FAILURE", "py-spy Speedscope profile is invalid"
            ) from error
        root = _object(document, "Speedscope profile")
        shared = _object(root.get("shared"), "Speedscope shared data")
        frames = shared.get("frames")
        profiles = root.get("profiles")
        if not isinstance(frames, list) or len(frames) > _MAX_FRAMES:
            raise ProviderFailure("LIMIT_EXCEEDED", "Speedscope frame count is invalid")
        if not isinstance(profiles, list) or not profiles:
            raise ProviderFailure("DECODE_FAILURE", "Speedscope profiles are missing")
        normalized_frames = [_frame(value, index) for index, value in enumerate(frames)]
        self_weights: defaultdict[int, float] = defaultdict(float)
        inclusive_weights: defaultdict[int, float] = defaultdict(float)
        sample_count = 0
        unresolved_sample_count = 0
        weight_units: set[str] = set()
        for profile_value in profiles:
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
                if not stack_value:
                    unresolved_sample_count += 1
                    continue
                stack = [_frame_index(value, len(normalized_frames)) for value in stack_value]
                weight = (
                    1.0 if weights is None else _number(weights[sample_index], "sample weight")
                ) * weight_scale
                self_weights[stack[-1]] += weight
                for frame_index in set(stack):
                    inclusive_weights[frame_index] += weight
        rows = [
            {
                **normalized_frames[index],
                "self_weight": self_weights[index],
                "inclusive_weight": inclusive_weights[index],
                "unit": next(iter(weight_units)),
            }
            for index in inclusive_weights
        ]
        rows.sort(key=lambda row: (-float(row["self_weight"]), str(row["function"])))
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
            "frame_count": len(frames),
            "sample_count": sample_count,
            "weight_unit": next(iter(weight_units)),
        }
        if unresolved_sample_count:
            metrics["unresolved_sample_count"] = unresolved_sample_count
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
            with path.open("rb") as stream:
                for raw in stream:
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
                    count = int(count_text)
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


def _object(value: object, subject: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProviderFailure("DECODE_FAILURE", f"{subject} must be an object")
    return cast(dict[str, Any], value)


def _speedscope_weight_unit(value: object) -> tuple[str, float]:
    if value == "bytes":
        raise ProviderFailure(
            "UNSUPPORTED_FORMAT",
            "Byte-weighted Speedscope profiles are not CPU hotspot evidence",
        )
    if value == "none":
        return "samples", 1.0
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


def _number(value: object, subject: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ProviderFailure("DECODE_FAILURE", f"{subject} is invalid")
    try:
        result = float(value)
    except OverflowError as error:
        raise ProviderFailure("DECODE_FAILURE", f"{subject} is invalid") from error
    if not math.isfinite(result) or result < 0:
        raise ProviderFailure("DECODE_FAILURE", f"{subject} is invalid")
    return result
