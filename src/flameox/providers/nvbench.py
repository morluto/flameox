from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from flameox.providers.benchmark_scaling import scaling_projection
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_BENCHMARKS = 10_000
_MAX_STATES = 100_000
_MAX_SUMMARIES = 1_000
_MAX_SAMPLES = 1_000_000
_HINTS = {
    "file/sample_times": ("seconds", "sample_times"),
    "file/sample_freqs": ("hertz", "sample_freqs"),
}


@dataclass(slots=True)
class _NvbenchBundle:
    version: str
    rows: list[dict[str, Any]]
    measurement_count: int
    benchmark_names: set[str]
    series: dict[tuple[str, str], tuple[float, int]]


class NvbenchProvider:
    """Read an explicit NVBench JSON-bin directory."""

    def analyze(
        self,
        capability_id: str,
        paths: Sequence[Path],
        formats: Sequence[str],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> ProviderAnalysis | None:
        if capability_id not in {"benchmark.summary", "benchmark.scaling", "benchmark.compare"}:
            return None
        if not paths or any(format_name != "nvbench" for format_name in formats):
            return None
        parsed = [self._read_bundle(path, max_rows=max_rows) for path in paths]
        if capability_id == "benchmark.compare":
            return self._compare(parsed, arguments, max_rows=max_rows)
        if capability_id == "benchmark.scaling":
            if any(bundle.measurement_count > len(bundle.rows) for bundle in parsed):
                raise ProviderFailure(
                    "LIMIT_EXCEEDED",
                    "benchmark.scaling requires complete samples; raise the startup row limit",
                )
            return scaling_projection(
                [row for bundle in parsed for row in bundle.rows],
                arguments,
                provider_id="nvbench",
                provider_version=parsed[0].version,
                max_rows=max_rows,
            )
        rows: list[dict[str, Any]] = []
        observed = 0
        for input_index, bundle in enumerate(parsed):
            observed += bundle.measurement_count
            for row in bundle.rows:
                if len(rows) < max_rows:
                    rows.append({"input_index": input_index, **row})
        return ProviderAnalysis(
            provider_id="nvbench",
            provider_version=parsed[0].version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "input_count": len(parsed),
                        "measurement_count": observed,
                        "benchmark_names": sorted(
                            {name for item in parsed for name in item.benchmark_names}
                        ),
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= len(rows),
            limitations=[
                "NVBench sample times and frequencies are provider measurements; device "
                "synchronization semantics come from NVBench."
            ],
        )

    @staticmethod
    def _compare(
        bundles: Sequence[_NvbenchBundle],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> ProviderAnalysis:
        if len(bundles) < 2:
            raise ProviderFailure("INVALID_INPUT", "benchmark.compare requires at least 2 inputs")
        baseline_index = int(arguments.get("baseline_index", 0))
        if baseline_index >= len(bundles):
            raise ProviderFailure("INVALID_INPUT", "baseline_index does not select an input")
        requested_metric = arguments.get("metric")
        common = set(bundles[baseline_index].series)
        for bundle in bundles:
            common.intersection_update(bundle.series)
        if requested_metric is not None:
            common = {identity for identity in common if identity[0] == requested_metric}
        baseline = bundles[baseline_index].series
        rows: list[dict[str, Any]] = []
        for identity in sorted(common):
            baseline_sum, baseline_count = baseline[identity]
            baseline_mean = baseline_sum / baseline_count
            for input_index, bundle in enumerate(bundles):
                if input_index == baseline_index:
                    continue
                candidate_sum, candidate_count = bundle.series[identity]
                candidate_mean = candidate_sum / candidate_count
                if len(rows) < max_rows:
                    rows.append(
                        {
                            "benchmark": identity[0],
                            "unit": identity[1],
                            "baseline_index": baseline_index,
                            "candidate_index": input_index,
                            "baseline_mean": baseline_mean,
                            "candidate_mean": candidate_mean,
                            "ratio": candidate_mean / baseline_mean if baseline_mean else None,
                        }
                    )
        observed = len(common) * (len(bundles) - 1)
        return ProviderAnalysis(
            provider_id="nvbench",
            provider_version=bundles[0].version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "input_count": len(bundles),
                        "compatible_metric_count": len(common),
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= len(rows),
            limitations=[
                "Ratios summarize observed sample means and do not establish causal improvement."
            ],
        )

    @staticmethod
    def _read_bundle(  # noqa: C901 - bounded traversal mirrors the NVBench document hierarchy
        path: Path, *, max_rows: int | None
    ) -> _NvbenchBundle:
        if not path.is_dir():
            raise ProviderFailure(
                "UNSUPPORTED_FORMAT",
                "NVBench evidence must be the explicit JSON-bin directory so sidecars are bound.",
            )
        json_files = sorted(path.glob("*.json"))
        if len(json_files) != 1:
            raise ProviderFailure(
                "UNSUPPORTED_FORMAT", "NVBench directory must contain exactly one root JSON file"
            )
        primary = json_files[0]
        if primary.stat().st_size > _MAX_JSON_BYTES:
            raise ProviderFailure("LIMIT_EXCEEDED", "NVBench JSON exceeds 64 MiB")
        try:
            document = json.loads(primary.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise ProviderFailure("DECODE_FAILURE", "NVBench JSON is invalid") from error
        root = _object(document, "NVBench document")
        meta = _object(root.get("meta", {}), "NVBench metadata")
        versions = _object(meta.get("version", {}), "NVBench versions")
        json_version = _object(versions.get("json"), "NVBench JSON version")
        if json_version.get("major") != 1:
            raise ProviderFailure(
                "UNSUPPORTED_FORMAT", "Only NVBench JSON schema major 1 is supported"
            )
        native_version = _object(versions.get("nvbench", {}), "NVBench version")
        provider_version = native_version.get("string")
        if not isinstance(provider_version, str) or not provider_version:
            provider_version = ".".join(
                str(native_version.get(name, 0)) for name in ("major", "minor", "patch")
            )
        benchmarks = root.get("benchmarks")
        if not isinstance(benchmarks, list) or len(benchmarks) > _MAX_BENCHMARKS:
            raise ProviderFailure("LIMIT_EXCEEDED", "NVBench benchmark count is invalid")
        rows: list[dict[str, Any]] = []
        series: dict[tuple[str, str], tuple[float, int]] = {}
        benchmark_names: set[str] = set()
        measurement_count = 0
        state_count = 0
        for benchmark_value in benchmarks:
            benchmark = _object(benchmark_value, "NVBench benchmark")
            benchmark_name = _text(benchmark.get("name"), "NVBench benchmark name")
            states = benchmark.get("states", [])
            if not isinstance(states, list):
                raise ProviderFailure("DECODE_FAILURE", "NVBench states must be an array")
            state_count += len(states)
            if state_count > _MAX_STATES:
                raise ProviderFailure("LIMIT_EXCEEDED", "NVBench state count exceeds the limit")
            for state_value in states:
                state = _object(state_value, "NVBench state")
                if state.get("is_skipped") is True:
                    continue
                summaries = state.get("summaries", [])
                if not isinstance(summaries, list) or len(summaries) > _MAX_SUMMARIES:
                    raise ProviderFailure("LIMIT_EXCEEDED", "NVBench summary count is invalid")
                for summary_value in summaries:
                    summary = _object(summary_value, "NVBench summary")
                    hint = summary.get("hint")
                    if isinstance(hint, str) and hint.startswith("file/") and hint not in _HINTS:
                        raise ProviderFailure(
                            "UNSUPPORTED_FORMAT", f"Unknown NVBench file hint: {hint}"
                        )
                    if hint not in _HINTS:
                        continue
                    filename, count = _sidecar_reference(summary)
                    sidecar = _contained_sidecar(path, filename)
                    unit = _HINTS[cast(str, hint)][0]
                    series_name = _HINTS[cast(str, hint)][1]
                    metric_name = f"{benchmark_name}.{series_name}"
                    benchmark_names.add(metric_name)
                    total, seen = series.get((metric_name, unit), (0.0, 0))
                    for row in _sidecar_rows(
                        sidecar,
                        count=count,
                        benchmark=benchmark_name,
                        state=state,
                        unit=unit,
                        series=series_name,
                    ):
                        measurement_count += 1
                        total += float(cast(float, row["value_float"]))
                        seen += 1
                        if max_rows is None or len(rows) < max_rows:
                            rows.append(row)
                    series[(metric_name, unit)] = total, seen
                    if measurement_count > _MAX_SAMPLES:
                        raise ProviderFailure(
                            "LIMIT_EXCEEDED", "NVBench sample count exceeds the limit"
                        )
        return _NvbenchBundle(
            provider_version,
            rows,
            measurement_count,
            benchmark_names,
            series,
        )


def _sidecar_reference(summary: Mapping[str, Any]) -> tuple[str, int]:
    data = summary.get("data")
    if not isinstance(data, list) or len(data) > 100:
        raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar metadata is invalid")
    values: dict[str, Any] = {}
    types: dict[str, Any] = {}
    for datum_value in data:
        datum = _object(datum_value, "NVBench summary datum")
        name = datum.get("name")
        if isinstance(name, str):
            values[name] = datum.get("value")
            types[name] = datum.get("type")
    filename = values.get("filename")
    raw_count = values.get("size")
    if types.get("filename") != "string" or not isinstance(filename, str) or not filename:
        raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar filename is invalid")
    if (
        types.get("size") != "int64"
        or not isinstance(raw_count, str)
        or not raw_count.isascii()
        or not raw_count.isdigit()
    ):
        raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar size is invalid")
    count = int(raw_count)
    if count > _MAX_SAMPLES:
        raise ProviderFailure("LIMIT_EXCEEDED", "NVBench sidecar sample count exceeds the limit")
    return filename, count


def _contained_sidecar(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar path escapes its bundle")
    candidate = root / value
    if candidate.is_symlink():
        raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar is missing or external") from error
    if not resolved.is_file():
        raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar is not a regular file")
    return resolved


def _sidecar_rows(
    path: Path,
    *,
    count: int,
    benchmark: str,
    state: Mapping[str, Any],
    unit: str,
    series: str,
) -> Iterator[dict[str, Any]]:
    expected_bytes = count * 4
    if path.stat().st_size != expected_bytes:
        raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar size contradicts its metadata")
    with path.open("rb") as stream:
        for index in range(count):
            raw = stream.read(4)
            if len(raw) != 4:
                raise ProviderFailure("DECODE_FAILURE", "NVBench sidecar ended early")
            value = struct.unpack("<f", raw)[0]
            if not math.isfinite(value):
                raise ProviderFailure(
                    "DECODE_FAILURE", "NVBench sidecar contains a non-finite value"
                )
            yield {
                "benchmark": f"{benchmark}.{series}",
                "unit": unit,
                "is_warmup": False,
                "value_int": None,
                "value_float": value,
                "value_index": index,
                "dimensions": _state_dimensions(state),
            }


def _state_dimensions(state: Mapping[str, Any]) -> dict[str, Any]:
    name = state.get("name")
    dimensions = {
        "state": name,
        "device": state.get("device"),
        "type_config_index": state.get("type_config_index"),
    }
    parsed_name = False
    if isinstance(name, str):
        for component in name.split(","):
            key, separator, value = component.strip().partition("=")
            if (
                separator
                and key
                and value
                and len(key) <= 120
                and len(value) <= 200
                and all(character.isalnum() or character in "._:/-" for character in key)
            ):
                dimensions.setdefault(key, value)
                parsed_name = True
    if parsed_name:
        dimensions.pop("state")
    return dimensions


def _object(value: object, subject: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProviderFailure("DECODE_FAILURE", f"{subject} must be an object")
    return cast(dict[str, Any], value)


def _text(value: object, subject: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 500:
        raise ProviderFailure("DECODE_FAILURE", f"{subject} is invalid")
    return value
