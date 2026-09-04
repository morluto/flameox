from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from flameox.benchmark_samples import BenchmarkSamplesV1, BenchmarkSeries
from flameox.workers.benchmark_samples_contract import (
    BENCHMARK_SAMPLES_WORKER,
    BenchmarkSamplesWorkerRequest,
    BenchmarkSamplesWorkerResult,
)
from flameox.workers.protocol import WorkerApplication, WorkerFailureKind, run_typed_worker


def _row(
    series: BenchmarkSeries,
    *,
    series_index: int,
    value_index: int,
    value: int | float,
    is_warmup: bool,
) -> dict[str, Any]:
    dimensions = {
        **series.dimensions,
        "measurement_clock": series.measurement_clock,
        "synchronization": series.synchronization,
    }
    if series.device is not None:
        dimensions["device.type"] = series.device.type
        if series.device.index is not None:
            dimensions["device.index"] = str(series.device.index)
        if series.device.stream is not None:
            dimensions["device.stream"] = series.device.stream
    integer_unit = series.unit in {"ns", "bytes", "count"}
    return {
        "benchmark": series.name,
        "unit": series.unit,
        "value_int": int(value) if integer_unit else None,
        "value_float": float(value) if not integer_unit else None,
        "series_index": series_index,
        "value_index": value_index,
        "loop_count": series.loop_count,
        "is_warmup": is_warmup,
        "scope": series.scope,
        "phase": "warmup" if is_warmup else series.phase,
        "worker_id": series.worker_id,
        "worker_run_index": series.worker_run_index,
        "trial_id": series.trial_id,
        "block_id": series.block_id,
        "variant_id": series.variant_id,
        "order_in_block": series.order_in_block,
        "dimensions": dimensions,
    }


def _series_row(series: BenchmarkSeries, *, series_index: int) -> dict[str, Any]:
    row = _row(
        series,
        series_index=series_index,
        value_index=0,
        value=series.samples[0],
        is_warmup=False,
    )
    row.pop("value_index")
    row["value_int"] = None
    row["value_float"] = None
    row["sample_sum"] = sum(series.samples)
    row["sample_count"] = len(series.samples)
    return row


def _handle(
    request: BenchmarkSamplesWorkerRequest, _job_root: Path
) -> BenchmarkSamplesWorkerResult:
    payload = BenchmarkSamplesV1.model_validate_json(Path(request.artifact_path).read_bytes())
    rows: list[dict[str, Any]] = []
    measurement_count = 0
    warmup_count = 0
    limitations: list[str] = []
    if payload.producer_version is None:
        limitations.append("The benchmark producer version was not declared.")
    for series_index, series in enumerate(payload.benchmarks):
        if series.measurement_clock in {"cuda_event", "hip_event", "device_event"} and (
            series.synchronization in {"none", "unknown"}
        ):
            limitations.append(
                f"{series.name} uses asynchronous device timing with "
                f"synchronization={series.synchronization}."
            )
        if request.projection == "series":
            measurement_count += len(series.samples)
            warmup_count += len(series.warmups)
            if request.metric in {None, series.name} and len(rows) < request.max_rows:
                rows.append(_series_row(series, series_index=series_index))
            continue
        for is_warmup, values in ((True, series.warmups), (False, series.samples)):
            for value_index, value in enumerate(values):
                if is_warmup:
                    warmup_count += 1
                else:
                    measurement_count += 1
                if len(rows) < request.max_rows:
                    rows.append(
                        _row(
                            series,
                            series_index=series_index,
                            value_index=value_index,
                            value=value,
                            is_warmup=is_warmup,
                        )
                    )
    observed = measurement_count + warmup_count
    selected_series_count = sum(
        request.metric in {None, series.name} for series in payload.benchmarks
    )
    truncated = (
        selected_series_count > len(rows)
        if request.projection == "series"
        else observed > len(rows)
    )
    if truncated:
        subject = "series" if request.projection == "series" else "rows"
        limitations.append(f"Benchmark {subject} were truncated to {request.max_rows} entries.")
    return BenchmarkSamplesWorkerResult(
        producer=payload.producer,
        producer_version=payload.producer_version,
        benchmark_names=tuple(series.name for series in payload.benchmarks),
        measurement_count=measurement_count,
        warmup_count=warmup_count,
        rows=cast(tuple[dict[str, JsonValue], ...], tuple(rows)),
        truncated=truncated,
        limitations=tuple(dict.fromkeys(limitations)),
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=BENCHMARK_SAMPLES_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="benchmark samples are unsupported or invalid",
            caught=(OSError, ValueError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
