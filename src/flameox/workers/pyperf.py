from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pyperf
from pydantic import JsonValue

from flameox.workers.protocol import WorkerApplication, WorkerFailureKind, run_typed_worker
from flameox.workers.pyperf_contract import PYPERF_WORKER, PyperfWorkerRequest, PyperfWorkerResult


def _normalized_value(value: float, unit: str) -> tuple[int | None, float | None, str]:
    if unit == "second":
        return round(value * 1_000_000_000), None, "ns"
    if unit == "byte":
        return round(value), None, "bytes"
    if unit == "integer":
        return round(value), None, "count"
    return None, float(value), unit


def _handle(request: PyperfWorkerRequest, _job_root: Path) -> PyperfWorkerResult:
    suite = pyperf.BenchmarkSuite.load(str(Path(request.artifact_path)))
    rows: list[dict[str, Any]] = []
    series_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    series_truncated = False
    measurement_count = 0
    warmup_count = 0
    for benchmark in suite.get_benchmarks():
        benchmark_name = benchmark.get_name()
        unit = benchmark.get_unit()
        for run_index, run in enumerate(benchmark.get_runs()):
            for value_index, (loops, value) in enumerate(run.warmups):
                warmup_count += 1
                if request.projection == "samples" and len(rows) < request.max_rows:
                    value_int, value_float, normalized_unit = _normalized_value(value, unit)
                    rows.append(
                        {
                            "benchmark": benchmark_name,
                            "unit": normalized_unit,
                            "value_int": value_int,
                            "value_float": value_float,
                            "worker_run_index": run_index,
                            "value_index": value_index,
                            "loop_count": loops,
                            "is_warmup": True,
                        }
                    )
            loops = run.get_loops()
            for value_index, value in enumerate(run.values):
                measurement_count += 1
                value_int, value_float, normalized_unit = _normalized_value(value, unit)
                normalized_value = value_int if value_int is not None else value_float
                assert normalized_value is not None
                if request.projection == "series":
                    if request.metric in {None, benchmark_name}:
                        key = (benchmark_name, normalized_unit, loops)
                        aggregate = series_rows.get(key)
                        if aggregate is None:
                            if len(series_rows) >= request.max_rows:
                                series_truncated = True
                                continue
                            aggregate = {
                                "benchmark": benchmark_name,
                                "unit": normalized_unit,
                                "value_int": None,
                                "value_float": None,
                                "loop_count": loops,
                                "is_warmup": False,
                                "sample_sum": 0.0,
                                "sample_count": 0,
                            }
                            series_rows[key] = aggregate
                        aggregate["sample_sum"] += float(normalized_value)
                        aggregate["sample_count"] += 1
                    continue
                if len(rows) < request.max_rows:
                    rows.append(
                        {
                            "benchmark": benchmark_name,
                            "unit": normalized_unit,
                            "value_int": value_int,
                            "value_float": value_float,
                            "worker_run_index": run_index,
                            "value_index": value_index,
                            "loop_count": loops,
                            "is_warmup": False,
                        }
                    )
    observed = measurement_count + warmup_count
    if request.projection == "series":
        rows = list(series_rows.values())
        truncated = series_truncated
    else:
        truncated = observed > len(rows)
    return PyperfWorkerResult(
        reader_version=version("pyperf"),
        benchmark_names=tuple(suite.get_benchmark_names()),
        measurement_count=measurement_count,
        warmup_count=warmup_count,
        rows=cast(tuple[dict[str, JsonValue], ...], tuple(rows)),
        truncated=truncated,
        limitations=(
            (
                f"pyperf {'series' if request.projection == 'series' else 'rows'} were "
                f"truncated to {request.max_rows} entries.",
            )
            if truncated
            else ()
        ),
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=PYPERF_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="pyperf benchmark suite is unsupported or invalid",
            caught=(OSError, ValueError, TypeError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
