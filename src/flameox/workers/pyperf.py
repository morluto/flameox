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
    measurement_count = 0
    warmup_count = 0
    for benchmark in suite.get_benchmarks():
        benchmark_name = benchmark.get_name()
        unit = benchmark.get_unit()
        for run_index, run in enumerate(benchmark.get_runs()):
            for value_index, (loops, value) in enumerate(run.warmups):
                warmup_count += 1
                if len(rows) < request.max_rows:
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
                if len(rows) < request.max_rows:
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
                            "is_warmup": False,
                        }
                    )
    observed = measurement_count + warmup_count
    return PyperfWorkerResult(
        reader_version=version("pyperf"),
        benchmark_names=tuple(suite.get_benchmark_names()),
        measurement_count=measurement_count,
        warmup_count=warmup_count,
        rows=cast(tuple[dict[str, JsonValue], ...], tuple(rows)),
        truncated=observed > len(rows),
        limitations=(
            (f"pyperf rows were truncated to {request.max_rows} entries.",)
            if observed > len(rows)
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
