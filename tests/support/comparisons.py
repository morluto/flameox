from __future__ import annotations

from pathlib import Path

import pyperf

from flameox.adapters import PyPerfExtractor
from flameox.application import ImportArtifactRequest, ImportService
from flameox.domain import ArtifactKind
from flameox.storage import Workspace


def benchmark(path: Path, values: tuple[float, float, float]) -> None:
    run = pyperf.Run(
        values,
        metadata={"name": "scan", "unit": "second", "loops": 1},
        collect_metadata=False,
    )
    pyperf.BenchmarkSuite([pyperf.Benchmark([run])]).dump(
        str(path),
        replace=True,
    )


def benchmark_workers(
    path: Path,
    workers: tuple[tuple[float, float, float], ...],
) -> None:
    runs = [
        pyperf.Run(
            values,
            metadata={"name": "scan", "unit": "second", "loops": 1},
            collect_metadata=False,
        )
        for values in workers
    ]
    pyperf.BenchmarkSuite([pyperf.Benchmark(runs)]).dump(str(path), replace=True)


def imported_benchmark(
    workspace: Workspace,
    path: Path,
    values: tuple[float, float, float],
) -> str:
    benchmark(path, values)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=path,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
        )
    )
    PyPerfExtractor(workspace).extract(imported.run.run_id)
    return imported.run.run_id


def imported_benchmark_workers(
    workspace: Workspace,
    path: Path,
    workers: tuple[tuple[float, float, float], ...],
) -> str:
    benchmark_workers(path, workers)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=path,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
        )
    )
    PyPerfExtractor(workspace).extract(imported.run.run_id)
    return imported.run.run_id


def measurement_row(
    run_id: str,
    value: int,
    *,
    measurement_id: str | None = None,
    worker_id: str = "additional-worker",
    worker_run_index: int = 0,
    value_index: int = 0,
) -> dict[str, object]:
    return {
        "measurement_id": measurement_id or f"additional-{run_id}",
        "run_id": run_id,
        "artifact_id": None,
        "name": "pyperf.scan",
        "value_int": value,
        "value_float": None,
        "unit": "ns",
        "aggregation": "sample",
        "scope": "process",
        "trial_id": None,
        "worker_id": worker_id,
        "worker_run_index": worker_run_index,
        "value_index": value_index,
        "loop_count": 1,
        "is_warmup": False,
        "block_id": None,
        "variant_id": None,
        "order_in_block": None,
        "phase": None,
        "dimensions": {},
        "evidence_level": "observed",
    }
