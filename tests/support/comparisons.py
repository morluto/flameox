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


def additional_measurement(run_id: str, value: int) -> dict[str, object]:
    return {
        "measurement_id": f"additional-{run_id}",
        "run_id": run_id,
        "artifact_id": None,
        "name": "pyperf.scan",
        "value_int": value,
        "value_float": None,
        "unit": "ns",
        "aggregation": "sample",
        "scope": "process",
        "trial_id": None,
        "worker_id": "additional-worker",
        "worker_run_index": 0,
        "value_index": 0,
        "loop_count": 1,
        "is_warmup": False,
        "block_id": None,
        "variant_id": None,
        "order_in_block": None,
        "phase": None,
        "dimensions": {},
        "evidence_level": "observed",
    }
