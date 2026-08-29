from __future__ import annotations

from typing import Any

from flameox.workers.aiperf_contract import AIPERF_WORKER
from flameox.workers.compute_sanitizer_contract import COMPUTE_SANITIZER_WORKER
from flameox.workers.memray_contract import MEMRAY_WORKER
from flameox.workers.nsight_compute_contract import NSIGHT_COMPUTE_WORKER
from flameox.workers.nsight_systems_contract import NSIGHT_SYSTEMS_WORKER
from flameox.workers.nvml_contract import NVML_WORKER
from flameox.workers.otlp_contract import OTLP_WORKER
from flameox.workers.perfetto_contract import PERFETTO_WORKER
from flameox.workers.protocol import WorkerDefinition
from flameox.workers.reduction_contract import SHRINKRAY_WORKER

ARTIFACT_WORKERS: tuple[WorkerDefinition[Any, Any], ...] = (
    AIPERF_WORKER,
    COMPUTE_SANITIZER_WORKER,
    MEMRAY_WORKER,
    NSIGHT_COMPUTE_WORKER,
    NSIGHT_SYSTEMS_WORKER,
    NVML_WORKER,
    OTLP_WORKER,
    PERFETTO_WORKER,
    SHRINKRAY_WORKER,
)


def worker_for_module(module: str) -> WorkerDefinition[Any, Any] | None:
    return next(
        (definition for definition in ARTIFACT_WORKERS if definition.module == module),
        None,
    )
