from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# The conditional imports preserve static types while runtime exports stay lazy.
# ruff: noqa: F405
if TYPE_CHECKING:
    from flameox.adapters.benchmark_samples import *  # noqa: F403
    from flameox.adapters.client_setup import *  # noqa: F403
    from flameox.adapters.coverage import *  # noqa: F403
    from flameox.adapters.inference import *  # noqa: F403
    from flameox.adapters.memray import *  # noqa: F403
    from flameox.adapters.nsight_systems import *  # noqa: F403
    from flameox.adapters.observations import *  # noqa: F403
    from flameox.adapters.perfetto import *  # noqa: F403
    from flameox.adapters.pyperf import *  # noqa: F403
    from flameox.adapters.pytest import *  # noqa: F403
    from flameox.adapters.python_startup import *  # noqa: F403
    from flameox.adapters.registry import *  # noqa: F403
    from flameox.adapters.setup_runtime import *  # noqa: F403
    from flameox.adapters.torch_profiler import *  # noqa: F403
    from flameox.adapters.toxiproxy import *  # noqa: F403


_MODULES = (
    "benchmark_samples",
    "client_setup",
    "coverage",
    "inference",
    "memray",
    "nsight_systems",
    "observations",
    "perfetto",
    "pyperf",
    "pytest",
    "python_startup",
    "registry",
    "setup_runtime",
    "torch_profiler",
    "toxiproxy",
)

__all__ = [
    "ALL_SETUP_CLIENTS",
    "AIPerfCorrelationSummary",
    "AIPerfInputsIndex",
    "AIPerfRecordParser",
    "AdapterApproval",
    "AdapterDescriptor",
    "AdapterDiscoveryResult",
    "AdapterRegistry",
    "BenchmarkSamplesExtractionResult",
    "BenchmarkSamplesExtractor",
    "BenchmarkSamplesV1",
    "ClientConfigEdit",
    "ClientConfigRegistry",
    "ClientPlanAction",
    "CoverageExtractionResult",
    "CoverageExtractor",
    "InferenceArtifactExtractor",
    "InferenceExtractionResult",
    "Launcher",
    "ManagedRuntime",
    "MemrayExtractionResult",
    "MemrayExtractor",
    "MooncakeRequestRow",
    "MooncakeTraceParser",
    "MooncakeTraceSummary",
    "NsightSystemsExtractionResult",
    "NsightSystemsExtractor",
    "ObservationExtractionResult",
    "ObservationExtractor",
    "PerfettoExtractionResult",
    "PerfettoExtractor",
    "PyPerfExtractionResult",
    "PyPerfExtractor",
    "PytestExtractionResult",
    "PytestExtractor",
    "PythonStartupExtractionResult",
    "PythonStartupExtractor",
    "RuntimeInstallation",
    "SdkTorchProfilerOptions",
    "SetupClient",
    "SglangResultDocument",
    "SglangResultParser",
    "TorchProfilerCaptureOptions",
    "TorchProfilerSchedule",
    "ToxiproxyApiError",
    "ToxiproxyClient",
    "ToxiproxyToolManager",
    "ToxiproxyToolReceipt",
    "TraceEvent",
    "TraceProcessorInstallation",
    "TraceWindowResult",
    "VllmAggregateMetrics",
    "VllmMeasurementRow",
    "VllmResultDocument",
    "VllmResultParser",
    "WholeEntrypointTorchProfilerOptions",
    "install_trace_processor",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module_name in _MODULES:
        module = import_module(f"{__name__}.{module_name}")
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
