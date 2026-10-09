"""Stable task names for the runtime's evidence capabilities."""

from __future__ import annotations

from flameox.runtime_contracts import CAPABILITIES, compatible_capture_providers

# Names describe the question; provider names remain typed collector choices.
ANALYSIS_TOOLS = {
    "trace.summary": "summarize_trace",
    "trace.call_graph": "inspect_trace_call_graph",
    "trace.pytorch": "summarize_pytorch_trace",
    "trace.operations": "summarize_trace_operations",
    "trace.lifecycle": "summarize_trace_lifecycle",
    "trace.window": "inspect_trace_window",
    "cpu.hotspots": "rank_cpu_hotspots",
    "cpu.callers": "inspect_cpu_callers",
    "memory.hotspots": "rank_allocation_hotspots",
    "memory.retained": "rank_retained_memory",
    "benchmark.summary": "summarize_benchmarks",
    "benchmark.scaling": "analyze_benchmark_scaling",
    "benchmark.compare": "compare_benchmarks",
    "inference.summary": "summarize_inference",
    "inference.compare": "compare_inference",
    "gpu.launches": "inspect_gpu_launches",
    "gpu.kernel_metrics": "inspect_gpu_kernel_metrics",
    "triton.autotune": "inspect_triton_autotune",
    "sanitizer.failures": "inspect_sanitizer_failures",
    "kernel.validation": "inspect_kernel_validation",
    "kernel.compare": "compare_kernel_validation",
    "failures.summary": "summarize_failures",
    "pytest.fixtures": "inspect_pytest_fixtures",
    "coverage.summary": "summarize_coverage",
    "static.performance_candidates": "inspect_performance_candidates",
    "artifact.preview": "preview_artifact",
}

CAPTURE_TOOLS = {
    capability.id: "capture_" + capability.id.replace(".", "_")
    for capability in CAPABILITIES
    if compatible_capture_providers(capability)
}

CAPABILITY_BY_TOOL = {name: identity for identity, name in ANALYSIS_TOOLS.items()} | {
    name: identity for identity, name in CAPTURE_TOOLS.items()
}
