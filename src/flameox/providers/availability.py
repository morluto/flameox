"""Declarative installation and workload requirements for profiler providers."""

from __future__ import annotations

MANAGED_PROVIDER_EXTRAS = {
    "aiperf": "inference",
    "memray": "memory",
    "otlp": "trace",
    "perfetto": "trace",
    "py-spy": "cpu",
    "torch": "torch",
}

SYSTEM_PROVIDER_GUIDANCE = {
    "compute-sanitizer": "Install NVIDIA Compute Sanitizer with the CUDA Toolkit.",
    "nsight-compute": "Install NVIDIA Nsight Compute with its extras/python interface.",
    "nsight-systems": "Install NVIDIA Nsight Systems and make nsys available on PATH.",
    "nvbench": "Build the target benchmark with NVBench and verify CUDA device access.",
    "perf": "Install Linux perf for the running kernel and grant profiling permission.",
    "perfetto": "Install Perfetto Trace Processor and make trace_processor_shell available.",
    "rocprofv3": "Install ROCProfiler SDK and make rocprofv3 available on PATH.",
    "triton": "Install Triton in the target Python environment and verify device access.",
}

WORKLOAD_PYTHON_REQUIREMENTS = {
    "coverage": ("coverage", "coverage", ">=7.14,<8"),
    "memray": ("memray", "memray", ">=1.17"),
    "torch-profiler": ("torch", "torch", ">=2.7"),
}

WORKLOAD_PROVIDER_GUIDANCE = {
    "memray": "Install compatible Memray in the exact workload Python interpreter; "
    "server preparation only supplies the analysis reader.",
    "torch": "Install compatible PyTorch in the exact workload Python interpreter and "
    "verify the requested CPU or accelerator activity is supported.",
}
