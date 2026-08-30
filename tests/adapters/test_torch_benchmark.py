from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

from flameox.sdk import torch_benchmark

pytestmark = pytest.mark.unit


class _Measurement:
    number_per_run = 4
    times = (0.001, 0.002, 0.003)


class _Timer:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        type(self).calls.append(kwargs)

    def blocked_autorange(self, *, min_run_time: float) -> _Measurement:
        self.kwargs["operation"]() if "operation" in self.kwargs else self.kwargs["globals"][
            "operation"
        ]()
        self.kwargs["min_run_time"] = min_run_time
        return _Measurement()


class _Event:
    def __init__(self, *, enable_timing: bool) -> None:
        assert enable_timing

    def record(self) -> None:
        pass

    def elapsed_time(self, end: _Event) -> float:
        assert isinstance(end, _Event)
        return 0.25


def _install_torch(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool) -> Any:
    class _Cuda:
        synchronized = 0

        @staticmethod
        def is_available() -> bool:
            return cuda_available

        @staticmethod
        def current_device() -> int:
            return 2

        @staticmethod
        def current_stream() -> SimpleNamespace:
            return SimpleNamespace(cuda_stream=17)

        @classmethod
        def synchronize(cls) -> None:
            cls.synchronized += 1

        Event = _Event

    torch: Any = ModuleType("torch")
    torch.__path__ = []
    torch.__version__ = "2.13.0"
    torch.cuda = _Cuda
    utils = ModuleType("torch.utils")
    utils.__path__ = []
    benchmark = ModuleType("torch.utils.benchmark")
    benchmark.Timer = _Timer  # type: ignore[attr-defined]
    utils.benchmark = benchmark  # type: ignore[attr-defined]
    torch.utils = utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.benchmark", benchmark)
    return _Cuda


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cuda_event_timing: bool,
) -> Any:
    output = tmp_path / "benchmark-samples.json"
    monkeypatch.setenv("FLAMEOX_BENCHMARK_OUTPUT", str(output))
    monkeypatch.setenv(
        "FLAMEOX_TORCH_BENCHMARK_CONFIG",
        json.dumps(
            {
                "min_run_time_seconds": 0.4,
                "max_samples": 2,
                "num_threads": 3,
                "cuda_event_timing": cuda_event_timing,
            }
        ),
    )
    return output


def test_torch_benchmark_uses_timer_samples_without_cuda_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Timer.calls.clear()
    _install_torch(monkeypatch, cuda_available=False)
    output = _configure(monkeypatch, tmp_path, cuda_event_timing=False)
    operations: list[str] = []

    torch_benchmark(
        "gae.step",
        lambda: operations.append("ran"),
        dimensions={"shape": "32x2048", "dtype": "float32"},
    )

    assert operations == ["ran"]
    assert len(_Timer.calls) == 1
    assert _Timer.calls[0]["stmt"] == "operation()"
    assert _Timer.calls[0]["num_threads"] == 3
    assert _Timer.calls[0]["min_run_time"] == 0.4
    document = json.loads(output.read_text())
    assert document == {
        "schema_version": "flameox.benchmark-samples.v1",
        "producer": "torch.utils.benchmark",
        "producer_version": "2.13.0",
        "benchmarks": [
            {
                "name": "gae.step",
                "unit": "ns",
                "measurement_clock": "host_monotonic",
                "synchronization": "not_required",
                "scope": "operator",
                "phase": "steady_state",
                "loop_count": 4,
                "worker_id": None,
                "worker_run_index": None,
                "trial_id": None,
                "block_id": None,
                "variant_id": None,
                "order_in_block": None,
                "device": None,
                "dimensions": {"shape": "32x2048", "dtype": "float32"},
                "warmups": [],
                "samples": [1_000_000, 2_000_000],
            }
        ],
    }


def test_torch_benchmark_emits_cuda_events_under_a_distinct_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Timer.calls.clear()
    cuda = _install_torch(monkeypatch, cuda_available=True)
    output = _configure(monkeypatch, tmp_path, cuda_event_timing=True)
    operations: list[str] = []

    torch_benchmark("gae.step", lambda: operations.append("ran"))

    document = json.loads(output.read_text())
    host, device = document["benchmarks"]
    assert operations == ["ran", "ran", "ran"]
    assert cuda.synchronized == 2
    assert host["name"] == "gae.step"
    assert host["measurement_clock"] == "host_monotonic"
    assert host["synchronization"] == "device_synchronize"
    assert device["name"] == "gae.step.cuda_event"
    assert device["measurement_clock"] == "cuda_event"
    assert device["synchronization"] == "device_synchronize"
    assert device["scope"] == "device"
    assert device["loop_count"] == 1
    assert device["device"] == {"type": "cuda", "index": 2, "stream": "17"}
    assert device["samples"] == [250_000, 250_000]


def test_torch_benchmark_rejects_cuda_event_requests_without_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torch(monkeypatch, cuda_available=False)
    _configure(monkeypatch, tmp_path, cuda_event_timing=True)

    with pytest.raises(RuntimeError, match="CUDA-event timing was requested"):
        torch_benchmark("gae.step", lambda: None)
