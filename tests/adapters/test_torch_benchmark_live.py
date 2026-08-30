from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.sdk import torch_benchmark

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optional,
    pytest.mark.serial,
    pytest.mark.requires_torch,
]


def test_torch_benchmark_records_distinct_host_and_cuda_event_series(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip(
        "torch",
        reason="optional provider unavailable: install the PyTorch provider",
    )
    if not torch.cuda.is_available():
        pytest.skip("PyTorch is installed, but no usable CUDA device is available")

    output = tmp_path / "benchmark-samples.json"
    monkeypatch.setenv("FLAMEOX_BENCHMARK_OUTPUT", str(output))
    monkeypatch.setenv(
        "FLAMEOX_TORCH_BENCHMARK_CONFIG",
        json.dumps(
            {
                "min_run_time_seconds": 0.01,
                "max_samples": 3,
                "num_threads": 1,
                "cuda_event_timing": True,
            }
        ),
    )

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        left = torch.randn((256, 256), device=device)
        right = torch.randn((256, 256), device=device)
        result = torch.empty((256, 256), device=device)

        def operation() -> None:
            torch.mm(left, right, out=result)

        operation()
        torch.cuda.synchronize()
        expected_stream = str(torch.cuda.current_stream().cuda_stream)

        torch_benchmark(
            "cuda.matmul",
            operation,
            dimensions={"shape": "256x256", "dtype": "float32"},
        )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["producer"] == "torch.utils.benchmark"
    assert document["producer_version"] == str(torch.__version__)
    assert len(document["benchmarks"]) == 2

    host, cuda_event = document["benchmarks"]
    assert host["name"] == "cuda.matmul"
    assert host["measurement_clock"] == "host_monotonic"
    assert host["synchronization"] == "device_synchronize"
    assert host["scope"] == "operator"
    assert host["device"] is None

    assert cuda_event["name"] == "cuda.matmul.cuda_event"
    assert cuda_event["measurement_clock"] == "cuda_event"
    assert cuda_event["synchronization"] == "device_synchronize"
    assert cuda_event["scope"] == "device"
    assert cuda_event["loop_count"] == 1
    assert cuda_event["device"] == {
        "type": "cuda",
        "index": device_index,
        "stream": expected_stream,
    }

    for series in (host, cuda_event):
        assert 1 <= len(series["samples"]) <= 3
        assert len(series["warmups"]) == 0
        assert series["dimensions"] == {"shape": "256x256", "dtype": "float32"}
    assert all(sample >= 0 for sample in host["samples"])
    assert all(sample > 0 for sample in cuda_event["samples"])
