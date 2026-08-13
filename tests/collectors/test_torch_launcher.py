from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from flameox.collectors import torch_launcher

pytestmark = pytest.mark.unit


def test_torch_launcher_records_workload_phase_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProfile:
        def __enter__(self) -> FakeProfile:
            return self

        def __exit__(self, *_: object) -> Literal[False]:
            return False

        def export_chrome_trace(self, path: str) -> None:
            Path(path).write_text("{}")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
            profile=lambda **_: FakeProfile(),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    workload = tmp_path / "failing_workload.py"
    workload.write_text("raise RuntimeError('synthetic workload failure')\n")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torch-launcher",
            "--output",
            str(output / "torch-trace.json"),
            "--config",
            json.dumps(
                {
                    "mode": "whole_entrypoint",
                    "activities": ["cpu"],
                    "record_shapes": False,
                    "profile_memory": False,
                    "with_stack": False,
                    "with_flops": False,
                    "with_modules": False,
                }
            ),
            "--script",
            str(workload),
        ],
    )

    with pytest.raises(RuntimeError, match="synthetic workload failure"):
        torch_launcher.main()

    diagnostics = json.loads(
        (output / "torch-profiler-diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["phase"] == "workload_execution"
    assert diagnostics["status"] == "failed"
    assert "synthetic workload failure" in diagnostics["detail"]
