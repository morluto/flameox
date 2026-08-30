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

        def events(self) -> tuple[SimpleNamespace, ...]:
            return (
                SimpleNamespace(
                    time_range=SimpleNamespace(start=4.0, end=9.0),
                ),
            )

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
    assert diagnostics["event_count"] == 1
    assert diagnostics["active_duration_us"] == 5.0


def test_torch_launcher_records_provider_event_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTimeRange:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class FakeEvent:
        def __init__(self, start: float, end: float) -> None:
            self.time_range = FakeTimeRange(start, end)

    class FakeProfile:
        def __enter__(self) -> FakeProfile:
            return self

        def __exit__(self, *_: object) -> Literal[False]:
            return False

        def events(self) -> tuple[FakeEvent, ...]:
            return (FakeEvent(10.0, 12.5), FakeEvent(11.0, 19.0))

        def export_chrome_trace(self, path: str) -> None:
            Path(path).write_text("trace")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
                profile=lambda **_: FakeProfile(),
            ),
        ),
    )
    workload = tmp_path / "workload.py"
    workload.write_text("pass\n")
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

    torch_launcher.main()

    diagnostics = json.loads(
        (output / "torch-profiler-diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["event_count"] == 2
    assert diagnostics["active_duration_us"] == 9.0
    assert diagnostics["artifact_size_bytes"] == len("trace")


@pytest.mark.parametrize("exit_code", [None, 0, 7])
def test_torch_launcher_handles_system_exit_by_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int | None,
) -> None:
    class FakeProfile:
        def __enter__(self) -> FakeProfile:
            return self

        def __exit__(self, *_: object) -> Literal[False]:
            return False

        def export_chrome_trace(self, path: str) -> None:
            Path(path).write_text("{}")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
                profile=lambda **_: FakeProfile(),
            ),
        ),
    )
    workload = tmp_path / "successful_module.py"
    workload.write_text(f"raise SystemExit({exit_code!r})\n")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "torch-trace.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torch-launcher",
            "--output",
            str(output),
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
            "--module",
            workload.stem,
            "--",
            "--workload-option",
        ],
    )

    if exit_code == 7:
        with pytest.raises(SystemExit) as failure:
            torch_launcher.main()
        assert failure.value.code == 7
        assert not output.exists()
    else:
        torch_launcher.main()
        assert output.read_text() == "{}"
    diagnostics = json.loads(
        (tmp_path / "torch-profiler-diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["status"] == ("failed" if exit_code == 7 else "succeeded")
