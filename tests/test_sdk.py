from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Self, cast

import pytest

import flameox.sdk as sdk
from flameox.sdk import TorchProfilerSession, memray_region, observe, phase

pytestmark = pytest.mark.unit


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _break_observation_parent(path: Path) -> None:
    path.unlink()
    path.parent.rmdir()
    path.parent.write_text("not a directory")


def _restore_observation_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "recovery" / "events.jsonl"
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(path))
    return path


def test_phase_restores_context_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(path))

    with phase("compile"):
        observe("inside")
    observe("after")

    events = _events(path)
    assert [event["phase"] for event in events] == ["compile", "compile", "compile", None]
    assert set(events[0]) == {"name", "phase", "monotonic_ns", "values"}


def test_phase_start_failure_restores_context_without_entering_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(blocked / "events.jsonl"))
    entered = False

    with pytest.raises(OSError), phase("compile"):
        entered = True

    assert entered is False
    recovered = _restore_observation_path(monkeypatch, tmp_path)
    observe("after")
    assert _events(recovered)[0]["phase"] is None


def test_phase_end_failure_is_fatal_after_success_and_restores_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "observations" / "events.jsonl"
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(path))

    with pytest.raises(OSError), phase("compile"):
        _break_observation_parent(path)

    recovered = _restore_observation_path(monkeypatch, tmp_path)
    observe("after")
    assert _events(recovered)[0]["phase"] is None


def test_phase_preserves_workload_failure_when_end_observation_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WorkloadFailure(Exception):
        pass

    path = tmp_path / "observations" / "events.jsonl"
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(path))

    with pytest.raises(WorkloadFailure, match="primary") as raised, phase("compile"):
        _break_observation_parent(path)
        raise WorkloadFailure("primary")

    assert any("phase-end observation also failed" in note for note in raised.value.__notes__)
    recovered = _restore_observation_path(monkeypatch, tmp_path)
    observe("after")
    assert _events(recovered)[0]["phase"] is None


def test_nested_phase_failures_restore_the_outer_and_root_contexts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WorkloadFailure(Exception):
        pass

    path = tmp_path / "observations" / "events.jsonl"
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(path))

    with pytest.raises(WorkloadFailure) as raised, phase("outer"), phase("inner"):
        _break_observation_parent(path)
        raise WorkloadFailure("primary")

    assert len(raised.value.__notes__) == 2
    recovered = _restore_observation_path(monkeypatch, tmp_path)
    observe("after")
    assert _events(recovered)[0]["phase"] is None


@pytest.mark.anyio
async def test_phase_context_is_task_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(path))

    async def worker(name: str) -> None:
        with phase(name):
            await asyncio.sleep(0)
            observe(f"inside-{name}")

    await asyncio.gather(worker("one"), worker("two"))
    inside = {
        event["name"]: event["phase"]
        for event in _events(path)
        if str(event["name"]).startswith("inside-")
    }
    assert inside == {"inside-one": "one", "inside-two": "two"}


def test_torch_profiler_exit_failure_remains_primary_when_phase_end_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProfilerExitFailure(Exception):
        pass

    class RecordFunction:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_error: object) -> None:
            raise ProfilerExitFailure("record function failed")

    torch = SimpleNamespace(
        profiler=SimpleNamespace(record_function=lambda _name: RecordFunction())
    )
    session = TorchProfilerSession(object(), torch)
    path = tmp_path / "observations" / "events.jsonl"
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(path))

    with pytest.raises(ProfilerExitFailure) as raised, session.phase("compile"):
        _break_observation_parent(path)

    assert any("phase-end observation also failed" in note for note in raised.value.__notes__)
    recovered = _restore_observation_path(monkeypatch, tmp_path)
    observe("after")
    assert _events(recovered)[0]["phase"] is None


def test_memray_region_binds_the_exact_plan_region_and_closes_one_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, bool, bool]] = []

    class Tracker:
        def __init__(
            self,
            output: object,
            *,
            native_traces: bool,
            trace_python_allocators: bool,
        ) -> None:
            calls.append((output, native_traces, trace_python_allocators))

        def __enter__(self) -> Tracker:
            return self

        def __exit__(self, *_error: object) -> None:
            return None

    path = tmp_path / "observations.jsonl"
    monkeypatch.setattr(sdk, "_MEMRAY_REGION_STATE", "idle")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(path))
    monkeypatch.setenv(
        "FLAMEOX_MEMRAY_CONFIG",
        json.dumps(
            {
                "mode": "sdk",
                "region": "steady_step",
                "native_traces": True,
                "trace_python_allocators": True,
            }
        ),
    )
    output = tmp_path / "memory.bin"
    monkeypatch.setenv("FLAMEOX_MEMRAY_OUTPUT", str(output))
    monkeypatch.setitem(sys.modules, "memray", SimpleNamespace(Tracker=Tracker))

    with memray_region("steady_step"):
        pass

    assert calls == [(str(output), True, True)]
    events = _events(path)
    assert [event["name"] for event in events] == [
        "flameox.memray.region.start",
        "flameox.memray.region.end",
    ]
    assert events[0]["values"] == {
        "region": "steady_step",
        "torch_cuda_initialized": None,
        "warmup_count": 0,
    }
    assert events[1]["values"] == {
        "completed": True,
        "region": "steady_step",
        "torch_cuda_initialized": None,
    }


def test_memray_region_rejects_wrong_nested_and_repeated_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tracker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Tracker:
            return self

        def __exit__(self, *_error: object) -> None:
            return None

    monkeypatch.setattr(sdk, "_MEMRAY_REGION_STATE", "idle")
    monkeypatch.setenv("FLAMEOX_OBSERVATIONS_PATH", str(tmp_path / "observations.jsonl"))
    monkeypatch.setenv(
        "FLAMEOX_MEMRAY_CONFIG",
        json.dumps({"mode": "sdk", "region": "steady_step"}),
    )
    monkeypatch.setenv("FLAMEOX_MEMRAY_OUTPUT", str(tmp_path / "memory.bin"))
    monkeypatch.setitem(sys.modules, "memray", SimpleNamespace(Tracker=Tracker))

    with pytest.raises(RuntimeError, match="exact region"), memray_region("other_step"):
        pass
    with (
        memray_region("steady_step"),
        pytest.raises(RuntimeError, match="exactly one region"),
        memray_region("steady_step"),
    ):
        pass
    with pytest.raises(RuntimeError, match="exactly one region"), memray_region("steady_step"):
        pass

    events = _events(tmp_path / "observations.jsonl")
    assert [event["name"] for event in events] == [
        "flameox.memray.region.start",
        "flameox.memray.region.error",
        "flameox.memray.region.end",
        "flameox.memray.region.error",
    ]
    assert [
        cast(dict[str, object], event["values"])["reason"]
        for event in events
        if event["name"] == "flameox.memray.region.error"
    ] == ["overlap", "repeated"]
