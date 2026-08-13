from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

from flameox.sdk import TorchProfilerSession, observe, phase

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
