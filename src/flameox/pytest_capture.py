"""Request-bound pytest plugin that emits bounded Flameox reliability events."""

from __future__ import annotations

import importlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

pytest = importlib.import_module("pytest")

_MAX_TEXT = 4096
_MAX_EVENT_BYTES = 64 * 1024
_FIXTURE_INVOCATIONS = itertools.count(1)
_output_path: Path | None = None
_current_worker_id = "main"


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--flameox-output", required=True, help="Flameox event stream path")


def pytest_configure(config: Any) -> None:
    global _current_worker_id, _output_path
    _output_path = Path(str(config.getoption("--flameox-output")))
    _current_worker_id = _worker_id(config)


def _write(event: dict[str, Any]) -> None:
    if _output_path is None:
        raise pytest.UsageError("--flameox-output is required by the Flameox plugin")
    encoded = (json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n").encode()
    if len(encoded) > _MAX_EVENT_BYTES:
        raise pytest.UsageError("Flameox pytest event exceeds 64 KiB")
    destination = _output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _text(value: object) -> str:
    return str(value)[:_MAX_TEXT]


def _worker_id(config: Any) -> str:
    worker_input = getattr(config, "workerinput", None)
    if isinstance(worker_input, dict) and isinstance(worker_input.get("workerid"), str):
        return _text(worker_input["workerid"])
    return "main"


@pytest.hookimpl(wrapper=True)  # type: ignore[untyped-decorator]
def pytest_fixture_setup(fixturedef: Any, request: Any) -> Any:
    worker_id = _worker_id(request.config)
    invocation_id = f"{worker_id}:{next(_FIXTURE_INVOCATIONS)}"
    fixture = _text(fixturedef.argname)
    scope = _text(fixturedef.scope)
    nodeid = _text(getattr(request.node, "nodeid", ""))
    teardown_started_ns: int | None = None

    def finish_teardown() -> None:
        finished_ns = time.perf_counter_ns()
        _write(
            {
                "event": "fixture_phase",
                "fixture": fixture,
                "scope": scope,
                "phase": "teardown",
                "outcome": "observed" if teardown_started_ns is not None else "incomplete",
                "duration_ns": (
                    max(0, finished_ns - teardown_started_ns)
                    if teardown_started_ns is not None
                    else None
                ),
                "worker_id": worker_id,
                "invocation_id": invocation_id,
                "nodeid": nodeid,
            }
        )

    def begin_teardown() -> None:
        nonlocal teardown_started_ns
        teardown_started_ns = time.perf_counter_ns()

    request.addfinalizer(finish_teardown)
    started_ns = time.perf_counter_ns()
    try:
        result = yield
    except BaseException:
        _write(
            {
                "event": "fixture_phase",
                "fixture": fixture,
                "scope": scope,
                "phase": "setup",
                "outcome": "failed",
                "duration_ns": max(0, time.perf_counter_ns() - started_ns),
                "worker_id": worker_id,
                "invocation_id": invocation_id,
                "nodeid": nodeid,
            }
        )
        raise
    else:
        _write(
            {
                "event": "fixture_phase",
                "fixture": fixture,
                "scope": scope,
                "phase": "setup",
                "outcome": "passed",
                "duration_ns": max(0, time.perf_counter_ns() - started_ns),
                "worker_id": worker_id,
                "invocation_id": invocation_id,
                "nodeid": nodeid,
            }
        )
        return result
    finally:
        request.addfinalizer(begin_teardown)


def pytest_sessionstart(session: object) -> None:
    del session
    _write({"event": "run_started", "run_started_at_ns": time.time_ns()})


def pytest_collection_modifyitems(items: list[Any]) -> None:
    for item in items:
        _write({"event": "test_collected", "nodeid": _text(item.nodeid)})


def pytest_collectreport(report: Any) -> None:
    if report.failed:
        _write(
            {
                "event": "collection_error",
                "nodeid": _text(report.nodeid),
                "outcome": _text(report.outcome),
            }
        )


def pytest_runtest_logreport(report: Any) -> None:
    if _current_worker_id == "main" and hasattr(report, "worker_id"):
        return
    _write(
        {
            "event": "test_phase",
            "nodeid": _text(report.nodeid),
            "phase": _text(report.when),
            "outcome": _text(report.outcome),
            "duration_ns": max(0, round(report.duration * 1_000_000_000)),
            "worker_id": _current_worker_id,
        }
    )


def pytest_keyboard_interrupt(excinfo: Any) -> None:
    if excinfo.type is KeyboardInterrupt:
        _write({"event": "interrupted"})


def pytest_internalerror(
    excrepr: object,
    excinfo: object,
) -> None:
    del excrepr, excinfo
    _write({"event": "internal_error"})


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    del session
    _write({"event": "run_finished", "exitstatus": int(exitstatus)})
