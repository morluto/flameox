"""Request-bound pytest plugin that emits bounded Flameox reliability events."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

pytest = importlib.import_module("pytest")

_OUTPUT_ENVIRONMENT = "FLAMEOX_PYTEST_OUTPUT"
_MAX_TEXT = 4096
_MAX_EVENT_BYTES = 64 * 1024


def _write(event: dict[str, Any]) -> None:
    output = os.environ.get(_OUTPUT_ENVIRONMENT)
    if output is None:
        raise pytest.UsageError(f"{_OUTPUT_ENVIRONMENT} is required by the Flameox plugin")
    encoded = (json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n").encode()
    if len(encoded) > _MAX_EVENT_BYTES:
        raise pytest.UsageError("Flameox pytest event exceeds 64 KiB")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _text(value: object) -> str:
    return str(value)[:_MAX_TEXT]


def pytest_sessionstart(session: object) -> None:
    del session
    _write({"event": "run_started", "run_started_at_ns": time.time_ns()})


def pytest_collection_modifyitems(items: list[Any]) -> None:
    for item in items:
        _write({"event": "test_collected", "nodeid": _text(item.nodeid)})


def pytest_runtest_logreport(report: Any) -> None:
    worker = getattr(report, "worker_id", "main")
    _write(
        {
            "event": "test_phase",
            "nodeid": _text(report.nodeid),
            "phase": _text(report.when),
            "outcome": _text(report.outcome),
            "duration_ns": max(0, round(report.duration * 1_000_000_000)),
            "worker_id": _text(worker),
        }
    )


def pytest_keyboard_interrupt(excinfo: object) -> None:
    del excinfo
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
