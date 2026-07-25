from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flamo_phase",
    default=None,
)
_WRITE_LOCK = threading.Lock()
_MAX_EVENT_BYTES = 16 * 1024


def observe(name: str, **values: Any) -> None:
    """Emit one bounded semantic observation when capture has enabled the SDK."""
    if not name or len(name) > 200:
        raise ValueError("observation names must contain 1 to 200 characters")
    path = os.environ.get("FLAMO_OBSERVATIONS_PATH")
    if path is None:
        return
    payload = {
        "schema_version": 1,
        "kind": "annotation",
        "name": name,
        "phase": _PHASE.get(),
        "monotonic_ns": time.monotonic_ns(),
        "values": _bounded_value(values),
    }
    encoded = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    if len(encoded) > _MAX_EVENT_BYTES:
        raise ValueError("observation exceeds the 16 KiB event limit")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, destination.open("ab") as stream:
        stream.write(encoded)
        stream.flush()


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Annotate a bounded logical phase around user code."""
    if not name or len(name) > 200:
        raise ValueError("phase names must contain 1 to 200 characters")
    token = _PHASE.set(name)
    observe("flamo.phase.start", phase_name=name)
    try:
        yield
    finally:
        observe("flamo.phase.end", phase_name=name)
        _PHASE.reset(token)


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError("observation nesting exceeds eight levels")
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("observations cannot contain non-finite numbers")
        return value
    if isinstance(value, list | tuple):
        if len(value) > 256:
            raise ValueError("observation lists cannot exceed 256 items")
        return [_bounded_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 256 or any(not isinstance(key, str) for key in value):
            raise ValueError("observation objects require at most 256 string keys")
        return {key: _bounded_value(item, depth=depth + 1) for key, item in value.items()}
    raise TypeError(f"unsupported observation value: {type(value).__name__}")
