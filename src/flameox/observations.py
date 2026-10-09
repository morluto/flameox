"""Shared bounds for SDK-produced semantic observations."""

from __future__ import annotations

import math
from typing import Any

MAX_OBSERVATION_EVENT_BYTES = 16 * 1024


def validate_observation_label(name: str) -> None:
    if not name or len(name) > 200:
        raise ValueError("observation and phase names must contain 1 to 200 characters")
    name.encode("utf-8")


def bounded_observation_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError("observation nesting exceeds eight levels")
    if isinstance(value, str):
        value.encode("utf-8")
        return value
    if value is None or isinstance(value, int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observations cannot contain non-finite numbers")
        return value
    if isinstance(value, list | tuple):
        if len(value) > 256:
            raise ValueError("observation lists cannot exceed 256 items")
        return [bounded_observation_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 256 or any(not isinstance(key, str) for key in value):
            raise ValueError("observation objects require at most 256 string keys")
        for key in value:
            key.encode("utf-8")
        return {
            key: bounded_observation_value(item, depth=depth + 1) for key, item in value.items()
        }
    raise TypeError(f"unsupported observation value: {type(value).__name__}")
