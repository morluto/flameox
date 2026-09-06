"""Single-pass preview projection over literal JSON events."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ijson
from ijson.common import ObjectBuilder

type Event = tuple[str, Any]


def _value_events(events: Iterator[Event], first: Event) -> Iterator[Event]:
    """Consume exactly one value, without interpreting map keys as paths."""
    yield first
    depth = int(first[0] in {"start_array", "start_map"})
    while depth:
        event = next(events)
        if event[0] in {"start_array", "start_map"}:
            depth += 1
        elif event[0] in {"end_array", "end_map"}:
            depth -= 1
        yield event


def _array_rows(events: Iterator[Event]) -> Iterator[dict[str, Any]]:
    for first in events:
        if first[0] == "end_array":
            return
        builder = ObjectBuilder()
        for event, value in _value_events(events, first):
            builder.event(event, value)
        value = builder.value
        yield value if isinstance(value, dict) else {"value": value}


def iter_json_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream array elements, object sections, or a scalar root in document order."""
    with path.open("rb") as stream:
        events = iter(ijson.basic_parse(stream, use_float=True))
        first = next(events)
        if first[0] == "start_array":
            yield from _array_rows(events)
        elif first[0] == "start_map":
            for event, key in events:
                if event == "end_map":
                    break
                first = next(events)
                if first[0] == "start_array":
                    for row in _array_rows(events):
                        yield {"section": key, **row}
                elif first[0] == "start_map":
                    for _ in _value_events(events, first):
                        pass
                    yield {"key": key, "value_type": "object"}
                else:
                    yield {"key": key, "value": first[1]}
        else:
            yield {"value": first[1]}
        # Drive the parser through EOF so trailing garbage cannot look complete.
        for _ in events:
            raise ValueError("Unexpected trailing JSON value")
