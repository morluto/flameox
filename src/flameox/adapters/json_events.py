"""Streaming JSON locations with literal object keys and distinct array positions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, BinaryIO

import ijson
from ijson.common import ObjectBuilder

type JsonLocation = tuple[str | None, ...]


def parse_json_events(stream: BinaryIO) -> Iterator[tuple[JsonLocation, str, Any]]:
    path: list[str | None] = []
    for event, value in ijson.basic_parse(stream):
        if event == "map_key":
            location = tuple(path[:-1])
            path[-1] = value
        elif event in {"end_map", "end_array"}:
            path.pop()
            location = tuple(path)
        else:
            location = tuple(path)
            if event in {"start_map", "start_array"}:
                path.append(None)
        yield location, event, value


def json_items(stream: BinaryIO, location: JsonLocation) -> Iterator[Any]:
    builder: ObjectBuilder | None = None
    depth = 0
    for current, event, value in parse_json_events(stream):
        if builder is None:
            if current != location or event in {"end_map", "end_array", "map_key"}:
                continue
            builder = ObjectBuilder()
        builder.event(event, value)
        if event in {"start_map", "start_array"}:
            depth += 1
        elif event in {"end_map", "end_array"}:
            depth -= 1
        if depth == 0:
            yield builder.value
            builder = None
