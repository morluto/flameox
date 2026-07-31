from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Ownership:
    owner: str
    lane: str
    paths: tuple[str, ...]
    markers: tuple[str, ...]


def load_ownership(path: Path) -> tuple[Ownership, ...]:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    return tuple(
        Ownership(
            owner=str(record["owner"]),
            lane=str(record["lane"]),
            paths=tuple(str(item) for item in record["paths"]),
            markers=tuple(str(item) for item in record["markers"]),
        )
        for record in document["ownership"]
    )


def ownership_by_path(records: tuple[Ownership, ...]) -> dict[str, Ownership]:
    by_path: dict[str, Ownership] = {}
    for record in records:
        for path in record.paths:
            if path in by_path:
                raise ValueError(f"Duplicate ownership path: {path}")
            by_path[path] = record
    return by_path
