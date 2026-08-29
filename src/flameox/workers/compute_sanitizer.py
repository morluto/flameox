from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

from defusedxml.ElementTree import iterparse  # type: ignore[import-untyped]
from pydantic import JsonValue

from flameox.workers.compute_sanitizer_contract import (
    COMPUTE_SANITIZER_WORKER,
    ComputeSanitizerWorkerRequest,
    ComputeSanitizerWorkerResult,
)
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerContext,
    WorkerFailureKind,
    run_typed_worker,
)

_KNOWN_RECORD_FIELDS = {
    "kind",
    "level",
    "who",
    "what",
    "where",
    "hostStack",
    "accessAddress",
    "accessSize",
    "errorAddress",
}


def _text(element: ET.Element | None, name: str) -> str | None:
    if element is None:
        return None
    child = element.find(name)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value or None


def _integer(element: ET.Element | None, name: str) -> int | None:
    value = _text(element, name)
    if value is None:
        return None
    parsed = int(value, 16 if value.casefold().startswith(("0x", "-0x")) else 10)
    if not -(2**31) <= parsed <= 2**31 - 1:
        raise ValueError(f"{name} is outside the supported 32-bit range")
    return parsed


def _address(element: ET.Element | None, name: str) -> int | None:
    value = _text(element, name)
    if value is None:
        return None
    parsed = int(value, 16 if value.casefold().startswith("0x") else 10)
    if not 0 <= parsed <= 2**64 - 1:
        raise ValueError(f"{name} is outside the supported unsigned 64-bit range")
    return parsed


def _normalize_path(value: str | None, project_root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except (OSError, ValueError):
        return f"<external>/{path.name}" if path.name else "<external>"


def _axes(element: ET.Element | None) -> dict[str, int] | None:
    if element is None:
        return None
    values = {
        axis: value for axis in ("x", "y", "z") if (value := _integer(element, axis)) is not None
    }
    return values or None


def _classification(kind: str | None, message: str | None, error: str | None) -> str:
    normalized_kind = (kind or "").casefold()
    combined = " ".join(item for item in (kind, message, error) if item).casefold()
    if normalized_kind in {"api", "api error"}:
        return "api_error"
    if normalized_kind in {"sanitizer", "sanitizer error"}:
        return "sanitizer_error"
    if normalized_kind in {"race", "hazard"}:
        return "race"
    if normalized_kind in {"initcheck", "uninitialized"}:
        return "uninitialized_memory"
    if normalized_kind in {"synccheck", "synchronization"}:
        return "synchronization"
    if normalized_kind == "precise":
        return "memory_access"
    if re.search(r"\b(race|hazard)\b", combined):
        return "race"
    if re.search(r"\b(uninitialized|initcheck)\b", combined):
        return "uninitialized_memory"
    if re.search(r"\b(sync|synccheck|barrier)\b", combined):
        return "synchronization"
    if re.search(r"\bapi\b", combined):
        return "api_error"
    if any(item in combined for item in ("out of bounds", "misaligned", "memory", "leak")):
        return "memory_access"
    if re.search(r"\b(sanitizer|internal)\b", combined):
        return "sanitizer_error"
    return "unknown"


def _record(
    element: ET.Element,
    *,
    project_root: Path,
    max_frames: int,
) -> tuple[dict[str, object], tuple[str, ...]]:
    kind = _text(element, "kind")
    level = _text(element, "level")
    what = element.find("what")
    where = element.find("where")
    message = _text(what, "text")
    error = _text(what, "error")
    who = element.find("who")
    stack = element.find("hostStack")
    frames: list[dict[str, object]] = []
    frame_elements = list(stack.findall("frame")) if stack is not None else []
    for frame in frame_elements[:max_frames]:
        frames.append(
            {
                "function": _text(frame, "func"),
                "module": _normalize_path(_text(frame, "module"), project_root),
                "path": _normalize_path(_text(frame, "path"), project_root),
                "line": _integer(frame, "line"),
                "pc": _text(frame, "pc"),
            }
        )
    limitations = [
        f"Unknown Compute Sanitizer record element: {child.tag}."
        for child in element
        if child.tag not in _KNOWN_RECORD_FIELDS
    ]
    if len(frame_elements) > max_frames:
        limitations.append(f"Host stack was truncated to {max_frames} frames.")
    access_address = _address(element, "accessAddress")
    if access_address is None:
        access_address = _address(what, "accessAddress")
    if access_address is None:
        access_address = _address(what, "address")
    access_size = _integer(element, "accessSize")
    if access_size is None:
        access_size = _integer(what, "accessSize")
    if access_size is None:
        access_size = _integer(what, "size")
    error_address = _address(element, "errorAddress")
    if error_address is None:
        error_address = _address(what, "errorAddress")
    return (
        {
            "kind": kind,
            "level": level,
            "classification": _classification(kind, message, error),
            "message": message,
            "memory_space": _text(what, "space"),
            "access_address": access_address,
            "access_size": access_size,
            "error_address": error_address,
            "direction": _text(what, "direction"),
            "error": error,
            "function": _text(where, "func"),
            "path": _normalize_path(_text(where, "path"), project_root),
            "line": _integer(where, "line"),
            "pc": _text(where, "pc"),
            "thread": _axes(who.find("threadIdx") if who is not None else None),
            "block": _axes(who.find("blockIdx") if who is not None else None),
            "frames": frames,
        },
        tuple(limitations),
    )


def _extract(
    artifact_path: Path,
    *,
    project_root: Path,
    max_records: int,
    max_frames: int,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    limitations: list[str] = []
    root_seen = False
    truncated = False
    for event, element in iterparse(artifact_path, events=("start", "end")):
        if event == "start" and not root_seen:
            root_seen = True
            if element.tag != "ComputeSanitizerOutput":
                raise ValueError("root element is not ComputeSanitizerOutput")
        if event != "end":
            continue
        if element.tag == "record":
            if len(records) >= max_records:
                truncated = True
            else:
                record, record_limitations = _record(
                    element,
                    project_root=project_root,
                    max_frames=max_frames,
                )
                records.append(record)
                limitations.extend(record_limitations)
            element.clear()
        elif element.tag not in {
            "ComputeSanitizerOutput",
            "kind",
            "level",
            "who",
            "threadIdx",
            "blockIdx",
            "x",
            "y",
            "z",
            "what",
            "text",
            "space",
            "size",
            "accessAddress",
            "accessSize",
            "errorAddress",
            "direction",
            "error",
            "address",
            "where",
            "func",
            "path",
            "line",
            "pc",
            "hostStack",
            "saveLocation",
            "frame",
            "module",
        }:
            limitations.append(f"Unknown Compute Sanitizer XML element: {element.tag}.")
    if not root_seen:
        raise ValueError("XML document is empty")
    if truncated:
        limitations.append(f"Sanitizer records were truncated to {max_records} entries.")
    classifications: dict[str, int] = {}
    for record in records:
        name = str(record["classification"])
        classifications[name] = classifications.get(name, 0) + 1
    return {
        "ok": True,
        "records": records,
        "classifications": classifications,
        "limitations": list(dict.fromkeys(limitations)),
        "truncated": truncated,
    }


def _handle(
    request: ComputeSanitizerWorkerRequest,
    _context: WorkerContext,
) -> ComputeSanitizerWorkerResult:
    result = _extract(
        Path(request.artifact_path),
        project_root=Path(request.project_root).resolve(),
        max_records=request.max_records,
        max_frames=request.max_frames,
    )
    return ComputeSanitizerWorkerResult(
        records=cast(
            tuple[dict[str, JsonValue], ...],
            tuple(cast(list[dict[str, object]], result["records"])),
        ),
        classifications=cast(dict[str, int], result["classifications"]),
        limitations=tuple(cast(list[str], result["limitations"])),
        truncated=cast(bool, result["truncated"]),
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=COMPUTE_SANITIZER_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="Compute Sanitizer XML is unsupported or invalid",
            caught=(OSError, ET.ParseError, ValueError, KeyError, TypeError, json.JSONDecodeError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
