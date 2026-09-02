from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn, cast

import ijson

from flameox.canonical import digest_model
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.workers.protocol import WorkerApplication, WorkerFailureKind, run_typed_worker
from flameox.workers.v8_profiles_contract import (
    V8_PROFILE_WORKER,
    V8ProfileRequest,
    V8ProfileResult,
)


def _parse_cpu(request: V8ProfileRequest) -> V8ProfileResult:
    nodes: dict[int, dict[str, Any]] = {}
    top_level = _top_level_keys(Path(request.artifact_path))
    if "nodes" not in top_level or "samples" not in top_level:
        _malformed("V8 CPU profile must contain nodes and samples arrays.")
    with Path(request.artifact_path).open("rb") as stream:
        for node in ijson.items(stream, "nodes.item"):
            if len(nodes) >= request.max_nodes:
                _limit("V8 CPU profile node limit exceeded.")
            if not isinstance(node, dict):
                _malformed("V8 CPU profile node must be an object.")
            node_id = _strict_int(node.get("id"), "node id")
            if node_id in nodes:
                _malformed("V8 CPU profile contains duplicate node IDs.")
            call_frame = _call_frame(node.get("callFrame"))
            children = node.get("children", [])
            if not isinstance(children, list):
                _malformed("V8 CPU profile node children must be an array.")
            child_ids = tuple(_strict_int(value, "child node id") for value in children)
            hit_count = _strict_int(node.get("hitCount", 0), "hit count")
            if hit_count < 0:
                _malformed("V8 CPU profile hit count cannot be negative.")
            nodes[node_id] = {
                "call_frame": call_frame,
                "children": child_ids,
                "hit_count": hit_count,
            }
    if not nodes:
        _malformed("V8 CPU profile nodes array cannot be empty.")
    sample_count = _count_cpu_samples(Path(request.artifact_path), request.max_samples)
    return _aggregate_cpu(request, nodes, sample_count)


def _count_cpu_samples(path: Path, limit: int) -> int:
    count = 0
    with path.open("rb") as stream:
        for sample in ijson.items(stream, "samples.item"):
            if count >= limit:
                _limit("V8 CPU profile sample limit exceeded.")
            _strict_int(sample, "sample node id")
            count += 1
    return count


def _aggregate_cpu(
    request: V8ProfileRequest,
    nodes: dict[int, dict[str, Any]],
    sample_count: int,
) -> V8ProfileResult:
    parents: dict[int, int] = {}
    for node_id, node in nodes.items():
        for child_id in node["children"]:
            if child_id not in nodes:
                _malformed("V8 CPU profile references an unknown child node.")
            if child_id in parents:
                _malformed("V8 CPU profile contains a node with multiple parents.")
            parents[child_id] = node_id
    roots = tuple(node_id for node_id in nodes if node_id not in parents)
    if not roots and nodes:
        _malformed("V8 CPU profile node tree contains a cycle.")

    frame_rows: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, dict[str, int]] = {}
    visited: set[int] = set()
    subtree_totals: dict[int, int] = {}
    for root in roots:
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            node_id, closing = stack.pop()
            if closing:
                node = nodes[node_id]
                identity = _frame_identity(node["call_frame"], request)
                subtree = node["hit_count"] + sum(
                    subtree_totals[child_id] for child_id in node["children"]
                )
                values = aggregates.setdefault(
                    identity["frame_id"], {"self": 0, "inclusive": 0, "samples": 0}
                )
                values["self"] += node["hit_count"]
                values["inclusive"] += subtree
                values["samples"] += node["hit_count"]
                subtree_totals[node_id] = subtree
                frame_rows.setdefault(identity["frame_id"], identity)
                continue
            if node_id in visited:
                _malformed("V8 CPU profile tree contains a repeated node.")
            visited.add(node_id)
            node = nodes[node_id]
            stack.append((node_id, True))
            for child_id in reversed(node["children"]):
                stack.append((child_id, False))

    if len(visited) != len(nodes):
        _malformed("V8 CPU profile contains disconnected or unreachable nodes.")
    _check_rows(request, len(frame_rows), len(aggregates))
    return V8ProfileResult(
        profile_kind="cpu",
        node_count=len(nodes),
        sample_count=sample_count,
        frames=tuple(frame_rows.values()),
        frame_measurements=tuple(
            {
                "frame_id": frame_id,
                "metric": "cpu.hit_count",
                "self_value": values["self"],
                "inclusive_value": values["inclusive"],
                "unit": "count",
                "sample_count": values["samples"],
            }
            for frame_id, values in sorted(aggregates.items())
        ),
        limitations=(
            "V8 CPU samples represent execution time, not allocation or memory evidence.",
            "The CPU profile contains sampled stack locations; source-map resolution is not "
            "applied.",
        ),
    )


def _parse_heap(request: V8ProfileRequest) -> V8ProfileResult:  # noqa: C901 - streaming validation
    path = Path(request.artifact_path)
    top_level = _top_level_keys(path)
    if "head" not in top_level or "samples" not in top_level:
        _malformed("V8 heap profile must contain head and samples.")
    frame_rows: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, dict[str, int]] = {}
    node_count = 0
    sample_count = 0
    total_sampled_bytes = 0
    node_stack: list[dict[str, Any]] = []
    sample: dict[str, Any] | None = None
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream):
            if event == "start_map" and (prefix == "head" or prefix.endswith(".children.item")):
                if len(node_stack) >= request.max_nodes:
                    _limit("V8 heap profile node limit exceeded.")
                node_stack.append(
                    {"call_frame": None, "self_size": None, "children": False, "child_total": 0}
                )
                continue
            if event == "start_map" and prefix == "samples.item":
                sample = {}
                continue
            if prefix == "samples.item" and event not in {"start_map", "end_map", "map_key"}:
                _malformed("V8 heap sample must be an object.")
            if node_stack:
                current = node_stack[-1]
                if event == "start_array" and prefix.endswith(".children"):
                    current["children"] = True
                elif event in {"string", "number", "null"}:
                    if prefix.endswith(".selfSize"):
                        current["self_size"] = value
                    elif prefix.endswith(".callFrame.functionName"):
                        current.setdefault("call_frame_values", {})["functionName"] = value
                    elif prefix.endswith(".callFrame.url"):
                        current.setdefault("call_frame_values", {})["url"] = value
                    elif prefix.endswith(".callFrame.lineNumber"):
                        current.setdefault("call_frame_values", {})["lineNumber"] = value
                    elif prefix.endswith(".callFrame.columnNumber"):
                        current.setdefault("call_frame_values", {})["columnNumber"] = value
                    elif prefix.endswith(".callFrame.scriptId"):
                        current.setdefault("call_frame_values", {})["scriptId"] = value
            if (
                sample is not None
                and event in {"string", "number", "null"}
                and prefix.endswith((".size", ".nodeId"))
            ):
                sample[prefix.rsplit(".", 1)[-1]] = value
            if event == "end_map" and prefix == "samples.item":
                if sample is None:
                    _malformed("V8 heap sample is malformed.")
                if sample_count >= request.max_samples:
                    _limit("V8 heap profile sample limit exceeded.")
                size = _strict_int(sample.get("size"), "sample size")
                _strict_int(sample.get("nodeId"), "sample node id")
                if size < 0:
                    _malformed("V8 heap sample size cannot be negative.")
                total_sampled_bytes += size
                sample_count += 1
                sample = None
                continue
            if (
                event == "end_map"
                and node_stack
                and (prefix == "head" or prefix.endswith(".children.item"))
            ):
                current = node_stack.pop()
                call_frame = current.get("call_frame_values")
                if not isinstance(call_frame, dict) or not current["children"]:
                    _malformed("V8 heap profile node is missing callFrame or children.")
                self_size = _strict_int(current["self_size"], "self size")
                if self_size < 0:
                    _malformed("V8 heap profile selfSize cannot be negative.")
                if node_stack:
                    node_stack[-1]["child_total"] += self_size + current["child_total"]
                identity = _frame_identity(call_frame, request)
                frame_rows.setdefault(identity["frame_id"], identity)
                values = aggregates.setdefault(
                    identity["frame_id"], {"self": 0, "inclusive": 0, "samples": 0}
                )
                values["self"] += self_size
                values["inclusive"] += self_size + current["child_total"]
                values["samples"] += 1
                node_count += 1
    if node_stack or sample is not None:
        _malformed("V8 heap profile contains an incomplete object.")
    _check_rows(request, len(frame_rows), len(aggregates))
    return V8ProfileResult(
        profile_kind="heap",
        node_count=node_count,
        sample_count=sample_count,
        total_sampled_bytes=total_sampled_bytes,
        frames=tuple(frame_rows.values()),
        frame_measurements=tuple(
            {
                "frame_id": frame_id,
                "metric": "memory.self_size",
                "self_value": values["self"],
                "inclusive_value": values["inclusive"],
                "unit": "bytes",
                "sample_count": values["samples"],
            }
            for frame_id, values in sorted(aggregates.items())
        ),
        limitations=(
            "Sampled allocation bytes are an estimate from V8's sampling heap profiler, not "
            "the exact retained heap or process RSS.",
            "Only allocations sampled by V8 are reported; small or short-lived allocations "
            "may be underrepresented.",
            "Source-map resolution is not applied by this extractor.",
        ),
    )


def _top_level_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream):
            if prefix == "" and event == "map_key":
                keys.add(str(value))
    return keys


def _call_frame(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _malformed("V8 CPU profile callFrame must be an object.")
    return cast(dict[str, Any], value)


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _malformed(f"V8 profile {field} must be an integer.")
    return value


def _frame_identity(call_frame: dict[str, Any], request: V8ProfileRequest) -> dict[str, Any]:
    function_value = call_frame.get("functionName")
    function = (
        function_value if isinstance(function_value, str) and function_value else "(anonymous)"
    )
    url_value = call_frame.get("url")
    url = url_value if isinstance(url_value, str) else ""
    normalized = url
    line = _strict_int(call_frame.get("lineNumber", 0), "line number")
    column = _strict_int(call_frame.get("columnNumber", 0), "column number")
    script_id = call_frame.get("scriptId")
    disambiguator = (
        f"v8-script:{script_id}"
        if not normalized and isinstance(script_id, (str, int)) and not isinstance(script_id, bool)
        else None
    )
    identity_payload: dict[str, object] = {
        "language": "JavaScript",
        "function": function,
        "file": normalized,
        "line": line,
        "column": column,
    }
    if disambiguator is not None:
        identity_payload["disambiguator"] = disambiguator
    frame_id = digest_model(identity_payload)
    return {
        "frame_id": frame_id,
        "language": "JavaScript",
        "function": function,
        "module": disambiguator,
        "file": normalized,
        "line": line,
        "column": column,
        "address": None,
        "build_id": None,
        "module_relative_address": None,
        "inline_chain_id": None,
        "source_state_id": None,
        "artifact_id": request.artifact_id,
        "inlined": False,
        "symbolization": (
            "complete"
            if function_value
            and normalized
            and normalized != "internal"
            and not normalized.startswith("node:")
            else "partial"
        ),
    }


def _check_rows(request: V8ProfileRequest, frame_count: int, measurement_count: int) -> None:
    if frame_count + measurement_count > request.max_rows:
        _limit("V8 profile normalized row limit exceeded.")


def _malformed(message: str) -> NoReturn:
    raise DomainError(ErrorCode.DECODE_FAILURE, message)


def _limit(message: str) -> NoReturn:
    raise DomainError(ErrorCode.LIMIT_EXCEEDED, message)


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=V8_PROFILE_WORKER,
            handler=lambda request, _context: (
                _parse_cpu(request) if request.profile_kind == "cpu" else _parse_heap(request)
            ),
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="V8 profile input is malformed",
            caught=(OSError, ValueError, TypeError, KeyError, ijson.common.JSONError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
