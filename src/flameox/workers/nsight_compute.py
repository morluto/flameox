from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any, cast

from flameox.domain import ErrorCode
from flameox.workers.protocol import run_worker

_NORMALIZED_MAX_COLLECTION_ITEMS = 8
_NORMALIZED_MAX_DEPTH = 5
_NORMALIZED_MAX_KEY_CHARS = 200
_NORMALIZED_MAX_NODES = 8
_NORMALIZED_MAX_STRING_CHARS = 256
_SOURCE_FILE_ID_CHARS = 200
_SOURCE_FILE_PATH_CHARS = 500


def _bounded_text(value: object, limit: int = 2_000) -> str:
    return str(value)[:limit]


def _bounded_items(
    values: Iterable[Any],
    *,
    limit: int,
    limitation: str,
    limitations: list[str],
) -> tuple[Any, ...]:
    items = tuple(islice(values, limit + 1))
    if len(items) > limit:
        limitations.append(limitation)
    return items[:limit]


def _normalized_text(
    value: object,
    *,
    limit: int,
    limitation: str,
    limitations: list[str],
) -> str:
    text = str(value)
    if len(text) > limit:
        limitations.append(limitation)
    return text[:limit]


def _safe_json(
    value: object,
    *,
    limitations: list[str],
    depth: int = 0,
    remaining_nodes: list[int] | None = None,
) -> object:
    if remaining_nodes is None:
        remaining_nodes = [_NORMALIZED_MAX_NODES]
    if remaining_nodes[0] <= 0:
        limitations.append(f"Nested provider values were bounded to {_NORMALIZED_MAX_NODES} nodes.")
        return "<truncated>"
    remaining_nodes[0] -= 1
    if depth >= _NORMALIZED_MAX_DEPTH:
        limitations.append(f"Nested provider values were bounded to depth {_NORMALIZED_MAX_DEPTH}.")
        return "<truncated>"
    if value is None or isinstance(value, bool | int | str):
        if not isinstance(value, str):
            return value
        return _normalized_text(
            value,
            limit=_NORMALIZED_MAX_STRING_CHARS,
            limitation=(
                "Nested provider strings were bounded to "
                f"{_NORMALIZED_MAX_STRING_CHARS} characters."
            ),
            limitations=limitations,
        )
    if isinstance(value, float):
        return (
            value
            if math.isfinite(value)
            else _normalized_text(
                value,
                limit=_NORMALIZED_MAX_STRING_CHARS,
                limitation=(
                    "Nested provider strings were bounded to "
                    f"{_NORMALIZED_MAX_STRING_CHARS} characters."
                ),
                limitations=limitations,
            )
        )
    if isinstance(value, Mapping):
        items = _bounded_items(
            value.items(),
            limit=_NORMALIZED_MAX_COLLECTION_ITEMS,
            limitation=(
                "Nested provider collections were bounded to "
                f"{_NORMALIZED_MAX_COLLECTION_ITEMS} entries."
            ),
            limitations=limitations,
        )
        normalized: dict[str, object] = {}
        for key, item in items:
            if remaining_nodes[0] <= 0:
                limitations.append(
                    f"Nested provider values were bounded to {_NORMALIZED_MAX_NODES} nodes."
                )
                break
            normalized_key = _normalized_text(
                key,
                limit=_NORMALIZED_MAX_KEY_CHARS,
                limitation=(
                    f"Nested provider keys were bounded to {_NORMALIZED_MAX_KEY_CHARS} characters."
                ),
                limitations=limitations,
            )
            normalized[normalized_key] = _safe_json(
                item,
                limitations=limitations,
                depth=depth + 1,
                remaining_nodes=remaining_nodes,
            )
        return normalized
    if isinstance(value, list | tuple):
        items = _bounded_items(
            value,
            limit=_NORMALIZED_MAX_COLLECTION_ITEMS,
            limitation=(
                "Nested provider collections were bounded to "
                f"{_NORMALIZED_MAX_COLLECTION_ITEMS} entries."
            ),
            limitations=limitations,
        )
    try:
        if not isinstance(value, list | tuple):
            items = _bounded_items(
                cast(Iterable[object], value),
                limit=_NORMALIZED_MAX_COLLECTION_ITEMS,
                limitation=(
                    "Nested provider collections were bounded to "
                    f"{_NORMALIZED_MAX_COLLECTION_ITEMS} entries."
                ),
                limitations=limitations,
            )
    except (TypeError, ValueError):
        return _normalized_text(
            value,
            limit=_NORMALIZED_MAX_STRING_CHARS,
            limitation=(
                "Nested provider strings were bounded to "
                f"{_NORMALIZED_MAX_STRING_CHARS} characters."
            ),
            limitations=limitations,
        )
    normalized_items: list[object] = []
    for item in items:
        if remaining_nodes[0] <= 0:
            limitations.append(
                f"Nested provider values were bounded to {_NORMALIZED_MAX_NODES} nodes."
            )
            break
        normalized_items.append(
            _safe_json(
                item,
                limitations=limitations,
                depth=depth + 1,
                remaining_nodes=remaining_nodes,
            )
        )
    return normalized_items


def _source_file_pairs(value: object, *, limitations: list[str]) -> list[dict[str, str]]:
    if isinstance(value, Mapping):
        raw_pairs: Iterable[tuple[object, object]] = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        raw_pairs = enumerate(value)
    else:
        limitations.append("Official ncu_report source_files returned an unsupported collection.")
        return []
    pairs = _bounded_items(
        raw_pairs,
        limit=_NORMALIZED_MAX_COLLECTION_ITEMS,
        limitation=(
            "Source files were bounded to "
            f"{_NORMALIZED_MAX_COLLECTION_ITEMS} identifier/path pairs."
        ),
        limitations=limitations,
    )
    return [
        {
            "id": _normalized_text(
                identifier,
                limit=_SOURCE_FILE_ID_CHARS,
                limitation=(
                    f"Source file identifiers were bounded to {_SOURCE_FILE_ID_CHARS} characters."
                ),
                limitations=limitations,
            ),
            "path": _normalized_text(
                path,
                limit=_SOURCE_FILE_PATH_CHARS,
                limitation=(
                    f"Source file paths were bounded to {_SOURCE_FILE_PATH_CHARS} characters."
                ),
                limitations=limitations,
            ),
        }
        for identifier, path in pairs
    ]


def _call(
    value: object,
    name: str,
    default: object = None,
    *,
    limitations: list[str] | None = None,
) -> Any:
    method = getattr(value, name, None)
    if not callable(method):
        return default
    try:
        return method()
    except Exception as exc:  # Official bindings can throw SWIG-specific exception classes.
        if limitations is not None:
            limitations.append(
                f"Official ncu_report access {name} failed with {type(exc).__name__}."
            )
        return default


def _metric_value(
    metric: object,
    *,
    limitations: list[str],
) -> tuple[str, int | float | str | None]:
    has_value = _call(metric, "has_value", True, limitations=limitations)
    if has_value is False:
        return "missing", None
    kind = _call(metric, "kind", limitations=limitations)
    if kind in {
        getattr(metric, "ValueKind_UINT32", object()),
        getattr(metric, "ValueKind_UINT64", object()),
    }:
        value = _call(metric, "as_uint64", limitations=limitations)
        return (
            ("integer", value)
            if isinstance(value, int) and not isinstance(value, bool)
            else (
                "unknown",
                None,
            )
        )
    if kind in {
        getattr(metric, "ValueKind_FLOAT", object()),
        getattr(metric, "ValueKind_DOUBLE", object()),
    }:
        value = _call(metric, "as_double", limitations=limitations)
        return (
            ("float", value)
            if isinstance(value, int | float) and math.isfinite(value)
            else (
                "unknown",
                None,
            )
        )
    if kind == getattr(metric, "ValueKind_STRING", object()):
        value = _call(metric, "as_string", limitations=limitations)
        return ("string", _bounded_text(value)) if value is not None else ("missing", None)
    value = _call(metric, "value", limitations=limitations)
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer", value
    if isinstance(value, float) and math.isfinite(value):
        return "float", value
    if isinstance(value, str):
        return "string", value[:2_000]
    return "unknown", None


def _extract(  # noqa: C901 - bounded traversal mirrors the official report hierarchy
    report_path: Path,
    *,
    interface_path: Path,
    max_ranges: int,
    max_actions: int,
    max_metrics: int,
    max_observations: int,
) -> dict[str, object]:
    if interface_path.name == "ncu_report.py":
        sys.path.insert(0, str(interface_path.parent))
    else:
        sys.path.insert(0, str(interface_path))
    module = importlib.import_module("ncu_report")
    report = module.load_report(str(report_path))
    if report is None:
        raise ValueError("ncu_report.load_report returned no report")

    measurements: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    limitations: list[str] = []
    section_ids: set[str] = set()
    metric_ids: set[str] = set()
    roofline_present = False
    range_total = int(report.num_ranges())
    action_seen = 0
    metric_seen = 0
    actions_truncated = False

    bounded_range_total = min(range_total, max_ranges)
    for range_index in range(bounded_range_total):
        selected_range = report.range_by_idx(range_index)
        action_total = int(selected_range.num_actions())
        range_name = _call(
            selected_range,
            "name",
            f"range-{range_index}",
            limitations=limitations,
        )
        if len(observations) < max_observations:
            observations.append(
                {
                    "kind": "profile.range",
                    "name": _bounded_text(range_name, 500),
                    "value": {"range_index": range_index, "action_count": action_total},
                }
            )
        action_budget = max_actions - action_seen
        actions_in_range = min(action_total, action_budget)
        for action_index in range(actions_in_range):
            action = selected_range.action_by_idx(action_index)
            action_seen += 1
            action_name = _bounded_text(action.name(), 500)
            workload_type = _call(action, "workload_type", limitations=limitations)
            action_identity: dict[str, object] = {}
            action_value = {
                "range_index": range_index,
                "range_name": _bounded_text(range_name, 500),
                "action_index": action_index,
                "workload_type": _safe_json(workload_type, limitations=limitations),
                "identity": action_identity,
            }
            if len(observations) < max_observations:
                observations.append(
                    {
                        "kind": "profile.action",
                        "name": action_name,
                        "value": action_value,
                    }
                )

            remaining_metrics = max_metrics - metric_seen
            raw_metric_names = action.metric_names() or ()
            metric_names = tuple(islice(raw_metric_names, remaining_metrics + 1))
            for raw_metric_name in metric_names[:remaining_metrics]:
                lookup_name = str(raw_metric_name)
                metric_name = _bounded_text(lookup_name, 500)
                metric = action.metric_by_name(lookup_name)
                value_kind, value = _metric_value(metric, limitations=limitations)
                metric_seen += 1
                metric_ids.add(metric_name)
                unit = _bounded_text(
                    _call(metric, "unit", "", limitations=limitations) or "unknown", 100
                )
                lower_identity = metric_name.casefold()
                roofline_present = roofline_present or "roofline" in lower_identity
                if (
                    any(
                        marker in lower_identity
                        for marker in (
                            "device__attribute",
                            "context_id",
                            "stream_id",
                            "process_id",
                            "launch__grid",
                            "launch__block",
                        )
                    )
                    and value is not None
                ):
                    if len(action_identity) < 8:
                        action_identity[metric_name] = _safe_json(
                            value,
                            limitations=limitations,
                        )
                    else:
                        limitations.append("Action identity was bounded to 8 entries.")
                common = {
                    "range_index": range_index,
                    "action_index": action_index,
                    "action_name": action_name,
                    "name": metric_name[:500],
                    "unit": unit,
                    "description": _bounded_text(
                        _call(metric, "description", "", limitations=limitations), 2_000
                    ),
                    "metric_type": _safe_json(
                        _call(metric, "metric_type", limitations=limitations),
                        limitations=limitations,
                    ),
                    "metric_subtype": _safe_json(
                        _call(metric, "metric_subtype", limitations=limitations),
                        limitations=limitations,
                    ),
                    "rollup_operation": _safe_json(
                        _call(metric, "rollup_operation", limitations=limitations),
                        limitations=limitations,
                    ),
                    "roofline": "roofline" in lower_identity,
                }
                if value_kind in {"integer", "float"} and isinstance(value, int | float):
                    measurements.append({**common, "value_kind": value_kind, "value": value})
                elif value_kind == "string" and len(observations) < max_observations:
                    observations.append(
                        {
                            "kind": "profile.attribute",
                            "name": metric_name[:500],
                            "value": {**common, "value": value},
                        }
                    )
                elif value_kind == "unknown":
                    limitations.append(f"Unsupported metric value type: {metric_name[:200]}.")
            if len(metric_names) > remaining_metrics:
                limitations.append(f"Metrics were truncated to {max_metrics} entries.")

            rules = _call(action, "rule_results_as_dicts", (), limitations=limitations) or ()
            rule_budget = max(0, max_observations - len(observations))
            bounded_rules = tuple(islice(cast(Iterable[object], rules), rule_budget + 1))
            for rule in bounded_rules[:rule_budget]:
                normalized = _safe_json(rule, limitations=limitations)
                if isinstance(normalized, dict):
                    section = normalized.get("section_identifier")
                    if isinstance(section, str):
                        section = section[:500]
                        section_ids.add(section)
                        roofline_present = roofline_present or "roofline" in section.casefold()
                    identifier = normalized.get("rule_identifier")
                    name = identifier[:500] if isinstance(identifier, str) else "rule"
                else:
                    name = "rule"
                observations.append({"kind": "profile.rule", "name": name, "value": normalized})
            if len(bounded_rules) > rule_budget:
                limitations.append("Rule results were truncated by the observation budget.")

            source_files = _call(action, "source_files", {}, limitations=limitations) or {}
            source_file_pairs = _source_file_pairs(source_files, limitations=limitations)
            if source_file_pairs:
                if len(observations) < max_observations:
                    observations.append(
                        {
                            "kind": "profile.source_files",
                            "name": action_name,
                            # Preserve identities only. Source bodies remain in the native report
                            # and are not imported into normalized evidence.
                            "value": {"files": source_file_pairs},
                        }
                    )
                else:
                    limitations.append("Source files were truncated by the observation budget.")
            markers = _call(action, "source_markers", (), limitations=limitations) or ()
            try:
                marker_iterator = iter(cast(Iterable[object], markers))
            except TypeError:
                limitations.append(
                    "Official ncu_report source_markers returned an unsupported collection."
                )
                marker_iterator = iter(())
            marker_exhausted = False
            marker_details_truncated = False
            while len(observations) < max_observations:
                try:
                    marker = next(marker_iterator)
                except StopIteration:
                    marker_exhausted = True
                    break
                normalized_marker = _safe_json(marker, limitations=limitations)
                observations.append(
                    {
                        "kind": "profile.source_reference",
                        "name": action_name,
                        "value": normalized_marker,
                    }
                )
                if not isinstance(marker, Mapping):
                    continue
                address = marker.get("source_address")
                if not isinstance(address, int):
                    continue
                if len(observations) >= max_observations:
                    marker_details_truncated = True
                    break
                reference = {
                    "address": address,
                    "source": _safe_json(
                        _call_with_arg(action, "source_info", address, limitations=limitations),
                        limitations=limitations,
                    ),
                    "sass": _safe_json(
                        _call_with_arg(action, "sass_by_pc", address, limitations=limitations),
                        limitations=limitations,
                    ),
                    "ptx": _safe_json(
                        _call_with_arg(action, "ptx_by_pc", address, limitations=limitations),
                        limitations=limitations,
                    ),
                }
                observations.append(
                    {
                        "kind": "profile.source_sass_reference",
                        "name": action_name,
                        "value": reference,
                    }
                )
            if marker_details_truncated:
                limitations.append(
                    "Source marker source/SASS/PTX details were truncated by the "
                    "observation budget."
                )
            if not marker_exhausted:
                try:
                    next(marker_iterator)
                except StopIteration:
                    pass
                else:
                    limitations.append("Source markers were truncated by the observation budget.")
        if actions_in_range < action_total or (
            action_seen >= max_actions and range_index + 1 < bounded_range_total
        ):
            actions_truncated = True
        if action_seen >= max_actions:
            break

    if range_total > max_ranges:
        limitations.append(f"Ranges were truncated to {max_ranges} entries.")
    if actions_truncated:
        limitations.append(f"Actions were bounded to {max_actions} entries.")
    if len(observations) >= max_observations:
        limitations.append(f"Observations were bounded to {max_observations} entries.")
    return {
        "ok": True,
        "report_version": _bounded_text(report.get_version(), 200),
        "measurements": measurements,
        "observations": observations,
        "metric_ids": sorted(metric_ids),
        "section_ids": sorted(section_ids),
        "range_count": min(range_total, max_ranges),
        "action_count": action_seen,
        "roofline_present": roofline_present,
        "limitations": list(dict.fromkeys(limitations)),
    }


def _call_with_arg(
    value: object,
    name: str,
    argument: object,
    *,
    limitations: list[str] | None = None,
) -> Any:
    method = getattr(value, name, None)
    if not callable(method):
        return None
    try:
        return method(argument)
    except Exception as exc:  # Official bindings can throw SWIG-specific exception classes.
        if limitations is not None:
            limitations.append(
                f"Official ncu_report access {name} failed with {type(exc).__name__}."
            )
        return None


def main() -> int:
    def handle(request: dict[str, object], _request_path: Path) -> dict[str, object]:
        return _extract(
            Path(str(request["artifact_path"])),
            interface_path=Path(str(request["interface_path"])),
            max_ranges=int(str(request["max_ranges"])),
            max_actions=int(str(request["max_actions"])),
            max_metrics=int(str(request["max_metrics"])),
            max_observations=int(str(request["max_observations"])),
        )

    return run_worker(
        handle,
        invalid_code=ErrorCode.ARTIFACT_PARSE_FAILED,
        invalid_message="Nsight Compute report extraction failed",
        caught=(Exception,),
    )


if __name__ == "__main__":
    raise SystemExit(main())
