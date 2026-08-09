from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any, cast


def _bounded_text(value: object, limit: int = 2_000) -> str:
    return str(value)[:limit]


def _safe_json(value: object, *, depth: int = 0) -> object:
    if depth >= 5:
        return _bounded_text(value)
    if value is None or isinstance(value, bool | int | str):
        return value if not isinstance(value, str) else value[:2_000]
    if isinstance(value, float):
        return value if math.isfinite(value) else _bounded_text(value)
    if isinstance(value, dict):
        return {
            _bounded_text(key, 200): _safe_json(item, depth=depth + 1)
            for key, item in islice(value.items(), 100)
        }
    if isinstance(value, list | tuple):
        return [_safe_json(item, depth=depth + 1) for item in value[:100]]
    try:
        return [
            _safe_json(item, depth=depth + 1) for item in islice(cast(Iterable[object], value), 100)
        ]
    except (TypeError, ValueError):
        return _bounded_text(value)


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
    total_actions = 0
    metric_seen = 0

    for range_index in range(min(range_total, max_ranges)):
        selected_range = report.range_by_idx(range_index)
        action_total = int(selected_range.num_actions())
        total_actions += action_total
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
        for action_index in range(min(action_total, max_actions - action_seen)):
            action = selected_range.action_by_idx(action_index)
            action_seen += 1
            action_name = _bounded_text(action.name(), 500)
            workload_type = _call(action, "workload_type", limitations=limitations)
            action_identity: dict[str, object] = {}
            action_value = {
                "range_index": range_index,
                "range_name": _bounded_text(range_name, 500),
                "action_index": action_index,
                "workload_type": _safe_json(workload_type),
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
            metric_names = tuple(
                str(item) for item in islice(raw_metric_names, remaining_metrics + 1)
            )
            for metric_name in metric_names[:remaining_metrics]:
                metric = action.metric_by_name(metric_name)
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
                    action_identity[metric_name[:500]] = value
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
                        _call(metric, "metric_type", limitations=limitations)
                    ),
                    "metric_subtype": _safe_json(
                        _call(metric, "metric_subtype", limitations=limitations)
                    ),
                    "rollup_operation": _safe_json(
                        _call(metric, "rollup_operation", limitations=limitations)
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
                normalized = _safe_json(rule)
                if isinstance(normalized, dict):
                    section = normalized.get("section_identifier")
                    if isinstance(section, str):
                        section_ids.add(section)
                        roofline_present = roofline_present or "roofline" in section.casefold()
                    identifier = normalized.get("rule_identifier")
                    name = identifier if isinstance(identifier, str) else "rule"
                else:
                    name = "rule"
                observations.append({"kind": "profile.rule", "name": name, "value": normalized})
            if len(bounded_rules) > rule_budget:
                limitations.append("Rule results were truncated by the observation budget.")

            source_files = _call(action, "source_files", {}, limitations=limitations) or {}
            try:
                source_names = [
                    _bounded_text(item, 500)
                    for item in islice(cast(Iterable[object], source_files), 100)
                ]
            except TypeError:
                source_names = []
            if source_names and len(observations) < max_observations:
                observations.append(
                    {
                        "kind": "profile.source_files",
                        "name": action_name,
                        # Preserve identities only. Source bodies remain in the native report and
                        # are not imported into normalized evidence.
                        "value": {"files": source_names},
                    }
                )
            markers = _call(action, "source_markers", (), limitations=limitations) or ()
            marker_budget = max(0, max_observations - len(observations))
            for marker in islice(cast(Iterable[object], markers), marker_budget):
                normalized_marker = _safe_json(marker)
                observations.append(
                    {
                        "kind": "profile.source_reference",
                        "name": action_name,
                        "value": normalized_marker,
                    }
                )
                if not isinstance(marker, dict):
                    continue
                address = marker.get("source_address")
                if not isinstance(address, int) or len(observations) >= max_observations:
                    continue
                reference = {
                    "address": address,
                    "source": _safe_json(
                        _call_with_arg(action, "source_info", address, limitations=limitations)
                    ),
                    "sass": _safe_json(
                        _call_with_arg(action, "sass_by_pc", address, limitations=limitations)
                    ),
                    "ptx": _safe_json(
                        _call_with_arg(action, "ptx_by_pc", address, limitations=limitations)
                    ),
                }
                observations.append(
                    {
                        "kind": "profile.source_sass_reference",
                        "name": action_name,
                        "value": reference,
                    }
                )
        if action_seen >= max_actions:
            break

    if range_total > max_ranges:
        limitations.append(f"Ranges were truncated to {max_ranges} entries.")
    if total_actions > max_actions:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        response = _extract(
            Path(request["artifact_path"]),
            interface_path=Path(request["interface_path"]),
            max_ranges=int(request["max_ranges"]),
            max_actions=int(request["max_actions"]),
            max_metrics=int(request["max_metrics"]),
            max_observations=int(request["max_observations"]),
        )
    except Exception as exc:
        response = {
            "ok": False,
            "code": "ARTIFACT_PARSE_FAILED",
            "message": f"Nsight Compute report extraction failed: {type(exc).__name__}: {exc}",
        }
    temporary = args.response.with_suffix(".tmp")
    temporary.write_text(json.dumps(response, allow_nan=False, sort_keys=True), encoding="utf-8")
    temporary.replace(args.response)
    # The response envelope, not the process status, carries parser failures so the
    # parent can preserve the bounded provider diagnostic.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
