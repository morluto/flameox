from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from flameox.nsight_compute import (
    NsightComputeFocusMetric,
    NsightComputeProviderRuleFact,
    NsightComputeReportLocation,
    NsightComputeRuleMessage,
    NsightComputeSpeedupEstimation,
    NsightComputeSpeedupMeaning,
)
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.workers.nsight_compute_contract import (
    NSIGHT_COMPUTE_WORKER,
    NsightComputeWorkerRequest,
    NsightComputeWorkerResult,
)
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerFailureKind,
    run_typed_worker,
)

_NORMALIZED_MAX_COLLECTION_ITEMS = 8
_NORMALIZED_MAX_DEPTH = 5
_NORMALIZED_MAX_KEY_CHARS = 200
_NORMALIZED_MAX_NODES = 8
_NORMALIZED_MAX_STRING_CHARS = 256
_SOURCE_FILE_ID_CHARS = 200
_SOURCE_FILE_PATH_CHARS = 500
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_RULE_MESSAGE_CHARS = 8_000
_RULE_TITLE_CHARS = 2_000
_RULE_TYPE_CHARS = 100
_RULE_FOCUS_METRIC_LIMIT = 32
_RULE_FOCUS_INFORMATION_CHARS = 2_000


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
    uint32_kind = getattr(metric, "ValueKind_UINT32", object())
    uint64_kind = getattr(metric, "ValueKind_UINT64", object())
    if kind in {uint32_kind, uint64_kind}:
        value = _call(metric, "as_uint64", limitations=limitations)
        maximum = _UINT32_MAX if kind == uint32_kind else _UINT64_MAX
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            raise DomainError(
                ErrorCode.UNSUPPORTED_FORMAT,
                "Official ncu_report returned a value outside its declared unsigned range.",
            )
        return ("uint32" if kind == uint32_kind else "uint64"), value
    float_kind = getattr(metric, "ValueKind_FLOAT", object())
    double_kind = getattr(metric, "ValueKind_DOUBLE", object())
    if kind in {float_kind, double_kind}:
        value = _call(metric, "as_double", limitations=limitations)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise DomainError(
                ErrorCode.UNSUPPORTED_FORMAT,
                "Official ncu_report returned an invalid floating-point metric value.",
            )
        return ("float32" if kind == float_kind else "float64"), float(value)
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


def _provider_rule_speedup_meaning(
    value: object,
) -> tuple[NsightComputeSpeedupMeaning, str | None]:
    """Classify only provider labels whose global/local meaning is explicit."""
    if value is None:
        return "unknown", None
    provider_type = _bounded_text(value, _RULE_TYPE_CHARS)
    folded = provider_type.casefold()
    if "global" in folded:
        return "global_runtime_reduction", provider_type
    if "local" in folded:
        return "local_hardware_efficiency_increase", provider_type
    return "unknown", provider_type


def _rule_message(value: object, *, limitations: list[str]) -> NsightComputeRuleMessage | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        limitations.append("ncu_report rule_results_as_dicts returned an invalid rule_message.")
        return None
    message = value.get("message")
    if not isinstance(message, str) or not message:
        limitations.append("ncu_report rule_results_as_dicts rule_message omitted its text.")
        return None
    title = value.get("title")
    provider_type = value.get("type")
    return NsightComputeRuleMessage(
        title=(
            _normalized_text(
                title,
                limit=_RULE_TITLE_CHARS,
                limitation=(f"Rule message titles were bounded to {_RULE_TITLE_CHARS} characters."),
                limitations=limitations,
            )
            if isinstance(title, str) and title
            else None
        ),
        message=_normalized_text(
            message,
            limit=_RULE_MESSAGE_CHARS,
            limitation=f"Rule messages were bounded to {_RULE_MESSAGE_CHARS} characters.",
            limitations=limitations,
        ),
        provider_type=(
            _normalized_text(
                provider_type,
                limit=_RULE_TYPE_CHARS,
                limitation=f"Rule message types were bounded to {_RULE_TYPE_CHARS} characters.",
                limitations=limitations,
            )
            if provider_type is not None
            else None
        ),
    )


def _rule_speedup(
    value: object,
    *,
    limitations: list[str],
) -> NsightComputeSpeedupEstimation | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        limitations.append(
            "ncu_report rule_results_as_dicts returned an invalid speedup_estimation."
        )
        return None
    speedup = value.get("speedup")
    if (
        isinstance(speedup, bool)
        or not isinstance(speedup, int | float)
        or not math.isfinite(speedup)
    ):
        limitations.append(
            "ncu_report rule_results_as_dicts speedup_estimation returned an invalid speedup."
        )
        return None
    if speedup < 0:
        limitations.append(
            "ncu_report rule_results_as_dicts speedup_estimation returned a negative speedup."
        )
        return None
    meaning, provider_type = _provider_rule_speedup_meaning(value.get("type"))
    return NsightComputeSpeedupEstimation(
        estimated_speedup=float(speedup),
        meaning=meaning,
        provider_type=provider_type,
    )


def _rule_focus_metrics(
    values: object,
    *,
    limitations: list[str],
) -> tuple[NsightComputeFocusMetric, ...]:
    if values is None:
        return ()
    try:
        iterator = iter(cast(Iterable[object], values))
    except TypeError:
        limitations.append(
            "ncu_report rule_results_as_dicts returned an invalid focus_metrics collection."
        )
        return ()
    bounded = _bounded_items(
        iterator,
        limit=_RULE_FOCUS_METRIC_LIMIT,
        limitation=(f"Rule focus metrics were bounded to {_RULE_FOCUS_METRIC_LIMIT} entries."),
        limitations=limitations,
    )
    facts: list[NsightComputeFocusMetric] = []
    for value in bounded:
        if not isinstance(value, Mapping):
            limitations.append(
                "ncu_report rule_results_as_dicts returned a non-mapping focus metric."
            )
            continue
        name = value.get("name")
        if not isinstance(name, str) or not name:
            limitations.append("ncu_report rule_results_as_dicts focus metric omitted its name.")
            continue
        numeric = value.get("value")
        if (
            isinstance(numeric, bool)
            or not isinstance(numeric, int | float)
            or not math.isfinite(numeric)
        ):
            numeric_value = None
            if numeric is not None:
                limitations.append(
                    "ncu_report rule_results_as_dicts focus metric has a non-finite value."
                )
        else:
            numeric_value = float(numeric)
        severity = value.get("severity")
        information = value.get("info")
        facts.append(
            NsightComputeFocusMetric(
                name=_normalized_text(
                    name,
                    limit=500,
                    limitation="Rule focus-metric names were bounded to 500 characters.",
                    limitations=limitations,
                ),
                value=numeric_value,
                severity=(
                    _normalized_text(
                        severity,
                        limit=_RULE_TYPE_CHARS,
                        limitation=(
                            "Rule focus-metric severities were bounded to "
                            f"{_RULE_TYPE_CHARS} characters."
                        ),
                        limitations=limitations,
                    )
                    if severity is not None
                    else None
                ),
                info=(
                    _normalized_text(
                        information,
                        limit=_RULE_FOCUS_INFORMATION_CHARS,
                        limitation=(
                            "Rule focus-metric information was bounded to "
                            f"{_RULE_FOCUS_INFORMATION_CHARS} characters."
                        ),
                        limitations=limitations,
                    )
                    if information is not None
                    else None
                ),
            )
        )
    return tuple(facts)


def _rule_fact(
    rule: object,
    *,
    range_index: int,
    action_index: int,
    action_name: str,
    limitations: list[str],
) -> NsightComputeProviderRuleFact | None:
    if not isinstance(rule, Mapping):
        limitations.append("ncu_report rule_results_as_dicts returned a non-mapping rule.")
        return None
    identifier = rule.get("rule_identifier")
    section = _provider_rule_section_identifier(rule, limitations=limitations)
    if not isinstance(identifier, str) or not identifier:
        limitations.append("ncu_report rule_results_as_dicts rule omitted rule_identifier.")
        return None
    if section is None:
        limitations.append("ncu_report rule_results_as_dicts rule omitted section_identifier.")
        return None
    return NsightComputeProviderRuleFact(
        location=NsightComputeReportLocation(
            range_index=range_index,
            action_index=action_index,
            action_name=action_name,
        ),
        rule_identifier=_normalized_text(
            identifier,
            limit=500,
            limitation="Rule identifiers were bounded to 500 characters.",
            limitations=limitations,
        ),
        section_identifier=section,
        rule_message=_rule_message(rule.get("rule_message"), limitations=limitations),
        speedup_estimation=_rule_speedup(
            rule.get("speedup_estimation"),
            limitations=limitations,
        ),
        focus_metrics=_rule_focus_metrics(rule.get("focus_metrics"), limitations=limitations),
    )


def _provider_rule_section_identifier(
    rule: object,
    *,
    limitations: list[str],
) -> str | None:
    """Read the documented raw section field independently of fact publication."""
    if not isinstance(rule, Mapping):
        return None
    section = rule.get("section_identifier")
    if not isinstance(section, str) or not section:
        return None
    return _normalized_text(
        section,
        limit=500,
        limitation="Rule section identifiers were bounded to 500 characters.",
        limitations=limitations,
    )


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
    range_total = int(report.num_ranges())
    action_seen = 0
    metric_seen = 0
    actions_truncated = False
    metrics_truncated = False
    rules_truncated = False

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
                if (
                    any(
                        marker in metric_name.casefold()
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
                }
                if value_kind in {
                    "integer",
                    "float",
                    "float32",
                    "float64",
                    "uint32",
                    "uint64",
                } and isinstance(value, int | float):
                    measurements.append(
                        {**common, "provider_value_kind": value_kind, "value": value}
                    )
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
                metrics_truncated = True
                limitations.append(f"Metrics were truncated to {max_metrics} entries.")

            rules = _call(action, "rule_results_as_dicts", (), limitations=limitations) or ()
            rule_budget = max(0, max_observations - len(observations))
            try:
                rule_iterator = iter(cast(Iterable[object], rules))
            except TypeError:
                limitations.append(
                    "ncu_report rule_results_as_dicts returned an unsupported collection."
                )
                rule_iterator = iter(())
            bounded_rules = tuple(islice(rule_iterator, rule_budget + 1))
            for rule in bounded_rules:
                section_identifier = _provider_rule_section_identifier(
                    rule,
                    limitations=limitations,
                )
                if section_identifier is not None:
                    section_ids.add(section_identifier)
            for rule in bounded_rules[:rule_budget]:
                fact = _rule_fact(
                    rule,
                    range_index=range_index,
                    action_index=action_index,
                    action_name=action_name,
                    limitations=limitations,
                )
                if fact is None:
                    continue
                observations.append(
                    {
                        "kind": "nsight_compute.rule",
                        "name": fact.rule_identifier,
                        "value": fact.model_dump(mode="json"),
                    }
                )
            if len(bounded_rules) > rule_budget:
                rules_truncated = True
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
        "truncated": any(
            (
                range_total > max_ranges,
                actions_truncated,
                metrics_truncated,
                rules_truncated,
                len(observations) >= max_observations,
            )
        ),
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
    def handle(
        request: NsightComputeWorkerRequest,
        _job_root: Path,
    ) -> NsightComputeWorkerResult:
        result = _extract(
            Path(request.artifact_path),
            interface_path=Path(request.interface_path),
            max_ranges=request.max_ranges,
            max_actions=request.max_actions,
            max_metrics=request.max_metrics,
            max_observations=request.max_observations,
        )
        return NsightComputeWorkerResult(
            report_version=cast(str, result["report_version"]),
            measurements=cast(
                tuple[dict[str, JsonValue], ...],
                tuple(cast(list[dict[str, object]], result["measurements"])),
            ),
            observations=cast(
                tuple[dict[str, JsonValue], ...],
                tuple(cast(list[dict[str, object]], result["observations"])),
            ),
            metric_ids=tuple(cast(list[str], result["metric_ids"])),
            section_ids=tuple(cast(list[str], result["section_ids"])),
            range_count=cast(int, result["range_count"]),
            action_count=cast(int, result["action_count"]),
            truncated=cast(bool, result["truncated"]),
            limitations=tuple(cast(list[str], result["limitations"])),
        )

    return run_typed_worker(
        WorkerApplication(
            definition=NSIGHT_COMPUTE_WORKER,
            handler=handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="Nsight Compute report extraction failed",
            caught=(Exception,),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
