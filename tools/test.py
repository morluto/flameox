#!/usr/bin/env python3
"""Run Flameox's explicitly owned pytest lanes.

The runner is intentionally small: ownership lives in tests/ownership.toml,
pytest remains the test framework, and no command retries a failed case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNERSHIP_PATH = ROOT / "tests" / "ownership.toml"
COLLECTION_BASELINE_PATH = ROOT / "tests" / "collection-baseline.toml"
RESULTS_DIR = ROOT / ".test-results"
PLAN_CONTRACT = "flameox.affected-plan"
PLAN_SCHEMA_VERSION = 3
PLANNER_VERSION = "affected-plan-v3"
PLAN_MAX_MATRIX_ITEMS = 256
SUPPORTED_EVENTS = (
    "local",
    "pull_request",
    "merge_group",
    "push",
    "schedule",
    "workflow_dispatch",
)
VALID_EVENTS = frozenset(SUPPORTED_EVENTS)
FULL_RUN_EVENTS = frozenset({"merge_group", "schedule", "workflow_dispatch"})
PLAN_KEYS = {
    "schema_version",
    "contract",
    "planner_version",
    "event",
    "base_revision",
    "head_revision",
    "base_available",
    "head_available",
    "changed_paths",
    "planner_digest",
    "topology_digest",
    "fallback_reason",
    "provenance",
    "full",
    "selected_lanes",
    "unselected_lanes",
    "matrix",
    "lanes",
    "optional_lanes",
    "run_quality",
    "run_deep_checks",
    "run_coverage",
    "run_collection",
    "run_npm",
    "run_performance",
}
PROVIDER_LANES = {
    "optional-compute-sanitizer": "optional and requires_compute_sanitizer",
    "optional-coverage": "optional and requires_coverage",
    "optional-memray": "optional and requires_memray",
    "optional-perfetto": "optional and requires_perfetto",
    "optional-pyspy": "optional and requires_pyspy",
    "optional-torch": "optional and requires_torch",
    "optional-host": (
        "optional and not requires_compute_sanitizer and not requires_coverage "
        "and not requires_memray "
        "and not requires_perfetto and not requires_pyspy and not requires_torch"
    ),
}
GPU_PROVIDER_LANES = frozenset({"optional-compute-sanitizer"})
# GitHub-hosted runners do not provide CUDA GPUs. Keep the live lane available
# as an explicit local command without emitting a job that can only skip.
CI_PROVIDER_LANES = {
    lane: expression
    for lane, expression in PROVIDER_LANES.items()
    if lane not in GPU_PROVIDER_LANES
}
TEST_LANES = (
    "core",
    "storage",
    "application",
    "analysis",
    "mcp",
    "process",
    "adapters",
    "cli",
    "security",
    "golden",
)
# These are the non-optional, non-performance paths previously covered by the
# repository-wide core invocation. Each is now executed once as its owned lane.
COVERAGE_LANES = (
    "core",
    "storage",
    "application",
    "analysis",
    "mcp",
    "adapters",
    "cli",
    "golden",
)
AGGREGATE_LANES = {"optional", "performance", "full"} | set(PROVIDER_LANES)
FULL_CHANGE_PATHS = {
    "AGENTS.md",
    "pyproject.toml",
    "uv.lock",
    "tests/conftest.py",
    "tests/ownership.toml",
    "tests/collection-baseline.toml",
}
ALL_PLANNED_LANES = (*TEST_LANES, "performance", *CI_PROVIDER_LANES)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support.ownership import Ownership, load_ownership  # noqa: E402
from tests.support.providers import PROVIDER_MARKERS, provider_inventory  # noqa: E402


class PlanValidationError(ValueError):
    """Raised when an affected-plan document violates its public contract."""


def _digest_document(document: object) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def planner_digest() -> str:
    """Bind the plan to the exact checked-in planner implementation."""
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def topology_digest(records: tuple[Ownership, ...]) -> str:
    """Bind the plan to the lane and ownership topology used to produce it."""
    return _digest_document(
        {
            "test_lanes": list(TEST_LANES),
            "coverage_lanes": list(COVERAGE_LANES),
            "provider_lanes": CI_PROVIDER_LANES,
            "full_change_paths": sorted(FULL_CHANGE_PATHS),
            "ownership": [
                {
                    "owner": record.owner,
                    "lane": record.lane,
                    "paths": list(record.paths),
                    "markers": list(record.markers),
                }
                for record in records
            ],
        }
    )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PlanValidationError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanValidationError(f"{field} must be a non-empty string")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise PlanValidationError(f"{field} must be a boolean")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PlanValidationError(f"{field} must be an array of strings")
    result = list(value)
    if len(result) != len(set(result)):
        raise PlanValidationError(f"{field} contains duplicates")
    return result


def _digest(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise PlanValidationError(f"{field} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in digest[7:]):
        raise PlanValidationError(f"{field} must be a lowercase sha256 digest")
    return digest


def _lane_decisions(value: object, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > len(ALL_PLANNED_LANES):
        raise PlanValidationError(f"{field} exceeds the lane bound")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, f"{field}[{index}]")
        if set(item) != {"lane", "reason"}:
            raise PlanValidationError(f"{field}[{index}] must contain lane and reason")
        lane = _string(item.get("lane"), f"{field}[{index}].lane")
        if lane not in ALL_PLANNED_LANES:
            raise PlanValidationError(f"{field} contains unknown lane {lane!r}")
        if lane in seen:
            raise PlanValidationError(f"{field} contains duplicate lane {lane!r}")
        seen.add(lane)
        result.append(
            {"lane": lane, "reason": _string(item.get("reason"), f"{field}[{index}].reason")}
        )
    return result


def _matrix(value: object) -> dict[str, list[dict[str, str]]]:
    matrix = _mapping(value, "matrix")
    if set(matrix) != {"lanes", "optional_lanes"}:
        raise PlanValidationError("matrix must contain lanes and optional_lanes")
    result: dict[str, list[dict[str, str]]] = {}
    for field in ("lanes", "optional_lanes"):
        raw_items = matrix[field]
        if not isinstance(raw_items, list) or len(raw_items) > PLAN_MAX_MATRIX_ITEMS:
            raise PlanValidationError(f"matrix.{field} exceeds the provider bound")
        allowed = set(TEST_LANES) if field == "lanes" else set(CI_PROVIDER_LANES)
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_items):
            item = _mapping(raw, f"matrix.{field}[{index}]")
            if set(item) != {"lane"}:
                raise PlanValidationError(f"matrix.{field}[{index}] must contain lane")
            lane = _string(item.get("lane"), f"matrix.{field}[{index}].lane")
            if lane not in allowed or lane in seen:
                raise PlanValidationError(
                    f"matrix.{field} contains invalid or duplicate lane {lane!r}"
                )
            seen.add(lane)
            items.append({"lane": lane})
        result[field] = items
    if len(result["lanes"]) + len(result["optional_lanes"]) > PLAN_MAX_MATRIX_ITEMS:
        raise PlanValidationError("matrix exceeds the provider job bound")
    return result


def _validate_plan_identity(
    plan: Mapping[str, object],
    records: tuple[Ownership, ...] | None,
) -> None:
    missing = sorted(PLAN_KEYS.difference(plan))
    unknown = sorted(set(plan).difference(PLAN_KEYS))
    details = ["missing " + ", ".join(missing)] if missing else []
    if unknown:
        details.append("unknown " + ", ".join(unknown))
    if details:
        raise PlanValidationError("invalid plan keys: " + "; ".join(details))
    if plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise PlanValidationError("unsupported planner schema version")
    if plan["contract"] != PLAN_CONTRACT or plan["planner_version"] != PLANNER_VERSION:
        raise PlanValidationError("unknown planner contract or planner version")
    event = _string(plan["event"], "event")
    if event not in VALID_EVENTS:
        raise PlanValidationError(f"unknown event {event!r}")
    for field in ("base_revision", "head_revision"):
        revision = plan[field]
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise PlanValidationError(f"{field} must be a string or null")
    changed = _string_list(plan["changed_paths"], "changed_paths")
    if changed != sorted(changed) or any(not _valid_changed_path(path) for path in changed):
        raise PlanValidationError("changed_paths must be sorted repository-relative paths")
    for field in ("base_available", "head_available", "full"):
        _boolean(plan[field], field)
    for field in ("planner_digest", "topology_digest"):
        _digest(plan[field], field)
    if plan["planner_digest"] != planner_digest():
        raise PlanValidationError("planner_digest does not match the current planner")
    if records is not None and plan["topology_digest"] != topology_digest(records):
        raise PlanValidationError("topology_digest does not match the current ownership")
    fallback = _string(plan["fallback_reason"], "fallback_reason")
    if not plan["full"] and fallback != "none":
        raise PlanValidationError("non-full plans must use fallback_reason='none'")
    if plan["full"] and fallback == "none":
        raise PlanValidationError("full plans must explain their fallback")
    provenance = _mapping(plan["provenance"], "provenance")
    expected = {
        field: plan[field]
        for field in (
            "event",
            "base_revision",
            "head_revision",
            "base_available",
            "head_available",
            "changed_paths",
            "planner_digest",
            "topology_digest",
            "fallback_reason",
        )
    }
    source_fields = ("event_source", "base_source", "head_source")
    if set(provenance) != set(expected) | set(source_fields):
        raise PlanValidationError("provenance does not match the top-level identity")
    for field in source_fields:
        _string(provenance.get(field), f"provenance.{field}")
    if any(provenance[field] != value for field, value in expected.items()):
        raise PlanValidationError("provenance does not match the top-level identity")


def _validate_plan_lanes(plan: Mapping[str, object]) -> list[str]:
    selected = _lane_decisions(plan["selected_lanes"], "selected_lanes")
    unselected = _lane_decisions(plan["unselected_lanes"], "unselected_lanes")
    selected_names = {item["lane"] for item in selected}
    unselected_names = {item["lane"] for item in unselected}
    if selected_names & unselected_names or selected_names | unselected_names != set(
        ALL_PLANNED_LANES
    ):
        raise PlanValidationError("selected and unselected lanes must partition the topology")
    matrices = _matrix(plan["matrix"])
    matrix_lanes = [item["lane"] for item in matrices["lanes"]]
    matrix_optional = [item["lane"] for item in matrices["optional_lanes"]]
    if set(matrix_lanes) != selected_names.intersection(TEST_LANES):
        raise PlanValidationError("matrix.lanes does not match selected lanes")
    if set(matrix_optional) != selected_names.intersection(PROVIDER_LANES):
        raise PlanValidationError("matrix.optional_lanes does not match selected lanes")
    lanes = _string_list(plan["lanes"], "lanes")
    optional_lanes = _string_list(plan["optional_lanes"], "optional_lanes")
    if lanes != matrix_lanes or optional_lanes != matrix_optional:
        raise PlanValidationError("legacy lane fields do not match the matrix")
    if any(lane not in TEST_LANES for lane in lanes):
        raise PlanValidationError("lanes contains an unknown lane")
    if any(lane not in CI_PROVIDER_LANES for lane in optional_lanes):
        raise PlanValidationError("optional_lanes contains an unknown lane")
    return lanes


def _validate_plan_flags(plan: Mapping[str, object], lanes: list[str]) -> None:
    for field in (
        "run_quality",
        "run_deep_checks",
        "run_coverage",
        "run_collection",
        "run_npm",
        "run_performance",
    ):
        _boolean(plan[field], field)
    if plan["run_deep_checks"] != plan["full"]:
        raise PlanValidationError("run_deep_checks must match full")
    if plan["run_quality"] != (plan["full"] or bool(lanes)):
        raise PlanValidationError("run_quality is inconsistent")
    if plan["run_coverage"] != (plan["full"] or bool(lanes)):
        raise PlanValidationError("run_coverage is inconsistent")
    if plan["run_collection"] != (plan["full"] or bool(lanes)):
        raise PlanValidationError("run_collection is inconsistent")
    if plan["run_npm"] != plan["full"]:
        raise PlanValidationError("run_npm must match full")
    selected_decisions = _lane_decisions(plan["selected_lanes"], "selected_lanes")
    performance_selected = any(item["lane"] == "performance" for item in selected_decisions)
    if plan["run_performance"] != performance_selected:
        raise PlanValidationError("run_performance is inconsistent")


def validate_affected_plan(
    plan: Mapping[str, object],
    records: tuple[Ownership, ...] | None = None,
) -> dict[str, object]:
    """Validate planner output before a CI matrix consumes it."""
    _validate_plan_identity(plan, records)
    lanes = _validate_plan_lanes(plan)
    _validate_plan_flags(plan, lanes)
    return dict(plan)


def validate_ownership(records: tuple[Ownership, ...]) -> None:
    test_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if "__pycache__" not in path.parts
    }
    listed: list[str] = [path for record in records for path in record.paths]
    duplicates = sorted({path for path in listed if listed.count(path) > 1})
    missing = sorted(test_files.difference(listed))
    unknown = sorted(set(listed).difference(test_files))
    if duplicates or missing or unknown:
        details = []
        if duplicates:
            details.append(f"duplicate paths: {', '.join(duplicates)}")
        if missing:
            details.append(f"unowned paths: {', '.join(missing)}")
        if unknown:
            details.append(f"manifest paths do not exist: {', '.join(unknown)}")
        raise SystemExit("Ownership validation failed: " + "; ".join(details))

    allowed_markers = {
        "unit",
        "integration",
        "process",
        "optional",
        "performance",
        "serial",
    } | set(PROVIDER_MARKERS)
    invalid = sorted({marker for record in records for marker in record.markers} - allowed_markers)
    if invalid:
        raise SystemExit("Ownership validation failed: unknown markers: " + ", ".join(invalid))


def lane_records(records: tuple[Ownership, ...], lane: str) -> tuple[Ownership, ...]:
    return tuple(record for record in records if record.lane == lane)


def lane_paths(records: tuple[Ownership, ...], lane: str) -> tuple[str, ...]:
    return tuple(path for record in lane_records(records, lane) for path in record.paths)


def marker_expression(lane: str) -> str | None:
    if lane == "full":
        return "not optional or optional"
    if lane == "core":
        return "not performance and not optional and not process"
    if lane == "process":
        return "process and not optional and not performance"
    if lane in PROVIDER_LANES:
        return PROVIDER_LANES[lane]
    if lane == "optional":
        return "optional"
    if lane == "performance":
        return "performance"
    if lane == "mcp":
        return "not optional and not performance"
    if lane in {"adapters", "application", "analysis", "cli", "golden", "security", "storage"}:
        return "not optional and not performance"
    raise SystemExit(f"Unknown test lane: {lane}")


def command_for(lane: str, records: tuple[Ownership, ...], *, coverage: bool = False) -> list[str]:
    paths = [str(ROOT / path) for path in lane_paths(records, lane)]
    if lane in AGGREGATE_LANES:
        paths = [str(ROOT / "tests")]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *paths,
        "-ra",
        "--strict-config",
        "--strict-markers",
        "--durations=20",
        "-p",
        "no:randomly",
    ]
    expression = marker_expression(lane)
    if expression is not None:
        command.extend(["-m", expression])
    if coverage:
        command.extend(
            [
                "--cov=flameox",
                "--cov-report=",
                "--cov-fail-under=0",
            ]
        )
    return command


def disk_telemetry(label: str) -> str:
    usage = shutil.disk_usage(ROOT)
    temp_usage = shutil.disk_usage(Path(tempfile.gettempdir()))
    return (
        f"{label}: workspace_free={usage.free} workspace_total={usage.total} "
        f"temp_free={temp_usage.free} temp_total={temp_usage.total} "
        f"pid={os.getpid()}"
    )


def collect_node_ids() -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT / "tests"),
        "--collect-only",
        "-q",
        "-p",
        "no:randomly",
        "-m",
        "not optional or optional",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)
    return tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )


def verify_collection(records: tuple[Ownership, ...]) -> int:
    del records
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    node_ids = collect_node_ids()
    (RESULTS_DIR / "collection.txt").write_text("\n".join(node_ids) + "\n", encoding="utf-8")
    with COLLECTION_BASELINE_PATH.open("rb") as stream:
        baseline = tomllib.load(stream)
    moved_paths = {
        str(path): str(target) for path, target in baseline.get("moved_paths", {}).items()
    }
    moved_digests = {
        str(path): str(digest) for path, digest in baseline.get("moved_digests", {}).items()
    }
    current: dict[str, list[str]] = {}
    for node_id in node_ids:
        path = node_id.split("::", 1)[0]
        canonical_path = moved_paths.get(path, path)
        canonical_id = canonical_path + node_id[len(path) :]
        current.setdefault(canonical_path, []).append(canonical_id)
    expected_count = int(baseline["expected_test_count"])
    expected_files = baseline["files"]
    differences: list[str] = []
    if len(node_ids) != expected_count:
        differences.append(f"count expected {expected_count}, got {len(node_ids)}")
    for path, expected in expected_files.items():
        actual = current.get(path, [])
        digest_items = sorted(actual) if path in moved_digests else actual
        digest = hashlib.sha256((r"\n".join(digest_items) + r"\n").encode()).hexdigest()
        expected_digest = moved_digests.get(path, str(expected["digest"]))
        if len(actual) != expected["count"] or digest != expected_digest:
            differences.append(
                f"{path}: expected {expected['count']} / {expected_digest}, "
                f"got {len(actual)} / {digest}"
            )
    for path in sorted(set(current).difference(expected_files)):
        differences.append(f"unexpected collected path: {path}")
    if differences:
        print("Collection preservation failed:")
        print("\n".join(f"- {difference}" for difference in differences))
        return 1
    print(f"Collection preserved: {len(node_ids)} node IDs across {len(current)} files.")
    print(f"Collection receipt: {RESULTS_DIR / 'collection.txt'}")
    return 0


def print_provider_inventory() -> int:
    print("Providers:")
    for marker, available in provider_inventory():
        print(f"  {marker:20} {'available' if available else 'unavailable'}")
    return 0


def validate_capability_contract() -> int:
    """Verify managed capability setup metadata matches published extras."""
    from flameox.adapters.builtins import BUILTIN_ADAPTERS

    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    extras = project["project"]["optional-dependencies"]
    managed: dict[str, str] = {}
    for adapter in BUILTIN_ADAPTERS.values():
        if adapter.managed_extra is None or adapter.managed_requirement is None:
            continue
        if adapter.name in managed:
            raise SystemExit(f"Duplicate managed capability metadata: {adapter.name}")
        requirements = extras.get(adapter.managed_extra, [])
        if adapter.managed_requirement not in requirements:
            raise SystemExit(
                f"{adapter.name}: {adapter.managed_requirement!r} is missing from "
                f"[project.optional-dependencies].{adapter.managed_extra}"
            )
        managed[adapter.name] = adapter.managed_extra
    print("Managed capability contract:")
    for adapter_name, extra in sorted(managed.items()):
        print(f"  {adapter_name:20} flameox[{extra}]")
    return 0


def _valid_changed_path(path: str) -> bool:
    if not path or path.startswith(("/", "\\")):
        return False
    parts = Path(path).parts
    return "." not in parts and ".." not in parts and "\\" not in path


def _revision_from_git() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _revision_exists(revision: str | None) -> bool:
    if not revision or set(revision) == {"0"}:
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _context(
    value: str | None, environment: str, default: str | None = None
) -> tuple[str | None, str]:
    if value is not None:
        return value, "argument"
    from_environment = os.environ.get(environment)
    if from_environment:
        return from_environment, "environment"
    if default is not None:
        return default, "default"
    return None, "missing"


def changed_paths(base: str | None, head: str | None) -> tuple[set[str], str | None]:
    """Return changed paths or a named conservative fallback reason."""
    if not base or set(base) == {"0"}:
        return set(), "missing_base_revision"
    if not head:
        return set(), "missing_head_revision"
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}", "--"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set(), "unavailable_revision"
    paths = {line for line in result.stdout.splitlines() if line}
    if any(not _valid_changed_path(path) for path in paths):
        return set(), "invalid_changed_path"
    return paths, None


def _reason_map_add(reasons: dict[str, list[str]], lane: str, reason: str) -> None:
    if reason not in reasons.setdefault(lane, []):
        reasons[lane].append(reason)


def _fallback_reason(reasons: list[str]) -> str:
    unique = list(dict.fromkeys(reasons))
    return "; ".join(unique) if unique else "none"


def affected_plan(  # noqa: C901
    records: tuple[Ownership, ...],
    base: str | None,
    *,
    event: str | None = None,
    head: str | None = None,
) -> dict[str, object]:
    """Build a bounded, provenance-bound plan while retaining legacy fields."""
    resolved_event, event_source = _context(event, "GITHUB_EVENT_NAME", "local")
    assert resolved_event is not None
    if resolved_event not in VALID_EVENTS:
        raise PlanValidationError(f"unknown event {resolved_event!r}")
    resolved_base, base_source = _context(base, "GITHUB_BASE_SHA")
    resolved_head, head_source = _context(head, "GITHUB_SHA", _revision_from_git())
    paths, revision_reason = changed_paths(resolved_base, resolved_head)
    by_path = {path: record for record in records for path in record.paths}
    fallback_reasons = [reason for reason in (revision_reason,) if reason]
    full = revision_reason is not None or resolved_event in FULL_RUN_EVENTS
    if resolved_event in FULL_RUN_EVENTS:
        fallback_reasons.append("event_requires_full_validation")
    lanes: set[str] = set()
    optional_lanes: set[str] = set()
    coverage = revision_reason is not None
    npm = revision_reason is not None
    performance = False
    lane_reasons: dict[str, list[str]] = {}
    optional_reasons: dict[str, list[str]] = {}

    for path in paths:
        if path in FULL_CHANGE_PATHS:
            full = True
            fallback_reasons.append("shared_configuration_change")
        elif path.startswith("src/"):
            full = True
            fallback_reasons.append("production_change")
        elif path.startswith(".github/"):
            full = True
            fallback_reasons.append("ci_configuration_change")
        elif path.startswith("tools/"):
            full = True
            fallback_reasons.append("planner_or_tooling_change")
        elif path.startswith("npm/"):
            full = True
            npm = True
            fallback_reasons.append("npm_change")
        elif path.startswith("tests/"):
            record = by_path.get(path)
            if record is None:
                full = True
                fallback_reasons.append("unowned_test_path")
                continue
            if "performance" in record.markers:
                performance = True
                _reason_map_add(lane_reasons, "performance", f"owned performance test path: {path}")
            if "optional" not in record.markers and "performance" not in record.markers:
                lanes.add(record.lane)
                coverage = True
                _reason_map_add(lane_reasons, record.lane, f"owned test path: {path}")
            provider_markers = (
                ("requires_compute_sanitizer", "optional-compute-sanitizer"),
                ("requires_coverage", "optional-coverage"),
                ("requires_memray", "optional-memray"),
                ("requires_perfetto", "optional-perfetto"),
                ("requires_pyspy", "optional-pyspy"),
                ("requires_torch", "optional-torch"),
            )
            marked_provider = False
            for marker, provider_lane in provider_markers:
                if marker in record.markers:
                    marked_provider = True
                    if provider_lane in CI_PROVIDER_LANES:
                        optional_lanes.add(provider_lane)
                        _reason_map_add(
                            optional_reasons,
                            provider_lane,
                            f"owned provider test path: {path}",
                        )
            if "optional" in record.markers and not marked_provider:
                optional_lanes.add("optional-host")
                _reason_map_add(
                    optional_reasons, "optional-host", f"owned optional test path: {path}"
                )
        elif path.startswith(("docs/", "README", "CHANGELOG", "LICENSE")):
            continue
        else:
            full = True
            fallback_reasons.append("unknown_path")

    if full:
        lanes.update(TEST_LANES)
        optional_lanes.update(CI_PROVIDER_LANES)
        performance = True
        coverage = True
        npm = True
        reason = _fallback_reason(fallback_reasons)
        for lane in TEST_LANES:
            _reason_map_add(lane_reasons, lane, f"conservative full plan: {reason}")
        _reason_map_add(lane_reasons, "performance", f"conservative full plan: {reason}")
        for lane in CI_PROVIDER_LANES:
            _reason_map_add(optional_reasons, lane, f"conservative full plan: {reason}")
    elif coverage:
        for lane in COVERAGE_LANES:
            lanes.add(lane)
            _reason_map_add(lane_reasons, lane, "coverage aggregation for affected tests")

    ordered_lanes = [lane for lane in TEST_LANES if lane in lanes]
    ordered_optional = [lane for lane in CI_PROVIDER_LANES if lane in optional_lanes]
    fallback = _fallback_reason(fallback_reasons)
    selected = (
        [
            {
                "lane": lane,
                "reason": "; ".join(lane_reasons.get(lane, ["selected by ownership policy"])),
            }
            for lane in ordered_lanes
        ]
        + (
            [
                {
                    "lane": "performance",
                    "reason": "; ".join(
                        lane_reasons.get(
                            "performance", ["selected by performance ownership policy"]
                        )
                    ),
                }
            ]
            if performance
            else []
        )
        + [
            {
                "lane": lane,
                "reason": "; ".join(optional_reasons.get(lane, ["selected by provider ownership"])),
            }
            for lane in ordered_optional
        ]
    )
    selected_names = {item["lane"] for item in selected}
    unselected = [
        {
            "lane": lane,
            "reason": "not selected by the changed-path ownership policy",
        }
        for lane in ALL_PLANNED_LANES
        if lane not in selected_names
    ]
    base_available = bool(resolved_base) and revision_reason != "missing_base_revision"
    head_available = bool(resolved_head) and revision_reason != "missing_head_revision"
    if revision_reason == "unavailable_revision":
        base_available = _revision_exists(resolved_base)
        head_available = _revision_exists(resolved_head)
    plan: dict[str, object] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "contract": PLAN_CONTRACT,
        "planner_version": PLANNER_VERSION,
        "event": resolved_event,
        "base_revision": resolved_base,
        "head_revision": resolved_head,
        "base_available": base_available,
        "head_available": head_available,
        "changed_paths": sorted(paths),
        "planner_digest": planner_digest(),
        "topology_digest": topology_digest(records),
        "fallback_reason": fallback,
        "provenance": {},
        "full": full,
        "selected_lanes": selected,
        "unselected_lanes": unselected,
        "matrix": {
            "lanes": [{"lane": lane} for lane in ordered_lanes],
            "optional_lanes": [{"lane": lane} for lane in ordered_optional],
        },
        "lanes": ordered_lanes,
        "optional_lanes": ordered_optional,
        "run_quality": full or bool(ordered_lanes),
        "run_deep_checks": full,
        "run_coverage": full or bool(ordered_lanes),
        "run_collection": full or bool(ordered_lanes),
        "run_npm": npm,
        "run_performance": performance,
    }
    plan["provenance"] = {
        "event": resolved_event,
        "base_revision": resolved_base,
        "head_revision": resolved_head,
        "base_available": plan["base_available"],
        "head_available": plan["head_available"],
        "changed_paths": plan["changed_paths"],
        "planner_digest": plan["planner_digest"],
        "topology_digest": plan["topology_digest"],
        "fallback_reason": fallback,
        "event_source": event_source,
        "base_source": base_source,
        "head_source": head_source,
    }
    # Keep provenance flat and exact; source metadata is useful but must be
    # represented in the contract rather than silently omitted.
    validate_affected_plan(plan)
    return plan


def print_affected_plan(
    records: tuple[Ownership, ...],
    base: str | None,
    *,
    event: str | None = None,
    head: str | None = None,
) -> int:
    try:
        plan = affected_plan(records, base, event=event, head=head)
    except (PlanValidationError, ValueError) as error:
        print(f"Invalid affected plan: {error}", file=sys.stderr)
        return 2
    print(json.dumps(plan, sort_keys=True))
    return 0


def run_lane(lane: str, records: tuple[Ownership, ...], *, coverage: bool = False) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    command = command_for(lane, records, coverage=coverage)
    junit = RESULTS_DIR / f"{lane}.xml"
    log_path = RESULTS_DIR / f"{lane}.log"
    command_path = RESULTS_DIR / f"{lane}.command.txt"
    command.extend([f"--junitxml={junit}"])
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
    print(f"Running lane '{lane}'", flush=True)
    print("Command: " + " ".join(command), flush=True)
    print(disk_telemetry("before"), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    print(disk_telemetry("after"), flush=True)
    print(f"JUnit: {junit}", flush=True)
    print(f"Log: {log_path}", flush=True)
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "list",
            "ownership",
            "collection",
            "providers",
            "capabilities",
            "affected",
            "core",
            "process",
            "optional",
            *PROVIDER_LANES,
            "performance",
            "full",
            "storage",
            "application",
            "mcp",
            "adapters",
            "analysis",
            "cli",
            "golden",
            "security",
        ),
    )
    parser.add_argument(
        "--base",
        help="git base revision for the affected CI plan; unavailable revisions select all lanes",
    )
    parser.add_argument(
        "--head",
        help="git head revision for the affected CI plan; defaults to GITHUB_SHA or HEAD",
    )
    parser.add_argument(
        "--event",
        choices=sorted(VALID_EVENTS),
        help="event context for the affected CI plan; defaults to GITHUB_EVENT_NAME or local",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="collect coverage without enforcing the threshold (for CI aggregation)",
    )
    args = parser.parse_args()
    records = load_ownership(OWNERSHIP_PATH)
    validate_ownership(records)
    if args.command == "ownership":
        print(f"Validated {len(records)} ownership records covering all test files.")
        return 0
    if args.command == "collection":
        return verify_collection(records)
    if args.command == "providers":
        return print_provider_inventory()
    if args.command == "capabilities":
        return validate_capability_contract()
    if args.command == "list":
        print("Available test lanes:")
        for lane in (*TEST_LANES, "optional", *PROVIDER_LANES, "performance", "full"):
            expression = marker_expression(lane)
            all_paths = lane in AGGREGATE_LANES
            scope = "all test paths" if all_paths else ", ".join(lane_paths(records, lane))
            scope = scope or "no paths"
            print(f"  {lane:12} markers={expression or 'all'} paths={scope}")
        print("\nMetadata commands:")
        print("  ownership    validate tests/ownership.toml coverage")
        print("  collection   compare collected node IDs with the baseline")
        print("  providers    report optional provider availability")
        print("  capabilities validate managed setup metadata against extras")
        print("  affected     print a conservative ownership-driven CI plan")
        print("\nEach run writes .test-results/<lane>.xml, .log, and .command.txt.")
        return 0
    if args.command == "affected":
        return print_affected_plan(records, args.base, event=args.event, head=args.head)
    return run_lane(args.command, records, coverage=args.coverage)


if __name__ == "__main__":
    raise SystemExit(main())
