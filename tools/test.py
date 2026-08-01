#!/usr/bin/env python3
"""Run Flameox's explicitly owned pytest lanes.

The runner is intentionally small: ownership lives in tests/ownership.toml,
pytest remains the test framework, and no command retries a failed case.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNERSHIP_PATH = ROOT / "tests" / "ownership.toml"
COLLECTION_BASELINE_PATH = ROOT / "tests" / "collection-baseline.toml"
RESULTS_DIR = ROOT / ".test-results"
PROVIDER_LANES = {
    "optional-coverage": "optional and requires_coverage",
    "optional-memray": "optional and requires_memray",
    "optional-perfetto": "optional and requires_perfetto",
    "optional-pyspy": "optional and requires_pyspy",
    "optional-torch": "optional and requires_torch",
    "optional-host": (
        "optional and not requires_coverage and not requires_memray "
        "and not requires_perfetto and not requires_pyspy and not requires_torch"
    ),
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flameox.adapters.builtins import BUILTIN_ADAPTERS  # noqa: E402
from tests.support.ownership import Ownership, load_ownership  # noqa: E402
from tests.support.providers import PROVIDER_MARKERS, provider_inventory  # noqa: E402


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


def command_for(lane: str, records: tuple[Ownership, ...]) -> list[str]:
    paths = [str(ROOT / path) for path in lane_paths(records, lane)]
    if lane in {"core", "optional", "performance", "full"} | set(PROVIDER_LANES):
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
    if lane == "core":
        command.extend(
            [
                "--cov=flameox",
                "--cov-report=xml:.test-results/core-coverage.xml",
                "--cov-report=term",
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


def run_lane(lane: str, records: tuple[Ownership, ...]) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    command = command_for(lane, records)
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
        for lane in (
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
            "capabilities",
            "optional",
            *PROVIDER_LANES,
            "performance",
            "full",
        ):
            expression = marker_expression(lane)
            all_paths = lane in {"core", "optional", "performance", "full"} | set(PROVIDER_LANES)
            scope = "all test paths" if all_paths else ", ".join(lane_paths(records, lane))
            scope = scope or "no paths"
            print(f"  {lane:12} markers={expression or 'all'} paths={scope}")
        print("\nEach run writes .test-results/<lane>.xml, .log, and .command.txt.")
        return 0
    return run_lane(args.command, records)


if __name__ == "__main__":
    raise SystemExit(main())
