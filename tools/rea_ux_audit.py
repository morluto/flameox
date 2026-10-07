"""Exercise relevant Flameox CLI and MCP workflows on an explicit REA checkout.

Run with uv run python -m tools.rea_ux_audit --rea /absolute/path/to/rea
--output /absolute/path/to/new-audit-directory. Optional --trace and --sarif accept
native upstream exports. No REA source or global client configuration is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import anyio
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp_types import TextResourceContents

from flameox import __version__
from flameox.runtime_contracts import CAPABILITY_BY_ID

type ToolCaller = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _snapshot(rea: Path) -> dict[str, Any]:
    files = sorted(
        {rea / "package.json", rea / "scripts/rea.mjs"}
        | set((rea / "dist").rglob("*.js"))
        | {
            path
            for path in (rea / "tests/conformance/readiness/javascript-cli").rglob("*")
            if path.is_file()
        }
    )
    identities = {}
    for path in files:
        with path.open("rb") as stream:
            identities[str(path.relative_to(rea))] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return {
        "commit": subprocess.check_output(
            ["git", "-C", str(rea), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip(),
        "dirty_state": subprocess.check_output(
            ["git", "-C", str(rea), "status", "--short"], text=True, timeout=10
        ),
        "workload_files": identities,
    }


def _cli(output: Path, rea: Path, environment: dict[str, str], node: Path) -> None:
    commands = {
        "cli-discovery": ["mcp", "inspect"],
        "cli-detail": ["mcp", "inspect", "--capability", "trace.window"],
        "cli-full": ["mcp", "inspect", "--full"],
        "cli-setup-plan": ["setup", "--client", "codex", "--dry-run", "--json"],
        "cli-location": ["evidence", "location"],
        "cli-query": ["evidence", "query", "--limit", "1"],
        "cli-preview": ["analyze", "artifact.preview", str(rea / "package.json")],
        "cli-capture": [
            "capture",
            "--provider",
            "node-cpu-profile",
            "--capability",
            "cpu.hotspots",
            "--cwd",
            str(rea),
            "--workload-budget",
            '{"timeout_seconds":30}',
            "--preserve",
            "--",
            str(node),
            "scripts/rea.mjs",
            "capabilities",
            "--format",
            "json",
        ],
    }
    for name, arguments in commands.items():
        result = subprocess.run(
            [sys.executable, "-m", "flameox", *arguments],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )
        value = json.loads(result.stdout)
        _save(output / f"{name}.json", value)
        if name == "cli-capture":
            reference = value["preserved"]["evidence_id"]
            shown = subprocess.run(
                [sys.executable, "-m", "flameox", "evidence", "show", reference],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            _save(output / "cli-show.json", json.loads(shown.stdout))
            page = value.get("next_page")
            if page:
                resumed = subprocess.run(
                    [sys.executable, "-m", "flameox", *page["argv"]],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                _save(output / "cli-page2.json", json.loads(resumed.stdout))


async def _audit(
    rea: Path, output: Path, trace: Path | None, sarif: Path | None, node: Path, experiments: bool
) -> None:
    started_at = datetime.now(UTC).isoformat()
    flameox_root = Path(__file__).resolve().parents[1]
    flameox_identity = {
        "commit": subprocess.check_output(
            ["git", "-C", str(flameox_root), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "-C", str(flameox_root), "branch", "--show-current"], text=True, timeout=10
        ).strip(),
        "dirty_state": subprocess.check_output(
            ["git", "-C", str(flameox_root), "status", "--short"], text=True, timeout=10
        ),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(
                ["git", "-C", str(flameox_root), "diff", "--binary", "HEAD"], timeout=10
            )
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    baseline = _snapshot(rea)
    _save(output / "rea-before.json", baseline)
    environment = {**os.environ, "FLAMEOX_DATA_DIR": str(output / "store")}
    records: list[dict[str, Any]] = []
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "flameox", "mcp", "serve"],
        env=environment,
    )
    async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
        initialized = await session.initialize()
        catalog = await session.list_tools()
        _save(output / "initialize.json", initialized.model_dump(mode="json", by_alias=True))
        _save(output / "catalog.json", catalog.model_dump(mode="json", by_alias=True))

        async def call(
            label: str, tool: str, arguments: dict[str, Any], *, error_code: str | None = None
        ) -> dict[str, Any]:
            start = time.monotonic()
            result = await session.call_tool(tool, arguments)
            await session.validate_tool_result(tool, result)
            value = cast(dict[str, Any], result.structured_content)
            _save(output / f"{label}.json", result.model_dump(mode="json", by_alias=True))
            passed = (
                result.is_error and value.get("code") == error_code
                if error_code
                else not result.is_error
            )
            record = {
                "label": label,
                "tool": tool,
                "seconds": round(time.monotonic() - start, 3),
                "is_error": result.is_error,
                "schema_valid": True,
                "passed": passed,
                "status": value.get("status"),
                "code": value.get("code"),
            }
            records.append(record)
            _save(output / "calls.json", records)
            print(json.dumps(record), flush=True)
            return value

        discovery = await call("discovery", "inspect_capabilities", {"mode": "list"})
        for descriptor in discovery["capabilities"]:
            capability_id = descriptor["capability_id"]
            detail = await call(
                "detail-" + capability_id,
                "inspect_capabilities",
                {"mode": "get", "capability_id": capability_id},
            )
            example = detail["capabilities"][0]["analysis_example"]["request"]
            CAPABILITY_BY_ID[capability_id].model.model_validate(example.get("options", {}))
        await call("query-absent", "query_evidence", {})
        target = {
            "argv": [
                str(node),
                "scripts/rea.mjs",
                "analyze-javascript-application",
                "tests/conformance/readiness/javascript-cli",
                "--format",
                "json",
            ],
            "cwd": str(rea),
            "budget": {"timeout_seconds": 30},
        }

        async def capture(
            label: str, capability_id: str, provider: str, *, preserve: bool = False
        ) -> dict[str, Any]:
            result = await call(
                label,
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": capability_id,
                        "target": target,
                        "provider": {"kind": provider},
                        "preserve": preserve,
                    },
                    "page_size": 10,
                },
            )
            assert result["analysis_failure"] is None
            assert result["capture"]["outcome"]["status"] == "succeeded"
            return result

        cpu = await capture("cpu", "cpu.hotspots", "node-cpu-profile")
        assert all("function" in row and "file" in row for row in cpu["blocks"][1]["rows"])
        await capture("heap", "memory.hotspots", "node-heap-profile", preserve=True)
        await capture("direct", "artifact.preview", "direct")
        benchmarks = []
        for index in range(2):
            benchmark = await call(
                f"benchmark-{index}",
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "benchmark.summary",
                        "target": target,
                        "preserve": True,
                        "provider": {
                            "kind": "pyperf",
                            "options": {
                                "processes": 1,
                                "values": 3,
                                "warmups": 0,
                                "loops": 1,
                                "name": "rea.javascript-analysis",
                            },
                        },
                    },
                    "page_size": 10,
                },
            )
            assert benchmark["analysis_failure"] is None
            assert benchmark["capture"]["outcome"]["status"] == "succeeded"
            benchmarks.append(benchmark)
        await call(
            "benchmark-compare",
            "analyze",
            {
                "request": {
                    "capability_id": "benchmark.compare",
                    "sources": [
                        {"kind": "evidence", "evidence_id": item["preserved"]["evidence_id"]}
                        for item in benchmarks
                    ],
                }
            },
        )
        preserved = await call("preserve", "preserve_evidence", {"analysis_id": cpu["analysis_id"]})
        await call("preserve-again", "preserve_evidence", {"analysis_id": cpu["analysis_id"]})
        resource = await session.read_resource(preserved["uri"])
        _save(output / "resource.json", resource.model_dump(mode="json", by_alias=True))
        assert isinstance(resource.contents[0], TextResourceContents)
        projection = json.loads(resource.contents[0].text)
        await call(
            "cpu-reopened",
            "analyze",
            {
                "request": {
                    "capability_id": "cpu.hotspots",
                    "sources": projection["analysis_sources"],
                },
                "page_size": 10,
            },
        )
        if page := preserved.get("next_page"):
            await call("cpu-page2", page["tool"], page["arguments"])
        rescued = await call(
            "rescue",
            "rescue_evidence",
            {
                "analysis_id": cpu["analysis_id"],
                "destination": str(output / "rescued-store"),
            },
        )
        await call("query", "query_evidence", {"page_size": 1})
        await call(
            "invalid-provider",
            "capture_and_analyze",
            {
                "request": {
                    "capability_id": "cpu.hotspots",
                    "target": target,
                    "provider": {"kind": "direct"},
                }
            },
            error_code="INVALID_INPUT",
        )
        await call(
            "expired",
            "preserve_evidence",
            {"analysis_id": "0" * 64},
            error_code="EXPIRED_SESSION_ANALYSIS",
        )
        if trace:
            for capability_id in ["trace.summary", "trace.call_graph", "trace.window"]:
                options = (
                    {"start_ns": 0, "end_ns": 2**53 - 1} if capability_id == "trace.window" else {}
                )
                await call(
                    capability_id,
                    "analyze",
                    {
                        "request": {
                            "capability_id": capability_id,
                            "options": options,
                            "sources": [
                                {"kind": "path", "path": str(trace), "format": "chrome-trace"}
                            ],
                        },
                        "page_size": 10,
                    },
                )
        if sarif:
            await call(
                "static",
                "analyze",
                {
                    "request": {
                        "capability_id": "static.performance_candidates",
                        "options": {"source_root": str(rea)},
                        "sources": [{"kind": "path", "path": str(sarif), "format": "sarif"}],
                    },
                    "page_size": 10,
                },
            )
        if experiments:
            await _experiment_workflows(call, rea, output, node)
    # A new process must consume the rescue handoff after the capture server exits.
    reopened_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "flameox", "mcp", "serve"],
        env={**os.environ, **rescued["next_action"]["environment"]},
    )
    async with stdio_client(reopened_parameters) as streams, ClientSession(*streams) as session:
        await session.initialize()
        resource = await session.read_resource(rescued["uri"])
        _save(output / "rescued-resource.json", resource.model_dump(mode="json", by_alias=True))
        assert isinstance(resource.contents[0], TextResourceContents)
        projection = json.loads(resource.contents[0].text)
        await call(
            "rescue-reopened",
            "analyze",
            {
                "request": {
                    "capability_id": "cpu.hotspots",
                    "sources": projection["analysis_sources"],
                }
            },
        )
    await anyio.to_thread.run_sync(_cli, output, rea, environment, node)
    final = _snapshot(rea)
    _save(output / "rea-after.json", final)
    _save(
        output / "summary.json",
        {
            "flameox_version": __version__,
            "flameox_source": flameox_identity,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "host": platform.platform(),
            "node_executable": str(node),
            "node_version": subprocess.check_output(
                [str(node), "--version"], text=True, timeout=10
            ).strip(),
            "rea_commit": baseline["commit"],
            "workload_files_unchanged": baseline["workload_files"] == final["workload_files"],
            "tool_count": len(catalog.tools),
            "catalog_bytes": len(catalog.model_dump_json(by_alias=True).encode()),
            "calls": len(records),
            "all_calls_passed": all(row["passed"] for row in records),
            "resource_reads": 2,
            "semantic_experiments": experiments,
            "optional_exports": {
                "trace": str(trace) if trace else None,
                "sarif": str(sarif) if sarif else None,
            },
            "limitations": [
                "These are contract and workflow checks, not model-selection-rate measurements.",
                "REA is a Node workload; no GPU, inference service, or Python-only workload "
                "is invented.",
                "Repeating one baseline measures compatibility, not an optimization or causality.",
            ],
        },
    )
    if not all(row["passed"] for row in records):
        raise SystemExit("Unexpected tool outcomes; inspect calls.json.")


async def _experiment_workflows(
    call: ToolCaller,
    rea: Path,
    output: Path,
    node: Path,
) -> None:
    """Check paired experiment and semantic rejection contracts on real REA analysis."""
    work = output / "experiments"
    work.mkdir()
    analysis_argv = [
        str(node),
        "scripts/rea.mjs",
        "analyze-javascript-application",
        "tests/conformance/readiness/javascript-cli",
        "--format",
        "json",
        "--filter-output",
        "normalized_result.root_artifact_sha256,normalized_result.statistics,"
        "normalized_result.summary",
    ]
    reference = await anyio.to_thread.run_sync(
        lambda: subprocess.check_output(analysis_argv, cwd=rea, timeout=30)
    )
    expected = work / "expected.json"
    _save(expected, json.loads(reference))
    oracle = work / "oracle.py"
    oracle.write_text(
        "import json, os, pathlib, sys\n"
        f"expected = json.loads(pathlib.Path({str(expected)!r}).read_text())\n"
        "observed = json.loads(pathlib.Path(os.environ['FLAMEOX_CAPTURE_STDOUT']).read_text())\n"
        "assert observed == expected, 'REA source findings differ'\n"
        "sys.exit(1 if os.environ.get('REA_AUDIT_REJECT') == '1' else 0)\n"
    )
    design = {
        "cases": [
            {"name": "baseline"},
            {"name": "repeat-control"},
        ],
        "blocks": 3,
        "seed": 23,
        "metric": "wall_time_ns",
        "estimand": "median_difference",
        "practical_threshold": 100_000_000,
        "semantic_oracle": [sys.executable, str(oracle)],
    }
    experiment = await call(
        "experiment-oracle",
        "capture_and_analyze",
        {
            "request": {
                "capability_id": "artifact.preview",
                "provider": {"kind": "direct"},
                "target": {
                    "argv": analysis_argv,
                    "cwd": str(rea),
                    "budget": {"timeout_seconds": 30},
                },
                "experiment": design,
                "preserve": True,
            },
            "page_size": 10,
        },
    )
    assert experiment["capture"]["outcome"]["succeeded_count"] == 6
    assert all(
        item["semantic_oracle"]["status"] == "passed"
        for item in experiment["capture"]["executions"]
    )
    rejected = await call(
        "experiment-rejected-oracle",
        "capture_and_analyze",
        {
            "request": {
                "capability_id": "artifact.preview",
                "provider": {"kind": "direct"},
                "target": {
                    "argv": analysis_argv,
                    "cwd": str(rea),
                    "budget": {"timeout_seconds": 30},
                },
                "experiment": {
                    **design,
                    "blocks": 1,
                    "cases": [
                        {"name": "baseline"},
                        {"name": "oracle-invalid", "environment": {"REA_AUDIT_REJECT": "1"}},
                    ],
                },
                "preserve": True,
            },
        },
    )
    assert rejected["status"] == "partial"
    assert rejected["capture"]["workload_status"] == "succeeded"
    assert any(
        item["semantic_oracle"]["status"] == "failed" for item in rejected["capture"]["executions"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rea", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--sarif", type=Path)
    parser.add_argument("--node", type=Path, help="Exact Node executable; defaults to PATH lookup.")
    parser.add_argument(
        "--experiments", action="store_true", help="Also check REA semantic experiment workflows."
    )
    arguments = parser.parse_args()
    rea = arguments.rea.resolve(strict=True)
    if not (rea / "scripts/rea.mjs").is_file():
        parser.error("--rea must name a built REA checkout with scripts/rea.mjs.")
    selected_node = arguments.node or shutil.which("node")
    if selected_node is None:
        parser.error("Node is unavailable; pass --node with an exact executable path.")
    node = Path(selected_node).resolve(strict=True)
    output = arguments.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    anyio.run(
        _audit,
        rea,
        output,
        arguments.trace.resolve(strict=True) if arguments.trace else None,
        arguments.sarif.resolve(strict=True) if arguments.sarif else None,
        node,
        arguments.experiments,
    )


if __name__ == "__main__":
    main()
