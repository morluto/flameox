from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import json5
import pytest
from click import unstyle
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

pytestmark = [pytest.mark.e2e, pytest.mark.process]


def run_cli(store: Path, *argv: str, exit_code: int = 0) -> dict[str, Any]:
    result = subprocess.run(
        [str(Path(sys.executable).with_name("flameox")), *argv],
        env={**os.environ, "FLAMEOX_DATA_DIR": str(store)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    payload: dict[str, Any] = json.loads(result.stdout)
    return payload


def test_invalid_cli_arguments_use_safe_typed_diagnostics(tmp_path: Path) -> None:
    store = tmp_path / "evidence"
    artifact = tmp_path / "input.txt"
    artifact.write_text("one\n")
    executable = str(Path(sys.executable).with_name("flameox"))
    environment = {**os.environ, "FLAMEOX_DATA_DIR": str(store)}

    invalid = subprocess.run(
        [
            executable,
            "analyze",
            "artifact.preview",
            str(artifact),
            "--arguments",
            '{"secret_field":"PRIVATE-CLI-INPUT"}',
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert invalid.returncode == 1
    assert invalid.stdout == ""
    failure = json.loads(invalid.stderr)
    assert failure["code"] == "INVALID_INPUT"
    assert failure["field_path"] == ["secret_field"]
    assert failure["details"]["error_type"] == "extra_forbidden"
    assert "PRIVATE-CLI-INPUT" not in invalid.stderr
    assert "input_value" not in invalid.stderr
    assert "errors.pydantic.dev" not in invalid.stderr

    marker = tmp_path / "started"
    experiment = (
        '{"cases":[{"name":"base"},{"name":"candidate"}],"blocks":1,"seed":1,'
        '"metric":"wall_time_ns","estimand":"mean_difference","practical_threshold":1e999}'
    )
    invalid_capture = subprocess.run(
        [
            executable,
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
            "--experiment",
            experiment,
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert invalid_capture.returncode == 1
    assert json.loads(invalid_capture.stderr)["field_path"] == ["practical_threshold"]
    assert not marker.exists()

    for invalid_json in (
        "{bad",
        '{"unused":' + "[" * 10_000 + "0" + "]" * 10_000 + "}",
        '{"unused":' + "9" * 5_000 + "}",
    ):
        malformed = subprocess.run(
            [executable, "analyze", "artifact.preview", str(artifact), "--arguments", invalid_json],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert malformed.returncode == 2
        assert malformed.stdout == ""
        assert "Usage: flameox analyze" in unstyle(malformed.stderr)
        assert "invalid JSON" in unstyle(malformed.stderr)
        assert "Traceback" not in unstyle(malformed.stderr)

    continuation = base64.urlsafe_b64encode(b"[" * 10_000 + b"0" + b"]" * 10_000).decode()
    malformed_token = subprocess.run(
        [executable, "analyze", "artifact.preview", str(artifact), "--continuation", continuation],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert malformed_token.returncode == 1
    assert json.loads(malformed_token.stderr)["code"] == "INVALID_INPUT"
    assert "Traceback" not in malformed_token.stderr


@pytest.mark.parametrize(
    "mode",
    [
        "line_comment",
        "block_comment",
        "duplicate_json",
        "duplicate_section",
        "duplicate_entry",
        "nested_json",
        "nested_jsonc",
    ],
)
def test_installed_setup_preserves_comments_and_rejects_ambiguous_configuration(
    tmp_path: Path, mode: str
) -> None:
    client = "cursor" if mode in {"nested_json", "duplicate_json"} else "opencode"
    relative = ".cursor/mcp.json" if client == "cursor" else ".config/opencode/opencode.jsonc"
    config = tmp_path / relative
    config.parent.mkdir(parents=True)
    source = {
        "line_comment": '{"enabled":true// keep this comment\n}',
        "block_comment": '{"enabled":true/* keep this comment */}',
        "duplicate_json": '{"mcpServers":{"important":{"command":"keep"}},"mcpServers":{}}',
        "duplicate_section": '{"mcp":{},"mcp":{}}',
        "duplicate_entry": '{"mcp":{"flameox":{},"flameox":{}}}',
        "nested_json": "[" * 10_000 + "0" + "]" * 10_000,
        "nested_jsonc": "[" * 10_000 + "0" + "]" * 10_000,
    }[mode]
    config.write_text(source)
    environment = {**os.environ, "HOME": str(tmp_path), "NO_COLOR": "1"}
    command = [
        str(Path(sys.executable).with_name("flameox")),
        "setup",
        "--client",
        client,
        "--yes",
        "--json",
    ]
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=30, check=False
    )
    if mode not in {"line_comment", "block_comment"}:
        assert result.returncode == 2
        assert "Could not read" in result.stderr
        assert "Traceback" not in result.stderr
        assert config.read_text() == source
        return
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["clients"][0]["status"] == "created"
    updated = config.read_text()
    document = json5.loads(updated)
    assert document["enabled"] is True
    assert "keep this comment" in updated
    assert document["mcp"]["flameox"]["command"][-3:] == ["flameox", "mcp", "serve"]
    repeated = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=30, check=False
    )
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout)["clients"][0]["status"] == "already_current"
    assert config.read_text() == updated


def test_capture_preserve_restart_and_replay_native_evidence(tmp_path: Path) -> None:
    store = tmp_path / "evidence"
    preview = run_cli(
        store,
        "capture",
        "--provider",
        "direct",
        "--cwd",
        str(tmp_path),
        "--",
        sys.executable,
        "-c",
        "print('ephemeral')",
    )
    assert preview["blocks"][1]["rows"][0]["text"] == "ephemeral"
    assert not store.exists()

    captured = run_cli(
        store,
        "capture",
        "--provider",
        "direct",
        "--cwd",
        str(tmp_path),
        "--limits",
        '{"max_rows":1}',
        "--preserve",
        "--",
        sys.executable,
        "-c",
        "print('first'); print('second')",
    )
    evidence_id = captured["preserved"]["evidence_id"]
    assert captured["capture"]["executions"][0]["status"] == "succeeded"
    assert captured["blocks"][1]["rows"][0]["text"] == "first"
    assert "analysis_id" not in captured
    assert evidence_id in captured["next_page"]["argv"]

    # Every call starts a fresh installed CLI process; no session handle can survive.
    resumed = run_cli(store, *captured["next_page"]["argv"])
    assert resumed["blocks"][1]["rows"][0]["text"] == "second"
    manifest = run_cli(store, "evidence", "show", evidence_id)
    assert manifest["evidence_id"] == evidence_id
    assert manifest["body"]["capability_id"] == "artifact.preview"
    native = b"first\nsecond\n"
    digest = hashlib.sha256(native).hexdigest()
    assert (store / "artifacts" / "sha256" / digest[:2] / digest / "payload").read_bytes() == native

    inventory = run_cli(store, "evidence", "query", "--capability", "artifact.preview")
    assert [item["evidence_id"] for item in inventory["evidence"]] == [evidence_id]
    replay = run_cli(store, "analyze", "artifact.preview", "--evidence", evidence_id)
    assert [row["text"] for row in replay["blocks"][1]["rows"]] == ["first", "second"]


def test_failed_capture_preserves_native_output_across_restart(tmp_path: Path) -> None:
    store = tmp_path / "evidence"
    captured = run_cli(
        store,
        "capture",
        "--provider",
        "direct",
        "--cwd",
        str(tmp_path),
        "--preserve",
        "--",
        sys.executable,
        "-c",
        "print('partial result'); raise SystemExit(7)",
        exit_code=1,
    )
    evidence_id = captured["preserved"]["evidence_id"]
    assert captured["capture"]["executions"][0]["status"] == "failed"
    manifest = run_cli(store, "evidence", "show", evidence_id)
    execution = manifest["body"]["capture_request"]["executions"][0]
    assert execution["status"] == "failed"
    assert execution["returncode"] == execution["workload_returncode"] == 7
    assert execution["returncode_scope"] == "workload"
    replay = run_cli(store, "analyze", "artifact.preview", "--evidence", evidence_id)
    assert replay["blocks"][1]["rows"][0]["text"] == "partial result"


def test_native_coverage_capture_survives_cli_to_mcp_handoff(tmp_path: Path) -> None:
    store = tmp_path / "evidence"
    script = tmp_path / "workload.py"
    script.write_text(
        "def double(value):\n    return value * 2\n\n"
        "assert double(3) == 6\nprint('semantic oracle passed')\n"
    )
    hostile_config = "[run]\nomit = *\n"
    (tmp_path / ".coveragerc").write_text(hostile_config)
    captured = run_cli(
        store,
        "capture",
        "--provider",
        "coverage",
        "--capability",
        "coverage.summary",
        "--cwd",
        str(tmp_path),
        "--capture-arguments",
        json.dumps({"branch": True, "source": [str(tmp_path)]}),
        "--preserve",
        "--",
        sys.executable,
        str(script),
    )
    evidence_id = captured["preserved"]["evidence_id"]
    assert captured["capture"]["executions"][0]["status"] == "succeeded"
    assert captured["provider"]["id"] == "coverage.py"
    assert captured["blocks"][0]["values"]["line_count"] == 4
    assert captured["blocks"][0]["values"]["arc_count"] > 0
    assert not (tmp_path / ".coverage").exists()
    assert (tmp_path / ".coveragerc").read_text() == hostile_config

    async def replay_over_stdio() -> None:
        parameters = StdioServerParameters(
            command=str(Path(sys.executable).with_name("flameox")),
            args=["mcp", "serve"],
            cwd=tmp_path,
            env={"FLAMEOX_DATA_DIR": str(store)},
        )
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            await session.initialize()
            marker = tmp_path / "invalid-request-executed"
            target = {
                "argv": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
                "cwd": str(tmp_path),
            }
            experiment = {
                "cases": [{"name": "baseline"}, {"name": "candidate"}],
                "blocks": 1,
                "seed": 1,
                "metric": "wall_time_ns",
                "estimand": "median_difference",
                "practical_threshold": 0,
            }
            invalid_targets = [
                {**target, "cwd": "."},
                {**target, "environment": {"PYTHONPATH": str(tmp_path)}},
                {**target, "environment": {"BAD=NAME": "value"}},
                {**target, "argv": [sys.executable, "bad\x00argument"]},
            ]
            invalid_experiments = [
                {**experiment, "seed": 2**53},
                {**experiment, "semantic_oracle": ["bad\x00command"]},
                {
                    **experiment,
                    "cases": [
                        {"name": "baseline", "argv": ["bad\x00argument"]},
                        {"name": "candidate"},
                    ],
                },
                {
                    **experiment,
                    "cases": [
                        {"name": "baseline", "environment": {"BAD=NAME": "value"}},
                        {"name": "candidate"},
                    ],
                },
            ]
            requests = [{"target": invalid} for invalid in invalid_targets] + [
                {"target": target, "experiment": invalid} for invalid in invalid_experiments
            ]
            for invalid in requests:
                rejected = await session.call_tool(
                    "capture_artifact_preview",
                    {"provider": {"kind": "direct"}, **invalid},
                )
                assert rejected.is_error is True
                assert rejected.structured_content is not None
                assert rejected.structured_content["code"] == "INVALID_REQUEST"
                assert not marker.exists()
            inspected = await session.call_tool("inspect_evidence", {"evidence_id": evidence_id})
            await session.validate_tool_result("inspect_evidence", inspected)
            assert inspected.is_error is False
            content = inspected.content[0]
            assert isinstance(content, TextContent)
            projection = json.loads(content.text)
            assert str(tmp_path) not in content.text
            assert "capture_argv" not in content.text
            unknown_source = await session.call_tool(
                "preview_artifact",
                {"sources": [{"kind": "mystery", "path": str(script)}]},
            )
            assert unknown_source.is_error is True
            assert unknown_source.structured_content is not None
            assert unknown_source.structured_content["code"] == "INVALID_REQUEST"
            replay = await session.call_tool(
                "summarize_coverage",
                {"sources": projection["analysis_sources"]},
            )
            assert replay.is_error is False
            assert replay.structured_content is not None
            assert replay.structured_content["blocks"] == captured["blocks"]
            await session.validate_tool_result("summarize_coverage", replay)

    anyio.run(replay_over_stdio)


def test_node_cpu_capture_and_saved_reanalysis_never_rerun_workload(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for native CPU profile capture")
    store = tmp_path / "evidence"
    marker = tmp_path / "workload-runs"
    workload = tmp_path / "workload.js"
    workload.write_text(
        "const fs = require('node:fs');\n"
        f"fs.appendFileSync({json.dumps(str(marker))}, 'run\\n');\n"
        "function work() {\n"
        "  const end = Date.now() + 100;\n"
        "  let total = 0;\n"
        "  while (Date.now() < end) total += Math.sqrt(total + 1);\n"
        "  return total;\n"
        "}\nconsole.log(work());\n"
    )

    async def exercise() -> tuple[str, list[dict[str, Any]]]:
        parameters = StdioServerParameters(
            command=str(Path(sys.executable).with_name("flameox")),
            args=["mcp", "serve"],
            cwd=tmp_path,
            env={"FLAMEOX_DATA_DIR": str(store)},
        )
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            await session.initialize()
            capture_arguments = {
                "target": {"argv": [node, str(workload)], "cwd": str(tmp_path)},
                "provider": {"kind": "node-cpu-profile"},
                "preserve": True,
            }
            rejected_capture = await session.call_tool(
                "capture_cpu_hotspots",
                {**capture_arguments, "metric": "self_time_seconds"},
            )
            assert rejected_capture.is_error is True
            assert rejected_capture.structured_content is not None
            assert rejected_capture.structured_content["code"] == "INVALID_INPUT"
            assert not marker.exists()

            captured = await session.call_tool("capture_cpu_hotspots", capture_arguments)
            await session.validate_tool_result("capture_cpu_hotspots", captured)
            assert captured.is_error is False
            assert captured.structured_content is not None
            payload = captured.structured_content
            assert payload["provider"]["id"] == "v8-cpu-profile"
            assert payload["blocks"][0]["values"]["sample_count"] > 0
            assert payload["capture"]["executions"][0]["status"] == "succeeded"
            inline = captured.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == payload
            assert marker.read_text() == "run\n"
            evidence_id = payload["preserved"]["evidence_id"]
            inspected = await session.call_tool("inspect_evidence", {"evidence_id": evidence_id})
            assert inspected.structured_content is not None
            sources = inspected.structured_content["analysis_sources"]
            rejected_analysis = await session.call_tool(
                "rank_cpu_hotspots", {"sources": sources, "metric": "self_time_seconds"}
            )
            assert rejected_analysis.is_error is True
            assert rejected_analysis.structured_content is not None
            assert rejected_analysis.structured_content["code"] == "INVALID_INPUT"
            assert marker.read_text() == "run\n"

            replayed = await session.call_tool("rank_cpu_hotspots", {"sources": sources})
            await session.validate_tool_result("rank_cpu_hotspots", replayed)
            assert replayed.is_error is False
            assert replayed.structured_content is not None
            assert replayed.structured_content["blocks"] == payload["blocks"]
            assert marker.read_text() == "run\n"
            return evidence_id, payload["blocks"]

    evidence_id, blocks = anyio.run(exercise)
    restarted = run_cli(store, "analyze", "cpu.hotspots", "--evidence", evidence_id)
    assert restarted["blocks"] == blocks
    assert marker.read_text() == "run\n"


@pytest.mark.optional
@pytest.mark.requires_torch
def test_torch_cpu_capture_replays_all_trace_projections_without_rerunning(tmp_path: Path) -> None:
    binary = (
        os.environ.get("FLAMEOX_TRACE_PROCESSOR")
        or shutil.which("trace_processor_shell")
        or shutil.which("trace_processor")
    )
    if binary is None:
        pytest.skip("A local Perfetto Trace Processor executable is required")
    store = tmp_path / "evidence"
    marker = tmp_path / "workload-runs"
    workload = tmp_path / "torch_workload.py"
    workload.write_text(
        "import torch\n"
        "from pathlib import Path\n"
        "from flameox.sdk import torch_profiler\n"
        f"with Path({str(marker)!r}).open('a') as stream: stream.write('run\\n')\n"
        "left = torch.rand((64, 64))\nright = torch.rand((64, 64))\n"
        "with torch_profiler() as session:\n"
        "    for _ in range(2):\n"
        "        with session.phase('multiply'):\n"
        "            result = left @ right\n"
        "        session.step()\n"
    )

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=str(Path(sys.executable).with_name("flameox")),
            args=["mcp", "serve"],
            cwd=tmp_path,
            env={"FLAMEOX_DATA_DIR": str(store), "FLAMEOX_TRACE_PROCESSOR": binary},
        )
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            await session.initialize()
            captured = await session.call_tool(
                "capture_trace_pytorch",
                {
                    "target": {"argv": [sys.executable, str(workload)], "cwd": str(tmp_path)},
                    "provider": {
                        "kind": "torch-profiler",
                        "activities": ["cpu"],
                        "record_shapes": True,
                    },
                    "preserve": True,
                },
            )
            await session.validate_tool_result("capture_trace_pytorch", captured)
            assert not captured.is_error
            assert captured.structured_content is not None
            payload = captured.structured_content
            assert payload["status"] == "complete"
            assert any(row["name"] == "aten::mm" for row in payload["blocks"][1]["rows"])
            inspected = await session.call_tool(
                "inspect_evidence", {"evidence_id": payload["preserved"]["evidence_id"]}
            )
            assert inspected.structured_content is not None
            for tool in (
                "summarize_trace",
                "inspect_trace_call_graph",
                "summarize_pytorch_trace",
                "inspect_trace_window",
            ):
                arguments = {"sources": inspected.structured_content["analysis_sources"]}
                if tool == "inspect_trace_window":
                    arguments.update(start_ns=0, end_ns=2**53 - 1)
                replayed = await session.call_tool(tool, arguments)
                await session.validate_tool_result(tool, replayed)
                assert not replayed.is_error
                assert replayed.structured_content is not None
                assert replayed.structured_content["blocks"][1]["rows"]
                if tool == "summarize_pytorch_trace":
                    assert replayed.structured_content["blocks"] == payload["blocks"]
                assert marker.read_text() == "run\n"

    anyio.run(exercise)
