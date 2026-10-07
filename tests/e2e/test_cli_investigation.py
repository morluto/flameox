from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp_types import TextResourceContents

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
                    "capture_and_analyze",
                    {
                        "request": {
                            "capability_id": "artifact.preview",
                            "provider": {"kind": "direct"},
                            **invalid,
                        }
                    },
                )
                assert rejected.is_error is True
                assert rejected.structured_content is not None
                assert rejected.structured_content["code"] == "INVALID_REQUEST"
                assert not marker.exists()
            resource = await session.read_resource(f"flameox://evidence/{evidence_id}")
            content = resource.contents[0]
            assert isinstance(content, TextResourceContents)
            projection = json.loads(content.text)
            assert str(tmp_path) not in content.text
            assert "capture_argv" not in content.text
            unknown_source = await session.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "mystery", "path": str(script)}],
                    }
                },
            )
            assert unknown_source.is_error is True
            assert unknown_source.structured_content is not None
            assert unknown_source.structured_content["code"] == "INVALID_REQUEST"
            replay = await session.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "coverage.summary",
                        "sources": projection["analysis_sources"],
                    }
                },
            )
            assert replay.is_error is False
            assert replay.structured_content is not None
            assert replay.structured_content["blocks"] == captured["blocks"]
            await session.validate_tool_result("analyze", replay)

    anyio.run(replay_over_stdio)
