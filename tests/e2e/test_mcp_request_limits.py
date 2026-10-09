from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

pytestmark = [pytest.mark.e2e, pytest.mark.process, pytest.mark.serial]


def test_stdio_inherits_startup_limits_and_preserves_request_limits_on_replay(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "rows.json"
    artifact.write_text('[{"value":1},{"value":2},{"value":3}]')
    oversized = tmp_path / "oversized.json"
    oversized.write_text('["' + "x" * 1_200 + '"]')
    marker = tmp_path / "started"
    target = {
        "argv": [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('started').touch(); print('first\\nsecond\\nthird')",
        ],
        "cwd": str(tmp_path),
    }

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "flameox",
                "mcp",
                "serve",
                "--limits",
                '{"max_rows":2,"max_input_bytes":4096,"max_output_bytes":8192}',
            ],
            cwd=tmp_path,
            env={**os.environ, "FLAMEOX_DATA_DIR": str(tmp_path / "store")},
        )
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            await session.initialize()

            async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                result = await session.call_tool(name, arguments)
                await session.validate_tool_result(name, result)
                assert result.structured_content is not None
                return cast(dict[str, Any], result.structured_content)

            preview = await call("preview_artifact", {"sources": [{"path": str(artifact)}]})
            assert [row["value"] for row in preview["blocks"][1]["rows"]] == [1, 2]

            for arguments in (
                {"limits": {"max_output_bytes": 16_384}},
                {"page_size": 3},
                {"page_size": 1, "limits": {"max_rows": 2}},
            ):
                rejected = await call(
                    "capture_artifact_preview",
                    {"target": target, "provider": {"kind": "direct"}, **arguments},
                )
                assert rejected["code"] in {"LIMIT_EXCEEDED", "INVALID_REQUEST"}
                assert not marker.exists()
                if arguments == {"page_size": 3}:
                    assert rejected["next_action"]["kind"] == "adjust_request"
                    assert rejected["next_action"]["field_path"] == ["page_size"]
                    assert "page_size=2" in rejected["next_action"]["message"]

            shorthand = {"sources": [{"path": str(artifact)}], "page_size": 3.0}
            rejected_shorthand = await call("preview_artifact", shorthand)
            retry_shorthand = dict(shorthand)
            retry_shorthand["page_size"] = rejected_shorthand["details"]["safe_retry"]["max_rows"]
            recovered_shorthand = await call("preview_artifact", retry_shorthand)
            assert recovered_shorthand["coverage"]["rows_returned"] == 2

            both_limits = {
                "sources": [{"path": str(artifact)}],
                "page_size": 3,
                "limits": {"max_rows": 3},
            }
            rejected_both = await call("preview_artifact", both_limits)
            assert rejected_both["next_action"]["field_path"] == ["page_size"]
            assert "page_size=2 and limits.max_rows=2" in rejected_both["next_action"]["message"]
            retry_both = dict(both_limits)
            retry_both["page_size"] = rejected_both["details"]["safe_retry"]["max_rows"]
            retry_both["limits"] = {"max_rows": rejected_both["details"]["safe_retry"]["max_rows"]}
            recovered_both = await call("preview_artifact", retry_both)
            assert recovered_both["coverage"]["rows_returned"] == 2

            rejected_input = await call(
                "preview_artifact",
                {"sources": [{"path": str(oversized)}], "limits": {"max_input_bytes": 1024}},
            )
            assert rejected_input["code"] == "LIMIT_EXCEEDED"
            inherited = await call(
                "capture_artifact_preview", {"target": target, "provider": {"kind": "direct"}}
            )
            assert inherited["coverage"]["rows_returned"] == 2

            captured = await call(
                "capture_artifact_preview",
                {
                    "target": target,
                    "provider": {"kind": "direct"},
                    "limits": {"max_rows": 1, "max_output_bytes": 2048, "max_input_bytes": 1024},
                },
            )
            handoff = captured["next_page"]
            assert handoff["arguments"]["limits"]["max_output_bytes"] == 2048
            second = await call(handoff["tool"], handoff["arguments"])
            assert second["blocks"][1]["rows"][0]["text"] == "second"
            preserved = await call("preserve_evidence", {"analysis_id": captured["analysis_id"]})
            handoff = preserved["next_page"]
            replayed = await call(handoff["tool"], handoff["arguments"])
            assert replayed["blocks"] == second["blocks"]

    anyio.run(exercise)
