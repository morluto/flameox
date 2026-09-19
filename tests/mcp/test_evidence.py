from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
import pytest
from coverage import CoverageData
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp_types import TextContent, TextResourceContents

from flameox.mcp import create_server
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    MAX_ROWS,
    PathSource,
)


@pytest.mark.process
def test_mcp_terminal_provider_limit_recommends_recovery_not_preservation(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("\n" * (MAX_ROWS + 204))
    artifact = tmp_path / ".coverage"
    data = CoverageData(basename=str(artifact))
    data.add_lines({str(source): set(range(1, MAX_ROWS + 205))})
    data.write()

    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            arguments: dict[str, Any] = {
                "request": {
                    "capability_id": "coverage.summary",
                    "sources": [
                        {
                            "kind": "path",
                            "path": str(artifact),
                            "format": "coverage",
                        }
                    ],
                },
                "page_size": 100,
            }
            terminal = None
            for _ in range(12):
                terminal = await client.call_tool("analyze", arguments)
                next_page = terminal.structured_content.get("next_page")
                if next_page is None:
                    break
                arguments = next_page["arguments"]

        assert terminal is not None
        assert terminal.structured_content["truncation"]["reason"] == "provider_limit"
        summary = terminal.content[0]
        assert isinstance(summary, TextContent)
        assert "no continuation is available" in summary.text
        assert "narrow" in summary.text
        assert "bounded evidence" in summary.text
        assert "preserve" not in summary.text

    anyio.run(exercise)


@pytest.mark.integration
def test_mcp_continuation_summary_names_safe_analysis_handoff(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            captured = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [
                                sys.executable,
                                "-c",
                                "[print(index) for index in range(4)]",
                            ],
                            "cwd": str(tmp_path),
                            "console_output": "full",
                        },
                        "provider": {"kind": "direct"},
                    },
                    "page_size": 2,
                },
            )
            preserved = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [
                                sys.executable,
                                "-c",
                                "[print(index) for index in range(4)]",
                            ],
                            "cwd": str(tmp_path),
                            "console_output": "full",
                        },
                        "provider": {"kind": "direct"},
                        "preserve": True,
                    },
                    "page_size": 2,
                },
            )
            captured_next = captured.structured_content["next_page"]
            captured_second = await client.call_tool(
                captured_next["tool"], captured_next["arguments"]
            )
            preserved_next = preserved.structured_content["next_page"]
            preserved_second = await client.call_tool(
                preserved_next["tool"], preserved_next["arguments"]
            )

        assert isinstance(captured.content[0], TextContent)
        assert "call analyze with the exact next_page arguments" in captured.content[0].text
        assert "do not rerun capture" in captured.content[0].text
        assert [row["text"] for row in captured_second.structured_content["blocks"][1]["rows"]] == [
            "2",
            "3",
        ]
        assert isinstance(preserved.content[0], TextContent)
        assert "call analyze with the exact next_page arguments" in preserved.content[0].text
        assert "do not rerun capture" in preserved.content[0].text
        assert preserved_next["arguments"]["request"]["sources"][0]["kind"] == "evidence"
        assert [
            row["text"] for row in preserved_second.structured_content["blocks"][1]["rows"]
        ] == ["2", "3"]

    anyio.run(exercise)


def test_mcp_keeps_complete_large_continuation_arguments(tmp_path: Path) -> None:
    artifact = tmp_path / "candidates.sarif"
    artifact.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "scanner"}},
                        "results": [
                            {
                                "ruleId": f"candidate-{index}",
                                "message": {"text": "candidate"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": f"src/work-{index}.py"}
                                        }
                                    }
                                ],
                            }
                            for index in range(2)
                        ],
                    }
                ],
            }
        )
    )
    include_paths = ["src/*", *[f"unused/{index}-{'x' * 240}*" for index in range(127)]]

    async def exercise() -> None:
        async with Client(
            create_server(
                evidence_directory=tmp_path / ".flameox",
            ),
            raise_exceptions=True,
        ) as client:
            result = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "static.performance_candidates",
                        "sources": [{"kind": "path", "path": str(artifact), "format": "sarif"}],
                        "options": {"include_paths": include_paths},
                    },
                    "page_size": 1,
                },
            )

        assert result.is_error is not True
        handoff = result.structured_content["next_page"]
        assert handoff["tool"] == "analyze"
        assert handoff["arguments"]["request"]["options"]["include_paths"] == include_paths
        assert result.structured_content["truncation"]["reason"] == "row_limit"
        summary = result.content[0]
        assert isinstance(summary, TextContent)
        assert "call analyze with the exact next_page arguments" in summary.text

    anyio.run(exercise)


def test_mcp_preservation_refreshes_a_live_capture_handoff(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            captured = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [sys.executable, "-c", "print('one'); print('two')"],
                            "cwd": str(tmp_path),
                            "console_output": "full",
                        },
                        "provider": {"kind": "direct"},
                    },
                    "page_size": 1,
                },
            )
            preserved = await client.call_tool(
                "preserve_evidence",
                {"analysis_id": captured.structured_content["analysis_id"]},
            )
            refreshed = preserved.structured_content["next_page"]
            second = await client.call_tool(refreshed["tool"], refreshed["arguments"])

        assert refreshed["arguments"]["request"]["sources"][0]["kind"] == "evidence"
        assert isinstance(preserved.content[0], TextContent)
        assert "refreshed evidence-backed next_page" in preserved.content[0].text
        assert second.structured_content["blocks"][1]["rows"][0]["text"] == "two"

    anyio.run(exercise)


def test_mcp_failed_capture_retains_an_executable_preserved_handoff(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            failed = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [
                                sys.executable,
                                "-c",
                                "print('one'); print('two'); raise SystemExit(7)",
                            ],
                            "cwd": str(tmp_path),
                            "console_output": "full",
                        },
                        "provider": {"kind": "direct"},
                        "preserve": True,
                    },
                    "page_size": 1,
                },
            )
            partial = failed.structured_content
            next_page = partial["next_page"]
            second = await client.call_tool(next_page["tool"], next_page["arguments"])

        assert failed.is_error is False
        assert partial["status"] == "partial"
        assert next_page["arguments"]["request"]["sources"][0]["kind"] == "evidence"
        assert second.structured_content["blocks"][1]["rows"][0]["text"] == "two"

    anyio.run(exercise)


def test_mcp_handoffs_are_derived_while_the_runtime_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "rows.json"
    artifact.write_text('[{"value":1},{"value":2}]')
    original = AnalysisRuntime.next_analysis_request
    observations: list[bool] = []

    def checked(runtime: AnalysisRuntime, result: Mapping[str, Any]) -> dict[str, Any] | None:
        observations.append(runtime._request_lock.locked())
        return original(runtime, result)

    monkeypatch.setattr(AnalysisRuntime, "next_analysis_request", checked)

    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            analyzed = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(artifact)}],
                    },
                    "page_size": 1,
                },
            )
            await client.call_tool(
                "preserve_evidence",
                {"analysis_id": analyzed.structured_content["analysis_id"]},
            )
            await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [sys.executable, "-c", "print('captured')"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "direct"},
                    }
                },
            )

    anyio.run(exercise)
    assert observations == [True, True, True]


@pytest.mark.skipif(sys.platform == "darwin", reason="requires descriptor-backed directory aliases")
def test_mcp_rescue_returns_a_restart_safe_next_page(tmp_path: Path) -> None:
    store = tmp_path / "store"
    rescue = tmp_path / "rescue"

    async def exercise() -> None:
        async with Client(create_server(evidence_directory=store), raise_exceptions=True) as client:
            captured = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [sys.executable, "-c", "print('one'); print('two')"],
                            "cwd": str(tmp_path),
                            "console_output": "full",
                        },
                        "provider": {"kind": "direct"},
                    },
                    "page_size": 1,
                },
            )
            rescued = await client.call_tool(
                "rescue_evidence",
                {
                    "analysis_id": captured.structured_content["analysis_id"],
                    "destination": str(rescue),
                },
            )
            next_page = rescued.structured_content["next_page"]

        async with Client(
            create_server(evidence_directory=rescue), raise_exceptions=True
        ) as client:
            second = await client.call_tool(next_page["tool"], next_page["arguments"])

        assert next_page["arguments"]["request"]["sources"][0]["kind"] == "evidence"
        assert second.structured_content["blocks"][1]["rows"][0]["text"] == "two"

    anyio.run(exercise)


def test_mcp_query_returns_an_exact_next_page(tmp_path: Path) -> None:
    store = tmp_path / ".flameox"
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        for index in range(2):
            artifact = tmp_path / f"query-{index}.json"
            artifact.write_text(json.dumps([{"value": index}]))
            result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            runtime.preserve_evidence(result["analysis_id"])
    finally:
        runtime.close()

    async def exercise() -> None:
        async with Client(create_server(evidence_directory=store), raise_exceptions=True) as client:
            first = await client.call_tool(
                "query_evidence", {"capability_id": "artifact.preview", "page_size": 1}
            )
            next_page = first.structured_content["next_page"]
            second = await client.call_tool(next_page["tool"], next_page["arguments"])

            first_summary = first.content[0]
            second_summary = second.content[0]
            assert isinstance(first_summary, TextContent)
            assert isinstance(second_summary, TextContent)
            assert "Evidence query partial" in first_summary.text
            assert "exact next_page arguments" in first_summary.text
            assert "Evidence query complete" in second_summary.text

        assert next_page["arguments"]["capability_id"] == "artifact.preview"
        assert next_page["arguments"]["page_size"] == 1
        assert second.structured_content.get("next_page") is None
        assert (
            first.structured_content["inventory_digest"]
            == second.structured_content["inventory_digest"]
        )

    anyio.run(exercise)


@pytest.mark.integration
def test_analysis_preservation_query_resource_and_restart(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1},{"value":2}]')

    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            analyzed = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(artifact)}],
                        "options": {},
                    }
                },
            )
            assert analyzed.is_error is False
            analysis_id = analyzed.structured_content["analysis_id"]
            preserved = await client.call_tool("preserve_evidence", {"analysis_id": analysis_id})
            assert preserved.is_error is False
            evidence_id = preserved.structured_content["evidence_id"]
            assert any(block.type == "resource_link" for block in preserved.content)
            queried = await client.call_tool("query_evidence", {"page_size": 10})
            assert queried.structured_content["evidence"][0]["evidence_id"] == evidence_id
            resource = await client.read_resource(f"flameox://evidence/{evidence_id}")
            assert resource.contents[0].mime_type == AGENT_EVIDENCE_MEDIA_TYPE

        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as restarted:
            reanalyzed = await restarted.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [
                            {
                                "kind": "evidence",
                                "evidence_id": evidence_id,
                                "artifact_role": "input",
                            }
                        ],
                        "options": {},
                    }
                },
            )
            assert reanalyzed.is_error is False
            assert reanalyzed.structured_content["blocks"][1]["rows"][0]["value"] == 1
            expired = await restarted.call_tool("preserve_evidence", {"analysis_id": analysis_id})
            assert expired.is_error is True
            assert expired.structured_content["code"] == "EXPIRED_SESSION_ANALYSIS"
            resource = await restarted.read_resource(f"flameox://evidence/{evidence_id}")
            content = resource.contents[0]
            assert isinstance(content, TextResourceContents)
            assert json.loads(content.text)["evidence_id"] == evidence_id
            with pytest.raises(MCPError):
                await restarted.read_resource(f"flameox://evidence/{'0' * 64}")

    anyio.run(exercise)


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "darwin", reason="requires descriptor-backed directory aliases")
def test_mcp_rescues_live_analysis_from_unusable_configured_store(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "unexpected").write_text("corrupt")
    rescue = tmp_path / "rescue"
    artifact = tmp_path / "input.json"
    artifact.write_text('[{"value": 1}]')

    async def exercise() -> str:
        async with Client(
            create_server(evidence_directory=configured), raise_exceptions=True
        ) as client:
            analyzed = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(artifact)}],
                    }
                },
            )
            rescued = await client.call_tool(
                "rescue_evidence",
                {
                    "analysis_id": analyzed.structured_content["analysis_id"],
                    "destination": str(rescue),
                },
            )
            assert rescued.is_error is False
            assert not any(block.type == "resource_link" for block in rescued.content)
            assert rescued.structured_content["next_action"]["environment"] == {
                "FLAMEOX_DATA_DIR": str(rescue)
            }
            return str(rescued.structured_content["evidence_id"])

    evidence_id = anyio.run(exercise)
    reopened = AnalysisRuntime(evidence_directory=rescue)
    try:
        assert reopened.read_evidence(evidence_id)["evidence_id"] == evidence_id
    finally:
        reopened.close()


@pytest.mark.integration
def test_mcp_evidence_resource_redacts_capture_provenance(tmp_path: Path) -> None:
    secret_argument = "known-safe-argument-placeholder"
    secret_environment = "known-safe-environment-placeholder"
    secret_path = str(tmp_path.resolve())

    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            captured = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [sys.executable, "-c", "print('ok')", secret_argument],
                            "cwd": str(tmp_path),
                            "environment": {"FLAMEOX_TEST_MARKER": secret_environment},
                        },
                        "provider": {"kind": "direct"},
                        "options": {},
                    }
                },
            )
            assert captured.is_error is False
            preserved = await client.call_tool(
                "preserve_evidence",
                {"analysis_id": captured.structured_content["analysis_id"]},
            )
            evidence_id = preserved.structured_content["evidence_id"]
            resource = await client.read_resource(f"flameox://evidence/{evidence_id}")
            content = resource.contents[0]
            assert isinstance(content, TextResourceContents)
            projection = content.text
            assert secret_argument not in projection
            assert secret_environment not in projection
            assert secret_path not in projection
            assert '"argv"' not in projection
            assert '"capture_argv"' not in projection
            assert '"cwd"' not in projection
            assert '"environment"' not in projection

            # The MCP projection must not weaken the canonical local provenance record.
            canonical = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
            try:
                manifest = json.dumps(canonical.read_evidence(evidence_id))
            finally:
                canonical.close()
            assert secret_argument in manifest
            assert secret_environment in manifest
            assert secret_path in manifest

    anyio.run(exercise)
