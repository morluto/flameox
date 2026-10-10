from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from coverage import CoverageData
from mcp import Client
from mcp_types import TextContent

from flameox.mcp.server import FlameoxServer
from flameox.repository import EvidenceRepository
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import MAX_ROWS


@pytest.mark.process
def test_mcp_terminal_provider_limit_is_visible_inline(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("\n" * (MAX_ROWS + 204))
    artifact = tmp_path / ".coverage"
    data = CoverageData(basename=str(artifact))
    data.add_lines({str(source): set(range(1, MAX_ROWS + 205))})
    data.write()

    async def exercise() -> None:
        async with Client(
            FlameoxServer(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            arguments: dict[str, Any] = {
                "sources": [{"kind": "path", "path": str(artifact), "format": "coverage"}],
                "page_size": 100,
            }
            terminal = None
            for _ in range(12):
                terminal = await client.call_tool("summarize_coverage", arguments)
                next_page = terminal.structured_content.get("next_page")
                if next_page is None:
                    break
                arguments = next_page["arguments"]

        assert terminal is not None
        assert terminal.structured_content["truncation"]["reason"] == "provider_limit"
        inline = terminal.content[0]
        assert isinstance(inline, TextContent)
        value = json.loads(inline.text)
        assert value == terminal.structured_content
        assert value["next_page"] is None
        assert value["coverage"]["complete"] is False

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
            FlameoxServer(
                evidence_directory=tmp_path / ".flameox",
            ),
            raise_exceptions=True,
        ) as client:
            result = await client.call_tool(
                "inspect_performance_candidates",
                {
                    "sources": [{"kind": "path", "path": str(artifact), "format": "sarif"}],
                    "include_paths": include_paths,
                    "page_size": 1,
                },
            )

        assert result.is_error is not True
        handoff = result.structured_content["next_page"]
        assert handoff["tool"] == "inspect_performance_candidates"
        assert handoff["arguments"]["include_paths"] == include_paths
        assert result.structured_content["truncation"]["reason"] == "row_limit"
        inline = result.content[0]
        assert isinstance(inline, TextContent)
        assert json.loads(inline.text) == result.structured_content

    anyio.run(exercise)


def test_mcp_rescue_returns_a_restart_safe_next_page(tmp_path: Path) -> None:
    store = tmp_path / "store"
    rescue = tmp_path / "rescue"

    async def exercise() -> None:
        async with Client(FlameoxServer(evidence_directory=store), raise_exceptions=True) as client:
            captured = await client.call_tool(
                "capture_artifact_preview",
                {
                    "target": {
                        "argv": [sys.executable, "-c", "print('one'); print('two')"],
                        "cwd": str(tmp_path),
                        "console_output": "full",
                    },
                    "provider": {"kind": "direct"},
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
            FlameoxServer(evidence_directory=rescue), raise_exceptions=True
        ) as client:
            second = await client.call_tool(next_page["tool"], next_page["arguments"])

        assert next_page["arguments"]["sources"][0]["kind"] == "evidence"
        assert second.structured_content["blocks"][1]["rows"][0]["text"] == "two"

    anyio.run(exercise)


@pytest.mark.integration
def test_mcp_query_filters_providers_and_returns_exact_next_pages(tmp_path: Path) -> None:
    artifact = tmp_path / "provider-filter.json"
    artifact.write_text('[{"value": 1}]')
    store = tmp_path / ".flameox"

    async def exercise() -> None:
        async with Client(FlameoxServer(evidence_directory=store), raise_exceptions=True) as client:
            analyzed = await client.call_tool(
                "preview_artifact",
                {"sources": [{"kind": "path", "path": str(artifact)}]},
            )
            analyzed_evidence = await client.call_tool(
                "preserve_evidence", {"analysis_id": analyzed.structured_content["analysis_id"]}
            )
            analysis_evidence_id = analyzed_evidence.structured_content["evidence_id"]

            capture_ids: list[str] = []
            for _ in range(2):
                captured = await client.call_tool(
                    "capture_artifact_preview",
                    {
                        "target": {
                            "argv": [sys.executable, "-c", "print('captured')"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "direct"},
                        "preserve": True,
                    },
                )
                assert captured.is_error is False
                capture_ids.append(captured.structured_content["preserved"]["evidence_id"])

            all_evidence = await client.call_tool("query_evidence", {})
            records = all_evidence.structured_content["evidence"]
            by_id = {item["evidence_id"]: item for item in records}
            analysis_provider_id = by_id[analysis_evidence_id]["provider"]["id"]

            flameox_matches = await client.call_tool(
                "query_evidence", {"provider_id": analysis_provider_id, "page_size": 10}
            )
            direct_first = await client.call_tool(
                "query_evidence",
                {"provider_id": "direct", "capability_id": "artifact.preview", "page_size": 1},
            )
            next_page = direct_first.structured_content["next_page"]
            assert next_page is not None
            direct_second = await client.call_tool(next_page["tool"], next_page["arguments"])
            assert next_page["arguments"]["provider_id"] == "direct"
            assert next_page["arguments"]["capability_id"] == "artifact.preview"
            assert next_page["arguments"]["page_size"] == 1
            inline = direct_first.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == direct_first.structured_content

            changed_filter = await client.call_tool(
                "query_evidence",
                {
                    **next_page["arguments"],
                    "provider_id": analysis_provider_id,
                },
            )
            unknown = await client.call_tool(
                "query_evidence", {"provider_id": "provider-that-does-not-exist"}
            )

        assert {item["evidence_id"] for item in flameox_matches.structured_content["evidence"]} == {
            analysis_evidence_id,
            *capture_ids,
        }
        assert {
            item["evidence_id"]
            for item in direct_first.structured_content["evidence"]
            + direct_second.structured_content["evidence"]
        } == set(capture_ids)
        assert direct_second.structured_content["next_page"] is None
        assert changed_filter.is_error is True
        assert changed_filter.structured_content["code"] == "INVALID_INPUT"
        assert unknown.structured_content["match_status"] == "no_matches"
        assert unknown.structured_content["evidence"] == []

    anyio.run(exercise)


@pytest.mark.integration
def test_analysis_preservation_query_inspection_and_restart(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1},{"value":2}]')

    async def exercise() -> None:
        async with Client(
            FlameoxServer(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            analyzed = await client.call_tool(
                "preview_artifact",
                {"sources": [{"kind": "path", "path": str(artifact)}]},
            )
            assert analyzed.is_error is False
            analysis_id = analyzed.structured_content["analysis_id"]
            preserved = await client.call_tool("preserve_evidence", {"analysis_id": analysis_id})
            assert preserved.is_error is False
            evidence_id = preserved.structured_content["evidence_id"]
            assert all(block.type == "text" for block in preserved.content)
            assert "uri" not in preserved.structured_content
            queried = await client.call_tool("query_evidence", {"page_size": 10})
            assert queried.structured_content["evidence"][0]["evidence_id"] == evidence_id
            inspected = await client.call_tool("inspect_evidence", {"evidence_id": evidence_id})
            assert inspected.is_error is False
            assert inspected.structured_content["evidence_id"] == evidence_id

        async with Client(
            FlameoxServer(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as restarted:
            reanalyzed = await restarted.call_tool(
                "preview_artifact",
                {
                    "sources": [
                        {"kind": "evidence", "evidence_id": evidence_id, "artifact_role": "input"}
                    ]
                },
            )
            assert reanalyzed.is_error is False
            assert reanalyzed.structured_content["blocks"][1]["rows"][0]["value"] == 1
            expired = await restarted.call_tool("preserve_evidence", {"analysis_id": analysis_id})
            assert expired.is_error is True
            assert expired.structured_content["code"] == "EXPIRED_SESSION_ANALYSIS"
            assert expired.structured_content["next_action"]["kind"] == "operator_action"
            assert "query_evidence" in expired.structured_content["next_action"]["message"]
            assert (
                "analysis_id alone cannot recover"
                in expired.structured_content["next_action"]["message"]
            )
            inspected = await restarted.call_tool("inspect_evidence", {"evidence_id": evidence_id})
            content = inspected.content[0]
            assert isinstance(content, TextContent)
            assert json.loads(content.text) == inspected.structured_content
            assert json.loads(content.text)["evidence_id"] == evidence_id
            missing = await restarted.call_tool("inspect_evidence", {"evidence_id": "0" * 64})
            assert missing.is_error is True
            assert missing.structured_content["code"] == "MISSING_EVIDENCE"

    anyio.run(exercise)


@pytest.mark.integration
def test_mcp_rescues_live_analysis_from_unusable_configured_store(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "unexpected").write_text("corrupt")
    rescue = tmp_path / "rescue"
    artifact = tmp_path / "input.json"
    artifact.write_text('[{"value": 1}]')

    async def exercise() -> str:
        async with Client(
            FlameoxServer(evidence_directory=configured), raise_exceptions=True
        ) as client:
            analyzed = await client.call_tool(
                "preview_artifact",
                {"sources": [{"kind": "path", "path": str(artifact)}]},
            )
            rescued = await client.call_tool(
                "rescue_evidence",
                {
                    "analysis_id": analyzed.structured_content["analysis_id"],
                    "destination": str(rescue),
                },
            )
            assert rescued.is_error is False
            inline = rescued.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == rescued.structured_content
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
def test_mcp_evidence_inspection_redacts_capture_provenance(tmp_path: Path) -> None:
    secret_argument = "known-safe-argument-placeholder"
    secret_environment = "known-safe-environment-placeholder"
    secret_path = str(tmp_path.resolve())

    async def exercise() -> None:
        async with Client(
            FlameoxServer(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            captured = await client.call_tool(
                "capture_artifact_preview",
                {
                    "target": {
                        "argv": [sys.executable, "-c", "print('ok')", secret_argument],
                        "cwd": str(tmp_path),
                        "environment": {"FLAMEOX_TEST_MARKER": secret_environment},
                    },
                    "provider": {"kind": "direct"},
                },
            )
            assert captured.is_error is False
            preserved = await client.call_tool(
                "preserve_evidence",
                {"analysis_id": captured.structured_content["analysis_id"]},
            )
            evidence_id = preserved.structured_content["evidence_id"]
            inspected = await client.call_tool("inspect_evidence", {"evidence_id": evidence_id})
            assert inspected.is_error is False
            content = inspected.content[0]
            assert isinstance(content, TextContent)
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


@pytest.mark.integration
def test_query_distinguishes_unavailable_empty_and_no_matches(tmp_path: Path) -> None:
    unavailable_store = tmp_path / "unavailable"
    empty_store = tmp_path / "empty"
    EvidenceRepository(empty_store, "test").initialize()

    async def query(directory: Path) -> dict[str, object]:
        async with Client(FlameoxServer(evidence_directory=directory)) as client:
            result = await client.call_tool("query_evidence", {})
            assert result.is_error is False
            return cast(dict[str, object], result.structured_content)

    async def populated_no_matches(directory: Path) -> dict[str, object]:
        artifact = tmp_path / "artifact.json"
        artifact.write_text("[1]")
        async with Client(FlameoxServer(evidence_directory=directory)) as client:
            analyzed = await client.call_tool(
                "preview_artifact",
                {"sources": [{"kind": "path", "path": str(artifact)}]},
            )
            await client.call_tool(
                "preserve_evidence",
                {"analysis_id": analyzed.structured_content["analysis_id"]},
            )
            result = await client.call_tool(
                "query_evidence", {"capability_id": "retired.capability"}
            )
            return cast(dict[str, object], result.structured_content)

    unavailable = anyio.run(query, unavailable_store)
    empty = anyio.run(query, empty_store)
    no_matches = anyio.run(populated_no_matches, tmp_path / "populated")
    assert unavailable["inventory_status"] == "absent"
    assert unavailable["inventory_size"] == 0
    assert empty["inventory_status"] == "empty"
    assert empty["inventory_size"] == 0
    assert empty["match_status"] == "no_matches"
    assert no_matches["inventory_status"] == "available"
    assert no_matches["inventory_size"] == 1
    assert no_matches["match_status"] == "no_matches"
