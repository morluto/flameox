from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp_types import TextContent

from flameox.mcp import create_server
from flameox.repository import EvidenceRepository


@pytest.mark.integration
def test_mcp_validation_unavailable_provider_and_failed_execution_are_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")

    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            invalid = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(artifact)}],
                        "options": {"unexpected": True},
                    }
                },
            )
            assert invalid.is_error is True
            # MCP SDK 2.0 rejects schema-invalid input before invoking the tool.
            assert invalid.structured_content is None

            for arguments in (
                {"page_size": 0},
                {"page_size": 201},
                {"input_sha256": "not-a-digest"},
            ):
                invalid_query = await client.call_tool("query_evidence", arguments)
                assert invalid_query.is_error is True
                assert invalid_query.structured_content is None
            for limit in (1, 200):
                valid_query = await client.call_tool("query_evidence", {"page_size": limit})
                assert valid_query.is_error is False
                assert valid_query.structured_content is not None

            empty_path = tmp_path / "empty-bin"
            empty_path.mkdir()
            unmanaged_python = empty_path / "python"
            unmanaged_python.symlink_to(sys.executable)
            monkeypatch.setattr("flameox.runtime.sys.executable", str(unmanaged_python))
            unavailable = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "cpu.hotspots",
                        "target": {
                            "argv": [sys.executable, "-c", "pass"],
                            "cwd": str(tmp_path),
                            "environment": {"PATH": str(empty_path)},
                        },
                        "provider": {"kind": "py-spy"},
                        "options": {},
                    }
                },
            )
            assert unavailable.is_error is True
            assert unavailable.structured_content["code"] == "UNAVAILABLE_CAPABILITY"
            assert unavailable.structured_content["details"] == {
                "provider_id": "py-spy",
                "preparation_tool": "prepare_providers",
                "provider_ids": ["py-spy"],
            }

            failed = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "direct"},
                        "options": {},
                    }
                },
            )
            assert failed.is_error is True
            assert failed.structured_content["code"] == "EXECUTION_FAILURE"
            partial = failed.structured_content["details"]["partial_evidence"]
            assert partial["analysis_id"]
            assert partial["capture"]["executions"][0]["returncode"] == 7

    anyio.run(exercise)


@pytest.mark.integration
@pytest.mark.process
@pytest.mark.parametrize("exit_code", [0, 7])
def test_mcp_one_run_capture_needs_no_execution_choice_and_exposes_its_handle(
    tmp_path: Path, exit_code: int
) -> None:
    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "provider": {"kind": "direct"},
                        "target": {
                            "argv": [sys.executable, "-c", f"print('evidence'); exit({exit_code})"],
                            "cwd": str(tmp_path),
                        },
                    }
                },
            )
            assert result.is_error is (exit_code != 0)
            value = result.structured_content
            partial = value["details"]["partial_evidence"] if exit_code else value
            assert partial["capture"]["mode"] == "single"
            assert len(partial["capture"]["executions"]) == 1
            assert partial["capture"]["executions"][0]["returncode"] == exit_code
            summary = result.content[0]
            assert isinstance(summary, TextContent)
            assert partial["analysis_id"] in summary.text
            assert "preserve" in summary.text
            preserved = await client.call_tool(
                "preserve_evidence", {"analysis_id": partial["analysis_id"]}
            )
            assert not preserved.is_error

    anyio.run(exercise)


@pytest.mark.integration
@pytest.mark.process
def test_mcp_experiment_design_alone_selects_paired_execution(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "provider": {"kind": "direct"},
                        "target": {
                            "argv": [sys.executable, "-c", "print('verified')"],
                            "cwd": str(tmp_path),
                        },
                        "experiment": {
                            "cases": [{"name": "baseline"}, {"name": "candidate"}],
                            "blocks": 1,
                            "seed": 7,
                            "metric": "wall_time_ns",
                            "estimand": "median_difference",
                            "practical_threshold": 0,
                        },
                    }
                },
            )
            assert not result.is_error, result.content
            capture = result.structured_content["capture"]
            assert capture["mode"] == "experiment"
            assert {item["case"] for item in capture["executions"]} == {"baseline", "candidate"}
            assert capture["outcome"]["succeeded_count"] == 2

    anyio.run(exercise)


@pytest.mark.integration
@pytest.mark.parametrize("unsupported", ["execution", "experiment"])
def test_mcp_rejects_removed_or_incompatible_execution_fields_before_capture(
    tmp_path: Path, unsupported: str
) -> None:
    marker = tmp_path / "executed"

    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "cpu.hotspots",
                        "provider": {"kind": "py-spy"},
                        "target": {
                            "argv": [
                                sys.executable,
                                "-c",
                                f"from pathlib import Path; Path({str(marker)!r}).touch()",
                            ],
                            "cwd": str(tmp_path),
                        },
                        unsupported: {"kind": "single"} if unsupported == "execution" else {},
                    }
                },
            )
            assert result.is_error
            assert unsupported in str(result.content)

    anyio.run(exercise)
    assert not marker.exists()
    assert not (tmp_path / "store").exists()


@pytest.mark.process
@pytest.mark.parametrize("argument_count", [1, 8])
def test_failed_capture_returns_full_provenance_once(tmp_path: Path, argument_count: int) -> None:
    directory = tmp_path / "store"
    arguments = [f"native-argument-{index}:".ljust(16_384, "x") for index in range(argument_count)]

    async def exercise() -> str:
        async with Client(
            create_server(
                evidence_directory=directory,
            )
        ) as client:
            result = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [sys.executable, "-c", "raise SystemExit(7)", *arguments],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "direct"},
                    }
                },
            )
            assert result.is_error
            value = result.structured_content
            assert value is not None, result.content
            assert value["code"] == "EXECUTION_FAILURE"
            summary = result.content[0]
            assert isinstance(summary, TextContent)
            assert "native-argument-" not in summary.text
            assert "structuredContent" in summary.text
            details = value["details"]
            assert "failed_executions" not in details
            analysis = details["partial_evidence"]
            assert analysis["capture"]["executions"][0]["argv"][3:] == arguments
            preserved = await client.call_tool(
                "preserve_evidence", {"analysis_id": analysis["analysis_id"]}
            )
            assert not preserved.is_error
            return str(preserved.structured_content["evidence_id"])

    evidence_id = anyio.run(exercise)
    manifest = EvidenceRepository(directory, "reader").read(evidence_id)
    execution = manifest["body"]["capture_request"]["executions"][0]
    assert execution["argv"][3:] == arguments
    assert execution["returncode"] == 7
    assert execution["status"] == "failed"
