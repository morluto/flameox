from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

from flameox import __version__
from flameox.mcp import create_server
from flameox.runtime import AnalysisRuntime
from flameox.setup import ExternalRequirement, ProviderPreparation, ProviderSelectionFailure


@pytest.mark.unit
def test_mcp_analysis_does_not_expose_unexpected_exception_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_path = "/private/agent-workspace/customer-secret.json"

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise OSError(2, "No such file", secret_path)

    monkeypatch.setattr(AnalysisRuntime, "analyze", fail)

    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            result = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(tmp_path / "input.json")}],
                    }
                },
            )

        assert result.is_error is True
        assert result.structured_content["code"] == "DECODE_FAILURE"
        assert result.structured_content["message"] == "Input could not be read during analysis."
        assert secret_path not in json.dumps(result.structured_content)

    anyio.run(exercise)


@pytest.mark.unit
def test_mcp_analysis_wraps_unexpected_provider_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("private provider state")

    monkeypatch.setattr(AnalysisRuntime, "analyze", fail)

    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            result = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(tmp_path / "input.json")}],
                    }
                },
            )

        assert result.is_error is True
        assert result.structured_content["code"] == "ANALYSIS_FAILURE"
        assert result.structured_content["message"] == "Analysis failed unexpectedly."
        assert "private provider state" not in json.dumps(result.structured_content)

    anyio.run(exercise)


@pytest.mark.unit
def test_mcp_prepares_managed_providers_and_only_guides_host_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparation_calls: list[list[str]] = []

    async def prepare(
        self: Any, provider_ids: list[str], timeout_seconds: int
    ) -> ProviderPreparation:
        assert timeout_seconds == 1_800
        if provider_ids == ["unknown-provider"]:
            raise ProviderSelectionFailure("Unknown provider 'unknown-provider'")
        preparation_calls.append(provider_ids)
        if provider_ids == ["nsight-compute"]:
            return ProviderPreparation(
                ["nsight-compute"],
                [],
                [
                    ExternalRequirement(
                        "nsight-compute",
                        "Install NVIDIA Nsight Compute with its extras/python interface.",
                    )
                ],
                [],
                "uvx",
                [
                    "--python",
                    "3.12",
                    "--from",
                    f"flameox=={__version__}",
                    "flameox",
                    "mcp",
                    "serve",
                ],
            )
        return ProviderPreparation(
            ["memray", "nsight-compute"],
            ["memray"],
            [
                ExternalRequirement(
                    "nsight-compute",
                    "Install NVIDIA Nsight Compute with its extras/python interface.",
                )
            ],
            ["/usr/bin/uvx", "--from", f"flameox[memory]=={__version__}", "--version"],
            "uvx",
            [
                "--python",
                "3.12",
                "--from",
                f"flameox[memory]=={__version__}",
                "flameox",
                "mcp",
                "serve",
            ],
        )

    monkeypatch.setattr("flameox.providers.preparation.ProviderDependencies.prepare", prepare)

    async def exercise() -> None:
        async with Client(
            create_server(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            result = await client.call_tool(
                "prepare_providers",
                {"provider_ids": ["memray", "nsight-compute", "memray"]},
            )
            host_only = await client.call_tool(
                "prepare_providers",
                {"provider_ids": ["nsight-compute"]},
            )
            invalid = await client.call_tool(
                "prepare_providers",
                {"provider_ids": ["unknown-provider"]},
            )

        assert result.is_error is False
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert "Provider preparation completed" in result.content[0].text
        assert "requested_providers" not in result.content[0].text
        assert result.structured_content["requested_providers"] == [
            "memray",
            "nsight-compute",
        ]
        assert result.structured_content["prepared_managed_providers"] == ["memray"]
        assert result.structured_content["external_requirements"] == [
            {
                "provider_id": "nsight-compute",
                "guidance": "Install NVIDIA Nsight Compute with its extras/python interface.",
            }
        ]
        assert result.structured_content["preparation"]["status"] == "prepared"
        assert result.structured_content["next_action"]["kind"] == "reconnect_mcp"
        assert result.structured_content["next_action"]["necessity"] == "conditional"
        assert result.structured_content["next_action"]["launcher"] == {
            "command": "uvx",
            "args": result.structured_content["launcher"]["args"],
        }
        assert "Preserve" in result.structured_content["next_action"]["message"]
        assert "external requirements" in result.content[0].text
        assert result.structured_content["launcher"]["args"][3] == (
            f"flameox[memory]=={__version__}"
        )
        assert result.structured_content["launcher"]["args"][-3:] == [
            "flameox",
            "mcp",
            "serve",
        ]
        assert host_only.is_error is False
        assert host_only.structured_content["next_action"] is None

        assert invalid.is_error is True
        assert invalid.structured_content["code"] == "INVALID_REQUEST"
        assert invalid.structured_content["field_path"] == ["provider_ids", 0]
        assert "py-spy" in invalid.structured_content["accepted_values"]

    anyio.run(exercise)
    assert preparation_calls == [
        ["memray", "nsight-compute", "memray"],
        ["nsight-compute"],
    ]


@pytest.mark.process
@pytest.mark.serial
def test_real_stdio_initialize_and_catalog_match_the_runtime_contract(tmp_path: Path) -> None:
    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "flameox",
                "mcp",
                "serve",
            ],
            cwd=tmp_path,
        )
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            initialized = await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            templates = await session.list_resource_templates()
            artifact = tmp_path / "sample.json"
            artifact.write_text('[{"value":1}]')
            inspected = await session.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(artifact)}],
                    }
                },
            )
            invalid = await session.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [],
                        "options": {},
                        "unexpected": True,
                    }
                },
            )
            captured = await session.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "target": {
                            "argv": [sys.executable, "-c", "print('stdio capture')"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "direct"},
                    }
                },
            )
            await session.validate_tool_result("analyze", inspected)
            await session.validate_tool_result("capture_and_analyze", captured)

        assert initialized.server_info.version == __version__
        assert len(tools.tools) == 7
        assert "inspect_capabilities" in [tool.name for tool in tools.tools]
        assert "analyze" in [tool.name for tool in tools.tools]
        assert "capture_and_analyze" in [tool.name for tool in tools.tools]
        assert all(tool.output_schema is not None for tool in tools.tools)
        assert invalid.is_error is True
        assert captured.structured_content["blocks"][1]["rows"][0]["text"] == "stdio capture"
        assert resources.resources == []
        assert [item.uri_template for item in templates.resource_templates] == [
            "flameox://evidence/{evidence_id}"
        ]

    anyio.run(exercise)
