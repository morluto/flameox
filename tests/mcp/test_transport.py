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

from flameox import __version__
from flameox.mcp import create_server
from flameox.providers.cpu import CpuProfileProvider
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE
from flameox.runtime import AnalysisRuntime
from flameox.runtime_errors import DomainError, ErrorCode


@pytest.mark.integration
def test_mcp_errors_do_not_expose_untrusted_paths_or_provider_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_path = "/private/customer/workspace/profile.json"
    private_diagnostic = "decoder stderr contains customer token"
    original_resolve = AnalysisRuntime._resolve_sources

    def fail_read(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(2, "No such file", private_path)

    def fail_provider(*_args: Any, **_kwargs: Any) -> Any:
        raise DomainError(
            ErrorCode.DECODE_FAILURE,
            "CPU profile worker transport failed before a trustworthy response.",
            details={"stderr": private_diagnostic, "path": private_path},
        )

    async def exercise() -> None:
        monkeypatch.setattr(AnalysisRuntime, "_resolve_sources", fail_read)
        async with Client(
            create_server(evidence_directory=tmp_path / "store"), raise_exceptions=True
        ) as client:
            unreadable = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(tmp_path / "input.json")}],
                    }
                },
            )
            monkeypatch.setattr(AnalysisRuntime, "_resolve_sources", original_resolve)
            monkeypatch.setattr(CpuProfileProvider, "analyze", fail_provider)
            profile = tmp_path / "profile.json"
            profile.write_text("{}")
            provider_failure = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "cpu.hotspots",
                        "sources": [{"kind": "path", "path": str(profile), "format": "py-spy"}],
                    }
                },
            )

        for result in (unreadable, provider_failure):
            assert result.is_error is True
            serialized = json.dumps(result.structured_content)
            assert private_path not in serialized
            assert private_diagnostic not in serialized

    anyio.run(exercise)


@pytest.mark.process
@pytest.mark.serial
@pytest.mark.e2e
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
        by_name = {tool.name: tool for tool in tools.tools}
        assert set(by_name) == {
            "inspect_capabilities",
            "prepare_providers",
            "analyze",
            "capture_and_analyze",
            "preserve_evidence",
            "rescue_evidence",
            "query_evidence",
        }
        assert all(tool.output_schema is not None for tool in tools.tools)
        for name in ("analyze", "capture_and_analyze"):
            schema = by_name[name].input_schema
            request_ref = schema["properties"]["request"]["$ref"].rsplit("/", 1)[-1]
            request = schema["$defs"][request_ref]
            assert request["type"] == "object"
            assert request["additionalProperties"] is False
            assert request["properties"]["capability_id"]["enum"]
            assert "capability_id" in request["required"]
        analysis_schema = by_name["analyze"].input_schema
        request_ref = analysis_schema["properties"]["request"]["$ref"].rsplit("/", 1)[-1]
        assert analysis_schema["$defs"][request_ref]["properties"]["sources"]["maxItems"] == 32
        analyze_annotations = by_name["analyze"].annotations
        capture_annotations = by_name["capture_and_analyze"].annotations
        assert analyze_annotations is not None
        assert capture_annotations is not None
        assert analyze_annotations.read_only_hint is True
        assert capture_annotations.destructive_hint is True
        assert invalid.is_error is True
        assert captured.structured_content["blocks"][1]["rows"][0]["text"] == "stdio capture"
        assert resources.resources == []
        assert [item.uri_template for item in templates.resource_templates] == [
            "flameox://evidence/{evidence_id}"
        ]
        assert templates.resource_templates[0].mime_type == AGENT_EVIDENCE_MEDIA_TYPE

    anyio.run(exercise)
