from __future__ import annotations

import json

import anyio
import pytest

from flameox.mcp import create_server
from flameox.mcp.request_contracts import (
    ARTIFACT_FORMATS,
    CAPABILITY_IDS,
    CAPTURE_PROVIDER_IDS,
    PREPARABLE_PROVIDER_IDS,
)
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE


@pytest.mark.unit
def test_mcp_catalog_is_compact_discoverable_and_non_recursive() -> None:
    async def inspect() -> None:
        server = create_server()
        tools = await server.list_tools()
        templates = await server.list_resource_templates()

        assert server.instructions is not None
        assert "inspect_capabilities" in server.instructions
        assert "SARIF" in server.instructions

        assert {tool.name for tool in tools} == {
            "inspect_capabilities",
            "prepare_providers",
            "analyze",
            "capture_and_analyze",
            "preserve_evidence",
            "rescue_evidence",
            "query_evidence",
        }
        assert all(tool.output_schema is not None for tool in tools)
        by_name = {tool.name: tool for tool in tools}

        serialized = json.dumps([tool.model_dump(mode="json") for tool in tools])
        # The catalog remains bounded after adding all typed recovery actions.
        assert len(serialized) < 100_000

        analyze = by_name["analyze"]
        capture = by_name["capture_and_analyze"]
        inspect_tool = by_name["inspect_capabilities"]
        analysis_request = analyze.input_schema["$defs"]["AnalysisRequest"]
        capture_request = capture.input_schema["$defs"]["CaptureRequest"]
        assert analysis_request["properties"]["capability_id"]["enum"] == list(CAPABILITY_IDS)
        assert capture_request["properties"]["capability_id"]["enum"] == list(CAPABILITY_IDS)
        assert analysis_request["properties"]["sources"]["maxItems"] == 32
        assert "examples" in analyze.input_schema
        assert "examples" in capture.input_schema
        assert "examples" in inspect_tool.input_schema

        provider = capture.input_schema["$defs"]["CaptureProviderRequest"]
        assert provider["properties"]["kind"]["enum"] == list(CAPTURE_PROVIDER_IDS)
        path_source = analyze.input_schema["$defs"]["McpPathSource"]
        format_schema = path_source["properties"]["format"]["anyOf"][0]
        assert format_schema["enum"] == list(ARTIFACT_FORMATS)
        prepare_ids = by_name["prepare_providers"].input_schema["properties"]["provider_ids"]
        assert prepare_ids["items"]["enum"] == list(PREPARABLE_PROVIDER_IDS)

        output = analyze.output_schema
        assert output is not None
        next_page = output["$defs"]["ToolCallEnvelope"]
        assert next_page["properties"]["arguments"]["type"] == "object"
        assert "AnalysisRequest" not in next_page["properties"]["arguments"].get("$ref", "")

        query_output = by_name["query_evidence"].output_schema
        assert query_output is not None
        query = query_output["$defs"]["QueryEnvelope"]
        assert query["properties"]["inventory_status"]["enum"] == [
            "absent",
            "empty",
            "available",
        ]
        evidence = query["properties"]["evidence"]
        assert evidence["items"]["$ref"].endswith("/EvidenceSummaryEnvelope")

        failure = output["$defs"]["ToolFailureEnvelope"]["properties"]
        assert {"code", "retryable", "field_path", "accepted_values", "next_action"} <= set(failure)
        capture_output = capture.output_schema
        assert capture_output is not None
        assert "CaptureExecution" in capture_output["$defs"]
        assert "partial_evidence" not in json.dumps(capture_output)

        assert capture.annotations is not None
        assert capture.annotations.destructive_hint is True
        assert by_name["analyze"].annotations is not None
        assert by_name["analyze"].annotations.read_only_hint is True
        assert await server.list_resources() == []
        assert [item.uri_template for item in templates] == ["flameox://evidence/{evidence_id}"]
        assert templates[0].mime_type == AGENT_EVIDENCE_MEDIA_TYPE

    anyio.run(inspect)
