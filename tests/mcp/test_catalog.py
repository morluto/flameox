from __future__ import annotations

import anyio
import pytest

from flameox.mcp import create_server
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE
from flameox.runtime_contracts import (
    CAPABILITIES,
    MAX_ROWS,
    compatible_capture_providers,
)


@pytest.mark.unit
def test_mcp_tools_consolidate_typed_capabilities_without_losing_contracts() -> None:
    async def inspect() -> None:
        server = create_server()
        tools = await server.list_tools()
        templates = await server.list_resource_templates()

        assert server.instructions is not None
        assert "request.experiment" in server.instructions

        assert {tool.name for tool in tools} == {
            "prepare_providers",
            "analyze",
            "capture_and_analyze",
            "preserve_evidence",
            "rescue_evidence",
            "query_evidence",
        }
        assert all(tool.output_schema is not None for tool in tools)
        by_name = {tool.name: tool for tool in tools}
        analysis_schema = by_name["analyze"].input_schema
        capture_schema = by_name["capture_and_analyze"].input_schema
        analysis_request = analysis_schema["properties"]["request"]
        capture_request = capture_schema["properties"]["request"]
        assert analysis_schema["required"] == ["request"]
        assert capture_schema["required"] == ["request"]
        assert analysis_request["discriminator"]["propertyName"] == "capability_id"
        assert capture_request["discriminator"]["propertyName"] == "capability_id"
        analysis_mapping = analysis_request["discriminator"]["mapping"]
        capture_mapping = capture_request["discriminator"]["mapping"]
        assert set(analysis_mapping) == {capability.id for capability in CAPABILITIES}
        assert set(capture_mapping) == {
            capability.id for capability in CAPABILITIES if compatible_capture_providers(capability)
        }
        analysis_definitions = {
            capability_id: analysis_schema["$defs"][reference.rsplit("/", 1)[-1]]
            for capability_id, reference in analysis_mapping.items()
        }
        capture_definitions = {
            capability_id: capture_schema["$defs"][reference.rsplit("/", 1)[-1]]
            for capability_id, reference in capture_mapping.items()
        }

        for capability in CAPABILITIES:
            analysis_definition = analysis_definitions[capability.id]
            sources = analysis_definition["properties"]["sources"]
            assert sources["minItems"] == capability.minimum_sources
            assert sources["maxItems"] == capability.maximum_sources
            assert analysis_definition["additionalProperties"] is False
            assert capability.summary in analysis_definition["description"]
            assert analysis_definition["properties"]["options"]["$ref"].endswith(
                f"/{capability.model.__name__}"
            )
            if capability.id not in capture_mapping:
                continue
            capture_definition = capture_definitions[capability.id]
            expected_providers = {
                provider.id for provider in compatible_capture_providers(capability)
            }
            provider_schema = capture_definition["properties"]["provider"]
            if len(expected_providers) == 1:
                provider_name = provider_schema["$ref"].rsplit("/", 1)[-1]
                assert capture_schema["$defs"][provider_name]["properties"]["kind"]["const"] in (
                    expected_providers
                )
            else:
                assert set(provider_schema["discriminator"]["mapping"]) == expected_providers
            assert "execution" not in capture_definition["properties"]
            if capability.maximum_sources == 1:
                assert "experiment" not in capture_definition["properties"]
            else:
                experiment_schema = capture_definition["properties"]["experiment"]
                assert experiment_schema["default"] is None
                assert experiment_schema["anyOf"][0]["$ref"].endswith("/ExperimentDesign")

        preview = analysis_definitions["artifact.preview"]
        comparison = analysis_definitions["benchmark.compare"]
        trace_window = analysis_definitions["trace.window"]
        assert preview["properties"]["sources"]["maxItems"] == 32
        assert comparison["properties"]["sources"]["minItems"] == 2
        assert set(preview["required"]) == {"capability_id", "sources"}
        assert set(trace_window["required"]) == {"capability_id", "sources", "options"}
        cpu_options_ref = analysis_definitions["cpu.hotspots"]["properties"]["options"]["$ref"]
        cpu_options = analysis_schema["$defs"][cpu_options_ref.rsplit("/", 1)[-1]]["properties"]
        assert set(cpu_options["metric"]["anyOf"][0]["enum"]) == {
            "self_time_seconds",
            "cumulative_time_seconds",
            "total_calls",
            "primitive_calls",
        }

        cpu_capture = capture_definitions["cpu.hotspots"]
        provider = cpu_capture["properties"]["provider"]
        provider_mapping = provider["discriminator"]["mapping"]
        assert set(provider_mapping) == {"node-cpu-profile", "perf", "py-spy"}
        for provider_id, reference in provider_mapping.items():
            definition = capture_schema["$defs"][reference.rsplit("/", 1)[-1]]
            assert definition["properties"]["kind"]["const"] == provider_id
        assert set(cpu_capture["required"]) == {"capability_id", "target", "provider"}
        assert "experiment" not in cpu_capture["properties"]
        benchmark = capture_definitions["benchmark.summary"]
        assert "experiment" not in benchmark["required"]
        target_properties = capture_schema["$defs"]["DirectTarget"]["properties"]
        assert "without a shell" in target_properties["argv"]["description"]
        assert "existing absolute directory" in target_properties["cwd"]["description"].lower()
        assert "minimal allowlisted environment" in target_properties["environment"]["description"]
        assert "limits" not in preview["properties"]
        page_size = analysis_schema["properties"]["page_size"]
        assert {key: page_size[key] for key in ("default", "maximum", "minimum", "type")} == {
            "default": 100,
            "maximum": MAX_ROWS,
            "minimum": 1,
            "type": "integer",
        }

        analysis_output = by_name["analyze"].output_schema
        assert analysis_output is not None
        output_properties = analysis_output["$defs"]["AnalysisEnvelope"]["properties"]
        assert set(output_properties) >= {
            "analysis_id",
            "capability_id",
            "inputs",
            "blocks",
            "coverage",
            "truncation",
            "limitations",
            "continuation",
            "next_page",
        }
        next_arguments = analysis_output["$defs"]["AnalyzeArgumentsEnvelope"]
        assert next_arguments["additionalProperties"] is False
        assert set(next_arguments["required"]) == {"request", "page_size"}
        next_request_schema = next_arguments["properties"]["request"]
        assert next_request_schema["discriminator"]["propertyName"] == "capability_id"
        assert set(next_request_schema["discriminator"]["mapping"]) == {
            capability.id for capability in CAPABILITIES
        }
        rescue_output = by_name["rescue_evidence"].output_schema
        assert rescue_output is not None
        rescue_properties = rescue_output["$defs"]["RescueEnvelope"]["properties"]
        assert set(rescue_properties) >= {
            "evidence_id",
            "uri",
            "artifact_count",
            "rescue_destination",
            "next_action",
            "next_page",
        }
        assert "alternate evidence store" in (
            rescue_output["$defs"]["RescueEnvironmentEnvelope"]["properties"]["FLAMEOX_DATA_DIR"][
                "description"
            ].lower()
        )
        query_output = by_name["query_evidence"].output_schema
        assert query_output is not None
        query_properties = query_output["$defs"]["QueryEnvelope"]["properties"]
        assert "continuation" not in query_properties
        assert "Exact query_evidence call" in query_properties["next_page"]["description"]

        assert by_name["analyze"].annotations is not None
        assert by_name["analyze"].annotations.read_only_hint is True
        assert by_name["capture_and_analyze"].annotations is not None
        assert by_name["capture_and_analyze"].annotations.destructive_hint is True
        assert by_name["capture_and_analyze"].annotations.open_world_hint is True
        prepare = by_name["prepare_providers"]
        assert prepare.annotations is not None
        assert prepare.annotations.open_world_hint is True
        assert prepare.annotations.read_only_hint is False
        assert prepare.annotations.destructive_hint is False

        query = by_name["query_evidence"]
        assert query.input_schema["properties"]["page_size"]["minimum"] == 1
        assert query.input_schema["properties"]["page_size"]["maximum"] == 200
        assert await server.list_resources() == []
        assert [item.uri_template for item in templates] == ["flameox://evidence/{evidence_id}"]
        assert templates[0].mime_type == AGENT_EVIDENCE_MEDIA_TYPE

    anyio.run(inspect)
