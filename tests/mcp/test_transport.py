from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

import anyio
import pytest
from jsonschema import Draft202012Validator
from mcp import Client, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

from flameox import __version__
from flameox.mcp.catalog import ANALYSIS_TOOLS, CAPTURE_TOOLS
from flameox.mcp.server import FlameoxServer
from flameox.providers.cpu import CpuProfileProvider
from flameox.runtime import AnalysisRuntime
from flameox.runtime_errors import DomainError, ErrorCode


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["analysis_io", "lifecycle_io", "provider"])
def test_mcp_failures_classify_operations_without_exposing_private_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    private_path = "/private/customer/workspace/profile.json"
    private_diagnostic = "decoder stderr contains customer token"
    error: Exception = OSError(2, "No such file", private_path)
    tool, arguments, expected_code = "preview_artifact", {}, "DECODE_FAILURE"
    owner: Any = AnalysisRuntime
    method = "_resolve_sources"
    if failure == "lifecycle_io":
        tool, method, expected_code = "query_evidence", "query_evidence", "IO_FAILURE"
    else:
        profile = tmp_path / "profile.json"
        profile.write_text("{}")
        arguments = {"sources": [{"path": str(profile)}]}
        if failure == "provider":
            arguments["sources"][0]["format"] = "py-spy"
            tool, owner, method = "rank_cpu_hotspots", CpuProfileProvider, "analyze"
            error = DomainError(
                ErrorCode.DECODE_FAILURE,
                "CPU profile worker failed before a trustworthy response.",
                details={"stderr": private_diagnostic, "path": private_path},
            )

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(owner, method, fail)

    async def exercise() -> None:
        async with Client(
            FlameoxServer(evidence_directory=tmp_path / "store"), raise_exceptions=True
        ) as client:
            result = await client.call_tool(tool, arguments)
            assert result.is_error is True
            assert result.structured_content["code"] == expected_code
            assert private_path not in result.model_dump_json()
            assert private_diagnostic not in result.model_dump_json()

    anyio.run(exercise)


@pytest.mark.integration
def test_catalog_projections_are_independent_of_caller_mutation() -> None:
    async def exercise() -> None:
        server = FlameoxServer()
        first = await server.list_tools()
        baseline = [tool.model_dump(mode="json") for tool in first]
        for tool in first:
            tool.input_schema.clear()
            assert tool.output_schema is not None
            tool.output_schema.clear()
            assert tool.annotations is not None
            tool.annotations.read_only_hint = not tool.annotations.read_only_hint
        assert [tool.model_dump(mode="json") for tool in await server.list_tools()] == baseline
        assert [
            tool.model_dump(mode="json") for tool in await FlameoxServer().list_tools()
        ] == baseline

    anyio.run(exercise)


@pytest.mark.integration
def test_mcp_rejects_invalid_runtime_results_before_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_projection(_self: AnalysisRuntime, _evidence_id: str) -> dict[str, Any]:
        return {
            "format_version": "3",
            "evidence_id": "0" * 64,
            "analysis_sources": [],
            "logical_sources": [],
            "body": {"invalid": object()},
        }

    monkeypatch.setattr(AnalysisRuntime, "read_evidence_agent_projection", invalid_projection)

    async def exercise() -> None:
        async with Client(FlameoxServer(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool("inspect_evidence", {"evidence_id": "0" * 64})
            assert result.is_error is True
            assert result.structured_content["code"] == "INTERNAL_CONTRACT_FAILURE"
            inline = result.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == result.structured_content

    anyio.run(exercise)


@pytest.mark.process
@pytest.mark.serial
@pytest.mark.e2e
def test_stdio_exposes_direct_tools_and_runs_typed_validation_and_capture(tmp_path: Path) -> None:
    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "flameox", "mcp", "serve"],
            cwd=tmp_path,
            env={"FLAMEOX_DATA_DIR": str(tmp_path / "store")},
        )
        artifact = tmp_path / "sample.json"
        artifact.write_text('[{"value":1},{"value":2},{"value":3}]')
        marker = tmp_path / "started.txt"
        command = (
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).open('a').write('started\\n'); "
            "print('stdio capture\\nsecond row\\nthird row'); sys.exit(int(sys.argv[2]))"
        )

        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            initialized = await session.initialize()
            listed = await session.list_tools()
            assert initialized.capabilities.resources is None
            by_name = {tool.name: tool for tool in listed.tools}

            assert initialized.server_info.version == __version__
            assert initialized.instructions and "native artifacts" in initialized.instructions
            expected = (
                set(ANALYSIS_TOOLS.values())
                | set(CAPTURE_TOOLS.values())
                | {
                    "prepare_providers",
                    "preserve_evidence",
                    "rescue_evidence",
                    "query_evidence",
                    "inspect_evidence",
                }
            )
            assert set(by_name) == expected
            assert not {"inspect_capabilities", "analyze", "capture_and_analyze"} & set(by_name)
            assert all(tool.output_schema for tool in listed.tools)
            for tool in listed.tools:
                Draft202012Validator.check_schema(tool.input_schema)
                assert tool.output_schema is not None
                Draft202012Validator.check_schema(tool.output_schema)
                input_validator = Draft202012Validator(tool.input_schema)
                for example in tool.input_schema.get("examples", []):
                    assert input_validator.is_valid(example), (tool.name, example)
            for name in ANALYSIS_TOOLS.values():
                assert by_name[name].input_schema["additionalProperties"] is False
                assert "sources" in by_name[name].input_schema["required"]

            evidence_source = {"kind": "evidence", "evidence_id": "0" * 64}

            query_schema = Draft202012Validator(by_name["query_evidence"].input_schema)
            iso_query = {"created_after": "2026-01-01T00:00:00Z"}
            assert query_schema.is_valid(iso_query)
            queried = await session.call_tool("query_evidence", iso_query)
            await session.validate_tool_result("query_evidence", queried)
            inline = queried.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == queried.structured_content
            assert queried.is_error is False
            timestamp_query = {"created_after": 0}
            assert not query_schema.is_valid(timestamp_query)
            invalid_timestamp = await session.call_tool("query_evidence", timestamp_query)
            await session.validate_tool_result("query_evidence", invalid_timestamp)
            assert invalid_timestamp.is_error is True
            assert invalid_timestamp.structured_content["code"] == "INVALID_REQUEST"
            assert invalid_timestamp.structured_content["field_path"] == ["created_after"]
            assert invalid_timestamp.structured_content["next_action"]["kind"] == "adjust_request"
            reversed_query = {
                "created_after": "2026-01-01T00:00:00Z",
                "created_before": "2025-01-01T00:00:00Z",
            }
            assert query_schema.is_valid(reversed_query)
            reversed_result = await session.call_tool("query_evidence", reversed_query)
            await session.validate_tool_result("query_evidence", reversed_result)
            assert reversed_result.is_error is True
            assert reversed_result.structured_content["code"] == "INVALID_REQUEST"
            assert reversed_result.structured_content["field_path"] == ["created_before"]
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest,
            )

            epoch_ns = 1_791_720_000_000_000_000
            otlp = ExportTraceServiceRequest()
            scope = otlp.resource_spans.add().scope_spans.add()
            for index in range(1, 4):
                span = scope.spans.add()
                span.trace_id = bytes.fromhex("01" * 16)
                span.span_id = index.to_bytes(8, "big")
                span.name = f"epoch-span-{index}"
                span.start_time_unix_nano = epoch_ns + index * 100
                span.end_time_unix_nano = span.start_time_unix_nano + 10
            trace = tmp_path / "epoch.otlp"
            trace.write_bytes(otlp.SerializeToString())
            window_schema = Draft202012Validator(by_name["inspect_trace_window"].input_schema)
            window_args: dict[str, Any] = {
                "sources": [{"path": str(trace), "format": "otlp"}],
                "start_ns": epoch_ns + 150,
                "end_ns": epoch_ns + 250,
                "page_size": 2,
            }
            for as_strings in (False, True):
                bounds = {
                    key: str(value) if as_strings and key in {"start_ns", "end_ns"} else value
                    for key, value in window_args.items()
                }
                assert window_schema.is_valid(bounds)
                window = await session.call_tool("inspect_trace_window", bounds)
                await session.validate_tool_result("inspect_trace_window", window)
                assert not window.is_error
                kept = await session.call_tool(
                    "preserve_evidence", {"analysis_id": window.structured_content["analysis_id"]}
                )
                await session.validate_tool_result("preserve_evidence", kept)
                assert not kept.is_error
                next_page = kept.structured_content["next_page"]
                assert next_page["arguments"]["start_ns"] == str(epoch_ns + 150)
                assert next_page["arguments"]["end_ns"] == str(epoch_ns + 250)
                continued = await session.call_tool(next_page["tool"], next_page["arguments"])
                await session.validate_tool_result(next_page["tool"], continued)
                assert not continued.is_error
                rows = continued.structured_content["blocks"][1]["rows"]
                assert [row["name"] for row in rows] == ["epoch-span-2"]
                assert rows[0]["start_time_unix_nano"] == str(epoch_ns + 200)
            integral_window = {**window_args, "start_ns": 2.0}
            assert window_schema.is_valid(integral_window)
            integral = await session.call_tool("inspect_trace_window", integral_window)
            await session.validate_tool_result("inspect_trace_window", integral)
            assert not integral.is_error
            for invalid_bound in (
                True,
                "01",
                "+1",
                "1.0",
                " 1",
                "\u0661",
                "1\n",
                f"{epoch_ns}\n",
                2**63,
                str(2**63),
            ):
                invalid_window = {**window_args, "start_ns": invalid_bound}
                assert not window_schema.is_valid(invalid_window)
                rejected = await session.call_tool("inspect_trace_window", invalid_window)
                await session.validate_tool_result("inspect_trace_window", rejected)
                assert rejected.is_error
                assert rejected.structured_content["field_path"] == ["start_ns"]
            preview = await session.call_tool(
                "preview_artifact",
                {"sources": [{"path": str(artifact)}], "page_size": 2.0},
            )
            assert Draft202012Validator(by_name["preview_artifact"].input_schema).is_valid(
                {"sources": [{"path": str(artifact)}], "page_size": 2}
            )
            await session.validate_tool_result("preview_artifact", preview)
            inline = preview.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == preview.structured_content
            assert preview.is_error is False
            previewed_values = [
                row["value"] for row in preview.structured_content["blocks"][1]["rows"]
            ]
            assert previewed_values == [1, 2]

            invalid_calls: tuple[tuple[str, dict[str, Any], list[str | int] | None], ...] = (
                ("preview_artifact", {}, None),
                ("inspect_evidence", {"evidence_id": "bad-id"}, ["evidence_id"]),
                ("inspect_evidence", {"evidence_id": "0" * 64, "uri": "removed"}, ["uri"]),
                ("preview_artifact", {"sources": []}, None),
                (
                    "preview_artifact",
                    {"sources": [{"path": str(artifact)}], "offset": 2**53},
                    ["offset"],
                ),
                (
                    "preview_artifact",
                    {"sources": [{"path": "/tmp/impossible\x00path"}]},
                    ["sources", 0, "path"],
                ),
                (
                    "rescue_evidence",
                    {"analysis_id": "0" * 64, "destination": "/tmp/impossible\x00path"},
                    ["destination"],
                ),
                (
                    "capture_artifact_preview",
                    {
                        "target": {"argv": [sys.executable], "cwd": "/tmp/impossible\x00path"},
                        "provider": {"kind": "direct"},
                    },
                    ["target", "cwd"],
                ),
                (
                    "preview_artifact",
                    {"sources": [{"path": str(artifact)}], "page_size": 0},
                    None,
                ),
                (
                    "preview_artifact",
                    {"sources": [{"path": str(artifact)}], "page_size": "2"},
                    ["page_size"],
                ),
                (
                    "preview_artifact",
                    {"sources": [{"path": str(artifact)}], "page_size": True},
                    ["page_size"],
                ),
                (
                    "preview_artifact",
                    {
                        "sources": [
                            {
                                **evidence_source,
                                "artifact_role": "stdout",
                                "artifact_selector": "1" * 64,
                            }
                        ]
                    },
                    ["sources", 0],
                ),
                (
                    "preview_artifact",
                    {"sources": [{"path": 1}]},
                    ["sources", 0, "path"],
                ),
                (
                    "preview_artifact",
                    {"sources": [{"kind": "evidence", "evidence_id": "bad"}]},
                    ["sources", 0, "evidence_id"],
                ),
                (
                    "preview_artifact",
                    {"request": {"capability_id": "artifact.preview", "sources": []}},
                    None,
                ),
                (
                    "preview_artifact",
                    {"sources": [{"path": str(artifact)}], "options": {}},
                    None,
                ),
                (
                    "rank_cpu_hotspots",
                    {
                        "sources": [
                            {"path": str(artifact)},
                            {"path": str(artifact)},
                        ]
                    },
                    None,
                ),
                (
                    "rank_cpu_hotspots",
                    {"sources": [{"path": str(artifact)}], "metric": "bogus"},
                    None,
                ),
                (
                    "rank_cpu_hotspots",
                    {"sources": [{"path": str(artifact), "format": "json"}]},
                    ["sources", 0, "format"],
                ),
                (
                    "inspect_trace_window",
                    {"sources": [{"path": str(artifact)}]},
                    None,
                ),
                (
                    "capture_benchmark_summary",
                    {
                        "target": {"argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path)},
                        "provider": {"kind": "pyperf", "name": "bad\x00name"},
                    },
                    ["provider", "name"],
                ),
                (
                    "capture_cpu_hotspots",
                    {
                        "target": {"argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path)},
                        "provider": {"kind": "py-spy", "rate": 0},
                    },
                    ["provider", "rate"],
                ),
                (
                    "capture_artifact_preview",
                    {
                        "target": {
                            "argv": [sys.executable, "-c", command, str(marker), "0"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "not-a-provider"},
                    },
                    None,
                ),
                (
                    "capture_artifact_preview",
                    {
                        "target": {
                            "argv": [sys.executable, "-c", command, str(marker), "0"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "py-spy"},
                    },
                    None,
                ),
                (
                    "capture_artifact_preview",
                    {
                        "target": {
                            "argv": [sys.executable, "-c", command, str(marker), "0"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "direct"},
                        "page_size": "1",
                    },
                    ["page_size"],
                ),
                (
                    "capture_artifact_preview",
                    {
                        "target": {
                            "argv": [sys.executable, "-c", command, str(marker), "0"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "direct"},
                        "page_size": True,
                    },
                    ["page_size"],
                ),
            )
            invalid_calls += tuple(
                (
                    "capture_artifact_preview",
                    {
                        "target": {"argv": argv, "cwd": str(tmp_path)},
                        "provider": {"kind": "direct"},
                    },
                    None,
                )
                for argv in ([""], [sys.executable, "bad\x00argument"], ["x" * 16_385])
            )
            invalid_calls += tuple(
                (
                    "capture_cpu_hotspots",
                    {
                        "target": {
                            "argv": [sys.executable, "-c", command, str(marker), "0"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": "py-spy"},
                        unsupported: {},
                    },
                    [unsupported],
                )
                for unsupported in ("execution", "experiment")
            )
            invalid_calls += tuple(
                (
                    "inspect_performance_candidates",
                    {"sources": [{"path": str(artifact)}], "include_paths": [pattern]},
                    ["include_paths", 0],
                )
                for pattern in ("", "x" * 257, "bad\x00pattern")
            )
            invalid_calls += tuple(
                (
                    tool,
                    {
                        "target": {
                            "argv": [sys.executable, "-c", command, str(marker), "0"],
                            "cwd": str(tmp_path),
                        },
                        "provider": {"kind": kind, field: [value]},
                    },
                    None,
                )
                for tool, kind, field, maximum in (
                    ("capture_gpu_kernel_metrics", "nsight-compute", "section", 200),
                    ("capture_coverage_summary", "coverage", "source", 256),
                    ("capture_coverage_summary", "coverage", "include", 256),
                    ("capture_coverage_summary", "coverage", "omit", 256),
                )
                for value in ("", "x" * (maximum + 1), "bad\x00selector")
            )
            for tool_name, arguments, expected_field_path in invalid_calls:
                assert not Draft202012Validator(by_name[tool_name].input_schema).is_valid(
                    arguments
                ), (tool_name, arguments)
                result = await session.call_tool(tool_name, arguments)
                await session.validate_tool_result(tool_name, result)
                inline = result.content[0]
                assert isinstance(inline, TextContent)
                assert json.loads(inline.text) == result.structured_content
                assert result.is_error is True
                assert result.structured_content["code"] == "INVALID_REQUEST"
                if expected_field_path is not None:
                    assert result.structured_content["field_path"] == expected_field_path
                    action = result.structured_content["next_action"]
                    assert action["kind"] == "adjust_request"
                    assert action["field_path"] == expected_field_path
                    assert action["message"] == result.structured_content["message"]
                if tool_name == "rank_cpu_hotspots" and arguments["sources"][0].get("format"):
                    assert "pstats" in result.structured_content["accepted_values"]
            assert not marker.exists(), (
                "invalid capture requests must fail before workload execution"
            )

            evidence_source_arguments = {
                "sources": [{"evidence_id": "0" * 64}],
            }
            assert Draft202012Validator(by_name["preview_artifact"].input_schema).is_valid(
                evidence_source_arguments
            )
            missing_evidence = await session.call_tool(
                "preview_artifact", evidence_source_arguments
            )
            await session.validate_tool_result("preview_artifact", missing_evidence)
            assert missing_evidence.is_error is True
            assert missing_evidence.structured_content["code"] == "MISSING_EVIDENCE"

            capture_arguments = {
                "target": {
                    "argv": [sys.executable, "-c", command, str(marker), "0"],
                    "cwd": str(tmp_path),
                },
                "provider": {"kind": "direct"},
                "page_size": 1,
            }
            assert Draft202012Validator(by_name["capture_artifact_preview"].input_schema).is_valid(
                capture_arguments
            )
            captured = await session.call_tool("capture_artifact_preview", capture_arguments)
            await session.validate_tool_result("capture_artifact_preview", captured)
            inline = captured.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == captured.structured_content
            assert captured.is_error is False
            assert marker.read_text().splitlines() == ["started"]
            assert captured.structured_content["blocks"][1]["rows"][0]["text"] == "stdio capture"

            capture_page = captured.structured_content.get("next_page")
            assert capture_page is not None
            assert capture_page["tool"] == "preview_artifact"
            assert capture_page["arguments"]["sources"][0]["kind"] == "path"
            page = await session.call_tool(capture_page["tool"], capture_page["arguments"])
            await session.validate_tool_result(capture_page["tool"], page)
            assert page.is_error is False
            assert page.structured_content["blocks"][1]["rows"][0]["text"] == "second row"
            assert marker.read_text().splitlines() == ["started"]
            preserved = await session.call_tool(
                "preserve_evidence",
                {"analysis_id": captured.structured_content["analysis_id"]},
            )
            await session.validate_tool_result("preserve_evidence", preserved)
            inline = preserved.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == preserved.structured_content
            assert preserved.is_error is False
            assert preserved.structured_content["next_page"] is not None
            assert (
                preserved.structured_content["next_page"]["arguments"]["sources"][0]["kind"]
                == "evidence"
            )
            analysis = await session.call_tool(
                preserved.structured_content["next_page"]["tool"],
                preserved.structured_content["next_page"]["arguments"],
            )
            await session.validate_tool_result("preview_artifact", analysis)
            assert analysis.is_error is False
            assert analysis.structured_content["blocks"][1]["rows"][0]["text"] == "second row"
            assert marker.read_text().splitlines() == ["started"]

            failed = await session.call_tool(
                "capture_artifact_preview",
                {
                    "target": {
                        "argv": [sys.executable, "-c", command, str(marker), "7"],
                        "cwd": str(tmp_path),
                    },
                    "provider": {"kind": "direct"},
                    "preserve": True,
                    "page_size": 1,
                },
            )
            await session.validate_tool_result("capture_artifact_preview", failed)
            inline = failed.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == failed.structured_content
            assert failed.structured_content["status"] == "partial"
            assert failed.structured_content["capture"]["workload_status"] == "failed"
            failed_page = failed.structured_content["next_page"]
            assert failed_page["tool"] == "preview_artifact"
            assert failed_page["arguments"]["sources"][0]["kind"] == "evidence"
            second = await session.call_tool(failed_page["tool"], failed_page["arguments"])
            await session.validate_tool_result("preview_artifact", second)
            assert second.is_error is False
            assert second.structured_content["blocks"][1]["rows"][0]["text"] == "second row"
            assert marker.read_text().splitlines() == ["started", "started"]
            inspected = await session.call_tool(
                "inspect_evidence", {"evidence_id": preserved.structured_content["evidence_id"]}
            )
            await session.validate_tool_result("inspect_evidence", inspected)
            assert inspected.is_error is False
            inline = inspected.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == inspected.structured_content
            assert "uri" not in preserved.structured_content
            replay = await session.call_tool(
                "preview_artifact", {"sources": inspected.structured_content["analysis_sources"]}
            )
            assert replay.is_error is False
            assert replay.structured_content["blocks"][1]["rows"][0]["text"] == "stdio capture"
            assert marker.read_text().splitlines() == ["started", "started"]

    anyio.run(exercise)


@pytest.mark.e2e
@pytest.mark.process
@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixtures")
def test_stdio_perf_failures_redact_decoder_console_for_analysis_and_capture(
    tmp_path: Path,
) -> None:
    private = "private-customer-decoder-token"
    collector = tmp_path / "perf"
    collector.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "if 'record' in sys.argv:\n"
        "    Path(sys.argv[sys.argv.index('--output') + 1]).write_bytes(b'native profile')\n"
        "elif '--version' in sys.argv:\n"
        "    print('perf version 6.0')\n"
        "else:\n"
        f"    print({private!r}, file=sys.stderr)\n"
        "    raise SystemExit(7)\n"
    )
    collector.chmod(0o755)
    native = tmp_path / "perf.data"
    native.write_bytes(b"native profile")

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=str(Path(sys.executable).with_name("flameox")),
            args=["mcp", "serve"],
            cwd=tmp_path,
            env={
                "FLAMEOX_DATA_DIR": str(tmp_path / "store"),
                "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
            },
        )
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            await session.initialize()
            analyzed = await session.call_tool(
                "rank_cpu_hotspots", {"sources": [{"path": str(native), "format": "perf-data"}]}
            )
            assert analyzed.is_error
            assert analyzed.structured_content is not None
            assert analyzed.structured_content["details"]["decoder_exit_code"] == 7
            captured = await session.call_tool(
                "capture_cpu_hotspots",
                {
                    "target": {"argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path)},
                    "provider": {"kind": "perf", "call_graph": "fp"},
                },
            )
            assert captured.structured_content is not None
            assert (
                captured.structured_content["analysis_failure"]["details"]["decoder_exit_code"] == 7
            )
            for result in (analyzed, captured):
                assert private not in result.model_dump_json()
                inline = result.content[0]
                assert isinstance(inline, TextContent)
                assert json.loads(inline.text) == result.structured_content

    anyio.run(exercise)
