from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp_types import TextContent

from flameox.mcp.server import FlameoxServer
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import CaptureTarget, EvidenceSource, RequestLimits


def _coverage_target(tmp_path: Path) -> CaptureTarget:
    marker = tmp_path / "workload-runs.txt"
    script = tmp_path / "covered.py"
    script.write_text(
        "from pathlib import Path\n"
        f"with Path({str(marker)!r}).open('a') as stream:\n"
        "    stream.write('run\\n')\n"
        "value = sum(range(100))\n"
        "print(value)\n"
    )
    return CaptureTarget(
        argv=[sys.executable, str(script)],
        cwd=str(tmp_path),
        provider_id="coverage",
        capture_arguments={"branch": True, "source": [str(tmp_path)]},
    )


@pytest.mark.process
@pytest.mark.parametrize("preserve", [False, True])
def test_native_capture_survives_tight_analysis_limit_and_can_be_reanalyzed(
    tmp_path: Path, preserve: bool
) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        limits = RequestLimits(max_input_bytes=1024, max_input_files=1, timeout_seconds=20)
        try:
            result = await runtime.capture_and_analyze(
                _coverage_target(tmp_path), "coverage.summary", limits=limits, preserve=preserve
            )

            assert result["analysis_failure"] is not None
            assert result["analysis_failure"]["code"] == "LIMIT_EXCEEDED"
            assert result["analysis_failure"]["details"]["analysis_source_count"] == 1
            assert result["capture"]["outcome"]["status"] == "succeeded"
            assert (tmp_path / "workload-runs.txt").read_text().splitlines() == ["run"]

            preserved = result.get("preserved") or runtime.preserve_evidence(result["analysis_id"])
            resource = runtime.read_evidence_agent_projection(preserved["evidence_id"])
            sources = resource["analysis_sources"]
            assert len(sources) == 1
            assert sources[0]["artifact_selector"]
            native = resource["body"]["artifacts"]
            assert len(native) == 1
            assert native[0]["format"] == "coverage"
            selection = runtime.repository.select_source(
                preserved["evidence_id"], selector=None, role=None
            )
            payload = selection.members[0][1].path.read_bytes()
            assert len(payload) == native[0]["size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == native[0]["sha256"]

            retried = runtime.analyze(
                "coverage.summary",
                [EvidenceSource.model_validate(sources[0])],
                {},
            )
            assert retried["analysis_failure"] is None
            assert retried["provider"]["id"] == "coverage.py"
            assert retried["blocks"][0]["values"]["line_count"] >= 1
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
@pytest.mark.parametrize("preserve", [False, True])
def test_mcp_native_analysis_failure_is_a_preservable_partial_result(
    tmp_path: Path,
    preserve: bool,
) -> None:
    async def exercise() -> None:
        target = _coverage_target(tmp_path)
        async with Client(
            FlameoxServer(
                evidence_directory=tmp_path / "store",
                limits=RequestLimits(max_input_bytes=1024, max_input_files=1),
            )
        ) as client:
            response = await client.call_tool(
                "capture_coverage_summary",
                {
                    "target": {"argv": target.argv, "cwd": target.cwd},
                    "provider": {"kind": "coverage", **target.capture_arguments},
                    "preserve": preserve,
                },
            )

            assert response.is_error is False
            partial = response.structured_content
            assert partial["status"] == "partial"
            assert partial["analysis_failure"]["code"] == "LIMIT_EXCEEDED"
            assert partial["analysis_id"]
            action = partial["next_action"]
            if preserve:
                preserved = partial["preserved"]
                assert action["kind"] == "call_tool"
                assert action["tool"] == "inspect_evidence"
                assert action["arguments"] == {"evidence_id": preserved["evidence_id"]}
                assert action["then_retry"] == "summarize_coverage"
            else:
                assert action["kind"] == "preserve_then_analyze"
                saved = await client.call_tool("preserve_evidence", action["preserve_arguments"])
                assert saved.is_error is False
                preserved = saved.structured_content
            assert preserved["evidence_id"]
            assert "uri" not in preserved
            inline = response.content[0]
            assert isinstance(inline, TextContent)
            assert json.loads(inline.text) == partial
            assert (tmp_path / "workload-runs.txt").read_text().splitlines() == ["run"]

            inspected = await client.call_tool(
                "inspect_evidence", {"evidence_id": preserved["evidence_id"]}
            )
            assert inspected.is_error is False
            projection = inspected.structured_content
            assert len(projection["analysis_sources"]) == 1
            assert projection["body"]["analysis_request"]["failure"]["code"] == "LIMIT_EXCEEDED"

        async with Client(FlameoxServer(evidence_directory=tmp_path / "store")) as restarted:
            retried = await restarted.call_tool(
                "summarize_coverage", {"sources": projection["analysis_sources"]}
            )
            assert retried.is_error is False
            assert retried.structured_content["analysis_failure"] is None
            assert retried.structured_content["blocks"][0]["values"]["line_count"] >= 1
            assert (tmp_path / "workload-runs.txt").read_text().splitlines() == ["run"]

    anyio.run(exercise)


@pytest.mark.process
@pytest.mark.parametrize("preserve", [False, True])
def test_rejected_sparse_native_artifact_keeps_capture_failure_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool
) -> None:
    monkeypatch.setattr("flameox.runtime.MAX_SESSION_SCRATCH_BYTES", 1024)
    workload = tmp_path / "sparse.py"
    workload.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "with Path(os.environ['FLAMEOX_OBSERVATIONS_PATH']).open('wb') as stream:\n"
        "    stream.seek(2048)\n"
        "    stream.write(b'x')\n"
        "print('rejected capture diagnostics')\n"
    )

    async def exercise() -> dict[str, object]:
        async with Client(FlameoxServer(evidence_directory=tmp_path / "store")) as client:
            response = await client.call_tool(
                "capture_failures_summary",
                {
                    "target": {"argv": [sys.executable, str(workload)], "cwd": str(tmp_path)},
                    "provider": {"kind": "observations"},
                    "limits": {"max_output_bytes": 1024},
                    "preserve": preserve,
                },
            )
            assert not response.is_error
            result = response.structured_content
            assert result["analysis_failure"]["details"]["analysis_source_count"] == 0
            action = result["next_action"]
            assert action["kind"] == "call_tool"
            assert action["then_retry"] is None
            assert "before a new capture" in action["message"]
            preserved = result.get("preserved")
            if preserve:
                assert action["tool"] == "inspect_evidence"
            else:
                assert action["tool"] == "preserve_evidence"
                saved = await client.call_tool(action["tool"], action["arguments"])
                assert not saved.is_error
                preserved = saved.structured_content
            assert preserved is not None
            inspected = await client.call_tool(
                "inspect_evidence", {"evidence_id": preserved["evidence_id"]}
            )
            assert not inspected.is_error
            resource = inspected.structured_content
            assert resource["analysis_sources"] == []
            assert resource["body"]["artifacts"] == []
            return dict(result)

    result = anyio.run(exercise)
    execution = result["capture"]["executions"][0]  # type: ignore[index]
    assert execution["status"] == "failed"
    assert execution["failure_code"] == "LIMIT_EXCEEDED"
    assert execution["limit"]["kind"] == "writable_limit_exceeded"
    assert execution["artifact_rejections"][0]["code"] == "LIMIT_EXCEEDED"
    assert "bundle was discarded" in execution["artifact_rejections"][0]["message"]
    assert execution["console_diagnostics"]["stdout"] == "rejected capture diagnostics\n"
    assert result["analysis_failure"]["code"] == "EXECUTION_FAILURE"  # type: ignore[index]
    assert result["inputs"] == []
    if preserve:
        assert result["preserved"]["evidence_id"]  # type: ignore[index]


@pytest.mark.process
def test_rejected_native_symlink_does_not_delete_external_target(tmp_path: Path) -> None:
    sentinel = tmp_path / "external-sentinel"
    sentinel.mkdir()
    marker = sentinel / "keep.txt"
    marker.write_text("keep me")
    workload = tmp_path / "symlink.py"
    workload.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "output = Path(os.environ['FLAMEOX_OBSERVATIONS_PATH'])\n"
        f"Path({str(marker)!r}).write_text('keep me')\n"
        "output.symlink_to(Path(" + repr(str(sentinel)) + "), target_is_directory=True)\n"
    )

    async def exercise() -> dict[str, object]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, str(workload)],
                    cwd=str(tmp_path),
                    provider_id="observations",
                ),
                "failures.summary",
                limits=RequestLimits(max_output_bytes=1024),
                preserve=True,
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    execution = result["capture"]["executions"][0]  # type: ignore[index]
    assert execution["failure_code"] == "CAPTURE_ARTIFACT_REJECTED"
    assert execution["artifact_rejections"][0]["code"] == "INVALID_INPUT"
    assert marker.read_text() == "keep me"
    assert result["preserved"]["evidence_id"]  # type: ignore[index]
