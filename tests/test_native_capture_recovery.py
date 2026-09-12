from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp_types import ResourceLink, TextResourceContents

from flameox.mcp import create_server
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
            assert result["capture"]["outcome"]["status"] == "succeeded"
            assert (tmp_path / "workload-runs.txt").read_text().splitlines() == ["run"]
            assert result["analysis_id"] in runtime.analyses

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
def test_mcp_native_analysis_failure_exposes_preservable_partial_evidence(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        target = _coverage_target(tmp_path)
        async with Client(
            create_server(
                evidence_directory=tmp_path / "store",
                limits=RequestLimits(max_input_bytes=1024, max_input_files=1),
            )
        ) as client:
            response = await client.call_tool(
                "capture_and_analyze",
                {
                    "request": {
                        "capability_id": "coverage.summary",
                        "target": {
                            "argv": target.argv,
                            "cwd": target.cwd,
                        },
                        "provider": {
                            "kind": "coverage",
                            "options": target.capture_arguments,
                        },
                        "preserve": True,
                    }
                },
            )

            assert response.is_error
            partial = response.structured_content["details"]["partial_evidence"]
            assert partial["analysis_failure"]["code"] == "LIMIT_EXCEEDED"
            assert partial["analysis_id"]
            preserved = partial["preserved"]
            assert preserved["evidence_id"]
            assert preserved["uri"]
            links = [item for item in response.content if isinstance(item, ResourceLink)]
            assert [item.uri for item in links] == [preserved["uri"]]
            assert (tmp_path / "workload-runs.txt").read_text().splitlines() == ["run"]

            resource = await client.read_resource(preserved["uri"])
            assert isinstance(resource.contents[0], TextResourceContents)
            projection = json.loads(resource.contents[0].text)
            assert len(projection["analysis_sources"]) == 1
            assert projection["body"]["analysis_request"]["failure"]["code"] == "LIMIT_EXCEEDED"

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
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, str(workload)],
                    cwd=str(tmp_path),
                    provider_id="observations",
                ),
                "failures.summary",
                limits=RequestLimits(max_output_bytes=1024),
                preserve=preserve,
            )
            assert result["analysis_id"] in runtime.analyses
            preserved = result.get("preserved") or runtime.preserve_evidence(result["analysis_id"])
            resource = runtime.read_evidence_agent_projection(preserved["evidence_id"])
            assert resource["body"]["artifacts"] == []
            return result
        finally:
            runtime.close()

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
