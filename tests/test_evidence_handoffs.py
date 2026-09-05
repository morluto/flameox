from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp_types import TextResourceContents

from flameox.canonical import canonical_bytes
from flameox.mcp import create_server
from flameox.runtime_contracts import (
    CaptureTarget,
    EvidenceSource,
    ExperimentCase,
    ExperimentDesign,
    PathSource,
    PreviewArguments,
    RequestLimits,
    RuntimeFailure,
)
from flameox.stateless import AnalysisRuntime


@pytest.mark.integration
@pytest.mark.parametrize("whitespace", ["", "\n", " \t\r\n" * 2000], ids=["compact", "lf", "long"])
def test_preview_json_whitespace_preserves_rows(tmp_path: Path, whitespace: str) -> None:
    artifact = tmp_path / "array.json"
    artifact.write_text(whitespace + '[{"value":1},{"value":2}]')
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        assert [row["value"] for row in result["blocks"][1]["rows"]] == [1, 2]
        assert result["coverage"]["complete"] is True
    finally:
        runtime.close()


@pytest.mark.integration
def test_preview_offset_is_a_logical_row(tmp_path: Path) -> None:
    assert "row" in PreviewArguments.model_json_schema()["properties"]["offset"]["description"]
    artifact = tmp_path / "lines.txt"
    artifact.write_text("a\nlong second line\nthird\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "artifact.preview", [PathSource(path=str(artifact))], {"offset": 1}
        )
        assert [row["text"] for row in result["blocks"][1]["rows"]] == ["long second line", "third"]
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.parametrize("inline", [False, True])
def test_preserved_capture_continuation_uses_discovered_sources(
    tmp_path: Path, inline: bool
) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            limits = RequestLimits(max_rows=2)
            first = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "print('one\\ntwo\\nthree\\nfour\\nfive')"],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
                limits=limits,
                preserve=inline,
            )
            preserved = first.get("preserved") or runtime.preserve_evidence(first["analysis_id"])
            execution = first["capture"]["executions"][0]
            assert execution["returncode_scope"] == "workload"
            assert execution["workload_returncode"] == 0
            projection = runtime.read_evidence_agent_projection(preserved["evidence_id"])
            sources = [
                EvidenceSource.model_validate(item) for item in projection["analysis_sources"]
            ]
            rows = list(first["blocks"][1]["rows"])
            token = first["continuation"]
            while token:
                page = runtime.analyze(
                    "artifact.preview", sources, {}, limits=limits, continuation=token
                )
                rows.extend(page["blocks"][1]["rows"])
                token = page["continuation"]
            assert [row["text"] for row in rows] == ["one", "two", "three", "four", "five"]
            assert not list(runtime.scratch.glob("capture-*"))
            artifacts = projection["body"]["artifacts"]
            for artifact in artifacts:
                selected = runtime.analyze(
                    "artifact.preview", [EvidenceSource.model_validate(artifact["source"])], {}
                )
                assert selected["inputs"][0]["sha256"] == artifact["sha256"]
            with pytest.raises(RuntimeFailure, match="Continuation"):
                runtime.analyze(
                    "artifact.preview",
                    sources,
                    {"offset": 1},
                    limits=limits,
                    continuation=first["continuation"],
                )
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.unit
@pytest.mark.parametrize("failed_index", [0, 15])
@pytest.mark.parametrize("text_width", [140, 170])
def test_capture_summary_survives_diagnostic_truncation(failed_index: int, text_width: int) -> None:
    executions = [
        dict(
            case=f"case-{i}",
            block=1,
            returncode=int(i == failed_index),
            status="failed" if i == failed_index else "succeeded",
            failure_code=None,
            wall_time_ns=100,
            containment="process_group",
            limit=None,
            semantic_oracle=None,
        )
        for i in range(16)
    ]
    result: dict[str, Any] = dict(
        analysis_id="a" * 64,
        capability_id="artifact.preview",
        coverage=dict(complete=True, rows_returned=100, rows_observed=100),
        blocks=[
            dict(type="metrics", values={}),
            dict(type="table", rows=[dict(text="x" * text_width) for _ in range(100)]),
        ],
        limitations=[],
        continuation=None,
        capture=dict(executions=executions, outcome=AnalysisRuntime._capture_outcome(executions)),
    )
    runtime = object.__new__(AnalysisRuntime)
    runtime._bound_capture_result(result, 16384, {}, 0)
    assert result["capture"]["outcome"] == {
        "status": "failed",
        "execution_count": 16,
        "succeeded_count": 15,
        "failed_count": 1,
    }
    assert len(canonical_bytes(result)) <= 16384
    if text_width == 170:
        assert result["capture"]["executions"] == []


@pytest.mark.integration
@pytest.mark.parametrize("failure_code", [None, "SEMANTIC_ORACLE_FAILED"])
def test_mcp_uses_capture_outcome_when_execution_diagnostics_are_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str | None,
) -> None:
    executions = [{"status": "failed", "failure_code": failure_code} for _ in range(16)]

    async def capture(self: AnalysisRuntime, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "capture": {
                "outcome": self._capture_outcome(executions),
                "executions": [],
                "executions_truncated": 16,
            }
        }

    monkeypatch.setattr(AnalysisRuntime, "capture_and_analyze", capture)

    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "capture_process_output",
                {
                    "target": {"argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path)},
                    "provider": {"kind": "direct"},
                    "execution": {"kind": "single"},
                },
            )
            assert result.is_error
            assert result.structured_content["code"] == "EXECUTION_FAILURE"
            outcome = result.structured_content["details"]["partial_evidence"]["capture"]["outcome"]
            assert outcome["failed_count"] == 16

    anyio.run(exercise)


@pytest.mark.unit
def test_experiment_classification_explicitly_describes_point_estimate() -> None:
    design = ExperimentDesign(
        cases=[ExperimentCase(name="base"), ExperimentCase(name="candidate")],
        blocks=3,
        seed=7,
        metric="wall_time_ns",
        estimand="mean_difference",
        practical_threshold=10,
    )
    executions = [
        dict(case=case, block=block, status="succeeded", wall_time_ns=value)
        for block, candidate in enumerate([1000, 2000, 3000], 1)
        for case, value in [("base", 2000), ("candidate", candidate)]
    ]
    blocks, _ = AnalysisRuntime._experiment_blocks(design, executions)
    row = blocks[-1]["rows"][0]
    assert row["estimate"] == 0
    assert row["confidence_low"] < -10 < 10 < row["confidence_high"]
    assert row["point_estimate_classification"] == "within_threshold"
    assert blocks[0]["values"]["decision_basis"] == "descriptive_point_estimate"


@pytest.mark.integration
def test_repository_corruption_has_path_free_recovery(tmp_path: Path) -> None:
    store = tmp_path / "private-store"
    store.mkdir()
    (store / "existing-evidence").write_text("retain")
    runtime = AnalysisRuntime(evidence_directory=store)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.query_evidence()
        assert failure.value.code == "REPOSITORY_CORRUPTION"
        details = failure.value.details
        assert details["configuration_variable"] == "FLAMEOX_DATA_DIR"
        assert "restore" in str(details).lower()
        assert str(store) not in str(details)
        assert (store / "existing-evidence").read_text() == "retain"
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.skipif(os.name == "nt", reason="POSIX collector fixture")
def test_mcp_collector_failure_retains_profile_and_unknown_workload_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = tmp_path / "collector"
    collector.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\n"
        "from pathlib import Path\n"
        "document = {'shared': {'frames': [{'name': 'work'}]}, "
        "'profiles': [{'type': 'sampled', 'samples': [[0]], 'weights': [1.0]}]}\n"
        "Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(document))\n"
        "print('collector could not reap child', file=sys.stderr)\n"
        "sys.exit(1)\n"
    )
    collector.chmod(0o755)
    monkeypatch.setattr(
        AnalysisRuntime, "_managed_executable", staticmethod(lambda _: str(collector))
    )

    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "capture_cpu_hotspots",
                {
                    "target": {"argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path)},
                    "provider": {"kind": "py-spy"},
                    "execution": {"kind": "single"},
                    "preserve": True,
                },
            )
            assert result.is_error
            partial = result.structured_content["details"]["partial_evidence"]
            assert partial["capture"]["outcome"]["failed_count"] == 1
            execution = partial["capture"]["executions"][0]
            assert execution["returncode"] == 1
            assert execution["returncode_scope"] == "collector"
            assert execution["workload_returncode"] is None
            resource = await client.read_resource(partial["preserved"]["uri"])
            assert isinstance(resource.contents[0], TextResourceContents)
            projection = json.loads(resource.contents[0].text)
            texts: list[str] = []
            for artifact in projection["body"]["artifacts"]:
                if artifact["format"] == "text":
                    preview = await client.call_tool(
                        "preview_artifact", {"sources": [artifact["source"]]}
                    )
                    texts.extend(
                        row["text"] for row in preview.structured_content["blocks"][1]["rows"]
                    )
            assert "collector could not reap child" in texts
            assert partial["blocks"][1]["rows"]

    anyio.run(exercise)


@pytest.mark.integration
def test_directory_sources_round_trip_without_revealing_member_names(tmp_path: Path) -> None:
    bundle = tmp_path / "private-bundle"
    bundle.mkdir()
    (bundle / "private-member.txt").write_text("one")
    (bundle / "other.txt").write_text("two")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        first = runtime.analyze(
            "artifact.preview", [PathSource(path=str(bundle), format="text")], {}
        )
        preserved = runtime.preserve_evidence(first["analysis_id"])
        resource = runtime.read_evidence_agent_projection(preserved["evidence_id"])
        assert "private-member" not in json.dumps(resource)
        assert "private-bundle" not in json.dumps(resource)
        assert len(resource["analysis_sources"]) == 1
        page = runtime.analyze(
            "artifact.preview",
            [EvidenceSource.model_validate(item) for item in resource["analysis_sources"]],
            {},
        )
        assert page["blocks"][1]["rows"] == first["blocks"][1]["rows"]
    finally:
        runtime.close()


@pytest.mark.integration
@pytest.mark.parametrize("content", ["[]", "\n[]", "{}", '\n{"key": 1}', "[", "\n[{]"])
def test_preview_json_empty_object_and_malformed_boundaries(tmp_path: Path, content: str) -> None:
    artifact = tmp_path / "input.json"
    artifact.write_text(content)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        if content in {"[", "\n[{]"}:
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            assert failure.value.code == "DECODE_FAILURE"
        else:
            result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            assert len(result["blocks"][1]["rows"]) == (1 if "key" in content else 0)
            assert result["coverage"]["complete"]
    finally:
        runtime.close()
