from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource

pytestmark = [pytest.mark.performance, pytest.mark.integration]


@pytest.mark.process
def test_v8_profile_aggregates_twenty_thousand_nodes_into_shared_frames(tmp_path: Path) -> None:
    profile = tmp_path / "repeated.cpuprofile"
    nodes = [{"id": 1, "callFrame": {"functionName": "root"}, "children": list(range(2, 20_002))}]
    nodes.extend(
        {
            "id": index,
            "callFrame": {
                "functionName": f"work{index % 10}",
                "url": "file:///app/work.js",
                "lineNumber": index % 10,
                "columnNumber": 0,
            },
            "children": [],
        }
        for index in range(2, 20_002)
    )
    profile.write_text(json.dumps({"nodes": nodes, "samples": list(range(2, 20_002))}))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        sources = [PathSource(path=str(profile), format="cpuprofile")]
        result = runtime.analyze("cpu.hotspots", sources, {})
        assert result["blocks"][0]["values"]["sample_count"] == 20_000
        rows = result["blocks"][1]["rows"]
        assert len(rows) == 11
        assert sum(row["self_value"] for row in rows) == 20_000
        for row in rows:
            if row["function"] == "root":
                assert row["inclusive_value"] == 20_000
            else:
                assert row["self_value"] == row["inclusive_value"] == 2_000
        rows.clear()
        assert len(runtime.analyze("cpu.hotspots", sources, {})["blocks"][1]["rows"]) == 11
    finally:
        runtime.close()


def test_fixture_analysis_keeps_other_workers_teardown_failures_separate(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    with events.open("w") as stream:
        for index in range(4_000):
            for phase, duration in (("setup", 10), ("teardown", 5)):
                stream.write(
                    json.dumps(
                        {
                            "event": "fixture_phase",
                            "fixture": f"fixture{index}",
                            "scope": "session",
                            "phase": phase,
                            "outcome": "passed" if phase == "setup" else "observed",
                            "duration_ns": duration,
                            "worker_id": "gw0",
                            "invocation_id": str(index),
                            "nodeid": "",
                        }
                    )
                    + "\n"
                )
        for index in range(4_000):
            stream.write(
                json.dumps(
                    {
                        "event": "test_phase",
                        "phase": "teardown",
                        "outcome": "failed",
                        "worker_id": "gw1",
                        "nodeid": f"test{index}",
                    }
                )
                + "\n"
            )
        stream.write('{"event":"run_finished","exitstatus":1}\n')
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "pytest.fixtures", [PathSource(path=str(events), format="pytest")], {}
        )
        metrics = result["blocks"][0]["values"]
        assert metrics["fixture_count"] == metrics["invocation_count"] == 4_000
        assert metrics["teardown_failure_count"] == 4_000
        assert metrics["incomplete_invocation_count"] == 0
        assert metrics["summed_fixture_work_ns"] == 60_000
        assert result["coverage"]["rows_observed"] == 8_000
    finally:
        runtime.close()
