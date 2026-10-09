from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


@pytest.mark.process
def test_cpu_profile_uses_explicit_isolated_worker_without_repository(tmp_path: Path) -> None:
    profile = tmp_path / "cpu.cpuprofile"
    profile.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 1,
                        "callFrame": {
                            "functionName": "main",
                            "url": (tmp_path / "index.js").as_uri(),
                            "scriptId": "1",
                            "lineNumber": 1,
                            "columnNumber": 0,
                        },
                        "hitCount": 2,
                        "children": [],
                    }
                ],
                "samples": [1, 1],
                "startTime": 0,
                "endTime": 100,
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "cpu.hotspots",
            [PathSource(path=str(profile), format="cpuprofile")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "v8-cpu-profile"
    assert result["blocks"][0]["values"]["sample_count"] == 2
    row = result["blocks"][1]["rows"][0]
    assert row["function"] == "main"
    assert row["file"] == (tmp_path / "index.js").as_uri()
    assert (row["line"], row["column"]) == (1, 0)
    assert row["symbolization"] == "complete"
    assert row["self_value"] == 2
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
@pytest.mark.parametrize("profile_kind", ["cpu", "heap"])
@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [
        pytest.param({}, (-1, -1), id="both-omitted"),
        pytest.param({"lineNumber": 0}, (0, -1), id="column-omitted"),
        pytest.param({"columnNumber": 0}, (-1, 0), id="line-omitted"),
        pytest.param({"lineNumber": 0, "columnNumber": 0}, (0, 0), id="explicit-zero"),
        pytest.param({"lineNumber": -1, "columnNumber": -1}, (-1, -1), id="explicit-unknown"),
    ],
)
def test_v8_hotspots_distinguish_omitted_coordinates_from_explicit_zero(
    tmp_path: Path,
    profile_kind: str,
    coordinates: dict[str, int],
    expected: tuple[int, int],
) -> None:
    frame = {"functionName": "work", "url": "app.js", **coordinates}
    if profile_kind == "cpu":
        payload = {
            "nodes": [{"id": 1, "callFrame": frame, "children": []}],
            "samples": [1],
        }
        capability_id, format_name = "cpu.hotspots", "cpuprofile"
    else:
        payload = {
            "head": {"id": 1, "callFrame": frame, "selfSize": 64, "children": []},
            "samples": [{"size": 64, "nodeId": 1}],
        }
        capability_id, format_name = "memory.hotspots", "heapprofile"
    profile = tmp_path / f"profile.{format_name}"
    profile.write_text(json.dumps(payload))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            capability_id, [PathSource(path=str(profile), format=format_name)], {}
        )
    finally:
        runtime.close()
    row = result["blocks"][1]["rows"][0]
    assert (row["line"], row["column"]) == expected
    assert row["function"] == "work"
    assert row["file"] == "app.js"


@pytest.mark.process
@pytest.mark.parametrize("hit_count", [None, 99])
def test_cpu_hotspots_count_exported_samples_when_hit_metadata_differs(
    tmp_path: Path, hit_count: int | None
) -> None:
    root = {
        "id": 1,
        "callFrame": {"functionName": "root", "url": "app.js"},
        "children": [2],
    }
    leaf = {
        "id": 2,
        "callFrame": {"functionName": "leaf", "url": "app.js"},
        "children": [],
    }
    if hit_count is not None:
        root["hitCount"] = hit_count
        leaf["hitCount"] = hit_count
    profile = tmp_path / "cpu.cpuprofile"
    profile.write_text(json.dumps({"nodes": [root, leaf], "samples": [1, 2, 2]}))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "cpu.hotspots", [PathSource(path=str(profile), format="cpuprofile")], {}
        )
    finally:
        runtime.close()
    rows = {row["function"]: row for row in result["blocks"][1]["rows"]}
    assert result["blocks"][0]["values"]["sample_count"] == 3
    assert rows["root"]["self_value"] == rows["root"]["sample_count"] == 1
    assert rows["root"]["inclusive_value"] == 3
    assert rows["leaf"]["self_value"] == rows["leaf"]["sample_count"] == 2
    assert sum(row["self_value"] for row in rows.values()) == 3


@pytest.mark.process
def test_cpu_profile_rejects_samples_for_unknown_nodes(tmp_path: Path) -> None:
    profile = tmp_path / "cpu.cpuprofile"
    profile.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 1,
                        "callFrame": {
                            "functionName": "main",
                            "url": "app.js",
                            "scriptId": "1",
                            "lineNumber": 1,
                            "columnNumber": 0,
                        },
                        "hitCount": 1,
                        "children": [],
                    }
                ],
                "samples": [2],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "cpu.hotspots",
                [PathSource(path=str(profile), format="cpuprofile")],
                {},
            )
    finally:
        runtime.close()

    assert failure.value.code == "DECODE_FAILURE"


@pytest.mark.process
def test_heap_sample_count_aggregates_records_for_repeated_frame_identity(
    tmp_path: Path,
) -> None:
    frame = {"functionName": "allocate", "url": "app.js", "lineNumber": 3, "columnNumber": 2}
    profile = tmp_path / "memory.heapprofile"
    profile.write_text(
        json.dumps(
            {
                "head": {
                    "callFrame": {"functionName": "root", "url": "app.js"},
                    "selfSize": 0,
                    "id": 1,
                    "children": [
                        {"callFrame": frame, "selfSize": 64, "id": 2, "children": []},
                        {"callFrame": frame, "selfSize": 128, "id": 3, "children": []},
                    ],
                },
                "samples": [{"size": 64, "nodeId": node} for node in [2, 2, 3, 3, 3]],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze("memory.hotspots", [PathSource(path=str(profile))], {})
    finally:
        runtime.close()

    allocation = next(row for row in result["blocks"][1]["rows"] if row["function"] == "allocate")
    assert allocation["sample_count"] == 5
    assert allocation["self_value"] == 192
    assert (allocation["line"], allocation["column"]) == (3, 2)
    assert result["blocks"][0]["values"]["sample_count"] == 5


@pytest.mark.process
def test_v8_heap_profile_reports_unresolved_samples_without_guessing_frames(tmp_path: Path) -> None:
    profile = tmp_path / "memory.heapprofile"
    profile.write_text(
        json.dumps(
            {
                "head": {
                    "callFrame": {"functionName": "allocate", "url": "app.js"},
                    "selfSize": 64,
                    "id": 1,
                    "children": [],
                },
                "samples": [{"size": 64, "nodeId": 2}],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("memory.hotspots", [PathSource(path=str(profile))], {})
    finally:
        runtime.close()

    metrics = result["blocks"][0]["values"]
    assert metrics["unresolved_sample_count"] == 1
    assert metrics["unresolved_sampled_bytes"] == 64
    assert metrics["sample_count"] == 1
    assert result["blocks"][1]["rows"][0]["sample_count"] == 0
    assert result["coverage"]["complete"] is False
    assert result["continuation"] is None
    assert any("frame attribution is unavailable" in item for item in result["limitations"])


@pytest.mark.process
def test_v8_heap_profile_rejects_malformed_sample_size(tmp_path: Path) -> None:
    profile = tmp_path / "memory.heapprofile"
    profile.write_text(
        json.dumps(
            {
                "head": {
                    "callFrame": {"functionName": "root", "url": "app.js"},
                    "selfSize": 0,
                    "id": 1,
                    "children": [],
                },
                "samples": [{"size": -1, "nodeId": 2}],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("memory.hotspots", [PathSource(path=str(profile))], {})
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


@pytest.mark.process
def test_v8_raw_node_bound_is_independent_from_hotspot_row_limit(tmp_path: Path) -> None:
    profile = tmp_path / "large.cpuprofile"
    nodes = [
        {
            "id": index,
            "callFrame": {
                "functionName": "shared",
                "url": "app.js",
                "scriptId": "1",
                "lineNumber": 1,
                "columnNumber": 0,
            },
            "hitCount": 1 if index == 1 else 0,
            "children": [index + 1] if index < 1_002 else [],
        }
        for index in range(1, 1_003)
    ]
    profile.write_text(json.dumps({"nodes": nodes, "samples": [1]}))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "cpu.hotspots",
            [PathSource(path=str(profile), format="cpuprofile")],
            {},
            limits=RequestLimits(max_rows=10),
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"] == {"node_count": 1_002, "sample_count": 1}
    assert len(result["blocks"][1]["rows"]) == 1
    assert result["coverage"] == {"rows_returned": 1, "rows_observed": 1, "complete": True}


@pytest.mark.process
def test_v8_hotspot_projection_bounds_distinct_aggregated_frames(tmp_path: Path) -> None:
    profile = tmp_path / "many-frames.cpuprofile"
    nodes = [
        {
            "id": index,
            "callFrame": {
                "functionName": f"frame-{index}",
                "url": "app.js",
                "scriptId": "1",
                "lineNumber": index,
                "columnNumber": 0,
            },
            "hitCount": 1 if index == 1 else 0,
            "children": [index + 1] if index < 1_002 else [],
        }
        for index in range(1, 1_003)
    ]
    profile.write_text(json.dumps({"nodes": nodes, "samples": [1]}))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "cpu.hotspots",
            [PathSource(path=str(profile), format="cpuprofile")],
            {},
            limits=RequestLimits(max_rows=10),
        )
    finally:
        runtime.close()

    assert result["coverage"] == {
        "rows_returned": 10,
        "rows_observed": 1_002,
        "complete": False,
    }
    assert result["blocks"][1]["rows"][0]["self_value"] == 1
