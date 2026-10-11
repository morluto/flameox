from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    PathSource,
    RequestLimits,
    RuntimeFailure,
)
from flameox.source_files import NativeSource


@pytest.mark.process
@pytest.mark.golden
@pytest.mark.parametrize("profile_kind", ["cpu", "heap"])
def test_v8_profiles_rank_globally_without_merging_artifact_weights(
    tmp_path: Path, profile_kind: str
) -> None:
    format_name = "cpuprofile" if profile_kind == "cpu" else "heapprofile"
    sources = []
    for index, weight in enumerate([1, 3]):
        work = {
            "id": 2,
            "callFrame": {"functionName": "work", "url": "work.js"},
            "children": [],
            "selfSize": weight * 10,
        }
        root = {
            "id": 1,
            "callFrame": {"functionName": "root"},
            "children": [2] if profile_kind == "cpu" else [work],
            "selfSize": 0,
        }
        payload = (
            {"nodes": [root, work], "samples": [2] * weight}
            if profile_kind == "cpu"
            else {"head": root, "samples": [{"nodeId": 2, "size": weight * 10}]}
        )
        path = tmp_path / f"profile-{index}.{format_name}"
        data = json.dumps(payload).encode()
        path.write_bytes(data)
        sources.append(
            NativeSource(
                path, hashlib.sha256(data).hexdigest(), len(data), format_name, None, str(index)
            )
        )
    operation = "rank_cpu_hotspots" if profile_kind == "cpu" else "rank_allocation_hotspots"
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.structured_workers.analyze(
            operation,
            sources,
            {},
            max_rows=2,
            timeout_seconds=10,
            maximum_rss_bytes=512 * 1024**2,
            maximum_output_bytes=1024**2,
        )
        assert result is not None
        assert result.rows_observed == 4
        assert result.complete is False
        metrics = result.blocks[0]["values"]
        assert metrics == (
            {"profile_count": 2, "node_count": 4, "sample_count": 4}
            if profile_kind == "cpu"
            else {"profile_count": 2, "node_count": 4, "sample_count": 2, "total_sampled_bytes": 40}
        )
        rows = result.blocks[1]["rows"]
        assert [row["self_value"] for row in rows] == (
            [3, 1] if profile_kind == "cpu" else [30, 10]
        )
        assert rows[0]["frame_id"] == rows[1]["frame_id"]
        assert [row["artifact_id"] for row in rows] == [
            source.sha256 for source in reversed(sources)
        ]

        request_sources = [
            PathSource(path=str(source.path), format=format_name) for source in sources
        ]
        with pytest.raises(RuntimeFailure) as too_many:
            runtime.analyze(operation, request_sources * 17, {})
        assert too_many.value.code == "INVALID_INPUT"
        mixed_sources = [
            request_sources[0],
            request_sources[1].model_copy(
                update={"format": "pstats" if profile_kind == "cpu" else "memray"}
            ),
        ]
        with pytest.raises(RuntimeFailure) as mixed:
            runtime.analyze(operation, mixed_sources, {})
        assert mixed.value.code == "INVALID_INPUT"
        first = runtime.analyze(operation, request_sources, {}, limits=RequestLimits(max_rows=1))
        second = runtime.analyze(
            operation,
            request_sources,
            {},
            limits=RequestLimits(max_rows=1),
            continuation=first["continuation"],
        )
        assert first["blocks"][1]["rows"] + second["blocks"][1]["rows"] == rows
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.parametrize("profile_kind", ["cpu", "heap"])
@pytest.mark.parametrize("field", ["tree", "samples"])
@pytest.mark.parametrize("invalid", [None, {}, 1, "invalid"])
def test_v8_profiles_reject_malformed_required_fields(
    tmp_path: Path, profile_kind: str, field: str, invalid: object
) -> None:
    node = {"id": 1, "callFrame": {"functionName": "work"}, "children": [], "selfSize": 0}
    tree_field = "nodes" if profile_kind == "cpu" else "head"
    payload: dict[str, object] = {
        tree_field: [node] if profile_kind == "cpu" else node,
        "samples": [],
    }
    payload[tree_field if field == "tree" else field] = invalid
    format_name = "cpuprofile" if profile_kind == "cpu" else "heapprofile"
    profile = tmp_path / f"profile.{format_name}"
    profile.write_text(json.dumps(payload))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "rank_cpu_hotspots" if profile_kind == "cpu" else "rank_allocation_hotspots",
                [PathSource(path=str(profile), format=format_name)],
                {},
            )
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.parametrize("profile_kind", ["cpu", "heap"])
def test_v8_profiles_accept_empty_sample_arrays(tmp_path: Path, profile_kind: str) -> None:
    node = {"id": 1, "callFrame": {"functionName": "work"}, "children": [], "selfSize": 0}
    payload = (
        {"nodes": [node], "samples": []} if profile_kind == "cpu" else {"head": node, "samples": []}
    )
    format_name = "cpuprofile" if profile_kind == "cpu" else "heapprofile"
    profile = tmp_path / f"profile.{format_name}"
    profile.write_text(json.dumps(payload))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "rank_cpu_hotspots" if profile_kind == "cpu" else "rank_allocation_hotspots",
            [PathSource(path=str(profile), format=format_name)],
            {},
        )
        assert result["coverage"]["complete"] is True
        assert result["blocks"][0]["values"]["sample_count"] == 0
    finally:
        runtime.close()


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
            "rank_cpu_hotspots",
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
        operation, format_name = "rank_cpu_hotspots", "cpuprofile"
    else:
        payload = {
            "head": {"id": 1, "callFrame": frame, "selfSize": 64, "children": []},
            "samples": [{"size": 64, "nodeId": 1}],
        }
        operation, format_name = "rank_allocation_hotspots", "heapprofile"
    profile = tmp_path / f"profile.{format_name}"
    profile.write_text(json.dumps(payload))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(operation, [PathSource(path=str(profile), format=format_name)], {})
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
            "rank_cpu_hotspots", [PathSource(path=str(profile), format="cpuprofile")], {}
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
                "rank_cpu_hotspots",
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
        result = runtime.analyze("rank_allocation_hotspots", [PathSource(path=str(profile))], {})
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
        result = runtime.analyze("rank_allocation_hotspots", [PathSource(path=str(profile))], {})
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
            runtime.analyze("rank_allocation_hotspots", [PathSource(path=str(profile))], {})
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
            "rank_cpu_hotspots",
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
            "rank_cpu_hotspots",
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


@pytest.mark.process
@pytest.mark.parametrize("children", [[42], [[]], [None], {}, False])
def test_v8_heap_rejects_malformed_child_nodes(tmp_path: Path, children: object) -> None:
    profile = tmp_path / "memory.heapprofile"
    profile.write_text(
        json.dumps(
            {
                "head": {
                    "id": 1,
                    "selfSize": 12,
                    "callFrame": {"functionName": "work", "url": "work.js"},
                    "children": children,
                },
                "samples": [],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("rank_allocation_hotspots", [PathSource(path=str(profile))], {})
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


@pytest.mark.process
def test_v8_heap_ignores_unrelated_nested_measurement_fields(tmp_path: Path) -> None:
    profile = tmp_path / "memory.heapprofile"
    profile.write_text(
        json.dumps(
            {
                "head": {
                    "id": 1,
                    "selfSize": 12,
                    "callFrame": {"functionName": "work", "url": "work.js"},
                    "children": [],
                    "extra": {"id": 99, "selfSize": 999, "children": []},
                },
                "samples": [{"size": 12, "nodeId": 1, "extra": {"size": 999, "nodeId": 2}}],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze("rank_allocation_hotspots", [PathSource(path=str(profile))], {})
        assert result["blocks"][0]["values"]["total_sampled_bytes"] == 12
        row = result["blocks"][1]["rows"][0]
        assert row["sample_count"] == 1
        assert row["self_value"] == 12
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.parametrize("profile_kind", ["cpu", "heap"])
@pytest.mark.parametrize("extra", ['"samples.item": 1', '"extra": {"id": 1, "id": 2}'])
def test_v8_validates_literal_native_object_fields(
    tmp_path: Path, profile_kind: str, extra: str
) -> None:
    if profile_kind == "cpu":
        payload: dict[str, object] = {
            "nodes": [{"id": 1, "callFrame": {"functionName": "work"}}],
            "samples": [],
        }
        operation, format_name = "rank_cpu_hotspots", "cpuprofile"
    else:
        payload = {
            "head": {
                "id": 1,
                "selfSize": 12,
                "callFrame": {"functionName": "work"},
                "children": [],
            },
            "samples": [],
        }
        operation, format_name = "rank_allocation_hotspots", "heapprofile"
    artifact = tmp_path / f"native.{format_name}"
    artifact.write_text(json.dumps(payload)[:-1] + "," + extra + "}")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        if "samples.item" in extra:
            result = runtime.analyze(operation, [PathSource(path=str(artifact))], {})
            assert result["blocks"][0]["values"]["sample_count"] == 0
        else:
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze(operation, [PathSource(path=str(artifact))], {})
            assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.parametrize("coordinate", [False, True, {}, [], None, "1", 1.5])
def test_v8_heap_rejects_supplied_invalid_coordinates(tmp_path: Path, coordinate: object) -> None:
    profile = tmp_path / "memory.heapprofile"
    profile.write_text(
        json.dumps(
            {
                "head": {
                    "id": 1,
                    "selfSize": 1,
                    "callFrame": {"functionName": "work", "lineNumber": coordinate},
                    "children": [],
                },
                "samples": [],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("rank_allocation_hotspots", [PathSource(path=str(profile))], {})
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()
