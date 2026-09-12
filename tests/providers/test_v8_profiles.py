from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.providers.capture import CAPTURE_BUILDERS
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CAPTURE_PROVIDER_CONTRACTS,
    CaptureTarget,
    PathSource,
    RequestLimits,
    RuntimeFailure,
)
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.workers.v8_profiles_contract import V8_PROFILE_WORKER, V8ProfileRequest


def test_every_capture_contract_has_exactly_one_registered_builder() -> None:
    assert set(CAPTURE_BUILDERS) == set(CAPTURE_PROVIDER_CONTRACTS)


@pytest.mark.process
def test_node_heap_capture_is_analyzable_memory_evidence(tmp_path: Path) -> None:
    executable = tmp_path / "fake-node"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

directory = Path(next(
    value.split("=", 1)[1]
    for value in sys.argv
    if value.startswith("--heap-prof-dir=")
))
name = next(
    value.split("=", 1)[1]
    for value in sys.argv
    if value.startswith("--heap-prof-name=")
)
(directory / name).write_text(json.dumps({
    "head": {
        "callFrame": {
            "functionName": "captured", "url": "app.js",
            "lineNumber": 0, "columnNumber": 0
        },
        "selfSize": 128,
        "id": 1,
        "children": []
    },
    "samples": [{"size": 128, "nodeId": 1}]
}))
"""
    )
    executable.chmod(0o755)

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[str(executable), "app.js"],
                    cwd=str(tmp_path),
                    provider_id="node-heap-profile",
                ),
                "memory.hotspots",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    capture = result["capture"]
    assert isinstance(capture, dict)
    capture_argv = capture["executions"][0]["capture_argv"]
    assert capture_argv[1] == "--heap-prof"
    assert any(value == "--heap-prof-name=profile.heapprofile" for value in capture_argv)
    assert result["provider"]["id"] == "v8-heap-profile"


def test_heap_profile_accepts_samples_before_referenced_nodes(tmp_path: Path) -> None:
    profile = tmp_path / "profile.heapprofile"
    profile.write_text(
        '{"samples":[{"size":128,"nodeId":1}],"head":'
        '{"callFrame":{"functionName":"work","url":"app.js",'
        '"lineNumber":0,"columnNumber":0},"selfSize":128,"id":1,"children":[]}}'
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "memory.hotspots", [PathSource(path=str(profile), format="heapprofile")], {}
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["sample_count"] == 1


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
    assert result["blocks"][1]["rows"]
    assert not (tmp_path / ".flameox").exists()


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
def test_v8_heap_profile_is_registered_as_memory_hotspot_evidence(tmp_path: Path) -> None:
    profile = tmp_path / "memory.heapprofile"
    profile.write_text(
        json.dumps(
            {
                "head": {
                    "callFrame": {
                        "functionName": "allocate",
                        "url": "app.js",
                        "scriptId": "1",
                        "lineNumber": 1,
                        "columnNumber": 0,
                    },
                    "selfSize": 64,
                    "id": 1,
                    "children": [],
                },
                "samples": [{"size": 64, "nodeId": 1}],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze("memory.hotspots", [PathSource(path=str(profile))], {})
    finally:
        runtime.close()

    assert result["provider"]["id"] == "v8-heap-profile"
    assert result["blocks"][0]["values"] == {
        "node_count": 1,
        "sample_count": 1,
        "total_sampled_bytes": 64,
    }
    assert result["blocks"][1]["rows"][0]["unit"] == "bytes"


@pytest.mark.process
def test_v8_heap_profile_rejects_samples_for_unknown_nodes(tmp_path: Path) -> None:
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
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("memory.hotspots", [PathSource(path=str(profile))], {})
    finally:
        runtime.close()

    assert failure.value.code == "DECODE_FAILURE"


@pytest.mark.process
def test_v8_heap_profile_enforces_total_node_limit(tmp_path: Path) -> None:
    profile = tmp_path / "wide.heapprofile"
    child = {
        "callFrame": {"functionName": "child", "url": "app.js"},
        "id": 2,
        "selfSize": 1,
        "children": [],
    }
    profile.write_text(
        json.dumps(
            {
                "head": {
                    "callFrame": {"functionName": "root", "url": "app.js"},
                    "id": 1,
                    "selfSize": 0,
                    "children": [child, child],
                },
                "samples": [],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(DomainError) as failure:
            runtime.workers.run_typed_sync(
                V8_PROFILE_WORKER,
                V8ProfileRequest(
                    profile_kind="heap",
                    artifact_path=str(profile),
                    artifact_id="test",
                    max_nodes=2,
                    max_samples=10,
                    max_rows=10,
                ),
                timeout_seconds=5,
                maximum_rss_bytes=256 * 1024 * 1024,
                maximum_writable_growth_bytes=1024,
            )
    finally:
        runtime.close()

    assert failure.value.code is ErrorCode.LIMIT_EXCEEDED


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

    assert result["blocks"][0]["values"] == {"node_count": 1_002, "sample_count": 1}
    assert result["coverage"] == {
        "rows_returned": 10,
        "rows_observed": 1_002,
        "complete": False,
    }
    assert result["blocks"][1]["rows"][0]["self_value"] == 1
