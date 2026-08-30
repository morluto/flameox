from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters.v8_cpu_prof import V8CpuProfExtractor
from flameox.adapters.v8_heap_prof import V8HeapProfExtractor
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace

pytestmark = [pytest.mark.integration]


def _write_cpu_profile(path: Path) -> None:
    index_url = (path.parent / "index.js").as_uri()
    utils_url = (path.parent / "utils.js").as_uri()
    profile = {
        "nodes": [
            {
                "id": 0,
                "callFrame": {
                    "functionName": "(root)",
                    "url": "internal",
                    "scriptId": "0",
                    "lineNumber": -1,
                    "columnNumber": -1,
                },
                "hitCount": 0,
                "children": [1, 2],
            },
            {
                "id": 1,
                "callFrame": {
                    "functionName": "main",
                    "url": index_url,
                    "scriptId": "1",
                    "lineNumber": 10,
                    "columnNumber": 5,
                },
                "hitCount": 100,
                "children": [3],
            },
            {
                "id": 2,
                "callFrame": {
                    "functionName": "idle",
                    "url": index_url,
                    "scriptId": "1",
                    "lineNumber": 20,
                    "columnNumber": 0,
                },
                "hitCount": 50,
                "children": [],
            },
            {
                "id": 3,
                "callFrame": {
                    "functionName": "helper",
                    "url": utils_url,
                    "scriptId": "2",
                    "lineNumber": 5,
                    "columnNumber": 10,
                },
                "hitCount": 30,
                "children": [],
            },
        ],
        "samples": [1, 1, 2, 3, 1],
        "startTime": 0,
        "endTime": 100_000,
    }
    path.write_text(json.dumps(profile), encoding="utf-8")


def _write_heap_profile(path: Path) -> None:
    alloc_url = (path.parent / "alloc.js").as_uri()
    proxy_url = (path.parent / "proxy.js").as_uri()
    profile = {
        "head": {
            "callFrame": {
                "functionName": "(root)",
                "url": "internal",
                "scriptId": "0",
                "lineNumber": -1,
                "columnNumber": -1,
            },
            "selfSize": 0,
            "id": 0,
            "children": [
                {
                    "callFrame": {
                        "functionName": "allocate",
                        "url": alloc_url,
                        "scriptId": "1",
                        "lineNumber": 1,
                        "columnNumber": 0,
                    },
                    "selfSize": 4096,
                    "id": 1,
                    "children": [],
                },
                {
                    "callFrame": {
                        "functionName": "proxy",
                        "url": proxy_url,
                        "scriptId": "2",
                        "lineNumber": 10,
                        "columnNumber": 5,
                    },
                    "selfSize": 2048,
                    "id": 2,
                    "children": [
                        {
                            "callFrame": {
                                "functionName": "nested",
                                "url": proxy_url,
                                "scriptId": "2",
                                "lineNumber": 11,
                                "columnNumber": 5,
                            },
                            "selfSize": 1024,
                            "id": 3,
                            "children": [],
                        }
                    ],
                },
            ],
        },
        "samples": [
            {"size": 4096, "nodeId": 1, "ordinal": 0},
            {"size": 2048, "nodeId": 2, "ordinal": 1},
        ],
    }
    path.write_text(json.dumps(profile), encoding="utf-8")


def test_v8_cpu_prof_extractor_publishes_frame_measurements(tmp_path: Path) -> None:
    capture = tmp_path / "cpu.cpuprofile"
    _write_cpu_profile(capture)

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.SAMPLE_PROFILE,
        )
    )
    result = V8CpuProfExtractor(workspace).extract(imported.run.run_id)

    assert result.node_count == 4
    assert result.sample_count == 5
    assert result.frame_count >= 3
    assert len(result.limitations) >= 2
    with Catalog(workspace).open_snapshot() as snapshot:
        frames = snapshot.execute(
            "SELECT language, function, file FROM frames ORDER BY function"
        ).fetchall()
        frame_measurements = snapshot.execute(
            "SELECT function, self_value, inclusive_value FROM frame_measurements "
            "JOIN frames USING (frame_id) WHERE metric = 'cpu.hit_count' ORDER BY function"
        ).fetchall()
    frame_tuples = [(f[0], f[1], f[2]) for f in frames]
    assert ("JavaScript", "helper", "utils.js") in frame_tuples
    assert ("JavaScript", "main", "index.js") in frame_tuples
    assert ("JavaScript", "idle", "index.js") in frame_tuples
    assert ("helper", 30, 30) in frame_measurements
    assert ("main", 100, 130) in frame_measurements


def test_v8_cpu_prof_extractor_rejects_non_cpu_profile(tmp_path: Path) -> None:
    bad = tmp_path / "bad.cpuprofile"
    bad.write_text('{"not_a_profile": true}', encoding="utf-8")

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=bad,
            kind=ArtifactKind.SAMPLE_PROFILE,
        )
    )
    with pytest.raises(DomainError) as failure:
        V8CpuProfExtractor(workspace).extract(imported.run.run_id)
    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_v8_cpu_prof_extractor_rejects_malformed_nodes_with_domain_error(tmp_path: Path) -> None:
    capture = tmp_path / "bad.cpuprofile"
    capture.write_text(json.dumps({"nodes": [{"callFrame": {}}], "samples": []}), encoding="utf-8")

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=capture, kind=ArtifactKind.SAMPLE_PROFILE)
    )
    with pytest.raises(DomainError) as failure:
        V8CpuProfExtractor(workspace).extract(imported.run.run_id)
    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_v8_heap_prof_extractor_publishes_sampled_bytes(tmp_path: Path) -> None:
    capture = tmp_path / "heap.heapprofile"
    _write_heap_profile(capture)

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
        )
    )
    result = V8HeapProfExtractor(workspace).extract(imported.run.run_id)

    assert result.sample_count == 2
    assert result.total_sampled_bytes == 6144
    assert result.frame_count >= 3
    assert len(result.limitations) >= 3
    with Catalog(workspace).open_snapshot() as snapshot:
        frames = snapshot.execute(
            "SELECT language, function, file FROM frames ORDER BY function"
        ).fetchall()
        measurements = snapshot.execute(
            "SELECT function, self_value, inclusive_value FROM frame_measurements "
            "JOIN frames USING (frame_id) WHERE metric = 'memory.self_size'"
        ).fetchall()
    frame_tuples = [(f[0], f[1], f[2]) for f in frames]
    assert ("JavaScript", "allocate", "alloc.js") in frame_tuples
    assert ("JavaScript", "proxy", "proxy.js") in frame_tuples
    assert ("proxy", 2048, 3072) in measurements


def test_v8_heap_prof_extractor_rejects_non_heap_profile(tmp_path: Path) -> None:
    bad = tmp_path / "bad.heapprofile"
    bad.write_text('{"no_head": true}', encoding="utf-8")

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=bad,
            kind=ArtifactKind.MEMORY_PROFILE,
        )
    )
    with pytest.raises(DomainError) as failure:
        V8HeapProfExtractor(workspace).extract(imported.run.run_id)
    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_v8_heap_prof_extractor_rejects_malformed_samples_with_domain_error(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "bad.heapprofile"
    capture.write_text(
        json.dumps(
            {
                "head": {
                    "callFrame": {"functionName": "root"},
                    "selfSize": 0,
                    "children": [],
                },
                "samples": [None],
            }
        ),
        encoding="utf-8",
    )

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=capture, kind=ArtifactKind.MEMORY_PROFILE)
    )
    with pytest.raises(DomainError) as failure:
        V8HeapProfExtractor(workspace).extract(imported.run.run_id)
    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
