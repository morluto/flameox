from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from flameox.action_graph import ActionId, ToolAction
from flameox.adapters.nsight_systems import NsightSystemsExtractor
from flameox.analysis import RecipeService
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.application.trace_windows import TraceWindowService
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


def _nsight_fixture(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    nameId INTEGER NOT NULL,
    correlationId INTEGER,
    globalTid INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    nameId INTEGER NOT NULL,
    correlationId INTEGER,
    globalTid INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    demangledName INTEGER NOT NULL,
    correlationId INTEGER,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER
);
CREATE TABLE NVTX_EVENTS (
    start INTEGER NOT NULL,
    end INTEGER,
    eventType INTEGER NOT NULL,
    text TEXT NOT NULL,
    globalTid INTEGER
);
CREATE TABLE META_DATA_EXPORT (name TEXT NOT NULL, value TEXT);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    bytes INTEGER,
    copyKind INTEGER,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    bytes INTEGER,
    value INTEGER,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER
);
"""
        )
        connection.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [
                (1, "cudaLaunchKernel"),
                (2, "cudaGraphLaunch"),
                (3, "projection_kernel"),
                (4, "cuCtxSynchronize"),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            [(1, 3, 1, 41, 7), (4, 6, 2, 42, 7)],
        )
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (6, 7, 4, 43, 7)")
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(10, 15, 3, 41, 0, 1, 9), (20, 24, 3, 42, 0, 1, 9)],
        )
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (0, 30, 59, 'decode', 7)")
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (0, NULL, 75, 'CCCL', 7)")
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (0, NULL, 59, 'open-range', 7)")
        connection.executemany(
            "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
            [
                ("EXPORT_PRODUCT_NAME", "NVIDIA Nsight Systems"),
                ("EXPORT_PRODUCT_VERSION", "2026.1.3.425"),
                ("EXPORT_SCHEMA_VERSION", "3.24.18"),
                ("EXPORT_SCHEMA_VERSION_CKSUM", "fixture-checksum"),
                ("EXPORT_PARAM_LAZY", "true"),
                ("EXPORT_PARAM_TIME_NORMALIZE", "false"),
                ("EXPORT_HOST", "must-not-publish"),
                ("EXPORT_USER", "must-not-publish"),
                ("EXPORT_PARAM_INPUT_PATH_ABS", "/sensitive/input.nsys-rep"),
            ],
        )
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (7, 9, 4096, 1, 0, 1, 9)")
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (9, 10, 1024, 0, 0, 1, 9)"
        )
        connection.commit()
    finally:
        connection.close()


def _minimal_nsight_fixture(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start INTEGER NOT NULL, end INTEGER NOT NULL, nameId INTEGER NOT NULL
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start INTEGER NOT NULL, end INTEGER NOT NULL, demangledName INTEGER NOT NULL
);
CREATE TABLE NVTX_EVENTS (
    start INTEGER NOT NULL, end INTEGER, text TEXT NOT NULL
);
INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel');
INSERT INTO StringIds VALUES (2, 'projection_kernel');
INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 1, 1);
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2, 3, 2);
INSERT INTO NVTX_EVENTS VALUES (0, 3, 'legacy-range');
"""
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.anyio
@pytest.mark.process
async def test_nsight_sqlite_extracts_launch_evidence_without_nsys_installed(
    tmp_path: Path,
) -> None:
    export = tmp_path / "decode.sqlite"
    _nsight_fixture(export)
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=export,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="nsight.systems",
            producer_version="2026.1.3.425",
        )
    )
    assert imported.run.artifacts[0].producer == "nsight.systems"
    with Catalog(workspace).open_snapshot(imported.corpus_commit_id) as snapshot:
        assert snapshot.execute(
            "SELECT producer FROM artifact_registrations WHERE artifact_id = ?",
            (imported.artifact_id,),
        ).fetchall() == [("nsight.systems",)]

    with pytest.raises(DomainError) as unavailable:
        RecipeService(workspace).accelerator_launches(imported.run.run_id)
    assert isinstance(unavailable.value.next_action, ToolAction)
    assert unavailable.value.next_action.action is ActionId.EXTRACT_NSIGHT_SYSTEMS
    assert unavailable.value.next_action.arguments == {"run_id": imported.run.run_id}

    extracted = await NsightSystemsExtractor(workspace).extract(imported.run.run_id)
    repeated = await NsightSystemsExtractor(workspace).extract(imported.run.run_id)
    analysis = RecipeService(workspace).accelerator_launches(
        imported.run.run_id,
        phase="decode",
    )
    first_window = await TraceWindowService(workspace).get(
        imported.artifact_id,
        start_ns=0,
        end_ns=31,
        limit=3,
    )
    second_import = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=export,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="nsight.systems",
            producer_version="2026.1.3.425",
        )
    )
    await NsightSystemsExtractor(workspace).extract(second_import.run.run_id)
    with pytest.raises(DomainError, match="multiple runs"):
        await TraceWindowService(workspace).get(
            imported.artifact_id,
            start_ns=0,
            end_ns=31,
            limit=3,
        )
    second_window = await TraceWindowService(workspace).get(
        imported.artifact_id,
        start_ns=0,
        end_ns=31,
        limit=3,
        cursor=first_window.next_cursor,
    )
    late_window = await TraceWindowService(workspace).get(
        imported.artifact_id,
        run_id=imported.run.run_id,
        start_ns=25,
        end_ns=29,
    )

    assert repeated.corpus_commit_id == extracted.corpus_commit_id
    assert extracted.event_count == 10
    assert extracted.compatibility_family == "nsight-systems.sqlite.cuda.v1"
    assert "CUPTI_ACTIVITY_KIND_RUNTIME" in extracted.observed_tables
    assert extracted.runtime_event_count == 2
    assert extracted.driver_event_count == 1
    assert extracted.kernel_event_count == 2
    assert extracted.nvtx_event_count == 3
    assert extracted.memory_copy_event_count == 1
    assert extracted.memory_set_event_count == 1
    assert extracted.coverage == {
        "cuda_runtime": True,
        "cuda_driver": True,
        "cuda_kernels": True,
        "nvtx": True,
        "memory_copies": True,
        "memory_sets": True,
        "correlation_ids": True,
        "stream_identity": True,
        "thread_identity": True,
        "process_identity": False,
    }
    assert extracted.asserted_producer_version == "2026.1.3.425"
    assert extracted.observed_product_name == "NVIDIA Nsight Systems"
    assert extracted.observed_product_version == "2026.1.3.425"
    assert extracted.export_schema_version == "3.24.18"
    assert extracted.export_schema_checksum == "fixture-checksum"
    assert extracted.export_settings == {"lazy_tables": "true", "time_normalized": "false"}
    assert "must-not-publish" not in extracted.model_dump_json()
    assert "/sensitive" not in extracted.model_dump_json()
    assert first_window.provider == "nsight.systems"
    assert first_window.total == 10
    assert first_window.returned == 3
    assert first_window.next_cursor is not None
    assert second_window.returned == 3
    assert {event.event_id for event in first_window.events}.isdisjoint(
        event.event_id for event in second_window.events
    )
    metadata_event = next(event for event in first_window.events if event.event_kind == "metadata")
    assert metadata_event.category == "nvtx"
    assert metadata_event.event_type == 75
    assert any(event.thread == 7 for event in first_window.events)
    assert "Cross-provider clock alignment is unavailable." in first_window.limitations
    incomplete = next(event for event in late_window.events if event.name == "open-range")
    assert incomplete.end_ns is None
    assert incomplete.event_kind == "incomplete_range"
    with Catalog(workspace).open_snapshot(extracted.corpus_commit_id) as snapshot:
        nvtx_values = snapshot.execute(
            "SELECT value_json FROM observations "
            "WHERE kind = 'trace.event' AND json_extract_string(value_json, '$.category') = 'nvtx' "
            "ORDER BY json_extract_string(value_json, '$.event_kind')"
        ).fetchall()
    assert any('"event_kind":"metadata"' in value for (value,) in nvtx_values)
    assert any('"event_type":75' in value for (value,) in nvtx_values)
    region = analysis.regions[0]
    assert region.region == "decode"
    assert region.direct_launch_count == 1
    assert region.graph_launch_count == 1
    assert region.kernel_count == 2
    assert region.idle_gap_total_ns == 5


@pytest.mark.anyio
@pytest.mark.process
async def test_nsight_certified_2025_5_export_preserves_graph_launches_and_identity(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "nsight_systems" / "nsight-2025.5.2.sqlite"
    payload = fixture.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "de09b9460b9f95a2b51b66d502e28cb7cf4e5dee6b748ab6c39cb6587dcb44a3"
    )
    export = tmp_path / fixture.name
    export.write_bytes(payload)
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=export,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="nsight.systems",
            producer_version="2025.5.2.266-255236693005v0",
        )
    )

    extracted = await NsightSystemsExtractor(workspace).extract(imported.run.run_id)
    repeated = await NsightSystemsExtractor(workspace).extract(imported.run.run_id)
    analysis = RecipeService(workspace).accelerator_launches(imported.run.run_id)
    region = analysis.regions[0]

    assert extracted.asserted_producer_version == "2025.5.2.266-255236693005v0"
    assert extracted.observed_product_version == "2025.5.2.266"
    assert extracted.export_schema_version == "3.24.0"
    assert (
        extracted.artifact_id
        == "sha256:de09b9460b9f95a2b51b66d502e28cb7cf4e5dee6b748ab6c39cb6587dcb44a3"
    )
    assert repeated.corpus_commit_id == extracted.corpus_commit_id
    assert extracted.event_count == 37
    assert extracted.runtime_event_count == 33
    assert extracted.kernel_event_count == 3
    assert extracted.memory_copy_event_count == 1
    assert extracted.coverage["cuda_runtime"]
    assert extracted.coverage["cuda_kernels"]
    assert extracted.coverage["correlation_ids"]
    assert extracted.coverage["stream_identity"]
    assert "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in extracted.observed_tables
    assert region.direct_launch_count == 4
    assert region.graph_launch_count == 2
    assert region.kernel_count == 3
    assert region.correlated_kernel_count == 3
    assert region.stream_count == 1
    assert region.kernel_names[0].name == "add_one(const float *, float *, int)"


@pytest.mark.anyio
@pytest.mark.process
async def test_nsight_extractor_rejects_unknown_sqlite_schema(tmp_path: Path) -> None:
    export = tmp_path / "unknown.sqlite"
    connection = sqlite3.connect(export)
    connection.execute("CREATE TABLE unrelated (value TEXT)")
    connection.close()
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=export,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="nsight.systems",
        )
    )

    with pytest.raises(DomainError) as failure:
        await NsightSystemsExtractor(workspace).extract(imported.run.run_id)

    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "CUDA runtime and kernel tables are required" in failure.value.message


@pytest.mark.anyio
@pytest.mark.process
async def test_nsight_extractor_reports_missing_optional_tables(tmp_path: Path) -> None:
    export = tmp_path / "minimal.sqlite"
    _minimal_nsight_fixture(export)
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=export,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="nsight.systems",
        )
    )

    extracted = await NsightSystemsExtractor(workspace).extract(imported.run.run_id)

    assert extracted.coverage["cuda_runtime"]
    assert extracted.coverage["cuda_kernels"]
    assert not extracted.coverage["cuda_driver"]
    assert extracted.coverage["nvtx"]
    assert not extracted.coverage["memory_copies"]
    assert not extracted.coverage["memory_sets"]
    assert extracted.driver_event_count == 0
    assert extracted.memory_copy_event_count == 0
    assert extracted.memory_set_event_count == 0
    assert extracted.nvtx_event_count == 1
    assert any("identity is unavailable" in item for item in extracted.limitations)


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.parametrize("asserted_version", ["2025.5.2", "2026.1.3.999"])
async def test_nsight_extractor_rejects_mismatched_asserted_version(
    tmp_path: Path, asserted_version: str
) -> None:
    export = tmp_path / "mismatch.sqlite"
    _nsight_fixture(export)
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=export,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="nsight.systems",
            producer_version=asserted_version,
        )
    )

    with pytest.raises(DomainError) as failure:
        await NsightSystemsExtractor(workspace).extract(imported.run.run_id)

    assert failure.value.code is ErrorCode.ADAPTER_INCOMPATIBLE
    assert failure.value.details["observed_product_version"] == "2026.1.3.425"
