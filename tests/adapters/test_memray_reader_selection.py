from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from flameox.action_graph import ActionId, ToolAction
from flameox.adapters import MemrayExtractor
from flameox.adapters.memray import memray_extraction_limits
from flameox.application import ImportArtifactRequest, ImportService
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace
from flameox.workers.memray_contract import (
    MemrayExtractionCoverage,
    MemrayMetricCoverage,
    MemrayWorkerResult,
)
from flameox.workers.protocol import WorkerOutputFile


@pytest.mark.anyio
async def test_memray_extraction_names_the_exact_missing_reader_setup(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    capture = tmp_path / "memory.bin"
    capture.write_bytes(b"preserved-native-profile")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version="1.20.0",
        )
    )

    with pytest.raises(DomainError) as raised:
        await MemrayExtractor(workspace).extract(
            imported.run.run_id, limits=memray_extraction_limits(workspace)
        )

    assert raised.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert raised.value.details == {
        "producer_version": "1.20.0",
        "required_reader": "memray==1.20.0",
    }
    assert "memray_reader_version='1.20.0'" in raised.value.remediation[0]
    assert raised.value.next_action == ToolAction(
        action=ActionId.START_CAPABILITY_SETUP,
        arguments={
            "adapters": ["memray"],
            "idempotency_key": "memray-reader-1.20.0",
            "memray_reader_version": "1.20.0",
        },
    )
    assert capture.read_bytes() == b"preserved-native-profile"


@pytest.mark.anyio
async def test_memray_extraction_rejects_missing_producer_identity(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    capture = tmp_path / "memory.bin"
    capture.write_bytes(b"preserved-native-profile")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=capture, kind=ArtifactKind.MEMORY_PROFILE)
    )

    with pytest.raises(DomainError) as raised:
        await MemrayExtractor(workspace).extract(
            imported.run.run_id, limits=memray_extraction_limits(workspace)
        )

    assert raised.value.code is ErrorCode.EVIDENCE_SCHEMA_MISMATCH


def test_memray_extraction_rejects_duplicate_worker_output_roles(tmp_path: Path) -> None:
    coverage = MemrayExtractionCoverage(
        high_watermark=MemrayMetricCoverage(
            records_seen=0,
            records_selected=0,
            record_bytes_seen=0,
            record_bytes_selected=0,
            dropped_stack_frames=0,
            dropped_stack_frame_bytes=0,
        ),
        retained_end=MemrayMetricCoverage(
            records_seen=0,
            records_selected=0,
            record_bytes_seen=0,
            record_bytes_selected=0,
            dropped_stack_frames=0,
            dropped_stack_frame_bytes=0,
        ),
        allocation_volume=MemrayMetricCoverage(
            records_seen=0,
            records_selected=0,
            record_bytes_seen=0,
            record_bytes_selected=0,
            dropped_stack_frames=0,
            dropped_stack_frame_bytes=0,
        ),
        temporary=MemrayMetricCoverage(
            records_seen=0,
            records_selected=0,
            record_bytes_seen=0,
            record_bytes_selected=0,
            dropped_stack_frames=0,
            dropped_stack_frame_bytes=0,
        ),
        frames_published=0,
        aggregate_rows_published=0,
        frame_contributions_dropped=0,
        frame_contribution_bytes_dropped=0,
        aggregate_rows_dropped=0,
        aggregate_inclusive_bytes_dropped=0,
        edge_rows_published=0,
        edge_rows_dropped=0,
        edge_weight_bytes_dropped=0,
        representative_stacks_published=0,
        representative_stacks_dropped=0,
        representative_stack_weight_bytes_dropped=0,
        output_bytes=0,
    )
    output = WorkerOutputFile(
        role="measurements",
        relative_path="measurements.parquet",
        media_type="application/vnd.apache.parquet",
        byte_length=0,
        sha256="sha256:" + "0" * 64,
    )
    result = MemrayWorkerResult(
        reader_version="1.20.0",
        peak_memory_bytes=0,
        retained_end_bytes=0,
        temporary_allocated_bytes=0,
        temporary_allocation_threshold=1,
        allocation_operations=0,
        total_allocated_bytes=0,
        capture_records=0,
        has_native_traces=False,
        coverage=coverage,
        files=(output, output, output, output, output),
    )

    with pytest.raises(DomainError) as raised:
        MemrayExtractor._stage_worker_outputs(
            cast(Any, object()),
            result,
            tmp_path,
            tmp_path,
        )

    assert raised.value.code is ErrorCode.EVIDENCE_SCHEMA_MISMATCH
