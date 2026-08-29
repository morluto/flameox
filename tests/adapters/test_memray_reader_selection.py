from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from flameox.action_graph import ActionId, ToolAction
from flameox.adapters import MemrayExtractor
from flameox.application import ImportArtifactRequest, ImportService
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace
from flameox.workers.memray_contract import MemrayWorkerResult
from flameox.workers.protocol import WorkerOutputFile


def test_memray_extraction_names_the_exact_missing_reader_setup(tmp_path: Path) -> None:
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
        MemrayExtractor(workspace).extract(imported.run.run_id)

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


def test_memray_extraction_rejects_missing_producer_identity(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    capture = tmp_path / "memory.bin"
    capture.write_bytes(b"preserved-native-profile")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=capture, kind=ArtifactKind.MEMORY_PROFILE)
    )

    with pytest.raises(DomainError) as raised:
        MemrayExtractor(workspace).extract(imported.run.run_id)

    assert raised.value.code is ErrorCode.EVIDENCE_SCHEMA_MISMATCH


def test_memray_extraction_rejects_duplicate_worker_output_roles(tmp_path: Path) -> None:
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
        allocation_operations=0,
        total_allocated_bytes=0,
        capture_records=0,
        frame_count=0,
        has_native_traces=False,
        files=(output, output, output),
    )

    with pytest.raises(DomainError) as raised:
        MemrayExtractor._stage_worker_outputs(
            cast(Any, object()),
            result,
            tmp_path,
            tmp_path,
        )

    assert raised.value.code is ErrorCode.EVIDENCE_SCHEMA_MISMATCH
