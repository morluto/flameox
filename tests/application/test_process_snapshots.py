from __future__ import annotations

from pathlib import Path

import pytest

from flameox.action_graph import ActionId, ManualAction
from flameox.application import ImportArtifactRequest, ImportService
from flameox.application.evidence_rows import (
    process_observation_coverage,
    process_observation_rows,
)
from flameox.application.lifecycle import LifecycleEvidenceService
from flameox.domain import DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.evidence_status import EvidenceStatus
from flameox.execution import ProcessDiscoverySource, ProcessObservation, ProcessSnapshotPhase
from flameox.storage import Workspace


def _create_run(workspace: Workspace, tmp_path: Path) -> str:
    source = tmp_path / "source.bin"
    source.write_bytes(b"evidence")
    return ImportService(workspace).import_artifact(ImportArtifactRequest(path=source)).run.run_id


@pytest.mark.parametrize(
    ("observations", "expected_status"),
    [
        ((), "unavailable"),
        (
            (
                ProcessObservation(
                    pid=100,
                    discovery_source=ProcessDiscoverySource.ROOT,
                    snapshot_phase=ProcessSnapshotPhase.RUNNING,
                ),
            ),
            "available",
        ),
        (
            (
                ProcessObservation(
                    pid=100,
                    discovery_source=ProcessDiscoverySource.ROOT,
                    snapshot_phase=ProcessSnapshotPhase.RUNNING,
                    failures=("memory_info",),
                ),
            ),
            "partial",
        ),
    ],
)
def test_process_snapshot_query_reports_observation_coverage(
    tmp_path: Path,
    observations: tuple[ProcessObservation, ...],
    expected_status: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id = _create_run(workspace, tmp_path)
    artifact_id = "sha256:" + "a" * 64
    summaries, entries = process_observation_rows(
        run_id,
        observations,
        artifact_id=artifact_id,
        evidence_status=process_observation_coverage(observations)[0],
    )
    GenerationPublisher(workspace).publish_rows(
        {"process_snapshots": summaries, "process_snapshot_entries": entries},
        publisher="process-observation-test",
        publisher_version="1",
        input_run_ids=(run_id,),
    )

    result = LifecycleEvidenceService(workspace).get_process_snapshot(run_id=run_id)

    assert result.evidence.status == expected_status
    assert result.run_id == run_id
    assert result.total == len(observations)
    if expected_status == "available":
        assert result.next_action is None
        assert result.limitations == ()
    else:
        assert isinstance(result.next_action, ManualAction)
        assert result.next_action.suggested_action is ActionId.PLAN_CAPTURE
        assert result.limitations


def test_missing_process_snapshot_is_unavailable_not_successfully_empty(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id = _create_run(workspace, tmp_path)

    result = LifecycleEvidenceService(workspace).get_process_snapshot(run_id=run_id)

    assert result.evidence.status == "unavailable"
    assert result.items == ()
    assert result.limitations == ("no_process_snapshot_evidence",)
    assert isinstance(result.next_action, ManualAction)


def test_process_snapshot_rejects_unknown_run(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as captured:
        LifecycleEvidenceService(workspace).get_process_snapshot(run_id="missing-run")

    assert captured.value.code is ErrorCode.RUN_NOT_FOUND


def test_complete_empty_process_snapshot_remains_distinct_from_unavailable(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id = _create_run(workspace, tmp_path)
    summaries, entries = process_observation_rows(
        run_id,
        (),
        artifact_id="sha256:" + "a" * 64,
        evidence_status=EvidenceStatus.EMPTY,
    )
    GenerationPublisher(workspace).publish_rows(
        {"process_snapshots": summaries, "process_snapshot_entries": entries},
        publisher="complete-empty-process-observation-test",
        publisher_version="1",
        input_run_ids=(run_id,),
    )

    result = LifecycleEvidenceService(workspace).get_process_snapshot(run_id=run_id)

    assert result.evidence.status == "empty"
    assert result.total == 0
    assert result.next_action is None


def test_process_snapshot_paginates_against_authoritative_entry_count(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id = _create_run(workspace, tmp_path)
    observations = tuple(
        ProcessObservation(
            pid=pid,
            discovery_source=ProcessDiscoverySource.ROOT,
            snapshot_phase=ProcessSnapshotPhase.RUNNING,
        )
        for pid in (100, 101, 102)
    )
    summaries, entries = process_observation_rows(
        run_id,
        observations,
        artifact_id="sha256:" + "a" * 64,
        evidence_status=EvidenceStatus.AVAILABLE,
    )
    GenerationPublisher(workspace).publish_rows(
        {"process_snapshots": summaries, "process_snapshot_entries": entries},
        publisher="paginated-process-observation-test",
        publisher_version="1",
        input_run_ids=(run_id,),
    )

    first = LifecycleEvidenceService(workspace).get_process_snapshot(run_id=run_id, limit=2)
    second = LifecycleEvidenceService(workspace).get_process_snapshot(
        run_id=run_id, limit=2, cursor=first.next_cursor
    )

    assert first.total == 3
    assert first.returned == 2
    assert first.truncated is True
    assert second.total == 3
    assert second.returned == 1
    assert second.next_cursor is None


@pytest.mark.parametrize(
    ("status", "entry_count"),
    [("unknown", 0), ("empty", 1)],
)
def test_process_snapshot_rejects_invalid_summary_contract(
    tmp_path: Path,
    status: str,
    entry_count: int,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id = _create_run(workspace, tmp_path)
    summaries, _ = process_observation_rows(
        run_id,
        (),
        artifact_id="sha256:" + "a" * 64,
        evidence_status=EvidenceStatus.EMPTY,
    )
    summaries[0]["evidence_status"] = status
    summaries[0]["entry_count"] = entry_count
    GenerationPublisher(workspace).publish_rows(
        {"process_snapshots": summaries},
        publisher="invalid-process-summary-test",
        publisher_version="1",
        input_run_ids=(run_id,),
    )

    with pytest.raises(DomainError) as captured:
        LifecycleEvidenceService(workspace).get_process_snapshot(run_id=run_id)

    assert captured.value.code is ErrorCode.WORKSPACE_INVALID
