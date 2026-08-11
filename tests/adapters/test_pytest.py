from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from flameox.adapters import PytestExtractor
from flameox.application import ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    DomainError,
    ErrorCode,
    Sensitivity,
    new_id,
)
from flameox.storage import ArtifactStore, RunStore, Workspace


def _event(event: str, **fields: Any) -> str:
    return json.dumps(
        {
            "schema": "flameox.pytest-event.v1",
            "event": event,
            "observed_at_ns": fields.pop("observed_at_ns", 1_000),
            **fields,
        }
    )


def _write_events(path: Path) -> None:
    events = [
        _event(
            "run_started",
            run_started_at_ns=1_000,
            pytest_version="9.0",
            python_version="3.12",
            platform="test",
            scheduler="load",
            requested_workers="2",
        ),
        _event("collection_started", observed_at_ns=1_010),
        _event("worker_created", worker_id="gw0"),
        _event("worker_ready", worker_id="gw0"),
        _event("test_collected", nodeid="test_suite.py::test_passes"),
        _event("test_collected", nodeid="test_suite.py::test_errors"),
        _event("test_collected", nodeid="test_suite.py::test_unexecuted"),
        _event(
            "collection_finished",
            observed_at_ns=1_060,
            test_count=3,
            worker_id="gw0",
        ),
        _event(
            "fixture_setup",
            observed_at_ns=1_100,
            duration_ns=500,
            fixture="database",
            scope="session",
            nodeid="",
            worker_id="gw0",
            outcome="passed",
        ),
        _event(
            "test_phase",
            nodeid="test_suite.py::test_passes",
            worker_id="gw0",
            phase="setup",
            outcome="passed",
            duration_ns=600,
            started_at_ns=1_100,
            stopped_at_ns=1_700,
            controller_received_at_ns=1_800,
            wasxfail=False,
        ),
        _event(
            "test_phase",
            nodeid="test_suite.py::test_passes",
            worker_id="gw0",
            phase="call",
            outcome="passed",
            duration_ns=200,
            started_at_ns=1_800,
            stopped_at_ns=2_000,
            controller_received_at_ns=2_100,
            wasxfail=False,
        ),
        _event(
            "test_phase",
            nodeid="test_suite.py::test_errors",
            worker_id="gw0",
            phase="setup",
            outcome="failed",
            duration_ns=300,
            started_at_ns=2_100,
            stopped_at_ns=2_400,
            controller_received_at_ns=2_800,
            wasxfail=False,
        ),
        _event("worker_down", worker_id="gw0", outcome="clean", error_type=None),
        _event("session_finished", exit_status=1),
    ]
    path.write_text("\n".join(events) + "\n")


def test_pytest_extracts_fixture_cost_outcomes_and_failure_latency(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "pytest-events.jsonl"
    _write_events(source)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.TEST_EXECUTION)
    )

    result = PytestExtractor(workspace).extract(imported.run.run_id)

    assert result.complete is True
    assert result.interrupted is False
    payload = result.model_dump(mode="json")
    assert "completion" not in payload
    assert type(result).model_validate(payload) == result
    assert result.validated_copy() == result
    with pytest.raises(ValidationError, match="cannot be complete and interrupted"):
        type(result).model_validate({**payload, "interrupted": True})
    assert result.collected_count == 3
    assert result.executed_count == 2
    assert result.passed_count == 1
    assert result.errored_count == 1
    assert result.unexecuted_count == 1
    assert result.fixture_setup_count == 1
    assert result.fixture_setup_ns == 500
    assert result.collection_duration_ns == 50
    assert result.first_failure_observed_ns == 1_400
    assert result.first_failure_reported_ns == 1_800
    assert result.workers == ("gw0",)
    assert any("queue latency" in item for item in result.limitations)
    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT name, value_int, worker_id, dimensions['fixture'] "
            "FROM measurements ORDER BY name"
        ).fetchall()
    assert ("pytest.fixture_setup", 500, "gw0", "database") in rows
    assert ("pytest.collection", 50, None, None) in rows
    assert ("pytest.tests.unexecuted", 1, None, None) in rows
    assert ("pytest.time_to_first_failure.reported", 1_800, None, None) in rows


def test_pytest_marks_external_cuda_compile_failure_as_environment_blocked(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "pytest-events.jsonl"
    source.write_text(
        "\n".join(
            (
                _event(
                    "run_started",
                    run_started_at_ns=1_000,
                    pytest_version="9.0",
                    python_version="3.12",
                    platform="test",
                    scheduler="no",
                    requested_workers="0",
                ),
                _event("collection_started"),
                _event("test_collected", nodeid="test_gpu.py::test_compile"),
                _event("test_collected", nodeid="test_gpu.py::test_unrelated"),
                _event(
                    "test_phase",
                    nodeid="test_gpu.py::test_compile",
                    worker_id="master",
                    phase="setup",
                    outcome="failed",
                    duration_ns=10,
                    started_at_ns=1_100,
                    stopped_at_ns=1_110,
                    controller_received_at_ns=1_120,
                    wasxfail=False,
                ),
                _event(
                    "test_phase",
                    nodeid="test_gpu.py::test_unrelated",
                    worker_id="master",
                    phase="call",
                    outcome="failed",
                    duration_ns=10,
                    started_at_ns=1_100,
                    stopped_at_ns=1_110,
                    controller_received_at_ns=1_120,
                    wasxfail=False,
                ),
                _event("session_finished", exit_status=1),
            )
        )
        + "\n"
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.TEST_EXECUTION)
    )
    stderr = tmp_path / "stderr.bin"
    stderr.write_text(
        "test_gpu.py::test_compile: fatal error: cuda_runtime.h: No such file or directory\n"
    )
    stored = ArtifactStore(workspace).import_path(
        stderr,
        allowed_roots=(tmp_path,),
        max_bytes=workspace.config.capture.max_artifact_bytes,
    )
    run = RunStore(workspace).read(imported.run.run_id)
    stderr_registration = ArtifactRegistration(
        registration_id=new_id(),
        run_id=run.run_id,
        artifact_id=stored.content.artifact_id,
        display_name="stderr.bin",
        media_type="application/octet-stream",
        kind=ArtifactKind.PROCESS_OUTPUT,
        role="stderr",
        sensitivity=Sensitivity.NORMAL,
    )
    RunStore(workspace).append(
        run.model_copy(update={"revision": 2, "artifacts": (*run.artifacts, stderr_registration)}),
        expected_revision=1,
    )

    result = PytestExtractor(workspace).extract(imported.run.run_id)

    assert result.failed_count == 1
    assert result.errored_count == 0
    assert result.environment_blocked_count == 1
    assert any("environment-blocked" in item for item in result.limitations)
    with Catalog(workspace).open_snapshot() as snapshot:
        row = snapshot.execute(
            "SELECT value_int FROM measurements WHERE name = 'pytest.tests.environment_blocked'"
        ).fetchone()
    assert row == (1,)
    with Catalog(workspace).open_snapshot() as snapshot:
        classifications = snapshot.execute(
            "SELECT dimensions['nodeid'], dimensions['classification'] "
            "FROM measurements WHERE dimensions['classification'] IS NOT NULL"
        ).fetchall()
    assert classifications == [("test_gpu.py::test_compile", "environment_blocked")]


@pytest.mark.parametrize("payload", ("", "{}\n", "{broken\n"))
def test_pytest_rejects_malformed_event_streams(tmp_path: Path, payload: str) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "pytest-events.jsonl"
    source.write_text(payload)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.TEST_EXECUTION)
    )

    with pytest.raises(DomainError) as error:
        PytestExtractor(workspace).extract(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_pytest_interrupt_is_explicitly_non_complete(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "pytest-events.jsonl"
    source.write_text(
        "\n".join(
            (
                _event(
                    "run_started",
                    run_started_at_ns=1_000,
                    pytest_version="9.0",
                    python_version="3.12",
                    platform="test",
                    scheduler="no",
                    requested_workers="0",
                ),
                _event("interrupted", exception_type="KeyboardInterrupt"),
                _event("session_finished", exit_status=2),
            )
        )
        + "\n"
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.TEST_EXECUTION)
    )

    result = PytestExtractor(workspace).extract(imported.run.run_id)

    assert result.interrupted is True
    assert result.complete is False
    assert any("interrupted" in item for item in result.limitations)


def test_pytest_recovers_valid_prefix_from_truncated_final_record(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "pytest-events.jsonl"
    source.write_text(
        "\n".join(
            (
                _event(
                    "run_started",
                    run_started_at_ns=1_000,
                    pytest_version="9.0",
                    python_version="3.12",
                    platform="test",
                    scheduler="no",
                    requested_workers="0",
                ),
                _event("session_finished", exit_status=2),
            )
        )
        + '\n{"schema":'
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.TEST_EXECUTION)
    )

    result = PytestExtractor(workspace).extract(imported.run.run_id)

    assert result.complete is False
    assert any("valid prefix" in item for item in result.limitations)
