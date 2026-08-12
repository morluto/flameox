from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from flameox.adapters import PytestExtractor
from flameox.adapters.pytest import PytestCompletionState
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


def _report(report_type: str, **fields: Any) -> str:
    return json.dumps({"$report_type": report_type, **fields})


def _test_report(nodeid: str, when: str, outcome: str, **fields: Any) -> str:
    return _report(
        "TestReport",
        nodeid=nodeid,
        when=when,
        outcome=outcome,
        duration=fields.pop("duration", 0.000_000_1),
        start=fields.pop("start", 0.000_001_1),
        stop=fields.pop("stop", 0.000_001_2),
        user_properties=fields.pop(
            "user_properties",
            [
                ["flameox.run_started_ns", 1_000],
                ["flameox.controller_received_ns", 1_300],
            ],
        ),
        **fields,
    )


def _write_events(path: Path) -> None:
    events = [
        _report("SessionStart", pytest_version="9.1.1"),
        _report(
            "CollectReport",
            nodeid="test_suite.py",
            outcome="passed",
            flameox={
                "collected_nodeids": [
                    "test_suite.py::test_passes",
                    "test_suite.py::test_errors",
                    "test_suite.py::test_unexecuted",
                ]
            },
        ),
        _test_report(
            "test_suite.py::test_passes",
            "setup",
            "passed",
            worker_id="gw0",
            duration=0.000_000_6,
            start=0.000_001_1,
            stop=0.000_001_7,
            user_properties=[
                ["flameox.run_started_ns", 1_000],
                ["flameox.controller_received_ns", 1_800],
                [
                    "flameox.fixture_setup",
                    {
                        "duration_ns": 500,
                        "fixture": "database",
                        "scope": "session",
                        "started_at_ns": 1_100,
                        "outcome": "passed",
                    },
                ],
            ],
        ),
        _test_report(
            "test_suite.py::test_passes",
            "call",
            "passed",
            worker_id="gw0",
            duration=0.000_000_2,
            start=0.000_001_8,
            stop=0.000_002,
        ),
        _test_report(
            "test_suite.py::test_errors",
            "setup",
            "failed",
            worker_id="gw0",
            duration=0.000_000_3,
            start=0.000_002_1,
            stop=0.000_002_4,
            user_properties=[
                ["flameox.run_started_ns", 1_000],
                ["flameox.controller_received_ns", 2_800],
            ],
        ),
        _report("SessionFinish", exitstatus=1),
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
    assert result.validated_copy() == result
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(result).model_validate({**payload, "interrupted": True})
    interrupted = result.validated_copy(update={"completion": PytestCompletionState.INTERRUPTED})
    assert interrupted.complete is False
    assert interrupted.interrupted is True
    assert result.collected_count == 3
    assert result.executed_count == 2
    assert result.passed_count == 1
    assert result.errored_count == 1
    assert result.unexecuted_count == 1
    assert result.fixture_setup_count == 1
    assert result.fixture_setup_ns == 500
    assert result.collection_duration_ns is None
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
    assert ("pytest.tests.unexecuted", 1, None, None) in rows
    assert ("pytest.time_to_first_failure.reported", 1_800, None, None) in rows


def test_pytest_reportlog_is_the_authoritative_producer_format(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "pytest-reportlog.jsonl"
    source.write_text(
        "\n".join(
            (
                _report("SessionStart", pytest_version="9.1.1"),
                _report("FutureReport", value="ignored by contract"),
                _report(
                    "CollectReport",
                    nodeid="tests/test_sample.py",
                    outcome="passed",
                    flameox={
                        "collected_nodeids": [
                            "tests/test_sample.py::test_passes",
                            "tests/test_sample.py::test_unexecuted",
                        ]
                    },
                ),
                _test_report(
                    "tests/test_sample.py::test_passes",
                    "setup",
                    "passed",
                    user_properties=[
                        ["flameox.run_started_ns", 1_000],
                        ["flameox.controller_received_ns", 1_800],
                        [
                            "flameox.fixture_setup",
                            {
                                "duration_ns": 500,
                                "fixture": "database",
                                "scope": "function",
                                "started_at_ns": 1_100,
                                "outcome": "passed",
                            },
                        ],
                    ],
                ),
                _test_report("tests/test_sample.py::test_passes", "call", "passed"),
                _report("SessionFinish", exitstatus=0),
            )
        )
        + "\n"
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.TEST_EXECUTION)
    )

    result = PytestExtractor(workspace).extract(imported.run.run_id)

    assert result.complete is True
    assert result.collected_count == 2
    assert result.executed_count == 1
    assert result.unexecuted_count == 1
    assert result.fixture_setup_count == 1
    assert result.fixture_setup_ns == 500


def test_pytest_marks_external_cuda_compile_failure_as_environment_blocked(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "pytest-events.jsonl"
    source.write_text(
        "\n".join(
            (
                _report("SessionStart", pytest_version="9.1.1"),
                _report(
                    "CollectReport",
                    nodeid="test_gpu.py",
                    outcome="passed",
                    flameox={
                        "collected_nodeids": [
                            "test_gpu.py::test_compile",
                            "test_gpu.py::test_unrelated",
                        ]
                    },
                ),
                _test_report(
                    "test_gpu.py::test_compile",
                    "setup",
                    "failed",
                    worker_id="master",
                ),
                _test_report(
                    "test_gpu.py::test_unrelated",
                    "call",
                    "failed",
                    worker_id="master",
                ),
                _report("SessionFinish", exitstatus=1),
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
                _report("SessionStart", pytest_version="9.1.1"),
                _report("SessionFinish", exitstatus=2),
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
                _report("SessionStart", pytest_version="9.1.1"),
                _report("SessionFinish", exitstatus=2),
            )
        )
        + '\n{"$report_type":'
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.TEST_EXECUTION)
    )

    result = PytestExtractor(workspace).extract(imported.run.run_id)

    assert result.complete is False
    assert any("valid prefix" in item for item in result.limitations)
