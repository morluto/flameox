from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    PathSource,
    RequestLimits,
)


@pytest.mark.unit
def test_pytest_stream_has_typed_summary_and_bounded_rows(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event": "run_started", "run_started_at_ns": 1},
                {"event": "test_collected", "nodeid": "test_ok"},
                {
                    "event": "test_phase",
                    "nodeid": "test_ok",
                    "phase": "call",
                    "outcome": "passed",
                    "duration_ns": 10,
                    "worker_id": "main",
                },
                {"event": "test_collected", "nodeid": "test_missing"},
                {"event": "run_finished"},
            )
        )
        + "\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "failures.summary",
            [PathSource(path=str(events), format="pytest")],
            {},
            limits=RequestLimits(max_rows=2),
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"] == {
        "completion": "complete",
        "collected": 2,
        "executed": 1,
        "unexecuted": 1,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "errored": 0,
    }
    assert result["coverage"] == {"rows_returned": 1, "rows_observed": 1, "complete": True}
    assert result["blocks"][1]["rows"][0]["classification"] == "unexecuted"


@pytest.mark.unit
def test_pytest_failure_identities_precede_large_successful_population(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    nodeids = [f"tests/test_large.py::test_{index:04d}" for index in range(1_472)]
    failing = set(nodeids[-15:])
    payloads = [{"event": "test_collected", "nodeid": nodeid} for nodeid in nodeids]
    payloads.extend(
        {
            "event": "test_phase",
            "nodeid": nodeid,
            "phase": "call",
            "outcome": "failed" if nodeid in failing else "passed",
        }
        for nodeid in nodeids
    )
    payloads.append({"event": "run_finished"})
    events.write_text("\n".join(json.dumps(item) for item in payloads) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "failures.summary",
            [PathSource(path=str(events), format="pytest")],
            {},
            limits=RequestLimits(max_rows=10),
        )
        second = runtime.analyze(
            "failures.summary",
            [PathSource(path=str(events), format="pytest")],
            {},
            limits=RequestLimits(max_rows=10),
            continuation=result["continuation"],
        )
    finally:
        runtime.close()

    rows = [*result["blocks"][1]["rows"], *second["blocks"][1]["rows"]]
    assert {row["nodeid"] for row in rows} == failing
    assert all(row["classification"] == "failed" for row in rows)
    assert second["coverage"]["complete"] is True


@pytest.mark.process
def test_pytest_collection_failure_identity_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "test_broken.py"
    broken.write_text("raise RuntimeError('collection failed')\n")

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-m", "pytest", str(broken), "-q"],
                    cwd=str(tmp_path),
                    provider_id="pytest",
                ),
                "failures.summary",
            )
        finally:
            runtime.close()

        assert result["blocks"][0]["values"]["errored"] == 1
        assert result["blocks"][1]["rows"] == [
            {
                "index": 1,
                "nodeid": "test_broken.py",
                "classification": "errored",
                "failing_phase": "collection",
                "phase_outcomes": {"collection": "failed"},
            }
        ]

    anyio.run(exercise)


@pytest.mark.process
def test_pytest_capture_produces_analyzable_session_evidence(tmp_path: Path) -> None:
    (tmp_path / "local_module.py").write_text("VALUE = 7\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "conftest.py").write_text("from local_module import VALUE\nassert VALUE == 7\n")
    test_file = tests_dir / "test_sample.py"
    test_file.write_text(
        "from local_module import VALUE\n\n"
        "def test_passes():\n"
        "    assert VALUE == 7\n\n"
        "def test_skips():\n"
        "    import pytest\n"
        "    pytest.skip('bounded example')\n"
    )

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-m", "pytest", "-q", str(test_file)],
                    cwd=str(tmp_path),
                    provider_id="pytest",
                ),
                "failures.summary",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["provider"] == {"id": "pytest", "version": "event-stream-v1"}
    assert result["blocks"][0]["values"] == {
        "completion": "complete",
        "collected": 2,
        "executed": 2,
        "unexecuted": 0,
        "passed": 1,
        "failed": 0,
        "skipped": 1,
        "errored": 0,
    }
    assert result["capture"]["executions"][0]["status"] == "succeeded"
    assert result["inputs"][0]["format"] == "pytest"
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
@pytest.mark.parametrize("workers", [0, 2])
def test_pytest_capture_uses_its_plugin_without_importing_workload_flameox(
    tmp_path: Path, workers: int
) -> None:
    shadow = tmp_path / "flameox"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise ImportError('workload Flameox is incompatible')\n")
    test_file = tmp_path / "test_workload.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.fixture(scope='session')\n"
        "def resource():\n"
        "    return 7\n"
        "@pytest.mark.parametrize('case', range(4))\n"
        "def test_case(resource, case):\n"
        "    assert resource == 7\n"
    )

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-m", "pytest", "-q", str(test_file)]
                    + (["-n", str(workers)] if workers else []),
                    cwd=str(tmp_path),
                    provider_id="pytest",
                ),
                "pytest.fixtures",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["capture"]["outcome"]["status"] == "succeeded"
    assert result["analysis_failure"] is None
    invocations = [
        row
        for row in result["blocks"][1]["rows"]
        if row["row_kind"] == "fixture_invocation" and row["fixture"] == "resource"
    ]
    assert len(invocations) == max(1, workers)
    assert all(row["complete"] for row in invocations)


@pytest.mark.process
def test_pytest_fixture_capture_attributes_session_work_per_xdist_worker(
    tmp_path: Path,
) -> None:
    (tmp_path / "conftest.py").write_text(
        "import time\n\n"
        "import pytest\n\n"
        "@pytest.fixture(scope='session', autouse=True)\n"
        "def database():\n"
        "    time.sleep(0.01)\n"
        "    yield\n"
        "    time.sleep(0.01)\n"
    )
    test_file = tmp_path / "test_parallel.py"
    test_file.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('case', range(4))\n"
        "def test_case(case):\n"
        "    assert case >= 0\n"
    )

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-m", "pytest", "-q", "-n", "2", str(test_file)],
                    cwd=str(tmp_path),
                    provider_id="pytest",
                ),
                "pytest.fixtures",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    rows = result["blocks"][1]["rows"]
    aggregate = next(
        row
        for row in rows
        if row["row_kind"] == "fixture_aggregate" and row["fixture"] == "database"
    )
    assert aggregate["scope"] == "session"
    assert aggregate["invocation_count"] == 2
    assert aggregate["worker_count"] == 2
    assert aggregate["setup_work_ns"] > 0
    assert aggregate["teardown_work_ns"] > 0
    assert aggregate["incomplete_invocation_count"] == 0
    invocations = [
        row
        for row in rows
        if row["row_kind"] == "fixture_invocation" and row["fixture"] == "database"
    ]
    assert {row["worker_id"] for row in invocations} == {"gw0", "gw1"}
    assert result["provider"] == {"id": "pytest", "version": "event-stream-v2"}


@pytest.mark.process
def test_pytest_fixture_capture_retains_teardown_failure(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(
        "import pytest\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def broken_teardown():\n"
        "    yield\n"
        "    raise RuntimeError('teardown failed')\n"
    )
    test_file = tmp_path / "test_teardown.py"
    test_file.write_text("def test_body_passes():\n    pass\n")

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-m", "pytest", "-q", str(test_file)],
                    cwd=str(tmp_path),
                    provider_id="pytest",
                ),
                "pytest.fixtures",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    metrics = result["blocks"][0]["values"]
    assert metrics["teardown_failure_count"] == 1
    invocation = next(
        row
        for row in result["blocks"][1]["rows"]
        if row["row_kind"] == "fixture_invocation" and row["fixture"] == "broken_teardown"
    )
    assert invocation["teardown_failure_reported"] is True
    assert invocation["complete"] is False


@pytest.mark.process
def test_pytest_capture_does_not_instrument_nested_pytest_processes(tmp_path: Path) -> None:
    nested = tmp_path / "test_nested.py"
    nested.write_text("def test_nested():\n    pass\n")
    outer = tmp_path / "test_outer.py"
    outer.write_text(
        "import subprocess\n"
        "import sys\n\n"
        "def test_outer():\n"
        f"    completed = subprocess.run([sys.executable, '-m', 'pytest', '-q', {str(nested)!r}])\n"
        "    assert completed.returncode == 0\n"
    )

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-m", "pytest", "-q", str(outer)],
                    cwd=str(tmp_path),
                    provider_id="pytest",
                ),
                "failures.summary",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["blocks"][0]["values"]["collected"] == 1
    assert result["blocks"][0]["values"]["executed"] == 1


@pytest.mark.unit
def test_pytest_interruption_is_not_overwritten_by_session_finish(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event": "run_started", "run_started_at_ns": 1},
                {"event": "interrupted"},
                {"event": "run_finished", "exitstatus": 2},
            )
        )
        + "\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "failures.summary", [PathSource(path=str(events), format="pytest")], {}
        )
        fixtures = runtime.analyze(
            "pytest.fixtures", [PathSource(path=str(events), format="pytest")], {}
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["completion"] == "interrupted"
    assert fixtures["blocks"][0]["values"]["completion"] == "interrupted"


@pytest.mark.unit
def test_pytest_nonzero_session_exit_remains_visible(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event": "run_started", "run_started_at_ns": 1},
                {"event": "run_finished", "exitstatus": 5},
            )
        )
        + "\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "failures.summary", [PathSource(path=str(events), format="pytest")], {}
        )
        fixtures = runtime.analyze(
            "pytest.fixtures", [PathSource(path=str(events), format="pytest")], {}
        )
    finally:
        runtime.close()

    metrics = result["blocks"][0]["values"]
    assert metrics["completion"] == "failed"
    assert metrics["exit_status"] == 5
    assert fixtures["blocks"][0]["values"]["completion"] == "failed"
    assert fixtures["blocks"][0]["values"]["exit_status"] == 5


@pytest.mark.unit
def test_pytest_retries_preserve_failed_attempts(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    payloads = [
        {"event": "test_collected", "nodeid": "test_flaky"},
        {"event": "test_phase", "nodeid": "test_flaky", "phase": "setup", "outcome": "passed"},
        {"event": "test_phase", "nodeid": "test_flaky", "phase": "call", "outcome": "failed"},
        {"event": "test_phase", "nodeid": "test_flaky", "phase": "call", "outcome": "passed"},
        {
            "event": "test_phase",
            "nodeid": "test_flaky",
            "phase": "teardown",
            "outcome": "passed",
        },
        {"event": "run_finished", "exitstatus": 0},
    ]
    events.write_text("\n".join(json.dumps(event) for event in payloads) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "failures.summary", [PathSource(path=str(events), format="pytest")], {}
        )
    finally:
        runtime.close()

    metrics = result["blocks"][0]["values"]
    assert metrics["passed"] == 1
    assert metrics["flaky"] == 1
    assert metrics["retried"] == 1
    assert result["blocks"][1]["rows"] == [
        {
            "index": 1,
            "nodeid": "test_flaky",
            "classification": "flaky",
            "failing_phase": "call",
            "phase_outcomes": {"setup": "passed", "call": "passed", "teardown": "passed"},
            "phase_attempts": {
                "setup": ["passed"],
                "call": ["failed", "passed"],
                "teardown": ["passed"],
            },
        }
    ]


@pytest.mark.unit
def test_pytest_rerun_outcome_is_flaky_attempt(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    payloads = [
        {"event": "test_collected", "nodeid": "test_flaky"},
        {"event": "test_phase", "nodeid": "test_flaky", "phase": "setup", "outcome": "passed"},
        {"event": "test_phase", "nodeid": "test_flaky", "phase": "call", "outcome": "rerun"},
        {"event": "test_phase", "nodeid": "test_flaky", "phase": "call", "outcome": "passed"},
        {"event": "test_phase", "nodeid": "test_flaky", "phase": "teardown", "outcome": "passed"},
        {"event": "run_finished", "exitstatus": 0},
    ]
    events.write_text("\n".join(json.dumps(event) for event in payloads) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "failures.summary", [PathSource(path=str(events), format="pytest")], {}
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["flaky"] == 1
    assert result["blocks"][1]["rows"][0]["classification"] == "flaky"
    assert result["blocks"][1]["rows"][0]["failing_phase"] == "call"


@pytest.mark.unit
def test_pytest_fixture_projection_aggregates_workers_and_preserves_incomplete_runs(
    tmp_path: Path,
) -> None:
    events = tmp_path / "pytest.jsonl"
    fixture_events = [
        {
            "event": "fixture_phase",
            "fixture": "database",
            "scope": "session",
            "phase": "setup",
            "outcome": "passed",
            "duration_ns": 10,
            "worker_id": "gw0",
            "invocation_id": "gw0:1",
            "nodeid": "",
        },
        {
            "event": "fixture_phase",
            "fixture": "database",
            "scope": "session",
            "phase": "teardown",
            "outcome": "observed",
            "duration_ns": 5,
            "worker_id": "gw0",
            "invocation_id": "gw0:1",
            "nodeid": "",
        },
        {
            "event": "fixture_phase",
            "fixture": "database",
            "scope": "session",
            "phase": "setup",
            "outcome": "passed",
            "duration_ns": 12,
            "worker_id": "gw1",
            "invocation_id": "gw1:1",
            "nodeid": "",
        },
        {"event": "interrupted"},
    ]
    events.write_text("\n".join(json.dumps(event) for event in fixture_events) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "pytest.fixtures", [PathSource(path=str(events), format="pytest")], {}
        )
    finally:
        runtime.close()

    metrics = result["blocks"][0]["values"]
    assert metrics["completion"] == "interrupted"
    assert metrics["worker_count"] == 2
    assert metrics["invocation_count"] == 2
    assert metrics["incomplete_invocation_count"] == 1
    aggregate = result["blocks"][1]["rows"][0]
    assert aggregate == {
        "row_kind": "fixture_aggregate",
        "fixture": "database",
        "scope": "session",
        "invocation_count": 2,
        "worker_count": 2,
        "setup_work_ns": 22,
        "teardown_work_ns": 5,
        "known_work_ns": 27,
        "incomplete_invocation_count": 1,
    }
