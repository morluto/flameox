from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from flameox.adapters import PytestExtractor, PythonStartupExtractor
from flameox.application import CaptureService, ExecutionPolicy
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, ExecutionStatus
from flameox.storage import ArtifactStore, Workspace


def _workspace_without_containment(tmp_path: Path) -> Workspace:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    Catalog(workspace).rebuild()
    return workspace


@pytest.mark.anyio
async def test_python_startup_capture_preserves_raw_importtime_and_samples(
    tmp_path: Path,
) -> None:
    workspace = _workspace_without_containment(tmp_path)
    (tmp_path / "startup_target.py").write_text("import json\nassert json.dumps({'ok': True})\n")
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.startup]
argv = [{json.dumps(sys.executable)}, "startup_target.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="startup",
        adapter="python-startup",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    captured = await service.execute(plan.plan_id)
    extracted = PythonStartupExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.SUCCEEDED
    assert extracted.sample_count == 5
    registration = next(
        item for item in captured.run.artifacts if item.kind is ArtifactKind.PYTHON_STARTUP
    )
    artifact = ArtifactStore(workspace).get(registration.artifact_id)
    payload = json.loads(artifact.payload_path.read_text())
    assert payload["samples"][0]["cache_semantics"] == "uncontrolled_initial"
    assert payload["samples"][1]["cache_semantics"] == "warm_process_restart"
    expected_rss_backend = "wait4_ru_maxrss" if hasattr(os, "wait4") else "psutil_polling"
    assert payload["samples"][0]["peak_rss_backend"] == expected_rss_backend
    assert "import time:" in payload["import_trace"]["raw_importtime"]


@pytest.mark.anyio
async def test_pytest_xdist_capture_attributes_repeated_fixture_cost_and_failure_latency(
    tmp_path: Path,
) -> None:
    workspace = _workspace_without_containment(tmp_path)
    (tmp_path / "test_example.py").write_text(
        "import time\n"
        "import pytest\n"
        "@pytest.fixture(scope='session', autouse=True)\n"
        "def expensive_session_fixture():\n"
        "    time.sleep(0.02)\n"
        "def test_one(): pass\n"
        "def test_two(): pass\n"
        "def test_three(): pass\n"
        "def test_failure(): assert False\n"
    )
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.tests]
argv = [{json.dumps(sys.executable)}, "-m", "pytest", "-q", "-n", "2", "test_example.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="tests",
        adapter="pytest",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    captured = await service.execute(plan.plan_id)
    extracted = PytestExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.FAILED
    assert extracted.complete is True
    assert extracted.failed_count == 1
    assert extracted.fixture_setup_count >= 2
    assert extracted.fixture_setup_ns >= 20_000_000
    assert extracted.first_failure_observed_ns is not None
    assert extracted.first_failure_reported_ns is not None
    assert extracted.first_failure_reported_ns >= extracted.first_failure_observed_ns
    assert len(extracted.workers) == 2
    with Catalog(workspace).open_snapshot() as snapshot:
        phase_workers = snapshot.execute(
            "SELECT DISTINCT worker_id FROM measurements "
            "WHERE name LIKE 'pytest.phase.%' ORDER BY worker_id"
        ).fetchall()
    assert phase_workers == [("gw0",), ("gw1",)]


@pytest.mark.anyio
async def test_pytest_timeout_preserves_partial_and_unexecuted_evidence(tmp_path: Path) -> None:
    workspace = _workspace_without_containment(tmp_path)
    (tmp_path / "test_timeout.py").write_text(
        "import time\ndef test_slow(): time.sleep(5)\ndef test_never_started(): pass\n"
    )
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.tests]
argv = [{json.dumps(sys.executable)}, "-m", "pytest", "-q", "-p", "no:randomly", "test_timeout.py"]
cwd = "."
timeout_seconds = 2
"""
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="tests",
        adapter="pytest",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    captured = await service.execute(plan.plan_id)
    extracted = PytestExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.TIMED_OUT
    assert extracted.execution_status == "timed_out"
    assert extracted.complete is False
    assert extracted.collected_count == 2
    assert extracted.unexecuted_count >= 1
    assert any("partial" in item for item in extracted.limitations)


@pytest.mark.anyio
async def test_pytest_worker_crash_recovers_fixture_sidecar(tmp_path: Path) -> None:
    workspace = _workspace_without_containment(tmp_path)
    (tmp_path / "test_crash.py").write_text(
        "import os\n"
        "import pytest\n"
        "@pytest.fixture(scope='session', autouse=True)\n"
        "def expensive_session_fixture():\n"
        "    return object()\n"
        "def test_crash(): os._exit(7)\n"
        "def test_unexecuted(): pass\n"
    )
    crash_argv = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-n",
        "1",
        "--max-worker-restart=0",
        "-p",
        "no:randomly",
        "test_crash.py",
    ]
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.tests]
argv = {json.dumps(crash_argv)}
cwd = "."
timeout_seconds = 30
"""
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="tests",
        adapter="pytest",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    captured = await service.execute(plan.plan_id)
    extracted = PytestExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.FAILED
    assert extracted.recovered_sidecar_events >= 2
    assert extracted.sidecar_recovery_failures == 0
    assert extracted.fixture_setup_count >= 1
    assert any("recovered from bounded sidecars" in item for item in extracted.limitations)
