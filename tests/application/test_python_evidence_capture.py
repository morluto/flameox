from __future__ import annotations

import json
import sys
from pathlib import Path

import pyperf
import pytest

from flameox.adapters.pytest import PytestExtractor
from flameox.adapters.python_startup import PythonStartupExtractor
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ExecutionStatus
from flameox.storage import ArtifactStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


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
async def test_python_startup_capture_preserves_native_pyperf_and_raw_importtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace_without_containment(tmp_path)
    (tmp_path / "startup_target.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "kind = 'import' if 'importtime' in sys._xoptions else 'wall'\n"
        "with Path('startup-executions.txt').open('a') as stream:\n"
        "    stream.write(kind + '\\n')\n"
        "import json\n"
        "assert json.dumps({'ok': True})\n"
    )
    (tmp_path / "flameox.toml").write_text(
        f"""
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

    captured = await service.execute(plan.plan_token)
    extracted = PythonStartupExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.SUCCEEDED
    assert captured.run.semantics.configuration["collector_implementation_id"]
    assert extracted.sample_count == 5
    wall_registration = next(
        item
        for item in captured.run.artifacts
        if item.kind is ArtifactKind.BENCHMARK_SAMPLES and item.role == "startup_wall"
    )
    trace_registration = next(
        item
        for item in captured.run.artifacts
        if item.kind is ArtifactKind.PYTHON_STARTUP and item.role == "import_trace"
    )
    artifacts = ArtifactStore(workspace)
    wall_artifact = artifacts.get(wall_registration.artifact_id)
    suite = pyperf.BenchmarkSuite.load(str(wall_artifact.payload_path))
    benchmark = suite.get_benchmark("flameox.python_startup.wall_time")
    assert benchmark.get_nrun() == 5
    assert all(run.get_loops() == 1 and len(run.values) == 1 for run in benchmark.get_runs())
    assert all(not run.warmups for run in benchmark.get_runs())
    assert "import time:" in artifacts.get(trace_registration.artifact_id).payload_path.read_text()
    executions = (tmp_path / "startup-executions.txt").read_text().splitlines()
    assert executions.count("wall") == 5
    assert executions.count("import") == 1

    drifted = await service.plan(
        workload_name="startup",
        adapter="python-startup",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    monkeypatch.setattr(
        "flameox.application.capture.collector_implementation_id",
        lambda _adapter: "sha256:" + "0" * 64,
    )
    with pytest.raises(DomainError, match="internal collector changed"):
        await service.execute(drifted.plan_token)


@pytest.mark.anyio
async def test_pytest_xdist_capture_attributes_repeated_fixture_cost_and_failure_latency(
    tmp_path: Path,
) -> None:
    hostile_plugin = tmp_path / "_flameox_bound_pytest_plugin.py"
    hostile_plugin.write_text("raise RuntimeError('stale collector loaded')\n")
    (tmp_path / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "for candidate in Path('.diagnostics').rglob('_flameox_bound_pytest_*.py'):\n"
        "    Path('collector-mutated.txt').write_text(str(candidate))\n"
        "    candidate.chmod(0o600)\n"
        "    candidate.write_text(\"raise RuntimeError('collector mutated')\\n\")\n"
    )
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

    captured = await service.execute(plan.plan_token)
    extracted = PytestExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.FAILED
    assert captured.run.semantics.configuration["collector_implementation_id"]
    assert extracted.completion == "complete"
    assert extracted.failed_count == 1
    assert extracted.fixture_setup_count >= 2
    assert not (tmp_path / "collector-mutated.txt").exists()
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

    captured = await service.execute(plan.plan_token)
    extracted = PytestExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.TIMED_OUT
    assert extracted.execution_status == "timed_out"
    assert extracted.completion != "complete"
    assert extracted.collected_count >= extracted.executed_count
    assert any("partial" in item for item in extracted.limitations)


@pytest.mark.anyio
async def test_pytest_worker_crash_preserves_partial_native_reportlog(tmp_path: Path) -> None:
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

    captured = await service.execute(plan.plan_token)
    extracted = PytestExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.FAILED
    assert extracted.completion != "complete"
    assert extracted.collected_count >= extracted.executed_count
    assert any("partial" in item for item in extracted.limitations)
