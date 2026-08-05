from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flameox.adapters import PyPerfExtractor
from flameox.adapters.builtins import build_capture_invocation
from flameox.analysis import RecipeService
from flameox.application import (
    CaptureService,
    ExecutionPolicy,
    ImportArtifactRequest,
    ImportService,
)
from flameox.domain import ArtifactKind, DomainError, ErrorCode, ExecutionStatus
from flameox.storage import Workspace
from flameox.storage.artifacts import StoredArtifact


def test_python_startup_invocation_uses_declared_workload_timeout(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "python-startup",
        (sys.executable, "startup_target.py"),
        tmp_path,
        executable=sys.executable,
        timeout_seconds=7.5,
    )

    timeout_index = invocation.argv.index("--timeout-seconds")
    assert invocation.argv[timeout_index + 1] == "7.5"


def test_pytest_capture_plan_rejects_non_pytest_workload(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as failure:
        build_capture_invocation(
            "pytest",
            ("not-pytest", "-m", "pytest", "-q"),
            tmp_path,
            executable=None,
        )

    assert failure.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "pytest" in failure.value.message


def test_torch_capture_launcher_does_not_require_flameox_in_workload_venv(
    tmp_path: Path,
) -> None:
    invocation = build_capture_invocation(
        "torch.profiler",
        ("/workload/.venv/bin/python", "-m", "benchmarks.benchmark_kda_decode", "--repeats", "1"),
        tmp_path,
        executable=None,
    )

    assert Path(invocation.argv[1]).name == "torch_launcher.py"
    assert invocation.argv[2:4] == (
        "--output",
        str(tmp_path / "torch-trace.json"),
    )
    assert invocation.argv[4] == "--config"
    assert json.loads(invocation.argv[5])["mode"] == "whole_entrypoint"
    assert invocation.argv[6:] == (
        "--module",
        "benchmarks.benchmark_kda_decode",
        "--repeats",
        "1",
    )


def test_torch_capture_launcher_binds_declared_inline_python(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "torch.profiler",
        (sys.executable, "-c", "print('inline')", "argument"),
        tmp_path,
        executable=None,
    )

    assert invocation.argv[6:] == (
        "--inline-code=print('inline')",
        "--",
        "argument",
    )


@pytest.mark.process
def test_torch_capture_launcher_runs_declared_inline_python(tmp_path: Path) -> None:
    (tmp_path / "torch.py").write_text(
        "from pathlib import Path\n"
        "class ProfilerActivity:\n"
        "    CPU = 'cpu'\n"
        "    CUDA = 'cuda'\n"
        "class _Profile:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *_): return False\n"
        "    def export_chrome_trace(self, path): Path(path).write_text('trace')\n"
        "class profiler:\n"
        "    ProfilerActivity = ProfilerActivity\n"
        "    @staticmethod\n"
        "    def profile(**_): return _Profile()\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return False\n"
    )
    invocation = build_capture_invocation(
        "torch.profiler",
        (
            sys.executable,
            "-c",
            "import json, sys; from pathlib import Path; "
            "Path('ran').write_text(json.dumps({'file': __file__, "
            "'argv0': sys.argv[0], 'args': sys.argv[1:]}))",
            "--size",
            "4",
        ),
        tmp_path,
        executable=None,
    )

    completed = subprocess.run(
        invocation.argv,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads((tmp_path / "ran").read_text()) == {
        "file": "<flameox-inline-python>",
        "argv0": "-c",
        "args": ["--size", "4"],
    }
    assert (tmp_path / "torch-trace.json").read_text() == "trace"


def test_torch_capture_rejects_schedule_without_explicit_sdk_steps(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as failure:
        build_capture_invocation(
            "torch.profiler",
            (sys.executable, "benchmark.py"),
            tmp_path,
            executable=None,
            options={
                "mode": "whole_entrypoint",
                "schedule": {"wait": 1, "warmup": 1, "active": 1, "repeat": 1},
            },
        )

    assert failure.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "explicit profile.step() calls" in failure.value.remediation[0]


@pytest.mark.process
def test_torch_capture_launcher_runs_script_with_sibling_imports(tmp_path: Path) -> None:
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "helper.py").write_text("MESSAGE = 'workload ran'\n")
    (tmp_path / "benchmarks" / "benchmark_demo.py").write_text(
        "from helper import MESSAGE\nprint(MESSAGE)\n"
    )
    (tmp_path / "benchmarks" / "torch.py").write_text(
        "from pathlib import Path\n"
        "class ProfilerActivity:\n"
        "    CPU = 'cpu'\n"
        "    CUDA = 'cuda'\n"
        "class _Profile:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *_): return False\n"
        "    def export_chrome_trace(self, path): Path(path).write_text('trace')\n"
        "class profiler:\n"
        "    ProfilerActivity = ProfilerActivity\n"
        "    @staticmethod\n"
        "    def profile(**_): return _Profile()\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return False\n"
    )
    invocation = build_capture_invocation(
        "torch.profiler",
        (sys.executable, "benchmarks/benchmark_demo.py"),
        tmp_path,
        executable=None,
    )

    completed = subprocess.run(
        invocation.argv,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "torch-trace.json").read_text() == "trace"


def test_import_detects_torch_profiler_trace_for_analysis_routing(tmp_path: Path) -> None:
    trace = tmp_path / "torch-trace.json"
    trace.write_text('{"traceEvents":[{"cat":"cpu_op","name":"aten::add"}]}')
    workspace = Workspace.initialize(tmp_path)

    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=trace, kind=ArtifactKind.EXECUTION_TRACE)
    )

    registration = imported.run.artifacts[0]
    assert registration.producer == "torch.profiler"


def test_import_infers_producer_from_staged_artifact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "torch-trace.json"
    trace.write_text('{"traceEvents":[{"cat":"cpu_op","name":"aten::add"}]}')
    workspace = Workspace.initialize(tmp_path)
    service = ImportService(workspace)
    import_path = service.artifacts.import_path

    def import_and_mutate_source(
        source: Path,
        *,
        allowed_roots: tuple[Path, ...],
        max_bytes: int,
    ) -> StoredArtifact:
        stored = import_path(source, allowed_roots=allowed_roots, max_bytes=max_bytes)
        trace.write_text("replacement bytes")
        return stored

    monkeypatch.setattr(service.artifacts, "import_path", import_and_mutate_source)
    imported = service.import_artifact(
        ImportArtifactRequest(path=trace, kind=ArtifactKind.EXECUTION_TRACE)
    )

    registration = imported.run.artifacts[0]
    assert registration.producer == "torch.profiler"


def test_imported_torch_trace_requires_perfetto_extraction_before_analysis(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "torch-trace.json"
    trace.write_text(
        '{"traceEvents":['
        '{"cat":"cpu_op","name":"aten::add","ph":"X","ts":0,"dur":5},'
        '{"cat":"kernel","name":"void aten_add_kernel","ph":"X","ts":10,"dur":2}'
        "]}"
    )
    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=trace, kind=ArtifactKind.EXECUTION_TRACE)
    )

    with pytest.raises(DomainError) as unavailable:
        RecipeService(workspace).pytorch(imported.run.run_id)

    assert unavailable.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert unavailable.value.details == {
        "next_tool": "extract_perfetto",
        "run_id": imported.run.run_id,
    }
    assert "extract_perfetto" in unavailable.value.remediation[0]


@pytest.mark.anyio
@pytest.mark.process
async def test_pyperf_capture_preserves_native_worker_hierarchy(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "scan.py").write_text(
        "values = list(range(2000))\n"
        "total = 0\n"
        "for value in reversed(values):\n"
        "    total += value\n"
        "assert total > 0\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.scan]
argv = ["python", "scan.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="scan",
        adapter="pyperf",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    captured = await service.execute(plan.plan_id)
    extracted = PyPerfExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.SUCCEEDED
    separator = captured.plan.collector_argv.index("--")
    assert Path(captured.plan.collector_argv[separator + 1]).name.startswith("python")
    assert captured.plan.collector_argv[separator + 2 :] == ("scan.py",)
    assert extracted.measurement_count == 9
    assert extracted.warmup_count >= 3
    primary = next(artifact for artifact in captured.run.artifacts if artifact.role == "primary")
    assert captured.plan.adapter_version is not None
    assert primary.producer == "pyperf"
    assert primary.producer_version == captured.plan.adapter_version
    assert extracted.limitations == ()
