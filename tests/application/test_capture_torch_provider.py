from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.action_graph import ToolAction
from flameox.application.capabilities import CapabilityService
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.python_environment import PythonEnvironmentObservation
from flameox.domain import (
    CapabilityPermissionStatus,
    CapabilityReport,
    CapabilityStatus,
    CaptureStatus,
    DomainError,
    ExecutionStatus,
    ProbeKind,
)
from flameox.storage import Workspace
from tests.support.capture import disable_containment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.process,
    pytest.mark.serial,
]


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_torch
@pytest.mark.process
async def test_torch_profiler_capture_registers_public_chrome_trace(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    if CapabilityService(workspace).get("torch.profiler").status is not CapabilityStatus.AVAILABLE:
        pytest.skip("PyTorch is not installed.")
    (tmp_path / "torch_workload.py").write_text(
        "import torch\n"
        "left = torch.ones((16, 16))\n"
        "right = torch.ones((16, 16))\n"
        "print(torch.mm(left, right).sum().item())\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.torch]
argv = ["python", "torch_workload.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="torch",
        adapter="torch.profiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    trace = next(
        registration
        for registration in result.run.artifacts
        if registration.kind.value == "execution_trace"
    )
    assert trace.display_name == "torch-trace.json"


@pytest.mark.anyio
@pytest.mark.process
async def test_torch_profiler_sdk_registers_every_scheduled_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "torch.py").write_text(
        "from contextlib import nullcontext\n"
        "from pathlib import Path\n"
        "class ProfilerActivity:\n"
        "    CPU = 'cpu'\n"
        "    CUDA = 'cuda'\n"
        "class _Profile:\n"
        "    def __init__(self, schedule=None, on_trace_ready=None, **_):\n"
        "        self.schedule = schedule\n"
        "        self.on_trace_ready = on_trace_ready\n"
        "        self.index = 0\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *_): return False\n"
        "    def step(self):\n"
        "        self.index += 1\n"
        "        if self.schedule is None: return\n"
        "        width = self.schedule['wait'] + self.schedule['warmup'] + "
        "self.schedule['active']\n"
        "        relative = self.index - self.schedule['skip_first']\n"
        "        if relative > 0 and relative % width == 0:\n"
        "            self.on_trace_ready(self)\n"
        "    def export_chrome_trace(self, path):\n"
        "        Path(path).write_text('{\\\"traceEvents\\\": []}')\n"
        "class profiler:\n"
        "    ProfilerActivity = ProfilerActivity\n"
        "    @staticmethod\n"
        "    def schedule(**values): return values\n"
        "    @staticmethod\n"
        "    def profile(**values): return _Profile(**values)\n"
        "    @staticmethod\n"
        "    def record_function(_): return nullcontext()\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return False\n"
    )
    (tmp_path / "scheduled_workload.py").write_text(
        "from flameox.sdk import torch_profiler\n"
        "with torch_profiler() as session:\n"
        "    for index in range(6):\n"
        "        with session.phase('warmup' if index < 2 else 'decode'):\n"
        "            pass\n"
        "        session.step()\n"
    )
    (tmp_path / "invalid_export.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['FLAMEOX_TORCH_PROFILER_OUTPUT_ROOT'])\n"
        "root.joinpath('torch-trace-cycle-0000.json').write_text('{}')\n"
        "root.joinpath('torch-trace-cycle-0001.json').write_text('{}')\n"
        "root.joinpath('torch-trace-cycle-0002.json').write_text('{}')\n"
    )
    (tmp_path / "missing_steps.py").write_text(
        "from flameox.sdk import torch_profiler\nwith torch_profiler():\n    pass\n"
    )
    (tmp_path / "timeout_workload.py").write_text(
        "import time\n"
        "from flameox.sdk import torch_profiler\n"
        "with torch_profiler():\n"
        "    time.sleep(10)\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.scheduled]
argv = ["python", "scheduled_workload.py"]
cwd = "."
timeout_seconds = 30
[workloads.invalid]
argv = ["python", "invalid_export.py"]
cwd = "."
timeout_seconds = 30
[workloads.missing_steps]
argv = ["python", "missing_steps.py"]
cwd = "."
timeout_seconds = 30
[workloads.timeout]
argv = ["python", "timeout_workload.py"]
cwd = "."
timeout_seconds = 0.1
"""
    )
    disable_containment(workspace)
    report = CapabilityReport(
        adapter="torch.profiler",
        status=CapabilityStatus.AVAILABLE,
        version="fixture",
        supported_modes=("whole_entrypoint", "sdk"),
        supported_formats=("chrome-trace",),
        permission_status=CapabilityPermissionStatus.GRANTED,
        probe_kind=ProbeKind.PASSIVE,
    )
    service = CaptureService(workspace)
    monkeypatch.setattr(service.capabilities, "get", lambda _adapter: report)

    async def inspect_torch_distribution(
        *_args: object,
        **_kwargs: object,
    ) -> PythonEnvironmentObservation:
        return PythonEnvironmentObservation(
            interpreter=Path("python"),
            interpreter_sha256="sha256:" + "0" * 64,
            versions={"torch": "2.7.0"},
        )

    monkeypatch.setattr(
        "flameox.application.capture.PythonEnvironmentProbe.inspect",
        inspect_torch_distribution,
    )
    plan = await service.plan(
        workload_name="scheduled",
        adapter="torch.profiler",
        adapter_options={
            "mode": "sdk",
            "activities": ["cpu"],
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
            "schedule": {
                "wait": 1,
                "warmup": 1,
                "active": 1,
                "repeat": 2,
                "skip_first": 0,
            },
        },
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert plan.adapter_options["mode"] == "sdk"
    schedule = plan.adapter_options["schedule"]
    assert isinstance(schedule, dict)
    assert schedule["repeat"] == 2
    assert "Selected activities: cpu." in plan.expected_overhead
    assert "High-cardinality options: none." in plan.expected_overhead
    assert plan.semantics.configuration["activities"] == ["cpu"]
    assert plan.semantics.configuration["record_shapes"] is False
    sdk_config = json.loads(plan.collector_environment["FLAMEOX_TORCH_PROFILER_CONFIG"])
    assert "expected_cycles" not in sdk_config
    traces = [
        registration
        for registration in result.run.artifacts
        if registration.kind.value == "execution_trace"
    ]
    assert [item.role for item in traces] == ["cycle_0000", "cycle_0001"]
    assert [item.display_name for item in traces] == [
        "torch-trace-cycle-0000.json",
        "torch-trace-cycle-0001.json",
    ]
    assert all(item.role != "torch_profiler_cycle_manifest" for item in result.run.artifacts)

    invalid_plan = await service.plan(
        workload_name="invalid",
        adapter="torch.profiler",
        adapter_options=plan.adapter_options,
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    invalid = await service.execute(invalid_plan.plan_token)
    assert invalid.run.execution_status is ExecutionStatus.SUCCEEDED
    assert invalid.run.capture_status is CaptureStatus.FAILED
    assert all(item.kind.value != "execution_trace" for item in invalid.run.artifacts)

    missing_steps_plan = await service.plan(
        workload_name="missing_steps",
        adapter="torch.profiler",
        adapter_options=plan.adapter_options,
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    missing_steps = await service.execute(missing_steps_plan.plan_token)
    assert missing_steps.run.execution_status is ExecutionStatus.FAILED
    assert missing_steps.run.capture_status is CaptureStatus.FAILED
    assert all(item.kind.value != "execution_trace" for item in missing_steps.run.artifacts)

    timeout_plan = await service.plan(
        workload_name="timeout",
        adapter="torch.profiler",
        adapter_options={
            "mode": "sdk",
            "activities": ["cpu"],
            "record_shapes": True,
            "schedule": {"wait": 0, "warmup": 0, "active": 1, "repeat": 1},
        },
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(DomainError) as timed_out:
        await service.execute(timeout_plan.plan_token)

    error = timed_out.value
    recovery = error.next_action
    assert isinstance(recovery, ToolAction)
    retry_options = recovery.arguments["torch_profiler_options"]
    assert retry_options == {
        "mode": "sdk",
        "activities": ["cpu"],
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "with_flops": False,
        "with_modules": False,
        "schedule": {"wait": 0, "warmup": 0, "active": 1, "repeat": 1, "skip_first": 0},
    }
    assert any("explicit step boundaries" in item for item in error.remediation)
