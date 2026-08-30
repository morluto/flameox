from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters.benchmark_samples import BenchmarkSamplesExtractor
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.python_environment import PythonEnvironmentObservation
from flameox.domain import ExecutionStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


@pytest.mark.anyio
async def test_torch_benchmark_capture_preserves_timer_samples_as_benchmark_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    torch_package = tmp_path / "torch"
    (torch_package / "utils").mkdir(parents=True)
    (torch_package / "__init__.py").write_text(
        "__version__ = '2.13.0'\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return False\n"
    )
    (torch_package / "utils" / "__init__.py").write_text("")
    (torch_package / "utils" / "benchmark.py").write_text(
        "class _Measurement:\n"
        "    number_per_run = 2\n"
        "    times = (0.000001, 0.000002)\n"
        "class Timer:\n"
        "    def __init__(self, **kwargs): self.kwargs = kwargs\n"
        "    def blocked_autorange(self, *, min_run_time):\n"
        "        self.kwargs['globals']['operation']()\n"
        "        return _Measurement()\n"
    )
    (tmp_path / "benchmark.py").write_text(
        "from flameox.sdk import torch_benchmark\n"
        "prepared = [1, 2, 3]\n"
        "torch_benchmark('gae.step', lambda: sum(prepared), dimensions={'shape': '32x2048'})\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.gae]
argv = ["python", "benchmark.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)

    async def inspect_torch_distribution(
        *_args: object,
        **_kwargs: object,
    ) -> PythonEnvironmentObservation:
        return PythonEnvironmentObservation(
            interpreter=Path("python"),
            interpreter_sha256="sha256:" + "0" * 64,
            versions={"torch": "2.13.0"},
        )

    monkeypatch.setattr(
        "flameox.application.capture.PythonEnvironmentProbe.inspect",
        inspect_torch_distribution,
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="gae",
        adapter="torch.benchmark",
        adapter_options={"min_run_time_seconds": 0.5, "max_samples": 2, "num_threads": 1},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    captured = await service.execute(plan.plan_token)
    extracted = BenchmarkSamplesExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.SUCCEEDED
    assert plan.semantics.scope.bounds == {
        "min_run_time_seconds": 0.5,
        "max_samples": 2,
    }
    artifact = next(
        item for item in captured.run.artifacts if item.producer == "torch.utils.benchmark"
    )
    assert artifact.producer == "torch.utils.benchmark"
    assert artifact.producer_version == "2.13.0"
    assert extracted.producer == "torch.utils.benchmark"
    assert extracted.measurement_count == 2
    assert extracted.warmup_count == 0
