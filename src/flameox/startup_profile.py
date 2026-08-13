from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PythonStartupProfile:
    """The one qualified pyperf profile for Python startup evidence."""

    profile_id: str
    benchmark_name: str
    process_count: int
    values_per_process: int
    loops_per_value: int
    warmups_per_process: int
    wall_output_name: str
    import_trace_output_name: str

    @property
    def sample_count(self) -> int:
        return self.process_count * self.values_per_process

    def pyperf_argv(
        self,
        *,
        python: str,
        output: Path,
        timeout_seconds: float,
        workload: tuple[str, ...],
    ) -> tuple[str, ...]:
        return (
            python,
            "-m",
            "pyperf",
            "command",
            "--output",
            str(output),
            "--processes",
            str(self.process_count),
            "--values",
            str(self.values_per_process),
            "--loops",
            str(self.loops_per_value),
            "--warmups",
            str(self.warmups_per_process),
            "--timeout",
            # pyperf 2.10 accepts integer seconds here. The canonical outer broker
            # retains the exact float deadline; rounding up avoids shortening it.
            str(math.ceil(timeout_seconds)),
            "--copy-env",
            "--name",
            self.benchmark_name,
            "--",
            *workload,
        )


PYTHON_STARTUP_PROFILE = PythonStartupProfile(
    profile_id="flameox.python-startup.pyperf.v1",
    benchmark_name="flameox.python_startup.wall_time",
    process_count=5,
    values_per_process=1,
    loops_per_value=1,
    warmups_per_process=0,
    wall_output_name="startup-wall.pyperf.json",
    import_trace_output_name="python-importtime.log",
)
