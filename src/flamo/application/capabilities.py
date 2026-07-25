from __future__ import annotations

import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from flamo.domain import CapabilityReport, CapabilityStatus
from flamo.storage import Workspace


class CapabilityList(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    capabilities: tuple[CapabilityReport, ...]


class CapabilityService:
    """Passive capability discovery; it never executes a discovered binary."""

    _EXECUTABLES: ClassVar[dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]]] = {
        "py-spy": (
            "py-spy",
            ("record", "attach", "chrome"),
            ("Install py-spy and grant ptrace access for attach mode.",),
        ),
        "perf": (
            "perf",
            ("record", "stat", "sched"),
            ("Install Linux perf matching the running kernel.",),
        ),
        "perfetto": (
            "trace_processor_shell",
            ("import", "query"),
            ("Install Perfetto Trace Processor or configure its local path.",),
        ),
    }
    _PACKAGES: ClassVar[dict[str, tuple[str, tuple[str, ...]]]] = {
        "pyperf": ("pyperf", ("import", "benchmark")),
        "torch.profiler": ("torch", ("trace_import", "launcher", "sdk")),
        "memray": ("memray", ("import", "run")),
        "coverage": ("coverage", ("import", "run")),
    }

    def __init__(self, workspace: Workspace | None = None) -> None:
        self.workspace = workspace

    def list(self) -> CapabilityList:
        reports: list[CapabilityReport] = [
            CapabilityReport(
                adapter="command",
                status=CapabilityStatus.AVAILABLE,
                supported_modes=("named_workload",),
                limitations=("Collects process output but no profiler evidence.",),
            )
        ]
        for adapter, (executable, modes, remediation) in self._EXECUTABLES.items():
            resolved = self._resolved_executable(adapter, executable)
            reports.append(
                CapabilityReport(
                    adapter=adapter,
                    status=(
                        CapabilityStatus.AVAILABLE
                        if resolved is not None
                        else CapabilityStatus.UNAVAILABLE
                    ),
                    executable=str(Path(resolved).resolve()) if resolved else None,
                    supported_modes=modes if resolved else (),
                    permissions=("ptrace",) if adapter == "py-spy" else (),
                    remediation=() if resolved else remediation,
                    limitations=(("Version not probed in passive mode.",) if resolved else ()),
                )
            )
        bwrap = shutil.which("bwrap")
        reports.append(
            CapabilityReport(
                adapter="containment.bubblewrap",
                status=(
                    CapabilityStatus.AVAILABLE
                    if bwrap is not None
                    else CapabilityStatus.UNAVAILABLE
                ),
                executable=str(Path(bwrap).absolute()) if bwrap else None,
                supported_modes=("pid_namespace", "network_namespace") if bwrap else (),
                remediation=() if bwrap else ("Install bubblewrap for contained MCP execution.",),
            )
        )
        for adapter, (package, modes) in self._PACKAGES.items():
            try:
                package_version = version(package)
            except PackageNotFoundError:
                package_version = None
            reports.append(
                CapabilityReport(
                    adapter=adapter,
                    status=(
                        CapabilityStatus.AVAILABLE
                        if package_version is not None
                        else CapabilityStatus.UNAVAILABLE
                    ),
                    version=package_version,
                    supported_modes=modes if package_version else (),
                    remediation=(
                        ()
                        if package_version
                        else (f"Install Flamo's optional dependency for {adapter}.",)
                    ),
                )
            )
        return CapabilityList(capabilities=tuple(reports))

    def get(self, adapter: str) -> CapabilityReport:
        for report in self.list().capabilities:
            if report.adapter == adapter:
                return report
        return CapabilityReport(
            adapter=adapter,
            status=CapabilityStatus.UNAVAILABLE,
            remediation=("Choose one of Flamo's registered adapters.",),
        )

    def _resolved_executable(self, adapter: str, executable: str) -> str | None:
        if (
            adapter == "perfetto"
            and self.workspace is not None
            and self.workspace.config.analysis.trace_processor_path is not None
        ):
            configured = Path(self.workspace.config.analysis.trace_processor_path)
            candidate = (
                configured if configured.is_absolute() else self.workspace.project_root / configured
            )
            if candidate.is_file():
                return str(candidate.absolute())
        return shutil.which(executable)
