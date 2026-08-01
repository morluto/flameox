from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Literal

import portalocker
from packaging.requirements import InvalidRequirement, Requirement

from flameox.application.preflight import PreflightService
from flameox.application.workloads import WorkloadService
from flameox.domain import DomainError, ErrorCode, PreflightReport
from flameox.models import ContractModel
from flameox.storage import Workspace


class WorkloadDependencySetupResult(ContractModel):
    schema_version: int = 1
    workload_name: str
    requested: tuple[str, ...]
    installed: tuple[str, ...]
    already_available: tuple[str, ...]
    preflight: PreflightReport
    status: Literal["ready", "blocked", "exploratory"]
    next_tool: Literal["plan_capture", "list_capabilities", "prepare_workload_dependencies"] | None
    workload_executed: bool = False


class WorkloadDependencyService:
    """Install only Python distributions declared by one named workload."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)

    async def prepare(self, workload_name: str) -> WorkloadDependencySetupResult:
        config = self.workloads.load().workloads.get(workload_name)
        if config is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unknown workload {workload_name!r}.",
                details={"next_tool": "list_declared_workflows"},
                remediation=(
                    "Call list_declared_workflows with kind='workload' and choose a declared name.",
                ),
            )
        requirements = tuple(
            self._validated_requirement(item)
            for item in config.requirements.python_distributions
        )
        names = tuple(str(item) for item in requirements)
        missing = tuple(item for item in requirements if not self._is_available(item))
        already_available = tuple(
            item for item in names if item not in {str(value) for value in missing}
        )
        if missing:
            uv = _uv_executable()
            if uv is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "uv is required to prepare declared workload Python distributions.",
                    details={
                        "next_tool": "prepare_workload_dependencies",
                        "workload_name": workload_name,
                        "missing_python_distributions": list(map(str, missing)),
                    },
                    remediation=(
                        "Install uv or make it available on PATH, then retry "
                        "prepare_workload_dependencies.",
                    ),
                )
            lock_path = Path(sys.executable).parent / ".flameox-workload-dependencies.lock"
            try:
                with portalocker.Lock(lock_path, mode="a", timeout=30):
                    completed = subprocess.run(
                        [
                            uv,
                            "pip",
                            "install",
                            "--python",
                            sys.executable,
                            *(str(item) for item in missing),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=1_800,
                        env={**os.environ, "UV_NO_PROGRESS": "1"},
                    )
            except (
                OSError,
                subprocess.SubprocessError,
                portalocker.exceptions.LockException,
            ) as exc:
                raise DomainError(
                    ErrorCode.PROCESS_FAILED,
                    "FlameOx could not install the declared workload distributions.",
                    retryable=True,
                    details={
                        "next_tool": "prepare_workload_dependencies",
                        "workload_name": workload_name,
                    },
                    remediation=(
                        "Retry prepare_workload_dependencies after checking uv and package-index "
                        "access.",
                    ),
                ) from exc
            if completed.returncode != 0:
                detail = (completed.stderr.strip() or completed.stdout.strip())[:500]
                raise DomainError(
                    ErrorCode.PROCESS_FAILED,
                    "FlameOx could not install the declared workload distributions.",
                    retryable=True,
                    details={
                        "next_tool": "prepare_workload_dependencies",
                        "workload_name": workload_name,
                        "error": detail,
                    },
                    remediation=(
                        "Retry prepare_workload_dependencies after checking the bounded installer "
                        "diagnostic.",
                    ),
                )

        preflight = await PreflightService(self.workspace).inspect(workload_name, mode="active")
        installed = tuple(item for item in names if item not in already_available)
        missing_after = tuple(
            item.requirement
            for item in preflight.requirements
            if item.kind == "python_distribution" and item.status != "available"
        )
        next_tool: Literal[
            "plan_capture", "list_capabilities", "prepare_workload_dependencies"
        ] | None
        if missing_after:
            next_tool = "prepare_workload_dependencies"
        elif preflight.disposition == "ready" or preflight.disposition == "exploratory":
            next_tool = "plan_capture"
        else:
            next_tool = "list_capabilities"
        return WorkloadDependencySetupResult(
            workload_name=workload_name,
            requested=names,
            installed=installed,
            already_available=already_available,
            preflight=preflight,
            status=preflight.disposition,
            next_tool=next_tool,
        )

    @staticmethod
    def _validated_requirement(value: str) -> Requirement:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Declared Python distribution requirement is invalid: {value!r}.",
                details={"requirement": value, "next_tool": "get_declared_workflow"},
                remediation=(
                    "Use a package name with an optional version specifier; direct URLs and "
                    "local paths are not workload dependency declarations.",
                ),
            ) from exc
        if requirement.url is not None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Declared Python distribution must come from a package index: {value!r}.",
                details={"requirement": value, "next_tool": "get_declared_workflow"},
                remediation=(
                    "Declare a distribution name and version range instead of a direct URL or "
                    "local path.",
                ),
            )
        return requirement

    @staticmethod
    def _is_available(requirement: Requirement) -> bool:
        try:
            installed = distribution(requirement.name)
        except PackageNotFoundError:
            return False
        return requirement.specifier.contains(installed.version, prereleases=True)


def _uv_executable() -> str | None:
    for candidate in ("uv", "uv.exe"):
        path = next(
            (
                item
                for item in os.environ.get("PATH", "").split(os.pathsep)
                if (item_path := Path(item) / candidate).is_file() and os.access(item_path, os.X_OK)
            ),
            None,
        )
        if path is not None:
            return str(Path(path) / candidate)
    return None
