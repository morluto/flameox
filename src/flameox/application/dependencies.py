from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Literal

import portalocker
from packaging.requirements import InvalidRequirement, Requirement
from pydantic import ConfigDict, computed_field, model_validator

from flameox.application.preflight import PreflightService
from flameox.application.workloads import WorkloadService
from flameox.domain import (
    DomainError,
    ErrorCode,
    PreflightDisposition,
    PreflightReport,
    ProbeKind,
    RequirementKind,
    RequirementStatus,
)
from flameox.execution import (
    INSTALLER_ENVIRONMENT_ALLOWLIST,
    ExecutionRequest,
    SubprocessBroker,
)
from flameox.models import ContractModel
from flameox.storage import Workspace


class WorkloadDependencySetupResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: int = 1
    workload_name: str
    requested: tuple[str, ...]
    already_available: tuple[str, ...]
    preflight: PreflightReport
    workload_executed: Literal[False] = False

    @model_validator(mode="after")
    def availability_is_a_partition(self) -> WorkloadDependencySetupResult:
        if len(set(self.requested)) != len(self.requested):
            raise ValueError("requested requirements must be unique")
        available = set(self.already_available)
        if tuple(item for item in self.requested if item in available) != self.already_available:
            raise ValueError("already-available requirements must be an ordered requested subset")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def installed(self) -> tuple[str, ...]:
        available = set(self.already_available)
        return tuple(item for item in self.requested if item not in available)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> PreflightDisposition:
        return self.preflight.disposition

    @computed_field  # type: ignore[prop-decorator]
    @property
    def next_tool(
        self,
    ) -> Literal["plan_capture", "list_capabilities", "prepare_workload_dependencies"]:
        return _dependency_next_tool(self.preflight)


def _dependency_next_tool(
    preflight: PreflightReport,
) -> Literal["plan_capture", "list_capabilities", "prepare_workload_dependencies"]:
    if any(
        item.kind is RequirementKind.PYTHON_DISTRIBUTION
        and item.status is not RequirementStatus.AVAILABLE
        for item in preflight.requirements
    ):
        return "prepare_workload_dependencies"
    if preflight.disposition in {
        PreflightDisposition.READY,
        PreflightDisposition.EXPLORATORY,
    }:
        return "plan_capture"
    return "list_capabilities"


class WorkloadDependencyService:
    """Install only Python distributions declared by one named workload."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.broker = broker or SubprocessBroker()

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
            self._validated_requirement(item) for item in config.requirements.python_distributions
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
                    completed = await self._run_install(
                        [
                            uv,
                            "pip",
                            "install",
                            "--python",
                            sys.executable,
                            *(str(item) for item in missing),
                        ]
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
            except DomainError as exc:
                details = {
                    **exc.details,
                    "next_tool": "prepare_workload_dependencies",
                    "workload_name": workload_name,
                }
                remediation = tuple(
                    dict.fromkeys(
                        (
                            *exc.remediation,
                            "Retry prepare_workload_dependencies after checking uv and "
                            "package-index access.",
                        )
                    )
                )
                raise DomainError(
                    exc.code,
                    "FlameOx could not install the declared workload distributions.",
                    retryable=exc.retryable,
                    details=details,
                    remediation=remediation,
                    run_id=exc.run_id,
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

        preflight = await PreflightService(self.workspace).inspect(
            workload_name,
            mode=ProbeKind.ACTIVE,
        )
        return WorkloadDependencySetupResult(
            workload_name=workload_name,
            requested=names,
            already_available=already_available,
            preflight=preflight,
        )

    async def _run_install(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        """Run installation through the shared bounded execution policy."""
        outcome = await self.broker.run(
            ExecutionRequest(
                argv=tuple(command),
                cwd=self.workspace.project_root,
                allowed_working_roots=(self.workspace.project_root,),
                environment_allowlist=INSTALLER_ENVIRONMENT_ALLOWLIST,
                environment_overrides={"UV_NO_PROGRESS": "1"},
                timeout_seconds=1_800,
            )
        )
        return subprocess.CompletedProcess(
            command,
            outcome.process.exit_code if outcome.process.exit_code is not None else -1,
            outcome.stdout.decode(errors="replace"),
            outcome.stderr.decode(errors="replace"),
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
