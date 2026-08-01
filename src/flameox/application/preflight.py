from __future__ import annotations

import shutil
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Literal

from packaging.requirements import InvalidRequirement, Requirement

from flameox.application.capabilities import CapabilityService
from flameox.application.workloads import WorkloadService
from flameox.domain import (
    CapabilityReport,
    CapabilityStatus,
    DomainError,
    PreflightReport,
    RequirementResult,
    digest_model,
)
from flameox.storage import Workspace


class PreflightService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        capabilities: CapabilityService | None = None,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.capabilities = capabilities or CapabilityService(workspace)

    async def inspect(
        self,
        workload_name: str,
        *,
        mode: Literal["passive", "active"],
    ) -> PreflightReport:
        config = self.workloads.load().workloads[workload_name]
        requirements = config.requirements
        results: list[RequirementResult] = []
        for name in requirements.executables:
            results.append(self._executable(name, required=name not in requirements.optional))
        for name in requirements.python_distributions:
            results.append(self._distribution(name, required=name not in requirements.optional))
        passive = {item.adapter: item for item in self.capabilities.list().capabilities}
        for name in requirements.capabilities:
            active = name in requirements.active
            report = passive.get(name)
            if active and mode == "active":
                try:
                    report = await self.capabilities.probe(name, refresh=True)
                except DomainError as error:
                    results.append(
                        RequirementResult(
                            requirement=name,
                            kind="capability",
                            required=name not in requirements.optional,
                            probe_kind="active",
                            status="probe_failed",
                            limitations=(error.message,),
                            remediation=error.remediation,
                        )
                    )
                    continue
            if active and mode == "passive":
                results.append(
                    RequirementResult(
                        requirement=name,
                        kind="capability",
                        required=name not in requirements.optional,
                        probe_kind="active",
                        status="unknown",
                        limitations=("Active probe was not requested for this planning call.",),
                        remediation=("Re-plan with preflight_mode='active' to request the probe.",),
                    )
                )
                continue
            results.append(
                self._capability(
                    name,
                    required=name not in requirements.optional,
                    active=active,
                    report=report,
                )
            )
        blocked = any(item.required and item.status != "available" for item in results)
        disposition: Literal["ready", "blocked", "exploratory"]
        if blocked and requirements.allow_exploratory:
            disposition = "exploratory"
        elif blocked:
            disposition = "blocked"
        else:
            disposition = "ready"
        limitations = tuple(
            f"{item.requirement}: {item.status}" for item in results if item.status != "available"
        )
        content = {
            "mode": mode,
            "requirements": [item.model_dump(mode="json") for item in results],
            "disposition": disposition,
        }
        return PreflightReport(
            preflight_id=digest_model(content),
            mode=mode,
            disposition=disposition,
            requirements=tuple(results),
            limitations=limitations,
        )

    def _executable(self, name: str, *, required: bool) -> RequirementResult:
        resolved = shutil.which(name)
        if resolved is None:
            return RequirementResult(
                requirement=name,
                kind="executable",
                required=required,
                probe_kind="passive",
                status="absent",
                remediation=(
                    f"FlameOx cannot install host executable {name!r}; install it in the local "
                    "environment or configure a workload that uses an available executable.",
                ),
            )
        path = Path(resolved).resolve()
        try:
            path.relative_to(self.workspace.project_root.resolve())
        except ValueError:
            pass
        else:
            return RequirementResult(
                requirement=name,
                kind="executable",
                required=required,
                probe_kind="passive",
                status="unsupported",
                evidence=(str(path),),
                limitations=("Repository-controlled executables are not probed during preflight.",),
            )
        return RequirementResult(
            requirement=name,
            kind="executable",
            required=required,
            probe_kind="passive",
            status="available",
            identity=str(path),
            evidence=(str(path),),
        )

    def _distribution(self, name: str, *, required: bool) -> RequirementResult:
        try:
            requirement = Requirement(name)
        except InvalidRequirement:
            return RequirementResult(
                requirement=name,
                kind="python_distribution",
                required=required,
                probe_kind="passive",
                status="unsupported",
                remediation=(
                    "Use a package name with an optional version specifier in the workload "
                    "requirements.",
                ),
            )
        try:
            value = distribution(requirement.name)
        except PackageNotFoundError:
            return RequirementResult(
                requirement=name,
                kind="python_distribution",
                required=required,
                probe_kind="passive",
                status="absent",
                remediation=(
                    f"Call prepare_workload_dependencies for workload dependencies including "
                    f"{name!r}.",
                ),
                next_tool="prepare_workload_dependencies",
            )
        identity = f"{value.metadata['Name']}=={value.version}"
        if not requirement.specifier.contains(value.version, prereleases=True):
            return RequirementResult(
                requirement=name,
                kind="python_distribution",
                required=required,
                probe_kind="passive",
                status="absent",
                identity=identity,
                evidence=(identity,),
                remediation=(
                    f"Call prepare_workload_dependencies to install a version matching {name!r}.",
                ),
                next_tool="prepare_workload_dependencies",
            )
        return RequirementResult(
            requirement=name,
            kind="python_distribution",
            required=required,
            probe_kind="passive",
            status="available",
            identity=identity,
            evidence=(identity,),
            limitations=("Distribution metadata does not prove which module will load.",),
        )

    def _capability(
        self,
        name: str,
        *,
        required: bool,
        active: bool,
        report: object,
    ) -> RequirementResult:
        if not isinstance(report, CapabilityReport):
            return RequirementResult(
                requirement=name,
                kind="capability",
                required=required,
                probe_kind="active" if active else "passive",
                status="unknown",
                limitations=("Flameox does not own a probe for this capability.",),
                next_tool="list_capabilities",
            )
        statuses: dict[
            CapabilityStatus,
            Literal[
                "available",
                "absent",
                "permission_denied",
                "unsupported",
                "unknown",
                "probe_failed",
            ],
        ] = {
            CapabilityStatus.AVAILABLE: "available",
            CapabilityStatus.UNAVAILABLE: "absent",
            CapabilityStatus.PERMISSION_REQUIRED: "permission_denied",
            CapabilityStatus.UNSUPPORTED_PLATFORM: "unsupported",
            CapabilityStatus.UNKNOWN: "unknown",
            CapabilityStatus.DEGRADED: "unknown",
        }
        return RequirementResult(
            requirement=name,
            kind="capability",
            required=required,
            probe_kind="active" if active else "passive",
            status=statuses[report.status],
            identity=report.version or report.executable or report.import_location,
            evidence=tuple(
                value
                for value in (report.executable, report.import_location, report.version)
                if value is not None
            ),
            limitations=report.limitations,
            remediation=report.remediation,
            next_tool=(
                "prepare_adapter"
                if getattr(report.setup, "method", None) == "prepare_adapter"
                else (
                    "prepare_capabilities"
                    if report.setup is not None
                    else (
                        "list_capabilities"
                        if report.status is not CapabilityStatus.AVAILABLE
                        else None
                    )
                )
            ),
        )
