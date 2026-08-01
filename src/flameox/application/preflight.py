from __future__ import annotations

import shutil
import tempfile
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
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.storage import Workspace


class PreflightService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        capabilities: CapabilityService | None = None,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.capabilities = capabilities or CapabilityService(workspace)
        self.broker = broker or SubprocessBroker()

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
            required = name not in requirements.optional
            if name == "nvcc":
                results.append(await self._cuda_toolkit(required=required, mode=mode))
            else:
                results.append(self._executable(name, required=required))
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

    async def _cuda_toolkit(
        self,
        *,
        required: bool,
        mode: Literal["passive", "active"],
    ) -> RequirementResult:
        resolved = shutil.which("nvcc")
        if resolved is None:
            return self._executable("nvcc", required=required)
        path = Path(resolved).resolve()
        try:
            path.relative_to(self.workspace.project_root.resolve())
        except ValueError:
            pass
        else:
            return RequirementResult(
                requirement="nvcc",
                kind="executable",
                required=required,
                probe_kind="active" if mode == "active" else "passive",
                status="unsupported",
                evidence=(str(path),),
                limitations=(
                    "Repository-controlled nvcc is not used for CUDA toolkit readiness checks.",
                ),
            )
        if mode == "passive":
            return RequirementResult(
                requirement="nvcc",
                kind="executable",
                required=required,
                probe_kind="active",
                status="unknown",
                identity=str(path),
                evidence=(str(path),),
                limitations=(
                    "nvcc is installed, but CUDA headers and host/device compilation were not "
                    "checked in passive preflight mode.",
                ),
                remediation=(
                    "Re-plan with preflight_mode='active' to run the bounded CUDA toolkit probe.",
                ),
            )

        try:
            with tempfile.TemporaryDirectory(
                dir=self.workspace.paths.staging,
                prefix="cuda-preflight-",
            ) as temporary:
                root = Path(temporary)
                source = root / "header_probe.cu"
                output = root / "header_probe.o"
                source.write_text(
                    "#include <cuda_runtime.h>\n"
                    "__global__ void flameox_probe_kernel() {}\n"
                    "int main() { return 0; }\n",
                    encoding="ascii",
                )
                outcome = await self.broker.run(
                    ExecutionRequest(
                        argv=(
                            str(path),
                            "-x",
                            "cu",
                            "-c",
                            str(source),
                            "-o",
                            str(output),
                        ),
                        cwd=self.workspace.project_root,
                        environment_allowlist=("PATH",),
                        allowed_working_roots=(self.workspace.project_root, root),
                        timeout_seconds=30,
                        max_output_bytes=64 * 1024,
                    )
                )
                diagnostic = self._bounded_diagnostic(
                    (outcome.stdout + b"\n" + outcome.stderr).decode(
                        "utf-8",
                        errors="replace",
                    )
                )
                if outcome.process.exit_code == 0 and output.is_file() and output.stat().st_size:
                    return RequirementResult(
                        requirement="nvcc",
                        kind="executable",
                        required=required,
                        probe_kind="active",
                        status="available",
                        identity=str(path),
                        evidence=(str(path), "bounded_cuda_header_compile"),
                    )
                return self._cuda_compile_failure(
                    required=required,
                    path=path,
                    diagnostic=diagnostic,
                )
        except DomainError as error:
            process = error.details.get("process")
            diagnostic = error.message
            if isinstance(process, dict):
                diagnostic = (
                    " ".join(
                        str(value)
                        for value in (process.get("stdout"), process.get("stderr"))
                        if value
                    )
                    or diagnostic
                )
            return self._cuda_compile_failure(
                required=required,
                path=path,
                diagnostic=self._bounded_diagnostic(diagnostic),
            )
        except (OSError, ValueError) as error:
            return self._cuda_compile_failure(
                required=required,
                path=path,
                diagnostic=self._bounded_diagnostic(f"{type(error).__name__}: {error}"),
            )

    @classmethod
    def _cuda_compile_failure(
        cls,
        *,
        required: bool,
        path: Path,
        diagnostic: str,
    ) -> RequirementResult:
        lowered = diagnostic.casefold()
        permission_denied = any(
            marker in lowered
            for marker in ("permission denied", "operation not permitted", "access denied")
        )
        missing_header = "cuda_runtime.h" in lowered and any(
            marker in lowered for marker in ("no such file", "not found", "cannot open")
        )
        if permission_denied:
            status: Literal["permission_denied", "environment_blocked"] = "permission_denied"
            limitation = "The bounded CUDA toolkit compile was denied by the host environment."
            remediation = (
                "Grant the configured process permission to invoke nvcc and access the CUDA "
                "toolkit, then refresh preflight.",
            )
        else:
            status = "environment_blocked"
            limitation = (
                "The CUDA toolkit is environment-blocked: the bounded nvcc compile did not "
                "produce an object file."
            )
            remediation = (
                "Install the CUDA development toolkit, including cuda_runtime.h, and ensure "
                "nvcc can find its include roots, then refresh preflight.",
            )
            if missing_header:
                limitation = (
                    "The CUDA toolkit is environment-blocked: cuda_runtime.h is missing from "
                    "nvcc's include path."
                )
        return RequirementResult(
            requirement="nvcc",
            kind="executable",
            required=required,
            probe_kind="active",
            status=status,
            identity=str(path),
            evidence=(str(path), diagnostic),
            limitations=(limitation,),
            remediation=remediation,
        )

    @staticmethod
    def _bounded_diagnostic(value: str) -> str:
        return " ".join(value.split())[:500] or "nvcc returned no diagnostic output."

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
