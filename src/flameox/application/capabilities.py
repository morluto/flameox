from __future__ import annotations

import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

from flameox.adapters.builtins import BUILTIN_ADAPTERS, BuiltinAdapter, builtin_adapter
from flameox.adapters.registry import AdapterRegistry
from flameox.domain import CapabilityReport, CapabilityStatus, DomainError
from flameox.domain.models import utc_now
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import Workspace


class CapabilityList(ContractModel):
    schema_version: int = 1
    capabilities: tuple[CapabilityReport, ...]


_CONTAINMENT_VERSION_ARGS = {
    "containment.bubblewrap": ("--version",),
    "containment.systemd": ("--version",),
}


class CapabilityService:
    """Passive discovery plus explicit, brokered, process-lifetime active probes."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        *,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()
        self._active_cache: dict[str, CapabilityReport] = {}

    def list(self) -> CapabilityList:
        system = platform.system().lower()
        architecture = platform.machine().lower()
        reports: list[CapabilityReport] = []
        command = BUILTIN_ADAPTERS["command"]
        reports.append(
            CapabilityReport(
                adapter=command.name,
                status=CapabilityStatus.AVAILABLE,
                supported_modes=command.supported_modes,
                supported_formats=command.supported_formats,
                platform=system,
                architecture=architecture,
                limitations=command.capture_limitations,
            )
        )
        executable_adapters = (
            adapter
            for adapter in BUILTIN_ADAPTERS.values()
            if adapter.dependency_kind == "executable"
        )
        for adapter in executable_adapters:
            assert adapter.dependency is not None
            resolved = self._resolved_executable(adapter.name, adapter.dependency)
            supported_platform = (
                adapter.supported_platforms is None or system in adapter.supported_platforms
            )
            reports.append(
                CapabilityReport(
                    adapter=adapter.name,
                    status=(
                        CapabilityStatus.UNSUPPORTED_PLATFORM
                        if not supported_platform
                        else (
                            CapabilityStatus.AVAILABLE
                            if resolved is not None
                            else CapabilityStatus.UNAVAILABLE
                        )
                    ),
                    executable=str(Path(resolved).resolve()) if resolved else None,
                    supported_modes=(
                        adapter.supported_modes if resolved and supported_platform else ()
                    ),
                    supported_formats=(
                        adapter.supported_formats if resolved and supported_platform else ()
                    ),
                    platform=system,
                    architecture=architecture,
                    permissions=adapter.permissions,
                    permission_status=(
                        "unknown_until_active_probe"
                        if adapter.name in {"py-spy", "perf"} and resolved
                        else None
                    ),
                    restrictions=self._platform_restrictions(adapter),
                    features=adapter.features,
                    remediation=(() if resolved or not supported_platform else adapter.remediation),
                    limitations=(("Version not probed in passive mode.",) if resolved else ()),
                )
            )
        reports.extend(self._containment_reports(system, architecture))
        package_adapters = (
            adapter for adapter in BUILTIN_ADAPTERS.values() if adapter.dependency_kind == "package"
        )
        for adapter in package_adapters:
            assert adapter.dependency is not None
            try:
                package_version = version(adapter.dependency)
            except PackageNotFoundError:
                package_version = None
            import_location = (
                self._import_location(adapter.dependency) if package_version is not None else None
            )
            reports.append(
                CapabilityReport(
                    adapter=adapter.name,
                    status=(
                        CapabilityStatus.AVAILABLE
                        if package_version is not None
                        else CapabilityStatus.UNAVAILABLE
                    ),
                    import_location=import_location,
                    version=package_version,
                    supported_modes=adapter.supported_modes if package_version else (),
                    supported_formats=adapter.supported_formats if package_version else (),
                    platform=system,
                    architecture=architecture,
                    features=adapter.features,
                    remediation=(
                        ()
                        if package_version
                        else (f"Install flameox's optional dependency for {adapter.name}.",)
                    ),
                )
            )
        if self.workspace is not None:
            for descriptor in AdapterRegistry(self.workspace).discover().adapters:
                reports.append(
                    CapabilityReport(
                        adapter=descriptor.adapter,
                        status=(
                            CapabilityStatus.UNKNOWN
                            if descriptor.approved
                            else CapabilityStatus.UNAVAILABLE
                        ),
                        version=descriptor.version,
                        platform=system,
                        architecture=architecture,
                        restrictions=(
                            f"distribution={descriptor.distribution}",
                            f"package_identity={descriptor.package_identity}",
                        ),
                        limitations=(
                            ("Approved third-party adapter has not been actively probed.",)
                            if descriptor.approved
                            else ("Third-party adapter is installed but not approved.",)
                        ),
                        remediation=(
                            ()
                            if descriptor.approved
                            else (
                                f"Approve distribution {descriptor.distribution!r} "
                                "through the local CLI.",
                            )
                        ),
                    )
                )
        return CapabilityList(capabilities=tuple(reports))

    async def list_active(self, *, refresh: bool = False) -> CapabilityList:
        passive = self.list()
        reports: list[CapabilityReport] = []
        for report in passive.capabilities:
            adapter = builtin_adapter(report.adapter)
            version_args = (
                adapter.version_args
                if adapter is not None
                else _CONTAINMENT_VERSION_ARGS.get(report.adapter, ())
            )
            if not version_args or report.executable is None:
                reports.append(report)
                continue
            reports.append(await self.probe(report.adapter, refresh=refresh))
        return CapabilityList(capabilities=tuple(reports))

    async def probe(self, adapter: str, *, refresh: bool = False) -> CapabilityReport:
        if not refresh and adapter in self._active_cache:
            return self._active_cache[adapter]
        passive = self.get(adapter)
        if passive.executable is None or passive.status is not CapabilityStatus.AVAILABLE:
            return passive
        definition = builtin_adapter(adapter)
        version_args = (
            definition.version_args
            if definition is not None
            else _CONTAINMENT_VERSION_ARGS.get(adapter, ())
        )
        if not version_args:
            return passive
        cwd = self.workspace.project_root if self.workspace is not None else Path.cwd()
        executable = Path(passive.executable)
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(str(executable), *version_args),
                    cwd=cwd,
                    environment_allowlist=(),
                    allowed_working_roots=(cwd,),
                    timeout_seconds=5,
                    max_output_bytes=32 * 1024,
                )
            )
            output = (outcome.stdout + b"\n" + outcome.stderr).decode(
                "utf-8",
                errors="replace",
            )
            first_line = next((line.strip() for line in output.splitlines() if line.strip()), None)
            succeeded = outcome.process.exit_code == 0
            status = CapabilityStatus.AVAILABLE if succeeded else CapabilityStatus.DEGRADED
            restrictions = passive.restrictions
            permission_status = passive.permission_status
            if adapter == "perf":
                restrictions = (*restrictions, "Sampling permissions were not exercised.")
            if adapter == "py-spy":
                permission_status = "not_exercised"
            result = passive.model_copy(
                update={
                    "status": status,
                    "version": first_line or passive.version,
                    "permission_status": permission_status,
                    "restrictions": restrictions,
                    "limitations": (
                        ()
                        if succeeded
                        else (f"Active version probe exited with {outcome.process.exit_code}.",)
                    ),
                    "probe_kind": "active",
                    "probed_at": utc_now(),
                }
            )
        except (DomainError, OSError, ValueError) as exc:
            result = passive.model_copy(
                update={
                    "status": CapabilityStatus.DEGRADED,
                    "limitations": (f"Active probe failed with {type(exc).__name__}.",),
                    "probe_kind": "active",
                    "probed_at": utc_now(),
                }
            )
        self._active_cache[adapter] = result
        return result

    def get(self, adapter: str) -> CapabilityReport:
        for report in self.list().capabilities:
            if report.adapter == adapter:
                return report
        return CapabilityReport(
            adapter=adapter,
            status=CapabilityStatus.UNAVAILABLE,
            platform=platform.system().lower(),
            architecture=platform.machine().lower(),
            remediation=("Choose one of flameox's registered adapters.",),
        )

    def _containment_reports(
        self,
        system: str,
        architecture: str,
    ) -> tuple[CapabilityReport, ...]:
        reports: list[CapabilityReport] = []
        for name, executable, modes in (
            ("containment.bubblewrap", "bwrap", ("pid_namespace", "network_namespace")),
            ("containment.systemd", "systemd-run", ("cgroup_scope", "resource_limits")),
        ):
            resolved = shutil.which(executable) if system == "linux" else None
            reports.append(
                CapabilityReport(
                    adapter=name,
                    status=(
                        CapabilityStatus.UNSUPPORTED_PLATFORM
                        if system != "linux"
                        else (
                            CapabilityStatus.AVAILABLE
                            if resolved is not None
                            else CapabilityStatus.UNAVAILABLE
                        )
                    ),
                    executable=str(Path(resolved).resolve()) if resolved else None,
                    supported_modes=modes if resolved else (),
                    platform=system,
                    architecture=architecture,
                    remediation=(
                        ()
                        if resolved or system != "linux"
                        else (f"Install {executable} for local process containment.",)
                    ),
                    limitations=(
                        ("User-manager or namespace permission not actively probed.",)
                        if resolved
                        else ()
                    ),
                )
            )
        return tuple(reports)

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

    @staticmethod
    def _import_location(package: str) -> str | None:
        try:
            package_spec = find_spec(package)
        except (ImportError, ValueError):
            return None
        return package_spec.origin if package_spec is not None else None

    @staticmethod
    def _platform_restrictions(adapter: BuiltinAdapter) -> tuple[str, ...]:
        path = adapter.restriction_path
        if path is None:
            return ()
        try:
            value = path.read_text().strip()
        except OSError:
            return (f"{path.name}=unknown",)
        return (f"{path.name}={value}",)
