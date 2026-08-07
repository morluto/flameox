from __future__ import annotations

import asyncio
import json
import platform
import re
import shutil
import socket
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import portalocker
from platformdirs import user_data_path

from flameox.adapters.builtins import BUILTIN_ADAPTERS, BuiltinAdapter, builtin_adapter
from flameox.adapters.registry import AdapterRegistry
from flameox.adapters.setup_runtime import install_trace_processor
from flameox.adapters.toxiproxy import ToxiproxyClient, ToxiproxyToolManager, ToxiproxyToolReceipt
from flameox.application.operations import OperationFailure, OperationRunner, OperationStatus
from flameox.atomic import atomic_write_json
from flameox.domain import (
    AdapterSetup,
    CapabilityProvisioning,
    CapabilityReport,
    CapabilitySetup,
    CapabilityStatus,
    DomainError,
    ErrorCode,
)
from flameox.domain.models import utc_now
from flameox.execution import (
    INSTALLER_ENVIRONMENT_ALLOWLIST,
    ExecutionOutcome,
    ExecutionRequest,
    ResourcePolicy,
    SubprocessBroker,
)
from flameox.models import ContractModel
from flameox.storage import Workspace


class CapabilityList(ContractModel):
    schema_version: int = 1
    capabilities: tuple[CapabilityReport, ...]
    setup_adapters: tuple[str, ...] = ()
    setup_third_party_adapters: tuple[str, ...] = ()
    available_setup_adapters: tuple[str, ...] = ()
    available_setup_third_party_adapters: tuple[str, ...] = ()
    recommendation_scope: str | None = None
    latest_setup: CapabilitySetupReceipt | None = None
    next_tool: Literal[
        "start_capability_setup",
        "prepare_adapter",
        "list_capabilities",
    ] | None = None


class SetupVerification(ContractModel):
    """Evidence that a setup action was checked before it returned success."""

    status: Literal["verified", "partial"]
    checked_adapters: tuple[str, ...]
    available_adapters: tuple[str, ...]
    unavailable_adapters: tuple[str, ...] = ()
    method: Literal["capability_scan"] = "capability_scan"


class CapabilitySetupReceipt(ContractModel):
    """Latest durable state for a managed capability setup request."""

    schema_version: int = 1
    requested: tuple[str, ...]
    completed: tuple[str, ...] = ()
    phase: Literal[
        "installing_packages",
        "staging_trace_processor",
        "staging_toxiproxy",
        "completed",
        "failed",
    ]
    error: str | None = None
    updated_at: datetime
    next_tool: Literal["list_capabilities"] = "list_capabilities"


class CapabilitySetupResult(ContractModel):
    schema_version: int = 1
    requested: tuple[str, ...]
    installed: tuple[str, ...]
    already_available: tuple[str, ...]
    next_tool: Literal["list_capabilities"] = "list_capabilities"
    setup_verification: SetupVerification
    workload_executed: bool = False


class AdapterPreparationResult(ContractModel):
    schema_version: int = 1
    adapter: str
    distribution: str
    version: str
    package_identity: str
    approval_provenance: Literal["agent"] = "agent"
    next_tool: Literal["list_capabilities"] = "list_capabilities"
    setup_verification: SetupVerification
    workload_executed: bool = False


_CONTAINMENT_VERSION_ARGS = {
    "containment.bubblewrap": ("--version",),
    "containment.systemd": ("--version",),
}

_CAPABILITY_INSTALL_TIMEOUT_SECONDS = 1_800


class CapabilityService:
    """Passive discovery plus explicit, brokered, process-lifetime active probes."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        *,
        broker: SubprocessBroker | None = None,
        capability_manifest: Path | None = None,
    ) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()
        self._uses_default_workspace_manifest = (
            capability_manifest is None and workspace is not None
        )
        self.capability_manifest = capability_manifest or (
            workspace.paths.records / "capabilities.json"
            if workspace is not None
            else Path(user_data_path("flameox", appauthor=False)) / "capabilities.json"
        )
        self.setup_receipt_path = self.capability_manifest.with_name("capability-setup.json")
        self._active_cache: dict[str, CapabilityReport] = {}

    def list(self) -> CapabilityList:
        return self._list()

    def list_for_adapter(self, adapter: str) -> CapabilityList:
        return self._list(recommendation_adapter=adapter)

    def _list(self, *, recommendation_adapter: str | None = None) -> CapabilityList:
        system = platform.system().lower()
        architecture = platform.machine().lower()
        reports: list[CapabilityReport] = []
        for adapter in BUILTIN_ADAPTERS.values():
            if adapter.dependency_kind != "internal":
                continue
            reports.append(
                CapabilityReport(
                    adapter=adapter.name,
                    status=CapabilityStatus.AVAILABLE,
                    provisioning=CapabilityProvisioning.BUNDLED,
                    supported_modes=adapter.supported_modes,
                    supported_formats=adapter.supported_formats,
                    platform=system,
                    architecture=architecture,
                    features=adapter.features,
                    limitations=adapter.capture_limitations,
                )
            )
        executable_adapters = (
            adapter
            for adapter in BUILTIN_ADAPTERS.values()
            if adapter.dependency_kind == "executable"
        )
        for adapter in executable_adapters:
            if adapter.dependency is None:
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "Builtin adapter is missing its dependency declaration.",
                )
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
                    provisioning=(
                        CapabilityProvisioning.UNSUPPORTED
                        if not supported_platform
                        else (
                            CapabilityProvisioning.MANAGED_RUNTIME
                            if adapter.managed_extra is not None
                            else CapabilityProvisioning.HOST
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
                    remediation=(
                        ()
                        if resolved or not supported_platform
                        else (
                            (
                                "Call start_capability_setup with this adapter to install it into "
                                "FlameOx's managed runtime.",
                            )
                            if adapter.managed_extra is not None
                            else adapter.remediation
                        )
                    ),
                    setup=self._setup(adapter),
                    setup_verification=(
                        "pending"
                        if resolved is None and adapter.managed_extra is not None
                        else ("passive" if resolved else "not_required")
                    ),
                    limitations=(("Version not probed in passive mode.",) if resolved else ()),
                )
            )
        reports.extend(self._containment_reports(system, architecture))
        package_adapters = (
            adapter for adapter in BUILTIN_ADAPTERS.values() if adapter.dependency_kind == "package"
        )
        for adapter in package_adapters:
            if adapter.dependency is None:
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "Builtin adapter is missing its dependency declaration.",
                )
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
                    provisioning=(
                        CapabilityProvisioning.MANAGED_RUNTIME
                        if adapter.managed_extra is not None
                        else CapabilityProvisioning.HOST
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
                        else (
                            (
                                "Call start_capability_setup with this adapter to install it into "
                                "FlameOx's managed runtime.",
                            )
                            if adapter.managed_extra is not None
                            else (f"Install flameox's optional dependency for {adapter.name}.",)
                        )
                    ),
                    setup=self._setup(adapter),
                    setup_verification=(
                        "pending"
                        if package_version is None and adapter.managed_extra is not None
                        else ("passive" if package_version is not None else "not_required")
                    ),
                )
            )
        if self.workspace is not None:
            reports.append(self._toxiproxy_report(system, architecture))
            for descriptor in AdapterRegistry(self.workspace).discover().adapters:
                reports.append(
                    CapabilityReport(
                        adapter=descriptor.adapter,
                        status=(
                            CapabilityStatus.UNKNOWN
                            if descriptor.approved
                            else CapabilityStatus.UNAVAILABLE
                        ),
                        provisioning=CapabilityProvisioning.THIRD_PARTY_APPROVAL,
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
                                "Call prepare_adapter with the reported adapter and "
                                f"distribution {descriptor.distribution!r}.",
                            )
                        ),
                        setup=(
                            None
                            if descriptor.approved
                            else AdapterSetup(
                                method="prepare_adapter",
                                adapter=descriptor.adapter,
                                distribution=descriptor.distribution,
                                package_identity=descriptor.package_identity,
                                next_tool="prepare_adapter",
                            )
                        ),
                        setup_verification=("pending" if not descriptor.approved else "passive"),
                    )
                )
        return self._finish(
            reports,
            latest_setup=self._read_setup_receipt(),
            recommendation_adapter=recommendation_adapter,
        )

    async def list_active(
        self,
        *,
        refresh: bool = False,
        recommendation_adapter: str | None = None,
    ) -> CapabilityList:
        passive = self.list()
        reports: list[CapabilityReport] = []
        for report in passive.capabilities:
            definition = builtin_adapter(report.adapter)
            version_args = (
                definition.version_args
                if definition is not None
                else _CONTAINMENT_VERSION_ARGS.get(report.adapter, ())
            )
            if not version_args or report.executable is None:
                reports.append(report)
                continue
            reports.append(await self.probe(report.adapter, refresh=refresh))
        return self._finish(
            reports,
            latest_setup=passive.latest_setup,
            recommendation_adapter=recommendation_adapter,
        )

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
        version_report = await self._probe_version(passive, version_args)
        if version_report.status is not CapabilityStatus.AVAILABLE:
            self._active_cache[adapter] = version_report
            return version_report
        if adapter == "perf":
            result = await self._probe_perf(passive)
            result = result.model_copy(update={"version": version_report.version})
            self._active_cache[adapter] = result
            return result
        result = version_report
        if adapter == "py-spy":
            result = result.model_copy(update={"permission_status": "not_exercised"})
        self._active_cache[adapter] = result
        return result

    async def _probe_version(
        self,
        passive: CapabilityReport,
        version_args: tuple[str, ...],
    ) -> CapabilityReport:
        if passive.executable is None:
            return passive
        cwd = self.workspace.project_root if self.workspace is not None else Path.cwd()
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(passive.executable, *version_args),
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
            return passive.model_copy(
                update={
                    "status": CapabilityStatus.AVAILABLE
                    if succeeded
                    else CapabilityStatus.DEGRADED,
                    "version": first_line or passive.version,
                    "limitations": (
                        ()
                        if succeeded
                        else (f"Active version probe exited with {outcome.process.exit_code}.",)
                    ),
                    "probe_kind": "active",
                    "setup_verification": "active",
                    "probed_at": utc_now(),
                }
            )
        except (DomainError, OSError, ValueError) as exc:
            return passive.model_copy(
                update={
                    "status": CapabilityStatus.DEGRADED,
                    "limitations": (f"Active version probe failed with {type(exc).__name__}.",),
                    "probe_kind": "active",
                    "setup_verification": "active",
                    "probed_at": utc_now(),
                }
            )

    async def _probe_perf(self, passive: CapabilityReport) -> CapabilityReport:
        """Exercise one bounded recording so availability includes event permissions."""
        if self.workspace is None or passive.executable is None:
            return passive
        staging_root = self.workspace.paths.staging
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                dir=staging_root,
                prefix="capability-perf-",
            ) as temporary:
                output = Path(temporary) / "perf.data"
                request = ExecutionRequest(
                    argv=(
                        passive.executable,
                        "record",
                        "-B",
                        "-N",
                        "--max-size=1M",
                        "-o",
                        str(output),
                        "--",
                        sys.executable,
                        "-I",
                        "-S",
                        "-c",
                        "pass",
                    ),
                    cwd=self.workspace.project_root,
                    environment_allowlist=(),
                    allowed_working_roots=(self.workspace.project_root,),
                    timeout_seconds=5,
                    max_output_bytes=self.workspace.config.execution.max_output_bytes,
                    resource_policy=ResourcePolicy(
                        filesystem_path=self.workspace.paths.root,
                        staging_root=Path(temporary),
                        minimum_free_bytes=self.workspace.config.storage.min_free_bytes,
                        sampling_interval_ms=(
                            self.workspace.config.execution.resource_sampling_interval_ms
                        ),
                        max_observed_files=self.workspace.config.execution.max_resource_observed_files,
                    ),
                )
                try:
                    outcome = await self.broker.run(request)
                except DomainError as error:
                    diagnostic = self._bounded_diagnostic(error.message)
                    process = error.details.get("process")
                    if isinstance(process, dict):
                        diagnostic = self._bounded_diagnostic(
                            " ".join(
                                str(value)
                                for value in (process.get("stdout"), process.get("stderr"))
                                if value
                            )
                            or diagnostic
                        )
                    if self._is_perf_permission_denial(diagnostic):
                        return self._perf_permission_failure(passive, diagnostic)
                    return self._perf_failure(passive, diagnostic)
        except (OSError, ValueError) as error:
            return self._perf_failure(passive, f"Active perf probe failed: {type(error).__name__}.")

        diagnostic = self._bounded_diagnostic(
            (outcome.stdout + b"\n" + outcome.stderr).decode("utf-8", errors="replace")
        )
        if outcome.process.exit_code == 0:
            return passive.model_copy(
                update={
                    "status": CapabilityStatus.AVAILABLE,
                    "permission_status": "granted",
                    "limitations": (),
                    "remediation": (),
                    "probe_kind": "active",
                    "probed_at": utc_now(),
                }
            )
        if self._is_perf_permission_denial(diagnostic):
            return self._perf_permission_failure(passive, diagnostic)
        return self._perf_failure(passive, diagnostic or "perf sampling probe failed.")

    def _perf_permission_failure(
        self,
        passive: CapabilityReport,
        diagnostic: str,
    ) -> CapabilityReport:
        return passive.model_copy(
            update={
                "status": CapabilityStatus.PERMISSION_REQUIRED,
                "permission_status": "denied",
                "limitations": (diagnostic or "perf event access was denied.",),
                "remediation": (self._perf_remediation(diagnostic),),
                "probe_kind": "active",
                "probed_at": utc_now(),
            }
        )

    def _perf_failure(self, passive: CapabilityReport, diagnostic: str) -> CapabilityReport:
        return passive.model_copy(
            update={
                "status": CapabilityStatus.DEGRADED,
                "permission_status": "unknown",
                "limitations": (diagnostic,),
                "remediation": (
                    "Inspect the bounded perf probe diagnostic and refresh capabilities.",
                ),
                "probe_kind": "active",
                "probed_at": utc_now(),
            }
        )

    @staticmethod
    def _is_perf_permission_denial(diagnostic: str) -> bool:
        lowered = diagnostic.casefold()
        return any(
            marker in lowered
            for marker in (
                "permission denied",
                "operation not permitted",
                "no permission",
                "perf_event_open",
                "perf_event_paranoid",
                "access denied",
            )
        )

    @staticmethod
    def _perf_remediation(diagnostic: str) -> str:
        lowered = diagnostic.casefold()
        match = re.search(r"perf_event_paranoid(?: setting is|=)\s*(\d+)", lowered)
        setting = match.group(1) if match is not None else None
        current = f" (observed perf_event_paranoid={setting})" if setting else ""
        return (
            "perf sampling is unusable because the kernel denied perf_event_open"
            f"{current}. Lower kernel.perf_event_paranoid to a policy value that permits "
            "this process, or grant CAP_PERFMON/CAP_SYS_ADMIN according to local policy, "
            "then call list_capabilities(mode='active_refresh') before planning."
        )

    @staticmethod
    def _bounded_diagnostic(value: str) -> str:
        normalized = " ".join(value.split())
        return normalized[:500] or "Active perf probe returned no diagnostic output."

    def get(self, adapter: str) -> CapabilityReport:
        for report in self.list().capabilities:
            if report.adapter == adapter:
                return report
        return CapabilityReport(
            adapter=adapter,
            status=CapabilityStatus.UNAVAILABLE,
            provisioning=CapabilityProvisioning.UNSUPPORTED,
            platform=platform.system().lower(),
            architecture=platform.machine().lower(),
            remediation=("Choose one of flameox's registered adapters.",),
        )

    def prepare(
        self,
        adapters: tuple[str, ...],
        *,
        cancel_event: threading.Event | None = None,
        phase_callback: Callable[[str], None] | None = None,
    ) -> CapabilitySetupResult:
        """Install only declared FlameOx-managed providers into this runtime."""
        reports = {item.adapter: item for item in self.list().capabilities}
        requested = tuple(dict.fromkeys(adapters))
        unsupported = tuple(
            adapter
            for adapter in requested
            if (
                adapter not in reports
                or reports[adapter].setup is None
                or reports[adapter].status is CapabilityStatus.UNSUPPORTED_PLATFORM
            )
        )
        if unsupported:
            unsupported_platform = tuple(
                adapter
                for adapter in unsupported
                if adapter in reports
                and reports[adapter].status is CapabilityStatus.UNSUPPORTED_PLATFORM
            )
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                (
                    "One or more requested capabilities are unavailable on this platform."
                    if unsupported_platform
                    else "One or more requested capabilities are not managed by FlameOx."
                ),
                details={
                    "unsupported_adapters": list(unsupported),
                    "unsupported_platform": list(unsupported_platform),
                    "next_tool": "list_capabilities",
                },
                remediation=(
                    (
                        "Use a supported platform for the managed capability, then retry setup."
                        if unsupported_platform
                        else "Request only capabilities whose report includes a "
                        "start_capability_setup setup action; host tools and permissions are "
                        "not installed by FlameOx."
                    ),
                ),
            )

        already_available = tuple(
            adapter
            for adapter in requested
            if reports[adapter].status is CapabilityStatus.AVAILABLE
        )
        pending = tuple(adapter for adapter in requested if adapter not in already_available)
        if not pending:
            self._record_managed_extras(
                tuple(
                    item.extra
                    for item in (reports[adapter].setup for adapter in requested)
                    if isinstance(item, CapabilitySetup)
                )
            )
            return CapabilitySetupResult(
                requested=requested,
                installed=(),
                already_available=already_available,
                setup_verification=self._verification(requested, reports),
            )

        setup = tuple(reports[adapter].setup for adapter in pending)
        requirements = tuple(
            dict.fromkeys(
                item.requirement
                for item in setup
                if isinstance(item, CapabilitySetup) and item.requirement
            )
        )
        pending_trace = "perfetto" in pending
        pending_toxiproxy = "toxiproxy" in pending
        self._record_setup_receipt(
            requested,
            completed=already_available,
            phase=(
                "installing_packages"
                if requirements
                else (
                    "staging_trace_processor"
                    if pending_trace
                    else ("staging_toxiproxy" if pending_toxiproxy else "completed")
                )
            ),
        )
        staging_phase: str | None = None
        lock_path = Path(sys.executable).parent / ".flameox-capability-setup.lock"
        uv = shutil.which("uv") if requirements else None
        if requirements and uv is None:
            self._record_setup_receipt(
                requested,
                completed=already_available,
                phase="failed",
                error="uv is missing from PATH.",
            )
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The managed runtime cannot prepare optional capabilities because uv is missing.",
                details={"next_tool": "start_capability_setup"},
                remediation=(
                    "Install uv, then reconnect FlameOx and call start_capability_setup again.",
                ),
            )
        try:
            if requirements:
                with portalocker.Lock(lock_path, mode="a", timeout=30):
                    self._check_cancelled(cancel_event)
                    command = [
                        str(uv),
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        *requirements,
                    ]
                    outcome = _run_brokered_sync(
                        self.broker,
                        ExecutionRequest(
                            argv=tuple(command),
                            cwd=Path.cwd(),
                            environment_allowlist=INSTALLER_ENVIRONMENT_ALLOWLIST,
                            environment_overrides={"UV_NO_PROGRESS": "1"},
                            allowed_working_roots=(Path.cwd(),),
                            timeout_seconds=_CAPABILITY_INSTALL_TIMEOUT_SECONDS,
                            max_output_bytes=16 * 1024 * 1024,
                        ),
                        cancel_event=cancel_event,
                        cancellation_message=(
                            "Capability setup was cancelled and its installer was terminated."
                        ),
                        cancellation_details={"next_tool": "start_capability_setup"},
                    )
                if outcome.process.exit_code != 0:
                    detail = _output_detail(outcome)[:500]
                    raise DomainError(
                        ErrorCode.PROCESS_FAILED,
                        "FlameOx could not prepare the requested optional capabilities.",
                        retryable=True,
                        details={"next_tool": "start_capability_setup", "error": detail},
                        remediation=(
                            "Retry start_capability_setup after checking the bounded installer "
                            "error.",
                        ),
                    )
                self._record_setup_receipt(
                    requested,
                    completed=self._available_requested(requested),
                    phase=(
                        "staging_trace_processor"
                        if pending_trace
                        else ("staging_toxiproxy" if pending_toxiproxy else "completed")
                    ),
                )
            if pending_trace:
                self._check_cancelled(cancel_event)
                if self.workspace is None:
                    raise DomainError(
                        ErrorCode.WORKSPACE_NOT_FOUND,
                        "A workspace is required to stage the managed Trace Processor.",
                        details={"next_tool": "initialize_workspace"},
                    )
                staging_phase = "staging_trace_processor"
                self._record_setup_receipt(
                    requested,
                    completed=self._available_requested(requested),
                    phase="staging_trace_processor",
                )
                if phase_callback is not None:
                    phase_callback("staging_trace_processor")
                install_trace_processor(
                    self.workspace,
                    cancel_event=cancel_event,
                    broker=self.broker,
                )
                self._check_cancelled(cancel_event)
            if pending_toxiproxy:
                self._check_cancelled(cancel_event)
                if self.workspace is None:
                    raise DomainError(
                        ErrorCode.WORKSPACE_NOT_FOUND,
                        "A workspace is required to stage managed Toxiproxy.",
                        details={"next_tool": "initialize_workspace"},
                    )
                staging_phase = "staging_toxiproxy"
                self._record_setup_receipt(
                    requested,
                    completed=self._available_requested(requested),
                    phase="staging_toxiproxy",
                )
                if phase_callback is not None:
                    phase_callback("staging_toxiproxy")
                receipt = ToxiproxyToolManager(self.workspace.paths.root).stage()
                self._verify_toxiproxy(receipt)
                self._check_cancelled(cancel_event)
        except DomainError as exc:
            failure = self._annotate_setup_phase(exc, staging_phase=staging_phase)
            self._record_setup_receipt(
                requested,
                completed=self._available_requested(requested),
                phase="failed",
                error=self._setup_failure_message(failure),
            )
            if failure is exc:
                raise
            raise failure from exc
        except (OSError, portalocker.exceptions.LockException) as exc:
            detail = self._bounded_setup_detail(exc)
            phase_detail = f" [phase={staging_phase}]" if staging_phase is not None else ""
            self._record_setup_receipt(
                requested,
                completed=self._available_requested(requested),
                phase="failed",
                error=f"Capability setup failed{phase_detail}: {detail}",
            )
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                "FlameOx could not prepare the requested optional capabilities.",
                retryable=True,
                details={
                    "next_tool": "start_capability_setup",
                    "error": detail,
                    **({"phase": staging_phase} if staging_phase is not None else {}),
                },
                remediation=(
                    "Retry start_capability_setup after checking uv and package-index access.",
                ),
            ) from exc
        refreshed = {item.adapter: item for item in self.list().capabilities}
        not_ready = tuple(
            adapter
            for adapter in pending
            if refreshed[adapter].status is not CapabilityStatus.AVAILABLE
        )
        if not_ready:
            self._record_setup_receipt(
                requested,
                completed=self._available_requested(requested),
                phase="failed",
                error="One or more requested capabilities did not become available.",
            )
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The installer completed but one or more capabilities did not become available.",
                details={"adapters": list(not_ready), "next_tool": "list_capabilities"},
                remediation=(
                    "Call list_capabilities to inspect the verified provider state before "
                    "planning.",
                ),
            )
        self._record_managed_extras(
            tuple(
                item.extra
                for item in (refreshed[adapter].setup for adapter in requested)
                if isinstance(item, CapabilitySetup)
            )
        )
        self._record_setup_receipt(
            requested,
            completed=requested,
            phase="completed",
        )
        return CapabilitySetupResult(
            requested=requested,
            installed=pending,
            already_available=already_available,
            setup_verification=self._verification(requested, refreshed),
        )

    @staticmethod
    def _bounded_setup_detail(error: object) -> str:
        detail = " ".join(str(error).split())
        return detail[:500] or "Capability setup returned no diagnostic detail."

    @staticmethod
    def _annotate_setup_phase(error: DomainError, *, staging_phase: str | None) -> DomainError:
        if staging_phase is None or isinstance(error.details.get("phase"), str):
            return error
        details = dict(error.details)
        details["phase"] = staging_phase
        return DomainError(
            error.code,
            error.message,
            retryable=error.retryable,
            details=details,
            remediation=error.remediation,
            run_id=error.run_id,
        )

    @classmethod
    def _setup_failure_message(cls, error: DomainError) -> str:
        phase = error.details.get("phase")
        category = error.details.get("failure_category")
        detail = error.details.get("failure_detail") or error.details.get("error")
        if not isinstance(category, str) or not isinstance(detail, str):
            return error.message
        phase_label = f" [phase={phase}]" if isinstance(phase, str) else ""
        return f"{error.message}{phase_label} [{category}] {cls._bounded_setup_detail(detail)}"

    @staticmethod
    def _check_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise DomainError(
                ErrorCode.PROCESS_CANCELLED,
                "Capability setup was cancelled before the next side effect.",
                retryable=True,
                details={"next_tool": "start_capability_setup"},
                remediation=("Retry the exact capability setup request when ready.",),
            )

    def prepare_adapter(self, adapter: str, distribution: str) -> AdapterPreparationResult:
        if self.workspace is None:
            raise DomainError(
                ErrorCode.WORKSPACE_NOT_FOUND,
                "A workspace is required to record an adapter approval.",
                details={"next_tool": "initialize_workspace"},
            )
        descriptor = AdapterRegistry(self.workspace).prepare(adapter, distribution)
        return AdapterPreparationResult(
            adapter=descriptor.adapter,
            distribution=descriptor.distribution,
            version=descriptor.version,
            package_identity=descriptor.package_identity,
            setup_verification=SetupVerification(
                status="verified",
                checked_adapters=(descriptor.adapter,),
                available_adapters=(descriptor.adapter,),
            ),
        )

    @staticmethod
    def _finish(
        reports: Sequence[CapabilityReport],
        *,
        latest_setup: CapabilitySetupReceipt | None = None,
        recommendation_adapter: str | None = None,
    ) -> CapabilityList:
        available_setup_adapters = tuple(
            sorted(
                item.adapter
                for item in reports
                if item.status is not CapabilityStatus.AVAILABLE
                and isinstance(item.setup, CapabilitySetup)
            )
        )
        available_setup_third_party_adapters = tuple(
            sorted(
                item.adapter
                for item in reports
                if item.status is not CapabilityStatus.AVAILABLE
                and isinstance(item.setup, AdapterSetup)
            )
        )
        scoped_reports = (
            tuple(item for item in reports if item.adapter == recommendation_adapter)
            if recommendation_adapter is not None
            else ()
        )
        setup_adapters = tuple(
            item.adapter
            for item in scoped_reports
            if item.status is not CapabilityStatus.AVAILABLE
            and isinstance(item.setup, CapabilitySetup)
        )
        setup_third_party_adapters = tuple(
            item.adapter
            for item in scoped_reports
            if item.status is not CapabilityStatus.AVAILABLE
            and isinstance(item.setup, AdapterSetup)
        )
        return CapabilityList(
            capabilities=tuple(reports),
            setup_adapters=setup_adapters,
            setup_third_party_adapters=setup_third_party_adapters,
            available_setup_adapters=available_setup_adapters,
            available_setup_third_party_adapters=available_setup_third_party_adapters,
            recommendation_scope=recommendation_adapter,
            latest_setup=latest_setup,
            next_tool=(
                "start_capability_setup"
                if setup_adapters
                else ("prepare_adapter" if setup_third_party_adapters else None)
            ),
        )

    def _available_requested(self, requested: tuple[str, ...]) -> tuple[str, ...]:
        reports = {item.adapter: item for item in self.list().capabilities}
        return tuple(
            adapter
            for adapter in requested
            if reports[adapter].status is CapabilityStatus.AVAILABLE
        )

    def _record_setup_receipt(
        self,
        requested: tuple[str, ...],
        *,
        completed: tuple[str, ...],
        phase: Literal[
            "installing_packages",
            "staging_trace_processor",
            "staging_toxiproxy",
            "completed",
            "failed",
        ],
        error: str | None = None,
    ) -> None:
        receipt = CapabilitySetupReceipt(
            requested=requested,
            completed=completed,
            phase=phase,
            error=error,
            updated_at=utc_now(),
        )
        atomic_write_json(self.setup_receipt_path, receipt.model_dump(mode="json"))

    def _read_setup_receipt(self) -> CapabilitySetupReceipt | None:
        try:
            return CapabilitySetupReceipt.model_validate_json(self.setup_receipt_path.read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _verification(
        requested: tuple[str, ...],
        reports: dict[str, CapabilityReport],
    ) -> SetupVerification:
        available = tuple(
            adapter
            for adapter in requested
            if reports[adapter].status is CapabilityStatus.AVAILABLE
        )
        unavailable = tuple(adapter for adapter in requested if adapter not in available)
        return SetupVerification(
            status="verified" if not unavailable else "partial",
            checked_adapters=requested,
            available_adapters=available,
            unavailable_adapters=unavailable,
        )

    @staticmethod
    def _setup(adapter: BuiltinAdapter) -> CapabilitySetup | None:
        if adapter.managed_extra is None or adapter.managed_requirement is None:
            return None
        return CapabilitySetup(
            method="start_capability_setup",
            extra=adapter.managed_extra,
            requirement=adapter.managed_requirement,
            next_tool="start_capability_setup",
        )

    def _toxiproxy_report(self, system: str, architecture: str) -> CapabilityReport:
        assert self.workspace is not None
        manager = ToxiproxyToolManager(self.workspace.paths.root)
        release = manager.release_for_host()
        receipt = manager.staged_receipt()
        if release is None:
            return CapabilityReport(
                adapter="toxiproxy",
                status=CapabilityStatus.UNSUPPORTED_PLATFORM,
                provisioning=CapabilityProvisioning.UNSUPPORTED,
                platform=system,
                architecture=architecture,
                features=("loopback_transport_faults",),
                limitations=("No pinned Toxiproxy release asset exists for this platform.",),
                setup_verification="not_required",
            )
        return CapabilityReport(
            adapter="toxiproxy",
            status=(
                CapabilityStatus.AVAILABLE
                if receipt is not None
                else CapabilityStatus.UNAVAILABLE
            ),
            provisioning=CapabilityProvisioning.MANAGED_RUNTIME,
            executable=str(receipt.executable) if receipt is not None else None,
            version=receipt.version if receipt is not None else None,
            supported_modes=("fault_experiment",) if receipt is not None else (),
            supported_formats=("loopback-transport",) if receipt is not None else (),
            platform=system,
            architecture=architecture,
            features=("loopback_transport_faults", "typed_toxics"),
            remediation=()
            if receipt is not None
            else ("Call start_capability_setup with adapter='toxiproxy'.",),
            setup=CapabilitySetup(
                method="start_capability_setup",
                extra="toxiproxy",
                requirement=None,
                next_tool="start_capability_setup",
            ),
            setup_verification="passive" if receipt is not None else "pending",
        )

    def _verify_toxiproxy(self, receipt: ToxiproxyToolReceipt) -> None:
        admin_port = _free_loopback_port()
        client = ToxiproxyClient(f"http://127.0.0.1:{admin_port}")

        async def verify() -> None:
            lease = await self.broker.start_toxiproxy(
                receipt.executable,
                admin_host="127.0.0.1",
                admin_port=admin_port,
                readiness=lambda: asyncio.to_thread(client.health),
                tool_receipt=receipt,
                readiness_timeout_seconds=5,
            )
            outcome = await lease.close()
            if not outcome.process.cleanup_complete:
                raise DomainError(
                    ErrorCode.PROCESS_FAILED,
                    "Managed Toxiproxy setup verification could not clean up its probe.",
                    retryable=True,
                )

        try:
            asyncio.run(verify())
        except DomainError:
            raise
        except (OSError, RuntimeError) as error:
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                "Managed Toxiproxy failed its bounded health verification.",
                retryable=True,
                details={"error": str(error)[:500]},
            ) from error

    def _record_managed_extras(self, extras: tuple[str, ...]) -> None:
        values = {
            value
            for value in extras
            if value in {"cpu", "execution", "memory", "test", "trace", "torch"}
        }
        if not values:
            return
        existing: set[str] = set()
        try:
            payload = json.loads(self.capability_manifest.read_text())
            if isinstance(payload, dict) and isinstance(payload.get("extras"), list):
                existing = {
                    value
                    for value in payload["extras"]
                    if isinstance(value, str)
                    and value in {"cpu", "execution", "memory", "test", "trace", "torch"}
                }
        except (OSError, ValueError):
            pass
        payload = {"schema_version": 1, "extras": sorted(existing | values)}
        atomic_write_json(self.capability_manifest, payload)
        if self.workspace is not None and self._uses_default_workspace_manifest:
            runtime_manifest = (
                Path(user_data_path("flameox", appauthor=False)) / "capabilities.json"
            )
            if runtime_manifest != self.capability_manifest:
                runtime_existing: set[str] = set()
                try:
                    runtime_payload = json.loads(runtime_manifest.read_text())
                    if isinstance(runtime_payload, dict) and isinstance(
                        runtime_payload.get("extras"), list
                    ):
                        runtime_existing = {
                            value
                            for value in runtime_payload["extras"]
                            if isinstance(value, str)
                            and value in {"cpu", "execution", "memory", "test", "trace", "torch"}
                        }
                except (OSError, ValueError):
                    pass
                atomic_write_json(
                    runtime_manifest,
                    {"schema_version": 1, "extras": sorted(runtime_existing | values)},
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
                    provisioning=(
                        CapabilityProvisioning.UNSUPPORTED
                        if system != "linux"
                        else CapabilityProvisioning.HOST
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


class CapabilitySetupManager:
    """Durable MCP lifecycle for providers and managed runtime tools."""

    def __init__(self, workspace: Workspace, service: CapabilityService) -> None:
        self.workspace = workspace
        self.service = service
        self.runner = OperationRunner(workspace, "capability.setup")

    async def start(
        self,
        adapters: tuple[str, ...],
        idempotency_key: str,
    ) -> OperationStatus:
        requested = tuple(dict.fromkeys(adapters))
        return await self.runner.start(
            {"adapters": requested},
            idempotency_key,
            self._run,
            items=requested,
        )

    async def status(self, operation_id: str) -> OperationStatus:
        return await self.runner.status(operation_id)

    async def cancel(self, operation_id: str) -> OperationStatus:
        return await self.runner.cancel(operation_id)

    async def shutdown(self) -> None:
        await self.runner.shutdown()

    async def _run(
        self,
        operation_id: str,
        progress: Callable[[str, float | None, float | None, str], Awaitable[None]],
    ) -> dict[str, object]:
        await progress("validating_request", 0, 3, "Validating managed capability request.")
        await progress("installing_packages", 1, 3, "Installing declared optional providers.")
        cancel_event = threading.Event()
        self.runner.set_cancel_hook(operation_id, cancel_event.set)
        loop = asyncio.get_running_loop()

        def report_phase(phase: str) -> None:
            if phase not in {"staging_trace_processor", "staging_toxiproxy"}:
                return

            message = (
                "Staging the managed Trace Processor."
                if phase == "staging_trace_processor"
                else "Staging and verifying managed Toxiproxy."
            )

            async def emit_progress() -> None:
                await progress(phase, 2, 3, message)

            future = asyncio.run_coroutine_threadsafe(emit_progress(), loop)
            future.result()

        try:
            try:
                requested = self._requested(operation_id)
                result = await asyncio.to_thread(
                    self.service.prepare,
                    requested,
                    cancel_event=cancel_event,
                    phase_callback=report_phase,
                )
            except DomainError as error:
                receipt = self.service._read_setup_receipt()
                raise OperationFailure(
                    error,
                    completed_items=receipt.completed if receipt is not None else (),
                ) from error
            await progress("verifying", 2, 3, "Refreshing and verifying requested capabilities.")
            await progress("completed", 3, 3, "Capability setup complete.")
            return {"setup": result.model_dump(mode="json")}
        finally:
            self.runner.clear_cancel_hook(operation_id)

    def _requested(self, operation_id: str) -> tuple[str, ...]:
        record = self.runner.store.read(operation_id)
        # Item identities retain the exact bounded request without exposing
        # package-manager arguments or host paths to the protocol.
        return tuple(item.item for item in record.item_outcomes)


def _free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _run_brokered(
    broker: SubprocessBroker,
    request: ExecutionRequest,
    *,
    cancel_event: threading.Event | None,
    cancellation_message: str,
    cancellation_details: dict[str, str],
) -> ExecutionOutcome:
    execution = asyncio.create_task(broker.run(request))
    if cancel_event is None:
        return await execution

    cancellation = asyncio.create_task(_wait_for_cancellation(cancel_event))
    done, _ = await asyncio.wait(
        (execution, cancellation),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if cancellation in done and execution not in done:
        execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
        raise DomainError(
            ErrorCode.PROCESS_CANCELLED,
            cancellation_message,
            retryable=True,
            details=cancellation_details,
        )
    cancellation.cancel()
    await asyncio.gather(cancellation, return_exceptions=True)
    return await execution


async def _wait_for_cancellation(cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(0.05)


def _run_brokered_sync(
    broker: SubprocessBroker,
    request: ExecutionRequest,
    *,
    cancel_event: threading.Event | None,
    cancellation_message: str,
    cancellation_details: dict[str, str],
) -> ExecutionOutcome:
    coroutine = _run_brokered(
        broker,
        request,
        cancel_event=cancel_event,
        cancellation_message=cancellation_message,
        cancellation_details=cancellation_details,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: list[ExecutionOutcome] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run, name="flameox-broker-bridge")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _output_detail(outcome: ExecutionOutcome) -> str:
    return (
        outcome.stderr.decode("utf-8", errors="replace").strip()
        or outcome.stdout.decode("utf-8", errors="replace").strip()
    )
