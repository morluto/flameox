from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import socket
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from platformdirs import user_data_path
from pydantic import ConfigDict, Field, TypeAdapter, computed_field, model_validator

from flameox.action_graph import (
    ActionId,
    NextAction,
    ToolAction,
    manual_action,
    tool_action,
)
from flameox.adapters.builtins import (
    BUILTIN_ADAPTERS,
    AdapterDependencyKind,
    BuiltinAdapter,
    builtin_adapter,
)
from flameox.adapters.nsight_compute import find_ncu_report_interface
from flameox.adapters.registry import AdapterRegistry
from flameox.adapters.setup_runtime import install_trace_processor
from flameox.adapters.toxiproxy import ToxiproxyClient, ToxiproxyToolManager, ToxiproxyToolReceipt
from flameox.application.concurrency import race_with_cancellation
from flameox.application.operations import (
    OperationAdapter,
    OperationFailure,
    OperationRunner,
    OperationStatus,
)
from flameox.application.provider_catalog import MANAGED_PROVIDERS, managed_provider
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.application.task_supervisor import TaskSupervisor
from flameox.atomic import atomic_write_json
from flameox.command_binding import ExecutableResolver
from flameox.domain import (
    AdapterSetup,
    CapabilityExtra,
    CapabilityPermissionStatus,
    CapabilityProvisioning,
    CapabilityReport,
    CapabilitySetup,
    CapabilitySetupVerification,
    CapabilityStatus,
    DomainError,
    ErrorCode,
    ProbeKind,
)
from flameox.domain.models import utc_now
from flameox.execution import ExecutionOutcome, ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import Workspace

type CapabilitySetupProgressPhase = Literal[
    "installing_packages",
    "staging_trace_processor",
    "staging_toxiproxy",
]


class _CapabilitySetupReceipt(ContractModel):
    """Fields shared by every durable capability-setup state."""

    schema_version: Literal[2] = 2
    requested: tuple[str, ...]
    updated_at: datetime
    next_action: ToolAction = Field(
        default_factory=lambda: tool_action(ActionId.INSPECT_CAPABILITIES)
    )


class _IncompleteCapabilitySetupReceipt(_CapabilitySetupReceipt):
    completed: tuple[str, ...] = ()

    @model_validator(mode="after")
    def completed_is_an_ordered_subset(self) -> _IncompleteCapabilitySetupReceipt:
        completed = set(self.completed)
        if (
            len(completed) != len(self.completed)
            or tuple(item for item in self.requested if item in completed) != self.completed
        ):
            raise ValueError("completed adapters must be an ordered subset of requested adapters")
        return self


class CapabilitySetupProgressReceipt(_IncompleteCapabilitySetupReceipt):
    phase: CapabilitySetupProgressPhase
    error: Literal[None] = None


class CapabilitySetupCompletedReceipt(_CapabilitySetupReceipt):
    phase: Literal["completed"] = "completed"
    completed: tuple[str, ...]
    error: Literal[None] = None

    @model_validator(mode="after")
    def completed_includes_every_requested_adapter(self) -> CapabilitySetupCompletedReceipt:
        if self.completed != self.requested:
            raise ValueError("completed setup must include every requested adapter")
        return self


class CapabilitySetupFailedReceipt(_IncompleteCapabilitySetupReceipt):
    phase: Literal["failed"] = "failed"
    error: str = Field(min_length=1, max_length=500)


type CapabilitySetupReceipt = (
    CapabilitySetupProgressReceipt | CapabilitySetupCompletedReceipt | CapabilitySetupFailedReceipt
)

_CAPABILITY_SETUP_RECEIPT_ADAPTER: TypeAdapter[CapabilitySetupReceipt] = TypeAdapter(
    CapabilitySetupReceipt
)


class CapabilityList(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: int = 1
    capabilities: tuple[CapabilityReport, ...]
    recommendation_scope: str | None = None
    latest_setup: CapabilitySetupReceipt | None = None
    next_action: NextAction | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def setup_adapters(self) -> tuple[str, ...]:
        return _capability_setup_projections(self.capabilities, self.recommendation_scope)[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def setup_third_party_adapters(self) -> tuple[str, ...]:
        return _capability_setup_projections(self.capabilities, self.recommendation_scope)[1]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available_setup_adapters(self) -> tuple[str, ...]:
        return _capability_setup_projections(self.capabilities, self.recommendation_scope)[2]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available_setup_third_party_adapters(self) -> tuple[str, ...]:
        return _capability_setup_projections(self.capabilities, self.recommendation_scope)[3]

    @model_validator(mode="after")
    def action_matches_selected_adapter(self) -> CapabilityList:
        expected = _capability_setup_projections(
            self.capabilities,
            self.recommendation_scope,
        )[4]
        if self.next_action != expected:
            raise ValueError("next action must match the selected adapter recommendation")
        return self


class SetupVerification(ContractModel):
    """Evidence that a setup action was checked before it returned success."""

    model_config = ConfigDict(json_schema_mode_override="serialization")

    checked_adapters: tuple[str, ...]
    available_adapters: tuple[str, ...]
    method: Literal["capability_scan"] = "capability_scan"

    @model_validator(mode="after")
    def availability_is_a_partition(self) -> SetupVerification:
        if len(set(self.checked_adapters)) != len(self.checked_adapters):
            raise ValueError("checked adapters must be unique")
        if len(set(self.available_adapters)) != len(self.available_adapters):
            raise ValueError("available adapters must be unique")
        available = set(self.available_adapters)
        if (
            tuple(item for item in self.checked_adapters if item in available)
            != self.available_adapters
        ):
            raise ValueError("available adapters must be an ordered subset of checked adapters")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unavailable_adapters(self) -> tuple[str, ...]:
        available = set(self.available_adapters)
        return tuple(item for item in self.checked_adapters if item not in available)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> Literal["verified", "partial"]:
        return "verified" if not self.unavailable_adapters else "partial"


class CapabilitySetupResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: int = 1
    requested: tuple[str, ...]
    already_available: tuple[str, ...]
    next_action: ToolAction = Field(
        default_factory=lambda: tool_action(ActionId.INSPECT_CAPABILITIES)
    )
    setup_verification: SetupVerification
    workload_executed: Literal[False] = False

    @model_validator(mode="after")
    def availability_is_a_partition(self) -> CapabilitySetupResult:
        if len(set(self.requested)) != len(self.requested):
            raise ValueError("requested adapters must be unique")
        available = set(self.already_available)
        if tuple(item for item in self.requested if item in available) != self.already_available:
            raise ValueError("already-available adapters must be an ordered requested subset")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def installed(self) -> tuple[str, ...]:
        available = set(self.already_available)
        return tuple(item for item in self.requested if item not in available)


class AdapterPreparationResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: int = 1
    adapter: str
    distribution: str
    version: str
    package_identity: str
    approval_provenance: Literal["agent"] = "agent"
    next_action: ToolAction = Field(
        default_factory=lambda: tool_action(ActionId.INSPECT_CAPABILITIES)
    )
    setup_verification: SetupVerification
    workload_executed: Literal[False] = False


def _capability_setup_projections(
    capabilities: tuple[CapabilityReport, ...],
    recommendation_scope: str | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    NextAction | None,
]:
    available_setup_adapters = tuple(
        sorted(
            item.adapter
            for item in capabilities
            if item.status is not CapabilityStatus.AVAILABLE
            and isinstance(item.setup, CapabilitySetup)
        )
    )
    available_setup_third_party_adapters = tuple(
        sorted(
            item.adapter
            for item in capabilities
            if item.status is not CapabilityStatus.AVAILABLE
            and isinstance(item.setup, AdapterSetup)
        )
    )
    scoped = (
        tuple(item for item in capabilities if item.adapter == recommendation_scope)
        if recommendation_scope is not None
        else ()
    )
    setup_adapters = tuple(
        item.adapter
        for item in scoped
        if item.status is not CapabilityStatus.AVAILABLE and isinstance(item.setup, CapabilitySetup)
    )
    third_party = tuple(
        item.adapter
        for item in scoped
        if item.status is not CapabilityStatus.AVAILABLE and isinstance(item.setup, AdapterSetup)
    )
    selected_setup = next(
        (
            item.setup
            for item in scoped
            if item.status is not CapabilityStatus.AVAILABLE and item.setup is not None
        ),
        None,
    )
    return (
        setup_adapters,
        third_party,
        available_setup_adapters,
        available_setup_third_party_adapters,
        selected_setup.next_action if selected_setup is not None else None,
    )


_CONTAINMENT_VERSION_ARGS = {
    "containment.bubblewrap": ("--version",),
    "containment.systemd": ("--version",),
}


def _retry_capability_setup_action() -> NextAction:
    return manual_action(
        "Choose an idempotency key and retry the reported capability setup request.",
        suggested_action=ActionId.START_CAPABILITY_SETUP,
        missing_arguments=("idempotency_key",),
    )


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
        self.capability_manifest = capability_manifest or (
            workspace.paths.records / "capabilities.json"
            if workspace is not None
            else Path(user_data_path("flameox", appauthor=False)) / "capabilities.json"
        )
        self.setup_receipt_path = self.capability_manifest.with_name("capability-setup.json")
        self.provider_runtimes = ProviderRuntimeManager(
            self.capability_manifest.parent / "provider-runtimes",
            broker=self.broker,
        )
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
            if adapter.dependency_kind is not AdapterDependencyKind.INTERNAL:
                continue
            supported_platform = (
                adapter.supported_platforms is None or system in adapter.supported_platforms
            )
            reports.append(
                CapabilityReport(
                    adapter=adapter.name,
                    status=(
                        CapabilityStatus.AVAILABLE
                        if supported_platform
                        else CapabilityStatus.UNSUPPORTED_PLATFORM
                    ),
                    provisioning=(
                        CapabilityProvisioning.BUNDLED
                        if supported_platform
                        else CapabilityProvisioning.UNSUPPORTED
                    ),
                    supported_modes=adapter.supported_modes if supported_platform else (),
                    supported_formats=adapter.supported_formats if supported_platform else (),
                    platform=system,
                    architecture=architecture,
                    features=adapter.features,
                    restrictions=self._platform_restrictions(adapter),
                    limitations=adapter.capture_limitations,
                )
            )
        executable_adapters = (
            adapter
            for adapter in BUILTIN_ADAPTERS.values()
            if adapter.dependency_kind is AdapterDependencyKind.EXECUTABLE
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
                        CapabilityPermissionStatus.UNKNOWN_UNTIL_ACTIVE_PROBE
                        if adapter.name in {"py-spy", "perf", "nsight.compute"} and resolved
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
                        CapabilitySetupVerification.PENDING
                        if resolved is None and adapter.managed_extra is not None
                        else (
                            CapabilitySetupVerification.PASSIVE
                            if resolved
                            else CapabilitySetupVerification.NOT_REQUIRED
                        )
                    ),
                    limitations=(("Version not probed in passive mode.",) if resolved else ()),
                )
            )
        reports.extend(self._containment_reports(system, architecture))
        package_adapters = (
            adapter
            for adapter in BUILTIN_ADAPTERS.values()
            if adapter.dependency_kind is AdapterDependencyKind.PACKAGE
        )
        for adapter in package_adapters:
            if adapter.dependency is None:
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "Builtin adapter is missing its dependency declaration.",
                )
            reports.append(
                CapabilityReport(
                    adapter=adapter.name,
                    status=CapabilityStatus.UNKNOWN,
                    provisioning=CapabilityProvisioning.WORKLOAD_ENVIRONMENT,
                    platform=system,
                    architecture=architecture,
                    features=adapter.features,
                    remediation=(
                        "Select a declared workload and plan capture so Flameox can inspect this "
                        "package through that workload's exact Python interpreter.",
                    ),
                    setup_verification=CapabilitySetupVerification.NOT_REQUIRED,
                    limitations=(
                        "Python package capabilities are workload-scoped and cannot be determined "
                        "from the Flameox control interpreter.",
                    ),
                )
            )
        if self.workspace is not None:
            reports.extend(self._managed_provider_reports(system, architecture))
            reports.append(self._toxiproxy_report(self.workspace, system, architecture))
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
                                adapter=descriptor.adapter,
                                distribution=descriptor.distribution,
                                package_identity=descriptor.package_identity,
                                next_action=tool_action(
                                    ActionId.PREPARE_ADAPTER,
                                    adapter=descriptor.adapter,
                                    distribution=descriptor.distribution,
                                ),
                                verification_action=tool_action(
                                    ActionId.INSPECT_CAPABILITIES,
                                    adapter=descriptor.adapter,
                                ),
                            )
                        ),
                        setup_verification=(
                            CapabilitySetupVerification.PENDING
                            if not descriptor.approved
                            else CapabilitySetupVerification.PASSIVE
                        ),
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
            provider = managed_provider(report.adapter)
            version_args = (
                definition.version_args
                if definition is not None
                else (
                    provider.version_args
                    if provider is not None
                    else _CONTAINMENT_VERSION_ARGS.get(report.adapter, ())
                )
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
        provider = managed_provider(adapter)
        version_args = (
            definition.version_args
            if definition is not None
            else (
                provider.version_args
                if provider is not None
                else _CONTAINMENT_VERSION_ARGS.get(adapter, ())
            )
        )
        if not version_args:
            return passive
        version_report = await self._probe_version(passive, version_args)
        if version_report.status is not CapabilityStatus.AVAILABLE:
            self._active_cache[adapter] = version_report
            return version_report
        if adapter == "perf":
            result = await self._probe_perf(passive)
            result = result.validated_copy(update={"version": version_report.version})
            self._active_cache[adapter] = result
            return result
        if adapter == "nsight.compute":
            result = await self._probe_nsight_compute(version_report)
            self._active_cache[adapter] = result
            return result
        result = version_report
        if adapter == "py-spy":
            result = result.validated_copy(
                update={"permission_status": CapabilityPermissionStatus.NOT_EXERCISED}
            )
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
                    executable_binding=ExecutableResolver().require_host_tool(
                        passive.executable, cwd=cwd
                    ),
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
            lines = tuple(line.strip() for line in output.splitlines() if line.strip())
            version_line = next(
                (
                    line
                    for line in lines
                    if passive.adapter in {"compute-sanitizer", "nsight.compute"}
                    and line.casefold().startswith("version ")
                ),
                lines[0] if lines else None,
            )
            succeeded = outcome.process.exit_code == 0
            return passive.validated_copy(
                update={
                    "status": CapabilityStatus.AVAILABLE
                    if succeeded
                    else CapabilityStatus.DEGRADED,
                    "version": version_line or passive.version,
                    "limitations": (
                        ()
                        if succeeded
                        else (f"Active version probe exited with {outcome.process.exit_code}.",)
                    ),
                    "probe_kind": ProbeKind.ACTIVE,
                    "setup_verification": CapabilitySetupVerification.ACTIVE,
                    "probed_at": utc_now(),
                }
            )
        except (DomainError, OSError, ValueError) as exc:
            return passive.validated_copy(
                update={
                    "status": CapabilityStatus.DEGRADED,
                    "limitations": (f"Active version probe failed with {type(exc).__name__}.",),
                    "probe_kind": ProbeKind.ACTIVE,
                    "setup_verification": CapabilitySetupVerification.ACTIVE,
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
                    executable_binding=ExecutableResolver().require_host_tool(
                        passive.executable, cwd=self.workspace.project_root
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
            return passive.validated_copy(
                update={
                    "status": CapabilityStatus.AVAILABLE,
                    "permission_status": CapabilityPermissionStatus.GRANTED,
                    "limitations": (),
                    "remediation": (),
                    "probe_kind": ProbeKind.ACTIVE,
                    "probed_at": utc_now(),
                }
            )
        if self._is_perf_permission_denial(diagnostic):
            return self._perf_permission_failure(passive, diagnostic)
        return self._perf_failure(passive, diagnostic or "perf sampling probe failed.")

    async def _probe_nsight_compute(self, passive: CapabilityReport) -> CapabilityReport:
        """Check the shipped report interface and map NVIDIA's counter diagnostic."""
        if self.workspace is None or passive.executable is None:
            return passive.validated_copy(
                update={"permission_status": CapabilityPermissionStatus.NOT_EXERCISED}
            )
        if find_ncu_report_interface(executable=passive.executable) is None:
            return passive.validated_copy(
                update={
                    "status": CapabilityStatus.DEGRADED,
                    "permission_status": CapabilityPermissionStatus.UNKNOWN,
                    "limitations": ("The official ncu_report Python interface is unavailable.",),
                    "remediation": (
                        "Install the Nsight Compute extras/python interface; FlameOx does not "
                        "decode NVIDIA report binaries.",
                    ),
                    "probe_kind": ProbeKind.ACTIVE,
                    "probed_at": utc_now(),
                }
            )
        restriction = self._nvidia_counter_access_restriction()
        if restriction is not None:
            return self._nsight_compute_permission_failure(passive, restriction)
        staging_root = self.workspace.paths.staging
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                dir=staging_root, prefix="capability-ncu-"
            ) as temporary:
                output = Path(temporary) / "probe.ncu-rep"
                request = ExecutionRequest(
                    argv=(
                        passive.executable,
                        "--set",
                        "basic",
                        "--launch-count",
                        "1",
                        "--export",
                        str(output),
                        "--force-overwrite",
                        sys.executable,
                        "-I",
                        "-S",
                        "-c",
                        "pass",
                    ),
                    executable_binding=ExecutableResolver().require_host_tool(
                        passive.executable, cwd=self.workspace.project_root
                    ),
                    cwd=self.workspace.project_root,
                    environment_allowlist=(),
                    allowed_working_roots=(self.workspace.project_root,),
                    timeout_seconds=10,
                    max_output_bytes=self.workspace.config.execution.max_output_bytes,
                    resource_policy=ResourcePolicy(
                        filesystem_path=self.workspace.paths.root,
                        staging_root=Path(temporary),
                        minimum_free_bytes=self.workspace.config.storage.min_free_bytes,
                        sampling_interval_ms=(
                            self.workspace.config.execution.resource_sampling_interval_ms
                        ),
                        max_observed_files=(
                            self.workspace.config.execution.max_resource_observed_files
                        ),
                    ),
                )
                try:
                    outcome = await self.broker.run(request)
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
                    diagnostic = self._bounded_diagnostic(diagnostic)
                    if "ERR_NVGPUCTRPERM" in diagnostic:
                        return self._nsight_compute_permission_failure(passive, diagnostic)
                    return self._nsight_compute_probe_failure(passive, diagnostic)
        except (OSError, ValueError) as error:
            return self._nsight_compute_probe_failure(passive, str(error))
        diagnostic = self._bounded_diagnostic(
            (outcome.stdout + b"\n" + outcome.stderr).decode("utf-8", errors="replace")
        )
        if "ERR_NVGPUCTRPERM" in diagnostic:
            return self._nsight_compute_permission_failure(passive, diagnostic)
        if outcome.process.exit_code != 0:
            return self._nsight_compute_probe_failure(passive, diagnostic)
        return passive.validated_copy(
            update={
                "permission_status": CapabilityPermissionStatus.NOT_EXERCISED,
                "limitations": (
                    "The official report interface is available, but the bounded probe did not "
                    "execute a CUDA kernel; counter permission remains unexercised.",
                ),
                "probe_kind": ProbeKind.ACTIVE,
                "probed_at": utc_now(),
            }
        )

    @staticmethod
    def _nsight_compute_permission_failure(
        passive: CapabilityReport,
        diagnostic: str,
    ) -> CapabilityReport:
        return passive.validated_copy(
            update={
                "status": CapabilityStatus.PERMISSION_REQUIRED,
                "permission_status": CapabilityPermissionStatus.DENIED,
                "limitations": (diagnostic,),
                "remediation": (
                    "Enable NVIDIA GPU performance-counter access following NVIDIA's "
                    "ERR_NVGPUCTRPERM guidance, then refresh capabilities; FlameOx will not "
                    "change system privileges.",
                ),
                "probe_kind": ProbeKind.ACTIVE,
                "probed_at": utc_now(),
            }
        )

    @staticmethod
    def _nvidia_counter_access_restriction() -> str | None:
        """Read NVIDIA's Linux driver policy without attempting a privilege change."""
        parameters = Path("/proc/driver/nvidia/params")
        if not parameters.is_file() or os.geteuid() == 0:
            return None
        try:
            admin_only = any(
                line.strip() == "RmProfilingAdminOnly: 1"
                for line in parameters.read_text(encoding="utf-8").splitlines()
            )
            status_lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
            effective = next(
                int(line.split(":", 1)[1].strip(), 16)
                for line in status_lines
                if line.startswith("CapEff:")
            )
        except (OSError, StopIteration, ValueError):
            return None
        cap_sys_admin = bool(effective & (1 << 21))
        if admin_only and not cap_sys_admin:
            return (
                "ERR_NVGPUCTRPERM: NVIDIA driver reports RmProfilingAdminOnly=1 and this "
                "process lacks CAP_SYS_ADMIN."
            )
        return None

    @staticmethod
    def _nsight_compute_probe_failure(
        passive: CapabilityReport,
        diagnostic: str,
    ) -> CapabilityReport:
        return passive.validated_copy(
            update={
                "status": CapabilityStatus.DEGRADED,
                "permission_status": CapabilityPermissionStatus.UNKNOWN,
                "limitations": (diagnostic or "Nsight Compute active probe failed.",),
                "remediation": ("Inspect the bounded Nsight Compute probe diagnostic.",),
                "probe_kind": ProbeKind.ACTIVE,
                "probed_at": utc_now(),
            }
        )

    def _perf_permission_failure(
        self,
        passive: CapabilityReport,
        diagnostic: str,
    ) -> CapabilityReport:
        return passive.validated_copy(
            update={
                "status": CapabilityStatus.PERMISSION_REQUIRED,
                "permission_status": CapabilityPermissionStatus.DENIED,
                "limitations": (diagnostic or "perf event access was denied.",),
                "remediation": (self._perf_remediation(diagnostic),),
                "probe_kind": ProbeKind.ACTIVE,
                "probed_at": utc_now(),
            }
        )

    def _perf_failure(self, passive: CapabilityReport, diagnostic: str) -> CapabilityReport:
        return passive.validated_copy(
            update={
                "status": CapabilityStatus.DEGRADED,
                "permission_status": CapabilityPermissionStatus.UNKNOWN,
                "limitations": (diagnostic,),
                "remediation": (
                    "Inspect the bounded perf probe diagnostic and refresh capabilities.",
                ),
                "probe_kind": ProbeKind.ACTIVE,
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
                next_action=tool_action(ActionId.INSPECT_CAPABILITIES),
            )

        already_available = tuple(
            adapter
            for adapter in requested
            if reports[adapter].status is CapabilityStatus.AVAILABLE
        )
        pending = tuple(adapter for adapter in requested if adapter not in already_available)
        if not pending:
            return CapabilitySetupResult(
                requested=requested,
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
        initial_phase: CapabilitySetupProgressPhase | None = (
            "installing_packages"
            if requirements
            else (
                "staging_trace_processor"
                if pending_trace
                else ("staging_toxiproxy" if pending_toxiproxy else None)
            )
        )
        if initial_phase is not None:
            self._record_setup_progress(
                requested,
                completed=already_available,
                phase=initial_phase,
            )
        staging_phase: CapabilitySetupProgressPhase | None = None
        try:
            if requirements:
                for adapter in pending:
                    capability_setup = reports[adapter].setup
                    if not isinstance(capability_setup, CapabilitySetup):
                        continue
                    self._check_cancelled(cancel_event)
                    self._prepare_provider(adapter, capability_setup, cancel_event=cancel_event)
                next_phase: CapabilitySetupProgressPhase | None = (
                    "staging_trace_processor"
                    if pending_trace
                    else ("staging_toxiproxy" if pending_toxiproxy else None)
                )
                if next_phase is not None:
                    self._record_setup_progress(
                        requested,
                        completed=self._available_requested(requested),
                        phase=next_phase,
                    )
            if pending_trace:
                self._check_cancelled(cancel_event)
                if self.workspace is None:
                    raise DomainError(
                        ErrorCode.WORKSPACE_NOT_FOUND,
                        "A workspace is required to stage the managed Trace Processor.",
                        next_action=tool_action(ActionId.INITIALIZE_WORKSPACE),
                    )
                staging_phase = "staging_trace_processor"
                self._record_setup_progress(
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
                        next_action=tool_action(ActionId.INITIALIZE_WORKSPACE),
                    )
                staging_phase = "staging_toxiproxy"
                self._record_setup_progress(
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
            self._record_setup_failure(
                requested,
                completed=self._available_requested(requested),
                error=self._setup_failure_message(failure),
            )
            if failure is exc:
                raise
            raise failure from exc
        except OSError as exc:
            detail = self._bounded_setup_detail(exc)
            phase_detail = f" [phase={staging_phase}]" if staging_phase is not None else ""
            self._record_setup_failure(
                requested,
                completed=self._available_requested(requested),
                error=f"Capability setup failed{phase_detail}: {detail}",
            )
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                "FlameOx could not prepare the requested optional capabilities.",
                retryable=True,
                details={
                    "adapters": list(requested),
                    "error": detail,
                    **({"phase": staging_phase} if staging_phase is not None else {}),
                },
                remediation=(
                    "Retry start_capability_setup after checking uv and package-index access.",
                ),
                next_action=_retry_capability_setup_action(),
            ) from exc
        refreshed = {item.adapter: item for item in self.list().capabilities}
        not_ready = tuple(
            adapter
            for adapter in pending
            if refreshed[adapter].status is not CapabilityStatus.AVAILABLE
        )
        if not_ready:
            self._record_setup_failure(
                requested,
                completed=self._available_requested(requested),
                error="One or more requested capabilities did not become available.",
            )
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The installer completed but one or more capabilities did not become available.",
                details={"adapters": list(not_ready)},
                remediation=(
                    "Call list_capabilities to inspect the verified provider state before "
                    "planning.",
                ),
                next_action=tool_action(ActionId.INSPECT_CAPABILITIES),
            )
        self._record_setup_completed(requested)
        return CapabilitySetupResult(
            requested=requested,
            already_available=already_available,
            setup_verification=self._verification(requested, refreshed),
        )

    def _prepare_provider(
        self,
        adapter: str,
        setup: CapabilitySetup,
        *,
        cancel_event: threading.Event | None,
    ) -> None:
        if setup.requirement is None:
            return
        definition = builtin_adapter(adapter)
        provider = managed_provider(adapter)
        executable_name = (
            definition.dependency
            if definition is not None
            and definition.dependency_kind is AdapterDependencyKind.EXECUTABLE
            and adapter != "perfetto"
            else None
        )
        if provider is not None:
            executable_name = provider.executable

        def run(request: ExecutionRequest) -> ExecutionOutcome:
            return _run_brokered_sync(
                self.broker,
                request,
                cancel_event=cancel_event,
                cancellation_message=(
                    "Capability setup was cancelled and its provider process was terminated."
                ),
                cancellation_details={"adapter": adapter},
                cancellation_next_action=_retry_capability_setup_action(),
            )

        self.provider_runtimes.prepare(
            extra=setup.extra,
            requirement=setup.requirement,
            executable_name=executable_name,
            request_runner=run,
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
            next_action=error.next_action,
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
                remediation=("Retry the exact capability setup request when ready.",),
                next_action=_retry_capability_setup_action(),
            )

    def prepare_adapter(self, adapter: str, distribution: str) -> AdapterPreparationResult:
        if self.workspace is None:
            raise DomainError(
                ErrorCode.WORKSPACE_NOT_FOUND,
                "A workspace is required to record an adapter approval.",
                next_action=tool_action(ActionId.INITIALIZE_WORKSPACE),
            )
        descriptor = AdapterRegistry(self.workspace).prepare(adapter, distribution)
        return AdapterPreparationResult(
            adapter=descriptor.adapter,
            distribution=descriptor.distribution,
            version=descriptor.version,
            package_identity=descriptor.package_identity,
            setup_verification=SetupVerification(
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
        capabilities = tuple(reports)
        return CapabilityList(
            capabilities=capabilities,
            recommendation_scope=recommendation_adapter,
            latest_setup=latest_setup,
            next_action=_capability_setup_projections(
                capabilities,
                recommendation_adapter,
            )[4],
        )

    def _available_requested(self, requested: tuple[str, ...]) -> tuple[str, ...]:
        reports = {item.adapter: item for item in self.list().capabilities}
        return tuple(
            adapter
            for adapter in requested
            if reports[adapter].status is CapabilityStatus.AVAILABLE
        )

    def _write_setup_receipt(self, receipt: CapabilitySetupReceipt) -> None:
        atomic_write_json(self.setup_receipt_path, receipt.model_dump(mode="json"))

    def _record_setup_progress(
        self,
        requested: tuple[str, ...],
        *,
        completed: tuple[str, ...],
        phase: CapabilitySetupProgressPhase,
    ) -> None:
        self._write_setup_receipt(
            CapabilitySetupProgressReceipt(
                requested=requested,
                completed=completed,
                phase=phase,
                updated_at=utc_now(),
            )
        )

    def _record_setup_completed(
        self,
        requested: tuple[str, ...],
    ) -> None:
        self._write_setup_receipt(
            CapabilitySetupCompletedReceipt(
                requested=requested,
                completed=requested,
                updated_at=utc_now(),
            )
        )

    def _record_setup_failure(
        self,
        requested: tuple[str, ...],
        *,
        completed: tuple[str, ...],
        error: str,
    ) -> None:
        self._write_setup_receipt(
            CapabilitySetupFailedReceipt(
                requested=requested,
                completed=completed,
                error=error,
                updated_at=utc_now(),
            )
        )

    def _read_setup_receipt(self) -> CapabilitySetupReceipt | None:
        try:
            payload = json.loads(self.setup_receipt_path.read_text())
            if (
                isinstance(payload, dict)
                and payload.get("schema_version") == 1
                and payload.get("next_tool") == "list_capabilities"
            ):
                payload = {
                    **payload,
                    "schema_version": 2,
                    "next_action": tool_action(ActionId.INSPECT_CAPABILITIES).model_dump(
                        mode="json"
                    ),
                }
                payload.pop("next_tool", None)
            return _CAPABILITY_SETUP_RECEIPT_ADAPTER.validate_python(payload)
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
        return SetupVerification(
            checked_adapters=requested,
            available_adapters=available,
        )

    @staticmethod
    def _setup(adapter: BuiltinAdapter) -> CapabilitySetup | None:
        if adapter.managed_extra is None or adapter.managed_requirement is None:
            return None
        return CapabilitySetup(
            extra=adapter.managed_extra,
            requirement=adapter.managed_requirement,
            next_action=manual_action(
                "Choose an idempotency key and start setup for the reported adapter.",
                suggested_action=ActionId.START_CAPABILITY_SETUP,
                missing_arguments=("idempotency_key",),
            ),
            verification_action=tool_action(
                ActionId.INSPECT_CAPABILITIES,
                adapter=adapter.name,
            ),
        )

    def _managed_provider_reports(
        self,
        system: str,
        architecture: str,
    ) -> tuple[CapabilityReport, ...]:
        reports: list[CapabilityReport] = []
        for provider in MANAGED_PROVIDERS.values():
            runtime = self.provider_runtimes.find(
                extra=provider.extra,
                requirement=provider.requirement,
            )
            executable = (
                str(runtime.executable)
                if runtime is not None and runtime.executable is not None
                else None
            )
            available = runtime is not None and (
                provider.executable is None or executable is not None
            )
            reports.append(
                CapabilityReport(
                    adapter=provider.name,
                    status=(
                        CapabilityStatus.AVAILABLE if available else CapabilityStatus.UNAVAILABLE
                    ),
                    provisioning=CapabilityProvisioning.MANAGED_RUNTIME,
                    executable=executable,
                    supported_modes=provider.supported_modes if available else (),
                    supported_formats=provider.supported_formats if available else (),
                    platform=system,
                    architecture=architecture,
                    features=provider.features,
                    setup=CapabilitySetup(
                        extra=provider.extra,
                        requirement=provider.requirement,
                        next_action=manual_action(
                            "Choose an idempotency key and start setup for this provider.",
                            suggested_action=ActionId.START_CAPABILITY_SETUP,
                            missing_arguments=("idempotency_key",),
                        ),
                        verification_action=tool_action(
                            ActionId.INSPECT_CAPABILITIES,
                            adapter=provider.name,
                        ),
                    ),
                    setup_verification=(
                        CapabilitySetupVerification.PASSIVE
                        if available
                        else CapabilitySetupVerification.PENDING
                    ),
                    remediation=(
                        ()
                        if executable
                        else (
                            "Call start_capability_setup with this provider name to create its "
                            "isolated runtime.",
                        )
                    ),
                    limitations=provider.limitations,
                )
            )
        return tuple(reports)

    def _toxiproxy_report(
        self,
        workspace: Workspace,
        system: str,
        architecture: str,
    ) -> CapabilityReport:
        manager = ToxiproxyToolManager(workspace.paths.root)
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
                setup_verification=CapabilitySetupVerification.NOT_REQUIRED,
            )
        return CapabilityReport(
            adapter="toxiproxy",
            status=(
                CapabilityStatus.AVAILABLE if receipt is not None else CapabilityStatus.UNAVAILABLE
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
                extra=CapabilityExtra.TOXIPROXY,
                requirement=None,
                next_action=manual_action(
                    "Choose an idempotency key and start setup for the toxiproxy adapter.",
                    suggested_action=ActionId.START_CAPABILITY_SETUP,
                    missing_arguments=("idempotency_key",),
                ),
                verification_action=tool_action(
                    ActionId.INSPECT_CAPABILITIES,
                    adapter="toxiproxy",
                ),
            ),
            setup_verification=(
                CapabilitySetupVerification.PASSIVE
                if receipt is not None
                else CapabilitySetupVerification.PENDING
            ),
        )

    def _verify_toxiproxy(self, receipt: ToxiproxyToolReceipt) -> None:
        admin_port = _free_loopback_port()
        client = ToxiproxyClient(f"http://127.0.0.1:{admin_port}")

        async def verify() -> None:
            lease = await self.broker.start_toxiproxy(
                receipt.executable,
                admin_host="127.0.0.1",
                admin_port=admin_port,
                readiness=client.health_async,
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
            binding = (
                ExecutableResolver().resolve_host_tool(executable) if system == "linux" else None
            )
            resolved = str(binding.invocation_path) if binding is not None else None
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
        definition = builtin_adapter(adapter)
        if (
            definition is not None
            and definition.managed_extra is not None
            and definition.managed_requirement is not None
        ):
            runtime = self.provider_runtimes.find(
                extra=definition.managed_extra,
                requirement=definition.managed_requirement,
            )
            if runtime is not None and runtime.executable is not None:
                return str(runtime.executable)
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
        binding = ExecutableResolver().resolve_host_tool(executable)
        return str(binding.invocation_path) if binding is not None else None

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

    def __init__(
        self,
        workspace: Workspace,
        service: CapabilityService,
        *,
        supervisor: TaskSupervisor | None = None,
    ) -> None:
        self.workspace = workspace
        self.service = service
        self.runner = OperationRunner(
            workspace,
            OperationAdapter(
                kind="capability.setup",
                start_action=ActionId.START_CAPABILITY_SETUP,
                status_action=ActionId.GET_CAPABILITY_SETUP,
            ),
            supervisor=supervisor,
        )

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
    cancellation_next_action: NextAction,
) -> ExecutionOutcome:
    if cancel_event is None:
        return await broker.run(request)
    return await race_with_cancellation(
        broker.run(request),
        lambda: _wait_for_cancellation(cancel_event),
        lambda: DomainError(
            ErrorCode.PROCESS_CANCELLED,
            cancellation_message,
            retryable=True,
            details=cancellation_details,
            next_action=cancellation_next_action,
        ),
    )


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
    cancellation_next_action: NextAction,
) -> ExecutionOutcome:
    coroutine = _run_brokered(
        broker,
        request,
        cancel_event=cancel_event,
        cancellation_message=cancellation_message,
        cancellation_details=cancellation_details,
        cancellation_next_action=cancellation_next_action,
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
