from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from pydantic import ValidationError

from flameox import __version__
from flameox.action_graph import ActionId, ManualAction, ToolAction, manual_action, tool_action
from flameox.adapters import AdapterDiscoveryResult, AdapterRegistry
from flameox.adapters.builtins import builtin_adapter
from flameox.adapters.toxiproxy import ToxiproxyToolReceipt
from flameox.application import CapabilityList, CapabilityService
from flameox.application.capabilities import (
    CapabilitySetupManager,
    CapabilitySetupResult,
    SetupVerification,
)
from flameox.application.dependencies import WorkloadDependencyService
from flameox.application.operations import (
    ActiveOperationRecord,
    OperationAdapter,
    OperationState,
    OperationStatus,
    operation_digests,
)
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.domain import (
    CapabilityExtra,
    CapabilityProvisioning,
    CapabilityReport,
    CapabilitySetup,
    CapabilityStatus,
    DomainError,
    ErrorCode,
    ProcessResult,
    process_termination_from_returncode,
)
from flameox.domain.models import utc_now
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ProcessContainment,
    SubprocessBroker,
)
from flameox.http_transport import DownloadProgress
from flameox.storage import Workspace
from tests.support.execution import executable_binding

pytestmark = pytest.mark.integration


class _ProbeBroker(SubprocessBroker):
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[ExecutionRequest] = []

    async def run(
        self,
        request: ExecutionRequest,
        **_: Any,
    ) -> ExecutionOutcome:
        self.calls += 1
        self.requests.append(request)
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(0),
                cleanup_complete=True,
            ),
            stdout=b"trace_processor_shell 99.1\n",
            stderr=b"",
            resolved_executable=Path(request.argv[0]),
            executable_binding=request.executable_binding,
            containment=ProcessContainment.PROCESS_GROUP,
        )


def _outcome(request: ExecutionRequest, *, stdout: bytes = b"") -> ExecutionOutcome:
    return ExecutionOutcome(
        process=ProcessResult(
            termination=process_termination_from_returncode(0),
            cleanup_complete=True,
        ),
        stdout=stdout,
        stderr=b"",
        resolved_executable=Path(request.argv[0]),
        executable_binding=request.executable_binding,
        containment=ProcessContainment.PROCESS_GROUP,
    )


class _PerfProbeBroker(SubprocessBroker):
    def __init__(self, outcomes: tuple[ExecutionOutcome, ...]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ExecutionRequest] = []

    async def run(
        self,
        request: ExecutionRequest,
        **_: Any,
    ) -> ExecutionOutcome:
        self.requests.append(request)
        return self.outcomes.pop(0)


class _BlockingBroker(SubprocessBroker):
    def __init__(self) -> None:
        self.started = threading.Event()

    async def run(self, request: ExecutionRequest, **_: Any) -> ExecutionOutcome:
        del request
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


def _probe_outcome(
    *,
    exit_code: int,
    stdout: bytes = b"perf version 1\n",
    stderr: bytes = b"",
) -> ExecutionOutcome:
    return ExecutionOutcome(
        process=ProcessResult(
            termination=process_termination_from_returncode(exit_code),
            cleanup_complete=True,
        ),
        stdout=stdout,
        stderr=stderr,
        resolved_executable=Path("/usr/bin/perf"),
        executable_binding=executable_binding("/usr/bin/perf"),
        containment=ProcessContainment.PROCESS_GROUP,
    )


def _managed_setup(
    extra: CapabilityExtra,
    requirement: str | None,
    *,
    adapter: str = "torch.profiler",
) -> CapabilitySetup:
    return CapabilitySetup(
        extra=extra,
        requirement=requirement,
        next_action=manual_action(
            "Choose an idempotency key before starting capability setup.",
            suggested_action=ActionId.START_CAPABILITY_SETUP,
            missing_arguments=("idempotency_key",),
        ),
        verification_action=tool_action(
            ActionId.INSPECT_CAPABILITIES,
            adapter=adapter,
        ),
    )


def test_capability_setup_fields_are_derived_from_authoritative_reports() -> None:
    report = CapabilityReport(
        adapter="torch.profiler",
        status=CapabilityStatus.UNAVAILABLE,
        setup=_managed_setup(CapabilityExtra.TORCH, "torch>=2.7"),
    )
    capabilities = CapabilityList(
        capabilities=(report,),
        recommendation_scope=report.adapter,
        next_action=report.setup.next_action if report.setup is not None else None,
    )

    assert capabilities.setup_adapters == (report.adapter,)
    assert isinstance(capabilities.next_action, ManualAction)
    assert capabilities.next_action.suggested_action is ActionId.START_CAPABILITY_SETUP
    assert capabilities.validated_copy() == capabilities

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CapabilityList.model_validate(
            {**capabilities.model_dump(), "next_action": {"kind": "manual"}}
        )


def test_setup_verification_fields_form_one_partition() -> None:
    verification = SetupVerification(
        checked_adapters=("available", "missing"),
        available_adapters=("available",),
    )

    assert verification.unavailable_adapters == ("missing",)
    assert verification.status == "partial"
    assert verification.validated_copy() == verification

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SetupVerification.model_validate({**verification.model_dump(), "status": "verified"})


@pytest.mark.anyio
async def test_active_capability_probe_is_brokered_cached_and_refreshable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    broker = _ProbeBroker()
    service = CapabilityService(workspace, broker=broker)
    monkeypatch.setattr(
        service,
        "_resolved_executable",
        lambda adapter, executable: "/usr/bin/true" if adapter == "perfetto" else None,
    )

    first = await service.probe("perfetto")
    cached = await service.probe("perfetto")
    refreshed = await service.probe("perfetto", refresh=True)

    assert first.status is CapabilityStatus.AVAILABLE
    assert first.probe_kind == "active"
    assert first.probed_at is not None
    assert first.version == "trace_processor_shell 99.1"
    assert "trace_sql" in first.features
    assert cached == first
    assert refreshed.probed_at is not None
    assert broker.calls == 2

    pyperf_report = service.get("pyperf")
    assert pyperf_report.status is CapabilityStatus.UNKNOWN
    assert pyperf_report.provisioning == "workload_environment"
    assert pyperf_report.import_location is None
    assert "raw_samples" in pyperf_report.features
    torch_report = service.get("torch.profiler")
    assert torch_report.status is CapabilityStatus.UNKNOWN
    assert torch_report.provisioning == "workload_environment"
    assert torch_report.setup is None


def test_toxiproxy_setup_uses_dedicated_staging_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    executable = Path("/bin/true")
    receipt = ToxiproxyToolReceipt(
        version="2.12.0",
        asset="toxiproxy_2.12.0_linux_amd64.tar.gz",
        sha256="a" * 64,
        executable=executable,
        executable_sha256="b" * 64,
        manifest_revision="test-manifest",
    )

    class _ToolManager:
        staged = False

        def __init__(self, _: Path) -> None:
            pass

        @staticmethod
        def release_for_host() -> tuple[str, str, str]:
            return (receipt.asset, receipt.sha256, "toxiproxy-server")

        @classmethod
        def staged_receipt(cls) -> ToxiproxyToolReceipt | None:
            return receipt if cls.staged else None

        @classmethod
        def stage(cls, **kwargs: object) -> ToxiproxyToolReceipt:
            progress = cast(Callable[[DownloadProgress], None], kwargs["progress"])
            progress(
                DownloadProgress(
                    received_bytes=128,
                    expected_bytes=256,
                    elapsed_seconds=1.5,
                    resume_possible=True,
                    validator='"fixture"',
                )
            )
            cls.staged = True
            return receipt

    monkeypatch.setattr("flameox.application.capabilities.ToxiproxyToolManager", _ToolManager)
    monkeypatch.setattr(CapabilityService, "_verify_toxiproxy", lambda self, value: None)
    service = CapabilityService(workspace)

    report = service.get("toxiproxy")
    phases: list[str] = []
    downloads: list[DownloadProgress] = []

    def record_progress(phase: str, download: DownloadProgress | None) -> None:
        phases.append(phase)
        if download is not None:
            downloads.append(download)

    result = service.prepare(("toxiproxy",), phase_callback=record_progress)

    assert isinstance(report.setup, CapabilitySetup)
    assert report.setup.extra == "toxiproxy"
    assert phases == ["staging_toxiproxy", "staging_toxiproxy"]
    assert downloads[0].received_bytes == 128
    assert downloads[0].resume_possible is True
    assert result.installed == ("toxiproxy",)
    assert service.get("toxiproxy").status is CapabilityStatus.AVAILABLE


def test_unsupported_toxiproxy_platform_has_platform_specific_setup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flameox.application.capabilities.ToxiproxyToolManager.release_for_host",
        staticmethod(lambda: None),
    )
    service = CapabilityService(Workspace.initialize(tmp_path))

    report = service.get("toxiproxy")

    assert report.status is CapabilityStatus.UNSUPPORTED_PLATFORM
    with pytest.raises(DomainError) as error:
        service.prepare(("toxiproxy",))
    assert error.value.details["unsupported_platform"] == ["toxiproxy"]
    assert "unavailable on this platform" in error.value.message


def test_internal_adapter_honors_declared_platform_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.application.capabilities.platform.system", lambda: "Darwin")

    report = CapabilityService(Workspace.initialize(tmp_path)).get("nvbench")

    assert report.status is CapabilityStatus.UNSUPPORTED_PLATFORM
    assert report.provisioning.value == "unsupported"
    assert report.supported_modes == ()
    assert report.supported_formats == ()


@pytest.mark.anyio
async def test_perf_probe_exercises_permissions_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    broker = _PerfProbeBroker(
        (
            _probe_outcome(exit_code=0),
            _probe_outcome(exit_code=0, stdout=b"", stderr=b""),
            _probe_outcome(exit_code=0),
            _probe_outcome(
                exit_code=1,
                stdout=b"",
                stderr=(
                    b"Error: perf_event_open: Operation not permitted\n"
                    b"perf_event_paranoid setting is 4\n"
                ),
            ),
            _probe_outcome(exit_code=0),
            _probe_outcome(exit_code=1, stdout=b"", stderr=b"unexpected perf failure\n"),
        )
    )
    service = CapabilityService(workspace, broker=broker)
    monkeypatch.setattr(
        service,
        "_resolved_executable",
        lambda adapter, executable: "/usr/bin/perf" if adapter == "perf" else None,
    )

    granted = await service.probe("perf")
    cached = await service.probe("perf")
    refreshed = await service.probe("perf", refresh=True)

    assert granted.status is CapabilityStatus.AVAILABLE
    assert granted.permission_status == "granted"
    assert cached == granted
    assert refreshed.status is CapabilityStatus.PERMISSION_REQUIRED
    assert refreshed.permission_status == "denied"
    assert "perf_event_paranoid=4" in refreshed.remediation[0]
    assert "active_refresh" in refreshed.remediation[0]
    assert len(broker.requests) == 4
    record_request = broker.requests[1]
    assert record_request.argv[1:8] == (
        "record",
        "-B",
        "-N",
        "--max-size=1M",
        "-o",
        record_request.argv[6],
        "--",
    )
    assert record_request.argv[8:] == (sys.executable, "-I", "-S", "-c", "pass")
    assert record_request.resource_policy is not None
    assert record_request.resource_policy.staging_root is not None
    assert not record_request.resource_policy.staging_root.exists()

    degraded_service = CapabilityService(
        workspace,
        broker=_PerfProbeBroker(
            (
                _probe_outcome(exit_code=0),
                _probe_outcome(exit_code=1, stdout=b"", stderr=b"unexpected perf failure\n"),
            )
        ),
    )
    monkeypatch.setattr(
        degraded_service,
        "_resolved_executable",
        lambda adapter, executable: "/usr/bin/perf" if adapter == "perf" else None,
    )
    degraded = await degraded_service.probe("perf")
    assert degraded.status is CapabilityStatus.DEGRADED
    assert degraded.permission_status == "unknown"
    assert "unexpected perf failure" in degraded.limitations[0]


def test_entry_point_approval_is_lazy_and_revoked_by_distribution_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    loaded: list[str] = []

    class FakeDistribution:
        metadata: ClassVar[dict[str, str]] = {"Name": "example-profiler"}
        version = "1.0"
        files: tuple[object, ...] = ()

    class FakeEntryPoint:
        name = "example"
        value = "example_plugin:adapter"
        dist = FakeDistribution()

        def load(self) -> object:
            loaded.append(self.value)
            return object()

    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "flameox.adapters.registry.entry_points",
        lambda *, group: (entry_point,),
    )
    monkeypatch.setattr(
        "flameox.adapters.registry._distribution_identity",
        lambda distribution: f"identity:{distribution.version}",
    )
    registry = AdapterRegistry(workspace)

    discovered = registry.discover()
    assert discovered.adapters[0].approved is False
    assert loaded == []
    with pytest.raises(DomainError) as refused:
        registry.load_approved("example")
    assert refused.value.code is ErrorCode.EXECUTION_REFUSED

    approved = registry.approve("example-profiler")
    assert approved.adapters[0].approved is True
    registry.load_approved("example")
    assert loaded == ["example_plugin:adapter"]

    approved_snapshot = AdapterDiscoveryResult(adapters=approved.adapters)
    entry_point.dist.version = "2.0"
    monkeypatch.setattr(registry, "discover", lambda: approved_snapshot)
    with pytest.raises(DomainError) as changed:
        registry.load_approved("example")
    assert changed.value.code is ErrorCode.REVISION_CONFLICT


def test_agent_adapter_preparation_records_exact_identity_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)

    class FakeDistribution:
        metadata: ClassVar[dict[str, str]] = {"Name": "example-profiler"}
        version = "1.0"
        files: tuple[object, ...] = ()

    class FakeEntryPoint:
        name = "example"
        value = "example_plugin:adapter"
        dist = FakeDistribution()

    monkeypatch.setattr(
        "flameox.adapters.registry.entry_points",
        lambda *, group: (FakeEntryPoint(),),
    )
    monkeypatch.setattr(
        "flameox.adapters.registry._distribution_identity",
        lambda distribution: f"identity:{distribution.version}",
    )

    prepared = AdapterRegistry(workspace).prepare("example", "example-profiler")
    payload = json.loads((workspace.paths.records / "adapter-approvals.json").read_text())

    assert prepared.approved is True
    assert payload["approvals"]["example-profiler"]["provenance"] == "agent"


def test_capability_setup_installs_only_declared_missing_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    monkeypatch.setattr(ProviderRuntimeManager, "_active_source_root", lambda _self: None)
    receipt_during_install: dict[str, object] | None = None

    class ObservingBroker(_ProbeBroker):
        async def run(
            self,
            request: ExecutionRequest,
            **kwargs: Any,
        ) -> ExecutionOutcome:
            nonlocal receipt_during_install
            self.calls += 1
            self.requests.append(request)
            receipt_during_install = json.loads((tmp_path / "capability-setup.json").read_text())
            if request.argv[1:] == ("--version",):
                return _outcome(request, stdout=b"uv 0.9.0\n")
            if request.argv[1] == "venv":
                python = Path(request.argv[-1]) / "bin" / "python"
                python.parent.mkdir(parents=True)
                python.write_text("#!/bin/sh\nexit 0\n")
                python.chmod(0o755)
                return _outcome(request)
            if request.argv[1:3] == ("pip", "compile"):
                lock = Path(request.argv[request.argv.index("--output-file") + 1])
                lock.write_text(
                    "flameox==0.1.14 --hash=sha256:" + "a" * 64 + "\n"
                    "torch==2.7 --hash=sha256:" + "b" * 64 + "\n"
                )
                return _outcome(request)
            if request.argv[1:3] == ("pip", "install"):
                return _outcome(request)
            if request.argv[1:3] == ("-I", "-c"):
                return _outcome(
                    request,
                    stdout=json.dumps(
                        {
                            "executable": request.argv[0],
                            "prefix": str(Path(request.argv[0]).parent.parent),
                            "versions": {"flameox": __version__, "torch": "2.7"},
                        }
                    ).encode(),
                )
            return _outcome(request, stdout=b"trace_processor_shell 99.1\n")

    broker = ObservingBroker()
    service = CapabilityService(
        workspace,
        broker=broker,
        capability_manifest=tmp_path / "capabilities.json",
    )
    setup = _managed_setup(CapabilityExtra.TORCH, "torch>=2.7")
    missing = CapabilityReport(
        adapter="torch.profiler",
        status=CapabilityStatus.UNAVAILABLE,
        setup=setup,
    )
    available = missing.model_copy(update={"status": CapabilityStatus.AVAILABLE})
    reports = iter(
        (
            CapabilityList(
                capabilities=(missing,),
                recommendation_scope="torch.profiler",
                next_action=setup.next_action,
            ),
            CapabilityList(
                capabilities=(available,),
            ),
            CapabilityList(
                capabilities=(available,),
            ),
        )
    )
    monkeypatch.setattr(service, "list", lambda: next(reports))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
    (tmp_path / "bin").mkdir()
    result = service.prepare(("torch.profiler",))

    assert result.installed == ("torch.profiler",)
    assert result.already_available == ()
    assert receipt_during_install is not None
    assert receipt_during_install["phase"] == "installing_packages"
    assert receipt_during_install["completed"] == []
    receipt = json.loads((tmp_path / "capability-setup.json").read_text())
    assert receipt | {"updated_at": None} == {
        "completed": ["torch.profiler"],
        "error": None,
        "next_action": {
            "kind": "tool",
            "action": "capabilities.inspect",
            "arguments": {"mode": "passive"},
        },
        "phase": "completed",
        "requested": ["torch.profiler"],
        "updated_at": None,
    }
    assert isinstance(receipt["updated_at"], str)
    assert not (tmp_path / "capabilities.json").exists()
    assert len(tuple((tmp_path / "provider-runtimes").glob("*/provider-runtime.json"))) == 1
    install = next(
        request for request in broker.requests if request.argv[1:3] == ("pip", "install")
    )
    target = install.argv[install.argv.index("--python") + 1]
    assert target.startswith(str(tmp_path / "provider-runtimes"))
    assert target != str(tmp_path / "bin" / "python")
    assert "--require-hashes" in install.argv
    assert "-r" in install.argv
    assert any(request.argv[1:3] == ("pip", "compile") for request in broker.requests)
    assert "HTTPS_PROXY" in install.environment_allowlist
    assert "NO_PROXY" in install.environment_allowlist
    assert "UV_INDEX_URL" in install.environment_allowlist
    assert "SSL_CERT_FILE" in install.environment_allowlist


def test_reduction_provider_is_discoverable_without_becoming_a_capture_adapter(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = CapabilityService(workspace)

    report = service.get("shrinkray")

    assert report.status is CapabilityStatus.UNAVAILABLE
    assert report.provisioning is CapabilityProvisioning.MANAGED_RUNTIME
    assert report.setup is not None
    assert isinstance(report.setup, CapabilitySetup)
    assert report.setup.extra is CapabilityExtra.REDUCTION
    assert report.setup.requirement == "shrinkray==26.7.8.0"
    assert builtin_adapter("shrinkray") is None


def test_capability_setup_rejects_unmanaged_provider(tmp_path: Path) -> None:
    service = CapabilityService(
        Workspace.initialize(tmp_path),
        capability_manifest=tmp_path / "capabilities.json",
    )

    with pytest.raises(DomainError) as refused:
        service.prepare(("perf",))

    assert refused.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert isinstance(refused.value.next_action, ToolAction)
    assert refused.value.next_action.action is ActionId.INSPECT_CAPABILITIES


def test_capability_setup_cancellation_cleans_up_brokered_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    broker = _BlockingBroker()
    service = CapabilityService(
        workspace,
        broker=broker,
        capability_manifest=tmp_path / "capabilities.json",
    )
    report = CapabilityReport(
        adapter="torch.profiler",
        status=CapabilityStatus.UNAVAILABLE,
        setup=_managed_setup(CapabilityExtra.TORCH, "torch>=2.7"),
    )
    monkeypatch.setattr(service, "list", lambda: CapabilityList(capabilities=(report,)))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
    (tmp_path / "bin").mkdir()
    cancel_event = threading.Event()
    failures: list[BaseException] = []

    def run() -> None:
        try:
            service.prepare(("torch.profiler",), cancel_event=cancel_event)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert broker.started.wait(1)
    cancel_event.set()
    thread.join(2)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], DomainError)
    assert failures[0].code is ErrorCode.PROCESS_CANCELLED


def test_capability_setup_records_failure_when_uv_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CapabilityService(
        Workspace.initialize(tmp_path),
        capability_manifest=tmp_path / "capabilities.json",
    )
    missing = CapabilityReport(
        adapter="torch.profiler",
        status=CapabilityStatus.UNAVAILABLE,
        setup=_managed_setup(CapabilityExtra.TORCH, "torch>=2.7"),
    )
    monkeypatch.setattr(service, "list", lambda: CapabilityList(capabilities=(missing,)))
    monkeypatch.setattr("flameox.command_binding.shutil.which", lambda _name, path=None: None)

    with pytest.raises(DomainError) as unavailable:
        service.prepare(("torch.profiler",))

    assert unavailable.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    receipt = json.loads((tmp_path / "capability-setup.json").read_text())
    assert receipt["phase"] == "failed"
    assert receipt["completed"] == []
    assert receipt["error"] == "Executable 'uv' was not found in the request PATH."


def test_trace_processor_staging_preserves_phase_and_bounded_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = CapabilityService(
        workspace,
        broker=_ProbeBroker(),
        capability_manifest=tmp_path / "capabilities.json",
    )
    report = CapabilityReport(
        adapter="perfetto",
        status=CapabilityStatus.UNAVAILABLE,
        setup=_managed_setup(
            CapabilityExtra.TRACE,
            "perfetto>=0.57,<0.58",
            adapter="perfetto",
        ),
    )
    monkeypatch.setattr(service, "list", lambda: CapabilityList(capabilities=(report,)))
    monkeypatch.setattr(service, "_prepare_provider", lambda *args, **kwargs: None)

    def fail_staging(*args: object, **kwargs: object) -> object:
        raise DomainError(
            ErrorCode.PROCESS_FAILED,
            "FlameOx could not stage the managed Trace Processor.",
            retryable=True,
            details={
                "adapter": "perfetto",
                "failure_category": "network",
                "failure_detail": "synthetic TLS failure",
            },
            next_action=manual_action(
                "Choose an idempotency key and retry perfetto setup.",
                suggested_action=ActionId.START_CAPABILITY_SETUP,
                missing_arguments=("idempotency_key",),
            ),
        )

    monkeypatch.setattr("flameox.application.capabilities.install_trace_processor", fail_staging)
    phases: list[str] = []

    with pytest.raises(DomainError):
        service.prepare(
            ("perfetto",), phase_callback=lambda phase, _download: phases.append(phase)
        )

    receipt = json.loads((tmp_path / "capability-setup.json").read_text())
    assert phases == ["staging_trace_processor"]
    assert receipt["phase"] == "failed"
    assert "phase=staging_trace_processor" in receipt["error"]
    assert "network" in receipt["error"]
    assert "synthetic TLS failure" in receipt["error"]


def test_capability_setup_is_idempotent_when_provider_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CapabilityService(
        Workspace.initialize(tmp_path),
        capability_manifest=tmp_path / "capabilities.json",
    )
    report = CapabilityReport(
        adapter="torch.profiler",
        status=CapabilityStatus.AVAILABLE,
        setup=_managed_setup(CapabilityExtra.TORCH, "torch>=2.7"),
    )
    monkeypatch.setattr(service, "list", lambda: CapabilityList(capabilities=(report,)))
    result = service.prepare(("torch.profiler", "torch.profiler"))

    assert result.installed == ()
    assert result.already_available == ("torch.profiler",)


def test_list_capabilities_exposes_latest_setup_receipt(tmp_path: Path) -> None:
    manifest = tmp_path / "capabilities.json"
    (tmp_path / "capability-setup.json").write_text(
        json.dumps(
            {
                "requested": ["torch.profiler", "perfetto"],
                "completed": ["torch.profiler"],
                "phase": "staging_trace_processor",
                "error": None,
                "updated_at": "2026-08-01T15:30:00Z",
                "next_action": {
                    "kind": "tool",
                    "action": "capabilities.inspect",
                    "arguments": {"mode": "passive"},
                },
            }
        )
    )
    service = CapabilityService(Workspace.initialize(tmp_path), capability_manifest=manifest)

    result = service.list()

    assert result.latest_setup is not None
    assert result.latest_setup.requested == ("torch.profiler", "perfetto")
    assert result.latest_setup.completed == ("torch.profiler",)
    assert result.latest_setup.phase == "staging_trace_processor"


@pytest.mark.parametrize(
    ("phase", "completed", "error"),
    [
        ("failed", (), None),
        ("staging_trace_processor", (), "unexpected failure"),
        ("completed", ("torch.profiler",), None),
        ("installing_packages", ("unknown",), None),
    ],
)
def test_capability_setup_receipt_rejects_contradictory_durable_states(
    phase: str,
    completed: tuple[str, ...],
    error: str | None,
) -> None:
    with pytest.raises(ValidationError):
        CapabilityList.model_validate(
            {
                "capabilities": [],
                "latest_setup": {
                    "requested": ["torch.profiler", "perfetto"],
                    "completed": list(completed),
                    "phase": phase,
                    "error": error,
                    "updated_at": "2026-08-01T15:30:00Z",
                    "next_tool": "list_capabilities",
                },
            }
        )


def test_capability_recommendations_are_scoped_to_selected_adapter(
    tmp_path: Path,
) -> None:
    service = CapabilityService(Workspace.initialize(tmp_path))

    inventory = service.list()
    selected = service.list_for_adapter("torch.profiler")

    assert inventory.setup_adapters == ()
    assert "torch.profiler" not in inventory.available_setup_adapters
    assert inventory.next_action is None
    assert selected.recommendation_scope == "torch.profiler"
    assert selected.setup_adapters == ()
    assert selected.next_action is None


def test_running_capability_setup_status_contains_exact_poll_action(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = {"adapters": ["perfetto"]}
    _, idempotency_digest = operation_digests(
        workspace,
        "capability.setup",
        request,
        "status-test",
    )
    record = ActiveOperationRecord(
        operation="capability.setup",
        workspace_id=workspace.identity.workspace_id,
        request=request,
        idempotency_digest=idempotency_digest,
        state=OperationState.RUNNING,
        phase="staging_trace_processor",
        owner_id="test-owner",
        owner_heartbeat_at=utc_now(),
    )

    adapter = OperationAdapter(
        kind="capability.setup",
        start_action=ActionId.START_CAPABILITY_SETUP,
        status_action=ActionId.GET_CAPABILITY_SETUP,
    )
    status = OperationStatus.from_record(record, adapter=adapter)

    assert status.poll_after_ms == 1_000
    assert status.recovery is not None
    assert status.recovery.action == "poll"
    assert isinstance(status.recovery.next_action, ToolAction)
    assert status.recovery.next_action.action is ActionId.GET_CAPABILITY_SETUP
    assert status.recovery.next_action.arguments == {"operation_id": record.operation_id}

    terminal = OperationStatus.from_record(
        record.completed(receipt={}, item_outcomes=()),
        adapter=adapter,
    )
    assert terminal.poll_after_ms is None
    assert terminal.recovery is None


@pytest.mark.anyio
async def test_capability_setup_manager_persists_progress_and_final_receipt(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)

    class FakeCapabilityService:
        def prepare(
            self,
            adapters: tuple[str, ...],
            *,
            memray_reader_version: str | None,
            cancel_event: object,
            phase_callback: Any,
        ) -> CapabilitySetupResult:
            del cancel_event, memray_reader_version
            phase_callback(
                "staging_trace_processor",
                DownloadProgress(
                    received_bytes=256,
                    expected_bytes=1024,
                    elapsed_seconds=2.0,
                    resume_possible=True,
                    validator='"fixture"',
                ),
            )
            return CapabilitySetupResult(
                requested=adapters,
                already_available=(),
                setup_verification=SetupVerification(
                    checked_adapters=adapters,
                    available_adapters=adapters,
                ),
            )

        def _read_setup_receipt(self) -> None:
            return None

    manager = CapabilitySetupManager(workspace, cast(CapabilityService, FakeCapabilityService()))
    try:
        started = await manager.start(("torch.profiler", "perfetto"), "proof-key")
        terminal = started
        for _ in range(100):
            terminal = await manager.status(started.operation_id)
            if terminal.state == "terminal":
                break
            await asyncio.sleep(0.01)

        assert terminal.state == "terminal"
        assert [item.phase for item in terminal.progress] == [
            "validating_request",
            "installing_packages",
            "staging_trace_processor",
            "verifying",
            "completed",
        ]
        assert [item.item for item in terminal.item_outcomes] == ["torch.profiler", "perfetto"]
        assert all(item.status == "complete" for item in terminal.item_outcomes)
        assert terminal.cleanup_status == "complete"
        download = terminal.progress[2]
        assert download.completed == 256
        assert download.total == 1024
        assert download.message == "Downloading managed Trace Processor: 256 of 1024 bytes."
        assert terminal.terminal_receipt is not None
        assert terminal.terminal_receipt["setup"]["requested"] == [
            "torch.profiler",
            "perfetto",
        ]

        replay = await manager.start(("torch.profiler", "perfetto"), "proof-key")
        assert replay.operation_id == terminal.operation_id
        assert replay.request_digest == terminal.request_digest
    finally:
        await manager.shutdown()


@pytest.mark.anyio
async def test_capability_setup_identity_binds_exact_memray_reader_version(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    observed: list[str | None] = []

    class FakeCapabilityService:
        def prepare(
            self,
            adapters: tuple[str, ...],
            *,
            memray_reader_version: str | None,
            cancel_event: object,
            phase_callback: Any,
        ) -> CapabilitySetupResult:
            del cancel_event, phase_callback
            observed.append(memray_reader_version)
            return CapabilitySetupResult(
                requested=adapters,
                already_available=(),
                provider_environment_ids={"memray": "sha256:" + "e" * 64},
                setup_verification=SetupVerification(
                    checked_adapters=adapters,
                    available_adapters=adapters,
                ),
            )

        def _read_setup_receipt(self) -> None:
            return None

    manager = CapabilitySetupManager(workspace, cast(CapabilityService, FakeCapabilityService()))
    try:
        started = await manager.start(
            ("memray",),
            "memray-reader-proof",
            memray_reader_version="1.20.0",
        )
        terminal = await manager.runner.wait(started.operation_id, timeout_seconds=2)

        assert terminal.state == "terminal"
        assert observed == ["1.20.0"]
        assert terminal.terminal_receipt is not None
        assert terminal.terminal_receipt["setup"]["provider_environment_ids"] == {
            "memray": "sha256:" + "e" * 64
        }
        with pytest.raises(DomainError) as conflict:
            await manager.start(
                ("memray",),
                "memray-reader-proof",
                memray_reader_version="1.19.3",
            )
        assert conflict.value.code is ErrorCode.REVISION_CONFLICT
    finally:
        await manager.shutdown()


def test_capability_setup_provisions_exact_memray_reader_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = CapabilityService(workspace)
    report = CapabilityReport(
        adapter="memray",
        status=CapabilityStatus.UNAVAILABLE,
        setup=_managed_setup(
            CapabilityExtra.MEMORY,
            "memray>=1.17",
            adapter="memray",
        ),
    )
    monkeypatch.setattr(
        service,
        "list",
        lambda: CapabilityList(capabilities=(report,)),
    )
    prepared: list[str] = []
    runtime = SimpleNamespace(
        receipt=SimpleNamespace(environment_id="sha256:" + "e" * 64)
    )

    def find_distribution(**_kwargs: object) -> object | None:
        return runtime if prepared else None

    def prepare_provider(**kwargs: object) -> object:
        prepared.append(str(kwargs["requirement"]))
        return runtime

    monkeypatch.setattr(service.provider_runtimes, "find_distribution", find_distribution)
    monkeypatch.setattr(service.provider_runtimes, "prepare", prepare_provider)

    result = service.prepare(("memray",), memray_reader_version="1.20.0")

    assert prepared == ["memray==1.20.0"]
    assert result.provider_environment_ids == {"memray": "sha256:" + "e" * 64}
    assert result.setup_verification.status == "verified"


@pytest.mark.anyio
async def test_capability_setup_failure_keeps_staging_phase_and_diagnostics(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)

    class FailingCapabilityService:
        def prepare(
            self,
            adapters: tuple[str, ...],
            *,
            memray_reader_version: str | None,
            cancel_event: object,
            phase_callback: Any,
        ) -> CapabilitySetupResult:
            del adapters, cancel_event, memray_reader_version
            phase_callback("staging_trace_processor", None)
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                "FlameOx could not stage the managed Trace Processor.",
                retryable=True,
                details={
                    "adapter": "perfetto",
                    "phase": "staging_trace_processor",
                    "failure_category": "network",
                    "failure_detail": "synthetic TLS failure",
                },
            )

        def _read_setup_receipt(self) -> None:
            return None

    manager = CapabilitySetupManager(
        workspace,
        cast(CapabilityService, FailingCapabilityService()),
    )
    try:
        started = await manager.start(("perfetto",), "staging-failure-proof")
        failed = started
        for _ in range(100):
            failed = await manager.status(started.operation_id)
            if failed.state == "failed":
                break
            await asyncio.sleep(0.01)

        assert failed.state == "failed"
        assert failed.phase == "staging_trace_processor"
        assert [item.phase for item in failed.progress] == [
            "validating_request",
            "installing_packages",
            "staging_trace_processor",
        ]
        assert failed.failure_details == {
            "phase": "staging_trace_processor",
            "failure_category": "network",
            "adapter": "perfetto",
            "failure_detail": "synthetic TLS failure",
        }
    finally:
        await manager.shutdown()


@pytest.mark.anyio
async def test_capability_setup_cancel_returns_while_staging_cleanup_continues(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    staging = threading.Event()
    release_cleanup = threading.Event()

    class BlockedCapabilityService:
        def prepare(
            self,
            adapters: tuple[str, ...],
            *,
            memray_reader_version: str | None,
            cancel_event: threading.Event,
            phase_callback: Any,
        ) -> CapabilitySetupResult:
            del memray_reader_version
            phase_callback("staging_trace_processor", None)
            staging.set()
            assert cancel_event.wait(timeout=2)
            assert release_cleanup.wait(timeout=2)
            raise DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled")

        def _read_setup_receipt(self) -> None:
            return None

    manager = CapabilitySetupManager(
        workspace,
        cast(CapabilityService, BlockedCapabilityService()),
    )
    try:
        started = await manager.start(("perfetto",), "bounded-staging-cancel")
        assert await asyncio.to_thread(staging.wait, 1)

        before = asyncio.get_running_loop().time()
        cancelling = await manager.cancel(started.operation_id)

        assert asyncio.get_running_loop().time() - before < 1
        assert cancelling.state == "running"
        assert cancelling.phase == "cancelling"
        assert cancelling.cancellation_requested is True
        assert cancelling.cleanup_status == "pending"

        release_cleanup.set()
        terminal = await manager.runner.wait(started.operation_id, timeout_seconds=1)
        assert terminal.state == "cancelled"
        assert terminal.cleanup_status == "complete"
    finally:
        release_cleanup.set()
        await manager.shutdown()


def test_entry_point_approval_is_revoked_when_installed_content_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    installed = tmp_path / "example_adapter.py"
    installed.write_text("VERSION = 1\n")
    loaded: list[str] = []

    class FakePackagePath:
        hash = None
        size = None

        def __str__(self) -> str:
            return "example_adapter.py"

    class FakeDistribution:
        metadata: ClassVar[dict[str, str]] = {"Name": "example-profiler"}
        version = "1.0"
        files = (FakePackagePath(),)

        def locate_file(self, _path: object) -> Path:
            return installed

    class FakeEntryPoint:
        name = "example"
        value = "example_adapter:adapter"
        dist = FakeDistribution()

        def load(self) -> object:
            loaded.append(self.value)
            return object()

    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "flameox.adapters.registry.entry_points",
        lambda *, group: (entry_point,),
    )
    registry = AdapterRegistry(workspace)
    registry.approve("example-profiler")
    installed.write_text("VERSION = 2\n")

    with pytest.raises(DomainError) as changed:
        registry.load_approved("example")

    assert changed.value.code is ErrorCode.EXECUTION_REFUSED
    assert loaded == []


@pytest.mark.anyio
async def test_prepare_workload_dependencies_inspects_declared_interpreter_without_mutation(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    python = tmp_path / "workload-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf \'{"agent-fixture":"2.1"}\\n\'\n')
    python.chmod(0o755)
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.probe]
argv = ["{python}", "-c", "pass"]
[workloads.probe.requirements]
python_distributions = ["agent-fixture>=2"]
"""
    )

    class RecordingBroker(SubprocessBroker):
        def __init__(self) -> None:
            self.requests: list[ExecutionRequest] = []

        async def run(
            self,
            request: ExecutionRequest,
            **kwargs: Any,
        ) -> ExecutionOutcome:
            self.requests.append(request)
            return await super().run(request, **kwargs)

    broker = RecordingBroker()

    result = await WorkloadDependencyService(workspace, broker=broker).prepare("probe")

    assert result.installed == ()
    assert result.environment_mutated is False
    assert result.already_available == ("agent-fixture>=2",)
    assert result.status == "ready"
    assert isinstance(result.next_action, ManualAction)
    assert result.next_action.suggested_action is ActionId.PLAN_CAPTURE
    assert result.next_action.missing_arguments == ("adapter", "parameters")
    assert result.preflight.requirements[0].status == "available"
    assert len(broker.requests) == 1
    request = broker.requests[0]
    assert request.argv[:3] == (str(python), "-I", "-c")
    assert request.argv[-1] == "agent-fixture"
    assert request.environment_allowlist == ()
    assert request.executable_binding.invocation_path == python


@pytest.mark.anyio
async def test_prepare_workload_dependencies_reports_missing_without_installing(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    python = tmp_path / "empty-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nprintf '{\"agent-fixture\":null}\\n'\n")
    python.chmod(0o755)
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.probe]
argv = ["{python}", "-c", "pass"]
[workloads.probe.requirements]
python_distributions = ["agent-fixture>=2"]
"""
    )

    result = await WorkloadDependencyService(workspace).prepare("probe")

    assert result.installed == ()
    assert result.environment_mutated is False
    assert result.already_available == ()
    assert result.status == "blocked"
    assert result.preflight.requirements[0].status == "absent"
    assert isinstance(result.next_action, ManualAction)
    assert result.next_action.suggested_action is ActionId.GET_DECLARED_WORKFLOW
    assert "declared Python environment" in result.next_action.instruction
