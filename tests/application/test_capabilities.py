from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest

from flameox.adapters import AdapterDiscoveryResult, AdapterRegistry
from flameox.application import CapabilityList, CapabilityService
from flameox.application.capabilities import (
    CapabilitySetupManager,
    CapabilitySetupResult,
    SetupVerification,
)
from flameox.application.dependencies import WorkloadDependencyService
from flameox.domain import (
    CapabilityReport,
    CapabilitySetup,
    CapabilityStatus,
    DomainError,
    ErrorCode,
    ProcessResult,
)
from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker
from flameox.storage import Workspace


class _ProbeBroker(SubprocessBroker):
    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self,
        request: ExecutionRequest,
        **_: Any,
    ) -> ExecutionOutcome:
        self.calls += 1
        return ExecutionOutcome(
            process=ProcessResult(exit_code=0, cleanup_complete=True),
            stdout=b"trace_processor_shell 99.1\n",
            stderr=b"",
            resolved_executable=Path(request.argv[0]),
            containment="process_group",
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


def _probe_outcome(
    *,
    exit_code: int,
    stdout: bytes = b"perf version 1\n",
    stderr: bytes = b"",
) -> ExecutionOutcome:
    return ExecutionOutcome(
        process=ProcessResult(exit_code=exit_code, cleanup_complete=True),
        stdout=stdout,
        stderr=stderr,
        resolved_executable=Path("/usr/bin/perf"),
        containment="process_group",
    )


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
    assert pyperf_report.import_location is not None
    assert "raw_samples" in pyperf_report.features
    torch_report = service.get("torch.profiler")
    assert isinstance(torch_report.setup, CapabilitySetup)
    assert torch_report.setup.extra == "torch"


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


def test_prepare_capabilities_installs_only_declared_missing_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = CapabilityService(workspace, capability_manifest=tmp_path / "capabilities.json")
    setup = CapabilitySetup(
        extra="torch",
        method="prepare_capabilities",
        next_tool="prepare_capabilities",
        requirement="torch>=2.7",
    )
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
                setup_adapters=("torch.profiler",),
                next_tool="prepare_capabilities",
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
    monkeypatch.setattr("flameox.application.capabilities.shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
    (tmp_path / "bin").mkdir()
    calls: list[list[str]] = []

    def run(command: list[str], **_: object) -> object:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("flameox.application.capabilities.subprocess.run", run)

    result = service.prepare(("torch.profiler",))

    assert result.installed == ("torch.profiler",)
    assert result.already_available == ()
    receipt = json.loads((tmp_path / "capability-setup.json").read_text())
    assert receipt | {"updated_at": None} == {
        "completed": ["torch.profiler"],
        "error": None,
        "next_tool": "list_capabilities",
        "phase": "completed",
        "requested": ["torch.profiler"],
        "schema_version": 1,
        "updated_at": None,
    }
    assert isinstance(receipt["updated_at"], str)
    assert (tmp_path / "capabilities.json").read_text() == (
        '{\n  "extras": [\n    "torch"\n  ],\n  "schema_version": 1\n}\n'
    )
    assert calls == [
        [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            str(tmp_path / "bin" / "python"),
            "torch>=2.7",
        ]
    ]


def test_prepare_capabilities_rejects_unmanaged_provider(tmp_path: Path) -> None:
    service = CapabilityService(
        Workspace.initialize(tmp_path),
        capability_manifest=tmp_path / "capabilities.json",
    )

    with pytest.raises(DomainError) as refused:
        service.prepare(("perf",))

    assert refused.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert refused.value.details["next_tool"] == "list_capabilities"


def test_prepare_capabilities_records_failure_when_uv_is_missing(
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
        setup=CapabilitySetup(
            extra="torch",
            method="prepare_capabilities",
            next_tool="prepare_capabilities",
            requirement="torch>=2.7",
        ),
    )
    monkeypatch.setattr(service, "list", lambda: CapabilityList(capabilities=(missing,)))
    monkeypatch.setattr("flameox.application.capabilities.shutil.which", lambda _: None)

    with pytest.raises(DomainError) as unavailable:
        service.prepare(("torch.profiler",))

    assert unavailable.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    receipt = json.loads((tmp_path / "capability-setup.json").read_text())
    assert receipt["phase"] == "failed"
    assert receipt["completed"] == []
    assert receipt["error"] == "uv is missing from PATH."


def test_trace_processor_staging_preserves_phase_and_bounded_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = CapabilityService(
        workspace,
        capability_manifest=tmp_path / "capabilities.json",
    )
    report = CapabilityReport(
        adapter="perfetto",
        status=CapabilityStatus.UNAVAILABLE,
        setup=CapabilitySetup(
            extra="trace",
            method="prepare_capabilities",
            next_tool="prepare_capabilities",
            requirement="perfetto>=0.57,<0.58",
        ),
    )
    monkeypatch.setattr(service, "list", lambda: CapabilityList(capabilities=(report,)))
    monkeypatch.setattr("flameox.application.capabilities.shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(
        "flameox.application.capabilities.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )

    def fail_staging(*args: object, **kwargs: object) -> object:
        raise DomainError(
            ErrorCode.PROCESS_FAILED,
            "FlameOx could not stage the managed Trace Processor.",
            retryable=True,
            details={
                "next_tool": "prepare_capabilities",
                "adapter": "perfetto",
                "failure_category": "network",
                "failure_detail": "synthetic TLS failure",
            },
        )

    monkeypatch.setattr("flameox.application.capabilities.install_trace_processor", fail_staging)
    phases: list[str] = []

    with pytest.raises(DomainError):
        service.prepare(("perfetto",), phase_callback=phases.append)

    receipt = json.loads((tmp_path / "capability-setup.json").read_text())
    assert phases == ["staging_trace_processor"]
    assert receipt["phase"] == "failed"
    assert "phase=staging_trace_processor" in receipt["error"]
    assert "network" in receipt["error"]
    assert "synthetic TLS failure" in receipt["error"]


def test_prepare_capabilities_is_idempotent_when_provider_is_available(
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
        setup=CapabilitySetup(
            extra="torch",
            method="prepare_capabilities",
            next_tool="prepare_capabilities",
            requirement="torch>=2.7",
        ),
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
                "schema_version": 1,
                "requested": ["torch.profiler", "perfetto"],
                "completed": ["torch.profiler"],
                "phase": "staging_trace_processor",
                "error": None,
                "updated_at": "2026-08-01T15:30:00Z",
                "next_tool": "list_capabilities",
            }
        )
    )
    service = CapabilityService(Workspace.initialize(tmp_path), capability_manifest=manifest)

    result = service.list()

    assert result.latest_setup is not None
    assert result.latest_setup.requested == ("torch.profiler", "perfetto")
    assert result.latest_setup.completed == ("torch.profiler",)
    assert result.latest_setup.phase == "staging_trace_processor"


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
            cancel_event: object,
            phase_callback: Any,
        ) -> CapabilitySetupResult:
            del cancel_event
            del phase_callback
            return CapabilitySetupResult(
                requested=adapters,
                installed=adapters,
                already_available=(),
                setup_verification=SetupVerification(
                    status="verified",
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
            "verifying",
            "completed",
        ]
        assert [item.item for item in terminal.item_outcomes] == ["torch.profiler", "perfetto"]
        assert all(item.status == "complete" for item in terminal.item_outcomes)
        assert terminal.cleanup_status == "complete"
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
async def test_capability_setup_failure_keeps_staging_phase_and_diagnostics(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)

    class FailingCapabilityService:
        def prepare(
            self,
            adapters: tuple[str, ...],
            *,
            cancel_event: object,
            phase_callback: Any,
        ) -> CapabilitySetupResult:
            del adapters, cancel_event
            phase_callback("staging_trace_processor")
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
async def test_prepare_workload_dependencies_installs_declared_spec_and_reruns_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["python", "-c", "pass"]
[workloads.probe.requirements]
python_distributions = ["agent-fixture>=2"]
"""
    )
    available = False

    def lookup(_: str) -> SimpleNamespace:
        if not available:
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError("agent-fixture")
        return SimpleNamespace(metadata={"Name": "agent-fixture"}, version="2.1")

    def install(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal available
        available = True
        assert command[-1] == "agent-fixture>=2"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("flameox.application.dependencies.distribution", lookup)
    monkeypatch.setattr("flameox.application.preflight.distribution", lookup)
    monkeypatch.setattr("flameox.application.dependencies._uv_executable", lambda: "/usr/bin/uv")

    async def install_async(
        _: object,
        command: list[str],
    ) -> subprocess.CompletedProcess[str]:
        return install(command)

    monkeypatch.setattr(
        "flameox.application.dependencies.WorkloadDependencyService._run_install",
        install_async,
    )

    result = await WorkloadDependencyService(Workspace(tmp_path / ".diagnostics")).prepare("probe")

    assert result.installed == ("agent-fixture>=2",)
    assert result.already_available == ()
    assert result.status == "ready"
    assert result.next_tool == "plan_capture"
    assert result.preflight.requirements[0].status == "available"
