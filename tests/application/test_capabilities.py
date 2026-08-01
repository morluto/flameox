from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from flameox.adapters import AdapterDiscoveryResult, AdapterRegistry
from flameox.application import CapabilityService
from flameox.domain import CapabilityStatus, DomainError, ErrorCode, ProcessResult
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
                stderr=b"Error: perf_event_open: Operation not permitted\n",
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
    assert refreshed.remediation
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
