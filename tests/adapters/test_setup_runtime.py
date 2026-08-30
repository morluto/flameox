from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from pathlib import Path

import httpx
import pytest

import flameox.adapters.setup_runtime as _setup_runtime
from flameox.action_graph import ActionId, manual_action, tool_action
from flameox.adapters.setup_runtime import (
    ManagedRuntime,
    install_trace_processor,
)
from flameox.application.capabilities import CapabilityList, CapabilityService
from flameox.domain import (
    CapabilityExtra,
    CapabilityReport,
    CapabilitySetup,
    CapabilityStatus,
    DomainError,
    ErrorCode,
    ProcessResult,
    process_termination_from_returncode,
)
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ProcessContainment,
    SubprocessBroker,
)
from flameox.http_transport import BoundedHttpClient
from flameox.managed_tools import (
    ManagedToolAsset,
    build_managed_tool_receipt,
    write_managed_tool_receipt,
)
from flameox.storage import Workspace

pytestmark = pytest.mark.unit


class RecordingRuntime(ManagedRuntime):
    def __init__(self, root: Path, *, broker: SubprocessBroker | None = None) -> None:
        super().__init__(root, broker=broker)
        self.verified: list[Path] = []

    async def verify(self, executable: Path, version: str) -> None:
        assert executable == self.executable(version)
        self.verified.append(executable)


class RecordingBroker(SubprocessBroker):
    def __init__(self, *, stdout: bytes = b"trace_processor_shell 55.1\n") -> None:
        self.requests: list[ExecutionRequest] = []
        self.stdout = stdout

    async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        self.requests.append(request)
        bin_directory = request.environment_overrides.get("UV_TOOL_BIN_DIR")
        if bin_directory is not None:
            path = Path(bin_directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / ("flameox.exe" if os.name == "nt" else "flameox")).write_text("")
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(0),
                cleanup_complete=True,
            ),
            stdout=self.stdout,
            stderr=b"",
            resolved_executable=Path(request.argv[0]),
            executable_binding=request.executable_binding,
            containment=ProcessContainment.PROCESS_GROUP,
        )


class BlockingBroker(SubprocessBroker):
    def __init__(self) -> None:
        self.started = threading.Event()

    async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        del request
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


@pytest.mark.anyio
async def test_runtime_install_uses_an_exact_isolated_uv_tool_environment(
    tmp_path: Path,
) -> None:
    broker = RecordingBroker()
    runtime = RecordingRuntime(tmp_path, broker=broker)

    result = await runtime.install("0.1.1")

    request = broker.requests[0]
    recorded_command = list(request.argv)
    recorded_environment = request.environment_overrides

    assert recorded_command == [
        "uv",
        "tool",
        "install",
        "--force",
        "--no-config",
        "--no-sources",
        "--prerelease",
        "allow",
        "--python",
        "3.12",
        "flameox==0.1.1",
    ]
    assert recorded_environment["UV_TOOL_DIR"] == str(tmp_path / "runtimes" / "0.1.1" / "tools")
    assert "PATH" in request.environment_allowlist
    assert "HTTPS_PROXY" in request.environment_allowlist
    assert "UV_INDEX_URL" in request.environment_allowlist
    assert result.installed is True
    assert runtime.verified == [result.executable]
    assert runtime.installed_versions() == ("0.1.1",)


@pytest.mark.anyio
async def test_runtime_upgrade_does_not_copy_provider_packages_into_control_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    runtime_root = tmp_path / "runtime"
    monkeypatch.setattr(
        "flameox.application.capabilities.user_data_path",
        lambda *_args, **_kwargs: runtime_root,
    )

    setup = CapabilitySetup(
        extra=CapabilityExtra.TORCH,
        requirement="torch>=2.7",
        next_action=manual_action(
            "Choose an idempotency key and start setup for torch.profiler.",
            suggested_action=ActionId.START_CAPABILITY_SETUP,
            missing_arguments=("idempotency_key",),
        ),
        verification_action=tool_action(
            ActionId.INSPECT_CAPABILITIES,
            adapter="torch.profiler",
        ),
    )
    available = CapabilityReport(
        adapter="torch.profiler",
        status=CapabilityStatus.AVAILABLE,
        setup=setup,
    )
    service = CapabilityService(workspace)
    monkeypatch.setattr(
        service,
        "list",
        lambda: CapabilityList(capabilities=(available,)),
    )

    prepared = service.prepare(("torch.profiler",))
    broker = RecordingBroker()
    installed = await RecordingRuntime(runtime_root, broker=broker).install("0.1.1")

    assert prepared.already_available == ("torch.profiler",)
    assert installed.installed is True
    assert broker.requests[0].argv[-1] == "flameox==0.1.1"
    assert not (runtime_root / "capabilities.json").exists()


def test_installed_version_discovery_ignores_unmanaged_directories(tmp_path: Path) -> None:
    unmanaged = tmp_path / "runtimes" / "not-a-version"
    unmanaged.mkdir(parents=True)
    (unmanaged / "runtime.json").write_text("{}")

    assert ManagedRuntime(tmp_path).installed_versions() == ()


def _patch_trace_processor_asset(
    monkeypatch: pytest.MonkeyPatch,
    expected_payload: bytes,
) -> ManagedToolAsset:
    digest = hashlib.sha256(expected_payload).hexdigest()
    asset = ManagedToolAsset(
        manifest_revision="test-manifest",
        tool="perfetto-trace-processor",
        version="v55.1",
        platform="linux",
        machine="x86_64",
        asset_name="trace_processor_shell-linux-amd64",
        url="https://downloads.example.com/trace_processor_shell",
        allowed_origins=("https://downloads.example.com",),
        sha256=digest,
        byte_length=len(expected_payload),
        max_bytes=1024 * 1024,
        executable_sha256=digest,
    )
    monkeypatch.setattr(_setup_runtime, "_trace_processor_asset", lambda: asset)
    return asset


def test_trace_processor_setup_stages_a_user_space_binary_and_updates_workspace_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    payload = b"trace-processor-binary"
    asset = _patch_trace_processor_asset(monkeypatch, payload)

    http_client = BoundedHttpClient(
        sync_transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                stream=httpx.ByteStream(payload),
            )
        )
    )
    broker = RecordingBroker()

    with http_client:
        result = install_trace_processor(workspace, broker=broker, http_client=http_client)

    assert result.installed is True
    assert result.executable.is_file()
    assert workspace.config.analysis.trace_processor_path == str(result.executable)
    assert broker.requests[0].argv[1:] == ("--version",)
    assert Path(broker.requests[0].argv[0]).parent == workspace.paths.staging
    assert broker.requests[0].environment_allowlist == ()
    assert result.asset_sha256 == asset.sha256
    assert result.executable_sha256 == asset.executable_sha256


def test_trace_processor_rejects_substituted_version_printer_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    substituted = b"#!/bin/sh\necho 55.1\n"
    _patch_trace_processor_asset(monkeypatch, b"x" * len(substituted))
    http_client = BoundedHttpClient(
        sync_transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                stream=httpx.ByteStream(substituted),
            )
        )
    )
    broker = RecordingBroker()

    with http_client, pytest.raises(DomainError) as caught:
        install_trace_processor(workspace, broker=broker, http_client=http_client)

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert broker.requests == []


def test_trace_processor_verification_cancellation_uses_broker_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    target = workspace.paths.root / "tools" / "trace_processor_shell"
    target.parent.mkdir(parents=True)
    target.write_text("placeholder")
    target.chmod(0o755)
    asset = _patch_trace_processor_asset(monkeypatch, b"placeholder")
    receipt = build_managed_tool_receipt(
        asset,
        target,
        trusted_root=target.parent,
    )
    write_managed_tool_receipt(target.parent / "trace-processor-receipt.json", receipt)
    broker = BlockingBroker()
    cancel_event = threading.Event()
    failures: list[BaseException] = []

    def run() -> None:
        try:
            install_trace_processor(
                workspace,
                cancel_event=cancel_event,
                broker=broker,
            )
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
