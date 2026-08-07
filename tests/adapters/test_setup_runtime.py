from __future__ import annotations

import asyncio
import os
import threading
from io import BytesIO
from pathlib import Path

import pytest

from flameox.adapters import ManagedRuntime, install_trace_processor
from flameox.application import CapabilityList
from flameox.application.capabilities import CapabilityService
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
            process=ProcessResult(exit_code=0, cleanup_complete=True),
            stdout=self.stdout,
            stderr=b"",
            resolved_executable=Path(request.argv[0]),
            containment="process_group",
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
async def test_prepared_workspace_capability_is_carried_into_runtime_upgrade(
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
        extra="torch",
        method="start_capability_setup",
        next_tool="start_capability_setup",
        requirement="torch>=2.7",
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
    assert broker.requests[0].argv[-1] == "flameox[torch]==0.1.1"


def test_installed_version_discovery_ignores_unmanaged_directories(tmp_path: Path) -> None:
    unmanaged = tmp_path / "runtimes" / "not-a-version"
    unmanaged.mkdir(parents=True)
    (unmanaged / "runtime.json").write_text("{}")

    assert ManagedRuntime(tmp_path).installed_versions() == ()


def test_trace_processor_setup_stages_a_user_space_binary_and_updates_workspace_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    monkeypatch.setattr("flameox.adapters.setup_runtime.sys.platform", "linux")
    monkeypatch.setattr("flameox.adapters.setup_runtime._machine", lambda: "x86_64")

    class Response:
        def __enter__(self) -> BytesIO:
            return BytesIO(b"trace-processor-binary")

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "flameox.adapters.setup_runtime.urllib.request.urlopen", lambda *args, **kwargs: Response()
    )
    broker = RecordingBroker()

    result = install_trace_processor(workspace, broker=broker)

    assert result.installed is True
    assert result.executable.is_file()
    assert workspace.config.analysis.trace_processor_path == str(result.executable)
    assert broker.requests[0].argv[1:] == ("--version",)
    assert Path(broker.requests[0].argv[0]).parent == workspace.paths.staging
    assert broker.requests[0].environment_allowlist == ()


def test_trace_processor_verification_cancellation_uses_broker_cleanup(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    target = workspace.paths.root / "tools" / "trace_processor_shell"
    target.parent.mkdir(parents=True)
    target.write_text("placeholder")
    target.chmod(0o755)
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
