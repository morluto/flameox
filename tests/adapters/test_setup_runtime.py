from __future__ import annotations

import os
import subprocess
from io import BytesIO
from pathlib import Path

import pytest

from flameox.adapters import ManagedRuntime, install_trace_processor
from flameox.application.capabilities import CapabilityService
from flameox.storage import Workspace


class RecordingRuntime(ManagedRuntime):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.verified: list[Path] = []

    async def verify(self, executable: Path, version: str) -> None:
        assert executable == self.executable(version)
        self.verified.append(executable)


@pytest.mark.anyio
async def test_runtime_install_uses_an_exact_isolated_uv_tool_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RecordingRuntime(tmp_path)
    recorded_command: list[str] = []
    recorded_environment: dict[str, str] = {}

    def install(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        recorded_command.extend(command)
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        recorded_environment.update(environment)
        bin_directory = Path(environment["UV_TOOL_BIN_DIR"])
        bin_directory.mkdir(parents=True)
        (bin_directory / ("flameox.exe" if os.name == "nt" else "flameox")).write_text("")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", install)

    result = await runtime.install("0.1.1")

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
    assert result.installed is True
    assert runtime.verified == [result.executable]
    assert runtime.installed_versions() == ("0.1.1",)


@pytest.mark.anyio
async def test_runtime_install_carries_prepared_capabilities_into_new_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RecordingRuntime(tmp_path)
    (tmp_path / "capabilities.json").write_text(
        '{"schema_version": 1, "extras": ["torch", "memory"]}\n'
    )
    recorded_command: list[str] = []

    def install(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded_command.extend(command)
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        bin_directory = Path(environment["UV_TOOL_BIN_DIR"])
        bin_directory.mkdir(parents=True, exist_ok=True)
        (bin_directory / "flameox").write_text("")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", install)

    await runtime.install("0.1.1")

    assert recorded_command[-1] == "flameox[memory,torch]==0.1.1"


def test_workspace_capability_setup_is_visible_to_runtime_upgrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    runtime_root = tmp_path / "runtime"
    monkeypatch.setattr(
        "flameox.application.capabilities.user_data_path",
        lambda *_args, **_kwargs: runtime_root,
    )

    CapabilityService(workspace)._record_managed_extras(("torch",))

    assert ManagedRuntime(runtime_root)._managed_extras() == ("torch",)


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
    monkeypatch.setattr(
        "flameox.adapters.setup_runtime.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            "trace_processor_shell 55.1\n",
            "",
        ),
    )

    result = install_trace_processor(workspace)

    assert result.installed is True
    assert result.executable.is_file()
    assert workspace.config.analysis.trace_processor_path == str(result.executable)
