from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from flameox.adapters import ManagedRuntime


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


def test_installed_version_discovery_ignores_unmanaged_directories(tmp_path: Path) -> None:
    unmanaged = tmp_path / "runtimes" / "not-a-version"
    unmanaged.mkdir(parents=True)
    (unmanaged / "runtime.json").write_text("{}")

    assert ManagedRuntime(tmp_path).installed_versions() == ()
