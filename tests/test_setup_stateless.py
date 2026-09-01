from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from flameox import __version__
from flameox.setup import (
    SetupFailure,
    install_providers,
    managed_tool_extras,
    provider_install_command,
)


def test_provider_install_command_adds_declared_extras_to_managed_tool() -> None:
    command = provider_install_command(
        ["perfetto", "memray", "otlp"],
        uv=Path("/usr/bin/uv"),
        installed_extras={"cpu"},
    )

    assert command == [
        "/usr/bin/uv",
        "tool",
        "install",
        "--force",
        "--python",
        "3.12",
        "--prerelease",
        "allow",
        f"flameox[cpu,memory,trace]=={__version__}",
    ]


def test_system_provider_has_guidance_but_no_python_install() -> None:
    assert provider_install_command(["nsight-compute"], uv=Path("uv")) == []


def test_managed_tool_extras_are_read_from_uv_receipt(tmp_path: Path) -> None:
    tool = tmp_path / "flameox"
    tool.mkdir()
    (tool / "uv-receipt.toml").write_text(
        """
[tool]
requirements = [{ name = "flameox", extras = ["cpu", "memory"], specifier = "==0.2.1" }]
"""
    )

    assert managed_tool_extras(tmp_path) == {"cpu", "memory"}


def test_managed_tool_extras_support_legacy_string_receipts(tmp_path: Path) -> None:
    tool = tmp_path / "flameox"
    tool.mkdir()
    (tool / "uv-receipt.toml").write_text('[tool]\nrequirements = ["flameox[cpu,memory]==0.2.1"]\n')

    assert managed_tool_extras(tmp_path) == {"cpu", "memory"}


def test_provider_install_retains_existing_managed_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tmp_path / "tools" / "flameox"
    tool.mkdir(parents=True)
    (tool / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "flameox", extras = ["cpu"] }]\n'
    )
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        if command[1:] == ["tool", "dir"]:
            return CompletedProcess(command, 0, f"{tmp_path / 'tools'}\n", "")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr("flameox.setup.subprocess.run", run)

    installed = install_providers(["memray"])

    assert calls[-1][-1] == f"flameox[cpu,memory]=={__version__}"
    assert installed.providers == ["memray", "py-spy"]


def test_unknown_provider_is_rejected_before_installation() -> None:
    with pytest.raises(SetupFailure, match="Unknown provider"):
        provider_install_command(["mystery"], uv=Path("uv"))
