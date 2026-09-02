from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from flameox import __version__
from flameox.setup import (
    SetupFailure,
    managed_tool_extras,
    mcp_launcher,
    prepare_providers,
    provider_install_command,
)


def test_mcp_launcher_pins_python_release_and_provider_extras() -> None:
    command, args = mcp_launcher(["memray", "nsight-compute", "py-spy"])

    assert command == "uvx"
    assert args[:5] == [
        "--python",
        "3.12",
        "--from",
        f"flameox[cpu,memory]=={__version__}",
        "flameox",
    ]


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


def test_system_only_preparation_does_not_require_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: None)

    prepared = prepare_providers(["nsight-compute"])

    assert prepared.configured_managed_providers == []
    assert [item.provider_id for item in prepared.external_requirements] == ["nsight-compute"]
    assert prepared.install_command == []
    assert prepared.changed is False


def test_provider_install_is_a_noop_when_requested_extra_is_already_managed() -> None:
    assert (
        provider_install_command(
            ["py-spy"],
            uv=Path("/usr/bin/uv"),
            installed_extras={"cpu", "memory"},
        )
        == []
    )


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
        (tool / "uv-receipt.toml").write_text(
            '[tool]\nrequirements = [{ name = "flameox", extras = ["cpu", "memory"] }]\n'
        )
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr("flameox.setup.subprocess.run", run)

    installed = prepare_providers(["memray"])

    assert calls[-1][-1] == f"flameox[cpu,memory]=={__version__}"
    assert installed.configured_managed_providers == ["memray", "py-spy"]
    assert installed.changed is True


def test_concurrent_preparation_retains_both_provider_extra_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    both_located = threading.Event()
    calls_guard = threading.Lock()
    tool_dir_calls = 0
    install_requirements: list[str] = []

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        nonlocal tool_dir_calls
        if command[1:] == ["tool", "dir"]:
            with calls_guard:
                tool_dir_calls += 1
                if tool_dir_calls == 2:
                    both_located.set()
            return CompletedProcess(command, 0, f"{tool_directory}\n", "")

        assert both_located.wait(timeout=2)
        requirement = command[-1]
        install_requirements.append(requirement)
        extras = requirement.partition("[")[2].partition("]")[0].split(",")
        receipt = tool_directory / "flameox" / "uv-receipt.toml"
        receipt.parent.mkdir(exist_ok=True)
        extras_toml = ", ".join(f'"{extra}"' for extra in extras)
        receipt.write_text(
            f'[tool]\nrequirements = [{{ name = "flameox", extras = [{extras_toml}] }}]\n'
        )
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr("flameox.setup.subprocess.run", run)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(prepare_providers, [["memray"], ["otlp"]]))

    assert len(install_requirements) == 2
    assert any("flameox[memory,trace]" in requirement for requirement in install_requirements)
    assert any(
        set(result.configured_managed_providers) == {"memray", "otlp", "perfetto"}
        for result in results
    )


def test_unknown_provider_is_rejected_before_installation() -> None:
    with pytest.raises(SetupFailure, match="Unknown provider"):
        provider_install_command(["mystery"], uv=Path("uv"))
