from __future__ import annotations

from pathlib import Path

import pytest

from flameox.setup import SetupFailure, provider_install_command


def test_provider_install_command_replaces_tool_with_declared_extra_set() -> None:
    command = provider_install_command(["perfetto", "memray", "otlp"], uv=Path("/usr/bin/uv"))

    assert command == [
        "/usr/bin/uv",
        "tool",
        "install",
        "--force",
        "--python",
        "3.12",
        "--prerelease",
        "allow",
        "flameox[memory,trace]==0.2.0",
    ]


def test_system_provider_has_guidance_but_no_python_install() -> None:
    assert provider_install_command(["nsight-compute"], uv=Path("uv")) == []


def test_unknown_provider_is_rejected_before_installation() -> None:
    with pytest.raises(SetupFailure, match="Unknown provider"):
        provider_install_command(["mystery"], uv=Path("uv"))
