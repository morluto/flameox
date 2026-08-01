from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flameox import __version__


@pytest.mark.skipif(os.name != "posix", reason="npx e2e fixture uses POSIX executable scripts")
@pytest.mark.process
@pytest.mark.serial
def test_npx_upgrade_activates_the_matching_managed_runtime(tmp_path: Path) -> None:
    home = tmp_path / "home"
    data = tmp_path / "data"
    project = tmp_path / "npx-project"
    home.mkdir()
    project.mkdir()
    (home / ".claude").mkdir()

    old_version = "0.1.3"
    old_executable = data / "runtimes" / old_version / "bin" / "flameox"
    old_executable.parent.mkdir(parents=True)
    old_executable.write_text("#!/bin/sh\nprintf '%s\\n' '0.1.3'\n")
    old_executable.chmod(0o700)
    (old_executable.parent.parent / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "distribution": "flameox",
                "version": old_version,
                "executable": str(old_executable),
            }
        )
        + "\n"
    )
    data.mkdir(exist_ok=True)
    (data / "install.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_version": old_version,
                "executable": str(old_executable),
            }
        )
        + "\n"
    )
    config = home / ".claude.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "flameox": {
                        "command": str(old_executable),
                        "args": ["mcp", "serve", "--project-root", "."],
                    }
                }
            }
        )
        + "\n"
    )

    npm_package = Path(__file__).parents[2] / "npm"
    node_modules = project / "node_modules"
    (node_modules / ".bin").mkdir(parents=True)
    (node_modules / "flameox").symlink_to(npm_package, target_is_directory=True)
    (node_modules / ".bin" / "flameox").symlink_to("../flameox/bin/flameox.cjs")

    python_cli = Path(sys.executable).with_name("flameox")
    fake_uvx = tmp_path / "uvx"
    fake_uvx.write_text(
        f"""#!{sys.executable}
import os
import sys

arguments = sys.argv[1:]
handoff = arguments.index("flameox")
os.execv({str(python_cli)!r}, [{str(python_cli)!r}, *arguments[handoff + 1:]])
"""
    )
    fake_uvx.chmod(0o700)

    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        f"""#!{sys.executable}
import os
from pathlib import Path
import sys

if sys.argv[1:3] != ["tool", "install"]:
    raise SystemExit("unexpected uv invocation")
runtime = Path(os.environ["UV_TOOL_BIN_DIR"]) / "flameox"
runtime.parent.mkdir(parents=True, exist_ok=True)
runtime.write_text(
    "#!{sys.executable}\\n"
    "import os, sys\\n"
    "os.execv({str(python_cli)!r}, [{str(python_cli)!r}, *sys.argv[1:]])\\n"
)
runtime.chmod(0o700)
"""
    )
    fake_uv.chmod(0o700)

    result = subprocess.run(
        ["npx", "--no-install", "flameox", "upgrade"],
        cwd=project,
        env={
            **os.environ,
            "FLAMEOX_SETUP_HOME": str(home),
            "FLAMEOX_SETUP_DATA_ROOT": str(data),
            "FLAMEOX_SETUP_UV": str(fake_uv),
            "FLAMEOX_UV_EXECUTABLE": str(fake_uvx),
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Managed runtime ready. Starting flameox setup..." in result.stderr

    manifest = json.loads((data / "install.json").read_text())
    current_executable = data / "runtimes" / __version__ / "bin" / "flameox"
    assert manifest == {
        "active_version": __version__,
        "executable": str(current_executable),
        "schema_version": 1,
    }
    assert json.loads(config.read_text())["mcpServers"]["flameox"] == {
        "command": str(current_executable),
        "args": ["mcp", "serve", "--project-root", "."],
    }
    version = subprocess.run(
        [str(current_executable), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert version.returncode == 0
    assert version.stdout.strip() == __version__
