from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flameox import __version__


def _next_patch_version(version: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


@pytest.mark.skipif(os.name != "posix", reason="npx e2e fixture uses POSIX executable scripts")
@pytest.mark.process
@pytest.mark.serial
@pytest.mark.parametrize("active_version", ("0.1.3", _next_patch_version(__version__)))
def test_npx_upgrade_activates_or_preserves_the_managed_runtime(
    tmp_path: Path,
    active_version: str,
) -> None:
    home = tmp_path / "home"
    data = tmp_path / "data"
    project = tmp_path / "npx-project"
    package_dist = tmp_path / "package-dist"
    home.mkdir()
    project.mkdir()
    package_dist.mkdir()
    (home / ".claude").mkdir()

    active_executable = data / "runtimes" / active_version / "bin" / "flameox"
    if active_version == "0.1.3":
        active_executable.parent.mkdir(parents=True)
        active_executable.write_text("#!/bin/sh\nprintf '%s\\n' '0.1.3'\n")
        active_executable.chmod(0o700)
        (active_executable.parent.parent / "runtime.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "distribution": "flameox",
                    "version": active_version,
                    "executable": str(active_executable),
                }
            )
            + "\n"
        )
    data.mkdir(exist_ok=True)
    (data / "install.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_version": active_version,
                "executable": str(active_executable),
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
                        "command": str(active_executable),
                        "args": ["mcp", "serve", "--project-root", "."],
                    }
                }
            }
        )
        + "\n"
    )

    npm_package = Path(__file__).parents[2] / "npm"
    npm_env = {
        **os.environ,
        "npm_config_cache": str(tmp_path / "npm-cache"),
        "npm_config_fund": "false",
        "npm_config_update_notifier": "false",
    }
    packed = subprocess.run(
        ["npm", "pack", str(npm_package), "--pack-destination", str(package_dist), "--json"],
        cwd=tmp_path,
        env=npm_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert packed.returncode == 0, packed.stdout + packed.stderr
    package_tarball = package_dist / json.loads(packed.stdout)[0]["filename"]

    dependency_fixture = tmp_path / "jsonc-parser-fixture"
    dependency_fixture.mkdir()
    (dependency_fixture / "package.json").write_text(
        json.dumps({"name": "jsonc-parser", "version": "3.3.1", "main": "index.js"}) + "\n"
    )
    (dependency_fixture / "index.js").write_text("module.exports = {};\n")
    dependency_pack = subprocess.run(
        [
            "npm",
            "pack",
            str(dependency_fixture),
            "--pack-destination",
            str(package_dist),
            "--json",
        ],
        cwd=tmp_path,
        env=npm_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert dependency_pack.returncode == 0, dependency_pack.stdout + dependency_pack.stderr
    dependency_tarball = package_dist / json.loads(dependency_pack.stdout)[0]["filename"]

    installed = subprocess.run(
        [
            "npm",
            "install",
            "--prefix",
            str(project),
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            "--offline",
            str(package_tarball),
            str(dependency_tarball),
        ],
        cwd=tmp_path,
        env=npm_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr

    python_cli = Path(sys.executable).with_name("flameox")
    fake_uvx = tmp_path / "uvx"
    fake_uvx.write_text(
        f"""#!{sys.executable}
import os
import sys

arguments = sys.argv[1:]
handoff = max(index for index, argument in enumerate(arguments) if argument == "flameox")
os.execv({str(python_cli)!r}, [{str(python_cli)!r}, *arguments[handoff + 1:]])
"""
    )
    fake_uvx.chmod(0o700)

    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

if sys.argv[1:3] != ["tool", "install"]:
    raise SystemExit("unexpected uv invocation")
version = next(
    argument.split("==", 1)[1]
    for argument in sys.argv
    if argument.startswith("flameox==")
)
Path(os.environ["FLAMEOX_UV_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))
runtime = Path(os.environ["UV_TOOL_BIN_DIR"]) / "flameox"
runtime.parent.mkdir(parents=True, exist_ok=True)
runtime.write_text(
    "#!{sys.executable}\\n"
    "import os, sys\\n"
    "if sys.argv[1:] == ['--version']:\\n"
    "    print(" + repr(version) + ")\\n"
    "else:\\n"
    "    os.execv({str(python_cli)!r}, [{str(python_cli)!r}, *sys.argv[1:]])\\n"
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
            "FLAMEOX_UV_CAPTURE": str(tmp_path / "uv-arguments.json"),
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Managed runtime ready. Starting flameox setup..." in result.stderr

    manifest = json.loads((data / "install.json").read_text())
    expected_version = __version__ if active_version == "0.1.3" else active_version
    current_executable = data / "runtimes" / expected_version / "bin" / "flameox"
    assert manifest == {
        "active_version": expected_version,
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
    assert version.stdout.strip() == expected_version
    assert "node_modules" not in json.loads(config.read_text())["mcpServers"]["flameox"]["command"]
    assert json.loads((tmp_path / "uv-arguments.json").read_text())[-1] == (
        f"flameox=={expected_version}"
    )
