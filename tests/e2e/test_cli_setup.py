from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flameox import __version__

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.process,
    pytest.mark.skipif(os.name != "posix", reason="Installer shim requires POSIX"),
]


@pytest.mark.parametrize("failure", [None, "missing", "version", "resolver", "output", "timeout"])
@pytest.mark.parametrize("provider", [None, "memray"])
def test_setup_verifies_release_before_publishing_and_settles_installer_children(
    tmp_path: Path, failure: str | None, provider: str | None
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    child_started = tmp_path / "child-started"
    child_finished = tmp_path / "child-finished"
    release_child = tmp_path / "release-child"
    configuration = tmp_path / ".cursor" / "mcp.json"
    configuration.parent.mkdir()
    original = json.dumps(
        {
            "mcpServers": {
                "flameox": {
                    "command": "custom",
                    "env": {"UV_DEFAULT_INDEX": "https://custom.invalid/simple"},
                }
            }
        }
    )
    configuration.write_text(original)
    if failure != "missing":
        uvx = bin_dir / "uvx"
        uvx.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys, time\n"
            "from pathlib import Path\n"
            f"failure = {failure!r}\n"
            "assert os.environ['UV_DEFAULT_INDEX'] == 'https://custom.invalid/simple'\n"
            "assert sys.argv[1:3] == ['--no-config', '--no-sources']\n"
            "if failure == 'timeout':\n"
            "    if os.fork() == 0:\n"
            f"        Path({str(child_started)!r}).touch()\n"
            f"        while not Path({str(release_child)!r}).exists():\n"
            "            time.sleep(0.01)\n"
            f"        Path({str(child_finished)!r}).touch()\n"
            "        sys.exit(0)\n"
            "    time.sleep(30)\n"
            "if failure == 'output':\n"
            "    sys.stderr.buffer.write(b'x' * (2 * 1024 * 1024))\n"
            "    sys.exit(0)\n"
            "if failure == 'resolver':\n"
            "    print('resolver diagnostic', file=sys.stderr)\n"
            "    sys.exit(23)\n"
            "if sys.argv[-2] == '-c':\n"
            f"    version = '0.0.0' if failure == 'version' else {__version__!r}\n"
            "    print(json.dumps({'version': version, 'extras': ['memory']}))\n"
            "else:\n"
            "    print(json.dumps({'tools': [{'name': 'cpu_hotspots'}]}))\n"
        )
        uvx.chmod(0o755)
    environment = {name: value for name, value in os.environ.items() if not name.startswith("UV_")}
    environment.update({"HOME": str(tmp_path), "PATH": str(bin_dir)})
    command = [
        str(Path(sys.executable).with_name("flameox")),
        "setup",
        "--client",
        "cursor",
        "--yes",
        "--json",
    ]
    if provider:
        command.extend(["--provider", provider])
    if failure == "timeout":
        command.extend(["--timeout-seconds", "5"])
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=25)
    if failure is None:
        assert result.returncode == 0, result.stderr
        assert json.loads(configuration.read_text())["mcpServers"]["flameox"]["command"] == "uvx"
    else:
        assert result.returncode != 0, result.stdout
        assert configuration.read_text() == original
        assert len(result.stderr) < 1100 * 1024
        if failure == "resolver":
            assert "resolver diagnostic" in result.stderr
        if failure == "timeout":
            assert child_started.exists()
            release_child.touch()
            time.sleep(1)
            assert not child_finished.exists()
