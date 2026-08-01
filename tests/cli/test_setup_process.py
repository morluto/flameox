from __future__ import annotations

import os
import pty
import selectors
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix", reason="PTY smoke test requires POSIX")
@pytest.mark.process
@pytest.mark.serial
def test_npm_bootstrap_pty_reaches_first_python_prompt(tmp_path: Path) -> None:
    fake_uvx = tmp_path / "uvx"
    flameox = Path(sys.executable).with_name("flameox")
    fake_uvx.write_text(
        f"""#!{sys.executable}
import os, sys
arguments = sys.argv[1:]
handoff = max(index for index, argument in enumerate(arguments) if argument == "flameox")
os.execv({str(flameox)!r}, [{str(flameox)!r}, *arguments[handoff + 1:]])
"""
    )
    fake_uvx.chmod(0o700)
    bootstrap = Path(__file__).parents[2] / "npm" / "bin" / "flameox.cjs"
    master, slave = pty.openpty()
    process = subprocess.Popen(
        ["node", str(bootstrap), "setup"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={
            **os.environ,
            "FLAMEOX_UV_EXECUTABLE": str(fake_uvx),
            "FLAMEOX_SETUP_HOME": str(tmp_path / "home"),
            "FLAMEOX_SETUP_DATA_ROOT": str(tmp_path / "data"),
        },
        close_fds=True,
    )
    os.close(slave)
    selector = selectors.DefaultSelector()
    selector.register(master, selectors.EVENT_READ)
    transcript = bytearray()
    cursor_queries_answered = 0
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline and b"Select MCP clients to connect" not in transcript:
            if selector.select(timeout=0.25):
                transcript.extend(os.read(master, 4_096))
                cursor_queries = transcript.count(b"\x1b[6n")
                while cursor_queries_answered < cursor_queries:
                    os.write(master, b"\x1b[1;1R")
                    cursor_queries_answered += 1
    finally:
        process.terminate()
        process.wait(timeout=5)
        selector.close()
        os.close(master)

    output = transcript.decode(errors="replace")
    assert "cached managed Python runtime" in output
    assert "Managed runtime ready. Starting flameox setup..." in output
    assert "Select MCP clients to connect" in output
