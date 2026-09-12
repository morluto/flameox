from __future__ import annotations

from pathlib import Path

import anyio
import psutil


def process_is_alive(pid: int) -> bool:
    """Treat a reaped or zombie process as stopped; surface permission failures."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


async def wait_for_pid_file(path: Path) -> int:
    """Wait for a workload's ready signal, including completion of its PID write."""
    with anyio.fail_after(10):
        while True:
            try:
                value = path.read_text().strip()
            except FileNotFoundError:
                value = ""
            if value:
                return int(value)
            await anyio.sleep(0.01)
