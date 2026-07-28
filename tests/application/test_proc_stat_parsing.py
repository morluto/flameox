from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from flameox.application.capture import CaptureService
from flameox.storage import Workspace


def _stat_line(pid: int, comm: str, starttime: int) -> str:
    """Build a minimal valid /proc/[pid]/stat line.

    Per proc(5), field 1 is pid, field 2 is ``(comm)``, and the remaining
    fields are space-separated. We only need fields up to 22 (starttime);
    the rest are filled with zeros.
    """
    rest = " ".join("0" for _ in range(19))
    return f"{pid} ({comm}) {rest} {starttime} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"


def test_lease_parses_starttime_for_multi_word_comm() -> None:
    """Regression test for the /proc/[pid]/stat parsing bug.

    A process whose name (field 2, ``comm``) contains spaces must still
    yield the correct ``starttime`` (field 22). The old code split the
    whole line on whitespace and indexed [21], which was wrong whenever
    ``comm`` had a space.
    """
    import tempfile

    workspace = Workspace.initialize(Path(tempfile.mkdtemp()))
    service = CaptureService(workspace)
    pid = 12345
    expected_starttime = 118
    stat_content = _stat_line(pid, "Web Content", expected_starttime).encode()

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if str(self) == "/proc/sys/kernel/random/boot_id":
            return "boot-id-1234\n"
        raise FileNotFoundError(str(self))

    def fake_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        if path == f"/proc/{pid}/stat":
            return 999
        raise FileNotFoundError(path)

    def fake_read(fd: int, n: int) -> bytes:
        if fd == 999:
            return stat_content
        raise OSError(fd)

    with (
        patch.object(Path, "read_text", fake_read_text),
        patch("os.open", fake_open),
        patch("os.read", fake_read),
        patch("os.close", lambda fd: 0),
    ):
        lease = service._lease(pid)

    assert lease is not None
    assert lease.process_start_identity == str(expected_starttime)
    assert lease.process_id == pid


def test_lease_parses_starttime_for_single_word_comm() -> None:
    """The fix must not regress the simple single-word ``comm`` case."""
    import tempfile

    workspace = Workspace.initialize(Path(tempfile.mkdtemp()))
    service = CaptureService(workspace)
    pid = 6789
    expected_starttime = 999
    stat_content = _stat_line(pid, "python", expected_starttime).encode()

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if str(self) == "/proc/sys/kernel/random/boot_id":
            return "boot-id-5678\n"
        raise FileNotFoundError(str(self))

    def fake_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        if path == f"/proc/{pid}/stat":
            return 888
        raise FileNotFoundError(path)

    def fake_read(fd: int, n: int) -> bytes:
        if fd == 888:
            return stat_content
        raise OSError(fd)

    with (
        patch.object(Path, "read_text", fake_read_text),
        patch("os.open", fake_open),
        patch("os.read", fake_read),
        patch("os.close", lambda fd: 0),
    ):
        lease = service._lease(pid)

    assert lease is not None
    assert lease.process_start_identity == str(expected_starttime)


def test_lease_returns_none_when_proc_missing() -> None:
    """When /proc/[pid]/stat is absent, _lease returns None (process gone)."""
    import tempfile

    workspace = Workspace.initialize(Path(tempfile.mkdtemp()))
    service = CaptureService(workspace)

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise FileNotFoundError(str(self))

    def fake_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        raise FileNotFoundError(path)

    with (
        patch.object(Path, "read_text", fake_read_text),
        patch("os.open", fake_open),
        patch("os.read", lambda fd, n: b""),
        patch("os.close", lambda fd: 0),
    ):
        lease = service._lease(99999)

    assert lease is None
