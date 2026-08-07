from __future__ import annotations

from pathlib import Path

PROC_ROOT = Path("/proc")
MAX_STAT_BYTES = 8192


def read_boot_id() -> str:
    boot_id = (PROC_ROOT / "sys/kernel/random/boot_id").read_text().strip()
    if not boot_id:
        raise ValueError("The kernel boot identifier is empty.")
    return boot_id


def parse_proc_stat_start_identity(stat_text: str) -> str:
    """Return field 22 (starttime) from a Linux ``/proc/<pid>/stat`` record."""
    comm_start = stat_text.find("(")
    comm_end = stat_text.rfind(")")
    if comm_start < 0 or comm_end < comm_start:
        raise ValueError("The process stat record has no complete comm field.")

    # The suffix begins at field 3. Field 22 is therefore index 19 here.
    fields = stat_text[comm_end + 1 :].split()
    if len(fields) < 20:
        raise ValueError("The process stat record ends before the starttime field.")
    return fields[19]


def read_proc_stat_start_identity(process_id: int) -> str:
    stat_path = PROC_ROOT / str(process_id) / "stat"
    with stat_path.open("rb") as stream:
        stat_raw = stream.read(MAX_STAT_BYTES + 1)
    if len(stat_raw) > MAX_STAT_BYTES:
        raise ValueError(f"The process stat record exceeds {MAX_STAT_BYTES} bytes.")
    return parse_proc_stat_start_identity(stat_raw.decode("utf-8", errors="replace"))


__all__ = [
    "MAX_STAT_BYTES",
    "parse_proc_stat_start_identity",
    "read_boot_id",
    "read_proc_stat_start_identity",
]
