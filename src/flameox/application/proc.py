from __future__ import annotations


def parse_proc_stat_starttime(stat_text: str) -> str:
    """Return Linux ``/proc/[pid]/stat`` field 22 without splitting ``comm``."""
    try:
        comm_start = stat_text.index("(")
        comm_end = stat_text.rindex(")")
    except ValueError as exc:
        raise ValueError("Malformed /proc stat process name") from exc
    if comm_end < comm_start:
        raise ValueError("Malformed /proc stat process name")
    fields_after_comm = stat_text[comm_end + 1 :].split()
    if len(fields_after_comm) < 20:
        raise ValueError(
            f"Insufficient /proc stat fields after process name: {len(fields_after_comm)}"
        )
    return fields_after_comm[19]
