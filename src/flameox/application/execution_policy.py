from __future__ import annotations

from enum import StrEnum


class ExecutionPolicy(StrEnum):
    """Named trust policies selected by transport composition roots."""

    TRUSTED_LOCAL = "trusted_local"
    APPROVED_AGENT = "approved_agent"

    def requires_containment(self, configured_mode: str) -> bool:
        return self is ExecutionPolicy.APPROVED_AGENT and configured_mode == "required_for_mcp"
