from __future__ import annotations

from enum import StrEnum

from flameox.config import ContainmentPolicy


class ExecutionPolicy(StrEnum):
    """Named trust policies selected by transport composition roots."""

    TRUSTED_LOCAL = "trusted_local"
    APPROVED_AGENT = "approved_agent"

    def requires_containment(self, configured_mode: ContainmentPolicy | str) -> bool:
        configured_mode = ContainmentPolicy(configured_mode)
        return (
            self is ExecutionPolicy.APPROVED_AGENT
            and configured_mode is ContainmentPolicy.REQUIRED_FOR_MCP
        )
