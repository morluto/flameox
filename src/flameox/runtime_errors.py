from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    UNAVAILABLE_CAPABILITY = "UNAVAILABLE_CAPABILITY"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    MISSING_OR_CHANGED_INPUT = "MISSING_OR_CHANGED_INPUT"
    DECODE_FAILURE = "DECODE_FAILURE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


class DomainError(Exception):
    """Typed internal failure translated at the CLI/MCP boundary."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        remediation: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = dict(details or {})
        self.remediation = remediation
