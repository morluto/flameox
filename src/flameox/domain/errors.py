from __future__ import annotations

from enum import StrEnum
from typing import Any

from flameox.action_graph import NextAction


class ErrorCode(StrEnum):
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    WORKSPACE_NOT_FOUND = "WORKSPACE_NOT_FOUND"
    WORKSPACE_INVALID = "WORKSPACE_INVALID"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    INVALID_CAPTURE_PLAN = "INVALID_CAPTURE_PLAN"
    EXECUTION_REFUSED = "EXECUTION_REFUSED"
    PROCESS_FAILED = "PROCESS_FAILED"
    PROCESS_TIMEOUT = "PROCESS_TIMEOUT"
    PROCESS_CANCELLED = "PROCESS_CANCELLED"
    ARTIFACT_TOO_LARGE = "ARTIFACT_TOO_LARGE"
    ARTIFACT_INTEGRITY_FAILED = "ARTIFACT_INTEGRITY_FAILED"
    ARTIFACT_PARSE_FAILED = "ARTIFACT_PARSE_FAILED"
    ADAPTER_INCOMPATIBLE = "ADAPTER_INCOMPATIBLE"
    EVIDENCE_SCHEMA_MISMATCH = "EVIDENCE_SCHEMA_MISMATCH"
    COMPARISON_INVALID = "COMPARISON_INVALID"
    QUERY_BUDGET_EXCEEDED = "QUERY_BUDGET_EXCEEDED"
    STORAGE_QUOTA_EXCEEDED = "STORAGE_QUOTA_EXCEEDED"
    WRITE_LOCK_TIMEOUT = "WRITE_LOCK_TIMEOUT"
    LOCK_ORDER_VIOLATION = "LOCK_ORDER_VIOLATION"
    SENSITIVE_ARTIFACT_REFUSED = "SENSITIVE_ARTIFACT_REFUSED"
    REVISION_CONFLICT = "REVISION_CONFLICT"
    STALE_CURSOR = "STALE_CURSOR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class DomainError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        remediation: tuple[str, ...] = (),
        run_id: str | None = None,
        next_action: NextAction | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = dict(details or {})
        self.remediation = remediation
        self.run_id = run_id
        forbidden_recovery_fields = {"next_tool", "next_arguments"} & self.details.keys()
        if forbidden_recovery_fields:
            raise ValueError(
                "Legacy recovery fields are not accepted; pass a validated next_action "
                f"instead: {', '.join(sorted(forbidden_recovery_fields))}."
            )
        self.next_action = next_action

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
            "remediation": list(self.remediation),
            "run_id": self.run_id,
            "next_action": (
                self.next_action.model_dump(mode="json") if self.next_action is not None else None
            ),
        }
