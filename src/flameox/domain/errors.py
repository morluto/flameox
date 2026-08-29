from __future__ import annotations

from enum import StrEnum
from typing import Any

from flameox.action_graph import ActionId, NextAction, manual_action


class ErrorCode(StrEnum):
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    WORKSPACE_NOT_FOUND = "WORKSPACE_NOT_FOUND"
    WORKSPACE_INVALID = "WORKSPACE_INVALID"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    INVALID_CAPTURE_PLAN = "INVALID_CAPTURE_PLAN"
    PLAN_ID_MISMATCH = "PLAN_ID_MISMATCH"
    PLAN_TOKEN_UNKNOWN = "PLAN_TOKEN_UNKNOWN"
    PLAN_TOKEN_EXPIRED = "PLAN_TOKEN_EXPIRED"
    PLAN_TOKEN_CONSUMED = "PLAN_TOKEN_CONSUMED"
    EXECUTION_REFUSED = "EXECUTION_REFUSED"
    PROCESS_FAILED = "PROCESS_FAILED"
    PROCESS_TIMEOUT = "PROCESS_TIMEOUT"
    PROCESS_CANCELLED = "PROCESS_CANCELLED"
    ARTIFACT_TOO_LARGE = "ARTIFACT_TOO_LARGE"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
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


def missing_artifact_input(
    *,
    run_id: str,
    requirement: str,
    artifact_kinds: tuple[str, ...],
    capture_adapters: tuple[str, ...],
    import_producers: tuple[str, ...] = (),
) -> DomainError:
    missing_arguments: tuple[str, ...]
    if capture_adapters:
        instruction = (
            "First plan a capture with one of the reported compatible adapters, then start "
            "that plan and extract the returned run; alternatively import supported evidence."
        )
        suggested_action = ActionId.START_DETACHED_CAPTURE
        missing_arguments = ("plan_token", "idempotency_key")
    else:
        instruction = "Import a supported artifact, then extract the returned run."
        suggested_action = ActionId.IMPORT_ARTIFACT
        missing_arguments = ("path", "kind", "sensitivity")
    return DomainError(
        ErrorCode.ARTIFACT_NOT_FOUND,
        f"The run contains no {requirement} artifact.",
        run_id=run_id,
        details={
            "missing_entity": "artifact_input",
            "required_artifact_kinds": artifact_kinds,
            "compatible_capture_adapters": capture_adapters,
            "compatible_import_producers": import_producers,
        },
        remediation=(
            "Capture a new run with a compatible adapter or import a supported artifact, "
            "then extract that returned run.",
        ),
        next_action=manual_action(
            instruction,
            suggested_action=suggested_action,
            missing_arguments=missing_arguments,
        ),
    )
