"""Normalize request-validation failures into actionable MCP results."""

from __future__ import annotations

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import best_match
from pydantic import ValidationError

from flameox.mcp.result_contracts import AdjustRequestAction, ToolFailureEnvelope
from flameox.validation import validation_failure


def normalize_schema_error(error: JsonSchemaValidationError) -> ToolFailureEnvelope:
    """Project schema failures without including rejected values or private paths."""
    issue = best_match(error.context) if error.context else error
    if issue is None:
        issue = error
    location = [part for part in issue.absolute_path if isinstance(part, (str, int))]
    constraint = str(issue.validator or "input")
    expected = issue.validator_value
    accepted: list[str] | None = None
    if (
        constraint == "enum"
        and isinstance(expected, list)
        and all(isinstance(item, str) for item in expected)
    ):
        accepted = expected
        message = "Value must be one of the advertised values."
    elif constraint == "type" and isinstance(expected, str):
        message = f"Input must match the advertised {expected} type."
    else:
        message = f"Input must match the advertised {constraint} constraint."
    next_action = AdjustRequestAction(
        kind="adjust_request",
        field_path=location or None,
        message=message,
    )
    return ToolFailureEnvelope(
        code="INVALID_REQUEST",
        message=message,
        field_path=location,
        accepted_values=accepted,
        next_action=next_action,
        details={"error_type": f"schema_{constraint}"},
    )


def normalize_validation_error(error: ValidationError) -> ToolFailureEnvelope:
    value = validation_failure(error)
    value["code"] = "INVALID_REQUEST"
    value["next_action"] = AdjustRequestAction(
        kind="adjust_request",
        field_path=value["field_path"] or None,
        message=value["message"],
    )
    return ToolFailureEnvelope.model_validate(value)
