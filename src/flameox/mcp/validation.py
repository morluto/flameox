"""Normalize request-validation failures into actionable MCP results."""

from __future__ import annotations

import re

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import best_match
from pydantic import ValidationError

from flameox.mcp.result_contracts import AdjustRequestAction, ToolFailureEnvelope
from flameox.runtime_contracts import CAPTURE_PROVIDER_CONTRACTS


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


def normalize_validation_error(
    error: ValidationError, *, prefix: tuple[str | int, ...] = ()
) -> ToolFailureEnvelope:
    issue = error.errors(include_url=False, include_context=True)[0]
    context = issue.get("ctx") or {}
    expected = context.get("expected")
    accepted: list[str] | None = None
    if isinstance(expected, str):
        accepted = re.findall(r"'([^']*)'", expected) or [expected]
    message = str(issue["msg"])
    marker = "accepted values: "
    if accepted is None and marker in message:
        accepted = [item.strip() for item in message.split(marker, 1)[1].split(",")]
    location = [*prefix, *issue["loc"]]
    # Pydantic inserts a selected union tag in loc. It is not an object key.
    if len(location) >= 2 and location[0] == "provider":
        if location[1] in CAPTURE_PROVIDER_CONTRACTS:
            location.pop(1)
    elif (
        len(location) >= 3
        and location[0] == "sources"
        and isinstance(location[1], int)
        and location[2] in {"path", "evidence"}
    ):
        location.pop(2)
    discriminator = context.get("discriminator")
    if issue["type"] in {"union_tag_not_found", "union_tag_invalid"} and isinstance(
        discriminator, str
    ):
        location.append(discriminator.strip("'"))
        expected_tags = context.get("expected_tags")
        if isinstance(expected_tags, str):
            accepted = [item.strip(" '") for item in expected_tags.split(",")]
        elif discriminator.strip("'") == "mode":
            accepted = ["list", "get"]
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
        details={"error_type": str(issue["type"])},
    )
