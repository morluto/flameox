"""Normalize request-validation failures into actionable MCP results."""

from __future__ import annotations

from pydantic import ValidationError

from flameox.mcp.result_contracts import ToolFailureEnvelope


def normalize_validation_error(
    error: ValidationError, *, prefix: tuple[str | int, ...] = ()
) -> ToolFailureEnvelope:
    issue = error.errors(include_url=False, include_context=True)[0]
    context = issue.get("ctx") or {}
    expected = context.get("expected")
    accepted: list[str] | None = None
    if isinstance(expected, str):
        accepted = [item.strip(" '") for item in expected.split(" or ")]
    message = str(issue["msg"])
    marker = "accepted values: "
    if accepted is None and marker in message:
        accepted = [item.strip() for item in message.split(marker, 1)[1].split(",")]
    location = [*prefix, *issue["loc"]]
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
    return ToolFailureEnvelope(
        code="INVALID_REQUEST",
        message=message,
        field_path=location,
        accepted_values=accepted,
        details={"error_type": str(issue["type"])},
    )
