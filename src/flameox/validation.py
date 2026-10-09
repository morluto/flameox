"""Safe request-validation diagnostics shared by CLI and MCP transports."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from flameox.runtime_contracts import CAPTURE_PROVIDER_CONTRACTS


def validation_failure(error: ValidationError) -> dict[str, Any]:
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
    location = list(issue["loc"])
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
    return {
        "code": "INVALID_INPUT",
        "message": message,
        "field_path": location,
        "accepted_values": accepted,
        "details": {"error_type": str(issue["type"])},
    }
