from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from flameox.domain import DomainError, ErrorCode, OracleReceiptV1

MAX_ORACLE_RECEIPT_BYTES = 64 * 1024


def parse_oracle_receipt(payload: bytes) -> OracleReceiptV1:
    if len(payload) > MAX_ORACLE_RECEIPT_BYTES:
        raise DomainError(
            ErrorCode.WORKSPACE_INVALID,
            f"Oracle receipt exceeds the {MAX_ORACLE_RECEIPT_BYTES}-byte limit.",
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DomainError(ErrorCode.WORKSPACE_INVALID, "Oracle receipt is not UTF-8 JSON.") from exc

    decoder = json.JSONDecoder(
        parse_constant=lambda value: _raise_invalid_number(value),
        object_pairs_hook=_unique_object,
    )
    try:
        text = text.lstrip()
        value, end = decoder.raw_decode(text)
        if text[end:].strip():
            raise ValueError("trailing JSON values are not allowed")
        if not isinstance(value, dict):
            raise ValueError("receipt must be a JSON object")
        return OracleReceiptV1.model_validate(value)
    except (RecursionError, ValueError, ValidationError) as exc:
        raise DomainError(
            ErrorCode.WORKSPACE_INVALID,
            f"Oracle receipt is invalid: {exc}",
        ) from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _raise_invalid_number(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")
