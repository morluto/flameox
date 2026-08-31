from __future__ import annotations

import hashlib
from typing import Any, cast

import rfc8785
from pydantic import BaseModel, JsonValue, TypeAdapter

_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_I_JSON_INTEGER_MIN = -(2**53) + 1
_I_JSON_INTEGER_MAX = 2**53 - 1


def canonical_bytes(value: object) -> bytes:
    """Encode a JSON value with the one canonical representation used for identities."""
    return rfc8785.dumps(cast(Any, value))


def digest_model(value: object, *, projection: str = "flameox.identity/v1") -> str:
    """Build a projection-bound identity without losing wide integer values."""
    if isinstance(value, BaseModel):
        normalized = value.model_dump(mode="json", exclude_none=False)
    else:
        normalized = _JSON_VALUE.dump_python(cast(JsonValue, value), mode="json", warnings=False)
    payload = {
        "algorithm": "rfc8785-sha256-v1",
        "projection": projection,
        "value": _normalize_wide_integers(normalized),
    }
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _normalize_wide_integers(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if _I_JSON_INTEGER_MIN <= value <= _I_JSON_INTEGER_MAX:
            return value
        return {"$flameox.integer": str(value)}
    if isinstance(value, list):
        return [_normalize_wide_integers(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_wide_integers(item) for key, item in value.items()}
    return value
