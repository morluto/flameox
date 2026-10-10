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


def sha256_id(hex_digest: str) -> str:
    """Format a SHA-256 hex digest as a content identifier."""
    return f"sha256:{hex_digest}"


def canonical_identity_bytes(value: object) -> bytes:
    """Encode identity keys losslessly, retaining ordinary I-JSON bytes and ordering."""
    return canonical_bytes(_normalize_wide_integers(value))


def content_id(data: bytes) -> str:
    """Hash bytes into the canonical SHA-256 content-identifier format."""
    return sha256_id(hashlib.sha256(data).hexdigest())


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
    return content_id(canonical_bytes(payload))


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
        normalized = {str(key): _normalize_wide_integers(item) for key, item in value.items()}
        if "$flameox.integer" in normalized or "$flameox.object" in normalized:
            # Native objects cannot impersonate the integer or escaped-object tags.
            return {"$flameox.object": [[key, normalized[key]] for key in sorted(normalized)]}
        return normalized
    return value
