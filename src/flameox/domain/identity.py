from __future__ import annotations

import hashlib
from typing import Any, Literal
from uuid import uuid4

import rfc8785
from pydantic import BaseModel, JsonValue, TypeAdapter

from flameox.models import ContractModel

_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_I_JSON_INTEGER_MIN = -(2**53) + 1
_I_JSON_INTEGER_MAX = 2**53 - 1


class SemanticIdentity(ContractModel):
    algorithm: Literal["rfc8785-sha256-v1"] = "rfc8785-sha256-v1"
    projection: str
    digest: str


def normalize_identity_value(value: Any) -> JsonValue:
    """Convert a declared domain value into the RFC 8785 JSON data model."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    else:
        value = _JSON_VALUE.dump_python(value, mode="json", warnings=False)
    value = _normalize_wide_integers(value)
    return _JSON_VALUE.validate_python(value)


def _normalize_wide_integers(value: Any) -> Any:
    """Project integers outside I-JSON's exact range without losing their value."""
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


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode current semantic identity bytes with maintained RFC 8785 JCS."""
    return rfc8785.dumps(normalize_identity_value(value))


def sha256_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_id(data: bytes) -> str:
    """Raw-byte content identity; deliberately independent of semantic JCS."""
    return f"sha256:{sha256_digest(data)}"


def semantic_identity(value: Any, *, projection: str) -> SemanticIdentity:
    normalized = normalize_identity_value(value)
    payload = {
        "algorithm": "rfc8785-sha256-v1",
        "projection": projection,
        "value": normalized,
    }
    return SemanticIdentity(
        projection=projection,
        digest=content_id(rfc8785.dumps(payload)),
    )


def digest_model(value: Any, *, projection: str = "flameox.generic/v1") -> str:
    """Digest a JSON-compatible model under an explicit semantic projection."""
    return semantic_identity(value, projection=projection).digest


def new_id() -> str:
    return str(uuid4())
