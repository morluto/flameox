from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from pydantic import BaseModel


def canonical_json_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def sha256_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_id(data: bytes) -> str:
    return f"sha256:{sha256_digest(data)}"


def digest_model(value: Any) -> str:
    return content_id(canonical_json_bytes(value))


def new_id() -> str:
    return str(uuid4())
