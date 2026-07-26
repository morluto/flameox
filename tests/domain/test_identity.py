from __future__ import annotations

import math

import pytest

from flameox.domain import canonical_json_bytes, content_id, digest_model


def test_canonical_json_is_order_independent() -> None:
    left = {"b": [2, 1], "a": {"x": True}}
    right = {"a": {"x": True}, "b": [2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert digest_model(left) == digest_model(right)


def test_canonical_json_rejects_non_finite_floats() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": math.nan})


def test_content_id_uses_prefixed_lowercase_sha256() -> None:
    assert content_id(b"flameox") == (
        "sha256:4bbbed24726110a321cbf5aff0c6a29478b60199ef1250b41af5eca6cfd009f6"
    )
