from __future__ import annotations

import math

import pytest

from flameox.domain import (
    canonical_json_bytes,
    content_id,
    digest_model,
    semantic_identity,
)

pytestmark = pytest.mark.unit


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


def test_rfc8785_unicode_key_order_and_number_encoding_match_golden_vector() -> None:
    value = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "😀": "Emoji: Grinning Face",
        "ö": "Latin Small Letter O With Diaeresis",
        "\u0080": "Control",
    }

    assert canonical_json_bytes(value) == (
        b'{"\\r":"Carriage Return","1":"One","\xc2\x80":"Control",'
        b'"\xc3\xb6":"Latin Small Letter O With Diaeresis","\xe2\x82\xac":"Euro Sign",'
        b'"\xf0\x9f\x98\x80":"Emoji: Grinning Face",'
        b'"\xef\xac\xb3":"Hebrew Letter Dalet With Dagesh"}'
    )
    assert canonical_json_bytes([-0.0, 333333333.3333333, 1e30, 4.5, 0.002, 1e-27]) == (
        b"[0,333333333.3333333,1e+30,4.5,0.002,1e-27]"
    )


def test_semantic_identity_records_algorithm_and_projection() -> None:
    value = {"value": 1.0}
    current = semantic_identity(value, projection="test-vector/v1")

    assert current.algorithm == "rfc8785-sha256-v1"
    assert current.projection == "test-vector/v1"
    assert current.digest == digest_model(value, projection="test-vector/v1")


def test_wide_integer_identity_is_exact_and_distinct() -> None:
    boundary = 2**53

    assert canonical_json_bytes(boundary) == b'{"$flameox.integer":"9007199254740992"}'
    assert digest_model(boundary) != digest_model(boundary + 1)
