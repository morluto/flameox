from __future__ import annotations

from flameox.canonical import canonical_bytes, canonical_identity_bytes, digest_model


def test_identity_encoding_distinguishes_native_values_from_reserved_tags() -> None:
    wide = 2**63 + 42
    values: list[object] = [
        wide,
        str(wide),
        {"$flameox.integer": str(wide)},
        {"$flameox.object": [["$flameox.integer", str(wide)]]},
        {"nested": [wide]},
        {"nested": [{"$flameox.integer": str(wide)}]},
        {"nested": [{"$flameox.object": [["$flameox.integer", str(wide)]]}]},
    ]
    assert len({canonical_identity_bytes(value) for value in values}) == len(values)
    assert len({digest_model(value) for value in values}) == len(values)


def test_identity_encoding_retains_ordinary_canonical_bytes() -> None:
    value = {"numbers": [-(2**53) + 1, 2**53 - 1, 1.5], "nested": {"label": "native", "flag": True}}
    assert canonical_identity_bytes(value) == canonical_bytes(value)
