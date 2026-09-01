from __future__ import annotations

import hashlib

from flameox.canonical import content_id, sha256_id


def test_sha256_id_formats_an_existing_digest() -> None:
    hex_digest = hashlib.sha256(b"flameox").hexdigest()

    assert sha256_id(hex_digest) == f"sha256:{hex_digest}"


def test_content_id_hashes_bytes_into_a_sha256_identifier() -> None:
    assert content_id(b"flameox") == sha256_id(hashlib.sha256(b"flameox").hexdigest())
