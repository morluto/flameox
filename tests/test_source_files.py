from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from flameox.runtime_contracts import RuntimeFailure
from flameox.source_files import NativeSource, copy_verified_file

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("actual", [b"old", b"", b"new", b"old" * 1000])
def test_copy_checks_identity_and_never_exceeds_admitted_size(
    tmp_path: Path, actual: bytes
) -> None:
    path = tmp_path / "source"
    path.write_bytes(actual)
    source = NativeSource(path, hashlib.sha256(b"old").hexdigest(), 3, "text", None, "input")
    destination = tmp_path / "copy"
    if actual == b"old":
        copy_verified_file(source, destination)
        assert destination.read_bytes() == actual
    else:
        with pytest.raises(RuntimeFailure) as failure:
            copy_verified_file(source, destination)
        assert failure.value.code == "MISSING_OR_CHANGED_INPUT"
        assert destination.stat().st_size <= source.size_bytes
