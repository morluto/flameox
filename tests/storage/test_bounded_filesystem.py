from __future__ import annotations

from pathlib import Path

import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.filesystem import BoundedFileSystem

pytestmark = pytest.mark.unit


def test_bounded_file_read_is_descriptor_relative_and_exact(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "nested" / "evidence.bin"
    source.parent.mkdir()
    source.write_bytes(b"evidence")
    files = BoundedFileSystem((root,))

    assert files.read_bytes(source, max_bytes=8) == b"evidence"
    with pytest.raises(DomainError) as oversized:
        files.read_bytes(source, max_bytes=7)
    assert oversized.value.code is ErrorCode.QUERY_BUDGET_EXCEEDED


def test_bounded_file_read_rejects_symlink_components(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"secret")
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DomainError) as refused:
        BoundedFileSystem((root,)).read_bytes(root / "escape" / "secret", max_bytes=10)
    assert refused.value.code is ErrorCode.EXECUTION_REFUSED
