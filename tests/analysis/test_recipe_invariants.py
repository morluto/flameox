from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

from flameox.analysis import RecipeService
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


def test_read_only_analysis_pins_snapshot_without_workspace_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    head = workspace.corpus.read_head().commit_id

    def fail_write_lock(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("read-only analysis acquired the workspace write lock")

    monkeypatch.setattr(workspace, "write_locked", fail_write_lock)

    result = RecipeService(workspace).failures(limit=10)

    assert result.corpus_commit_id == head
    assert workspace.corpus.read_head().commit_id == head


def test_analysis_rejects_input_absent_from_pinned_corpus(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as error:
        RecipeService(workspace).hotspots("missing-run")

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
