from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flameox.application.source import collect_source_state
from flameox.domain import IdentityQuality
from flameox.execution import SubprocessBroker
from flameox.storage import Workspace


def git(project: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=project,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.mark.anyio
async def test_git_source_identity_includes_dirty_and_untracked_content(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n")
    git(tmp_path, "add", "tracked.py")
    git(tmp_path, "commit", "-m", "initial")
    workspace = Workspace.initialize(tmp_path)
    broker = SubprocessBroker()

    clean = await collect_source_state(
        workspace,
        workload_executable=sys.executable,
        broker=broker,
    )
    tracked.write_text("value = 2\n")
    (tmp_path / "untracked.py").write_text("new = True\n")
    dirty = await collect_source_state(
        workspace,
        workload_executable=sys.executable,
        broker=broker,
    )

    assert clean.identity_quality is IdentityQuality.CLEAN
    assert dirty.identity_quality is IdentityQuality.EXACT
    assert clean.source_state_id != dirty.source_state_id
    assert clean.diff_digest != dirty.diff_digest
    assert dirty.fields["untracked_inputs"]
