from __future__ import annotations

import json
from pathlib import Path

from flameox.storage import Workspace


def test_workspace_identity_persists_only_workspace_semantics(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    assert set(json.loads(workspace.paths.identity.read_text())) == {
        "workspace_id",
        "created_at",
        "project_root",
    }
