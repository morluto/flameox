from __future__ import annotations

import os
import time
from pathlib import Path

from flamo.application import (
    GarbageCollector,
    ImportArtifactRequest,
    ImportService,
)
from flamo.storage import Workspace


def test_gc_is_dry_run_first_and_moves_only_plan_to_recoverable_trash(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "referenced.bin"
    source.write_bytes(b"keep")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=source))
    orphan = workspace.paths.staging / "abandoned"
    orphan.mkdir()
    (orphan / "partial.bin").write_bytes(b"recoverable")
    old = time.time() - 48 * 3600
    os.utime(orphan, (old, old))

    collector = GarbageCollector(workspace)
    plan = collector.plan(minimum_age_hours=24)

    assert orphan.exists()
    assert [entry.path for entry in plan.entries] == ["staging/abandoned"]
    assert all(imported.artifact_id not in entry.path for entry in plan.entries)

    result = collector.apply(plan)

    assert not orphan.exists()
    trash = Path(result.trash_root)
    assert (trash / "manifest.json").is_file()
    assert (trash / "objects" / "staging" / "abandoned" / "partial.bin").read_bytes() == (
        b"recoverable"
    )
