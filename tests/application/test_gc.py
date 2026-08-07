from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from flameox.application import (
    CompactionService,
    FreezeRunSetRequest,
    GarbageCollector,
    GarbagePlan,
    ImportArtifactRequest,
    ImportService,
    RunSetService,
)
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


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


def test_gc_discovers_unreferenced_final_evidence_from_interrupted_publication(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    orphan = workspace.paths.evidence / "runs" / "generation=interrupted"
    orphan.mkdir(parents=True)
    (orphan / "part-00000.parquet").write_bytes(b"unreachable")
    old = time.time() - 48 * 3600
    os.utime(orphan, (old, old))

    collector = GarbageCollector(workspace)
    plan = collector.plan(minimum_age_hours=24)

    assert [(entry.path, entry.kind) for entry in plan.entries] == [
        ("evidence/runs/generation=interrupted", "evidence")
    ]

    result = collector.apply(plan)

    assert not orphan.exists()
    assert (
        Path(result.trash_root)
        / "objects"
        / "evidence"
        / "runs"
        / "generation=interrupted"
        / "part-00000.parquet"
    ).read_bytes() == b"unreachable"


def test_gc_rejects_symlink_candidate_without_moving_its_target(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    victim = workspace.paths.runs / "preserved"
    victim.mkdir(parents=True)
    payload = victim / "manifest.json"
    payload.write_text("must remain")
    old = time.time() - 48 * 3600
    os.utime(victim, (old, old))
    candidate = workspace.paths.staging / "linked-candidate"
    candidate.symlink_to(victim, target_is_directory=True)

    with pytest.raises(DomainError) as error:
        GarbageCollector(workspace).plan(minimum_age_hours=24)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert payload.read_text() == "must remain"
    assert candidate.is_symlink()


def test_gc_apply_rejects_candidate_swapped_to_symlink_after_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    candidate = workspace.paths.staging / "abandoned"
    candidate.mkdir()
    (candidate / "partial.bin").write_bytes(b"recoverable")
    old = time.time() - 48 * 3600
    os.utime(candidate, (old, old))
    victim = workspace.paths.runs / "preserved"
    victim.mkdir(parents=True)
    payload = victim / "manifest.json"
    payload.write_text("must remain")
    collector = GarbageCollector(workspace)
    authorized = collector.plan(minimum_age_hours=24)
    real_plan = collector.plan

    def swap_after_recheck(*, minimum_age_hours: int = 24) -> GarbagePlan:
        current = real_plan(minimum_age_hours=minimum_age_hours)
        candidate.rename(tmp_path / "displaced-candidate")
        candidate.symlink_to(victim, target_is_directory=True)
        return current

    monkeypatch.setattr(collector, "plan", swap_after_recheck)

    with pytest.raises(DomainError) as error:
        collector.apply(authorized)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert payload.read_text() == "must remain"
    assert candidate.is_symlink()
    assert not any(workspace.paths.trash.iterdir())


def test_gc_retains_generations_reachable_from_a_pinned_run_set(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "profile.bin"
    source.write_bytes(b"profile")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=source))
    run_set = RunSetService(workspace).freeze(FreezeRunSetRequest(run_ids=(imported.run.run_id,)))
    pinned_commit = workspace.corpus.read_commit(run_set.corpus_commit_id)
    pinned_generation_ids = {
        Path(relative).parent.name for relative in pinned_commit.generation_manifests
    }

    compacted = CompactionService(workspace).compact()
    assert compacted.superseded_generation_count >= 2
    old = time.time() - 48 * 3600
    for generation_id in pinned_generation_ids:
        generation = workspace.paths.generations / generation_id
        os.utime(generation, (old, old))

    plan = GarbageCollector(workspace).plan(minimum_age_hours=24)

    candidates = {Path(entry.path).name for entry in plan.entries if entry.kind == "generation"}
    assert run_set.corpus_commit_id in plan.root_corpus_commit_ids
    assert pinned_generation_ids <= set(plan.root_generation_ids)
    assert candidates.isdisjoint(pinned_generation_ids)


def test_gc_manifest_can_resume_an_interrupted_multi_object_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    old = time.time() - 48 * 3600
    for name in ("first", "second"):
        path = workspace.paths.staging / name
        path.mkdir()
        (path / "partial.bin").write_bytes(name.encode())
        os.utime(path, (old, old))
    collector = GarbageCollector(workspace)
    plan = collector.plan(minimum_age_hours=24)
    real_replace = os.replace
    candidate_moves = 0

    def interrupted_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal candidate_moves
        source_path = Path(source)
        if source_path.parent == workspace.paths.staging:
            candidate_moves += 1
            if candidate_moves == 2:
                raise OSError("injected crash boundary")
        real_replace(source, destination)

    monkeypatch.setattr(
        "flameox.application.recoverable_move.os.replace",
        interrupted_replace,
    )
    with pytest.raises(OSError, match="injected crash"):
        collector.apply(plan)
    (manifest_id,) = collector.moving_manifests()

    monkeypatch.setattr("flameox.application.recoverable_move.os.replace", real_replace)
    manifest = collector.resume(manifest_id)

    assert manifest.state == "recoverable"
    assert set(manifest.moved_paths) == {"staging/first", "staging/second"}
    assert not (workspace.paths.staging / "first").exists()
    assert not (workspace.paths.staging / "second").exists()


def test_gc_restore_and_explicit_expired_purge_are_manifest_specific(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    abandoned = workspace.paths.staging / "abandoned"
    abandoned.mkdir()
    (abandoned / "partial.bin").write_bytes(b"recoverable")
    old = time.time() - 48 * 3600
    os.utime(abandoned, (old, old))
    collector = GarbageCollector(workspace)
    first = collector.apply(
        collector.plan(minimum_age_hours=24),
        recovery_window_hours=0,
    )

    restored = collector.restore(first.trash_manifest_id)
    assert restored.restored[0].path == "staging/abandoned"
    assert (abandoned / "partial.bin").read_bytes() == b"recoverable"

    os.utime(abandoned, (old, old))
    second = collector.apply(
        collector.plan(minimum_age_hours=24),
        recovery_window_hours=0,
    )
    purged = collector.purge(second.trash_manifest_id)

    assert purged.trash_manifest_id == second.trash_manifest_id
    assert purged.purged_entries == 1
    assert not (workspace.paths.trash / second.trash_manifest_id).exists()


def test_gc_resume_completes_restore_interrupted_after_object_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    abandoned = workspace.paths.staging / "abandoned"
    abandoned.mkdir()
    (abandoned / "partial.bin").write_bytes(b"recoverable")
    old = time.time() - 48 * 3600
    os.utime(abandoned, (old, old))
    collector = GarbageCollector(workspace)
    applied = collector.apply(
        collector.plan(minimum_age_hours=24),
        recovery_window_hours=0,
    )
    real_replace = os.replace

    def fail_after_move(source: str | Path, destination: str | Path) -> None:
        real_replace(source, destination)
        if "objects" in Path(source).parts:
            raise OSError("injected restore crash")

    monkeypatch.setattr(
        "flameox.application.recoverable_move.os.replace",
        fail_after_move,
    )
    with pytest.raises(OSError, match="injected restore crash"):
        collector.restore(applied.trash_manifest_id)

    monkeypatch.setattr("flameox.application.recoverable_move.os.replace", real_replace)
    assert collector.moving_manifests() == (applied.trash_manifest_id,)
    manifest = collector.resume(applied.trash_manifest_id)

    assert manifest.state == "restored"
    assert (abandoned / "partial.bin").read_bytes() == b"recoverable"
