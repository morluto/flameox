from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import JsonValue

from flameox.application.projections import ProjectionCoordinator
from flameox.application.run_projection import (
    ProjectionVisibilityState,
    RunProjectionService,
)
from flameox.catalog import Catalog
from flameox.domain import (
    CaptureStatus,
    EnvironmentRecord,
    ExecutionStatus,
    IdentityQuality,
    ProjectionState,
    ValidationStatus,
    digest_model,
)
from flameox.domain.models import ImportRunManifest
from flameox.storage import ProjectionIntentStore, RunStore, Workspace
from flameox.storage.control_plane import ControlPlane

pytestmark = pytest.mark.integration


class InjectedCrash(BaseException):
    pass


def _environment() -> EnvironmentRecord:
    fields: dict[str, JsonValue] = {"platform": "test"}
    return EnvironmentRecord(
        environment_id=digest_model(
            {"identity_quality": IdentityQuality.EXACT.value, "fields": fields}
        ),
        identity_quality=IdentityQuality.EXACT,
        fields=fields,
    )


def _run(run_id: str, environment: EnvironmentRecord) -> ImportRunManifest:
    return ImportRunManifest(
        run_id=run_id,
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=environment.environment_id,
        collector="test-import",
    )


def _crash_at(target: str) -> Callable[[str, object], None]:
    def inject(phase: str, _intent: object) -> None:
        if phase == target:
            raise InjectedCrash(phase)

    return inject


def test_run_revision_and_projection_intent_commit_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    environment = _environment()
    run = _run("atomic-run", environment)

    def reject_intent(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("injected intent failure")

    monkeypatch.setattr(ControlPlane, "_insert_projection_intent", reject_intent)

    with pytest.raises(RuntimeError, match="injected intent failure"):
        ProjectionCoordinator(workspace).create_run(
            run,
            environment=environment,
            source_state=None,
        )

    assert RunStore(workspace).list() == ()
    assert ProjectionIntentStore(workspace).list() == ()


def test_crash_after_domain_commit_replays_exact_run_projection(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    environment = _environment()
    run = _run("domain-first", environment)

    with pytest.raises(InjectedCrash):
        ProjectionCoordinator(
            workspace,
            fault_injector=_crash_at("after_domain_commit"),
        ).create_run(run, environment=environment, source_state=None)

    [pending] = ProjectionIntentStore(workspace).list(state=ProjectionState.PENDING)
    assert pending.domain_revision == 0
    assert pending.domain_digest == digest_model(run.model_dump(mode="json"))
    assert workspace.corpus.read_head().generation_manifests == ()
    pending_view = RunProjectionService(workspace).get(run.run_id)
    assert pending_view.manifest_source == "control_plane"
    assert pending_view.projection.state is ProjectionVisibilityState.PENDING
    assert pending_view.projection.current is False
    assert pending_view.projection.projected_revision is None

    reconciled = ProjectionCoordinator(workspace).reconcile_intent(pending.intent_id)

    assert reconciled.state is ProjectionState.PUBLISHED
    published_view = RunProjectionService(workspace).get(run.run_id)
    assert published_view.manifest_source == "corpus_projection"
    assert published_view.projection.state is ProjectionVisibilityState.PUBLISHED
    assert published_view.projection.current is True
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute(
            "SELECT run_revision, run_manifest_digest FROM current_runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone() == (0, pending.domain_digest)


def test_crash_after_head_publication_finalizes_without_duplicate_generation(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    environment = _environment()
    run = _run("head-first", environment)

    with pytest.raises(InjectedCrash):
        ProjectionCoordinator(
            workspace,
            fault_injector=_crash_at("after_corpus_publish"),
        ).create_run(run, environment=environment, source_state=None)

    [pending] = ProjectionIntentStore(workspace).list(state=ProjectionState.PENDING)
    first_head = workspace.corpus.read_head()
    assert len(first_head.generation_manifests) == 1

    reconciled = ProjectionCoordinator(workspace).reconcile_intent(pending.intent_id)

    assert reconciled.state is ProjectionState.PUBLISHED
    assert reconciled.corpus_commit_id == first_head.commit_id
    assert workspace.corpus.read_head() == first_head


def test_head_failure_leaves_typed_retryable_projection_and_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    environment = _environment()
    run = _run("head-failure", environment)
    publish_head = workspace.corpus.publish_head
    calls = 0

    def fail_once(commit_id: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected HEAD failure")
        publish_head(commit_id)

    monkeypatch.setattr(workspace.corpus, "publish_head", fail_once)

    with pytest.raises(OSError, match="injected HEAD failure"):
        ProjectionCoordinator(workspace).create_run(
            run,
            environment=environment,
            source_state=None,
        )

    [failed] = ProjectionIntentStore(workspace).list(state=ProjectionState.FAILED)
    assert failed.failure_code == "publication_failed"
    assert workspace.corpus.read_head().generation_manifests == ()

    reconciled = ProjectionCoordinator(workspace).reconcile_intent(failed.intent_id)

    assert reconciled.state is ProjectionState.PUBLISHED
    assert len(workspace.corpus.read_head().generation_manifests) == 1


def test_crash_after_intent_completion_is_already_converged(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    environment = _environment()
    run = _run("completed-crash", environment)

    with pytest.raises(InjectedCrash):
        ProjectionCoordinator(
            workspace,
            fault_injector=_crash_at("after_intent_commit"),
        ).create_run(run, environment=environment, source_state=None)

    [published] = ProjectionIntentStore(workspace).list(state=ProjectionState.PUBLISHED)
    assert ProjectionCoordinator(workspace).reconcile_intent(published.intent_id) == published
    assert len(workspace.corpus.read_head().generation_manifests) == 1
