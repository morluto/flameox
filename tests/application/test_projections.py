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
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    EnvironmentRecord,
    ExecutionStatus,
    IdentityQuality,
    ProjectionState,
    RunSemantics,
    Sensitivity,
    ValidationStatus,
    digest_model,
)
from flameox.domain.models import ImportRunManifest
from flameox.storage import ArtifactStore, RunStore, Workspace
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
        semantics=RunSemantics.unavailable(origin="import", adapter="test-import"),
    )


def _crash_at(target: str) -> Callable[[str, object], None]:
    def inject(phase: str, _intent: object) -> None:
        if phase == target:
            raise InjectedCrash(phase)

    return inject


def test_projection_intent_binds_one_run_revision_and_publication_context(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    environment = _environment()
    spec = ProjectionCoordinator(workspace).run_projection_spec(
        _run("one-kind", environment),
        environment=environment,
        source_state=None,
    )

    assert spec.run_id == "one-kind"
    assert spec.run_revision == 0
    assert spec.context.environment == environment
    assert spec.context.source_state is None


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
    assert ControlPlane(workspace).list_projection_intents() == ()


def test_crash_after_domain_commit_recovers_exact_run_projection(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    environment = _environment()
    output = tmp_path / "stderr.bin"
    output.write_text("failed")
    stored = ArtifactStore(workspace).import_path(
        output,
        allowed_roots=(tmp_path,),
        max_bytes=1_024,
    )
    run = _run("domain-first", environment).validated_copy(
        update={
            "artifacts": (
                ArtifactRegistration(
                    registration_id="pending-output",
                    run_id="domain-first",
                    artifact_id=stored.content.artifact_id,
                    display_name="stderr.bin",
                    media_type="application/octet-stream",
                    kind=ArtifactKind.PROCESS_OUTPUT,
                    role="stderr",
                    sensitivity=Sensitivity.INTERNAL,
                ),
            )
        }
    )

    with pytest.raises(InjectedCrash):
        ProjectionCoordinator(
            workspace,
            fault_injector=_crash_at("after_domain_commit"),
        ).create_run(run, environment=environment, source_state=None)

    [pending] = ControlPlane(workspace).list_projection_intents(state=ProjectionState.PENDING)
    assert pending.run_revision == 0
    assert pending.run_digest == digest_model(run.model_dump(mode="json"))
    assert workspace.corpus.read_head().generation_ids == ()
    pending_view = RunProjectionService(workspace).get(run.run_id)
    assert pending_view.manifest_source == "control_plane"
    assert pending_view.projection.state is ProjectionVisibilityState.PENDING
    assert pending_view.projection.current is False
    assert pending_view.projection.projected_revision is None
    assert pending_view.recovery_actions == ()

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
        ).fetchone() == (0, pending.run_digest)


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

    [pending] = ControlPlane(workspace).list_projection_intents(state=ProjectionState.PENDING)
    first_head = workspace.corpus.read_head()
    assert len(first_head.generation_ids) == 1

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

    [failed] = ControlPlane(workspace).list_projection_intents(state=ProjectionState.FAILED)
    assert failed.failure_code == "publication_failed"
    assert workspace.corpus.read_head().generation_ids == ()

    reconciled = ProjectionCoordinator(workspace).reconcile_intent(failed.intent_id)

    assert reconciled.state is ProjectionState.PUBLISHED
    assert len(workspace.corpus.read_head().generation_ids) == 1


def test_crash_after_intent_completion_is_already_converged(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    environment = _environment()
    run = _run("completed-crash", environment)

    with pytest.raises(InjectedCrash):
        ProjectionCoordinator(
            workspace,
            fault_injector=_crash_at("after_intent_commit"),
        ).create_run(run, environment=environment, source_state=None)

    [published] = ControlPlane(workspace).list_projection_intents(state=ProjectionState.PUBLISHED)
    assert ProjectionCoordinator(workspace).reconcile_intent(published.intent_id) == published
    assert len(workspace.corpus.read_head().generation_ids) == 1
