from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from flameox.application import (
    ArtifactPipelineService,
    ImportArtifactRequest,
    ImportService,
    PipelineComparison,
    PipelineStageComparison,
    PipelineStageDeclaration,
    PipelineStageStatus,
    RegisteredPipelineStage,
    RegisteredPipelineStageDeclaration,
    RegisterPipelineRequest,
    UnregisteredPipelineStageDeclaration,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, Sensitivity, digest_model
from flameox.storage import RunStore, Workspace

_STAGE_ADAPTER: TypeAdapter[PipelineStageDeclaration] = TypeAdapter(PipelineStageDeclaration)
_STAGE_COMPARISON_ADAPTER: TypeAdapter[PipelineStageComparison] = TypeAdapter(
    PipelineStageComparison
)
type StageStatus = PipelineStageStatus


@pytest.mark.parametrize(
    "status",
    list(PipelineStageStatus),
)
def test_pipeline_stage_status_routes_to_registration_variant(status: StageStatus) -> None:
    payload: dict[str, object] = {
        "name": "generated",
        "ordinal": 1,
        "status": status,
        "registration_id": "registration-1" if status in {"available", "cached"} else None,
        "format": "text",
        "format_schema": "source-v1",
    }

    stage = _STAGE_ADAPTER.validate_python(payload)

    if status in {PipelineStageStatus.AVAILABLE, PipelineStageStatus.CACHED}:
        assert isinstance(stage, RegisteredPipelineStageDeclaration)
    else:
        assert isinstance(stage, UnregisteredPipelineStageDeclaration)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "name": "generated",
            "ordinal": 1,
            "status": "available",
            "format": "text",
            "format_schema": "source-v1",
        },
        {
            "name": "generated",
            "ordinal": 1,
            "status": "skipped",
            "registration_id": "registration-1",
            "format": "text",
            "format_schema": "source-v1",
        },
    ],
)
def test_pipeline_stage_rejects_registration_status_mismatch(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _STAGE_ADAPTER.validate_python(payload)


def test_pipeline_stage_comparison_requires_the_side_named_by_its_disposition() -> None:
    with pytest.raises(ValidationError):
        _STAGE_COMPARISON_ADAPTER.validate_python(
            {
                "stage_name": "generated",
                "disposition": "added",
                "baseline_ordinal": 0,
                "candidate_ordinal": 1,
            }
        )


def test_pipeline_stage_comparison_rejects_a_stale_short_circuit_projection() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _STAGE_COMPARISON_ADAPTER.validate_python(
            {
                "stage_name": "generated",
                "disposition": "identical",
                "baseline_ordinal": 1,
                "candidate_ordinal": 1,
                "baseline_artifact_id": "sha256:same",
                "candidate_artifact_id": "sha256:same",
                "extraction_short_circuited": False,
            }
        )


def test_pipeline_comparison_rejects_stale_derived_projections() -> None:
    payload = {
        "baseline_pipeline_id": "baseline",
        "candidate_pipeline_id": "candidate",
        "compatibility": "compatible",
        "identity_mismatches": [],
        "stages": [
            {
                "stage_name": "generated",
                "disposition": "changed",
                "baseline_ordinal": 1,
                "candidate_ordinal": 1,
            }
        ],
        "input_artifact_ids": [],
        "extractor_identities": [],
        "limitations": [],
    }
    comparison = PipelineComparison.model_validate(payload)

    assert comparison.first_observed_divergent_stage == "generated"
    assert comparison.comparison_id == comparison.result_digest
    with pytest.raises(ValidationError, match="first divergent stage"):
        PipelineComparison.model_validate(
            {**payload, "first_observed_divergent_stage": "different"}
        )
    with pytest.raises(ValidationError, match="digest does not match"):
        PipelineComparison.model_validate({**payload, "result_digest": "sha256:stale"})


def _import(workspace: Workspace, path: Path, content: str, *, sensitive: bool = False) -> str:
    path.write_text(content)
    return (
        ImportService(workspace)
        .import_artifact(
            ImportArtifactRequest(
                path=path,
                kind=ArtifactKind.COLLECTOR_METADATA,
                sensitivity=Sensitivity.SENSITIVE if sensitive else Sensitivity.INTERNAL,
            )
        )
        .run.run_id
    )


def _add_registration(
    workspace: Workspace,
    run_id: str,
    registration_id: str,
    *,
    artifact_id: str | None = None,
) -> None:
    store = RunStore(workspace)
    run = store.read(run_id)
    extra = run.artifacts[0].model_copy(
        update={
            "registration_id": registration_id,
            "artifact_id": artifact_id or run.artifacts[0].artifact_id,
            "role": "generated",
            "display_name": f"{registration_id}.txt",
        }
    )
    store.append(
        run.model_copy(update={"revision": run.revision + 1, "artifacts": (*run.artifacts, extra)}),
        expected_revision=run.revision,
    )
    Catalog(workspace).rebuild()


def _pipeline(
    service: ArtifactPipelineService,
    workspace: Workspace,
    run_id: str,
    *,
    second_status: StageStatus = PipelineStageStatus.AVAILABLE,
    second_ordinal: int = 1,
    schema: str = "ir-v1",
    generated_lines: int = 1,
    producer_version: str = "1.0",
    workload_identity: str | None = "workload",
    device_identity: str | None = "device",
) -> str:
    registrations = RunStore(workspace).read(run_id).artifacts
    if second_status in {PipelineStageStatus.AVAILABLE, PipelineStageStatus.CACHED}:
        second_stage: PipelineStageDeclaration = RegisteredPipelineStageDeclaration(
            name="generated",
            ordinal=second_ordinal,
            predecessor="input",
            status=second_status,
            registration_id=registrations[1].registration_id,
            format="text",
            format_schema=schema,
            extractor="line-summary",
            extractor_version="1",
            structural_summary={"lines": generated_lines},
        )
    else:
        second_stage = UnregisteredPipelineStageDeclaration(
            name="generated",
            ordinal=second_ordinal,
            predecessor="input",
            status=second_status,
            format="text",
            format_schema=schema,
        )
    request = RegisterPipelineRequest(
        run_id=run_id,
        pipeline_name="compiler",
        pipeline_schema="pipeline-v1",
        producer="example-compiler",
        producer_version=producer_version,
        workload_identity=workload_identity,
        device_identity=device_identity,
        stages=(
            RegisteredPipelineStageDeclaration(
                name="input",
                ordinal=0,
                status=PipelineStageStatus.AVAILABLE,
                registration_id=registrations[0].registration_id,
                format="text",
                format_schema="source-v1",
                extractor="line-summary",
                extractor_version="1",
                structural_summary={"lines": 1},
            ),
            second_stage,
        ),
    )
    return service.register(request).pipeline_id


def test_identical_pipeline_short_circuits_content_addressed_artifacts(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    left_run = _import(workspace, tmp_path / "left.txt", "same")
    right_run = _import(workspace, tmp_path / "right.txt", "same")
    _add_registration(workspace, left_run, "left-generated")
    _add_registration(workspace, right_run, "right-generated")
    service = ArtifactPipelineService(workspace)

    left_pipeline_id = _pipeline(service, workspace, left_run)
    right_pipeline_id = _pipeline(service, workspace, right_run)
    comparison = service.compare(left_pipeline_id, right_pipeline_id)

    assert all(
        isinstance(stage, RegisteredPipelineStage)
        for stage in service.pipelines.read(left_pipeline_id).stages
    )
    assert [stage.disposition for stage in comparison.stages] == ["identical", "identical"]
    assert all(stage.extraction_short_circuited for stage in comparison.stages)
    assert len(comparison.input_artifact_ids) == 1
    assert comparison.first_observed_divergent_stage is None
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM artifact_pipelines").fetchone() == (2,)
        assert snapshot.execute("SELECT count(*) FROM pipeline_comparisons").fetchone() == (1,)


def test_pipeline_reports_late_difference_without_claiming_root_cause(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    left_run = _import(workspace, tmp_path / "left.txt", "same")
    right_run = _import(workspace, tmp_path / "right.txt", "same")
    left_generated = _import(workspace, tmp_path / "left-generated.txt", "left")
    right_generated = _import(workspace, tmp_path / "right-generated.txt", "right")
    _add_registration(
        workspace,
        left_run,
        "left-generated",
        artifact_id=RunStore(workspace).read(left_generated).artifacts[0].artifact_id,
    )
    _add_registration(
        workspace,
        right_run,
        "right-generated",
        artifact_id=RunStore(workspace).read(right_generated).artifacts[0].artifact_id,
    )
    service = ArtifactPipelineService(workspace)

    comparison = service.compare(
        _pipeline(service, workspace, left_run),
        _pipeline(service, workspace, right_run, generated_lines=2),
    )

    assert comparison.stages[0].disposition == "identical"
    assert comparison.stages[1].disposition == "changed"
    assert comparison.stages[1].artifact_length_change == 1
    assert comparison.first_observed_divergent_stage == "generated"
    assert any("not a root-cause" in item for item in comparison.limitations)

    same_structure = service.compare(
        _pipeline(service, workspace, left_run),
        _pipeline(service, workspace, right_run),
    )
    assert same_structure.stages[1].disposition == "content_changed"


def test_pipeline_marks_skipped_and_incompatible_stages_explicitly(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    left_run = _import(workspace, tmp_path / "left.txt", "same", sensitive=True)
    right_run = _import(workspace, tmp_path / "right.txt", "same", sensitive=True)
    _add_registration(workspace, left_run, "left-generated")
    _add_registration(workspace, right_run, "right-generated")
    service = ArtifactPipelineService(workspace)
    baseline = _pipeline(service, workspace, left_run)

    skipped = service.compare(
        baseline,
        _pipeline(
            service,
            workspace,
            right_run,
            second_status=PipelineStageStatus.SKIPPED,
        ),
    )
    incompatible = service.compare(
        baseline,
        _pipeline(service, workspace, right_run, schema="ir-v2"),
    )

    assert skipped.stages[1].disposition == "uninspectable"
    assert skipped.first_observed_divergent_stage is None
    assert incompatible.stages[1].disposition == "incompatible"
    assert incompatible.first_observed_divergent_stage is None


def test_pipeline_identity_mismatch_invalidates_stage_comparison(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    left_run = _import(workspace, tmp_path / "left.txt", "same")
    right_run = _import(workspace, tmp_path / "right.txt", "same")
    _add_registration(workspace, left_run, "left-generated")
    _add_registration(workspace, right_run, "right-generated")
    runs = RunStore(workspace)
    right = runs.read(right_run)
    runs.append(
        right.model_copy(
            update={
                "revision": right.revision + 1,
                "environment_id": digest_model({"environment": "different"}),
            }
        ),
        expected_revision=right.revision,
    )
    service = ArtifactPipelineService(workspace)

    comparison = service.compare(
        _pipeline(service, workspace, left_run),
        _pipeline(service, workspace, right_run),
    )

    assert comparison.compatibility == "incompatible"
    assert comparison.identity_mismatches == ("environment_id",)
    assert all(stage.disposition == "incompatible" for stage in comparison.stages)
    assert comparison.first_observed_divergent_stage is None


def test_pipeline_missing_critical_identity_is_unknown_until_known_mismatch(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    left_run = _import(workspace, tmp_path / "left.txt", "same")
    right_run = _import(workspace, tmp_path / "right.txt", "same")
    _add_registration(workspace, left_run, "left-generated")
    _add_registration(workspace, right_run, "right-generated")
    service = ArtifactPipelineService(workspace)

    both_missing = service.compare(
        _pipeline(
            service,
            workspace,
            left_run,
            workload_identity=None,
            device_identity=None,
        ),
        _pipeline(
            service,
            workspace,
            right_run,
            workload_identity=None,
            device_identity=None,
        ),
    )
    known_vs_missing = service.compare(
        _pipeline(service, workspace, left_run),
        _pipeline(
            service,
            workspace,
            right_run,
            workload_identity=None,
            device_identity=None,
        ),
    )
    known_mismatch = service.compare(
        _pipeline(service, workspace, left_run, workload_identity="left"),
        _pipeline(service, workspace, right_run, workload_identity="right"),
    )

    assert both_missing.compatibility == "unknown"
    assert both_missing.identity_mismatches == ()
    assert known_vs_missing.compatibility == "unknown"
    assert known_vs_missing.identity_mismatches == ()
    assert known_mismatch.compatibility == "incompatible"
    assert known_mismatch.identity_mismatches == ("workload_identity",)
