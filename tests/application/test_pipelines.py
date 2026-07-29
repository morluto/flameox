from __future__ import annotations

from pathlib import Path

from flameox.application import (
    ArtifactPipelineService,
    ImportArtifactRequest,
    ImportService,
    PipelineStageDeclaration,
    RegisterPipelineRequest,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, Sensitivity
from flameox.storage import RunStore, Workspace


def _import(workspace: Workspace, path: Path, content: str, *, sensitive: bool = False) -> str:
    path.write_text(content)
    return ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=path,
            kind=ArtifactKind.COLLECTOR_METADATA,
            sensitivity=Sensitivity.SENSITIVE if sensitive else Sensitivity.INTERNAL,
        )
    ).run.run_id


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
    second_status: str = "available",
    second_ordinal: int = 1,
    schema: str = "ir-v1",
    generated_lines: int = 1,
) -> str:
    registrations = RunStore(workspace).read(run_id).artifacts
    request = RegisterPipelineRequest(
        run_id=run_id,
        pipeline_name="compiler",
        pipeline_schema="pipeline-v1",
        producer="example-compiler",
        producer_version="1.0",
        stages=(
            PipelineStageDeclaration(
                name="input",
                ordinal=0,
                status="available",
                registration_id=registrations[0].registration_id,
                format="text",
                format_schema="source-v1",
                extractor="line-summary",
                extractor_version="1",
                structural_summary={"lines": 1},
            ),
            PipelineStageDeclaration(
                name="generated",
                ordinal=second_ordinal,
                predecessor="input",
                status=second_status,  # type: ignore[arg-type]
                registration_id=(
                    registrations[1].registration_id
                    if second_status in {"available", "cached"}
                    else None
                ),
                format="text",
                format_schema=schema,
                extractor=(
                    "line-summary" if second_status in {"available", "cached"} else None
                ),
                extractor_version=(
                    "1" if second_status in {"available", "cached"} else None
                ),
                structural_summary=(
                    {"lines": generated_lines}
                    if second_status in {"available", "cached"}
                    else None
                ),
            ),
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

    comparison = service.compare(
        _pipeline(service, workspace, left_run),
        _pipeline(service, workspace, right_run),
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
        _pipeline(service, workspace, right_run, second_status="skipped"),
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
                "environment_id": "different-environment",
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
