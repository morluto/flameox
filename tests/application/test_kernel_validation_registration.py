from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flameox.adapters import KernelValidationExtractor
from flameox.application import (
    ArtifactPipelineService,
    CaptureService,
    ExecutionPolicy,
    ImportArtifactRequest,
    ImportService,
    KernelValidationRegistrationService,
    PipelineStageStatus,
    RegisteredPipelineStageDeclaration,
    RegisterKernelValidationRequest,
    RegisterPipelineRequest,
)
from flameox.catalog import Catalog
from flameox.cli import app
from flameox.domain import ArtifactKind, DomainError, ErrorCode, ExecutionStatus, ValidationStatus
from flameox.storage import RunStore, Workspace
from tests.support.capture import disable_containment, write_workload

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kernel_validation" / "pass.json"


async def _capture(workspace: Workspace) -> tuple[str, int]:
    write_workload(workspace.project_root)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    return result.run.run_id, result.run.revision


@pytest.mark.anyio
async def test_register_kernel_validation_updates_the_producing_run(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id, revision = await _capture(workspace)
    run_before = RunStore(workspace).read(run_id)
    source_registration = run_before.artifacts[0]
    source_pipeline = ArtifactPipelineService(workspace).register(
        RegisterPipelineRequest(
            run_id=run_id,
            pipeline_name="validation-target",
            pipeline_schema="validation-target.v1",
            producer="fixture",
            producer_version="1",
            stages=(
                RegisteredPipelineStageDeclaration(
                    name="execution_output",
                    ordinal=0,
                    status=PipelineStageStatus.AVAILABLE,
                    registration_id=source_registration.registration_id,
                    format=source_registration.media_type,
                    format_schema="raw",
                ),
            ),
        )
    )
    validation_path = tmp_path / "validation.json"
    shutil.copy2(FIXTURE, validation_path)

    registered = KernelValidationRegistrationService(workspace).register(
        RegisterKernelValidationRequest(
            run_id=run_id,
            expected_run_revision=revision,
            path=validation_path,
        )
    )
    extracted = KernelValidationExtractor(workspace).extract(run_id)
    extracted_again = KernelValidationExtractor(workspace).extract(run_id)
    run = RunStore(workspace).read(run_id)

    assert registered.run_id == run_id
    assert registered.run_revision == revision + 1
    assert registered.environment_id == run.environment_id
    assert registered.source_state_id == run.source_state_id
    assert registered.workload_definition_id == run.workload_definition_id
    assert registered.workload_instance_id == run.workload_instance_id
    assert run.execution_identity is not None
    assert registered.execution_identity_id == run.execution_identity.identity_id
    assert len(registered.pipeline_ids) == 1
    derived_pipeline = ArtifactPipelineService(workspace).pipelines.read(
        registered.pipeline_ids[0]
    )
    assert derived_pipeline.pipeline_id != source_pipeline.pipeline_id
    assert derived_pipeline.run_id == run_id
    assert [stage.name for stage in derived_pipeline.stages] == [
        "execution_output",
        "kernel_validation",
    ]
    assert derived_pipeline.environment_id == source_pipeline.environment_id
    assert derived_pipeline.source_state_id == source_pipeline.source_state_id
    assert derived_pipeline.producer == source_pipeline.producer
    assert derived_pipeline.producer_version == source_pipeline.producer_version
    assert run.validation_status is ValidationStatus.PASSED
    linked_artifacts = [
        (item.role, item.artifact_id)
        for item in run.artifacts
        if item.role == "kernel_validation"
    ]
    assert linked_artifacts == [
        ("kernel_validation", registered.artifact_id)
    ]
    assert extracted.run_id == run_id
    assert extracted.artifact_id == registered.artifact_id
    assert extracted_again.corpus_commit_id == extracted.corpus_commit_id
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute(
            "SELECT environment_id FROM environments WHERE environment_id = ?",
            (run.environment_id,),
        ).fetchone() == (run.environment_id,)
        assert snapshot.execute(
            "SELECT source_state_id FROM source_states WHERE source_state_id = ?",
            (run.source_state_id,),
        ).fetchone() == (run.source_state_id,)


@pytest.mark.anyio
async def test_register_kernel_validation_rejects_a_stale_source_run(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id, revision = await _capture(workspace)
    validation_path = tmp_path / "validation.json"
    shutil.copy2(FIXTURE, validation_path)

    with pytest.raises(DomainError) as error:
        KernelValidationRegistrationService(workspace).register(
            RegisterKernelValidationRequest(
                run_id=run_id,
                expected_run_revision=revision + 1,
                path=validation_path,
            )
        )

    assert error.value.code is ErrorCode.REVISION_CONFLICT


def test_register_kernel_validation_rejects_import_runs_and_missing_runs(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    validation_path = tmp_path / "validation.json"
    shutil.copy2(FIXTURE, validation_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=validation_path, kind=ArtifactKind.VALIDATION_OUTPUT)
    )

    with pytest.raises(DomainError) as wrong_type:
        KernelValidationRegistrationService(workspace).register(
            RegisterKernelValidationRequest(
                run_id=imported.run.run_id,
                expected_run_revision=imported.run.revision,
                path=validation_path,
            )
        )
    with pytest.raises(DomainError) as missing:
        KernelValidationRegistrationService(workspace).register(
            RegisterKernelValidationRequest(
                run_id="missing",
                expected_run_revision=0,
                path=validation_path,
            )
        )

    assert wrong_type.value.code is ErrorCode.INVALID_ARGUMENTS
    assert missing.value.code is ErrorCode.RUN_NOT_FOUND


@pytest.mark.anyio
async def test_cli_registers_kernel_validation_against_a_reviewed_run(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    run_id, revision = await _capture(workspace)
    validation_path = tmp_path / "validation.json"
    shutil.copy2(FIXTURE, validation_path)

    result = CliRunner().invoke(
        app,
        [
            "register-kernel-validation",
            run_id,
            str(validation_path),
            "--expected-run-revision",
            str(revision),
            "--workspace",
            str(workspace.paths.root),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run_id"] == run_id
    assert payload["run_revision"] == revision + 1
    assert payload["validation_status"] == "passed"
