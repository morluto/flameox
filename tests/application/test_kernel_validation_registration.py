from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flameox.adapters.kernel_validation import KernelValidationExtractor
from flameox.application.capabilities import CapabilityService
from flameox.application.capture import CaptureService
from flameox.application.evidence_query import EvidenceQueryService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.application.kernel_validation import (
    KernelValidationRegistrationService,
    RegisterKernelValidationRequest,
)
from flameox.application.pipelines import (
    ArtifactPipelineService,
    PipelineStageStatus,
    RegisteredPipelineStageDeclaration,
    RegisterPipelineRequest,
)
from flameox.catalog import Catalog
from flameox.cli import app
from flameox.domain import (
    ArtifactKind,
    CapabilityReport,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ValidationStatus,
)
from flameox.storage import ArtifactStore, RunStore, Workspace
from tests.support.capture import disable_containment, write_workload

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kernel_validation" / "pass.json"


class _VersionedCompilerCapabilities(CapabilityService):
    def get(self, adapter: str) -> CapabilityReport:
        report = super().get(adapter)
        if adapter in {"triton.compiler", "cute.compiler"}:
            return report.model_copy(update={"version": "fixture-compiler-1"})
        return report


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
            producer="fixture",
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
    derived_pipeline = ArtifactPipelineService(workspace).pipelines.read(registered.pipeline_ids[0])
    assert derived_pipeline.pipeline_id != source_pipeline.pipeline_id
    assert derived_pipeline.run_id == run_id
    assert [stage.name for stage in derived_pipeline.stages] == [
        "execution_output",
        "kernel_validation",
    ]
    assert derived_pipeline.producer == source_pipeline.producer
    assert {
        "environment_id",
        "source_state_id",
        "workload_definition_id",
        "producer_version",
    }.isdisjoint(derived_pipeline.model_dump())
    assert run.validation_status is ValidationStatus.PASSED
    linked_artifacts = [
        (item.role, item.artifact_id) for item in run.artifacts if item.role == "kernel_validation"
    ]
    assert linked_artifacts == [("kernel_validation", registered.artifact_id)]
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
async def test_public_compile_validation_benchmark_pipeline_preserves_provenance(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    compiler = tmp_path / "compile.py"
    compiler.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "dump = Path(os.environ['CUTE_DSL_DUMP_DIR'])\n"
        "dump.mkdir(parents=True, exist_ok=True)\n"
        "(dump / 'kernel.mlir').write_text('candidate ir')\n"
        "(dump / 'kernel.ptx').write_text('candidate ptx')\n"
        "(dump / 'kernel.cubin').write_bytes(b'candidate cubin')\n"
        "print('benchmark sample: 42')\n",
        encoding="utf-8",
    )
    (tmp_path / "flameox.toml").write_text(
        f"[workloads.compile]\nargv = ['python', '{compiler}']\ncwd = '.'\ntimeout_seconds = 30\n",
        encoding="utf-8",
    )
    disable_containment(workspace)

    capture = CaptureService(
        workspace,
        capabilities=_VersionedCompilerCapabilities(workspace),
    )
    plan = await capture.plan(
        workload_name="compile",
        adapter="cute.compiler",
        adapter_options={"keep_allowlist": ["ir", "ptx", "cubin"]},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    compiled = await capture.execute(plan.plan_token)
    source = RunStore(workspace).read(compiled.run.run_id)
    source_artifact_ids = tuple(item.artifact_id for item in source.artifacts)
    assert source.execution_status is ExecutionStatus.SUCCEEDED
    assert source.measurement_protocol_id is not None
    assert source.execution_identity is not None
    assert compiled.pipeline_ids
    benchmark_registration = next(item for item in source.artifacts if item.role == "stdout")
    benchmark_artifact = ArtifactStore(workspace).get(benchmark_registration.artifact_id)
    assert benchmark_artifact.payload_path.read_bytes() == b"benchmark sample: 42\n"

    benchmark = EvidenceQueryService(workspace).measurements(
        run_id=source.run_id,
        name_prefix="process.wall_time",
        limit=1,
    )
    assert benchmark.total == 1
    assert benchmark.measurements[0].run_id == source.run_id

    validation_path = tmp_path / "validation.json"
    shutil.copy2(FIXTURE, validation_path)
    registered = KernelValidationRegistrationService(workspace).register(
        RegisterKernelValidationRequest(
            run_id=source.run_id,
            expected_run_revision=source.revision,
            path=validation_path,
            pipeline_id=compiled.pipeline_ids[0],
        )
    )
    extracted = KernelValidationExtractor(workspace).extract(source.run_id)
    repeated = KernelValidationExtractor(workspace).extract(source.run_id)

    run = RunStore(workspace).read(source.run_id)
    pipeline = ArtifactPipelineService(workspace).get(registered.pipeline_ids[0]).pipeline
    validation_registration = next(
        item for item in run.artifacts if item.role == "kernel_validation"
    )
    validation_artifact = ArtifactStore(workspace).get(validation_registration.artifact_id)

    assert registered.run_id == source.run_id
    assert registered.run_revision == source.revision + 1
    assert registered.workload_definition_id == source.workload_definition_id
    assert registered.workload_instance_id == source.workload_instance_id
    assert registered.environment_id == source.environment_id
    assert registered.source_state_id == source.source_state_id
    assert registered.execution_identity_id == source.execution_identity.identity_id
    assert run.measurement_protocol_id == source.measurement_protocol_id
    assert run.semantics.semantic_id == source.semantics.semantic_id
    assert tuple(item.artifact_id for item in run.artifacts[: len(source.artifacts)]) == (
        source_artifact_ids
    )
    assert validation_artifact.content.artifact_id == validation_registration.artifact_id
    assert extracted.artifact_id == validation_registration.artifact_id
    assert extracted.case_count == 1
    assert extracted.metric_count == 2
    assert repeated.corpus_commit_id == extracted.corpus_commit_id
    assert [stage.name for stage in pipeline.stages] == [
        "cute_dsl_ir",
        "ptx",
        "cubin",
        "kernel_validation",
    ]
    assert pipeline.run_id == source.run_id
    assert pipeline.stages[-1].artifact_id == validation_registration.artifact_id
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute(
            "SELECT count(*) FROM kernel_validation_metrics WHERE run_id = ?",
            (source.run_id,),
        ).fetchone() == (2,)


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


@pytest.mark.anyio
async def test_register_kernel_validation_rejects_a_pipeline_from_another_run(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_run_id, source_revision = await _capture(workspace)
    other_run_id, _ = await _capture(workspace)
    other_run = RunStore(workspace).read(other_run_id)
    other_pipeline = ArtifactPipelineService(workspace).register(
        RegisterPipelineRequest(
            run_id=other_run_id,
            pipeline_name="other-run",
            producer="fixture",
            stages=(
                RegisteredPipelineStageDeclaration(
                    name="execution_output",
                    ordinal=0,
                    status=PipelineStageStatus.AVAILABLE,
                    registration_id=other_run.artifacts[0].registration_id,
                    format=other_run.artifacts[0].media_type,
                    format_schema="raw",
                ),
            ),
        )
    )
    validation_path = tmp_path / "validation.json"
    shutil.copy2(FIXTURE, validation_path)

    with pytest.raises(DomainError) as error:
        KernelValidationRegistrationService(workspace).register(
            RegisterKernelValidationRequest(
                run_id=source_run_id,
                expected_run_revision=source_revision,
                path=validation_path,
                pipeline_id=other_pipeline.pipeline_id,
            )
        )

    source_run = RunStore(workspace).read(source_run_id)
    assert error.value.code is ErrorCode.INVALID_ARGUMENTS
    assert source_run.revision == source_revision
    assert not any(item.role == "kernel_validation" for item in source_run.artifacts)


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
