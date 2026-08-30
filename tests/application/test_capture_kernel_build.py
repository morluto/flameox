"""Kernel-build capture lifecycle integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flameox.application.capabilities import CapabilityService
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.pipelines import (
    ArtifactPipeline,
    ArtifactPipelineService,
)
from flameox.application.triton_autotune import TritonAutotuneService
from flameox.cli import app
from flameox.domain import (
    ArtifactKind,
    CapabilityReport,
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
)
from flameox.storage import ArtifactStore, Workspace
from tests.support.capture import disable_containment

pytestmark = [pytest.mark.integration, pytest.mark.process]


class _VersionedCompilerCapabilities(CapabilityService):
    def __init__(self, workspace: Workspace) -> None:
        pytest.importorskip("triton")
        super().__init__(workspace)

    def get(self, adapter: str) -> CapabilityReport:
        report = super().get(adapter)
        if adapter == "cute.compiler":
            return report.model_copy(update={"version": "fixture-compiler-1"})
        return report


def _write_workload(
    project: Path,
    *,
    script: str,
    environment: dict[str, str] | None = None,
) -> None:
    env_section = ""
    if environment:
        env_lines = "\n".join(f'  {key} = "{value}"' for key, value in environment.items())
        env_section = f"\n[workloads.compile.environment]\n{env_lines}"
    (project / "flameox.toml").write_text(
        f"""

[workloads.compile]
argv = ["python", "{script}"]
cwd = "."
timeout_seconds = 30{env_section}
"""
    )


def _dump_script(path: Path, *, exit_code: int = 0) -> None:
    """A fake compiler that writes IR files to TRITON_DUMP_DIR and exits."""
    path.write_text(
        f"""
import os, sys, pathlib
dump = pathlib.Path(os.environ["TRITON_DUMP_DIR"])
dump.mkdir(parents=True, exist_ok=True)
(dump / "kernel.ttir").write_text("ttir content")
(dump / "kernel.ptx").write_text("ptx content")
(dump / "kernel.cubin").write_bytes(b"\\x00\\x01\\x02\\x03")
sys.exit({exit_code})
"""
    )


def _two_group_dump_script(path: Path) -> None:
    path.write_text(
        """
import os, pathlib
dump = pathlib.Path(os.environ["TRITON_DUMP_DIR"])
for group in ("source-hash-a", "source-hash-b"):
    root = dump / group
    root.mkdir(parents=True, exist_ok=True)
    for extension in ("ttir", "ttgir", "llir", "ptx", "cubin", "sass"):
        payload = f"{group}:{extension}".encode()
        (root / f"kernel.{extension}").write_bytes(payload)
"""
    )


def _write_identity_workloads(project: Path, script: Path, *, changed_argv: bool = False) -> None:
    suffix = ', "--changed"' if changed_argv else ""
    (project / "flameox.toml").write_text(
        f"""

[workloads.compile]
argv = ["python", "{script}", "{{size}}"{suffix}]
cwd = "."
timeout_seconds = 30

[workloads.compile.parameters]
size = [1, 2]

[workloads.renamed]
argv = ["python", "{script}", "{{size}}"]
cwd = "."
timeout_seconds = 30

[workloads.renamed.parameters]
size = [1, 2]
"""
    )


@pytest.mark.anyio
async def test_capture_plan_rejects_conflicting_compiler_environment(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script)
    _write_workload(
        tmp_path,
        script=str(script),
        environment={"TRITON_DUMP_DIR": "/elsewhere"},
    )
    disable_containment(workspace)
    service = CaptureService(
        workspace,
        capabilities=_VersionedCompilerCapabilities(workspace),
    )
    with pytest.raises(DomainError) as error:
        await service.plan(
            workload_name="compile",
            adapter="triton.compiler",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "conflict" in error.value.message.lower()


@pytest.mark.anyio
async def test_capture_plan_accepts_identical_compiler_environment(
    tmp_path: Path,
) -> None:
    pytest.importorskip("triton")
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script)
    _write_workload(
        tmp_path,
        script=str(script),
        environment={"TRITON_KERNEL_DUMP": "1"},
    )
    disable_containment(workspace)

    plan = await CaptureService(workspace).plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert plan.collector_environment["TRITON_KERNEL_DUMP"] == "1"


@pytest.mark.anyio
async def test_triton_compiler_capture_emits_manifest_with_native_artifacts(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script)
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(
        workspace,
        capabilities=_VersionedCompilerCapabilities(workspace),
    )
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        adapter_options={"target": {"backend": "cuda", "architecture": "sm_86", "warp_size": 32}},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    manifest_reg = next(reg for reg in result.run.artifacts if reg.role == "kernel_build_manifest")
    assert manifest_reg.kind is ArtifactKind.KERNEL_BUILD
    native_regs = [reg for reg in result.run.artifacts if reg.role == "compiler_output"]
    assert len(native_regs) == 3
    extensions = {reg.display_name.rsplit(".", 1)[1] for reg in native_regs}
    assert extensions == {"ttir", "ptx", "cubin"}
    media_types = {reg.display_name.rsplit(".", 1)[1]: reg.media_type for reg in native_regs}
    assert media_types == {
        "ttir": "text/plain",
        "ptx": "text/plain",
        "cubin": "application/octet-stream",
    }
    pipelines = ArtifactPipelineService(workspace).pipelines.list()
    assert len(pipelines) == 1
    assert result.pipeline_ids == (pipelines[0].pipeline_id,)
    assert pipelines[0].run_id == result.run.run_id
    assert pipelines[0].identity_quality == "managed_partial"
    assert pipelines[0].compiler_identity_id is not None
    assert pipelines[0].target_identity_id is None
    assert "Compiler target identity was unavailable" in pipelines[0].limitations[0]
    assert {
        "workload_definition_id",
        "workload_instance_id",
        "command_digest",
        "environment_id",
        "source_state_id",
        "build_protocol_id",
    }.isdisjoint(pipelines[0].model_dump())
    assert {stage.artifact_id for stage in pipelines[0].stages} == {
        registration.artifact_id for registration in native_regs
    }
    assert [stage.predecessor for stage in pipelines[0].stages] == [None, "ttir", "ptx"]
    artifact = ArtifactStore(workspace).get(manifest_reg.artifact_id)
    manifest = json.loads(artifact.payload_path.read_text())
    assert set(manifest) == {"producer", "native_groups", "attachments"}
    assert manifest["producer"] == "triton"
    assert len(manifest["native_groups"]) == 1
    group = manifest["native_groups"][0]
    assert group["path"] == "triton-dumps"
    assert len(group["artifacts"]) == 3
    native_artifacts = {
        registration.display_name: ArtifactStore(workspace)
        .get(registration.artifact_id)
        .payload_path
        for registration in native_regs
    }
    for declaration in group["artifacts"]:
        payload = native_artifacts[declaration["path"]].read_bytes()
        assert declaration["byte_length"] == len(payload)
        assert declaration["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.anyio
async def test_triton_capture_projects_listener_selection_without_cache_or_pipeline_inference(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "triton.py").write_text(
        "class _Autotuning:\n"
        "    listener = None\n"
        "class _Knobs:\n"
        "    autotuning = _Autotuning()\n"
        "knobs = _Knobs()\n"
    )
    script = tmp_path / "compile.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import triton\n"
        "class Config:\n"
        "    def __init__(self, block, warps):\n"
        "        self.kwargs = {'BLOCK': block}\n"
        "        self.num_warps = warps\n"
        "        self.num_stages = 1\n"
        "        self.num_ctas = 1\n"
        "        self.maxnreg = None\n"
        "        self.ir_override = None\n"
        "class Kernel:\n"
        "    pass\n"
        "Kernel.__module__ = 'workload'\n"
        "Kernel.__name__ = 'vector_add'\n"
        "first = Config(128, 4)\n"
        "winner = Config(256, 8)\n"
        "triton.knobs.autotuning.listener(\n"
        "    fn=Kernel, key=(1024,), best_config=winner,\n"
        "    configs_timings={first: [2.0, 1.8, 2.2], winner: [1.0, 0.9, 1.1]},\n"
        "    duration=0.032, cache_hit=False,\n"
        ")\n"
        "triton.knobs.autotuning.listener(\n"
        "    fn=Kernel, key=(2048,), best_config=winner,\n"
        "    configs_timings={first: [2.1, 1.9, 2.3], winner: [1.1, 1.0, 1.2]},\n"
        "    duration=0.016, cache_hit=False,\n"
        ")\n"
        "dump = Path(os.environ['TRITON_DUMP_DIR'])\n"
        "dump.mkdir(parents=True, exist_ok=True)\n"
        "(dump / 'kernel.ttir').write_text('ttir')\n"
    )
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(
        workspace,
        capabilities=_VersionedCompilerCapabilities(workspace),
    )
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    captured = await service.execute(plan.plan_token)
    selections = TritonAutotuneService(workspace).selections(
        run_id=captured.run.run_id,
        limit=1,
    )

    assert captured.run.execution_status is ExecutionStatus.SUCCEEDED
    assert captured.run.semantics.configuration["collector_implementation_id"]
    assert selections.total == 2
    assert selections.returned == 1
    assert selections.next_cursor is not None
    selection = selections.selections[0]
    assert selection.function_name == "workload.vector_add"
    assert selection.cache_hit is False
    assert selection.duration_ms in {16.0, 32.0}
    assert selection.candidate_count == 2
    assert selection.winner_config_id in {item.config_id for item in selection.candidates}
    assert all("cache" not in limitation.lower() for limitation in selection.limitations)
    assert all(item.role != "triton_autotune" for item in captured.run.artifacts)
    assert all("pipeline_id" not in item.model_dump() for item in selection.candidates)
    next_page = TritonAutotuneService(workspace).selections(
        run_id=captured.run.run_id,
        limit=1,
        cursor=selections.next_cursor,
    )
    assert next_page.returned == 1
    assert next_page.next_cursor is None
    assert {selection.duration_ms, next_page.selections[0].duration_ms} == {16.0, 32.0}


@pytest.mark.anyio
async def test_triton_capture_registers_one_pipeline_per_complete_dump_group(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _two_group_dump_script(script)
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(
        workspace,
        capabilities=_VersionedCompilerCapabilities(workspace),
    )
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        adapter_options={"target": {"backend": "cuda", "architecture": "sm_86", "warp_size": 32}},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_token)
    registrations = {item.artifact_id: item.display_name for item in result.run.artifacts}
    pipelines = [
        pipeline
        for pipeline in ArtifactPipelineService(workspace).pipelines.list()
        if pipeline.run_id == result.run.run_id
    ]

    assert len(result.pipeline_ids) == 2
    assert {pipeline.pipeline_id for pipeline in pipelines} == set(result.pipeline_ids)
    pipeline_groups: set[str] = set()
    for pipeline in pipelines:
        assert [stage.name for stage in pipeline.stages] == [
            "ttir",
            "ttgir",
            "llir",
            "ptx",
            "cubin",
            "sass",
        ]
        assert [stage.predecessor for stage in pipeline.stages] == [
            None,
            "ttir",
            "ttgir",
            "llir",
            "ptx",
            "cubin",
        ]
        first_artifact_id = pipeline.stages[0].artifact_id
        assert first_artifact_id is not None
        group_path = registrations[first_artifact_id].rsplit("/", 1)[0]
        pipeline_groups.add(group_path)
        for stage in pipeline.stages:
            assert stage.artifact_id is not None
            assert registrations[stage.artifact_id].startswith(f"{group_path}/")
    assert pipeline_groups == {"triton-dumps/source-hash-a", "triton-dumps/source-hash-b"}


def test_cli_capture_show_and_compare_uses_returned_pipeline_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script)
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    monkeypatch.setattr(
        "flameox.cli.CaptureService",
        lambda selected: CaptureService(
            selected,
            capabilities=_VersionedCompilerCapabilities(selected),
        ),
    )
    runner = CliRunner()
    capture_args = [
        "capture",
        "run",
        "triton.compiler",
        "--workload",
        "compile",
        "--workspace",
        str(workspace.paths.root),
        "--json",
    ]

    baseline = runner.invoke(app, capture_args)
    candidate = runner.invoke(app, capture_args)

    assert baseline.exit_code == 0, baseline.output
    assert candidate.exit_code == 0, candidate.output
    baseline_payload = json.loads(baseline.stdout)
    candidate_payload = json.loads(candidate.stdout)
    baseline_pipeline_id = baseline_payload["pipeline_ids"][0]
    candidate_pipeline_id = candidate_payload["pipeline_ids"][0]
    shown = runner.invoke(
        app,
        [
            "pipelines",
            "show",
            baseline_pipeline_id,
            "--workspace",
            str(workspace.paths.root),
            "--json",
        ],
    )
    compared = runner.invoke(
        app,
        [
            "pipelines",
            "compare",
            baseline_payload["run"]["run_id"],
            candidate_payload["run"]["run_id"],
            "--workspace",
            str(workspace.paths.root),
            "--json",
        ],
    )

    assert shown.exit_code == 0, shown.output
    assert compared.exit_code == 0, compared.output
    assert json.loads(shown.stdout)["compatible_pipeline_ids"] == [candidate_pipeline_id]
    comparison = json.loads(compared.stdout)
    assert comparison["baseline_pipeline_id"] == baseline_pipeline_id
    assert comparison["candidate_pipeline_id"] == candidate_pipeline_id


@pytest.mark.anyio
async def test_triton_compiler_capture_preserves_manifest_on_nonzero_exit(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script, exit_code=1)
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(
        workspace,
        capabilities=_VersionedCompilerCapabilities(workspace),
    )
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)
    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    manifest_regs = [reg for reg in result.run.artifacts if reg.role == "kernel_build_manifest"]
    assert len(manifest_regs) == 1
    native_regs = [reg for reg in result.run.artifacts if reg.role == "compiler_output"]
    assert len(native_regs) == 3


@pytest.mark.anyio
async def test_cute_compiler_capture_emits_manifest(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    script.write_text(
        """
import os, sys, pathlib
dump = pathlib.Path(os.environ["CUTE_DSL_DUMP_DIR"])
dump.mkdir(parents=True, exist_ok=True)
(dump / ("cute_dsl_" + "long_kernel_identity_" * 6 + ".mlir")).write_text("cute ir")
(dump / "kernel.ptx").write_text("ptx")
sys.exit(0)
"""
    )
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="compile",
        adapter="cute.compiler",
        adapter_options={"keep_allowlist": ["ir", "ptx"]},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    native_regs = [reg for reg in result.run.artifacts if reg.role == "compiler_output"]
    assert len(native_regs) == 2
    assert all(len(reg.role) <= 100 for reg in native_regs)
    assert any("long_kernel_identity" in reg.display_name for reg in native_regs)
    pipelines = ArtifactPipelineService(workspace).pipelines.list()
    assert len(pipelines) == 1
    assert {stage.artifact_id for stage in pipelines[0].stages} == {
        registration.artifact_id for registration in native_regs
    }


@pytest.mark.anyio
async def test_managed_kernel_build_comparison_uses_known_identity_without_inventing_target(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script)
    _write_identity_workloads(tmp_path, script)
    disable_containment(workspace)
    service = CaptureService(
        workspace,
        capabilities=_VersionedCompilerCapabilities(workspace),
    )

    async def capture(
        *,
        workload_name: str = "compile",
        size: int = 1,
        architecture: str = "sm_86",
    ) -> ArtifactPipeline:
        plan = await service.plan(
            workload_name=workload_name,
            parameters={"size": size},
            adapter="triton.compiler",
            adapter_options={
                "target": {
                    "backend": "cuda",
                    "architecture": architecture,
                    "warp_size": 32,
                }
            },
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        result = await service.execute(plan.plan_token)
        return next(
            pipeline
            for pipeline in ArtifactPipelineService(workspace).pipelines.list()
            if pipeline.run_id == result.run.run_id
        )

    baseline = await capture()
    exact_repeat = await capture()
    changed_parameter = await capture(size=2)
    renamed = await capture(workload_name="renamed")
    changed_target = await capture(architecture="sm_90")
    _write_identity_workloads(tmp_path, script, changed_argv=True)
    changed_command = await capture()

    comparisons = ArtifactPipelineService(workspace)
    repeat_result = comparisons.compare(baseline.pipeline_id, exact_repeat.pipeline_id)
    parameter_result = comparisons.compare(baseline.pipeline_id, changed_parameter.pipeline_id)
    renamed_result = comparisons.compare(baseline.pipeline_id, renamed.pipeline_id)
    target_result = comparisons.compare(baseline.pipeline_id, changed_target.pipeline_id)
    command_result = comparisons.compare(baseline.pipeline_id, changed_command.pipeline_id)

    assert repeat_result.compatibility == "unknown"
    assert repeat_result.identity_mismatches == ()
    assert parameter_result.compatibility == "incompatible"
    assert parameter_result.identity_mismatches == ("workload_instance_id",)
    assert renamed_result.compatibility == "incompatible"
    assert {"workload_definition_id", "workload_instance_id"}.issubset(
        renamed_result.identity_mismatches
    )
    assert target_result.compatibility == "unknown"
    assert target_result.identity_mismatches == ()
    assert any("target_identity_id" in limitation for limitation in target_result.limitations)
    assert command_result.compatibility == "incompatible"
    assert {"workload_definition_id", "workload_instance_id"}.issubset(
        command_result.identity_mismatches
    )
    assert all(
        result.first_observed_divergent_stage is None
        for result in (parameter_result, renamed_result, target_result, command_result)
    )
