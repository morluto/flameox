from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from flameox.adapters import AdapterDescriptor
from flameox.application import CaptureService, ExecutionPolicy
from flameox.catalog import Catalog
from flameox.domain import (
    ADAPTER_API_VERSION,
    AdapterArtifactDeclaration,
    AdapterExecutionPlan,
    AdapterExtractionResult,
    AdapterPlanRequest,
    AdapterProbeContext,
    AdapterProbeResult,
    AdapterProbeStatus,
    AdapterValidationResult,
    ArtifactKind,
    DomainError,
    ErrorCode,
    ExecutionStatus,
)
from flameox.storage import ArtifactStore, RunStore, Workspace

_WRAPPER = """\
import pathlib
import subprocess
import sys

output = pathlib.Path(sys.argv[1])
separator = sys.argv.index("--")
subprocess.run(sys.argv[separator + 1:], check=True)
output.write_text('samples=7\\n')
"""


class FixtureAdapter:
    name = "fixture"
    api_version = ADAPTER_API_VERSION

    def __init__(
        self,
        *,
        valid: bool = True,
        fail_plan: bool = False,
        fail_extract: bool = False,
    ) -> None:
        self.valid = valid
        self.fail_plan = fail_plan
        self.fail_extract = fail_extract
        self.phases: list[str] = []

    async def probe(self, context: AdapterProbeContext) -> AdapterProbeResult:
        self.phases.append("probe")
        return AdapterProbeResult(
            status=AdapterProbeStatus.AVAILABLE,
            adapter_version="fixture-1",
        )

    async def plan(self, request: AdapterPlanRequest) -> AdapterExecutionPlan:
        self.phases.append("plan")
        if self.fail_plan:
            raise RuntimeError("fixture plan failure")
        return AdapterExecutionPlan(
            adapter=self.name,
            argv_prefix=(
                sys.executable,
                "-c",
                _WRAPPER,
                str(Path(request.output_root) / "fixture.txt"),
            ),
            artifacts=(
                AdapterArtifactDeclaration(
                    relative_path="fixture.txt",
                    kind=ArtifactKind.SAMPLE_PROFILE,
                    role="primary",
                    media_type="text/plain",
                ),
            ),
            permissions=("process_observation",),
            expected_overhead="Fixture wrapper overhead.",
            limitations=("Fixture evidence is intentionally minimal.",),
            extractor_version="extractor-1",
        )

    async def validate(
        self,
        artifact_path: str,
        declaration: AdapterArtifactDeclaration,
    ) -> AdapterValidationResult:
        self.phases.append("validate")
        return AdapterValidationResult(
            valid=self.valid and Path(artifact_path).read_text().startswith("samples="),
            limitations=(() if self.valid else ("invalid fixture",)),
        )

    async def extract(
        self,
        artifact_path: str,
        declaration: AdapterArtifactDeclaration,
    ) -> AdapterExtractionResult:
        self.phases.append("extract")
        if self.fail_extract:
            raise RuntimeError("fixture extraction failure")
        assert ".diagnostics/artifacts/" in artifact_path
        samples = int(Path(artifact_path).read_text().split("=", 1)[1])
        return AdapterExtractionResult(
            extractor_version="extractor-1",
            summary={"samples": samples},
        )


def _workspace(tmp_path: Path, command: str = "print('workload')") -> Workspace:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.fixture]
argv = [{json.dumps(sys.executable)}, "-c", {json.dumps(command)}]
timeout_seconds = 30
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    return workspace


def _install_fixture(
    monkeypatch: pytest.MonkeyPatch,
    adapter: FixtureAdapter,
    *,
    identity: dict[str, str] | None = None,
) -> dict[str, str]:
    current = identity or {"value": "sha256:" + "a" * 64}

    def descriptor(_registry: object, name: str) -> AdapterDescriptor:
        assert name == "fixture"
        return AdapterDescriptor(
            adapter="fixture",
            entry_point="fixture_package:adapter",
            distribution="fixture-package",
            version="1.0",
            package_identity=current["value"],
            approved=True,
        )

    def contract(registry: object, name: str) -> tuple[AdapterDescriptor, FixtureAdapter]:
        return descriptor(registry, name), adapter

    monkeypatch.setattr(
        "flameox.application.capture.AdapterRegistry.approved_descriptor",
        descriptor,
    )
    monkeypatch.setattr(
        "flameox.application.capture.AdapterRegistry.load_contract",
        contract,
    )
    return current


@pytest.mark.anyio
async def test_approved_adapter_runs_full_lifecycle_and_publishes_linked_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    adapter = FixtureAdapter()
    _install_fixture(monkeypatch, adapter)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="fixture",
        adapter="fixture",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )

    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert plan.permissions == ("process_observation",)
    assert plan.adapter_execution_plan is not None
    assert adapter.phases == ["probe", "plan", "validate", "extract"]
    assert len(result.run.artifacts) == 3
    assert any(item.kind.value == "process_tree_snapshot" for item in result.run.artifacts)
    native = next(item for item in result.run.artifacts if item.role == "primary")
    stdout = next(item for item in result.run.artifacts if item.role == "stdout")
    assert ArtifactStore(workspace).get(stdout.artifact_id).payload_path.read_text().strip() == (
        "workload"
    )
    with Catalog(workspace).open_snapshot() as snapshot:
        extraction = snapshot.execute(
            "SELECT input_artifact_id, adapter, adapter_package_identity, "
            "extractor_version, summary_json FROM adapter_extractions WHERE run_id = ?",
            (result.run.run_id,),
        ).fetchone()
    assert extraction == (
        native.artifact_id,
        "fixture",
        "sha256:" + "a" * 64,
        "extractor-1",
        '{"samples":7}',
    )


@pytest.mark.anyio
async def test_adapter_package_change_invalidates_unconsumed_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    adapter = FixtureAdapter()
    identity = _install_fixture(monkeypatch, adapter)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="fixture",
        adapter="fixture",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )
    identity["value"] = "sha256:" + "b" * 64

    with pytest.raises(DomainError) as changed:
        await service.execute(plan.plan_token)

    assert changed.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert RunStore(workspace).list() == ()


@pytest.mark.anyio
async def test_adapter_planning_failure_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    adapter = FixtureAdapter(fail_plan=True)
    _install_fixture(monkeypatch, adapter)

    with pytest.raises(DomainError, match="bounded planning") as failure:
        await CaptureService(workspace).plan(
            workload_name="fixture",
            adapter="fixture",
            execution_policy=ExecutionPolicy.APPROVED_AGENT,
        )

    assert failure.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert not (workspace.paths.staging / "captures").exists()


@pytest.mark.parametrize(
    ("adapter_factory", "message"),
    [
        pytest.param(
            lambda: FixtureAdapter(valid=False),
            "rejected its declared artifact",
            id="validation",
        ),
        pytest.param(
            lambda: FixtureAdapter(fail_extract=True),
            "extraction failed",
            id="extraction",
        ),
    ],
)
@pytest.mark.anyio
async def test_adapter_execution_failures_are_structured_and_leave_no_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_factory: Callable[[], FixtureAdapter],
    message: str,
) -> None:
    workspace = _workspace(tmp_path)
    adapter = adapter_factory()
    _install_fixture(monkeypatch, adapter)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="fixture",
        adapter="fixture",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )
    with pytest.raises(DomainError, match=message) as failure:
        await service.execute(plan.plan_token)

    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert failure.value.run_id == plan.run_id
    assert not (workspace.paths.staging / "captures" / plan.plan_id).exists()


@pytest.mark.anyio
async def test_adapter_capture_cancellation_uses_normal_process_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path, "import time; time.sleep(10)")
    adapter = FixtureAdapter()
    _install_fixture(monkeypatch, adapter)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="fixture",
        adapter="fixture",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )
    task = asyncio.create_task(service.execute(plan.plan_token))
    for _ in range(200):
        try:
            running = service.runs.read(plan.run_id)
        except DomainError:
            await asyncio.sleep(0.01)
            continue
        if running.lease is not None:
            break
        await asyncio.sleep(0.01)
    assert running.lease is not None
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):
        run = service.runs.read(plan.run_id)
        if run.execution_status is ExecutionStatus.CANCELLED:
            break
        await asyncio.sleep(0.05)
    assert run.execution_status is ExecutionStatus.CANCELLED
    assert run.process is not None
    assert run.process.cleanup_complete is True


@pytest.mark.anyio
async def test_adapter_artifact_quota_uses_normal_capture_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "capture": workspace.config.capture.validated_copy(update={"max_artifact_bytes": 1}),
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            ),
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    _install_fixture(monkeypatch, FixtureAdapter())
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="fixture",
        adapter="fixture",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )

    with pytest.raises(DomainError) as limited:
        await service.execute(plan.plan_token)

    assert limited.value.code is ErrorCode.ARTIFACT_TOO_LARGE
    assert service.runs.read(plan.run_id).execution_status is ExecutionStatus.FAILED
