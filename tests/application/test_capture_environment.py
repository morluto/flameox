from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application.capture import CaptureService
from flameox.application.discovery import (
    RunDiscoveryService,
    RunFilter,
)
from flameox.application.execution_identity import ExecutionIdentityService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.catalog import Catalog
from flameox.domain import (
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ExternalExecutionContext,
    Sensitivity,
)
from flameox.storage import Workspace
from tests.support.capture import disable_containment

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_external_execution_context_is_bound_preserved_and_discoverable(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.fail]
argv = ["python", "-c", "raise SystemExit(3)"]
"""
    )
    disable_containment(workspace)
    context = ExternalExecutionContext(
        orchestrator="crabbox",
        provider="runpod",
        lease_id="lease-123",
        worker_id="worker-7",
        orchestration_run_id="validation-9",
        sensitivity=Sensitivity.SENSITIVE,
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="fail",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        external_context=context,
    )
    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.external_context == context
    discovered = RunDiscoveryService(workspace).list(
        filter=RunFilter(provider="runpod", lease_id="lease-123"),
        limit=10,
    )
    assert [item.run_id for item in discovered.runs] == [result.run.run_id]
    assert discovered.runs[0].worker_id == "worker-7"

    replacement = context.model_copy(update={"lease_id": "lease-replaced"})
    second = await service.plan(
        workload_name="fail",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        external_context=context,
    )
    with pytest.raises(DomainError) as tampered:
        await service.plans.issue(second.model_copy(update={"external_context": replacement}))
    assert tampered.value.code is ErrorCode.REVISION_CONFLICT

    with pytest.raises(ValueError):
        ExternalExecutionContext(
            orchestrator="crabbox",
            provider="runpod",
            lease_id="contains secret whitespace",
            worker_id="worker",
            orchestration_run_id="run",
        )
    with pytest.raises(ValueError):
        ExternalExecutionContext.model_validate(
            {**context.model_dump(mode="json"), "sensitivity": Sensitivity.NORMAL}
        )


@pytest.mark.anyio
async def test_declared_module_and_native_file_identity_is_observed_and_revalidated(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "project_module.py").write_text("VALUE = 1\n")
    (tmp_path / "lib").mkdir()
    native = tmp_path / "lib" / "extension.so"
    native.write_bytes(b"candidate-a")
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.identity]
argv = ["python", "-c", "import project_module; print(project_module.VALUE)"]
[workloads.identity.identity]
python_modules = ["project_module"]
native_files = ["lib/extension.so"]
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="identity",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    assert plan.planned_execution_identity.quality == "exact"
    result = await service.execute(plan.plan_token)

    identity = result.run.execution_identity
    assert identity is not None
    assert identity.quality == "exact"
    module, library = identity.inputs
    assert module.requested == "project_module"
    assert module.identity_basis == "project_source"
    assert module.resolved_path == str(tmp_path / "project_module.py")
    assert module.loaded_path is None
    assert module.content_digest is not None
    assert "runtime import use is not observed" in module.limitations[0]
    assert library.identity_basis == "explicit_file"
    assert library.configured_path == str(native)
    assert library.resolved_path == str(native)
    assert library.content_digest is not None
    assert library.loaded_path is None

    first_digest = library.content_digest
    native.write_bytes(b"candidate-b")
    changed = ExecutionIdentityService(workspace).plan("identity")
    assert changed.inputs[1].content_digest != first_digest

    stale_plan = await service.plan(
        workload_name="identity",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    native.write_bytes(b"candidate-c")
    with pytest.raises(DomainError) as stale:
        await service.execute(stale_plan.plan_token)
    assert stale.value.code is ErrorCode.INVALID_CAPTURE_PLAN


@pytest.mark.anyio
async def test_python_module_identity_never_executes_declared_module(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    sentinel = tmp_path / "identity-probe-imported"
    (tmp_path / "hostile_module.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('imported')\n"
        "raise RuntimeError('private-target-exception')\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.identity]
argv = ["python", "-c", "print('workload does not import the declared module')"]
[workloads.identity.identity]
python_modules = ["hostile_module"]
"""
    )

    identity = await ExecutionIdentityService(workspace).observe(
        "identity",
        parameters={},
    )

    assert not sentinel.exists()
    assert identity.quality == "exact"
    assert identity.inputs[0].identity_basis == "project_source"
    assert identity.inputs[0].loaded_path is None
    assert "private-target-exception" not in repr(identity)


@pytest.mark.anyio
async def test_python_module_identity_uses_interpreter_and_distribution_metadata(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.identity]
argv = ["python", "-c", "print('ok')"]
[workloads.identity.identity]
python_modules = ["json", "pydantic"]
"""
    )

    identity = await ExecutionIdentityService(workspace).observe(
        "identity",
        parameters={},
    )

    standard_library, distribution = identity.inputs
    assert identity.quality == "exact"
    assert standard_library.identity_basis == "interpreter_stdlib"
    assert standard_library.content_digest is not None
    assert standard_library.resolved_path is None
    assert standard_library.loaded_path is None
    assert distribution.identity_basis == "distribution_metadata"
    assert distribution.distribution == "pydantic"
    assert distribution.version is not None
    assert distribution.content_digest is not None
    assert distribution.resolved_path is None
    assert distribution.loaded_path is None


def test_missing_declared_native_identity_is_explicitly_partial(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.identity]
argv = ["python", "-c", "print('ok')"]
[workloads.identity.identity]
native_files = ["build/missing.so"]
"""
    )

    identity = ExecutionIdentityService(workspace).plan("identity")

    assert identity.quality == "partial"
    assert identity.missing_inputs == ("build/missing.so",)
    assert identity.inputs[0].status == "missing"


@pytest.mark.anyio
async def test_declared_missing_accelerator_downgrades_captured_environment_identity(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.cuda]
argv = ["python", "-c", "print('bounded')"]
[workloads.cuda.identity.environment]
required = ["cuda.driver", "cuda.runtime", "cuda.devices", "cuda.peer_topology"]
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="cuda",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    with Catalog(workspace).open_snapshot() as snapshot:
        row = snapshot.execute(
            "SELECT identity_quality, missing_fields, fields_json "
            "FROM environments WHERE environment_id = ?",
            (result.run.environment_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "partial"
    assert set(row[1]) == {
        "cuda.driver",
        "cuda.runtime",
        "cuda.devices",
        "cuda.peer_topology",
    }
    assert '"status":"missing"' in row[2]
