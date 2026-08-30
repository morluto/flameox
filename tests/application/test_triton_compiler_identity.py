from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application.capabilities import CapabilityService
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.domain import CapabilityReport, CapabilityStatus, CompilerIdentity, DomainError
from flameox.storage import Workspace
from tests.support.capture import disable_containment


def _write_workload(project: Path) -> None:
    script = project / "compile.py"
    script.write_text("print('compiler fixture')\n")
    (project / "flameox.toml").write_text(
        f"""
[workloads.compile]
argv = ["python", "{script}"]
cwd = "."
timeout_seconds = 30
"""
    )


def _identity(
    version: str = "3.7.1",
    content_digest: str = "sha256:" + "1" * 64,
) -> CompilerIdentity:
    return CompilerIdentity(
        adapter="triton.compiler",
        distribution="triton",
        version=version,
        content_digest=content_digest,
        interpreter_digest="sha256:" + "2" * 64,
    )


@pytest.mark.anyio
async def test_triton_plan_binds_and_rechecks_exact_interpreter_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    capability = (
        CapabilityService(workspace)
        .get("triton.compiler")
        .model_copy(update={"status": CapabilityStatus.AVAILABLE, "version": "3.7.1"})
    )

    async def adapter_capability(*_args: object, **_kwargs: object) -> CapabilityReport:
        return capability

    async def compiler_identity(workload_name: str) -> CompilerIdentity:
        assert workload_name == "compile"
        return _identity()

    monkeypatch.setattr(service, "_adapter_capability", adapter_capability)
    monkeypatch.setattr(service, "_triton_compiler_identity", compiler_identity)
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert plan.adapter_version == "3.7.1"
    assert plan.compiler_identity == _identity()
    await service._recheck(plan)

    async def changed_identity(_workload_name: str) -> CompilerIdentity:
        return _identity(content_digest="sha256:" + "3" * 64)

    monkeypatch.setattr(service, "_triton_compiler_identity", changed_identity)
    with pytest.raises(DomainError, match="Triton distribution changed"):
        await service._recheck(plan)
