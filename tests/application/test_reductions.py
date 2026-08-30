from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox import __version__
from flameox.application.artifacts import ArtifactService
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.application.projections import ProjectionCoordinator
from flameox.application.provider_runtime import (
    ProviderRuntime,
    ProviderRuntimeReceipt,
)
from flameox.application.reduction_contracts import ReductionMinimality
from flameox.application.reductions import (
    PlanReductionRequest,
    ReductionLimits,
    ReductionResult,
    ReductionService,
)
from flameox.domain import ArtifactKind, CapabilityExtra, DomainError, ErrorCode, digest_model
from flameox.storage import ArtifactStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_executable(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o755)


def _provider(tmp_path: Path) -> ProviderRuntime:
    root = tmp_path / "provider"
    scripts = root / "bin"
    scripts.mkdir(parents=True)
    python = scripts / "python"
    python.write_text(f'#!/bin/sh\nexec {Path(sys.executable)} "$@"\n')
    python.chmod(0o755)
    shrinkray = scripts / "shrinkray"
    _write_executable(
        shrinkray,
        """
import os, pathlib, shlex, subprocess, sys
args = sys.argv[1:]
target = pathlib.Path(args[-1])
test = shlex.split(args[-2])
history = pathlib.Path(os.environ['SHRINKRAY_DIRECTORY']) / '.shrinkray' / 'run'
(history / 'initial').mkdir(parents=True)
(history / 'initial' / target.name).write_bytes(target.read_bytes())
initial = subprocess.run([*test, str(target)], check=False).returncode
if initial != 0:
    raise SystemExit(1)
lines = target.read_bytes().splitlines(keepends=True)
for candidate in (b''.join(line for line in lines if b'KEEP' in line), b''):
    candidate_path = target.with_name(target.stem + '-candidate' + target.suffix)
    candidate_path.write_bytes(candidate)
    outcome = subprocess.run([*test, str(candidate_path)], check=False).returncode
    if outcome == 0:
        target.write_bytes(candidate)
        reduction = history / 'reductions' / '0001'
        reduction.mkdir(parents=True, exist_ok=True)
        (reduction / target.name).write_bytes(candidate)
        break
print('fake shrinkray completed')
raise SystemExit(0)
""".strip(),
    )
    bridge = scripts / "flameox-reduction-predicate"
    _write_executable(
        bridge,
        "from flameox.workers.reduction_predicate import main\nraise SystemExit(main())\n",
    )
    receipt = ProviderRuntimeReceipt(
        environment_id="sha256:" + "a" * 64,
        flameox_version=__version__,
        flameox_package_source="index",
        extra=CapabilityExtra.REDUCTION,
        requirement="shrinkray==26.7.8.0",
        python_requirement=f"{sys.version_info.major}.{sys.version_info.minor}",
        platform=sys.platform,
        architecture="test",
        uv_version="test",
        uv_sha256="sha256:" + "b" * 64,
        python_relative_path="bin/python",
        python_sha256=_sha256(python),
        distributions={"flameox": __version__, "shrinkray": "26.7.8.0"},
        executable_relative_path="bin/shrinkray",
        executable_sha256=_sha256(shrinkray),
    )
    return ProviderRuntime(root, receipt)


def _configure(project: Path, predicate_code: str) -> None:
    (project / "flameox.toml").write_text(
        f"""
[workloads.predicate]
argv = ["python", "-c", {json.dumps(predicate_code)}]
cwd = "."
timeout_seconds = 30
"""
    )


def _original(
    workspace: Workspace,
    path: Path,
    content: bytes,
    *,
    kind: ArtifactKind = ArtifactKind.COLLECTOR_METADATA,
    media_type: str | None = None,
) -> tuple[str, str, str]:
    path.write_bytes(content)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=path, kind=kind, media_type=media_type)
    )
    registration = imported.run.artifacts[0]
    return imported.artifact_id, imported.run.run_id, registration.registration_id


def _service(tmp_path: Path, predicate_code: str) -> tuple[Workspace, ReductionService]:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, predicate_code)
    return workspace, ReductionService(workspace, provider_runtime=_provider(tmp_path))


def test_reduction_contract_rejects_parallel_candidate_execution() -> None:
    with pytest.raises(ValidationError):
        ReductionLimits(parallelism=2)  # type: ignore[arg-type]


def test_planning_requires_an_exact_managed_shrinkray_provider(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "raise SystemExit(0)")
    _artifact_id, source_run_id, source_registration_id = _original(
        workspace, tmp_path / "input", b"KEEP\n"
    )

    with pytest.raises(DomainError) as failure:
        ReductionService(workspace).plan(
            PlanReductionRequest(
                source_run_id=source_run_id,
                source_registration_id=source_registration_id,
                predicate_workload="predicate",
            )
        )

    assert failure.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert "start_capability_setup" in " ".join(failure.value.remediation)


def test_planning_rejects_a_registration_not_owned_by_the_source_run(tmp_path: Path) -> None:
    workspace, service = _service(tmp_path, "raise SystemExit(0)")
    _artifact_id, source_run_id, _registration_id = _original(
        workspace, tmp_path / "input", b"KEEP\n"
    )

    with pytest.raises(DomainError) as failure:
        service.plan(
            PlanReductionRequest(
                source_run_id=source_run_id,
                source_registration_id="forged-registration",
                predicate_workload="predicate",
            )
        )

    assert failure.value.code is ErrorCode.ARTIFACT_NOT_FOUND


@pytest.mark.anyio
async def test_shrinkray_reduction_preserves_receipts_and_revalidates_final_candidate(
    tmp_path: Path,
) -> None:
    workspace, service = _service(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if b'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_bytes() else 1)",
    )
    original_id, source_run_id, source_registration_id = _original(
        workspace,
        tmp_path / "original.data",
        b"discard\nKEEP\ndiscard\n",
        kind=ArtifactKind.PROCESS_OUTPUT,
        media_type="text/plain",
    )
    plan = service.plan(
        PlanReductionRequest(
            source_run_id=source_run_id,
            source_registration_id=source_registration_id,
            predicate_workload="predicate",
            limits=ReductionLimits(
                max_attempts=16,
                max_staging_files=64,
                predicate_repetitions=2,
            ),
        )
    )

    result = await service.execute(plan.plan_id)

    assert plan.engine == "shrinkray"
    assert plan.shrinkray_version == "26.7.8.0"
    assert result.disposition == "succeeded"
    assert result.final_revalidation_status == "interesting"
    assert result.minimality is ReductionMinimality.NOT_CLAIMED
    assert result.final_artifact_id is not None
    assert result.source_run_id == source_run_id
    assert result.source_registration_id == source_registration_id
    assert result.reduced_registration_id is not None
    assert result.attempt_receipts_artifact_id is not None
    assert result.shrinkray_history_artifact_id is not None
    assert result.reducer_stdout_artifact_id is not None
    assert result.final_predicate_stdout_artifact_id is not None
    assert result.repeatability_status == "consistent"
    assert result.cleanup_complete is True
    final = ArtifactStore(workspace).get(result.final_artifact_id)
    assert final.payload_path.read_bytes() == b"KEEP\n"
    assert final.content.payload_name.endswith(".data")
    metadata = ArtifactService(workspace).get(result.final_artifact_id)
    reduced_registration = next(
        item
        for item in metadata.registrations
        if item.registration_id == result.reduced_registration_id
    )
    assert reduced_registration.run_id == source_run_id
    assert reduced_registration.display_name == "original.reduced.data"
    assert reduced_registration.kind is ArtifactKind.PROCESS_OUTPUT
    assert reduced_registration.media_type == "text/plain"
    assert reduced_registration.role == "reduction_final"
    assert reduced_registration.producer == "shrinkray"
    assert metadata.reduction_provenance[0].reduction_id == result.reduction_id
    assert metadata.reduction_provenance[0].original_artifact_id == original_id
    assert metadata.reduction_provenance[0].role == "final"
    assert metadata.total_reductions == 1
    assert metadata.reduction_provenance_next_cursor is None
    preview = ArtifactService(workspace).preview_text(
        result.final_artifact_id,
        offset=0,
        max_bytes=64,
        max_lines=10,
    )
    assert preview.text == "KEEP\n"
    receipts = ArtifactStore(workspace).get(result.attempt_receipts_artifact_id)
    parsed = [json.loads(line) for line in receipts.payload_path.read_text().splitlines()]
    assert [item["attempt_id"] for item in parsed] == [
        f"attempt-{index:08d}" for index in range(len(parsed))
    ]
    assert all(item["classification"] != "unresolved" for item in parsed)
    assert service.get(result.reduction_id) == result
    head_before_reuse = workspace.corpus.read_head().commit_id
    assert await service.execute(plan.plan_id) == result
    assert workspace.corpus.read_head().commit_id == head_before_reuse
    assert (
        sum(
            item.registration_id == result.reduced_registration_id
            for item in service.runs.read(source_run_id).artifacts
        )
        == 1
    )


@pytest.mark.anyio
async def test_failed_reduction_operation_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, service = _service(tmp_path, "raise SystemExit(0)")
    _original_id, source_run_id, source_registration_id = _original(
        workspace, tmp_path / "retry.data", b"KEEP\n"
    )
    plan = service.plan(
        PlanReductionRequest(
            source_run_id=source_run_id,
            source_registration_id=source_registration_id,
            predicate_workload="predicate",
            limits=ReductionLimits(max_attempts=16, max_staging_files=64),
        )
    )
    execute = service._execute_shrinkray
    attempts = 0

    async def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DomainError(
                ErrorCode.PROCESS_FAILED, "transient provider failure", retryable=True
            )
        return await execute(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service, "_execute_shrinkray", fail_once)

    with pytest.raises(DomainError):
        await service.execute(plan.plan_id)
    result = await service.execute(plan.plan_id)

    assert result.reduction_id == digest_model({"operation": "reduction", "plan_id": plan.plan_id})
    assert attempts == 2


@pytest.mark.anyio
@pytest.mark.parametrize("failed_projection", ["_register_candidate", "_publish_rows"])
async def test_completed_reduction_reconciles_projection_failures_without_rerunning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_projection: str,
) -> None:
    workspace, service = _service(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if b'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_bytes() else 1)",
    )
    _original_id, source_run_id, source_registration_id = _original(
        workspace,
        tmp_path / "reconcile.data",
        b"discard\nKEEP\ndiscard\n",
        kind=ArtifactKind.PROCESS_OUTPUT,
        media_type="text/plain",
    )
    plan = service.plan(
        PlanReductionRequest(
            source_run_id=source_run_id,
            source_registration_id=source_registration_id,
            predicate_workload="predicate",
            limits=ReductionLimits(max_attempts=16, max_staging_files=64),
        )
    )
    execute = service._execute_shrinkray
    provider_calls = 0

    async def count_provider_calls(*args: object, **kwargs: object) -> object:
        nonlocal provider_calls
        provider_calls += 1
        return await execute(*args, **kwargs)  # type: ignore[arg-type]

    projection = getattr(service, failed_projection)
    projection_calls = 0

    def fail_projection_once(*args: object, **kwargs: object) -> object:
        nonlocal projection_calls
        projection_calls += 1
        if projection_calls == 1:
            raise DomainError(ErrorCode.INTERNAL_ERROR, "injected projection failure")
        return projection(*args, **kwargs)

    monkeypatch.setattr(service, "_execute_shrinkray", count_provider_calls)
    monkeypatch.setattr(service, failed_projection, fail_projection_once)

    with pytest.raises(DomainError):
        await service.execute(plan.plan_id)
    result = await service.execute(plan.plan_id)

    assert provider_calls == 1
    assert result.reduced_registration_id is not None
    assert (
        sum(
            item.registration_id == result.reduced_registration_id
            for item in service.runs.read(source_run_id).artifacts
        )
        == 1
    )
    assert result.final_artifact_id is not None
    metadata = ArtifactService(workspace).get(result.final_artifact_id)
    assert metadata.total_reductions == 1


@pytest.mark.anyio
async def test_completed_reduction_reconciles_a_run_projection_after_domain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, service = _service(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if b'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_bytes() else 1)",
    )
    _original_id, source_run_id, source_registration_id = _original(
        workspace,
        tmp_path / "projection.data",
        b"discard\nKEEP\ndiscard\n",
        kind=ArtifactKind.PROCESS_OUTPUT,
        media_type="text/plain",
    )
    plan = service.plan(
        PlanReductionRequest(
            source_run_id=source_run_id,
            source_registration_id=source_registration_id,
            predicate_workload="predicate",
            limits=ReductionLimits(max_attempts=16, max_staging_files=64),
        )
    )
    execute = service._execute_shrinkray
    provider_calls = 0
    injected = False

    async def count_provider_calls(*args: object, **kwargs: object) -> object:
        nonlocal provider_calls
        provider_calls += 1
        return await execute(*args, **kwargs)  # type: ignore[arg-type]

    def fail_after_domain_commit(phase: object, _intent: object) -> None:
        nonlocal injected
        if phase == "after_domain_commit" and not injected:
            injected = True
            raise DomainError(ErrorCode.INTERNAL_ERROR, "injected projection failure")

    service.projections = ProjectionCoordinator(
        workspace,
        fault_injector=fail_after_domain_commit,
    )
    monkeypatch.setattr(service, "_execute_shrinkray", count_provider_calls)

    with pytest.raises(DomainError):
        await service.execute(plan.plan_id)
    result = await service.execute(plan.plan_id)

    assert provider_calls == 1
    assert result.final_artifact_id is not None
    metadata = ArtifactService(workspace).get(result.final_artifact_id)
    assert any(
        item.registration_id == result.reduced_registration_id for item in metadata.registrations
    )


@pytest.mark.anyio
async def test_reduction_marks_an_unqualified_binary_candidate_explicitly(
    tmp_path: Path,
) -> None:
    workspace, service = _service(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if b'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_bytes() else 1)",
    )
    _original_id, source_run_id, source_registration_id = _original(
        workspace,
        tmp_path / "opaque.bin",
        b"discard\nKEEP\ndiscard\n",
    )
    plan = service.plan(
        PlanReductionRequest(
            source_run_id=source_run_id,
            source_registration_id=source_registration_id,
            predicate_workload="predicate",
            limits=ReductionLimits(max_attempts=16, max_staging_files=64),
        )
    )

    result = await service.execute(plan.plan_id)

    assert result.final_artifact_id is not None
    metadata = ArtifactService(workspace).get(result.final_artifact_id)
    assert metadata.registrations[0].kind is ArtifactKind.REDUCED_CANDIDATE
    assert metadata.registrations[0].media_type == "application/octet-stream"
    assert any("not requalified" in item for item in result.limitations)


@pytest.mark.anyio
async def test_artifact_reduction_lineage_is_cursor_paginated(tmp_path: Path) -> None:
    workspace, service = _service(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if b'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_bytes() else 1)",
    )
    _original_id, source_run_id, source_registration_id = _original(
        workspace,
        tmp_path / "lineage.data",
        b"discard\nKEEP\ndiscard\n",
        kind=ArtifactKind.PROCESS_OUTPUT,
        media_type="text/plain",
    )
    results = []
    for max_attempts in (16, 17):
        plan = service.plan(
            PlanReductionRequest(
                source_run_id=source_run_id,
                source_registration_id=source_registration_id,
                predicate_workload="predicate",
                limits=ReductionLimits(
                    max_attempts=max_attempts,
                    max_staging_files=64,
                ),
            )
        )
        results.append(await service.execute(plan.plan_id))

    artifact_id = results[0].final_artifact_id
    assert artifact_id is not None
    assert results[1].final_artifact_id == artifact_id
    artifacts = ArtifactService(workspace)
    metadata = artifacts.get(artifact_id, limit=1)

    assert metadata.total_reductions == 2
    assert metadata.reduction_provenance_next_cursor is not None
    continuation = artifacts.list_reductions(
        artifact_id,
        limit=1,
        cursor=metadata.reduction_provenance_next_cursor,
    )
    assert continuation.next_cursor is None
    assert {
        metadata.reduction_provenance[0].reduction_id,
        continuation.reductions[0].reduction_id,
    } == {result.reduction_id for result in results}


@pytest.mark.anyio
async def test_unresolved_candidate_is_recorded_but_never_adopted(tmp_path: Path) -> None:
    workspace, service = _service(
        tmp_path,
        "import os,pathlib,time; data=pathlib.Path("
        "os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_bytes(); "
        "time.sleep(0 if b'KEEP' in data and b'discard' in data else 1); "
        "raise SystemExit(0 if b'KEEP' in data and b'discard' in data else 1)",
    )
    original_id, source_run_id, source_registration_id = _original(
        workspace,
        tmp_path / "unresolved.data",
        b"discard\nKEEP\n",
    )
    plan = service.plan(
        PlanReductionRequest(
            source_run_id=source_run_id,
            source_registration_id=source_registration_id,
            predicate_workload="predicate",
            limits=ReductionLimits(
                max_attempts=16,
                max_staging_files=64,
                predicate_timeout_seconds=0.3,
            ),
        )
    )

    result = await service.execute(plan.plan_id)

    assert result.disposition == "unchanged"
    assert result.final_artifact_id == original_id
    assert result.attempts.unresolved >= 1


@pytest.mark.anyio
async def test_execution_refuses_changed_provider_bytes(tmp_path: Path) -> None:
    workspace, service = _service(tmp_path, "raise SystemExit(0)")
    _original_id, source_run_id, source_registration_id = _original(
        workspace, tmp_path / "changed.data", b"KEEP\n"
    )
    plan = service.plan(
        PlanReductionRequest(
            source_run_id=source_run_id,
            source_registration_id=source_registration_id,
            predicate_workload="predicate",
            limits=ReductionLimits(max_attempts=16, max_staging_files=64),
        )
    )
    service._provided_runtime.executable.write_text("changed")  # type: ignore[union-attr]

    with pytest.raises(DomainError) as failure:
        await service.execute(plan.plan_id)

    assert failure.value.code is ErrorCode.EXECUTION_REFUSED


def test_result_rejects_a_minimality_claim_from_tool_completion() -> None:
    with pytest.raises(ValidationError):
        ReductionResult.model_validate(
            {
                "reduction_id": "r",
                "plan_id": "p",
                "disposition": "unchanged",
                "source_run_id": "run",
                "source_registration_id": "registration",
                "original_artifact_id": "sha256:" + "0" * 64,
                "predicate_definition_id": "d",
                "predicate_instance_id": "i",
                "attempts": {
                    "attempted": 0,
                    "passed": 0,
                    "failed": 0,
                    "unresolved": 0,
                    "contradictory": 0,
                    "timed_out": 0,
                },
                "cleanup_complete": True,
                "provider_environment_id": "provider",
                "provider_python_digest": "python",
                "shrinkray_executable_digest": "tool",
                "predicate_bridge_digest": "bridge",
                "original_size_bytes": 1,
                "minimality": "one_minimal",
                "final_revalidation_status": "interesting",
                "predicate_repetitions": 1,
                "repeatability_status": "not_assessed",
                "staging_byte_limit": 1024,
                "retained_candidate_byte_limit": 1024,
            }
        )
