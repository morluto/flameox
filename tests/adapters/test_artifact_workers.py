from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from flameox.adapters.artifact_workers import IsolatedWorkerHarness
from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel
from flameox.storage import Workspace
from flameox.workers.protocol import (
    WorkerDefinition,
    WorkerOperationId,
    WorkerOutputFile,
)

pytestmark = pytest.mark.unit


class _Request(ContractModel):
    value: int


class _Response(ContractModel):
    value: int


_DEFINITION = WorkerDefinition(
    operation=WorkerOperationId.OTLP_PARSE,
    module="example.worker",
    request=TypeAdapter(_Request),
    response=TypeAdapter(_Response),
    name="example",
    implementation="example.worker/v1",
)


def test_typed_worker_response_is_bound_and_exact(tmp_path: Path) -> None:
    worker = IsolatedWorkerHarness(Workspace.initialize(tmp_path))
    response = tmp_path / "response.json"
    response.write_text(
        '{"transport":"flameox.artifact-worker/v1","request_id":"'
        + "a" * 32
        + '","operation":"otlp.parse","implementation":"example.worker/v1",'
        '"kind":"success","payload":{"value":7}}'
    )

    parsed = worker._load_typed_response(
        0,
        b"",
        response,
        definition=_DEFINITION,
        request_id="a" * 32,
    )

    assert parsed == _Response(value=7)


@pytest.mark.parametrize(
    ("exit_code", "request_id", "implementation"),
    [
        (1, "a" * 32, "example.worker/v1"),
        (0, "b" * 32, "example.worker/v1"),
        (0, "a" * 32, "example.worker/v2"),
    ],
)
def test_typed_worker_rejects_exit_or_identity_mismatch(
    tmp_path: Path,
    exit_code: int,
    request_id: str,
    implementation: str,
) -> None:
    worker = IsolatedWorkerHarness(Workspace.initialize(tmp_path))
    response = tmp_path / "response.json"
    response.write_text(
        '{"transport":"flameox.artifact-worker/v1","request_id":"'
        + request_id
        + '","operation":"otlp.parse","implementation":"'
        + implementation
        + '","kind":"success","payload":{"value":7}}'
    )

    with pytest.raises(DomainError):
        worker._load_typed_response(
            exit_code,
            b"",
            response,
            definition=_DEFINITION,
            request_id="a" * 32,
        )


def test_worker_output_file_rejects_symlink(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    worker = IsolatedWorkerHarness(workspace)
    job_root = workspace.paths.staging / "worker-output-proof"
    job_root.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"{}\n")
    (job_root / "projection.jsonl").symlink_to(outside)
    output = WorkerOutputFile(
        role="projection",
        relative_path="projection.jsonl",
        media_type="application/x-ndjson",
        byte_length=3,
        sha256="sha256:" + "0" * 64,
    )

    with pytest.raises(DomainError) as error:
        worker.validate_output_file(job_root, output)

    assert error.value.code is ErrorCode.EXECUTION_REFUSED


def test_worker_side_channel_is_byte_bounded_and_rejects_symlinks(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    worker = IsolatedWorkerHarness(workspace)
    job_root = workspace.paths.staging / "worker-progress-proof"
    job_root.mkdir(parents=True)
    progress = job_root / "progress.json"
    progress.write_bytes(b'{"phase":"reading"}')

    assert (
        worker.read_staged_bytes(job_root, "progress.json", max_bytes=64)
        == progress.read_bytes()
    )

    progress.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    progress.symlink_to(outside)
    with pytest.raises(DomainError) as error:
        worker.read_staged_bytes(job_root, "progress.json", max_bytes=64)

    assert error.value.code is ErrorCode.EXECUTION_REFUSED


def test_worker_execution_applies_explicit_memory_and_staging_growth_limits(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    worker = IsolatedWorkerHarness(workspace)
    job_root = workspace.paths.staging / "worker-resource-proof"

    request = worker._execution_request(
        "example.worker",
        job_root / "request.json",
        job_root / "response.json",
        job_root,
        timeout_seconds=12,
        maximum_rss_bytes=34,
        maximum_writable_growth_bytes=56,
    )

    assert request.timeout_seconds == 12
    assert request.resource_policy is not None
    assert request.resource_policy.maximum_rss_bytes == 34
    assert request.resource_policy.maximum_writable_growth_bytes == 56
