from __future__ import annotations

import json
import secrets
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import JsonValue

from flameox.atomic import atomic_write_json
from flameox.domain import DomainError, ErrorCode
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.storage import Workspace

T = TypeVar("T")


class ArtifactWorker:
    """Run an artifact-facing parser behind the canonical subprocess boundary."""

    def __init__(self, workspace: Workspace, *, broker: SubprocessBroker | None = None) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()

    def run_sync(
        self,
        module: str,
        request: dict[str, JsonValue],
        *,
        name: str,
        timeout_seconds: float = 120,
    ) -> dict[str, Any]:
        job_root = self._job_root(name)
        job_root.mkdir(parents=True, exist_ok=False)
        request_path = job_root / "request.json"
        response_path = job_root / "response.json"
        atomic_write_json(request_path, request)
        try:
            outcome = self.broker.run_sync(
                self._execution_request(
                    module,
                    request_path,
                    response_path,
                    job_root,
                    timeout_seconds=timeout_seconds,
                )
            )
            return self._load_response(
                outcome.process.exit_code,
                outcome.stderr,
                response_path,
                name=name,
            )
        finally:
            shutil.rmtree(job_root, ignore_errors=True)

    async def run(
        self,
        module: str,
        request: dict[str, JsonValue],
        *,
        name: str,
        timeout_seconds: float,
        consume: Callable[[dict[str, Any], Path], T],
    ) -> T:
        job_root = self._job_root(name)
        job_root.mkdir(parents=True, exist_ok=False)
        request_path = job_root / "request.json"
        response_path = job_root / "response.json"
        atomic_write_json(request_path, request)
        try:
            outcome = await self.broker.run(
                self._execution_request(
                    module,
                    request_path,
                    response_path,
                    job_root,
                    timeout_seconds=timeout_seconds,
                )
            )
            payload = self._load_response(
                outcome.process.exit_code,
                outcome.stderr,
                response_path,
                name=name,
            )
            return consume(payload, job_root)
        finally:
            shutil.rmtree(job_root, ignore_errors=True)

    def _job_root(self, name: str) -> Path:
        return self.workspace.paths.staging / "artifact-workers" / f"{name}-{secrets.token_hex(16)}"

    def _execution_request(
        self,
        module: str,
        request_path: Path,
        response_path: Path,
        job_root: Path,
        *,
        timeout_seconds: float,
    ) -> ExecutionRequest:
        return ExecutionRequest(
            argv=(
                sys.executable,
                "-m",
                module,
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ),
            cwd=self.workspace.project_root,
            environment_allowlist=tuple(
                self.workspace.config.execution.child_environment_allowlist
            ),
            allowed_working_roots=(self.workspace.project_root,),
            timeout_seconds=timeout_seconds,
            max_output_bytes=1_048_576,
            resource_policy=ResourcePolicy(
                filesystem_path=self.workspace.paths.root,
                staging_root=self.workspace.paths.staging,
                writable_roots=(job_root,),
                minimum_free_bytes=self.workspace.config.storage.min_free_bytes,
                maximum_rss_bytes=self.workspace.config.execution.max_memory_bytes,
                sampling_interval_ms=self.workspace.config.execution.resource_sampling_interval_ms,
                max_observed_files=self.workspace.config.execution.max_resource_observed_files,
            ),
        )

    def _load_response(
        self,
        exit_code: int | None,
        stderr: bytes,
        response_path: Path,
        *,
        name: str,
    ) -> dict[str, Any]:
        if exit_code != 0 or not response_path.is_file():
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"{name} worker exited without a valid response.",
                details={
                    "exit_code": exit_code,
                    "stderr": stderr.decode(errors="replace")[-2_000:],
                },
            )
        response_size = response_path.stat().st_size
        response_limit = self.workspace.config.execution.max_output_bytes
        if response_size > response_limit:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"{name} worker response exceeds the configured output budget.",
                details={"byte_length": response_size, "max_bytes": response_limit},
            )
        try:
            payload = json.loads(response_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"{name} worker response is invalid.",
            ) from exc
        if not isinstance(payload, dict):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"{name} worker response is invalid.",
            )
        if payload.get("ok") is not True:
            raw_code = payload.get("code")
            try:
                code = ErrorCode(str(raw_code))
            except ValueError:
                code = ErrorCode.INTERNAL_ERROR
            raise DomainError(code, str(payload.get("message", f"{name} worker failed.")))
        return payload
