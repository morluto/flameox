from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from flameox.domain import (
    DomainError,
    ErrorCode,
    IdentityQuality,
    SourceState,
    digest_model,
)
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.storage import Workspace


def collect_partial_source_state(
    workspace: Workspace,
    *,
    executable: Path | None = None,
) -> SourceState:
    del workspace
    executable = (executable or Path(sys.executable)).resolve()
    executable_digest = _file_digest(executable)
    fields: dict[str, JsonValue] = {
        "resolved_executable": str(executable),
    }
    content: dict[str, JsonValue] = {
        "identity_quality": "partial",
        "repository_root": ".",
        "executable_digest": executable_digest,
        "fields": fields,
        "missing_fields": ["git_diff", "untracked_inputs", "submodules"],
    }
    return SourceState(
        source_state_id=digest_model(content),
        identity_quality=IdentityQuality.PARTIAL,
        repository_root=".",
        executable_digest=executable_digest,
        fields=fields,
        missing_fields=("git_diff", "untracked_inputs", "submodules"),
    )


async def collect_source_state(
    workspace: Workspace,
    *,
    workload_executable: str,
    broker: SubprocessBroker,
) -> SourceState:
    git = shutil.which("git")
    resolved_executable = _resolve_executable(
        workload_executable,
        workspace.project_root,
    )
    executable_digest = _file_digest(resolved_executable)
    if git is None or not (workspace.project_root / ".git").exists():
        return collect_partial_source_state(
            workspace,
            executable=resolved_executable,
        )
    try:
        head = await _git(workspace, broker, "rev-parse", "HEAD")
        branch = await _git(
            workspace,
            broker,
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        )
        diff = await _git_bytes(
            workspace,
            broker,
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            max_output_bytes=64 * 1024 * 1024,
        )
        untracked_output = await _git_bytes(
            workspace,
            broker,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        submodules = await _git(
            workspace,
            broker,
            "submodule",
            "status",
            "--recursive",
        )
        untracked = _hash_untracked(workspace.project_root, untracked_output)
    except (DomainError, OSError, UnicodeDecodeError):
        return collect_partial_source_state(
            workspace,
            executable=resolved_executable,
        )
    diff_digest = f"sha256:{hashlib.sha256(diff).hexdigest()}"
    quality = (
        IdentityQuality.CLEAN
        if not diff and not untracked and not submodules
        else IdentityQuality.EXACT
    )
    fields: dict[str, JsonValue] = {
        "branch": branch,
        "resolved_executable": str(resolved_executable),
        "untracked_inputs": cast(JsonValue, untracked),
        "submodules": submodules,
    }
    content: dict[str, JsonValue] = {
        "identity_quality": quality.value,
        "repository_root": ".",
        "head_commit": head,
        "diff_digest": diff_digest,
        "executable_digest": executable_digest,
        "fields": fields,
        "missing_fields": [],
    }
    return SourceState(
        source_state_id=digest_model(content),
        identity_quality=quality,
        repository_root=".",
        head_commit=head,
        diff_digest=diff_digest,
        executable_digest=executable_digest,
        fields=fields,
    )


async def _git(
    workspace: Workspace,
    broker: SubprocessBroker,
    *arguments: str,
) -> str:
    return (
        (await _git_bytes(workspace, broker, *arguments)).decode("utf-8", errors="strict").strip()
    )


async def _git_bytes(
    workspace: Workspace,
    broker: SubprocessBroker,
    *arguments: str,
    max_output_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    outcome = await broker.run(
        ExecutionRequest(
            argv=("git", "-c", "core.quotepath=false", *arguments),
            cwd=workspace.project_root,
            environment_allowlist=("PATH",),
            environment_overrides={
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_OPTIONAL_LOCKS": "0",
            },
            allowed_working_roots=(workspace.project_root,),
            timeout_seconds=30,
            max_output_bytes=max_output_bytes,
        )
    )
    if outcome.process.exit_code != 0:
        raise DomainError(
            code=ErrorCode.PROCESS_FAILED,
            message="Git source-state inspection failed.",
        )
    return outcome.stdout


def _resolve_executable(value: str, cwd: Path) -> Path:
    if os.sep in value:
        candidate = Path(value)
        resolved = (candidate if candidate.is_absolute() else cwd / candidate).resolve()
    else:
        located = shutil.which(value)
        if located is None:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The workload executable could not be resolved.",
                details={"executable": value},
                remediation=(
                    "Install the workload executable or declare a resolvable path, then retry "
                    "capture.",
                ),
            )
        resolved = Path(located).resolve()
    if not resolved.is_file():
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "The workload executable could not be resolved.",
            details={"executable": value},
            remediation=(
                "Install the workload executable or declare a resolvable path, then retry capture.",
            ),
        )
    return resolved


def _hash_untracked(project_root: Path, output: bytes) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    for raw in sorted(item for item in output.split(b"\0") if item):
        relative = raw.decode("utf-8", errors="strict")
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise OSError("untracked path escaped repository") from exc
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                continue
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        records.append(
            {
                "path": relative,
                "sha256": digest.hexdigest(),
                "byte_length": metadata.st_size,
            }
        )
    return records


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
