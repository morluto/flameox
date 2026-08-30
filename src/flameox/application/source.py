from __future__ import annotations

import hashlib
import os
import stat
import sys
from asyncio import run as run_async
from contextlib import suppress
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from flameox.command_binding import ExecutableResolver
from flameox.domain import (
    DomainError,
    ErrorCode,
    IdentityQuality,
    SourceState,
    digest_model,
    process_exit_code,
)
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.storage import Workspace


def collect_partial_source_state(
    workspace: Workspace,
    *,
    executable: Path | None = None,
) -> SourceState:
    return _partial_source_state(executable or Path(sys.executable))


def _partial_source_state(executable: Path | None) -> SourceState:
    resolved_executable = executable.resolve() if executable is not None else None
    executable_digest = _file_digest(resolved_executable) if resolved_executable else None
    fields: dict[str, JsonValue] = {}
    if resolved_executable is not None:
        fields["resolved_executable"] = str(resolved_executable)
    content: dict[str, JsonValue] = {
        "identity_quality": "partial",
        "repository_root": ".",
        "fields": fields,
        "missing_fields": ["git_diff", "untracked_inputs", "submodules"],
    }
    if executable_digest is not None:
        content["executable_digest"] = executable_digest
    return SourceState(
        source_state_id=digest_model(content),
        identity_quality=IdentityQuality.PARTIAL,
        repository_root=".",
        executable_digest=executable_digest,
        fields=fields,
        missing_fields=("git_diff", "untracked_inputs", "submodules"),
    )


def collect_import_source_state(workspace: Workspace) -> SourceState:
    """Collect a repository identity for a synchronous evidence import.

    Imports do not have a workload executable.  Reuse the capture source-state
    collector with that fact made explicit instead of assigning the controller
    interpreter to imported evidence.
    """
    return run_async(
        collect_source_state(
            workspace,
            workload_executable=None,
            broker=SubprocessBroker(),
        )
    )


async def collect_source_state(
    workspace: Workspace,
    *,
    workload_executable: str | None,
    broker: SubprocessBroker,
) -> SourceState:
    git_binding = ExecutableResolver().resolve_host_tool("git")
    resolved_executable = (
        _resolve_executable(workload_executable, workspace.project_root)
        if workload_executable is not None
        else None
    )
    executable_digest = (
        _file_digest(resolved_executable) if resolved_executable is not None else None
    )
    if git_binding is None or not (workspace.project_root / ".git").exists():
        return _partial_source_state(resolved_executable)
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
        untracked = _hash_untracked(
            workspace.project_root,
            untracked_output,
            ignored_root=workspace.paths.root,
        )
    except (DomainError, OSError, UnicodeDecodeError):
        return _partial_source_state(resolved_executable)
    diff_digest = f"sha256:{hashlib.sha256(diff).hexdigest()}"
    quality = (
        IdentityQuality.CLEAN
        if not diff and not untracked and not submodules
        else IdentityQuality.EXACT
    )
    fields: dict[str, JsonValue] = {
        "branch": branch,
        "untracked_inputs": cast(JsonValue, untracked),
        "submodules": submodules,
    }
    if resolved_executable is not None:
        fields["resolved_executable"] = str(resolved_executable)
    content: dict[str, JsonValue] = {
        "identity_quality": quality.value,
        "repository_root": ".",
        "head_commit": head,
        "diff_digest": diff_digest,
        "fields": fields,
        "missing_fields": [],
    }
    if executable_digest is not None:
        content["executable_digest"] = executable_digest
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
            executable_binding=ExecutableResolver().require_host_tool(
                "git", cwd=workspace.project_root
            ),
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
    if process_exit_code(outcome.process.termination) != 0:
        raise DomainError(
            code=ErrorCode.PROCESS_FAILED,
            message="Git source-state inspection failed.",
        )
    return outcome.stdout


def _resolve_executable(value: str, cwd: Path) -> Path:
    binding = ExecutableResolver().resolve_host_tool(value, cwd=cwd)
    if binding is None:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "The workload executable could not be resolved.",
            details={"executable": value},
            remediation=(
                "Install the workload executable or declare a resolvable path, then retry capture.",
            ),
        )
    return binding.canonical_target


def _hash_untracked(
    project_root: Path,
    output: bytes,
    *,
    ignored_root: Path | None = None,
) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    ignored_relative: Path | None = None
    if ignored_root is not None:
        with suppress(ValueError):
            ignored_relative = ignored_root.resolve().relative_to(project_root.resolve())
    for raw in sorted(item for item in output.split(b"\0") if item):
        relative = raw.decode("utf-8", errors="strict")
        relative_path = Path(relative)
        if ignored_relative is not None and (
            relative_path == ignored_relative or ignored_relative in relative_path.parents
        ):
            continue
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
