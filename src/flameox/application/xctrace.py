from __future__ import annotations

import os
import secrets
import shutil
import stat
import sys
import tarfile
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

from pydantic import Field

from flameox.application.imports import (
    BundleMember,
    ImportBundleRequest,
    ImportService,
)
from flameox.command_binding import ExecutableResolver
from flameox.domain import ArtifactKind, DomainError, ErrorCode, Sensitivity
from flameox.domain.executables import ResolvedExecutable
from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker
from flameox.filesystem import BoundedFileSystem
from flameox.models import ContractModel
from flameox.storage import Workspace

_MAX_TRACE_MEMBERS = 10_000


class XctraceCapability(ContractModel):
    schema_version: Literal[1] = 1
    status: Literal["available", "missing", "unsupported", "permission_denied", "unknown"]
    executable: str | None = None
    version: str | None = None
    qualified_version: bool = False
    metal_system_trace_template: bool = False
    platform_supported: bool = False
    limitations: tuple[str, ...] = ()


class XctraceImportRequest(ContractModel):
    trace_path: Path
    allow_external_path: bool = False
    max_export_bytes: Annotated[int, Field(gt=0, le=16 * 1024 * 1024)] = 4 * 1024 * 1024


class XctraceImportResult(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    trace_artifact_id: str
    toc_artifact_id: str
    corpus_commit_id: str
    xctrace_version: str
    template: Literal["Metal System Trace"] = "Metal System Trace"
    native_member_count: int
    native_byte_length: int
    limitations: tuple[str, ...]
    run_resource_uri: str
    trace_resource_uri: str


class _Broker(Protocol):
    async def run(self, request: ExecutionRequest) -> ExecutionOutcome: ...


class XctraceService:
    """Preserve and inspect native Metal System Trace bundles through Apple's CLI."""

    def __init__(self, workspace: Workspace, *, broker: _Broker | None = None) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()

    async def capability(self) -> XctraceCapability:
        if sys.platform != "darwin":
            return XctraceCapability(
                status="unsupported",
                limitations=("xctrace is supported only on macOS.",),
            )
        executable = ExecutableResolver().resolve_host_tool(
            "xcrun", cwd=self.workspace.project_root
        )
        if executable is None:
            return XctraceCapability(
                status="missing",
                platform_supported=True,
                limitations=("xcrun was not found.",),
            )
        version = await self._run(executable, ("xctrace", "version"), max_output_bytes=64 * 1024)
        if version.process.exit_code != 0:
            detail = version.stderr.decode(errors="replace").lower()
            status = "permission_denied" if "permission" in detail else "unsupported"
            return XctraceCapability(
                status=cast(
                    Literal["permission_denied", "unsupported"],
                    status,
                ),
                executable=str(executable.invocation_path),
                platform_supported=True,
                limitations=("xcrun xctrace version failed.",),
            )
        version_text = version.stdout.decode(errors="replace").strip()[:200]
        qualified_version = version_text.startswith("xctrace version 16.0 ")
        templates = await self._run(
            executable,
            ("xctrace", "list", "templates"),
            max_output_bytes=512 * 1024,
        )
        template_available = (
            templates.process.exit_code == 0
            and "Metal System Trace" in templates.stdout.decode(errors="replace")
        )
        return XctraceCapability(
            status="available" if template_available and qualified_version else "unsupported",
            executable=str(executable.invocation_path),
            version=version_text or None,
            qualified_version=qualified_version,
            metal_system_trace_template=template_available,
            platform_supported=True,
            limitations=()
            if template_available and qualified_version
            else (
                "The installed xctrace version is not qualified for Metal trace import."
                if not qualified_version
                else "The installed xctrace does not list the Metal System Trace template."
            ),
        )

    async def import_trace(self, request: XctraceImportRequest) -> XctraceImportResult:
        capability = await self.capability()
        if capability.status != "available" or capability.version is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                capability.limitations[0] if capability.limitations else "xctrace is unavailable.",
            )
        trace = Path(os.path.abspath(request.trace_path))
        if trace.suffix != ".trace" or not trace.is_dir():
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "xctrace import requires an existing .trace directory bundle.",
            )
        if not request.allow_external_path:
            try:
                trace.relative_to(Path(os.path.abspath(self.workspace.project_root)))
            except ValueError as error:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "The trace bundle is outside the declared project root.",
                ) from error
        staging = self.workspace.paths.staging / f"xctrace-import-{secrets.token_hex(16)}"
        staging.mkdir(parents=True, mode=0o700)
        archive = staging / f"{trace.name}.tar"
        toc = staging / "toc.xml"
        try:
            member_count, byte_length = _archive_trace_bundle(
                trace,
                archive,
                max_bytes=self.workspace.config.capture.max_artifact_bytes,
            )
            executable = ExecutableResolver().require_host_tool(
                "xcrun", cwd=self.workspace.project_root
            )
            exported = await self._run(
                executable,
                (
                    "xctrace",
                    "export",
                    "--input",
                    str(trace),
                    "--toc",
                    "--output",
                    str(toc),
                ),
                max_output_bytes=request.max_export_bytes,
                allowed_roots=(self.workspace.project_root, trace.parent, staging),
            )
            if exported.process.exit_code != 0 or not toc.is_file():
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "xctrace could not export a table of contents for the trace bundle.",
                    details={"provider": "xctrace", "outcome": "export_failed"},
                )
            if toc.stat().st_size > request.max_export_bytes:
                raise DomainError(
                    ErrorCode.QUERY_BUDGET_EXCEEDED,
                    "xctrace table-of-contents export exceeded its byte limit.",
                )
            imported = ImportService(self.workspace).import_provider_bundle(
                ImportBundleRequest(
                    primary=BundleMember(
                        path=archive,
                        role="native_trace_bundle",
                        media_type="application/x-tar",
                    ),
                    sidecars=(
                        BundleMember(
                            path=toc,
                            role="xctrace_toc",
                            media_type="application/xml",
                        ),
                    ),
                    kind=ArtifactKind.METAL_TRACE,
                    sensitivity=Sensitivity.SENSITIVE,
                    producer="xctrace",
                    producer_version=capability.version,
                )
            )
            return XctraceImportResult(
                run_id=imported.run.run_id,
                trace_artifact_id=imported.primary_artifact_id,
                toc_artifact_id=imported.sidecar_artifact_ids[0],
                corpus_commit_id=imported.corpus_commit_id,
                xctrace_version=capability.version,
                native_member_count=member_count,
                native_byte_length=byte_length,
                limitations=(
                    "The native bundle and TOC are preserved; no schema-qualified curated "
                    "Metal event table was available in this import profile.",
                ),
                run_resource_uri=f"flameox://runs/{imported.run.run_id}",
                trace_resource_uri=f"flameox://artifacts/{imported.primary_artifact_id}",
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    async def _run(
        self,
        executable: ResolvedExecutable,
        arguments: tuple[str, ...],
        *,
        max_output_bytes: int,
        allowed_roots: tuple[Path, ...] | None = None,
    ) -> ExecutionOutcome:
        return await self.broker.run(
            ExecutionRequest(
                argv=(str(executable.invocation_path), *arguments),
                cwd=self.workspace.project_root,
                environment_allowlist=("PATH", "DEVELOPER_DIR"),
                allowed_working_roots=allowed_roots or (self.workspace.project_root,),
                timeout_seconds=60,
                max_output_bytes=max_output_bytes,
                executable_binding=executable,
            )
        )


def _archive_trace_bundle(trace: Path, destination: Path, *, max_bytes: int) -> tuple[int, int]:
    members = 0
    total = 0
    with tarfile.open(destination, mode="x") as archive:
        for root, directories, files in os.walk(trace, followlinks=False):
            directories.sort()
            files.sort()
            root_path = Path(root)
            for name in (*directories, *files):
                path = root_path / name
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or (
                    stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1
                ):
                    raise DomainError(
                        ErrorCode.EXECUTION_REFUSED,
                        "Native trace bundles cannot contain links.",
                    )
            for name in files:
                path = root_path / name
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise DomainError(
                        ErrorCode.EXECUTION_REFUSED,
                        "Native trace bundles may contain only directories and regular files.",
                    )
                members += 1
                total += metadata.st_size
                if members > _MAX_TRACE_MEMBERS or total > max_bytes:
                    raise DomainError(
                        ErrorCode.ARTIFACT_TOO_LARGE,
                        "Native trace bundle exceeds its member or byte budget.",
                    )
                relative = path.relative_to(trace).as_posix()
                info = tarfile.TarInfo(relative)
                info.size = metadata.st_size
                info.mode = metadata.st_mode & 0o777
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                with (
                    BoundedFileSystem((trace,)).open_regular(
                        path,
                        max_bytes=max_bytes - (total - metadata.st_size),
                        require_single_link=True,
                    ) as descriptor,
                    os.fdopen(os.dup(descriptor), "rb") as stream,
                ):
                    archive.addfile(info, stream)
    return members, total
