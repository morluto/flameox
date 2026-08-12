from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from flameox.application.artifacts import ArtifactService
from flameox.command_binding import ExecutableResolver
from flameox.domain import ArtifactKind, DomainError, ErrorCode, ProcessResult
from flameox.domain.executables import ResolvedExecutable
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, Workspace


class _NativeViewerPlan(ContractModel):
    schema_version: int = 1
    artifact_id: str
    artifact_path: str
    artifact_kinds: tuple[ArtifactKind, ...]
    viewer: str
    argv: tuple[str, ...]
    executable_binding: ResolvedExecutable
    limitations: tuple[str, ...] = (
        "Viewer behavior and format support are controlled by the installed application.",
    )


class NativeViewerPlan(_NativeViewerPlan):
    launches: Literal[False] = False


class _LaunchedNativeViewerPlan(_NativeViewerPlan):
    launches: Literal[True] = True


class NativeViewerLaunchResult(ContractModel):
    schema_version: int = 1
    plan: _LaunchedNativeViewerPlan
    process: ProcessResult


class NativeViewerService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def plan(self, artifact_id: str) -> NativeViewerPlan:
        artifact = ArtifactStore(self.workspace).get(artifact_id)
        metadata = ArtifactService(self.workspace).get(artifact_id)
        kinds = tuple(
            sorted(
                {item.kind for item in metadata.registrations},
                key=lambda kind: kind.value,
            )
        )
        producers = {
            item.producer.lower() for item in metadata.registrations if item.producer is not None
        }
        viewer, argv, executable_binding = self._viewer_for(
            artifact.payload_path,
            kinds=kinds,
            producers=producers,
        )
        return NativeViewerPlan(
            artifact_id=artifact_id,
            artifact_path=str(artifact.payload_path),
            artifact_kinds=kinds,
            viewer=viewer,
            argv=argv,
            executable_binding=executable_binding,
        )

    async def launch(self, artifact_id: str) -> NativeViewerLaunchResult:
        plan = self.plan(artifact_id)
        outcome = await SubprocessBroker().run(
            ExecutionRequest(
                argv=plan.argv,
                executable_binding=plan.executable_binding,
                cwd=self.workspace.project_root,
                allowed_working_roots=(self.workspace.project_root,),
                timeout_seconds=30,
                max_output_bytes=1_048_576,
            )
        )
        return NativeViewerLaunchResult(
            plan=_LaunchedNativeViewerPlan.model_validate(
                {**plan.model_dump(mode="python"), "launches": True}
            ),
            process=outcome.process,
        )

    def _viewer_for(
        self,
        path: Path,
        *,
        kinds: tuple[ArtifactKind, ...],
        producers: set[str],
    ) -> tuple[str, tuple[str, ...], ResolvedExecutable]:
        kind_set = set(kinds)
        if ArtifactKind.MEMORY_PROFILE in kind_set or "memray" in producers:
            executable = self._required_executable(
                "memray",
                "Install Memray to inspect memory-profile artifacts.",
            )
            return "memray tree", (str(executable.invocation_path), "tree", str(path)), executable
        if ArtifactKind.BENCHMARK_SAMPLES in kind_set or "pyperf" in producers:
            executable = self._required_executable(
                "pyperf",
                "Install pyperf to inspect benchmark artifacts.",
            )
            return "pyperf show", (str(executable.invocation_path), "show", str(path)), executable
        if kind_set & {
            ArtifactKind.EXECUTION_TRACE,
            ArtifactKind.SAMPLE_PROFILE,
        }:
            configured = self.workspace.config.analysis.trace_processor_path
            trace_binding = (
                ExecutableResolver().resolve_host_tool(
                    str(
                        (
                            Path(configured)
                            if Path(configured).is_absolute()
                            else self.workspace.project_root / configured
                        ).resolve()
                    ),
                    cwd=self.workspace.project_root,
                )
                if configured is not None
                else self._optional_executable("trace_processor_shell")
                or self._optional_executable("trace_processor")
            )
            if trace_binding is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "No Perfetto Trace Processor is installed for this trace artifact.",
                    remediation=(
                        "Install trace_processor_shell or configure analysis.trace_processor_path.",
                    ),
                )
            return (
                "trace_processor_shell",
                (str(trace_binding.invocation_path), str(path)),
                trace_binding,
            )
        if ArtifactKind.CORE_DUMP in kind_set:
            executable = self._required_executable(
                "gdb",
                "Install gdb and supply symbols when inspecting the core.",
            )
            return "gdb", (str(executable.invocation_path), "-c", str(path)), executable
        return self._opener(path)

    def _required_executable(self, name: str, remediation: str) -> ResolvedExecutable:
        executable = self._optional_executable(name)
        if executable is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"The {name!r} ecosystem viewer is not installed.",
                remediation=(remediation,),
            )
        return executable

    def _opener(self, path: Path) -> tuple[str, tuple[str, ...], ResolvedExecutable]:
        if sys.platform == "darwin":
            executable = self._optional_executable("open")
            name = "open"
        elif os.name == "nt":
            executable = self._optional_executable("explorer")
            name = "explorer"
        else:
            executable = self._optional_executable("xdg-open")
            name = "xdg-open"
        if executable is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "No operating-system file viewer command is installed.",
                remediation=("Install xdg-utils or open the reported artifact path manually.",),
            )
        return name, (str(executable.invocation_path), str(path)), executable

    @staticmethod
    def _optional_executable(name: str) -> ResolvedExecutable | None:
        return ExecutableResolver().resolve_host_tool(name)
