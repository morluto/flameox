from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from flameox.application.artifacts import ArtifactService
from flameox.command_binding import ExecutableResolver
from flameox.domain import ArtifactKind, DomainError, ErrorCode, ProcessResult
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
        viewer, argv = self._viewer_for(
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
        )

    async def launch(self, artifact_id: str) -> NativeViewerLaunchResult:
        plan = self.plan(artifact_id)
        outcome = await SubprocessBroker().run(
            ExecutionRequest(
                argv=plan.argv,
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
    ) -> tuple[str, tuple[str, ...]]:
        kind_set = set(kinds)
        if ArtifactKind.MEMORY_PROFILE in kind_set or "memray" in producers:
            executable = self._required_executable(
                "memray",
                "Install Memray to inspect memory-profile artifacts.",
            )
            return "memray tree", (executable, "tree", str(path))
        if ArtifactKind.BENCHMARK_SAMPLES in kind_set or "pyperf" in producers:
            executable = self._required_executable(
                "pyperf",
                "Install pyperf to inspect benchmark artifacts.",
            )
            return "pyperf show", (executable, "show", str(path))
        if kind_set & {
            ArtifactKind.EXECUTION_TRACE,
            ArtifactKind.SAMPLE_PROFILE,
        }:
            configured = self.workspace.config.analysis.trace_processor_path
            trace_executable = (
                str(
                    (
                        Path(configured)
                        if Path(configured).is_absolute()
                        else self.workspace.project_root / configured
                    ).resolve()
                )
                if configured is not None
                else self._optional_executable("trace_processor_shell")
                or self._optional_executable("trace_processor")
            )
            if (
                trace_executable is None
                or not Path(trace_executable).is_file()
                or not os.access(trace_executable, os.X_OK)
            ):
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "No Perfetto Trace Processor is installed for this trace artifact.",
                    remediation=(
                        "Install trace_processor_shell or configure analysis.trace_processor_path.",
                    ),
                )
            return "trace_processor_shell", (trace_executable, str(path))
        if ArtifactKind.CORE_DUMP in kind_set:
            executable = self._required_executable(
                "gdb",
                "Install gdb and supply symbols when inspecting the core.",
            )
            return "gdb", (executable, "-c", str(path))
        return self._opener(path)

    def _required_executable(self, name: str, remediation: str) -> str:
        executable = self._optional_executable(name)
        if executable is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"The {name!r} ecosystem viewer is not installed.",
                remediation=(remediation,),
            )
        return executable

    def _opener(self, path: Path) -> tuple[str, tuple[str, ...]]:
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
        return name, (executable, str(path))

    @staticmethod
    def _optional_executable(name: str) -> str | None:
        binding = ExecutableResolver().resolve_host_tool(name)
        return str(binding.invocation_path) if binding is not None else None
