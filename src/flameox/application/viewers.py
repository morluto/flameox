from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from packaging.version import InvalidVersion, Version

from flameox.action_graph import ActionId, tool_action
from flameox.adapters.builtins import builtin_adapter
from flameox.application.artifacts import ArtifactService
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.command_binding import ExecutableResolver
from flameox.domain import ArtifactKind, DomainError, ErrorCode, ProcessResult
from flameox.domain.executables import ResolvedExecutable
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, Workspace

_DEFAULT_LIMITATIONS = (
    "Viewer behavior and format support are controlled by the installed application.",
)


class _NativeViewerPlan(ContractModel):
    artifact_id: str
    artifact_path: str
    artifact_kinds: tuple[ArtifactKind, ...]
    viewer: str
    argv: tuple[str, ...]
    executable_binding: ResolvedExecutable
    provider_environment_id: str | None = None
    limitations: tuple[str, ...] = _DEFAULT_LIMITATIONS


class NativeViewerPlan(_NativeViewerPlan):
    launches: Literal[False] = False


class _LaunchedNativeViewerPlan(_NativeViewerPlan):
    launches: Literal[True] = True


class NativeViewerLaunchResult(ContractModel):
    plan: _LaunchedNativeViewerPlan
    process: ProcessResult


class NativeViewerService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.provider_runtimes = ProviderRuntimeManager(
            workspace.paths.records / "provider-runtimes"
        )

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
        raw_producer_versions = {
            item.producer_version
            for item in metadata.registrations
            if item.producer is not None
            and item.producer.lower() == "memray"
            and item.producer_version is not None
        }
        memray_version = self._memray_version(raw_producer_versions)
        viewer, argv, executable_binding, provider_environment_id, limitations = self._viewer_for(
            artifact.payload_path,
            kinds=kinds,
            producers=producers,
            memray_version=memray_version,
        )
        return NativeViewerPlan(
            artifact_id=artifact_id,
            artifact_path=str(artifact.payload_path),
            artifact_kinds=kinds,
            viewer=viewer,
            argv=argv,
            executable_binding=executable_binding,
            provider_environment_id=provider_environment_id,
            limitations=limitations,
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
        memray_version: str | None,
    ) -> tuple[str, tuple[str, ...], ResolvedExecutable, str | None, tuple[str, ...]]:
        kind_set = set(kinds)
        if ArtifactKind.MEMORY_PROFILE in kind_set or "memray" in producers:
            executable, environment_id, limitations = self._provider_executable(
                "memray", version=memray_version
            )
            return (
                "memray tree",
                (str(executable.invocation_path), "tree", str(path)),
                executable,
                environment_id,
                limitations,
            )
        if ArtifactKind.BENCHMARK_SAMPLES in kind_set or "pyperf" in producers:
            executable = self._required_executable(
                sys.executable,
                "Run Flameox from an environment containing its required pyperf dependency.",
            )
            return (
                "pyperf show",
                (str(executable.invocation_path), "-m", "pyperf", "show", str(path)),
                executable,
                None,
                _DEFAULT_LIMITATIONS,
            )
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
                None,
                _DEFAULT_LIMITATIONS,
            )
        if ArtifactKind.CORE_DUMP in kind_set:
            executable = self._required_executable(
                "gdb",
                "Install gdb and supply symbols when inspecting the core.",
            )
            return (
                "gdb",
                (str(executable.invocation_path), "-c", str(path)),
                executable,
                None,
                _DEFAULT_LIMITATIONS,
            )
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

    def _provider_executable(
        self,
        name: str,
        *,
        version: str | None,
    ) -> tuple[ResolvedExecutable, str | None, tuple[str, ...]]:
        definition = builtin_adapter(name)
        if (
            definition is not None
            and definition.managed_extra is not None
            and definition.managed_requirement is not None
        ):
            requirement = (
                f"{name}=={version}" if version is not None else definition.managed_requirement
            )
            runtime = self.provider_runtimes.find_distribution(
                extra=definition.managed_extra,
                requirement=requirement,
            )
            if runtime is not None and runtime.executable is not None:
                binding = ExecutableResolver().require_host_tool(
                    str(runtime.executable), cwd=runtime.root
                )
                return (
                    binding,
                    runtime.receipt.environment_id,
                    (
                        _DEFAULT_LIMITATIONS
                        if version is not None
                        else (
                            *_DEFAULT_LIMITATIONS,
                            "The artifact has no Memray producer version; the viewer satisfies "
                            "Flameox's supported range but exact producer compatibility is "
                            "unknown.",
                        )
                    ),
                )
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"No verified managed {name!r} viewer is available.",
            details={"producer_version": version},
            remediation=(f"Call start_capability_setup with adapters=[{name!r}], then retry.",),
            next_action=tool_action(
                ActionId.START_CAPABILITY_SETUP,
                adapters=[name],
                idempotency_key=f"native-viewer-{name}-{version or 'compatible'}",
                **({"memray_reader_version": version} if version is not None else {}),
            ),
        )

    @staticmethod
    def _memray_version(versions: set[str]) -> str | None:
        if len(versions) > 1:
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The artifact has conflicting Memray producer versions.",
                details={"producer_versions": sorted(versions)},
                remediation=("Select a registration with unambiguous producer provenance.",),
            )
        if not versions:
            return None
        raw = next(iter(versions))
        try:
            return str(Version(raw))
        except InvalidVersion as error:
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The artifact has an invalid Memray producer version.",
                details={"producer_version": raw},
            ) from error

    def _opener(
        self,
        path: Path,
    ) -> tuple[str, tuple[str, ...], ResolvedExecutable, str | None, tuple[str, ...]]:
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
        return (
            name,
            (str(executable.invocation_path), str(path)),
            executable,
            None,
            _DEFAULT_LIMITATIONS,
        )

    @staticmethod
    def _optional_executable(name: str) -> ResolvedExecutable | None:
        return ExecutableResolver().resolve_host_tool(name)
