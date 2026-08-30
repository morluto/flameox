from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath

from flameox.application.workloads import WorkloadService
from flameox.domain import (
    DomainError,
    ErrorCode,
    ExecutionIdentityBasis,
    ExecutionIdentityInput,
    ExecutionIdentityInputKind,
    ExecutionIdentityInputStatus,
    ExecutionIdentityQuality,
    WorkloadExecutionIdentity,
    digest_model,
    process_exit_code,
)
from flameox.domain.executables import ResolvedExecutable
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.storage import Workspace

_MODULE_METADATA_PROBE = """
import hashlib
import importlib.metadata as metadata
import json
import sys

MAX_DISTRIBUTION_FILES = 16_384


def failed(name, status, failure_code):
    return {
        "name": name,
        "basis": None,
        "distribution": None,
        "version": None,
        "content_digest": None,
        "status": status,
        "failure_code": failure_code,
    }


try:
    package_distributions = metadata.packages_distributions()
except BaseException:
    package_distributions = None

values = []
for name in sys.argv[1:]:
    top_level = name.split(".", 1)[0]
    if top_level in sys.builtin_module_names or top_level in sys.stdlib_module_names:
        values.append({
            "name": name,
            "basis": "interpreter_stdlib",
            "distribution": None,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "content_digest": None,
            "status": "exact",
            "failure_code": None,
        })
        continue
    if package_distributions is None:
        values.append(failed(name, "resolution_failed", "metadata_catalog_unavailable"))
        continue
    distributions = package_distributions.get(top_level, [])
    if not distributions:
        values.append(failed(name, "not_observed", "distribution_mapping_missing"))
        continue
    if len(distributions) != 1:
        values.append(failed(name, "ambiguous", "distribution_mapping_ambiguous"))
        continue
    distribution_name = distributions[0]
    try:
        distribution = metadata.distribution(distribution_name)
        files = distribution.files
        if files is None:
            values.append(failed(name, "ambiguous", "distribution_files_unavailable"))
            continue
        if len(files) > MAX_DISTRIBUTION_FILES:
            values.append(failed(name, "ambiguous", "distribution_file_limit_exceeded"))
            continue
        inventory = []
        incomplete = False
        for entry in files:
            entry_path = str(entry).replace("\\\\", "/")
            entry_hash = entry.hash
            if entry_hash is None:
                if entry_path.endswith(".dist-info/RECORD"):
                    continue
                incomplete = True
                break
            inventory.append([
                entry_path,
                f"{entry_hash.mode}:{entry_hash.value}",
                entry.size,
            ])
        if incomplete:
            values.append(failed(name, "ambiguous", "distribution_record_incomplete"))
            continue
        canonical_name = distribution.metadata.get("Name") or distribution_name
        distribution_version = distribution.version
        identity_payload = json.dumps(
            {
                "distribution": canonical_name,
                "version": distribution_version,
                "files": sorted(inventory),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        values.append({
            "name": name,
            "basis": "distribution_metadata",
            "distribution": canonical_name,
            "version": distribution_version,
            "content_digest": "sha256:" + hashlib.sha256(identity_payload).hexdigest(),
            "status": "exact",
            "failure_code": None,
        })
    except metadata.PackageNotFoundError:
        values.append(failed(name, "resolution_failed", "distribution_not_found"))
    except BaseException:
        values.append(failed(name, "resolution_failed", "distribution_metadata_unavailable"))
print(json.dumps(values, separators=(",", ":")))
"""

_METADATA_FAILURE_LIMITATIONS = {
    "metadata_catalog_unavailable": "Python distribution metadata could not be enumerated.",
    "distribution_mapping_missing": (
        "No isolated package-to-distribution mapping was available; runtime import use was not "
        "observed."
    ),
    "distribution_mapping_ambiguous": (
        "The import package maps to multiple distributions, so its identity is ambiguous."
    ),
    "distribution_files_unavailable": (
        "The installed distribution does not publish a file inventory."
    ),
    "distribution_file_limit_exceeded": (
        "The installed distribution exceeds the bounded metadata inventory limit."
    ),
    "distribution_record_incomplete": (
        "The installed distribution file inventory contains unhashed entries."
    ),
    "distribution_not_found": "The mapped Python distribution was not installed.",
    "distribution_metadata_unavailable": "Python distribution metadata could not be read.",
}


class ExecutionIdentityService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.broker = broker or SubprocessBroker()

    def plan(
        self,
        workload_name: str,
        *,
        cwd: Path | None = None,
    ) -> WorkloadExecutionIdentity:
        config = self.workloads.load().workloads[workload_name]
        inputs: list[ExecutionIdentityInput] = []
        for name in config.identity.python_modules:
            project_input = self._project_module(name, cwd) if cwd is not None else None
            inputs.append(
                project_input
                or ExecutionIdentityInput(
                    kind=ExecutionIdentityInputKind.PYTHON_MODULE,
                    requested=name,
                    status=ExecutionIdentityInputStatus.NOT_OBSERVED,
                    limitations=(
                        "Passive distribution metadata is observed immediately before execution; "
                        "the target module is never imported by identity collection.",
                    ),
                )
            )
        inputs.extend(self._native(value) for value in config.identity.native_files)
        return self._report(inputs)

    async def observe(
        self,
        workload_name: str,
        *,
        parameters: dict[str, str | int | float | bool],
        dynamic_parameters: tuple[str, ...] = (),
    ) -> WorkloadExecutionIdentity:
        config = self.workloads.load().workloads[workload_name]
        instance = self.workloads.resolve(
            workload_name,
            parameters,
            dynamic_parameters=dynamic_parameters,
        )
        inputs: list[ExecutionIdentityInput] = []
        if config.identity.python_modules:
            inputs.extend(
                await self._modules(
                    config.identity.python_modules,
                    executable=instance.command.argv[0],
                    executable_binding=instance.executable_binding,
                    cwd=Path(instance.command.cwd),
                    environment=instance.command.env_overrides,
                )
            )
        inputs.extend(self._native(value) for value in config.identity.native_files)
        return self._report(inputs)

    async def distribution_identity(
        self,
        workload_name: str,
        distribution: str,
    ) -> ExecutionIdentityInput:
        """Bind one installed distribution through the declared workload interpreter.

        This deliberately bypasses project-module lookup: compiler identity is
        the installed distribution selected by the interpreter, not a local
        file which happens to shadow its import name.
        """

        if not distribution.isidentifier():
            raise ValueError("distribution probe names must be Python identifiers")
        instance = self.workloads.resolve(workload_name)
        return (
            await self._metadata_modules(
                (distribution,),
                executable=instance.command.argv[0],
                executable_binding=instance.executable_binding,
                cwd=Path(instance.command.cwd),
                environment=instance.command.env_overrides,
            )
        )[0]

    async def _modules(
        self,
        names: tuple[str, ...],
        *,
        executable: str,
        executable_binding: ResolvedExecutable,
        cwd: Path,
        environment: dict[str, str],
    ) -> tuple[ExecutionIdentityInput, ...]:
        results: list[ExecutionIdentityInput | None] = []
        unresolved: list[str] = []
        for name in names:
            project_input = self._project_module(name, cwd)
            results.append(project_input)
            if project_input is None:
                unresolved.append(name)
        if not unresolved:
            return tuple(item for item in results if item is not None)

        metadata_results = await self._metadata_modules(
            tuple(unresolved),
            executable=executable,
            executable_binding=executable_binding,
            cwd=cwd,
            environment=environment,
        )
        metadata = iter(metadata_results)
        return tuple(item if item is not None else next(metadata) for item in results)

    async def _metadata_modules(
        self,
        names: tuple[str, ...],
        *,
        executable: str,
        executable_binding: ResolvedExecutable,
        cwd: Path,
        environment: dict[str, str],
    ) -> tuple[ExecutionIdentityInput, ...]:
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(executable, "-I", "-c", _MODULE_METADATA_PROBE, *names),
                    executable_binding=executable_binding,
                    cwd=cwd,
                    environment_allowlist=(
                        self.workspace.config.execution.child_environment_allowlist
                    ),
                    environment_overrides=environment,
                    allowed_working_roots=(self.workspace.project_root,),
                    timeout_seconds=10,
                    max_output_bytes=65_536,
                )
            )
            if process_exit_code(outcome.process.termination) != 0:
                raise ValueError("metadata probe exited unsuccessfully")
            values = json.loads(outcome.stdout)
            if (
                not isinstance(values, list)
                or len(values) != len(names)
                or any(not isinstance(value, dict) for value in values)
            ):
                raise ValueError("module identity probe returned an invalid result set")
        except (DomainError, json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            return tuple(
                ExecutionIdentityInput(
                    kind=ExecutionIdentityInputKind.PYTHON_MODULE,
                    requested=name,
                    status=ExecutionIdentityInputStatus.RESOLUTION_FAILED,
                    limitations=("Passive Python metadata probing failed.",),
                )
                for name in names
            )
        results: list[ExecutionIdentityInput] = []
        for name, value in zip(names, values, strict=True):
            if value.get("name") != name:
                results.append(self._metadata_failure(name, "metadata_catalog_unavailable"))
                continue
            basis_value = value.get("basis")
            distribution = value.get("distribution")
            distribution_version = value.get("version")
            content_digest = value.get("content_digest")
            if value.get("status") == "exact" and basis_value == "interpreter_stdlib":
                results.append(
                    ExecutionIdentityInput(
                        kind=ExecutionIdentityInputKind.PYTHON_MODULE,
                        requested=name,
                        identity_basis=ExecutionIdentityBasis.INTERPRETER_STDLIB,
                        version=(
                            distribution_version if isinstance(distribution_version, str) else None
                        ),
                        content_digest=executable_binding.identity.sha256,
                        status=ExecutionIdentityInputStatus.EXACT,
                        limitations=(
                            "Standard-library membership is bound to the exact interpreter; "
                            "runtime import use is not observed.",
                        ),
                    )
                )
                continue
            if (
                value.get("status") == "exact"
                and basis_value == "distribution_metadata"
                and isinstance(distribution, str)
                and isinstance(distribution_version, str)
                and isinstance(content_digest, str)
            ):
                try:
                    results.append(
                        ExecutionIdentityInput(
                            kind=ExecutionIdentityInputKind.PYTHON_MODULE,
                            requested=name,
                            identity_basis=ExecutionIdentityBasis.DISTRIBUTION_METADATA,
                            distribution=distribution,
                            version=distribution_version,
                            content_digest=content_digest,
                            status=ExecutionIdentityInputStatus.EXACT,
                            limitations=(
                                "Installed distribution metadata is bound; runtime import use is "
                                "not observed.",
                            ),
                        )
                    )
                    continue
                except ValueError:
                    pass
            failure_code = value.get("failure_code")
            results.append(
                self._metadata_failure(
                    name,
                    (
                        failure_code
                        if isinstance(failure_code, str)
                        else "metadata_catalog_unavailable"
                    ),
                    status=value.get("status"),
                )
            )
        return tuple(results)

    def _metadata_failure(
        self,
        name: str,
        failure_code: str,
        *,
        status: object = None,
    ) -> ExecutionIdentityInput:
        statuses = {
            "ambiguous": ExecutionIdentityInputStatus.AMBIGUOUS,
            "not_observed": ExecutionIdentityInputStatus.NOT_OBSERVED,
            "resolution_failed": ExecutionIdentityInputStatus.RESOLUTION_FAILED,
        }
        return ExecutionIdentityInput(
            kind=ExecutionIdentityInputKind.PYTHON_MODULE,
            requested=name,
            status=(
                statuses.get(status, ExecutionIdentityInputStatus.RESOLUTION_FAILED)
                if isinstance(status, str)
                else ExecutionIdentityInputStatus.RESOLUTION_FAILED
            ),
            limitations=(
                _METADATA_FAILURE_LIMITATIONS.get(
                    failure_code,
                    "Python distribution metadata could not be validated.",
                ),
            ),
        )

    def _project_module(self, name: str, cwd: Path) -> ExecutionIdentityInput | None:
        parts = name.split(".")
        if not parts or any(not part.isidentifier() for part in parts):
            return ExecutionIdentityInput(
                kind=ExecutionIdentityInputKind.PYTHON_MODULE,
                requested=name,
                status=ExecutionIdentityInputStatus.RESOLUTION_FAILED,
                limitations=("Declared Python module names must contain identifiers only.",),
            )
        try:
            project_root = self.workspace.project_root.resolve(strict=True)
            resolved_cwd = cwd.resolve(strict=True)
            resolved_cwd.relative_to(project_root)
        except (FileNotFoundError, OSError, ValueError):
            return ExecutionIdentityInput(
                kind=ExecutionIdentityInputKind.PYTHON_MODULE,
                requested=name,
                status=ExecutionIdentityInputStatus.RESOLUTION_FAILED,
                limitations=("The workload directory is not a readable project directory.",),
            )
        module_path = resolved_cwd.joinpath(*parts).with_suffix(".py")
        package_path = resolved_cwd.joinpath(*parts, "__init__.py")
        configured = tuple(path for path in (module_path, package_path) if path.is_file())
        if not configured:
            return None
        if len(configured) != 1:
            return ExecutionIdentityInput(
                kind=ExecutionIdentityInputKind.PYTHON_MODULE,
                requested=name,
                status=ExecutionIdentityInputStatus.AMBIGUOUS,
                limitations=("Both module and package source candidates exist in the project.",),
            )
        candidate = configured[0]
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(project_root)
            digest = self._digest(resolved)
        except (FileNotFoundError, OSError, ValueError):
            return ExecutionIdentityInput(
                kind=ExecutionIdentityInputKind.PYTHON_MODULE,
                requested=name,
                configured_path=str(candidate),
                status=ExecutionIdentityInputStatus.RESOLUTION_FAILED,
                limitations=("The project module source candidate could not be bound safely.",),
            )
        return ExecutionIdentityInput(
            kind=ExecutionIdentityInputKind.PYTHON_MODULE,
            requested=name,
            identity_basis=ExecutionIdentityBasis.PROJECT_SOURCE,
            configured_path=str(candidate),
            resolved_path=str(resolved),
            loaded_path=None,
            content_digest=digest,
            status=ExecutionIdentityInputStatus.EXACT,
            limitations=(
                "Project source is bound without importing the module; runtime import use is not "
                "observed.",
            ),
        )

    def _native(self, value: str) -> ExecutionIdentityInput:
        relative = PurePath(value)
        if relative.is_absolute() or ".." in relative.parts or "\x00" in value:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Native identity path {value!r} must be project-relative.",
            )
        candidate = self.workspace.project_root / Path(relative)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.workspace.project_root.resolve())
        except FileNotFoundError:
            return ExecutionIdentityInput(
                kind=ExecutionIdentityInputKind.NATIVE_FILE,
                requested=value,
                configured_path=str(candidate),
                status=ExecutionIdentityInputStatus.MISSING,
            )
        except ValueError:
            return ExecutionIdentityInput(
                kind=ExecutionIdentityInputKind.NATIVE_FILE,
                requested=value,
                configured_path=str(candidate),
                status=ExecutionIdentityInputStatus.RESOLUTION_FAILED,
                limitations=("Resolved native file escapes the project root.",),
            )
        if not resolved.is_file():
            return ExecutionIdentityInput(
                kind=ExecutionIdentityInputKind.NATIVE_FILE,
                requested=value,
                configured_path=str(candidate),
                resolved_path=str(resolved),
                status=ExecutionIdentityInputStatus.RESOLUTION_FAILED,
                limitations=("Resolved native identity input is not a regular file.",),
            )
        return ExecutionIdentityInput(
            kind=ExecutionIdentityInputKind.NATIVE_FILE,
            requested=value,
            identity_basis=ExecutionIdentityBasis.EXPLICIT_FILE,
            configured_path=str(candidate),
            resolved_path=str(resolved),
            loaded_path=None,
            content_digest=self._digest(resolved),
            status=ExecutionIdentityInputStatus.EXACT,
            limitations=(
                "Configured native identity is bound; actual loader use is not observed.",
            ),
        )

    def _report(
        self,
        inputs: list[ExecutionIdentityInput] | tuple[ExecutionIdentityInput, ...],
    ) -> WorkloadExecutionIdentity:
        values = tuple(inputs)
        missing = tuple(
            item.requested
            for item in values
            if item.status
            in {
                ExecutionIdentityInputStatus.MISSING,
                ExecutionIdentityInputStatus.AMBIGUOUS,
                ExecutionIdentityInputStatus.RESOLUTION_FAILED,
                ExecutionIdentityInputStatus.NOT_OBSERVED,
            }
        )
        if not values:
            quality = ExecutionIdentityQuality.NOT_APPLICABLE
        elif not missing:
            quality = ExecutionIdentityQuality.EXACT
        else:
            quality = ExecutionIdentityQuality.PARTIAL
        content = {
            "quality": quality,
            "inputs": [item.model_dump(mode="json") for item in values],
            "missing_inputs": missing,
        }
        return WorkloadExecutionIdentity(
            identity_id=digest_model(content),
            quality=quality,
            inputs=values,
            missing_inputs=missing,
        )

    def _digest(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
