from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Literal

from flameox.application.workloads import WorkloadService
from flameox.domain import (
    DomainError,
    ErrorCode,
    ExecutionIdentityInput,
    WorkloadExecutionIdentity,
    digest_model,
)
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.storage import Workspace

_MODULE_PROBE = """
import importlib
import importlib.metadata
import json
import sys

values = []
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
        path = getattr(module, "__file__", None)
        distributions = importlib.metadata.packages_distributions().get(
            name.split(".", 1)[0], []
        )
        distribution = distributions[0] if len(distributions) == 1 else None
        try:
            distribution_version = (
                importlib.metadata.version(distribution) if distribution else None
            )
        except importlib.metadata.PackageNotFoundError:
            distribution_version = None
        values.append({
            "name": name,
            "path": path,
            "distribution": distribution,
            "version": distribution_version,
            "status": "exact" if path else "ambiguous",
        })
    except BaseException as error:
        values.append({
            "name": name,
            "path": None,
            "status": "resolution_failed",
            "error": f"{type(error).__name__}: {error}",
        })
print(json.dumps(values, separators=(",", ":")))
"""


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

    def plan(self, workload_name: str) -> WorkloadExecutionIdentity:
        config = self.workloads.load().workloads[workload_name]
        inputs = [
            ExecutionIdentityInput(
                kind="python_module",
                requested=name,
                status="not_observed",
                limitations=("Resolved and loaded path is observed immediately before execution.",),
            )
            for name in config.identity.python_modules
        ]
        inputs.extend(self._native(value) for value in config.identity.native_files)
        return self._report(inputs)

    async def observe(
        self,
        workload_name: str,
        *,
        parameters: dict[str, str | int | float | bool],
    ) -> WorkloadExecutionIdentity:
        config = self.workloads.load().workloads[workload_name]
        instance = self.workloads.resolve(
            workload_name,
            parameters,
            require_approval=False,
        )
        inputs: list[ExecutionIdentityInput] = []
        if config.identity.python_modules:
            inputs.extend(
                await self._modules(
                    config.identity.python_modules,
                    executable=instance.command.argv[0],
                    cwd=Path(instance.command.cwd),
                    environment=instance.command.env_overrides,
                )
            )
        inputs.extend(self._native(value) for value in config.identity.native_files)
        return self._report(inputs)

    async def _modules(
        self,
        names: tuple[str, ...],
        *,
        executable: str,
        cwd: Path,
        environment: dict[str, str],
    ) -> tuple[ExecutionIdentityInput, ...]:
        if "python" not in Path(executable).name.lower():
            return tuple(
                ExecutionIdentityInput(
                    kind="python_module",
                    requested=name,
                    status="resolution_failed",
                    limitations=("Declared Python modules require a Python workload executable.",),
                )
                for name in names
            )
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(executable, "-c", _MODULE_PROBE, *names),
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
            if outcome.process.exit_code != 0:
                raise ValueError(outcome.stderr.decode(errors="replace"))
            values = json.loads(outcome.stdout)
            if (
                not isinstance(values, list)
                or len(values) != len(names)
                or any(not isinstance(value, dict) for value in values)
            ):
                raise ValueError("module identity probe returned an invalid result set")
        except (DomainError, json.JSONDecodeError, ValueError) as error:
            return tuple(
                ExecutionIdentityInput(
                    kind="python_module",
                    requested=name,
                    status="resolution_failed",
                    limitations=(f"Module identity probe failed: {error}",),
                )
                for name in names
            )
        results: list[ExecutionIdentityInput] = []
        for name, value in zip(names, values, strict=True):
            path_value = value.get("path")
            resolved = Path(path_value).resolve() if isinstance(path_value, str) else None
            distribution = value.get("distribution")
            distribution_version = value.get("version")
            results.append(
                ExecutionIdentityInput(
                    kind="python_module",
                    requested=name,
                    resolved_path=str(resolved) if resolved is not None else None,
                    loaded_path=str(resolved) if resolved is not None else None,
                    distribution=distribution if isinstance(distribution, str) else None,
                    version=(
                        distribution_version
                        if isinstance(distribution_version, str)
                        else None
                    ),
                    content_digest=self._digest(resolved) if resolved is not None else None,
                    status=(
                        "exact"
                        if value.get("status") == "exact" and resolved is not None
                        else "resolution_failed"
                    ),
                    limitations=(
                        (str(value.get("error")),)
                        if value.get("error") is not None
                        else ()
                    ),
                )
            )
        return tuple(results)

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
                kind="native_file",
                requested=value,
                configured_path=str(candidate),
                status="missing",
            )
        except ValueError:
            return ExecutionIdentityInput(
                kind="native_file",
                requested=value,
                configured_path=str(candidate),
                status="resolution_failed",
                limitations=("Resolved native file escapes the project root.",),
            )
        if not resolved.is_file():
            return ExecutionIdentityInput(
                kind="native_file",
                requested=value,
                configured_path=str(candidate),
                resolved_path=str(resolved),
                status="resolution_failed",
                limitations=("Resolved native identity input is not a regular file.",),
            )
        return ExecutionIdentityInput(
            kind="native_file",
            requested=value,
            configured_path=str(candidate),
            resolved_path=str(resolved),
            loaded_path=None,
            content_digest=self._digest(resolved),
            status="exact",
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
            if item.status in {"missing", "ambiguous", "resolution_failed", "not_observed"}
        )
        quality: Literal["exact", "partial", "unknown", "not_applicable"]
        if not values:
            quality = "not_applicable"
        elif not missing:
            quality = "exact"
        else:
            quality = "partial"
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
