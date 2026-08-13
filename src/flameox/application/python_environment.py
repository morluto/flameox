from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement

from flameox.application.workloads import WorkloadService
from flameox.domain import DomainError, ErrorCode
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.storage import Workspace

_DISTRIBUTION_PROBE = """
import importlib.metadata
import json
import sys

result = {}
for name in sys.argv[1:]:
    try:
        result[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result[name] = None
print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
""".strip()


@dataclass(frozen=True, slots=True)
class PythonEnvironmentObservation:
    interpreter: Path
    interpreter_sha256: str
    versions: dict[str, str | None]


class PythonEnvironmentProbe:
    """Inspect distributions through the exact interpreter a workload executes."""

    def __init__(self, workspace: Workspace, *, broker: SubprocessBroker | None = None) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()

    async def inspect(
        self,
        workload_name: str,
        requirements: tuple[Requirement, ...],
    ) -> PythonEnvironmentObservation:
        instance = WorkloadService(self.workspace).resolve(workload_name)
        interpreter = instance.executable_binding.invocation_path
        if not interpreter.name.casefold().startswith("python"):
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Python distribution requirements need a workload whose declared executable "
                "is a Python interpreter.",
                details={
                    "workload_name": workload_name,
                    "executable": str(interpreter),
                },
                remediation=(
                    "Declare the workload with an explicit Python interpreter, or remove its "
                    "Python distribution requirements.",
                ),
            )
        names = tuple(dict.fromkeys(requirement.name for requirement in requirements))
        outcome = await self.broker.run(
            ExecutionRequest(
                argv=(str(interpreter), "-I", "-c", _DISTRIBUTION_PROBE, *names),
                executable_binding=instance.executable_binding,
                cwd=Path(instance.command.cwd),
                environment_allowlist=(),
                allowed_working_roots=(self.workspace.project_root,),
                timeout_seconds=min(30, instance.command.timeout_seconds),
                max_output_bytes=64 * 1024,
            )
        )
        if outcome.process.exit_code != 0:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The workload Python environment could not report distribution metadata.",
                details={
                    "workload_name": workload_name,
                    "interpreter": str(interpreter),
                    "exit_code": outcome.process.exit_code,
                },
                remediation=(
                    "Check that the declared interpreter can run isolated stdlib metadata "
                    "queries, then retry preflight.",
                ),
            )
        try:
            payload = json.loads(outcome.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The workload Python environment returned invalid distribution metadata.",
                details={"workload_name": workload_name, "interpreter": str(interpreter)},
            ) from exc
        if not isinstance(payload, dict) or set(payload) != set(names):
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The workload Python environment returned incomplete distribution metadata.",
                details={"workload_name": workload_name, "interpreter": str(interpreter)},
            )
        versions: dict[str, str | None] = {}
        for name in names:
            value = payload[name]
            if value is not None and (not isinstance(value, str) or not value or len(value) > 200):
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "The workload Python environment returned an invalid package version.",
                    details={"workload_name": workload_name, "distribution": name},
                )
            versions[name] = value
        return PythonEnvironmentObservation(
            interpreter=interpreter,
            interpreter_sha256=instance.executable_binding.identity.sha256,
            versions=versions,
        )
