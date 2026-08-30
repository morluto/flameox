from __future__ import annotations

from typing import Literal

from packaging.requirements import InvalidRequirement, Requirement
from pydantic import ConfigDict, computed_field, model_validator

from flameox.action_graph import ActionId, NextAction, manual_action, tool_action
from flameox.application.preflight import PreflightService
from flameox.application.workloads import WorkloadService
from flameox.domain import (
    DomainError,
    ErrorCode,
    PreflightDisposition,
    PreflightReport,
    ProbeKind,
    RequirementKind,
    RequirementStatus,
)
from flameox.execution import SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import Workspace


class WorkloadDependencySetupResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    workload_name: str
    requested: tuple[str, ...]
    already_available: tuple[str, ...]
    preflight: PreflightReport
    workload_executed: Literal[False] = False
    environment_mutated: Literal[False] = False
    next_action: NextAction

    @model_validator(mode="after")
    def availability_is_a_partition(self) -> WorkloadDependencySetupResult:
        if len(set(self.requested)) != len(self.requested):
            raise ValueError("requested requirements must be unique")
        available = set(self.already_available)
        if tuple(item for item in self.requested if item in available) != self.already_available:
            raise ValueError("already-available requirements must be an ordered requested subset")
        expected = _dependency_next_action(self.preflight)
        if self.next_action != expected:
            raise ValueError("next action must match the active preflight result")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> PreflightDisposition:
        return self.preflight.disposition


def _dependency_next_action(
    preflight: PreflightReport,
) -> NextAction:
    if any(
        item.kind is RequirementKind.PYTHON_DISTRIBUTION
        and item.status is not RequirementStatus.AVAILABLE
        for item in preflight.requirements
    ):
        return manual_action(
            "Install the missing distributions in the workload's declared Python environment "
            "or select a workload that already provides them, then inspect dependencies again.",
            suggested_action=ActionId.GET_DECLARED_WORKFLOW,
        )
    if preflight.disposition in {
        PreflightDisposition.READY,
        PreflightDisposition.EXPLORATORY,
    }:
        return manual_action(
            "Choose a compatible adapter and declared parameters before planning capture.",
            suggested_action=ActionId.PLAN_CAPTURE,
            missing_arguments=("adapter", "parameters"),
        )
    return tool_action(ActionId.INSPECT_CAPABILITIES)


class WorkloadDependencyService:
    """Inspect distributions declared by one workload without mutating any environment."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.broker = broker or SubprocessBroker()

    async def prepare(self, workload_name: str) -> WorkloadDependencySetupResult:
        config = self.workloads.load().workloads.get(workload_name)
        if config is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unknown workload {workload_name!r}.",
                remediation=(
                    "Call list_declared_workflows with kind='workload' and choose a declared name.",
                ),
                next_action=tool_action(ActionId.LIST_DECLARED_WORKFLOWS, kind="workload"),
            )
        requirements = tuple(
            self._validated_requirement(item, workload_name=workload_name)
            for item in config.requirements.python_distributions
        )
        names = tuple(str(item) for item in requirements)
        preflight = await PreflightService(self.workspace, broker=self.broker).inspect(
            workload_name,
            mode=ProbeKind.ACTIVE,
        )
        available_names = {
            item.requirement
            for item in preflight.requirements
            if item.kind is RequirementKind.PYTHON_DISTRIBUTION
            and item.status is RequirementStatus.AVAILABLE
        }
        already_available = tuple(item for item in names if item in available_names)
        return WorkloadDependencySetupResult(
            workload_name=workload_name,
            requested=names,
            already_available=already_available,
            preflight=preflight,
            next_action=_dependency_next_action(preflight),
        )

    @staticmethod
    def _validated_requirement(value: str, *, workload_name: str) -> Requirement:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Declared Python distribution requirement is invalid: {value!r}.",
                details={"requirement": value},
                remediation=(
                    "Use a package name with an optional version specifier; direct URLs and "
                    "local paths are not workload dependency declarations.",
                ),
                next_action=tool_action(
                    ActionId.GET_DECLARED_WORKFLOW,
                    kind="workload",
                    name=workload_name,
                ),
            ) from exc
        if requirement.url is not None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Declared Python distribution must come from a package index: {value!r}.",
                details={"requirement": value},
                remediation=(
                    "Declare a distribution name and version range instead of a direct URL or "
                    "local path.",
                ),
                next_action=tool_action(
                    ActionId.GET_DECLARED_WORKFLOW,
                    kind="workload",
                    name=workload_name,
                ),
            )
        return requirement
