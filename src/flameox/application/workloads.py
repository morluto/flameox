from __future__ import annotations

import json
import os
import shutil
import string
import tomllib
from pathlib import Path, PurePath
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, model_validator

from flameox.domain import (
    CommandSpec,
    CursorCodec,
    DomainError,
    ErrorCode,
    OracleStrength,
    WorkloadDefinition,
    WorkloadInstance,
    digest_model,
)
from flameox.models import ContractModel
from flameox.storage import Workspace
from flameox.storage.atomic import atomic_write_json

Scalar = str | int | float | bool


class WorkloadOracleConfig(ContractModel):
    strength: OracleStrength = OracleStrength.EXECUTION_CHECK
    argv: Annotated[tuple[str, ...], Field(max_length=1_024)] = ()


class WorkloadRequirementsConfig(ContractModel):
    executables: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    python_distributions: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    capabilities: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    optional: Annotated[tuple[str, ...], Field(max_length=192)] = ()
    active: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    allow_exploratory: bool = False

    @model_validator(mode="after")
    def references_are_unique_and_declared(self) -> WorkloadRequirementsConfig:
        declared = (*self.executables, *self.python_distributions, *self.capabilities)
        if len(set(declared)) != len(declared):
            raise ValueError("workload requirements must be unique")
        if not set(self.optional).issubset(declared):
            raise ValueError("optional requirements must also be declared")
        if not set(self.active).issubset(self.capabilities):
            raise ValueError("active requirements must be declared capabilities")
        return self


AcceleratorIdentityRequirement = Literal[
    "cuda.driver",
    "cuda.runtime",
    "cuda.devices",
    "cuda.peer_topology",
]


class WorkloadEnvironmentIdentityConfig(ContractModel):
    required: Annotated[tuple[AcceleratorIdentityRequirement, ...], Field(max_length=4)] = ()

    @model_validator(mode="after")
    def requirements_are_unique(self) -> WorkloadEnvironmentIdentityConfig:
        if len(set(self.required)) != len(self.required):
            raise ValueError("environment identity requirements must be unique")
        return self


class WorkloadIdentityConfig(ContractModel):
    python_modules: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    native_files: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    environment: WorkloadEnvironmentIdentityConfig = Field(
        default_factory=WorkloadEnvironmentIdentityConfig
    )


class WorkloadConfig(ContractModel):
    argv: Annotated[tuple[str, ...], Field(min_length=1, max_length=1_024)]
    cwd: str = "."
    timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300
    parameters: dict[str, tuple[Scalar, ...]] = Field(
        default_factory=dict,
        max_length=128,
    )
    environment: dict[str, str] = Field(default_factory=dict, max_length=128)
    oracle: WorkloadOracleConfig | None = None
    requirements: WorkloadRequirementsConfig = Field(default_factory=WorkloadRequirementsConfig)
    writable_paths: Annotated[tuple[str, ...], Field(max_length=16)] = ()
    identity: WorkloadIdentityConfig = Field(default_factory=WorkloadIdentityConfig)

    @model_validator(mode="after")
    def validate_templates(self) -> WorkloadConfig:
        fields: set[str] = set()
        for value in (*self.argv, self.cwd, *self.environment.values()):
            fields.update(_template_fields(value))
        if self.oracle is not None:
            for value in self.oracle.argv:
                fields.update(_template_fields(value))
        undeclared = fields - set(self.parameters)
        if undeclared:
            raise ValueError(
                "template fields are not declared parameters: " + ", ".join(sorted(undeclared))
            )
        return self


class ExperimentConfig(ContractModel):
    workload: str
    variants: Annotated[tuple[str, ...], Field(max_length=16)] = ()
    design: Literal[
        "randomized_complete_blocks",
        "randomized",
        "fixed_order",
    ] = "randomized_complete_blocks"
    blocks: Annotated[int, Field(gt=0, le=1_000)] = 1
    factors: dict[str, Annotated[tuple[Scalar, ...], Field(min_length=1, max_length=32)]] = Field(
        default_factory=dict, max_length=8
    )
    combination_policy: Literal["cartesian", "explicit"] = "cartesian"
    combinations: Annotated[tuple[dict[str, Scalar], ...], Field(max_length=10_000)] = ()
    exclude: Annotated[tuple[dict[str, Scalar], ...], Field(max_length=1_000)] = ()
    treatment_factor: str | None = None
    max_trials: Annotated[int, Field(gt=0, le=100_000)] = 10_000
    analysis: Literal["performance", "outcome"] = "performance"
    outcome_goal: Literal["equivalence", "absence_of_failure", "bounded_rate"] | None = None
    minimum_attempts: Annotated[int, Field(gt=0, le=1_000)] | None = None
    maximum_attempts: Annotated[int, Field(gt=0, le=1_000)] | None = None
    primary_metric: str = "categorical_outcome"
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"] = "neutral"
    estimand: str = "median_paired_log_ratio"
    practical_threshold: Annotated[float, Field(ge=0)] = 0
    confidence_level: Annotated[float, Field(gt=0, lt=1)] = 0.95
    random_seed: Annotated[int, Field(ge=0)] = 0
    scaling_parameter: str | None = None
    scaling_values: Annotated[tuple[Scalar, ...], Field(max_length=1_000)] = ()

    @model_validator(mode="after")
    def valid_design(self) -> ExperimentConfig:
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("experiment variants must be unique")
        if bool(self.variants) == bool(self.factors):
            raise ValueError("declare either legacy variants or factors")
        if self.factors:
            if self.treatment_factor not in self.factors:
                raise ValueError("factor experiments require a declared treatment_factor")
            if self.scaling_parameter is not None or self.scaling_values:
                raise ValueError("factor experiments cannot use legacy scaling fields")
            if self.combination_policy == "cartesian" and self.combinations:
                raise ValueError("cartesian experiments cannot declare explicit combinations")
            if self.combination_policy == "explicit" and not self.combinations:
                raise ValueError("explicit experiments require combinations")
        elif self.combinations or self.exclude or self.treatment_factor is not None:
            raise ValueError("combination fields require factors")
        if bool(self.scaling_parameter) != bool(self.scaling_values):
            raise ValueError("scaling_parameter and scaling_values must be declared together")
        if len(set(self.scaling_values)) != len(self.scaling_values):
            raise ValueError("experiment scaling values must be unique")
        if self.analysis == "outcome":
            if self.outcome_goal is None:
                raise ValueError("outcome experiments require outcome_goal")
            minimum = self.minimum_attempts or self.blocks
            maximum = self.maximum_attempts or self.blocks
            if minimum > self.blocks or maximum < self.blocks or minimum > maximum:
                raise ValueError("fixed blocks must lie within declared attempt bounds")
        elif (
            self.outcome_goal is not None
            or self.minimum_attempts is not None
            or self.maximum_attempts is not None
        ):
            raise ValueError("outcome settings require analysis='outcome'")
        return self


class ProjectConfig(ContractModel):
    schema_version: Literal[1] = 1
    workloads: dict[str, WorkloadConfig] = Field(default_factory=dict, max_length=1_000)
    experiments: dict[str, ExperimentConfig] = Field(
        default_factory=dict,
        max_length=1_000,
    )

    @model_validator(mode="after")
    def experiments_reference_workloads(self) -> ProjectConfig:
        missing = sorted(
            {
                experiment.workload
                for experiment in self.experiments.values()
                if experiment.workload not in self.workloads
            }
        )
        if missing:
            raise ValueError("experiments reference unknown workloads: " + ", ".join(missing))
        return self


class ApprovalRecord(ContractModel):
    schema_version: Literal[1] = 1
    workloads: dict[str, str] = Field(default_factory=dict)


class ResolvedOracle(ContractModel):
    strength: OracleStrength
    command: CommandSpec


class DeclaredWorkflowSummary(ContractModel):
    kind: Literal["workload", "experiment"]
    name: str
    approval: Literal["approved", "unapproved", "not_applicable"]
    definition_id: str
    parameter_names: tuple[str, ...] = ()
    oracle_strength: OracleStrength | None = None
    timeout_seconds: float | None = None


class DeclaredWorkflowList(ContractModel):
    schema_version: int = 1
    configuration_id: str
    workflows: tuple[DeclaredWorkflowSummary, ...]
    returned: int
    truncated: bool
    next_cursor: str | None


class DeclaredWorkflowDetail(ContractModel):
    schema_version: int = 1
    configuration_id: str
    summary: DeclaredWorkflowSummary
    allowed_parameters: dict[str, tuple[Scalar, ...]]
    workload_name: str | None = None
    variants: tuple[str, ...] = ()
    design: str | None = None
    blocks: int | None = None
    primary_metric: str | None = None
    polarity: str | None = None
    estimand: str | None = None
    validation_spec_id: str | None = None


def _template_fields(value: str) -> set[str]:
    fields: set[str] = set()
    try:
        parsed = string.Formatter().parse(value)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                not field_name.isidentifier()
                or format_spec
                or conversion is not None
                or "." in field_name
                or "[" in field_name
            ):
                raise ValueError("only plain scalar placeholders such as {length} are allowed")
            fields.add(field_name)
    except ValueError as exc:
        raise ValueError(f"invalid workload template {value!r}: {exc}") from exc
    return fields


def _render(value: str, parameters: dict[str, Scalar]) -> str:
    _template_fields(value)
    try:
        return value.format_map(parameters)
    except KeyError as exc:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"Missing workload parameter {exc.args[0]!r}.",
        ) from exc


class WorkloadService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.project_config_path = workspace.project_root / "flameox.toml"
        self.approvals_path = workspace.paths.root / "approvals.json"

    def load(self) -> ProjectConfig:
        try:
            with self.project_config_path.open("rb") as stream:
                return ProjectConfig.model_validate(tomllib.load(stream))
        except FileNotFoundError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Project workload configuration is missing: {self.project_config_path}",
                remediation=("Create flameox.toml at the project root.",),
            ) from exc
        except (tomllib.TOMLDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Project workload configuration is invalid: {exc}",
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.load().workloads))

    def list_declared(
        self,
        *,
        kind: Literal["workload", "experiment"],
        approval: Literal["approved", "unapproved", "any"] = "any",
        limit: int,
        cursor: str | None = None,
    ) -> DeclaredWorkflowList:
        project = self.load()
        configuration_id = digest_model(project)
        query_id = digest_model({"kind": kind, "approval": approval})
        offset = 0
        if cursor is not None:
            position = CursorCodec.decode(
                cursor,
                namespace="declared_workflows",
                snapshot_id=configuration_id,
                scope_digest=query_id,
            )
            try:
                offset = int(position[0])
            except (IndexError, TypeError, ValueError) as exc:
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.") from exc

        names = sorted(project.workloads if kind == "workload" else project.experiments)
        summaries = tuple(self._workflow_summary(project, kind, name) for name in names)
        if kind == "workload" and approval != "any":
            summaries = tuple(item for item in summaries if item.approval == approval)
        selected = summaries[offset : offset + limit]
        next_offset = offset + len(selected)
        next_cursor = (
            CursorCodec.encode(
                namespace="declared_workflows",
                snapshot_id=configuration_id,
                scope_digest=query_id,
                position=(next_offset,),
            )
            if next_offset < len(summaries)
            else None
        )
        return DeclaredWorkflowList(
            configuration_id=configuration_id,
            workflows=selected,
            returned=len(selected),
            truncated=next_cursor is not None,
            next_cursor=next_cursor,
        )

    def get_declared(
        self,
        *,
        kind: Literal["workload", "experiment"],
        name: str,
    ) -> DeclaredWorkflowDetail:
        project = self.load()
        configuration_id = digest_model(project)
        summary = self._workflow_summary(project, kind, name)
        if kind == "workload":
            config = project.workloads[name]
            definition = self.definition(name)
            return DeclaredWorkflowDetail(
                configuration_id=configuration_id,
                summary=summary,
                allowed_parameters=config.parameters,
                validation_spec_id=definition.validation_spec_id,
            )
        experiment = project.experiments[name]
        workload = project.workloads[experiment.workload]
        return DeclaredWorkflowDetail(
            configuration_id=configuration_id,
            summary=summary,
            allowed_parameters=workload.parameters,
            workload_name=experiment.workload,
            variants=experiment.variants,
            design=experiment.design,
            blocks=experiment.blocks,
            primary_metric=experiment.primary_metric,
            polarity=experiment.polarity,
            estimand=experiment.estimand,
            validation_spec_id=self.definition(experiment.workload).validation_spec_id,
        )

    def _workflow_summary(
        self,
        project: ProjectConfig,
        kind: Literal["workload", "experiment"],
        name: str,
    ) -> DeclaredWorkflowSummary:
        try:
            if kind == "workload":
                workload_config = project.workloads[name]
                definition = self.definition(name)
                return DeclaredWorkflowSummary(
                    kind=kind,
                    name=name,
                    approval=(
                        "approved"
                        if definition.approved_definition_digest is not None
                        else "unapproved"
                    ),
                    definition_id=definition.workload_definition_id,
                    parameter_names=tuple(sorted(workload_config.parameters)),
                    oracle_strength=(
                        workload_config.oracle.strength
                        if workload_config.oracle is not None
                        else None
                    ),
                    timeout_seconds=workload_config.timeout_seconds,
                )
            experiment_config = project.experiments[name]
            return DeclaredWorkflowSummary(
                kind=kind,
                name=name,
                approval="not_applicable",
                definition_id=digest_model(experiment_config),
                parameter_names=tuple(
                    sorted(project.workloads[experiment_config.workload].parameters)
                ),
            )
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unknown declared {kind} {name!r}.",
                remediation=(f"Call list_declared_workflows with kind={kind!r}.",),
                details={"next_tool": "list_declared_workflows", "kind": kind},
            ) from exc

    def definition(self, name: str) -> WorkloadDefinition:
        config = self._selected(name)
        content = self._definition_content(name, config)
        definition_id = digest_model(content)
        approvals = self._approvals()
        approved = approvals.workloads.get(name)
        return WorkloadDefinition(
            workload_definition_id=definition_id,
            name=name,
            command_template=config.argv,
            parameter_names=tuple(sorted(config.parameters)),
            validation_spec_id=(
                digest_model(config.oracle.model_dump(mode="json"))
                if config.oracle is not None
                else None
            ),
            approved_definition_digest=approved if approved == definition_id else None,
        )

    def approve(self, name: str) -> WorkloadDefinition:
        definition = self.definition(name)
        approvals = self._approvals()
        updated = ApprovalRecord(
            workloads={
                **approvals.workloads,
                name: definition.workload_definition_id,
            }
        )
        with self.workspace.write_locked():
            atomic_write_json(self.approvals_path, updated.model_dump(mode="json"))
        return definition.model_copy(
            update={"approved_definition_digest": definition.workload_definition_id}
        )

    def resolve(
        self,
        name: str,
        overrides: dict[str, Scalar] | None = None,
        *,
        require_approval: bool,
    ) -> WorkloadInstance:
        config = self._selected(name)
        definition = self.definition(name)
        if require_approval and definition.approved_definition_digest is None:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Workload {name!r} is not approved at its current definition hash.",
                remediation=(f"Run `flameox workload approve {name}` after review.",),
                details={"workload_definition_id": definition.workload_definition_id},
            )
        selected = self._parameters(config, overrides or {})
        cwd = (self.workspace.project_root / _render(config.cwd, selected)).resolve()
        try:
            cwd.relative_to(self.workspace.project_root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Resolved workload directory leaves the project root.",
            ) from exc
        rendered_argv = tuple(_render(value, selected) for value in config.argv)
        command = CommandSpec(
            argv=(
                str(self._resolve_executable(rendered_argv[0], cwd)),
                *rendered_argv[1:],
            ),
            cwd=str(cwd),
            env_overrides={
                name: _render(value, selected) for name, value in config.environment.items()
            },
            timeout_seconds=config.timeout_seconds,
        )
        json_parameters = {name: cast(JsonValue, value) for name, value in selected.items()}
        content: dict[str, JsonValue] = {
            "workload_definition_id": definition.workload_definition_id,
            "command": command.model_dump(mode="json"),
            "parameters": json_parameters,
        }
        return WorkloadInstance(
            workload_instance_id=digest_model(content),
            workload_definition_id=definition.workload_definition_id,
            command=command,
            parameters=json_parameters,
        )

    def resolve_oracle(
        self,
        name: str,
        overrides: dict[str, Scalar] | None = None,
    ) -> ResolvedOracle | None:
        config = self._selected(name)
        if config.oracle is None:
            return None
        if not config.oracle.argv:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Workload {name!r} declares an oracle without argv.",
            )
        selected = self._parameters(config, overrides or {})
        cwd = (self.workspace.project_root / _render(config.cwd, selected)).resolve()
        rendered = tuple(_render(value, selected) for value in config.oracle.argv)
        return ResolvedOracle(
            strength=config.oracle.strength,
            command=CommandSpec(
                argv=(
                    str(self._resolve_executable(rendered[0], cwd)),
                    *rendered[1:],
                ),
                cwd=str(cwd),
                timeout_seconds=config.timeout_seconds,
            ),
        )

    def writable_targets(self, name: str) -> tuple[tuple[Path, str], ...]:
        config = self._selected(name)
        project_root = self.workspace.project_root.resolve()
        resolved: list[tuple[Path, str]] = []
        for value in config.writable_paths:
            relative = PurePath(value)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or "\x00" in value
                or relative.parts[0] in {".git", ".diagnostics", "flameox.toml"}
            ):
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Writable path {value!r} is not an allowed project build-output root.",
                )
            target = (project_root / Path(relative)).resolve(strict=True)
            try:
                target.relative_to(project_root)
            except ValueError as exc:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Writable path {value!r} escapes the project root.",
                ) from exc
            if not target.is_dir():
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Writable path {value!r} must be a pre-existing directory.",
                )
            stat = target.stat()
            identity = digest_model(
                {
                    "path": str(target),
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "mode": stat.st_mode,
                }
            )
            resolved.append((target, identity))
        if len({item[0] for item in resolved}) != len(resolved):
            raise DomainError(ErrorCode.EXECUTION_REFUSED, "Writable paths must be unique.")
        return tuple(resolved)

    def _resolve_executable(self, value: str, cwd: Path) -> Path:
        if os.sep in value or (os.altsep is not None and os.altsep in value):
            candidate = Path(value)
            candidate = candidate if candidate.is_absolute() else cwd / candidate
            resolved = candidate.parent.resolve() / candidate.name
        else:
            located = shutil.which(value)
            if located is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"Workload executable {value!r} is unavailable.",
                )
            resolved = Path(located).absolute()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Workload executable is not runnable: {resolved}",
            )
        return resolved

    def _selected(self, name: str) -> WorkloadConfig:
        try:
            return self.load().workloads[name]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unknown workload {name!r}.",
            ) from exc

    def _parameters(
        self,
        config: WorkloadConfig,
        overrides: dict[str, Scalar],
    ) -> dict[str, Scalar]:
        unexpected = set(overrides) - set(config.parameters)
        if unexpected:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Unknown workload parameters.",
                details={"parameters": sorted(unexpected)},
            )
        selected: dict[str, Scalar] = {}
        for name, choices in config.parameters.items():
            if not choices:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Workload parameter {name!r} has no allowed values.",
                )
            value = overrides.get(name, choices[0])
            if value not in choices:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    f"Value for {name!r} is outside the declared choices.",
                    details={"allowed": list(choices), "received": value},
                )
            selected[name] = value
        return selected

    def _definition_content(
        self,
        name: str,
        config: WorkloadConfig,
    ) -> dict[str, JsonValue]:
        return {
            "name": name,
            "definition": config.model_dump(mode="json"),
        }

    def _approvals(self) -> ApprovalRecord:
        if not self.approvals_path.exists():
            return ApprovalRecord()
        try:
            return ApprovalRecord.model_validate(json.loads(self.approvals_path.read_text()))
        except (json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Workload approval file is invalid.",
            ) from exc
