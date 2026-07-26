from __future__ import annotations

import json
import os
import shutil
import string
import tomllib
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, model_validator

from flamo.domain import (
    CommandSpec,
    DomainError,
    ErrorCode,
    OracleStrength,
    WorkloadDefinition,
    WorkloadInstance,
    digest_model,
)
from flamo.models import ContractModel
from flamo.storage import Workspace
from flamo.storage.atomic import atomic_write_json

Scalar = str | int | float | bool


class WorkloadOracleConfig(ContractModel):
    strength: OracleStrength = OracleStrength.EXECUTION_CHECK
    argv: Annotated[tuple[str, ...], Field(max_length=1_024)] = ()


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
    variants: Annotated[tuple[str, ...], Field(min_length=2, max_length=16)]
    design: Literal[
        "randomized_complete_blocks",
        "randomized",
        "fixed_order",
    ] = "randomized_complete_blocks"
    blocks: Annotated[int, Field(gt=0, le=1_000)] = 1
    primary_metric: str
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"]
    estimand: str = "median_paired_log_ratio"
    practical_threshold: Annotated[float, Field(ge=0)] = 0
    confidence_level: Annotated[float, Field(gt=0, lt=1)] = 0.95
    random_seed: Annotated[int, Field(ge=0)] = 0
    scaling_parameter: str | None = None
    scaling_values: Annotated[tuple[Scalar, ...], Field(max_length=1_000)] = ()

    @model_validator(mode="after")
    def unique_variants(self) -> ExperimentConfig:
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("experiment variants must be unique")
        if bool(self.scaling_parameter) != bool(self.scaling_values):
            raise ValueError("scaling_parameter and scaling_values must be declared together")
        if len(set(self.scaling_values)) != len(self.scaling_values):
            raise ValueError("experiment scaling values must be unique")
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
        self.project_config_path = workspace.project_root / "flamo.toml"
        self.approvals_path = workspace.paths.root / "approvals.json"

    def load(self) -> ProjectConfig:
        try:
            with self.project_config_path.open("rb") as stream:
                return ProjectConfig.model_validate(tomllib.load(stream))
        except FileNotFoundError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Project workload configuration is missing: {self.project_config_path}",
                remediation=("Create flamo.toml at the project root.",),
            ) from exc
        except (tomllib.TOMLDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Project workload configuration is invalid: {exc}",
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.load().workloads))

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
                remediation=(f"Run `flamo workload approve {name}` after review.",),
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
