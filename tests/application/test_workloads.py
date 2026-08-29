from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from flameox.action_graph import ActionId, ManualAction, ToolAction
from flameox.application import (
    ConfigurationOperation,
    ConfigureWorkloadRequest,
    DeclaredWorkflowKind,
    InferenceConfigurationResult,
    ProjectConfig,
    WorkloadConfig,
    WorkloadConfigurationResult,
    WorkloadConfigurationStatus,
    WorkloadOracleConfig,
    WorkloadService,
    parse_experiment_config,
)
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace

pytestmark = pytest.mark.integration


def test_oracle_declaration_requires_a_command() -> None:
    with pytest.raises(ValidationError, match="argv"):
        WorkloadOracleConfig.model_validate({"strength": "execution_check"})


def _request(
    name: str,
    *,
    operation: ConfigurationOperation = ConfigurationOperation.CREATE,
    argv: tuple[str, ...] = ("python", "-c", "print('ok')"),
    parameters: dict[str, tuple[str | int | float | bool, ...]] | None = None,
    environment: dict[str, str] | None = None,
    expected_configuration_id: str | None = None,
) -> ConfigureWorkloadRequest:
    return ConfigureWorkloadRequest(
        name=name,
        operation=operation,
        config=WorkloadConfig(
            argv=argv,
            parameters=parameters or {},
            environment=environment or {},
        ),
        expected_configuration_id=expected_configuration_id,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "missing",
            "configuration_id": "sha256:" + "1" * 64,
            "diagnostics": ["missing"],
        },
        {
            "status": "invalid",
            "diagnostics": [],
        },
        {
            "status": "valid",
            "configuration_id": "sha256:" + "1" * 64,
            "workload_names": [],
            "diagnostics": [],
            "next_action": {
                "kind": "tool",
                "action": "workflow.list",
                "arguments": {"kind": "workload", "limit": 50},
            },
        },
    ],
)
def test_workload_configuration_status_rejects_contradictory_states(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(WorkloadConfigurationStatus).validate_python(payload)


def test_configuration_receipts_reject_changed_paths_that_contradict_the_action() -> None:
    digest = "sha256:" + "1" * 64
    with pytest.raises(ValidationError):
        WorkloadConfigurationResult.model_validate(
            {
                "action": "unchanged",
                "name": "probe",
                "configuration_id": digest,
                "workload_definition_id": digest,
                "changed_paths": ["flameox.toml"],
            }
        )
    with pytest.raises(ValidationError):
        InferenceConfigurationResult.model_validate(
            {
                "kind": "server",
                "action": "updated",
                "name": "local",
                "configuration_id": digest,
                "definition_id": digest,
                "changed_paths": [],
            }
        )


@pytest.mark.parametrize("name", ("../escape", "nested/name", ".", "name with space"))
def test_inference_declaration_names_are_safe_identifiers(name: str) -> None:
    with pytest.raises(ValidationError, match="inference declaration names"):
        ProjectConfig.model_validate(
            {
                "inference_servers": {
                    name: {
                        "provider": "vllm",
                        "mode": "existing_local",
                        "base_url": "http://127.0.0.1:8000",
                        "model": "model",
                    }
                }
            }
        )


def test_configure_workload_is_idempotent_and_records_configuration_source(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    request = _request("probe")

    first = service.configure(request)
    config_text = (tmp_path / "flameox.toml").read_text()
    second = service.configure(request)

    assert first.action == "created"
    assert first.changed_paths == ("flameox.toml",)
    assert first.configuration_source == "agent"
    assert second.action == "unchanged"
    assert second.configuration_id == first.configuration_id
    assert second.workload_definition_id == first.workload_definition_id
    assert second.changed_paths == ()
    assert (tmp_path / "flameox.toml").read_text() == config_text
    assert (
        service.list_declared(kind=DeclaredWorkflowKind.WORKLOAD, limit=10).workflows[0].name
        == "probe"
    )


def test_literal_braces_round_trip_while_declared_placeholders_render(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    service.configure(
        _request(
            "json",
            argv=("python", "-c", 'print({"key": "{size}"})'),
            parameters={"size": (4,)},
        )
    )

    instance = service.resolve("json", {"size": 4})

    assert instance.command.argv[-1] == 'print({"key": "4"})'


def test_placeholder_renders_before_escaped_closing_brace(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    service.configure(
        _request(
            "json",
            argv=("python", "-c", 'print({{"batch": {batch}}})'),
            parameters={"batch": (4,)},
        )
    )

    instance = service.resolve("json", {"batch": 4})

    assert instance.command.argv[-1] == 'print({"batch": 4})'


def test_resolve_binds_the_workload_executable_using_its_effective_path(tmp_path: Path) -> None:
    executable = tmp_path / "bin" / ("tool.exe" if os.name == "nt" else "tool")
    executable.parent.mkdir()
    shutil.copy2(sys.executable, executable)
    executable.chmod(0o755)
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    service.configure(
        _request(
            "bound",
            argv=("tool", "--version"),
            environment={"PATH": "bin"},
        )
    )

    instance = service.resolve("bound")

    assert instance.executable_binding is not None
    assert instance.executable_binding.requested_token == "tool"
    assert instance.executable_binding.invocation_path == executable.absolute()
    assert instance.command.argv[0] == str(instance.executable_binding.invocation_path)


def test_unknown_plain_placeholder_is_still_rejected(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)

    with pytest.raises(ValueError, match="template fields are not declared parameters"):
        service.configure(_request("unknown", argv=("python", "-c", "print({missing})")))


def test_schema_one_legacy_experiment_fields_remain_loadable() -> None:
    project = ProjectConfig.model_validate(
        {
            "schema_version": 1,
            "workloads": {
                "scan": {
                    "argv": ["python", "-c", "print('{variant}', '{length}')"],
                    "parameters": {
                        "variant": ["baseline", "candidate"],
                        "length": [1, 2],
                    },
                }
            },
            "experiments": {
                "scan": {
                    "workload": "scan",
                    "variants": ["baseline", "candidate"],
                    "scaling_parameter": "length",
                    "scaling_values": [1, 2],
                }
            },
        }
    )

    experiment = project.experiments["scan"]
    assert experiment.variants == ("baseline", "candidate")
    assert experiment.scaling_parameter == "length"
    assert experiment.scaling_values == (1, 2)


def test_legacy_scaled_experiment_accepts_distinct_typed_scaling_values() -> None:
    project = ProjectConfig.model_validate(
        {
            "schema_version": 1,
            "workloads": {
                "scan": {
                    "argv": ["python", "-c", "print('{length}')"],
                    "parameters": {"length": [1, 1.0]},
                }
            },
            "experiments": {
                "scan": {
                    "workload": "scan",
                    "variants": ["baseline", "candidate"],
                    "scaling_parameter": "length",
                    "scaling_values": [1, 1.0],
                }
            },
        }
    )

    assert project.experiments["scan"].scaling_values == (1, 1.0)


@pytest.mark.parametrize(
    "shape",
    (
        {"variants": ["baseline", "candidate"]},
        {
            "variants": ["baseline", "candidate"],
            "scaling_parameter": "length",
            "scaling_values": [1, 2],
        },
        {
            "treatment_factor": "mode",
            "factors": {"mode": ["baseline", "candidate"]},
        },
        {
            "treatment_factor": "mode",
            "factors": {"mode": ["baseline", "candidate"]},
            "combination_policy": "explicit",
            "combinations": [{"mode": "baseline"}],
        },
    ),
)
@pytest.mark.parametrize(
    "analysis",
    (
        {},
        {"analysis": "outcome", "outcome_goal": "absence_of_failure"},
    ),
)
def test_experiment_parser_round_trips_each_legal_case(
    shape: dict[str, object],
    analysis: dict[str, object],
) -> None:
    parsed = parse_experiment_config({"workload": "scan", **shape, **analysis})

    reparsed = parse_experiment_config(parsed.model_dump(mode="python"))

    assert reparsed == parsed


@pytest.mark.parametrize(
    "config",
    (
        {
            "workload": "scan",
            "variants": ["baseline", "candidate"],
            "factors": {"mode": ["baseline", "candidate"]},
            "treatment_factor": "mode",
        },
        {
            "workload": "scan",
            "factors": {"mode": ["baseline", "candidate"]},
            "treatment_factor": "mode",
            "combinations": [{"mode": "baseline"}],
        },
        {
            "workload": "scan",
            "factors": {"mode": ["baseline", "candidate"]},
            "treatment_factor": "mode",
            "combination_policy": "explicit",
        },
        {
            "workload": "scan",
            "variants": ["baseline", "candidate"],
            "scaling_parameter": "length",
        },
        {
            "workload": "scan",
            "variants": ["baseline", "candidate"],
            "outcome_goal": "absence_of_failure",
        },
        {
            "workload": "scan",
            "variants": ["baseline", "candidate"],
            "analysis": "outcome",
        },
    ),
)
def test_experiment_parser_rejects_cross_case_states(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_experiment_config(config)


@pytest.mark.parametrize("goal", ("equivalence", "bounded_rate"))
def test_experiment_parser_rejects_outcome_goals_without_evaluable_contracts(
    goal: str,
) -> None:
    with pytest.raises(ValueError):
        parse_experiment_config(
            {
                "workload": "scan",
                "variants": ["baseline", "candidate"],
                "analysis": "outcome",
                "outcome_goal": goal,
            }
        )


def test_replace_requires_current_configuration_digest_and_preserves_unrelated_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "flameox.toml").write_text(
        """# keep this project note
schema_version = 1

[workloads.alpha]
argv = ["python", "-c", "print('alpha')"]

[workloads.other]
# keep this unrelated workload
argv = ["python", "-c", "print('other')"]

[experiments.compare]
workload = "alpha"
treatment_factor = "mode"
[experiments.compare.factors]
mode = ["baseline", "candidate"]
"""
    )
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    current = service.configuration_status()
    assert current.status == "valid"
    assert current.configuration_id is not None
    assert current.workload_names == ("alpha", "other")
    assert isinstance(current.next_action, ToolAction)
    assert current.next_action.action is ActionId.LIST_DECLARED_WORKFLOWS
    original = (tmp_path / "flameox.toml").read_text()

    with pytest.raises(DomainError) as stale:
        service.configure(
            _request(
                "alpha",
                operation=ConfigurationOperation.REPLACE,
                argv=("python", "-c", "print('new alpha')"),
                expected_configuration_id="sha256:" + "0" * 64,
            )
        )

    assert stale.value.code is ErrorCode.REVISION_CONFLICT
    assert (tmp_path / "flameox.toml").read_text() == original

    updated = service.configure(
        _request(
            "alpha",
            operation=ConfigurationOperation.REPLACE,
            argv=("python", "-c", "print('new alpha')"),
            expected_configuration_id=current.configuration_id,
        )
    )
    updated_text = (tmp_path / "flameox.toml").read_text()

    assert updated.action == "updated"
    assert "# keep this project note" in updated_text
    assert "# keep this unrelated workload" in updated_text
    assert "[workloads.other]" in updated_text
    assert "[experiments.compare]" in updated_text
    assert "print('new alpha')" in updated_text
    assert "print('alpha')" not in updated_text
    assert updated.configuration_source == "agent"


def test_invalid_configuration_is_reported_and_never_overwritten(tmp_path: Path) -> None:
    invalid = """schema_version = 1

[workloads.probe]
argv = ["python", "-c", "print('ok')"]

[experiments.broken]
workload = "missing"
treatment_factor = "mode"
[experiments.broken.factors]
mode = ["one", "two"]
"""
    (tmp_path / "flameox.toml").write_text(invalid)
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)

    status = service.configuration_status()
    assert status.status == "invalid"
    assert status.config_path == "flameox.toml"
    assert status.configuration_id is None
    assert isinstance(status.next_action, ManualAction)
    assert status.next_action.suggested_action is ActionId.CONFIGURE_WORKLOAD
    assert len(status.diagnostics) == 1
    assert len(status.diagnostics[0]) <= 512

    with pytest.raises(DomainError) as refused:
        service.configure(_request("new"))

    assert refused.value.code is ErrorCode.WORKSPACE_INVALID
    assert (tmp_path / "flameox.toml").read_text() == invalid


def test_structured_workload_recovery_repairs_semantically_invalid_configuration(
    tmp_path: Path,
) -> None:
    (tmp_path / "flameox.toml").write_text(
        """# preserve this note
schema_version = 1

[experiments.broken]
workload = "missing"
treatment_factor = "mode"
[experiments.broken.factors]
mode = ["one", "two"]
"""
    )
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)

    result = service.configure(_request("missing"))

    assert result.action == "created"
    repaired = (tmp_path / "flameox.toml").read_text()
    assert "# preserve this note" in repaired
    assert "[workloads.missing]" in repaired
    assert service.configuration_status().status == "valid"


def test_experiment_requires_explicit_treatment_factor_and_factors() -> None:
    config = parse_experiment_config(
        {
            "workload": "probe",
            "treatment_factor": "mode",
            "factors": {"mode": ("baseline", "candidate"), "length": (128, 256)},
        }
    )
    assert config.treatment_factor == "mode"
    assert config.factors["length"] == (128, 256)


def test_factor_experiment_baseline_value_uses_exact_scalar_identity() -> None:
    config = parse_experiment_config(
        {
            "workload": "probe",
            "treatment_factor": "mode",
            "baseline_value": 1.0,
            "factors": {"mode": (1, 1.0)},
        }
    )

    assert config.model_dump(mode="python")["baseline_value"] == 1.0
    with pytest.raises(ValueError, match="baseline_value must be one of"):
        parse_experiment_config(
            {
                "workload": "probe",
                "treatment_factor": "mode",
                "baseline_value": True,
                "factors": {"mode": (1, 1.0)},
            }
        )


def test_scalar_identity_distinguishes_exact_json_types() -> None:
    """Scalar identity distinguishes bool/int/float even when Python equality treats them equal."""
    from flameox.application import (
        scalar_contains,
        scalar_equal,
        scalar_identity,
        scalar_identity_set,
        scalar_subset,
    )

    # True is not 1, 1 is not 1.0
    assert scalar_identity(True) != scalar_identity(1)
    assert scalar_identity(False) != scalar_identity(0)
    assert scalar_identity(1) != scalar_identity(1.0)
    assert scalar_identity(0) != scalar_identity(0.0)
    assert scalar_identity(1) != scalar_identity("1")

    assert not scalar_equal(True, 1)
    assert not scalar_equal(False, 0)
    assert not scalar_equal(1, 1.0)
    assert not scalar_equal(1, "1")
    assert scalar_equal(1, 1)
    assert scalar_equal(True, True)
    assert scalar_equal("a", "a")

    # scalar_contains uses exact type
    assert scalar_contains(1, (1,))
    assert not scalar_contains(True, (1,))
    assert not scalar_contains(1, (1.0,))
    assert not scalar_contains(1.0, (1,))
    assert not scalar_contains(1, (True,))
    assert not scalar_contains(0, (False,))
    assert scalar_contains(True, (True,))
    assert scalar_contains(False, (False,))

    # scalar_identity_set treats distinct-typed values as distinct
    ids = scalar_identity_set([True, 1, 1.0, "1", True])
    assert len(ids) == 4
    ids2 = scalar_identity_set([1, 1, 1])
    assert len(ids2) == 1

    assert scalar_subset([1, 2], [1, 2, 3])
    assert not scalar_subset([True], [1])
    assert not scalar_subset([1.0], [1])
    assert not scalar_subset([1], [1.0])
    assert scalar_subset([True], [True])


def test_resolve_rejects_int_for_bool_choice(tmp_path: Path) -> None:
    """Regression for #258: integer 1 must not authorize a boolean True choice."""
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    service.configure(
        ConfigureWorkloadRequest(
            name="probe",
            operation=ConfigurationOperation.CREATE,
            config=WorkloadConfig(
                argv=("python", "-c", "print({mode})"),
                parameters={"mode": (True,)},
            ),
        )
    )

    with pytest.raises(DomainError) as exc:
        service.resolve("probe", {"mode": 1})
    assert exc.value.code is ErrorCode.INVALID_CAPTURE_PLAN

    with pytest.raises(DomainError) as exc2:
        service.resolve("probe", {"mode": 1.0})
    assert exc2.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_resolve_rejects_float_for_int_choice(tmp_path: Path) -> None:
    """Regression for #258: float 1.0 must not authorize an integer 1 choice."""
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    service.configure(
        ConfigureWorkloadRequest(
            name="probe",
            operation=ConfigurationOperation.CREATE,
            config=WorkloadConfig(
                argv=("python", "-c", "print({mode})"),
                parameters={"mode": (1,)},
            ),
        )
    )

    with pytest.raises(DomainError) as exc:
        service.resolve("probe", {"mode": 1.0})
    assert exc.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_resolve_rejects_bool_for_int_choice(tmp_path: Path) -> None:
    """Regression for #258: bool True must not authorize an integer 1 choice."""
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    service.configure(
        ConfigureWorkloadRequest(
            name="probe",
            operation=ConfigurationOperation.CREATE,
            config=WorkloadConfig(
                argv=("python", "-c", "print({mode})"),
                parameters={"mode": (1,)},
            ),
        )
    )

    with pytest.raises(DomainError) as exc:
        service.resolve("probe", {"mode": True})
    assert exc.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_resolve_accepts_same_type_choices(tmp_path: Path) -> None:
    """Regression for #258: same-typed values must still authorize normally."""
    workspace = Workspace.initialize(tmp_path)
    service = WorkloadService(workspace)
    service.configure(
        ConfigureWorkloadRequest(
            name="probe",
            operation=ConfigurationOperation.CREATE,
            config=WorkloadConfig(
                argv=("python", "-c", "print({mode})"),
                parameters={"mode": (1, 2, 3)},
            ),
        )
    )

    instance = service.resolve("probe", {"mode": 2})
    assert instance.parameters["mode"] == 2
