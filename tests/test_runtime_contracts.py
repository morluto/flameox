from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from flameox.runtime_contracts import (
    CaptureTarget,
    CompareArguments,
    EvidenceSource,
    ExperimentCase,
    ExperimentDesign,
    PathSource,
    Source,
)


@pytest.mark.unit
def test_capture_target_rejects_policy_blocked_environment_override() -> None:
    with pytest.raises(ValidationError, match=r"PYTHONPATH.*blocked by policy"):
        CaptureTarget(
            argv=["python"],
            cwd=str(Path.cwd()),
            provider_id="direct",
            environment={"PYTHONPATH": "unsafe"},
        )


@pytest.mark.unit
def test_capture_target_requires_an_absolute_working_directory() -> None:
    with pytest.raises(ValidationError, match="cwd must be an absolute path"):
        CaptureTarget(argv=["python"], cwd=".", provider_id="direct")


@pytest.mark.unit
def test_path_source_omitted_kind_uses_advertised_default() -> None:
    """The public schema advertises PathSource.kind as optional (default "path")."""
    source: Source = TypeAdapter(Source).validate_python({"path": "/tmp/artifact.perf"})
    assert isinstance(source, PathSource)
    assert source.kind == "path"


@pytest.mark.unit
def test_evidence_source_omitted_kind_still_selects_evidence_member() -> None:
    source: Source = TypeAdapter(Source).validate_python({"evidence_id": "a" * 64})
    assert isinstance(source, EvidenceSource)
    assert source.kind == "evidence"


@pytest.mark.unit
def test_source_union_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Source).validate_python({"kind": "mystery", "path": "/tmp/x"})


@pytest.mark.unit
def test_advertised_schema_matches_accepted_source_inputs() -> None:
    """Fields the generated schema leaves optional must not fail request validation."""
    source_schema = TypeAdapter(Source).json_schema(ref_template="#/$defs/{model}")
    assert "kind" not in source_schema["$defs"]["PathSource"]["required"]
    assert source_schema["$defs"]["PathSource"]["properties"]["kind"]["default"] == "path"


@pytest.mark.unit
def test_compare_metric_rejects_empty_value_at_public_contract() -> None:
    with pytest.raises(ValidationError):
        CompareArguments(metric="")


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExperimentCase(name="case", argv=["bad\x00argument"]),
        lambda: ExperimentCase(name="case", environment={"BAD=NAME": "value"}),
        lambda: ExperimentDesign(
            cases=[ExperimentCase(name="a"), ExperimentCase(name="b")],
            blocks=1,
            seed=1,
            metric="wall_time_ns",
            estimand="median_difference",
            practical_threshold=0,
            semantic_oracle=["bad\x00command"],
        ),
    ],
)
def test_experiment_commands_use_the_same_strict_validation_as_capture_targets(
    factory: Any,
) -> None:
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.unit
def test_experiment_seed_rejects_values_outside_canonical_json_domain() -> None:
    def design(seed: int) -> ExperimentDesign:
        return ExperimentDesign(
            cases=[ExperimentCase(name="a"), ExperimentCase(name="b")],
            blocks=1,
            seed=seed,
            metric="wall_time_ns",
            estimand="mean_difference",
            practical_threshold=0,
        )

    design(2**53 - 1)
    design(-(2**53) + 1)
    with pytest.raises(ValidationError, match="less than or equal to"):
        design(2**53)
    with pytest.raises(ValidationError, match="greater than or equal to"):
        design(-(2**53))
