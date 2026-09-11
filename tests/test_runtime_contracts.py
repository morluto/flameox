from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from flameox.mcp.capability_tools import Execution, SingleExecution
from flameox.runtime_contracts import CaptureTarget, EvidenceSource, PathSource, Source


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
def test_single_execution_omitted_kind_uses_advertised_default() -> None:
    """The public schema advertises SingleExecution.kind as optional (default "single")."""
    execution: Execution = TypeAdapter(Execution).validate_python({})
    assert isinstance(execution, SingleExecution)
    assert execution.kind == "single"


@pytest.mark.unit
def test_execution_union_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Execution).validate_python({"kind": "mystery"})


@pytest.mark.unit
def test_advertised_schema_matches_accepted_source_and_execution_inputs() -> None:
    """Fields the generated schema leaves optional must not fail request validation."""
    source_schema = TypeAdapter(Source).json_schema(ref_template="#/$defs/{model}")
    assert "kind" not in source_schema["$defs"]["PathSource"]["required"]
    assert source_schema["$defs"]["PathSource"]["properties"]["kind"]["default"] == "path"
    execution_schema = TypeAdapter(Execution).json_schema(ref_template="#/$defs/{model}")
    assert "kind" not in execution_schema["$defs"]["SingleExecution"].get("required", [])
    assert execution_schema["$defs"]["SingleExecution"]["properties"]["kind"]["default"] == "single"
