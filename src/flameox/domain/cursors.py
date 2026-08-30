from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from flameox.domain.errors import DomainError, ErrorCode

type CursorValue = str | int
type CursorPosition = tuple[CursorValue, ...]
type CursorComponentKind = Literal["string", "integer"]


class CursorNamespace(StrEnum):
    """Closed set of query families that may issue continuation handles."""

    ARTIFACTS = "artifacts"
    ARTIFACT_REDUCTIONS = "artifact_reductions"
    CALL_EDGES = "call_edges"
    DECLARED_WORKFLOWS = "declared_workflows"
    EXPERIMENT_TRIALS = "experiment-trials"
    FINDINGS = "findings"
    FIND_REPEATED_OPERATION_SEQUENCES = "find_repeated_operation_sequences"
    GET_OPERATION_WINDOW = "get_operation_window"
    GET_PROCESS_SNAPSHOT = "get_process_snapshot"
    INFERENCE_REQUESTS = "inference_requests"
    INVESTIGATIONS = "investigations"
    MEASUREMENTS = "measurements"
    EXECUTION_ANALYSIS = "execution_analysis"
    NORMALIZED_TRACE_WINDOW = "normalized_trace_window"
    OPERATION_TRANSITIONS = "operation_transitions"
    PIPELINES = "pipelines"
    RUNS = "runs"
    STACK_EXAMPLES = "stack_examples"
    STATIC_CANDIDATES = "static_candidates"
    TRACE_WINDOW = "trace_window"
    TRITON_AUTOTUNE_SELECTIONS = "triton_autotune_selections"


@dataclass(frozen=True, slots=True)
class CursorPositionSpec:
    components: tuple[CursorComponentKind, ...]
    max_age_seconds: int = 900


CURSOR_POSITION_SPECS = MappingProxyType(
    {
        CursorNamespace.ARTIFACTS: CursorPositionSpec(("string",)),
        CursorNamespace.ARTIFACT_REDUCTIONS: CursorPositionSpec(("string",)),
        CursorNamespace.CALL_EDGES: CursorPositionSpec(("integer", "string", "string", "string")),
        CursorNamespace.DECLARED_WORKFLOWS: CursorPositionSpec(("integer",)),
        CursorNamespace.EXPERIMENT_TRIALS: CursorPositionSpec(("integer",)),
        CursorNamespace.FINDINGS: CursorPositionSpec(("integer",)),
        CursorNamespace.FIND_REPEATED_OPERATION_SEQUENCES: CursorPositionSpec(("integer",)),
        CursorNamespace.GET_OPERATION_WINDOW: CursorPositionSpec(("integer",)),
        CursorNamespace.GET_PROCESS_SNAPSHOT: CursorPositionSpec(("integer",)),
        CursorNamespace.INFERENCE_REQUESTS: CursorPositionSpec(("string",)),
        CursorNamespace.INVESTIGATIONS: CursorPositionSpec(("integer",)),
        CursorNamespace.MEASUREMENTS: CursorPositionSpec(("string",)),
        CursorNamespace.EXECUTION_ANALYSIS: CursorPositionSpec(("integer",)),
        CursorNamespace.NORMALIZED_TRACE_WINDOW: CursorPositionSpec(
            ("string", "string", "integer", "string")
        ),
        CursorNamespace.OPERATION_TRANSITIONS: CursorPositionSpec(("integer",)),
        CursorNamespace.PIPELINES: CursorPositionSpec(("string", "string")),
        CursorNamespace.RUNS: CursorPositionSpec(("string", "string")),
        CursorNamespace.STACK_EXAMPLES: CursorPositionSpec(
            ("integer", "string", "string", "string")
        ),
        CursorNamespace.STATIC_CANDIDATES: CursorPositionSpec(("string",)),
        CursorNamespace.TRACE_WINDOW: CursorPositionSpec(("integer", "string")),
        CursorNamespace.TRITON_AUTOTUNE_SELECTIONS: CursorPositionSpec(("string",)),
    }
)


def validate_cursor_position(
    namespace: CursorNamespace,
    position: object,
) -> CursorPosition:
    """Validate the complete position shape before it can influence a query."""

    spec = CURSOR_POSITION_SPECS[namespace]
    if not isinstance(position, (tuple, list)) or len(position) != len(spec.components):
        raise _invalid_position(namespace)
    validated: list[CursorValue] = []
    for value, component in zip(position, spec.components, strict=True):
        if component == "string":
            if not isinstance(value, str) or not value or len(value) > 512:
                raise _invalid_position(namespace)
        elif (
            not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 2**63 - 1
        ):
            raise _invalid_position(namespace)
        validated.append(value)
    return tuple(validated)


def _invalid_position(namespace: CursorNamespace) -> DomainError:
    return DomainError(
        ErrorCode.STALE_CURSOR,
        "Cursor position is invalid.",
        details={"namespace": namespace.value},
    )
