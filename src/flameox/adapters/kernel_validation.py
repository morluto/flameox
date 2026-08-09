from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from flameox.domain import ArtifactKind, DomainError, ErrorCode, digest_model
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace

ValidationStatus = Literal["pass", "fail", "inconclusive", "unsupported"]
MetricName = Literal[
    "max_abs_error",
    "max_rel_error",
    "mse",
    "rmse",
    "psnr",
    "cosine_similarity",
]
Comparator = Literal["<=", ">="]
BoundedName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"),
]
_DimensionValue = str | int | float | bool
Int64 = Annotated[int, Field(ge=-(2**63), le=2**63 - 1)]

_MAX_KERNEL_VALIDATION_BYTES = 64 * 1024 * 1024
_MAX_SUMMARY_LIMITATIONS = 100

_LOWER_IS_BETTER = {"max_abs_error", "max_rel_error", "mse", "rmse"}
_HIGHER_IS_BETTER = {"psnr", "cosine_similarity"}


def _require_finite(value: float | None, field: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


class KernelValidationReference(ContractModel):
    name: BoundedName
    version: Annotated[str, StringConstraints(max_length=200)] | None = None
    identity: Annotated[str, StringConstraints(min_length=1, max_length=500)]


class KernelValidationInput(ContractModel):
    dtype: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    shape: Annotated[tuple[Int64, ...], Field(max_length=16)]
    role: Annotated[str, StringConstraints(max_length=100)] | None = None

    @field_validator("shape")
    @classmethod
    def nonnegative_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item < 0 for item in value):
            raise ValueError("input shape dimensions must be nonnegative")
        return value


class KernelValidationMetric(ContractModel):
    name: MetricName
    value: float | None
    comparator: Comparator | None
    threshold: float | None
    unit: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    status: ValidationStatus
    limitation: Annotated[str, StringConstraints(max_length=500)] | None = None

    @model_validator(mode="after")
    def coherent_result(self) -> KernelValidationMetric:
        _require_finite(self.value, "metric value")
        _require_finite(self.threshold, "metric threshold")
        expected = "<=" if self.name in _LOWER_IS_BETTER else ">="
        if self.status in {"inconclusive", "unsupported"}:
            if self.limitation is None:
                raise ValueError(f"{self.status} metrics require a limitation")
            if any(item is not None for item in (self.value, self.comparator, self.threshold)):
                raise ValueError(
                    f"{self.status} metrics cannot declare a value, comparator, or threshold"
                )
            return self
        if self.value is None or self.threshold is None or self.comparator is None:
            raise ValueError("supported metrics require value, comparator, and threshold")
        if self.comparator != expected:
            raise ValueError(f"{self.name} requires comparator {expected}")
        if self.name in _LOWER_IS_BETTER and (self.value < 0 or self.threshold < 0):
            raise ValueError(f"{self.name} value and threshold must be nonnegative")
        if self.name == "cosine_similarity" and not (
            -1 <= self.value <= 1 and -1 <= self.threshold <= 1
        ):
            raise ValueError("cosine_similarity value and threshold must be between -1 and 1")
        passed = (
            self.value <= self.threshold
            if self.comparator == "<="
            else self.value >= self.threshold
        )
        expected_status: ValidationStatus = "pass" if passed else "fail"
        if self.status != expected_status:
            raise ValueError("metric status contradicts its value and threshold")
        return self


class KernelValidationFailure(ContractModel):
    coordinates: Annotated[tuple[int, ...], Field(max_length=16)] = ()
    expected: float | None = None
    actual: float | None = None
    absolute_error: float | None = None
    relative_error: float | None = None
    detail: Annotated[str, StringConstraints(max_length=500)] | None = None

    @model_validator(mode="after")
    def finite_values(self) -> KernelValidationFailure:
        for name in ("expected", "actual", "absolute_error", "relative_error"):
            _require_finite(getattr(self, name), name)
        if not (
            self.coordinates
            or self.expected is not None
            or self.actual is not None
            or self.absolute_error is not None
            or self.relative_error is not None
            or self.detail
        ):
            raise ValueError("representative failures must contain a substantive witness")
        return self


class KernelValidationOutput(ContractModel):
    name: BoundedName
    dtype: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    shape: Annotated[tuple[Int64, ...], Field(max_length=16)]
    status: ValidationStatus
    metrics: Annotated[tuple[KernelValidationMetric, ...], Field(max_length=16)] = ()
    representative_failures: Annotated[
        tuple[KernelValidationFailure, ...], Field(max_length=8)
    ] = ()
    limitations: Annotated[tuple[str, ...], Field(max_length=20)] = ()

    @field_validator("shape")
    @classmethod
    def nonnegative_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item < 0 for item in value):
            raise ValueError("output shape dimensions must be nonnegative")
        return value

    @model_validator(mode="after")
    def aggregate_metrics(self) -> KernelValidationOutput:
        names = [item.name for item in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("output metric names must be unique")
        if not self.metrics:
            if self.status not in {"inconclusive", "unsupported"} or not self.limitations:
                raise ValueError(
                    "outputs without metrics must be inconclusive or unsupported with a limitation"
                )
        else:
            expected = _aggregate_status(tuple(item.status for item in self.metrics))
            if self.status != expected:
                raise ValueError("output status contradicts its metrics")
        if self.status == "fail" and not self.representative_failures:
            raise ValueError("failed outputs require at least one representative failure")
        if self.status != "fail" and self.representative_failures:
            raise ValueError("representative failures require failed output status")
        return self


class KernelValidationCase(ContractModel):
    case_id: BoundedName
    dimensions: Annotated[dict[BoundedName, _DimensionValue], Field(max_length=32)] = Field(
        default_factory=dict
    )
    inputs: Annotated[dict[BoundedName, KernelValidationInput], Field(max_length=32)] = Field(
        default_factory=dict
    )
    seed: Int64 | None = None
    device: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    status: ValidationStatus
    outputs: Annotated[tuple[KernelValidationOutput, ...], Field(min_length=1, max_length=32)]
    limitations: Annotated[tuple[str, ...], Field(max_length=20)] = ()

    @field_validator("dimensions")
    @classmethod
    def finite_dimensions(cls, value: dict[str, _DimensionValue]) -> dict[str, _DimensionValue]:
        if any(isinstance(item, float) and not math.isfinite(item) for item in value.values()):
            raise ValueError("case dimensions must be finite")
        return value

    @model_validator(mode="after")
    def aggregate_outputs(self) -> KernelValidationCase:
        names = [item.name for item in self.outputs]
        if len(names) != len(set(names)):
            raise ValueError("case output names must be unique")
        expected = _aggregate_status(tuple(item.status for item in self.outputs))
        if self.status != expected:
            raise ValueError("case status contradicts its outputs")
        return self


class KernelValidationV1(ContractModel):
    schema_version: Literal["flameox.kernel-validation.v1"]
    producer: BoundedName
    producer_version: Annotated[str, StringConstraints(max_length=200)] | None = None
    reference: KernelValidationReference
    status: ValidationStatus
    coverage_complete: bool
    cases: Annotated[tuple[KernelValidationCase, ...], Field(min_length=1, max_length=1_000)]
    limitations: Annotated[tuple[str, ...], Field(max_length=20)] = ()

    @model_validator(mode="after")
    def aggregate_cases(self) -> KernelValidationV1:
        identifiers = [item.case_id for item in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("case IDs must be unique")
        child_status = _aggregate_status(tuple(item.status for item in self.cases))
        expected: ValidationStatus
        if child_status == "fail":
            expected = "fail"
        elif child_status == "pass" and self.coverage_complete:
            expected = "pass"
        elif child_status == "unsupported":
            expected = "unsupported"
        else:
            expected = "inconclusive"
        if self.status != expected:
            raise ValueError("document status contradicts case outcomes or coverage")
        if not self.coverage_complete and not self.limitations:
            raise ValueError("incomplete coverage requires a limitation")
        return self


def _aggregate_status(statuses: tuple[ValidationStatus, ...]) -> ValidationStatus:
    if "fail" in statuses:
        return "fail"
    if statuses and all(item == "pass" for item in statuses):
        return "pass"
    if statuses and all(item == "unsupported" for item in statuses):
        return "unsupported"
    return "inconclusive"


class KernelValidationExtractionResult(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    producer: str
    producer_version: str | None
    status: ValidationStatus
    coverage_complete: bool
    case_count: int
    output_count: int
    metric_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...]


class KernelValidationExtractor:
    name = "flameox.kernel-validation"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> KernelValidationExtractionResult:
        run = self.runs.read(run_id)
        registrations = tuple(
            item for item in run.artifacts if item.kind is ArtifactKind.VALIDATION_OUTPUT
        )
        if len(registrations) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one kernel-validation artifact.",
                run_id=run_id,
            )
        registration = registrations[0]
        artifact = self.artifacts.get(registration.artifact_id)
        document = self._load(artifact.payload_path)
        if registration.producer not in {None, "flameox.import", document.producer}:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Kernel-validation producer identity conflicts with the registration.",
                run_id=run_id,
            )
        if (
            registration.producer_version is not None
            and document.producer_version is not None
            and registration.producer_version != document.producer_version
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Kernel-validation producer version conflicts with the registration.",
                run_id=run_id,
            )

        case_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        for case in document.cases:
            for output in case.outputs:
                output_id = digest_model(
                    {
                        "artifact_id": registration.artifact_id,
                        "case_id": case.case_id,
                        "output": output.name,
                    }
                )
                case_rows.append(
                    {
                        "case_output_id": output_id,
                        "run_id": run_id,
                        "artifact_id": registration.artifact_id,
                        "case_id": case.case_id,
                        "output_name": output.name,
                        "case_status": case.status,
                        "output_status": output.status,
                        "coverage_complete": document.coverage_complete,
                        "producer": document.producer,
                        "producer_version": document.producer_version,
                        "reference_name": document.reference.name,
                        "reference_version": document.reference.version,
                        "reference_identity": document.reference.identity,
                        "device": case.device,
                        "seed": case.seed,
                        "dimensions_json": _json(case.dimensions),
                        "inputs_json": _json(
                            {
                                name: value.model_dump(mode="json")
                                for name, value in case.inputs.items()
                            }
                        ),
                        "dtype": output.dtype,
                        "shape": list(output.shape),
                        "representative_failures_json": _json(
                            [
                                item.model_dump(mode="json")
                                for item in output.representative_failures
                            ]
                        ),
                        "limitations": list(
                            dict.fromkeys((*case.limitations, *output.limitations))
                        ),
                    }
                )
                for metric in output.metrics:
                    metric_rows.append(
                        {
                            "metric_id": digest_model(
                                {"case_output_id": output_id, "metric": metric.name}
                            ),
                            "case_output_id": output_id,
                            "run_id": run_id,
                            "artifact_id": registration.artifact_id,
                            "case_id": case.case_id,
                            "output_name": output.name,
                            "metric_name": metric.name,
                            "value": metric.value,
                            "comparator": metric.comparator,
                            "threshold": metric.threshold,
                            "unit": metric.unit,
                            "status": metric.status,
                            "limitation": metric.limitation,
                        }
                    )
        if (
            len(case_rows) + len(metric_rows)
            > self.workspace.config.storage.max_rows_per_generation
        ):
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Kernel-validation evidence exceeds the workspace generation row limit.",
            )
        published = self.publisher.publish_rows_idempotent(
            {
                "kernel_validation_cases": case_rows,
                "kernel_validation_metrics": metric_rows,
            },
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "schema_version": document.schema_version,
                "document_digest": digest_model(document.model_dump(mode="json")),
            },
        )
        return KernelValidationExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            producer=document.producer,
            producer_version=document.producer_version,
            status=document.status,
            coverage_complete=document.coverage_complete,
            case_count=len(document.cases),
            output_count=len(case_rows),
            metric_count=len(metric_rows),
            corpus_commit_id=published.commit.commit_id,
            limitations=_summary_limitations(document),
        )

    @staticmethod
    def _load(path: Path) -> KernelValidationV1:
        try:
            if path.stat().st_size > _MAX_KERNEL_VALIDATION_BYTES:
                raise DomainError(
                    ErrorCode.ARTIFACT_TOO_LARGE,
                    "Kernel-validation JSON exceeds the 64 MiB contract limit.",
                )
            payload = path.read_text(encoding="utf-8")
            return KernelValidationV1.model_validate_json(payload, strict=True)
        except DomainError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a valid flameox kernel-validation v1 document.",
            ) from exc


def kernel_validation_json_schema() -> dict[str, JsonValue]:
    return KernelValidationV1.model_json_schema(mode="validation")


def _summary_limitations(document: KernelValidationV1) -> tuple[str, ...]:
    nested = (
        *document.limitations,
        *(
            limitation
            for case in document.cases
            for limitation in (
                *case.limitations,
                *(
                    limitation
                    for output in case.outputs
                    for limitation in (
                        *output.limitations,
                        *(metric.limitation for metric in output.metrics if metric.limitation),
                    )
                ),
            )
        ),
    )
    unique = tuple(dict.fromkeys(nested))
    if len(unique) <= _MAX_SUMMARY_LIMITATIONS:
        return unique
    return (
        *unique[: _MAX_SUMMARY_LIMITATIONS - 1],
        "Additional nested limitations were omitted from this bounded summary.",
    )


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
