from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from flameox.adapters.kernel_validation import (
    KernelValidationExtractor,
    KernelValidationV1,
    kernel_validation_json_schema,
)
from flameox.application import ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _document() -> dict[str, Any]:
    return {
        "schema_version": "flameox.kernel-validation.v1",
        "producer": "kernel-tests",
        "producer_version": "1.2.3",
        "reference": {
            "name": "torch.reference",
            "version": "2.9",
            "identity": "sha256:" + "a" * 64,
        },
        "status": "pass",
        "coverage_complete": True,
        "cases": [
            {
                "case_id": "square-fp32-128",
                "dimensions": {"size": 128, "transposed": False},
                "inputs": {"left": {"dtype": "float32", "shape": [128, 128], "role": "input"}},
                "seed": 42,
                "device": "cuda:0-sm86",
                "status": "pass",
                "outputs": [
                    {
                        "name": "result",
                        "dtype": "float32",
                        "shape": [128, 128],
                        "status": "pass",
                        "metrics": [
                            {
                                "name": "max_abs_error",
                                "value": 0.0001,
                                "comparator": "<=",
                                "threshold": 0.001,
                                "unit": "absolute",
                                "status": "pass",
                            },
                            {
                                "name": "cosine_similarity",
                                "value": 0.9999,
                                "comparator": ">=",
                                "threshold": 0.999,
                                "unit": "ratio",
                                "status": "pass",
                            },
                        ],
                    }
                ],
            }
        ],
    }


def _import(workspace: Workspace, path: Path) -> str:
    return (
        ImportService(workspace)
        .import_artifact(ImportArtifactRequest(path=path, kind=ArtifactKind.VALIDATION_OUTPUT))
        .run.run_id
    )


def test_kernel_validation_round_trips_and_extracts_idempotently(tmp_path: Path) -> None:
    document = KernelValidationV1.model_validate(_document())
    assert KernelValidationV1.model_validate_json(document.model_dump_json()) == document
    assert kernel_validation_json_schema()["additionalProperties"] is False
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "validation.json"
    source.write_text(document.model_dump_json())
    run_id = _import(workspace, source)

    result = KernelValidationExtractor(workspace).extract(run_id)
    repeated = KernelValidationExtractor(workspace).extract(run_id)

    assert result.status == "pass"
    assert result.case_count == 1
    assert result.output_count == 1
    assert result.metric_count == 2
    assert repeated.corpus_commit_id == result.corpus_commit_id
    with Catalog(workspace).open_snapshot() as snapshot:
        cases = snapshot.execute(
            "SELECT case_id, output_name, shape FROM kernel_validation_cases"
        ).fetchall()
        metrics = snapshot.execute(
            "SELECT metric_name, comparator, status FROM kernel_validation_metrics "
            "ORDER BY metric_name"
        ).fetchall()
    assert cases == [("square-fp32-128", "result", [128, 128])]
    assert metrics == [
        ("cosine_similarity", ">=", "pass"),
        ("max_abs_error", "<=", "pass"),
    ]


def test_published_kernel_validation_schema_matches_the_model() -> None:
    published = json.loads(
        (
            PROJECT_ROOT / "src" / "flameox" / "schemas" / "kernel-validation-v1.schema.json"
        ).read_text()
    )

    assert published == kernel_validation_json_schema()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda value: value.update(status="fail"), "document status contradicts"),
        (
            lambda value: value["cases"][0]["outputs"][0]["metrics"][0].update(comparator=">="),
            "requires comparator",
        ),
        (
            lambda value: value["cases"][0]["outputs"][0]["metrics"][0].update(value=float("nan")),
            "must be finite",
        ),
        (
            lambda value: value.update(coverage_complete=False),
            "document status contradicts",
        ),
    ),
)
def test_kernel_validation_rejects_contradictory_or_nonfinite_documents(
    mutation: object,
    match: str,
) -> None:
    payload = _document()
    assert callable(mutation)
    mutation(payload)

    with pytest.raises(ValidationError, match=match):
        KernelValidationV1.model_validate(payload)


def test_kernel_validation_failed_output_requires_bounded_examples() -> None:
    payload = _document()
    metric = payload["cases"][0]["outputs"][0]["metrics"][0]
    metric.update(value=0.1, status="fail")
    payload["cases"][0]["outputs"][0].update(status="fail")
    payload["cases"][0].update(status="fail")
    payload.update(status="fail")

    with pytest.raises(ValidationError, match="representative failure"):
        KernelValidationV1.model_validate(payload)

    payload["cases"][0]["outputs"][0]["representative_failures"] = [
        {"coordinates": [3, 7], "expected": 1.0, "actual": 1.1, "absolute_error": 0.1}
    ]
    assert KernelValidationV1.model_validate(payload).status == "fail"


def test_kernel_validation_extractor_rejects_unknown_schema(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "validation.json"
    payload = _document()
    payload["schema_version"] = "flameox.kernel-validation.v2"
    source.write_text(json.dumps(payload))
    run_id = _import(workspace, source)

    with pytest.raises(DomainError) as error:
        KernelValidationExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_kernel_validation_does_not_allow_passing_without_metric_evidence() -> None:
    payload = _document()
    payload["cases"][0]["outputs"][0]["metrics"] = []

    with pytest.raises(ValidationError, match="without metrics"):
        KernelValidationV1.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["cases"][0].update(seed=2**63),
        lambda payload: payload["cases"][0]["inputs"]["left"].update(shape=[2**63]),
        lambda payload: payload["cases"][0]["outputs"][0].update(shape=[2**63]),
    ),
)
def test_kernel_validation_rejects_values_outside_evidence_int64(
    mutation: object,
) -> None:
    payload = _document()
    assert callable(mutation)
    mutation(payload)

    with pytest.raises(ValidationError):
        KernelValidationV1.model_validate(payload)
