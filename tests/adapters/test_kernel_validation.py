from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from flameox.adapters.kernel_validation import (
    FiniteKernelMetricValue,
    KernelValidationExtractor,
    KernelValidationV2,
    kernel_validation_json_schema,
)
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "kernel_validation" / "pass.json"


def _document() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(VALIDATION_FIXTURE.read_text(encoding="utf-8")))


def _import(workspace: Workspace, path: Path) -> str:
    return (
        ImportService(workspace)
        .import_artifact(ImportArtifactRequest(path=path, kind=ArtifactKind.VALIDATION_OUTPUT))
        .run.run_id
    )


def test_kernel_validation_round_trips_and_extracts_idempotently(tmp_path: Path) -> None:
    document = KernelValidationV2.model_validate(_document())
    assert KernelValidationV2.model_validate_json(document.model_dump_json()) == document
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
            PROJECT_ROOT / "src" / "flameox" / "schemas" / "kernel-validation-v2.schema.json"
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
            lambda value: value["cases"][0]["outputs"][0]["metrics"][0].update(
                value={"kind": "finite", "value": float("nan")}
            ),
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
        KernelValidationV2.model_validate(payload)


def test_kernel_validation_failed_output_requires_bounded_examples() -> None:
    payload = _document()
    metric = payload["cases"][0]["outputs"][0]["metrics"][0]
    metric.update(value={"kind": "finite", "value": 0.1}, status="fail")
    payload["cases"][0]["outputs"][0].update(status="fail")
    payload["cases"][0].update(status="fail")
    payload.update(status="fail")

    with pytest.raises(ValidationError, match="representative failure"):
        KernelValidationV2.model_validate(payload)

    payload["cases"][0]["outputs"][0]["representative_failures"] = [{}]
    with pytest.raises(ValidationError, match="substantive witness"):
        KernelValidationV2.model_validate(payload)

    payload["cases"][0]["outputs"][0]["representative_failures"] = [
        {"coordinates": [3, 7], "expected": 1.0, "actual": 1.1, "absolute_error": 0.1}
    ]
    assert KernelValidationV2.model_validate(payload).status == "fail"


@pytest.mark.parametrize("metric_status", ("inconclusive", "unsupported"))
def test_kernel_validation_surfaces_nonpassing_metric_limitation(
    tmp_path: Path,
    metric_status: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    payload = _document()
    metric = payload["cases"][0]["outputs"][0]["metrics"][0]
    metric.update(
        value=None,
        comparator=None,
        threshold=None,
        status=metric_status,
        limitation=f"Metric was {metric_status} for this device.",
    )
    payload["cases"][0]["outputs"][0].update(status="inconclusive")
    payload["cases"][0].update(status="inconclusive")
    payload.update(status="inconclusive")
    source = tmp_path / "inconclusive.json"
    source.write_text(json.dumps(payload))

    result = KernelValidationExtractor(workspace).extract(_import(workspace, source))

    assert result.status == "inconclusive"
    assert f"Metric was {metric_status} for this device." in result.limitations


@pytest.mark.parametrize(
    "updates",
    (
        {"status": "inconclusive", "limitation": "Unavailable."},
        {"value": None},
    ),
)
def test_kernel_validation_metric_status_determines_evidence_shape(
    updates: dict[str, object],
) -> None:
    payload = _document()
    payload["cases"][0]["outputs"][0]["metrics"][0].update(updates)

    with pytest.raises(ValidationError):
        KernelValidationV2.model_validate(payload)


@pytest.mark.parametrize(
    ("name", "value", "threshold", "match"),
    (
        ("max_abs_error", -0.1, 0.0, "must be nonnegative"),
        ("max_rel_error", -0.1, 0.0, "must be nonnegative"),
        ("mse", -1.0, 0.0, "must be nonnegative"),
        ("rmse", -1.0, 0.0, "must be nonnegative"),
        ("cosine_similarity", 1.1, 0.9, "between -1 and 1"),
        ("cosine_similarity", 0.9, -1.1, "between -1 and 1"),
    ),
)
def test_kernel_validation_rejects_impossible_metric_domains(
    name: str,
    value: float,
    threshold: float,
    match: str,
) -> None:
    payload = _document()
    metric = payload["cases"][0]["outputs"][0]["metrics"][0]
    metric.update(
        name=name,
        value={"kind": "finite", "value": value},
        threshold=threshold,
        comparator=">=" if name == "cosine_similarity" else "<=",
        status="pass",
    )

    with pytest.raises(ValidationError, match=match):
        KernelValidationV2.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["cases"][0].update(seed="42"),
        lambda payload: payload["cases"][0]["outputs"][0]["metrics"][0].update(value="0.0001"),
        lambda payload: payload.update(coverage_complete=1),
    ),
)
def test_kernel_validation_extractor_rejects_json_type_coercion(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    workspace = Workspace.initialize(tmp_path)
    payload = _document()
    mutation(payload)
    source = tmp_path / "coerced.json"
    source.write_text(json.dumps(payload))

    with pytest.raises(DomainError) as error:
        KernelValidationExtractor(workspace).extract(_import(workspace, source))

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_kernel_validation_result_bounds_nested_limitation_summary(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    payload = _document()
    template = payload["cases"][0]
    cases = []
    for index in range(101):
        case = deepcopy(template)
        case["case_id"] = f"case-{index}"
        case["limitations"] = [f"limitation-{index}"]
        cases.append(case)
    payload["cases"] = cases
    source = tmp_path / "many-limitations.json"
    source.write_text(json.dumps(payload))

    result = KernelValidationExtractor(workspace).extract(_import(workspace, source))

    assert len(result.limitations) == 100
    assert result.limitations[0] == "limitation-0"
    assert result.limitations[-1] == (
        "Additional nested limitations were omitted from this bounded summary."
    )


def test_kernel_validation_extractor_rejects_unknown_schema(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "validation.json"
    payload = _document()
    payload["schema_version"] = "flameox.kernel-validation.v3"
    source.write_text(json.dumps(payload))
    run_id = _import(workspace, source)

    with pytest.raises(DomainError) as error:
        KernelValidationExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_kernel_validation_does_not_allow_passing_without_metric_evidence() -> None:
    payload = _document()
    payload["cases"][0]["outputs"][0]["metrics"] = []

    with pytest.raises(ValidationError, match="without metrics"):
        KernelValidationV2.model_validate(payload)


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
        KernelValidationV2.model_validate(payload)


def _psnr_metric(
    *,
    value: dict[str, object],
    data_range: float = 1.0,
    threshold: float = 60.0,
) -> dict[str, object]:
    return {
        "name": "psnr",
        "value": value,
        "comparator": ">=",
        "threshold": threshold,
        "unit": "dB",
        "profile": {
            "identity_quality": "exact",
            "data_range": data_range,
            "log_base": 10,
            "reduction": "mean_squared_error",
            "zero_mse_convention": "positive_infinity",
        },
        "status": "pass",
    }


def test_exact_agreement_psnr_uses_tagged_positive_infinity(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    payload = _document()
    metrics = payload["cases"][0]["outputs"][0]["metrics"]
    metrics[:] = [
        {
            "name": "mse",
            "value": {"kind": "finite", "value": 0.0},
            "comparator": "<=",
            "threshold": 0.0,
            "unit": "squared_error",
            "status": "pass",
        },
        _psnr_metric(
            value={
                "kind": "positive_infinity",
                "reason": "zero_mse_exact_agreement",
            },
            threshold=1e308,
        ),
    ]
    document = KernelValidationV2.model_validate(payload)
    source = tmp_path / "exact-psnr.json"
    source.write_text(document.model_dump_json())

    result = KernelValidationExtractor(workspace).extract(_import(workspace, source))

    assert result.status == "pass"
    with Catalog(workspace).open_snapshot() as snapshot:
        row = snapshot.execute(
            "SELECT value, value_kind, positive_infinity_reason, metric_profile_json "
            "FROM kernel_validation_metrics WHERE metric_name = 'psnr'"
        ).fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] == "positive_infinity"
    assert row[2] == "zero_mse_exact_agreement"
    assert json.loads(row[3])["data_range"] == 1.0


def test_finite_psnr_remains_exactly_finite_for_nonzero_mse() -> None:
    payload = _document()
    payload["cases"][0]["outputs"][0]["metrics"] = [
        {
            "name": "mse",
            "value": {"kind": "finite", "value": 1e-12},
            "comparator": "<=",
            "threshold": 1e-9,
            "unit": "squared_error",
            "status": "pass",
        },
        _psnr_metric(value={"kind": "finite", "value": 120.0}),
    ]

    document = KernelValidationV2.model_validate(payload)

    psnr = document.cases[0].outputs[0].metrics[1]
    assert isinstance(psnr.value, FiniteKernelMetricValue)
    assert psnr.value.value == 120.0


@pytest.mark.parametrize("mse_value", (None, 1e-12))
def test_positive_infinity_psnr_requires_zero_mse_witness(
    mse_value: float | None,
) -> None:
    payload = _document()
    metrics: list[dict[str, object]] = []
    if mse_value is not None:
        metrics.append(
            {
                "name": "mse",
                "value": {"kind": "finite", "value": mse_value},
                "comparator": "<=",
                "threshold": 1e-9,
                "unit": "squared_error",
                "status": "pass",
            }
        )
    metrics.append(
        _psnr_metric(
            value={
                "kind": "positive_infinity",
                "reason": "zero_mse_exact_agreement",
            }
        )
    )
    payload["cases"][0]["outputs"][0]["metrics"] = metrics

    with pytest.raises(ValidationError, match="requires an evaluated zero-mse witness"):
        KernelValidationV2.model_validate(payload)


@pytest.mark.parametrize(
    ("name", "value", "profile", "match"),
    (
        (
            "mse",
            {"kind": "positive_infinity", "reason": "zero_mse_exact_agreement"},
            None,
            "only for exact-agreement psnr",
        ),
        (
            "psnr",
            {"kind": "finite", "value": float("inf")},
            {"identity_quality": "exact", "data_range": 1.0},
            "must be finite",
        ),
        (
            "psnr",
            {"kind": "finite", "value": float("nan")},
            {"identity_quality": "exact", "data_range": 1.0},
            "must be finite",
        ),
    ),
)
def test_kernel_validation_rejects_illegal_infinity_variants(
    name: str,
    value: dict[str, object],
    profile: dict[str, object] | None,
    match: str,
) -> None:
    payload = _document()
    metric = payload["cases"][0]["outputs"][0]["metrics"][0]
    metric.update(
        name=name,
        value=value,
        comparator=">=" if name == "psnr" else "<=",
        threshold=60.0 if name == "psnr" else 0.0,
        unit="dB" if name == "psnr" else "squared_error",
        profile=profile,
    )

    with pytest.raises(ValidationError, match=match):
        KernelValidationV2.model_validate(payload)


def test_psnr_profile_changes_metric_identity(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    identities: list[str] = []
    for data_range in (1.0, 255.0):
        payload = _document()
        payload["cases"][0]["outputs"][0]["metrics"] = [
            _psnr_metric(value={"kind": "finite", "value": 80.0}, data_range=data_range)
        ]
        source = tmp_path / f"psnr-{data_range}.json"
        source.write_text(json.dumps(payload))
        run_id = _import(workspace, source)
        KernelValidationExtractor(workspace).extract(run_id)
        with Catalog(workspace).open_snapshot() as snapshot:
            row = snapshot.execute(
                "SELECT metric_identity_id FROM kernel_validation_metrics "
                "WHERE run_id = ? AND metric_name = 'psnr'",
                (run_id,),
            ).fetchone()
        assert row is not None
        identities.append(row[0])

    assert len(set(identities)) == 2


def test_v1_kernel_validation_is_rejected(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    payload = _document()
    payload["schema_version"] = "flameox.kernel-validation.v1"
    payload["cases"][0]["outputs"][0]["metrics"] = [
        {
            "name": "psnr",
            "value": 80.0,
            "comparator": ">=",
            "threshold": 60.0,
            "unit": "dB",
            "status": "pass",
        }
    ]
    source = tmp_path / "legacy-psnr.json"
    source.write_text(json.dumps(payload))

    with pytest.raises(DomainError) as error:
        KernelValidationExtractor(workspace).extract(_import(workspace, source))

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
