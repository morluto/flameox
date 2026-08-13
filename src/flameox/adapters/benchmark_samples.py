from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from flameox.domain import ArtifactKind, DomainError, ErrorCode, digest_model
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace

MetricName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    ),
]
DimensionValue = Annotated[str, StringConstraints(max_length=200)]
SampleValue = StrictInt | StrictFloat


class AcceleratorDevice(ContractModel):
    type: Literal["cuda", "hip", "xpu", "mps", "other"]
    index: Annotated[int, Field(ge=0)] | None = None
    stream: Annotated[str, StringConstraints(max_length=200)] | None = None


class BenchmarkSeries(ContractModel):
    name: MetricName
    unit: Literal["ns", "bytes", "count", "ratio"]
    measurement_clock: Literal[
        "host_monotonic",
        "cuda_event",
        "hip_event",
        "device_event",
        "unknown",
    ]
    synchronization: Literal[
        "not_required",
        "device_synchronize",
        "event_synchronize",
        "stream_synchronize",
        "none",
        "unknown",
    ]
    scope: Literal["process", "thread", "operator", "device", "workload"] = "workload"
    phase: Annotated[str, StringConstraints(max_length=100)] | None = "steady_state"
    loop_count: Annotated[int, Field(gt=0)] | None = None
    worker_id: Annotated[str, StringConstraints(max_length=200)] | None = None
    worker_run_index: Annotated[int, Field(ge=0)] | None = None
    trial_id: Annotated[str, StringConstraints(max_length=200)] | None = None
    block_id: Annotated[str, StringConstraints(max_length=200)] | None = None
    variant_id: Annotated[str, StringConstraints(max_length=200)] | None = None
    order_in_block: Annotated[int, Field(ge=0)] | None = None
    device: AcceleratorDevice | None = None
    dimensions: dict[MetricName, DimensionValue] = Field(default_factory=dict, max_length=32)
    warmups: Annotated[tuple[SampleValue, ...], Field(max_length=1_000_000)] = ()
    samples: Annotated[
        tuple[SampleValue, ...],
        Field(min_length=1, max_length=1_000_000),
    ]

    @field_validator("warmups", "samples")
    @classmethod
    def finite_values(cls, values: tuple[SampleValue, ...]) -> tuple[SampleValue, ...]:
        if any(isinstance(value, float) and not math.isfinite(value) for value in values):
            raise ValueError("benchmark samples must be finite")
        return values

    @model_validator(mode="after")
    def values_match_unit(self) -> BenchmarkSeries:
        values = (*self.warmups, *self.samples)
        if self.unit in {"ns", "bytes", "count"} and any(
            not isinstance(value, int) for value in values
        ):
            raise ValueError(f"{self.unit} samples require exact integer values")
        reserved_dimensions = {
            "measurement_clock",
            "producer",
            "producer_version",
            "synchronization",
        }
        conflicts = sorted(
            key
            for key in self.dimensions
            if key in reserved_dimensions or key.startswith("device.")
        )
        if conflicts:
            raise ValueError("benchmark dimensions use reserved keys: " + ", ".join(conflicts))
        return self


class BenchmarkSamplesV1(ContractModel):
    schema_version: Literal["flameox.benchmark-samples.v1"]
    producer: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    producer_version: Annotated[str, StringConstraints(max_length=200)] | None = None
    benchmarks: Annotated[tuple[BenchmarkSeries, ...], Field(min_length=1, max_length=1_000)]


class BenchmarkSamplesExtractionResult(ContractModel):
    schema_version: int = 1
    run_id: str
    artifact_id: str
    producer: str
    producer_version: str | None
    benchmark_names: tuple[str, ...]
    measurement_count: int
    warmup_count: int
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class BenchmarkSamplesExtractor:
    name = "flameox.benchmark-samples"
    version = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    def extract(self, run_id: str) -> BenchmarkSamplesExtractionResult:
        run = self.runs.read(run_id)
        registrations = tuple(
            item for item in run.artifacts if item.kind is ArtifactKind.BENCHMARK_SAMPLES
        )
        if len(registrations) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The run must contain exactly one structured benchmark-samples artifact.",
                run_id=run_id,
            )
        registration = registrations[0]
        stored = self.artifacts.get(registration.artifact_id)
        payload = self._load(stored.payload_path)
        self._check_registration_identity(
            registration.producer,
            registration.producer_version,
            payload,
        )

        total_rows = sum(len(item.warmups) + len(item.samples) for item in payload.benchmarks)
        if total_rows > self.workspace.config.storage.max_rows_per_generation:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Structured benchmark samples exceed the workspace generation row limit.",
                details={
                    "rows": total_rows,
                    "max_rows": self.workspace.config.storage.max_rows_per_generation,
                },
            )

        rows: list[dict[str, object]] = []
        limitations: list[str] = []
        producer_version = payload.producer_version or registration.producer_version
        if producer_version is None:
            limitations.append("The benchmark producer version was not declared.")
        for series_index, series in enumerate(payload.benchmarks):
            if series.measurement_clock in {
                "cuda_event",
                "hip_event",
                "device_event",
            } and series.synchronization in {"none", "unknown"}:
                limitations.append(
                    f"{series.name} uses asynchronous device timing with "
                    f"synchronization={series.synchronization}."
                )
            for is_warmup, values in ((True, series.warmups), (False, series.samples)):
                for value_index, value in enumerate(values):
                    rows.append(
                        self._measurement_row(
                            run_id=run_id,
                            artifact_id=registration.artifact_id,
                            producer=payload.producer,
                            producer_version=producer_version,
                            series=series,
                            series_index=series_index,
                            value_index=value_index,
                            value=value,
                            is_warmup=is_warmup,
                        )
                    )

        published = self.publisher.publish_rows_idempotent(
            {"measurements": rows},
            publisher=self.name,
            publisher_version=self.version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={
                "schema_version": payload.schema_version,
                "producer": payload.producer,
                "producer_version": producer_version,
            },
        )
        return BenchmarkSamplesExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            producer=payload.producer,
            producer_version=producer_version,
            benchmark_names=tuple(item.name for item in payload.benchmarks),
            measurement_count=sum(len(item.samples) for item in payload.benchmarks),
            warmup_count=sum(len(item.warmups) for item in payload.benchmarks),
            corpus_commit_id=published.commit.commit_id,
            limitations=tuple(dict.fromkeys(limitations)),
        )

    @staticmethod
    def _load(path: Path) -> BenchmarkSamplesV1:
        try:
            with path.open(encoding="utf-8") as stream:
                return BenchmarkSamplesV1.model_validate(json.load(stream))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact is not a valid flameox benchmark-samples v1 document.",
            ) from exc

    @staticmethod
    def _check_registration_identity(
        registration_producer: str | None,
        registration_version: str | None,
        payload: BenchmarkSamplesV1,
    ) -> None:
        transport_producers = {None, "flameox.import", "benchmark-samples"}
        if registration_producer not in {*transport_producers, payload.producer}:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Benchmark producer identity conflicts with the artifact registration.",
                details={
                    "registered_producer": registration_producer,
                    "document_producer": payload.producer,
                },
            )
        if (
            registration_producer not in transport_producers
            and registration_version is not None
            and payload.producer_version is not None
            and registration_version != payload.producer_version
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Benchmark producer version conflicts with the artifact registration.",
                details={
                    "registered_version": registration_version,
                    "document_version": payload.producer_version,
                },
            )

    @staticmethod
    def _measurement_row(
        *,
        run_id: str,
        artifact_id: str,
        producer: str,
        producer_version: str | None,
        series: BenchmarkSeries,
        series_index: int,
        value_index: int,
        value: SampleValue,
        is_warmup: bool,
    ) -> dict[str, object]:
        dimensions = dict(series.dimensions)
        dimensions.update(
            {
                "measurement_clock": series.measurement_clock,
                "synchronization": series.synchronization,
                "producer": producer,
            }
        )
        if producer_version is not None:
            dimensions["producer_version"] = producer_version
        if series.device is not None:
            dimensions["device.type"] = series.device.type
            if series.device.index is not None:
                dimensions["device.index"] = str(series.device.index)
            if series.device.stream is not None:
                dimensions["device.stream"] = series.device.stream
        identity = {
            "run_id": run_id,
            "artifact_id": artifact_id,
            "series_index": series_index,
            "name": series.name,
            "is_warmup": is_warmup,
            "value_index": value_index,
        }
        integer_unit = series.unit in {"ns", "bytes", "count"}
        return {
            "measurement_id": digest_model(identity),
            "run_id": run_id,
            "artifact_id": artifact_id,
            "name": series.name,
            "value_int": int(value) if integer_unit else None,
            "value_float": float(value) if not integer_unit else None,
            "unit": series.unit,
            "aggregation": "sample",
            "scope": series.scope,
            "trial_id": series.trial_id,
            "worker_id": series.worker_id,
            "worker_run_index": series.worker_run_index,
            "value_index": value_index,
            "loop_count": series.loop_count,
            "is_warmup": is_warmup,
            "block_id": series.block_id,
            "variant_id": series.variant_id,
            "order_in_block": series.order_in_block,
            "phase": "warmup" if is_warmup else series.phase,
            "dimensions": dimensions,
            "evidence_level": "observed",
        }
