from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from flameox.models import ContractModel

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


class BenchmarkDevice(ContractModel):
    type: Literal["cuda", "hip", "xpu", "mps", "other"]
    index: Annotated[int, Field(ge=0)] | None = None
    stream: Annotated[str, StringConstraints(max_length=200)] | None = None


class BenchmarkSeries(ContractModel):
    name: MetricName
    unit: Literal["ns", "bytes", "count", "ratio"]
    measurement_clock: Literal[
        "host_monotonic", "cuda_event", "hip_event", "device_event", "unknown"
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
    device: BenchmarkDevice | None = None
    dimensions: dict[MetricName, DimensionValue] = Field(default_factory=dict, max_length=32)
    warmups: Annotated[tuple[SampleValue, ...], Field(max_length=1_000_000)] = ()
    samples: Annotated[tuple[SampleValue, ...], Field(min_length=1, max_length=1_000_000)]

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
        reserved = {
            "measurement_clock",
            "producer",
            "producer_version",
            "synchronization",
        }
        if conflicts := sorted(
            key for key in self.dimensions if key in reserved or key.startswith("device.")
        ):
            raise ValueError("benchmark dimensions use reserved keys: " + ", ".join(conflicts))
        return self


class BenchmarkSamplesV1(ContractModel):
    schema_version: Literal["flameox.benchmark-samples.v1"]
    producer: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    producer_version: Annotated[str, StringConstraints(max_length=200)] | None = None
    benchmarks: Annotated[tuple[BenchmarkSeries, ...], Field(min_length=1, max_length=1_000)]

    @model_validator(mode="after")
    def metric_names_do_not_mix_timing_semantics(self) -> BenchmarkSamplesV1:
        semantics_by_name: dict[str, tuple[str, str, str]] = {}
        for series in self.benchmarks:
            semantics = (series.unit, series.measurement_clock, series.synchronization)
            prior = semantics_by_name.setdefault(series.name, semantics)
            if prior != semantics:
                raise ValueError(
                    "a benchmark metric name cannot mix timing clocks or synchronization"
                )
        return self
