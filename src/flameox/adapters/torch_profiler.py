from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Discriminator,
    Field,
    StrictBool,
    StrictInt,
    Tag,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel


class TorchProfilerSchedule(ContractModel):
    wait: StrictInt = Field(default=1, ge=0, le=10_000)
    warmup: StrictInt = Field(default=1, ge=0, le=10_000)
    active: StrictInt = Field(default=1, ge=1, le=10_000)
    repeat: StrictInt = Field(default=1, ge=1, le=100)
    skip_first: StrictInt = Field(default=0, ge=0, le=10_000)


class _TorchProfilerCaptureOptions(ContractModel):
    activities: tuple[Literal["cpu", "cuda", "cuda_if_available"], ...] = (
        "cpu",
        "cuda_if_available",
    )
    record_shapes: StrictBool = True
    profile_memory: StrictBool = True
    with_stack: StrictBool = True
    with_flops: StrictBool = False
    with_modules: StrictBool = False

    @model_validator(mode="after")
    def unique_activities(self) -> Self:
        if not self.activities:
            raise ValueError("at least one torch.profiler activity is required")
        if len(set(self.activities)) != len(self.activities):
            raise ValueError("torch.profiler activities must be unique")
        if {"cuda", "cuda_if_available"}.issubset(self.activities):
            raise ValueError("choose either cuda or cuda_if_available, not both")
        return self


class WholeEntrypointTorchProfilerOptions(_TorchProfilerCaptureOptions):
    mode: Literal["whole_entrypoint"] = "whole_entrypoint"
    schedule: Literal[None] = None

    @property
    def expected_cycles(self) -> int:
        return 1

    @property
    def output_filenames(self) -> tuple[str, ...]:
        return ("torch-trace.json",)


class SdkTorchProfilerOptions(_TorchProfilerCaptureOptions):
    mode: Literal["sdk"] = "sdk"
    schedule: TorchProfilerSchedule

    @property
    def expected_cycles(self) -> int:
        return self.schedule.repeat

    @property
    def output_filenames(self) -> tuple[str, ...]:
        return tuple(f"torch-trace-cycle-{cycle:04d}.json" for cycle in range(self.expected_cycles))


def _torch_profiler_variant(value: Any) -> Literal["whole_entrypoint", "sdk"]:
    if isinstance(value, SdkTorchProfilerOptions):
        return "sdk"
    if isinstance(value, Mapping) and value.get("mode") == "sdk":
        return "sdk"
    return "whole_entrypoint"


type TorchProfilerCaptureOptions = Annotated[
    Annotated[WholeEntrypointTorchProfilerOptions, Tag("whole_entrypoint")]
    | Annotated[SdkTorchProfilerOptions, Tag("sdk")],
    Discriminator(_torch_profiler_variant),
]

_TORCH_PROFILER_OPTIONS: TypeAdapter[TorchProfilerCaptureOptions] = TypeAdapter(
    TorchProfilerCaptureOptions
)


def torch_profiler_options(
    value: dict[str, object] | None,
) -> TorchProfilerCaptureOptions:
    try:
        return _TORCH_PROFILER_OPTIONS.validate_python(value or {})
    except ValidationError as exc:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "Invalid torch.profiler capture options.",
            details={"validation_errors": exc.errors(include_url=False)},
            remediation=(
                "Use mode='whole_entrypoint' without a schedule, or mode='sdk' with a "
                "bounded wait/warmup/active/repeat schedule and explicit profile.step() calls.",
            ),
        ) from exc
