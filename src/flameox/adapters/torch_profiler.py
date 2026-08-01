from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, StrictInt, ValidationError, model_validator

from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel


class TorchProfilerSchedule(ContractModel):
    wait: StrictInt = Field(default=1, ge=0, le=10_000)
    warmup: StrictInt = Field(default=1, ge=0, le=10_000)
    active: StrictInt = Field(default=1, ge=1, le=10_000)
    repeat: StrictInt = Field(default=1, ge=1, le=100)
    skip_first: StrictInt = Field(default=0, ge=0, le=10_000)


class TorchProfilerCaptureOptions(ContractModel):
    mode: Literal["whole_entrypoint", "sdk"] = "whole_entrypoint"
    activities: tuple[Literal["cpu", "cuda", "cuda_if_available"], ...] = (
        "cpu",
        "cuda_if_available",
    )
    record_shapes: StrictBool = True
    profile_memory: StrictBool = True
    with_stack: StrictBool = True
    with_flops: StrictBool = False
    with_modules: StrictBool = False
    schedule: TorchProfilerSchedule | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> TorchProfilerCaptureOptions:
        if not self.activities:
            raise ValueError("at least one torch.profiler activity is required")
        if len(set(self.activities)) != len(self.activities):
            raise ValueError("torch.profiler activities must be unique")
        if {"cuda", "cuda_if_available"}.issubset(self.activities):
            raise ValueError("choose either cuda or cuda_if_available, not both")
        if self.mode == "whole_entrypoint" and self.schedule is not None:
            raise ValueError(
                "a torch.profiler schedule requires SDK mode and explicit profile.step() calls"
            )
        if self.mode == "sdk" and self.schedule is None:
            raise ValueError("SDK mode requires an explicit bounded torch.profiler schedule")
        return self

    @property
    def expected_cycles(self) -> int:
        return self.schedule.repeat if self.schedule is not None else 1

    @property
    def output_filenames(self) -> tuple[str, ...]:
        if self.mode == "whole_entrypoint":
            return ("torch-trace.json",)
        return tuple(f"torch-trace-cycle-{cycle:04d}.json" for cycle in range(self.expected_cycles))


def torch_profiler_options(
    value: dict[str, object] | None,
) -> TorchProfilerCaptureOptions:
    try:
        return TorchProfilerCaptureOptions.model_validate(value or {})
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
