from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, StringConstraints, model_validator

from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel

MemrayRegionName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
    ),
]


class MemrayCaptureOptions(ContractModel):
    """The supported Memray capture scopes.

    ``sdk`` relies on one explicit workload-owned context. Memray tracks every
    thread in the process while its tracker is active, so it is an exact time
    region rather than a single-thread filter.
    """

    mode: Literal["whole_entrypoint", "sdk"] = "whole_entrypoint"
    region: MemrayRegionName | None = None
    warmup_count: Annotated[int, Field(strict=True, ge=0, le=1_000_000)] = 0
    native_traces: StrictBool = False
    trace_python_allocators: StrictBool = False

    @model_validator(mode="after")
    def region_matches_capture_mode(self) -> MemrayCaptureOptions:
        if (self.mode == "sdk") != (self.region is not None):
            raise ValueError("region is required exactly for sdk Memray capture")
        if self.mode == "whole_entrypoint" and self.warmup_count:
            raise ValueError("warmup_count is valid only for sdk Memray capture")
        return self


def memray_capture_options(value: dict[str, object] | None) -> MemrayCaptureOptions:
    try:
        return MemrayCaptureOptions.model_validate(value or {})
    except ValueError as exc:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "Invalid Memray capture options.",
            details={"validation_error": str(exc)},
            remediation=(
                "Use mode='whole_entrypoint' without a region, or mode='sdk' with one "
                "declared region and one flameox.sdk.memray_region() context.",
            ),
        ) from exc
