from __future__ import annotations

from typing import Annotated, cast

from pydantic import Field, StrictBool, StrictFloat, StrictInt, ValidationError

from flameox.models import ContractModel
from flameox.runtime_errors import DomainError, ErrorCode


class TorchBenchmarkOptions(ContractModel):
    """Bounded controls owned by the Torch-aware operation benchmark adapter."""

    min_run_time_seconds: Annotated[StrictFloat, Field(gt=0, le=60)] = 0.2
    max_samples: Annotated[StrictInt, Field(ge=1, le=1_000)] = 100
    num_threads: Annotated[StrictInt, Field(ge=1, le=256)] = 1
    cuda_event_timing: StrictBool = False


def torch_benchmark_options(value: dict[str, object] | None) -> TorchBenchmarkOptions:
    try:
        return TorchBenchmarkOptions.model_validate(value or {})
    except ValidationError as exc:
        raise DomainError(
            ErrorCode.INVALID_INPUT,
            "Invalid torch.benchmark capture options.",
            details={"validation_errors": exc.errors(include_url=False)},
            remediation=(
                "Use bounded min_run_time_seconds, max_samples, and num_threads values; "
                "enable cuda_event_timing only when a separate device-time metric is needed.",
            ),
        ) from exc


def parse_torch_benchmark_options(value: object) -> TorchBenchmarkOptions:
    """Validate request-bound SDK configuration without compatibility shapes."""

    if not isinstance(value, dict):
        raise ValueError("Flameox torch.benchmark configuration must be an object")
    return TorchBenchmarkOptions.model_validate(cast(dict[str, object], value))
