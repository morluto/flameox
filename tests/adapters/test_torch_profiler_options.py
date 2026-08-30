from __future__ import annotations

import pytest
from pydantic import ValidationError

from flameox.adapters.torch_profiler import (
    SdkTorchProfilerOptions,
    TorchProfilerSchedule,
    WholeEntrypointTorchProfilerOptions,
    torch_profiler_options,
    torch_profiler_trace_filenames,
)

pytestmark = pytest.mark.unit


def test_torch_profiler_parser_routes_mode_to_schedule_variant() -> None:
    whole = torch_profiler_options({})
    sdk = torch_profiler_options(
        {
            "mode": "sdk",
            "schedule": {"wait": 1, "warmup": 1, "active": 2, "repeat": 3},
        }
    )

    assert isinstance(whole, WholeEntrypointTorchProfilerOptions)
    assert whole.record_shapes is False
    assert whole.profile_memory is False
    assert whole.with_stack is False
    assert whole.with_flops is False
    assert whole.with_modules is False
    assert torch_profiler_trace_filenames(whole) == ("torch-trace.json",)
    assert whole.model_dump(mode="json")["schedule"] is None
    assert isinstance(sdk, SdkTorchProfilerOptions)
    assert torch_profiler_trace_filenames(sdk)[-1] == "torch-trace-cycle-0002.json"


def test_torch_profiler_variants_cannot_cross_schedule_modes() -> None:
    with pytest.raises(ValidationError, match="schedule"):
        WholeEntrypointTorchProfilerOptions.model_validate(
            {"schedule": TorchProfilerSchedule().model_dump(mode="json")}
        )
    with pytest.raises(ValidationError, match="schedule"):
        SdkTorchProfilerOptions.model_validate({"mode": "sdk"})
