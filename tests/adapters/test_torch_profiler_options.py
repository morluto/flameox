from __future__ import annotations

import pytest
from pydantic import ValidationError

from flameox.adapters import (
    SdkTorchProfilerOptions,
    TorchProfilerSchedule,
    WholeEntrypointTorchProfilerOptions,
)
from flameox.adapters.torch_profiler import torch_profiler_options


def test_torch_profiler_parser_routes_mode_to_schedule_variant() -> None:
    whole = torch_profiler_options({})
    sdk = torch_profiler_options(
        {
            "mode": "sdk",
            "schedule": {"wait": 1, "warmup": 1, "active": 2, "repeat": 3},
        }
    )

    assert isinstance(whole, WholeEntrypointTorchProfilerOptions)
    assert whole.expected_cycles == 1
    assert whole.output_filenames == ("torch-trace.json",)
    assert whole.model_dump(mode="json")["schedule"] is None
    assert isinstance(sdk, SdkTorchProfilerOptions)
    assert sdk.expected_cycles == 3
    assert sdk.output_filenames[-1] == "torch-trace-cycle-0002.json"


def test_torch_profiler_variants_cannot_cross_schedule_modes() -> None:
    with pytest.raises(ValidationError, match="schedule"):
        WholeEntrypointTorchProfilerOptions.model_validate(
            {"schedule": TorchProfilerSchedule().model_dump(mode="json")}
        )
    with pytest.raises(ValidationError, match="schedule"):
        SdkTorchProfilerOptions.model_validate({"mode": "sdk"})
