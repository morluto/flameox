from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.runtime_contracts import CaptureTarget


@pytest.mark.unit
def test_capture_target_rejects_policy_blocked_environment_override() -> None:
    with pytest.raises(ValidationError, match=r"PYTHONPATH.*blocked by policy"):
        CaptureTarget(
            argv=["python"],
            cwd=str(Path.cwd()),
            provider_id="direct",
            environment={"PYTHONPATH": "unsafe"},
        )


@pytest.mark.unit
def test_capture_target_requires_an_absolute_working_directory() -> None:
    with pytest.raises(ValidationError, match="cwd must be an absolute path"):
        CaptureTarget(argv=["python"], cwd=".", provider_id="direct")
