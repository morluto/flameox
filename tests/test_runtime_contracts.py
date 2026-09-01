from __future__ import annotations

import pytest
from pydantic import ValidationError

from flameox.runtime_contracts import CaptureTarget


@pytest.mark.unit
def test_capture_target_rejects_policy_blocked_environment_override() -> None:
    with pytest.raises(ValidationError, match=r"PYTHONPATH.*blocked by policy"):
        CaptureTarget(argv=["python"], environment={"PYTHONPATH": "unsafe"})
