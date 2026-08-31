from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from flameox.command_binding import ExecutableResolver
from flameox.executable_models import (
    ExecutableResolutionRequest,
    ExecutableTrustPolicy,
)
from flameox.execution import ExecutionRequest, SubprocessBroker

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


@pytest.mark.anyio
async def test_broker_executes_the_bound_executable_without_repeating_path_search(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "bin" / ("tool.exe" if os.name == "nt" else "tool")
    executable.parent.mkdir()
    try:
        executable.symlink_to(sys.executable)
    except OSError:
        pytest.skip("creating an executable symlink is unavailable on this host")
    binding = ExecutableResolver().resolve(
        ExecutableResolutionRequest(
            token="tool",
            cwd=tmp_path,
            environment={"PATH": "bin"},
            policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
        )
    )

    outcome = await SubprocessBroker().run(
        ExecutionRequest(
            argv=("tool", "-c", "print('bound executable')"),
            executable_binding=binding,
            cwd=tmp_path,
            environment_allowlist=(),
            environment_overrides={"PATH": ""},
            allowed_working_roots=(tmp_path,),
        )
    )

    assert outcome.stdout == b"bound executable\n"
    assert outcome.executable_binding == binding
