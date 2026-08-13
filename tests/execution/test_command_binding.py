from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from flameox.command_binding import ExecutableResolver
from flameox.domain import DomainError, ErrorCode
from flameox.domain.executables import (
    ExecutableResolutionRequest,
    ExecutableTrustPolicy,
)

pytestmark = pytest.mark.unit


def test_path_search_is_bound_to_the_request_cwd_and_environment(tmp_path: Path) -> None:
    executable_name = "tool.exe" if os.name == "nt" else "tool"
    executable = tmp_path / "bin" / executable_name
    executable.parent.mkdir()
    shutil.copy2(sys.executable, executable)
    executable.chmod(0o755)

    resolved = ExecutableResolver().resolve(
        ExecutableResolutionRequest(
            token="tool",
            cwd=tmp_path,
            environment={"PATH": "bin"},
            policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
        )
    )

    assert resolved.requested_token == "tool"
    assert resolved.invocation_path == executable.absolute()
    assert resolved.canonical_target == executable.resolve()
    assert resolved.matched_path_entry == executable.parent.absolute()
    assert resolved.identity.sha256.startswith("sha256:")
    assert resolved.policy_decision.allowed is True


def test_revalidation_rejects_an_executable_changed_after_binding(tmp_path: Path) -> None:
    executable = tmp_path / ("tool.exe" if os.name == "nt" else "tool")
    shutil.copy2(sys.executable, executable)
    executable.chmod(0o755)
    resolver = ExecutableResolver()
    resolved = resolver.resolve(
        ExecutableResolutionRequest(
            token=str(executable),
            cwd=tmp_path,
            environment={},
            policy=ExecutableTrustPolicy.PROJECT_BOUND,
            allowed_roots=(tmp_path,),
        )
    )

    with executable.open("ab") as stream:
        stream.write(b"changed after planning")

    with pytest.raises(DomainError) as caught:
        resolver.revalidate(resolved)

    assert caught.value.code is ErrorCode.INVALID_CAPTURE_PLAN
