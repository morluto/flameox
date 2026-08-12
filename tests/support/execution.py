from pathlib import Path

from flameox.command_binding import ExecutableResolver
from flameox.domain.executables import (
    ExecutableIdentity,
    ExecutablePolicyDecision,
    ExecutableResolutionOrigin,
    ExecutableTrustPolicy,
    ResolvedExecutable,
)


def executable_binding(path: str | Path) -> ResolvedExecutable:
    """Build an inert binding for tests whose broker never starts a process."""

    executable = Path(path).absolute()
    if executable.is_file():
        return ExecutableResolver().require_host_tool(str(executable), cwd=executable.parent)
    return ResolvedExecutable(
        requested_token=str(path),
        invocation_path=executable,
        canonical_target=executable,
        origin=ExecutableResolutionOrigin.EXPLICIT_PATH,
        identity=ExecutableIdentity(
            sha256="sha256:" + "0" * 64,
            size=0,
            mode=0o755,
            device=0,
            inode=0,
            modified_ns=0,
        ),
        policy_decision=ExecutablePolicyDecision(
            policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
            allowed=True,
        ),
    )
