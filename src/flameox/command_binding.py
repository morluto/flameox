from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path

from flameox.canonical import sha256_id
from flameox.executable_models import (
    ExecutableIdentity,
    ExecutablePolicyDecision,
    ExecutableResolutionOrigin,
    ExecutableResolutionRequest,
    ExecutableTrustPolicy,
    ResolvedExecutable,
)
from flameox.runtime_errors import DomainError, ErrorCode


class ExecutableResolver:
    """Resolve, authorize, and identify the exact executable selected for a request."""

    def resolve(self, request: ExecutableResolutionRequest) -> ResolvedExecutable:
        cwd = request.cwd.resolve(strict=True)
        if not cwd.is_dir():
            raise self._refused(request.token, "Executable resolution cwd is not a directory.")

        path_like = _is_path_like(request.token)
        if (
            request.policy is ExecutableTrustPolicy.EXACT_PATH
            and not Path(request.token).is_absolute()
        ):
            raise self._refused(
                request.token,
                "Exact-path executable policy requires an absolute path.",
            )

        matched_path_entry: Path | None = None
        if path_like:
            candidate = Path(request.token)
            invocation_path = candidate if candidate.is_absolute() else cwd / candidate
            invocation_path = invocation_path.absolute()
            origin = ExecutableResolutionOrigin.EXPLICIT_PATH
        else:
            search_path, entries = _normalized_search_path(request.environment.get("PATH"), cwd)
            located = shutil.which(request.token, path=search_path)
            if located is None:
                raise DomainError(
                    ErrorCode.UNAVAILABLE_CAPABILITY,
                    f"Executable {request.token!r} was not found in the request PATH.",
                    details={"executable": request.token},
                )
            invocation_path = Path(located).absolute()
            matched_path_entry = next(
                (entry for entry in entries if invocation_path.parent == entry),
                invocation_path.parent,
            )
            origin = ExecutableResolutionOrigin.PATH_SEARCH

        try:
            canonical_target = invocation_path.resolve(strict=True)
            target_stat = canonical_target.stat()
        except OSError as exc:
            raise self._refused(
                request.token,
                "Executable target is missing or cannot be resolved.",
            ) from exc
        if not canonical_target.is_file() or not os.access(invocation_path, os.X_OK):
            raise self._refused(request.token, "Executable target is not a runnable regular file.")

        decision = self._authorize(request, canonical_target)
        identity = _identity(canonical_target, target_stat)
        return ResolvedExecutable(
            requested_token=request.token,
            invocation_path=invocation_path,
            canonical_target=canonical_target,
            origin=origin,
            matched_path_entry=matched_path_entry,
            identity=identity,
            policy_decision=decision,
        )

    def resolve_host_tool(
        self,
        token: str,
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> ResolvedExecutable | None:
        """Resolve an optional host tool through the same authoritative binding path."""
        working_directory = (cwd or Path.cwd()).resolve()
        selected_environment = dict(os.environ if environment is None else environment)
        try:
            return self.resolve(
                ExecutableResolutionRequest(
                    token=token,
                    cwd=working_directory,
                    environment=selected_environment,
                    policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
                )
            )
        except DomainError as error:
            if error.code is ErrorCode.UNAVAILABLE_CAPABILITY:
                return None
            raise

    def require_host_tool(
        self,
        token: str,
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> ResolvedExecutable:
        """Resolve a required host tool or raise the resolver's typed unavailable error."""

        binding = self.resolve_host_tool(token, cwd=cwd, environment=environment)
        if binding is None:
            raise DomainError(
                ErrorCode.UNAVAILABLE_CAPABILITY,
                f"Executable {token!r} was not found in the request PATH.",
                details={"executable": token},
            )
        return binding

    def revalidate(self, resolved: ResolvedExecutable) -> ResolvedExecutable:
        """Recheck that a previously bound invocation still selects the same file bytes."""

        try:
            canonical_target = resolved.invocation_path.resolve(strict=True)
            target_stat = canonical_target.stat()
        except OSError as exc:
            raise self._changed(resolved, "The bound executable is no longer available.") from exc
        if canonical_target != resolved.canonical_target:
            raise self._changed(
                resolved,
                "The bound executable now resolves to a different target.",
            )
        if _identity(canonical_target, target_stat) != resolved.identity:
            raise self._changed(resolved, "The bound executable changed after planning.")
        return resolved

    def _authorize(
        self,
        request: ExecutableResolutionRequest,
        canonical_target: Path,
    ) -> ExecutablePolicyDecision:
        if request.policy is ExecutableTrustPolicy.TRUSTED_HOST_TOOL:
            return ExecutablePolicyDecision(policy=request.policy, allowed=True)

        roots = tuple(root.resolve(strict=True) for root in request.allowed_roots)
        for root in roots:
            try:
                canonical_target.relative_to(root)
            except ValueError:
                continue
            return ExecutablePolicyDecision(
                policy=request.policy,
                allowed=True,
                matched_root=root,
            )
        raise self._refused(
            request.token,
            f"Executable target is not authorized by {request.policy.value!r} policy.",
        )

    @staticmethod
    def _refused(token: str, message: str) -> DomainError:
        return DomainError(
            ErrorCode.EXECUTION_FAILURE,
            message,
            details={"executable": token},
        )

    @staticmethod
    def _changed(resolved: ResolvedExecutable, message: str) -> DomainError:
        return DomainError(
            ErrorCode.MISSING_OR_CHANGED_INPUT,
            message,
            details={"executable": resolved.requested_token},
        )


def _is_path_like(value: str) -> bool:
    return "/" in value or "\\" in value or Path(value).is_absolute()


def _normalized_search_path(value: str | None, cwd: Path) -> tuple[str, tuple[Path, ...]]:
    if value is None:
        raise DomainError(
            ErrorCode.UNAVAILABLE_CAPABILITY,
            "A bare executable name requires PATH in the request environment.",
        )
    entries = tuple(
        (cwd if not item else (Path(item) if Path(item).is_absolute() else cwd / item)).absolute()
        for item in value.split(os.pathsep)
    )
    return os.pathsep.join(str(item) for item in entries), entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return sha256_id(digest.hexdigest())


def _identity(path: Path, path_stat: os.stat_result | None = None) -> ExecutableIdentity:
    target_stat = path_stat or path.stat()
    return ExecutableIdentity(
        sha256=_sha256(path),
        size=target_stat.st_size,
        mode=stat.S_IMODE(target_stat.st_mode),
        device=target_stat.st_dev,
        inode=target_stat.st_ino,
        modified_ns=target_stat.st_mtime_ns,
    )
