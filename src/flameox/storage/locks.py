from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import portalocker

from flameox.domain import DomainError, ErrorCode


class WorkspaceLockResource(StrEnum):
    WRITE = "write"
    RETENTION = "retention"
    CATALOG = "catalog"


class WorkspaceLockMode(StrEnum):
    SHARED = "shared"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True, slots=True)
class WorkspaceLockIntent:
    resource: WorkspaceLockResource
    mode: WorkspaceLockMode


WRITE_EXCLUSIVE = WorkspaceLockIntent(
    WorkspaceLockResource.WRITE,
    WorkspaceLockMode.EXCLUSIVE,
)
RETENTION_SHARED = WorkspaceLockIntent(
    WorkspaceLockResource.RETENTION,
    WorkspaceLockMode.SHARED,
)
RETENTION_EXCLUSIVE = WorkspaceLockIntent(
    WorkspaceLockResource.RETENTION,
    WorkspaceLockMode.EXCLUSIVE,
)
CATALOG_SHARED = WorkspaceLockIntent(
    WorkspaceLockResource.CATALOG,
    WorkspaceLockMode.SHARED,
)
CATALOG_EXCLUSIVE = WorkspaceLockIntent(
    WorkspaceLockResource.CATALOG,
    WorkspaceLockMode.EXCLUSIVE,
)


LOCK_ORDER: tuple[WorkspaceLockResource, ...] = (
    WorkspaceLockResource.RETENTION,
    WorkspaceLockResource.CATALOG,
    WorkspaceLockResource.WRITE,
)
_LOCK_RANK = {resource: rank for rank, resource in enumerate(LOCK_ORDER)}


@dataclass(frozen=True, slots=True)
class _HeldLock:
    workspace_root: Path
    intent: WorkspaceLockIntent


_HELD_LOCKS: ContextVar[tuple[_HeldLock, ...]] = ContextVar(
    "flameox_workspace_locks",
    default=(),
)


class WorkspaceLockManager:
    """Acquire named workspace locks in one globally enforced order."""

    def __init__(self, workspace_root: Path, paths: dict[WorkspaceLockResource, Path]) -> None:
        self.workspace_root = workspace_root.resolve()
        self.paths = paths

    @contextmanager
    def acquire(
        self,
        *intents: WorkspaceLockIntent,
        timeout: float = 30,
        phase: str = "workspace mutation",
    ) -> Iterator[tuple[object, ...]]:
        if not intents:
            raise ValueError("at least one workspace lock intent is required")
        by_resource = {intent.resource: intent for intent in intents}
        if len(by_resource) != len(intents):
            raise DomainError(
                ErrorCode.LOCK_ORDER_VIOLATION,
                "A workspace lock resource cannot be acquired twice in one operation.",
                details={"phase": phase},
            )
        ordered = tuple(sorted(intents, key=lambda intent: _LOCK_RANK[intent.resource]))
        held = tuple(
            item for item in _HELD_LOCKS.get() if item.workspace_root == self.workspace_root
        )
        held_resources = {item.intent.resource for item in held}
        repeated = held_resources.intersection(by_resource)
        if repeated:
            resource = min(repeated, key=lambda item: _LOCK_RANK[item])
            raise DomainError(
                ErrorCode.LOCK_ORDER_VIOLATION,
                "Workspace locks are not reentrant and shared-to-exclusive upgrades are refused.",
                details={"resource": resource.value, "phase": phase},
            )
        if held:
            highest = max(_LOCK_RANK[item.intent.resource] for item in held)
            first_requested = _LOCK_RANK[ordered[0].resource]
            if first_requested <= highest:
                raise DomainError(
                    ErrorCode.LOCK_ORDER_VIOLATION,
                    "Workspace lock acquisition would invert the global lock order.",
                    details={
                        "held": [item.intent.resource.value for item in held],
                        "requested": [intent.resource.value for intent in ordered],
                        "phase": phase,
                    },
                )

        acquired = tuple(_HeldLock(self.workspace_root, intent) for intent in ordered)
        token = _HELD_LOCKS.set((*_HELD_LOCKS.get(), *acquired))
        stack = ExitStack()
        streams: list[object] = []
        try:
            for intent in ordered:
                flag = (
                    portalocker.LOCK_SH
                    if intent.mode is WorkspaceLockMode.SHARED
                    else portalocker.LOCK_EX
                )
                lock = cast(
                    AbstractContextManager[object],
                    portalocker.Lock(
                        self.paths[intent.resource],
                        mode="a",
                        timeout=timeout,
                        flags=flag | portalocker.LOCK_NB,
                    ),
                )
                stream: object = stack.enter_context(lock)
                streams.append(stream)
            yield tuple(streams)
        except portalocker.exceptions.LockException as error:
            requested = (
                ordered[len(streams)].resource
                if len(streams) < len(ordered)
                else ordered[-1].resource
            )
            raise DomainError(
                ErrorCode.WRITE_LOCK_TIMEOUT,
                f"Timed out waiting for workspace {requested.value!r} lock.",
                retryable=True,
                details={
                    "resource": requested.value,
                    "phase": phase,
                    "timeout_seconds": timeout,
                },
            ) from error
        finally:
            stack.close()
            _HELD_LOCKS.reset(token)
