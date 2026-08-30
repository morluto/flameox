from __future__ import annotations

import hashlib
import json
from importlib.metadata import Distribution, EntryPoint, entry_points
from pathlib import Path
from typing import Literal, cast

from flameox.action_graph import ActionId, tool_action
from flameox.atomic import atomic_write_json
from flameox.domain import AdapterV1, DomainError, ErrorCode, digest_model
from flameox.models import ContractModel
from flameox.storage import Workspace

ENTRY_POINT_GROUP = "flameox.adapters"
_ADAPTER_API_VERSION = 1


class AdapterApproval(ContractModel):
    distribution: str
    version: str
    package_identity: str
    provenance: Literal["agent", "cli"]


class AdapterDescriptor(ContractModel):
    adapter: str
    entry_point: str
    distribution: str
    version: str
    package_identity: str
    approved: bool


class AdapterDiscoveryResult(ContractModel):
    adapters: tuple[AdapterDescriptor, ...]


class AdapterRegistry:
    """Discover entry points without importing them and bind approvals to wheel identity."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.approvals_path = workspace.paths.records / "adapter-approvals.json"

    def discover(self) -> AdapterDiscoveryResult:
        approvals = self._approvals()
        descriptors: list[AdapterDescriptor] = []
        for entry_point in sorted(
            entry_points(group=ENTRY_POINT_GROUP),
            key=lambda item: (item.name, item.value),
        ):
            distribution = entry_point.dist
            if distribution is None:
                continue
            name = distribution.metadata["Name"] or "unknown"
            identity = _distribution_identity(distribution)
            approval = approvals.get(name.casefold())
            descriptors.append(
                AdapterDescriptor(
                    adapter=entry_point.name,
                    entry_point=entry_point.value,
                    distribution=name,
                    version=distribution.version,
                    package_identity=identity,
                    approved=(
                        approval is not None
                        and approval.version == distribution.version
                        and approval.package_identity == identity
                    ),
                )
            )
        return AdapterDiscoveryResult(adapters=tuple(descriptors))

    def approve(self, distribution_name: str) -> AdapterDiscoveryResult:
        matches = [
            descriptor
            for descriptor in self.discover().adapters
            if descriptor.distribution.casefold() == distribution_name.casefold()
        ]
        if not matches:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"No {ENTRY_POINT_GROUP!r} entry point is installed for "
                f"distribution {distribution_name!r}.",
            )
        versions = {(item.version, item.package_identity) for item in matches}
        if len(versions) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "One distribution resolved to conflicting package identities.",
            )
        version, package_identity = versions.pop()
        if package_identity.startswith("unverifiable:"):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The installed distribution contains unreadable or untracked files.",
            )
        with self.workspace.write_locked():
            approvals = self._approvals()
            approvals[matches[0].distribution.casefold()] = AdapterApproval(
                distribution=matches[0].distribution,
                version=version,
                package_identity=package_identity,
                provenance="cli",
            )
            self._write_approvals(approvals)
        return self.discover()

    def prepare(self, adapter: str, distribution_name: str) -> AdapterDescriptor:
        """Approve one exact installed entry point with agent provenance."""
        matches = [
            descriptor
            for descriptor in self.discover().adapters
            if descriptor.adapter == adapter
            and descriptor.distribution.casefold() == distribution_name.casefold()
        ]
        if not matches:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"No installed adapter {adapter!r} belongs to distribution {distribution_name!r}.",
                details={"adapter": adapter},
                remediation=(
                    "Use the adapter and distribution identity returned by list_capabilities, "
                    "then retry prepare_adapter.",
                ),
                next_action=tool_action(ActionId.INSPECT_CAPABILITIES, adapter=adapter),
            )
        descriptor = matches[0]
        if descriptor.package_identity.startswith("unverifiable:"):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The installed adapter distribution has no verifiable package identity.",
                details={"adapter": adapter, "distribution": descriptor.distribution},
                remediation=(
                    "Reinstall the distribution with intact metadata, then retry prepare_adapter.",
                ),
            )
        with self.workspace.write_locked():
            approvals = self._approvals()
            approvals[descriptor.distribution.casefold()] = AdapterApproval(
                distribution=descriptor.distribution,
                version=descriptor.version,
                package_identity=descriptor.package_identity,
                provenance="agent",
            )
            self._write_approvals(approvals)
        refreshed = next(
            item for item in self.discover().adapters if item.adapter == adapter and item.approved
        )
        return refreshed

    def revoke(self, distribution_name: str) -> AdapterDiscoveryResult:
        with self.workspace.write_locked():
            approvals = self._approvals()
            removed = approvals.pop(distribution_name.casefold(), None)
            if removed is None:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Distribution {distribution_name!r} is not approved.",
                )
            self._write_approvals(approvals)
        return self.discover()

    def approved_descriptor(self, adapter: str) -> AdapterDescriptor:
        descriptor = next(
            (
                item
                for item in self.discover().adapters
                if item.adapter == adapter and item.approved
            ),
            None,
        )
        if descriptor is None:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Third-party adapter {adapter!r} is not installed and approved.",
            )
        return descriptor

    def load_contract(self, adapter: str) -> tuple[AdapterDescriptor, AdapterV1]:
        descriptor = self.approved_descriptor(adapter)
        entry_point = self._entry_point(descriptor)
        try:
            loaded = entry_point.load()
        except Exception as exc:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Approved adapter {adapter!r} could not be loaded.",
                details={"exception_type": type(exc).__name__},
            ) from exc
        if (
            getattr(loaded, "name", None) != adapter
            or getattr(loaded, "api_version", None) != _ADAPTER_API_VERSION
            or any(
                not callable(getattr(loaded, method, None))
                for method in ("probe", "plan", "validate", "extract")
            )
        ):
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Approved adapter {adapter!r} does not implement the Flameox v1 contract.",
            )
        return descriptor, cast(AdapterV1, loaded)

    def _entry_point(self, descriptor: AdapterDescriptor) -> EntryPoint:
        for entry_point in entry_points(group=ENTRY_POINT_GROUP):
            distribution = entry_point.dist
            if distribution is None:
                continue
            if (
                entry_point.name == descriptor.adapter
                and entry_point.value == descriptor.entry_point
                and (distribution.metadata["Name"] or "").casefold()
                == descriptor.distribution.casefold()
                and distribution.version == descriptor.version
                and _distribution_identity(distribution) == descriptor.package_identity
            ):
                return entry_point
        raise DomainError(
            ErrorCode.REVISION_CONFLICT,
            "The approved entry point changed after discovery.",
            retryable=True,
            details={"adapter": descriptor.adapter},
            remediation=(
                "Refresh list_capabilities and use the currently reported adapter identity.",
            ),
            next_action=tool_action(
                ActionId.INSPECT_CAPABILITIES,
                adapter=descriptor.adapter,
            ),
        )

    def _approvals(self) -> dict[str, AdapterApproval]:
        if not self.approvals_path.exists():
            return {}
        try:
            payload = json.loads(self.approvals_path.read_text())
            if not isinstance(payload, dict) or set(payload) != {"approvals"}:
                raise ValueError("adapter approval registry fields are invalid")
            approvals = payload["approvals"]
            if not isinstance(approvals, dict) or not all(
                isinstance(key, str) for key in approvals
            ):
                raise ValueError("adapter approvals must be a string-keyed object")
            return {key: AdapterApproval.model_validate(value) for key, value in approvals.items()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "The third-party adapter approval registry is invalid.",
            ) from exc

    def _write_approvals(self, approvals: dict[str, AdapterApproval]) -> None:
        atomic_write_json(
            self.approvals_path,
            {
                "approvals": {
                    key: value.model_dump(mode="json") for key, value in sorted(approvals.items())
                },
            },
        )


def _distribution_identity(distribution: Distribution) -> str:
    files: list[dict[str, str | int | None]] = []
    installed_paths = tuple(sorted(distribution.files or (), key=str))
    verifiable = bool(installed_paths)
    for path in installed_paths:
        file_hash = path.hash
        installed = Path(str(distribution.locate_file(path)))
        content_digest: str | None
        try:
            digest = hashlib.sha256()
            with installed.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            content_digest = digest.hexdigest()
        except (OSError, ValueError):
            content_digest = None
            verifiable = False
        files.append(
            {
                "path": str(path),
                "hash_mode": file_hash.mode if file_hash is not None else None,
                "hash_value": file_hash.value if file_hash is not None else None,
                "size": path.size,
                "content_sha256": content_digest,
            }
        )
    identity = digest_model(
        {
            "name": distribution.metadata["Name"],
            "version": distribution.version,
            "files": files,
        }
    )
    return identity if verifiable else f"unverifiable:{identity}"
