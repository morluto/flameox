from __future__ import annotations

from typing import overload

from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import digest_model
from flameox.domain.models import (
    ExecutionRunManifest,
    ImportRunManifest,
    RunManifest,
    parse_run_manifest,
    parse_run_manifest_json,
)
from flameox.domain.projections import ProjectionIntentSpec
from flameox.storage.control_plane import ControlPlane, canonical_json
from flameox.storage.workspace import Workspace

_OUTPUT_ONLY_FIELDS = {"process": {"timed_out"}}


class RunStore:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.control_plane = ControlPlane(workspace)

    @overload
    def create(
        self,
        manifest: ImportRunManifest,
        *,
        projection_intent: ProjectionIntentSpec | None = None,
    ) -> ImportRunManifest: ...

    @overload
    def create(
        self,
        manifest: ExecutionRunManifest,
        *,
        projection_intent: ProjectionIntentSpec | None = None,
    ) -> ExecutionRunManifest: ...

    def create(
        self,
        manifest: RunManifest,
        *,
        projection_intent: ProjectionIntentSpec | None = None,
    ) -> RunManifest:
        manifest = self._canonical(manifest)
        if manifest.revision != 0:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "A new run must start at revision zero.",
            )
        self._require_projection_binding(manifest, projection_intent)
        with self.workspace.write_locked():
            self.control_plane.create_run(
                run_id=manifest.run_id,
                run_type=type(manifest).__name__,
                revision=manifest.revision,
                payload_json=self._json(manifest),
                projection_intent=projection_intent,
            )
        return manifest

    def read(self, run_id: str) -> RunManifest:
        try:
            return parse_run_manifest_json(self.control_plane.read_run(run_id))
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Run {run_id!r} has an invalid manifest.",
            ) from exc

    def list(self) -> tuple[RunManifest, ...]:
        try:
            return tuple(
                parse_run_manifest_json(payload) for payload in self.control_plane.list_runs()
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "The SQLite control plane contains an invalid run manifest.",
            ) from exc

    def read_revision(self, run_id: str, revision: int) -> RunManifest:
        try:
            return parse_run_manifest_json(self.control_plane.read_run_revision(run_id, revision))
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Run {run_id!r} revision {revision} has an invalid manifest.",
            ) from exc

    def exists(self, run_id: str) -> bool:
        try:
            self.control_plane.read_run(run_id)
        except DomainError as error:
            if error.code is ErrorCode.RUN_NOT_FOUND:
                return False
            raise
        return True

    @overload
    def append(
        self,
        manifest: ImportRunManifest,
        *,
        expected_revision: int,
        projection_intent: ProjectionIntentSpec | None = None,
    ) -> ImportRunManifest: ...

    @overload
    def append(
        self,
        manifest: ExecutionRunManifest,
        *,
        expected_revision: int,
        projection_intent: ProjectionIntentSpec | None = None,
    ) -> ExecutionRunManifest: ...

    def append(
        self,
        manifest: RunManifest,
        *,
        expected_revision: int,
        projection_intent: ProjectionIntentSpec | None = None,
    ) -> RunManifest:
        manifest = self._canonical(manifest)
        self._require_projection_binding(manifest, projection_intent)
        with self.workspace.write_locked():
            self.control_plane.append_run(
                run_id=manifest.run_id,
                run_type=type(manifest).__name__,
                expected_revision=expected_revision,
                next_revision=manifest.revision,
                payload_json=self._json(manifest),
                projection_intent=projection_intent,
            )
        return manifest

    @staticmethod
    def _canonical(manifest: RunManifest) -> RunManifest:
        try:
            canonical = parse_run_manifest(
                manifest.model_dump(mode="python", exclude=_OUTPUT_ONLY_FIELDS)
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Run manifest is invalid and cannot be persisted.",
            ) from exc
        if type(canonical) is not type(manifest):
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "A run revision cannot change its run type.",
            )
        return canonical

    @staticmethod
    def _json(manifest: RunManifest) -> str:
        return canonical_json(manifest.model_dump(mode="json", exclude=_OUTPUT_ONLY_FIELDS))

    def _require_projection_binding(
        self,
        manifest: RunManifest,
        intent: ProjectionIntentSpec | None,
    ) -> None:
        if intent is None:
            return
        expected_digest = digest_model(manifest.model_dump(mode="json"))
        if (
            intent.workspace_id != self.workspace.identity.workspace_id
            or intent.domain_kind != "run"
            or intent.domain_id != manifest.run_id
            or intent.domain_revision != manifest.revision
            or intent.domain_digest != expected_digest
        ):
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Projection intent does not match the exact canonical run revision.",
            )
