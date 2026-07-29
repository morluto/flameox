from __future__ import annotations

from pathlib import Path

from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.models import RunManifest
from flameox.storage.atomic import atomic_write_json
from flameox.storage.workspace import Workspace


class RunStore:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def create(self, manifest: RunManifest) -> RunManifest:
        if manifest.revision != 0:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "A new run must start at revision zero.",
            )
        with self.workspace.write_locked():
            run_root = self._run_root(manifest.run_id)
            if run_root.exists():
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Run {manifest.run_id!r} already exists.",
                )
            (run_root / "revisions").mkdir(parents=True)
            self._write_revision(manifest)
            self._write_projection(manifest)
        return manifest

    def read(self, run_id: str) -> RunManifest:
        try:
            return RunManifest.model_validate_json(
                (self._run_root(run_id) / "manifest.json").read_text()
            )
        except FileNotFoundError as exc:
            raise DomainError(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} does not exist.",
                remediation=("Call list_runs to choose an existing run.",),
                details={"missing_entity": "run"},
            ) from exc
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Run {run_id!r} has an invalid manifest.",
            ) from exc

    def append(
        self,
        manifest: RunManifest,
        *,
        expected_revision: int,
    ) -> RunManifest:
        with self.workspace.write_locked():
            current = self.read(manifest.run_id)
            if current.revision != expected_revision:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Run {manifest.run_id!r} changed before the update.",
                    retryable=True,
                    details={
                        "expected_revision": expected_revision,
                        "actual_revision": current.revision,
                    },
                )
            if manifest.revision != expected_revision + 1:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "The next run revision is not consecutive.",
                )
            self._write_revision(manifest)
            self._write_projection(manifest)
        return manifest

    def _run_root(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or "\x00" in run_id:
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Invalid run identifier.")
        return self.workspace.paths.runs / run_id

    def _write_revision(self, manifest: RunManifest) -> None:
        path = self._run_root(manifest.run_id) / "revisions" / f"{manifest.revision:08d}.json"
        if path.exists():
            existing = RunManifest.model_validate_json(path.read_text())
            if existing != manifest:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "An immutable run revision already contains different data.",
                )
            return
        atomic_write_json(path, manifest.model_dump(mode="json"))

    def _write_projection(self, manifest: RunManifest) -> None:
        atomic_write_json(
            self._run_root(manifest.run_id) / "manifest.json",
            manifest.model_dump(mode="json"),
        )
