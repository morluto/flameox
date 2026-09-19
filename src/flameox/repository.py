"""Lazy, immutable evidence repository for the process-lifespan runtime."""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from flameox.canonical import canonical_bytes
from flameox.evidence_models import (
    AnalysisRequest,
    ArtifactMetadata,
    CaptureExecution,
    EvidenceManifest,
    LogicalSource,
    ManifestBody,
    RepositoryMetadata,
    SourceLayout,
    VersionHeader,
)
from flameox.runtime_contracts import LOWERCASE_SHA256_PATTERN, RuntimeFailure
from flameox.source_files import (
    NativeSource,
    bundle_digest,
    copy_verified_file,
    directory_files,
    sha256_file,
)

REPOSITORY_FORMAT = "3"
EVIDENCE_MEDIA_TYPE = "application/vnd.flameox.evidence+json;version=3"
AGENT_EVIDENCE_MEDIA_TYPE = "application/vnd.flameox.evidence-projection+json;version=1"


class RepositoryError(RuntimeError):
    """A stable repository failure suitable for projection by a transport."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def artifact_selector(evidence_id: str, collection: str, index: int) -> str:
    """Address an immutable public position, never a guessable private filename."""
    return hashlib.sha256(canonical_bytes([evidence_id, collection, index])).hexdigest()


def evidence_source(evidence_id: str, collection: str, index: int) -> dict[str, str]:
    return {
        "kind": "evidence",
        "evidence_id": evidence_id,
        "artifact_selector": artifact_selector(evidence_id, collection, index),
    }


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    evidence_id: str
    source: LogicalSource
    members: tuple[tuple[str | None, NativeSource], ...]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EvidenceRepository:
    """Content-addressed repository with no mutable catalog or control plane."""

    def __init__(self, root: Path, session_id: str) -> None:
        self.root = root.expanduser().absolute()
        self.session_id = session_id

    @property
    def exists(self) -> bool:
        return (self.root / "repository.json").is_file()

    def _metadata_missing_is_corruption(self) -> bool:
        if not self.root.exists():
            return False
        if self.root.is_symlink() or not self.root.is_dir():
            return True
        for relative in ("artifacts/sha256", "evidence/sha256"):
            path = self.root / relative
            if path.exists() and any(path.iterdir()):
                return True
        return any(
            child.name not in {"artifacts", "evidence", ".staging"} for child in self.root.iterdir()
        )

    def _require_metadata_or_absent(self) -> bool:
        if self.exists:
            return True
        if self._metadata_missing_is_corruption():
            if self.exists:
                return True
            raise RepositoryError(
                "REPOSITORY_CORRUPTION",
                "repository.json is missing from an existing evidence repository.",
            )
        return False

    def initialize(self) -> None:
        metadata_path = self.root / "repository.json"
        if metadata_path.is_file():
            self._validate_repository()
            return
        if self._metadata_missing_is_corruption():
            if metadata_path.is_file():
                self._validate_repository()
                return
            raise RepositoryError(
                "REPOSITORY_CORRUPTION",
                "repository.json is missing from an existing evidence repository.",
            )
        if self.root.is_symlink() or (self.root.exists() and not self.root.is_dir()):
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "The Flameox data path must be a directory."
            )
        self.root.mkdir(mode=0o700, exist_ok=True)
        for relative in (
            "artifacts/sha256",
            "evidence/sha256",
            f".staging/{self.session_id}",
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": REPOSITORY_FORMAT,
            "created_at": datetime.now(UTC).isoformat(),
        }
        try:
            self._write_file(metadata_path, canonical_bytes(metadata))
        except FileExistsError:
            self._validate_repository()
        _fsync_directory(self.root)
        self._validate_repository()

    def preserve(
        self,
        *,
        manifest_body: Mapping[str, Any],
        sources: Sequence[NativeSource],
        analysis_source_indices: Sequence[int] | None,
        analysis: Mapping[str, Any],
    ) -> dict[str, Any]:
        artifacts, _layout, artifact_refs, analysis_bytes, manifest = self._publication_plan(
            manifest_body, sources, analysis_source_indices, analysis
        )
        self.initialize()
        published_refs = [self._publish_artifact(item) for item in artifacts]
        if published_refs != artifact_refs:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Artifact publication changed identity.")
        evidence_id = str(manifest["evidence_id"])
        destination = self._evidence_path(evidence_id)
        if destination.exists():
            self._validate_evidence(destination, expected=manifest)
        else:
            stage = self._new_stage("evidence")
            try:
                (stage / "data").mkdir()
                self._write_file(stage / "data" / "analysis.json", analysis_bytes)
                _fsync_directory(stage / "data")
                self._write_file(stage / "manifest.json", canonical_bytes(manifest))
                self._validate_evidence(stage, expected=manifest)
                self._publish_directory(stage, destination)
                if stage.exists():
                    self._validate_evidence(destination, expected=manifest)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        return {
            "evidence_id": evidence_id,
            "uri": f"flameox://evidence/{evidence_id}",
            "artifact_count": len(artifact_refs),
        }

    def expected_evidence_id(
        self,
        *,
        manifest_body: Mapping[str, Any],
        sources: Sequence[NativeSource],
        analysis_source_indices: Sequence[int] | None,
        analysis: Mapping[str, Any],
    ) -> str:
        """Derive a publication identity without initializing or writing the repository."""
        return str(
            self._publication_plan(manifest_body, sources, analysis_source_indices, analysis)[4][
                "evidence_id"
            ]
        )

    def _publication_plan(
        self,
        manifest_body: Mapping[str, Any],
        sources: Sequence[NativeSource],
        analysis_source_indices: Sequence[int] | None,
        analysis: Mapping[str, Any],
    ) -> tuple[list[NativeSource], SourceLayout, list[dict[str, Any]], bytes, dict[str, Any]]:
        artifacts, layout = self._publication_layout(
            sources, manifest_body, analysis_source_indices
        )
        artifact_refs = [
            {
                "format_version": REPOSITORY_FORMAT,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "role": artifact.role,
                "format": artifact.format,
                "producer": artifact.producer,
            }
            for artifact in artifacts
        ]
        analysis_bytes = canonical_bytes(analysis)
        body = dict(manifest_body)
        body["artifacts"] = artifact_refs
        body["source_layout"] = layout.model_dump(mode="json")
        body["data_files"] = [
            {
                "path": "data/analysis.json",
                "sha256": hashlib.sha256(analysis_bytes).hexdigest(),
                "size_bytes": len(analysis_bytes),
                "media_type": EVIDENCE_MEDIA_TYPE,
            }
        ]
        evidence_id = hashlib.sha256(canonical_bytes(body)).hexdigest()
        manifest = {
            "format_version": REPOSITORY_FORMAT,
            "evidence_id": evidence_id,
            "body": body,
        }
        return artifacts, layout, artifact_refs, analysis_bytes, manifest

    @staticmethod
    def _publication_layout(
        sources: Sequence[NativeSource],
        body: Mapping[str, Any],
        analysis_source_indices: Sequence[int] | None,
    ) -> tuple[list[NativeSource], SourceLayout]:
        groups: list[list[tuple[str | None, NativeSource]]] = []
        for source in sources:
            is_directory = source.path.is_dir()
            files = directory_files(source.path) if is_directory else [source.path]
            members = []
            remaining = source.size_bytes
            for path in files:
                try:
                    digest, size = sha256_file(path, max_bytes=remaining)
                except RuntimeFailure as error:
                    raise RepositoryError(
                        "MISSING_OR_CHANGED_INPUT", "Input changed before preservation."
                    ) from error
                remaining -= size
                relative = path.relative_to(source.path).as_posix() if is_directory else None
                role = source.role if relative is None else f"{source.role}:{relative}"
                members.append(
                    (
                        relative,
                        NativeSource(path, digest, size, source.format, source.producer, role),
                    )
                )
            digest = (
                bundle_digest(
                    (relative, member.sha256)
                    for relative, member in members
                    if relative is not None
                )
                if is_directory
                else members[0][1].sha256
            )
            if (digest, sum(member.size_bytes for _, member in members)) != (
                source.sha256,
                source.size_bytes,
            ):
                raise RepositoryError(
                    "MISSING_OR_CHANGED_INPUT", "Input changed before preservation."
                )
            groups.append(members)
        expanded_roles = Counter(member.role for members in groups for _, member in members)
        namespace = len({source.role for source in sources}) != len(sources) or any(
            count > 1 for count in expanded_roles.values()
        )
        artifacts: list[NativeSource] = []
        logical: list[LogicalSource] = []
        for index, (source, members) in enumerate(zip(sources, groups, strict=True)):
            role = f"source-{index + 1:04d}/{source.role}" if namespace else source.role
            indices = list(range(len(artifacts), len(artifacts) + len(members)))
            for relative, member in members:
                member.role = role if relative is None else f"{role}:{relative}"
                artifacts.append(member)
            logical.append(
                LogicalSource(
                    role=role,
                    sha256=source.sha256,
                    size_bytes=source.size_bytes,
                    format=source.format,
                    producer=source.producer,
                    is_directory=source.path.is_dir(),
                    artifact_indices=indices,
                    relative_paths=[relative for relative, _ in members if relative is not None],
                )
            )
        request = AnalysisRequest.model_validate(body["analysis_request"])
        indices = list(
            analysis_source_indices if analysis_source_indices is not None else range(len(sources))
        )
        if len(indices) != len(request.inputs):
            raise RepositoryError(
                "MISSING_OR_CHANGED_INPUT",
                "Analysis source mapping does not match the analyzed input count.",
            )
        for item, source_index in zip(request.inputs, indices, strict=True):
            if not 0 <= source_index < len(sources):
                raise RepositoryError(
                    "MISSING_OR_CHANGED_INPUT", "Analysis source mapping is out of range."
                )
            source = sources[source_index]
            if (source.sha256, source.format, source.producer) != (
                item.sha256,
                item.format,
                item.producer,
            ):
                raise RepositoryError(
                    "MISSING_OR_CHANGED_INPUT", "Analysis input is absent from preserved sources."
                )
        return artifacts, SourceLayout(sources=logical, analysis_sources=indices)

    def read(self, evidence_id: str) -> dict[str, Any]:
        return self._read_manifest(evidence_id).model_dump(mode="json", exclude_unset=True)

    def _require_evidence_path(self, evidence_id: str) -> Path:
        self._validate_id(evidence_id)
        if not self._require_metadata_or_absent():
            raise RepositoryError("MISSING_EVIDENCE", f"Evidence {evidence_id} does not exist.")
        self._validate_repository()
        path = self._evidence_path(evidence_id)
        if not path.is_dir():
            raise RepositoryError("MISSING_EVIDENCE", f"Evidence {evidence_id} does not exist.")
        return path

    def _read_manifest(self, evidence_id: str) -> EvidenceManifest:
        return self._validate_evidence(self._require_evidence_path(evidence_id))

    def select_source(
        self, evidence_id: str, *, selector: str | None, role: str | None
    ) -> EvidenceSelection:
        """Resolve validated metadata without reading payloads before runtime admission."""
        manifest = self._parse_evidence(self._require_evidence_path(evidence_id))
        logical = manifest.body.source_layout.sources
        files = [
            LogicalSource(
                role=item.role,
                sha256=item.sha256,
                size_bytes=item.size_bytes,
                format=item.format,
                producer=item.producer,
                is_directory=False,
                artifact_indices=[index],
                relative_paths=[],
            )
            for index, item in enumerate(manifest.body.artifacts)
        ]
        selected = None
        if selector is not None:
            selected = next(
                (
                    item
                    for collection, items in (("artifact", files), ("logical", logical))
                    for index, item in enumerate(items)
                    if artifact_selector(evidence_id, collection, index) == selector
                ),
                None,
            )
        elif role is not None:
            selected = next((item for item in [*files, *logical] if item.role == role), None)
        else:
            candidates = [
                item
                for item in logical
                if not item.role.endswith(
                    ("/stdout", "/stderr", "/oracle_stdout", "/oracle_stderr")
                )
            ]
            if len(candidates) != 1:
                raise RepositoryError(
                    "INVALID_INPUT", "A selector is required for multiple logical sources."
                )
            selected = candidates[0]
        if selected is None:
            raise RepositoryError("MISSING_EVIDENCE", "The requested evidence source is absent.")
        paths = selected.relative_paths
        members = []
        for position, index in enumerate(selected.artifact_indices):
            item = manifest.body.artifacts[index]
            relative = None
            if selected.is_directory:
                relative = paths[position]
            members.append(
                (
                    relative,
                    NativeSource(
                        self._artifact_path(item.sha256) / "payload",
                        item.sha256,
                        item.size_bytes,
                        item.format,
                        item.producer,
                        item.role,
                    ),
                )
            )
        return EvidenceSelection(evidence_id, selected, tuple(members))

    def verify_source(self, selection: EvidenceSelection) -> None:
        for _, source in selection.members:
            self._validate_artifact(
                source.path.parent,
                {
                    "format_version": REPOSITORY_FORMAT,
                    "sha256": source.sha256,
                    "size_bytes": source.size_bytes,
                },
            )

    def read_agent_projection(self, evidence_id: str) -> dict[str, Any]:
        """Return a bounded MCP-safe view without weakening canonical provenance."""

        manifest = self._read_manifest(evidence_id)
        body = manifest.body
        capture = body.capture_request
        safe_capture = None
        if capture is not None:
            safe_capture = {
                "request_sha256": hashlib.sha256(
                    canonical_bytes(capture.model_dump(mode="json", exclude_unset=True))
                ).hexdigest(),
                "mode": capture.mode,
                "target": {
                    "provider_id": capture.target.provider_id,
                    **(
                        {"budget": capture.target.budget.model_dump(mode="json")}
                        if "budget" in capture.target.model_fields_set
                        else {}
                    ),
                    "argument_count": len(capture.target.argv),
                    "environment_override_count": len(capture.target.environment),
                },
                "experiment_present": capture.experiment is not None,
                "executions": [
                    self._safe_execution_projection(item) for item in capture.executions
                ],
            }
        analysis = body.analysis_request
        safe_analysis = {
            "request_sha256": hashlib.sha256(
                canonical_bytes(analysis.model_dump(mode="json", exclude_unset=True))
            ).hexdigest(),
            "capability_id": analysis.capability_id,
            "inputs": [{"sha256": item.sha256, "format": item.format} for item in analysis.inputs],
            "offset": analysis.offset,
            "failure": {"code": analysis.failure.code} if analysis.failure is not None else None,
        }
        return {
            "format_version": manifest.format_version,
            "evidence_id": manifest.evidence_id,
            "analysis_sources": [
                evidence_source(evidence_id, "logical", index)
                for index in body.source_layout.analysis_sources
            ],
            "logical_sources": [
                {"sha256": item.sha256, "size_bytes": item.size_bytes, "format": item.format}
                | {"source": evidence_source(evidence_id, "logical", index)}
                for index, item in enumerate(body.source_layout.sources)
            ],
            "body": {
                "evidence_kind": body.evidence_kind,
                "capability_id": body.capability_id,
                "provider": body.provider.model_dump(mode="json"),
                "inputs": [
                    item.model_dump(include={"sha256", "size_bytes", "format"})
                    for item in body.inputs
                ],
                "capture_request": safe_capture,
                "analysis_request": safe_analysis,
                "episode": body.episode.model_dump(mode="json"),
                "coverage": body.coverage.model_dump(mode="json"),
                "limitations": {
                    "count": len(body.limitations),
                    "sha256": hashlib.sha256(canonical_bytes(body.limitations)).hexdigest(),
                },
                "artifacts": [
                    item.model_dump(include={"sha256", "size_bytes", "format"})
                    | {"source": evidence_source(evidence_id, "artifact", index)}
                    for index, item in enumerate(body.artifacts)
                ],
                "data_files": [
                    item.model_dump(include={"sha256", "size_bytes", "media_type"})
                    for item in body.data_files
                ],
            },
        }

    @staticmethod
    def _safe_execution_projection(execution: CaptureExecution) -> dict[str, Any]:
        diagnostic_counts = {
            "stdout_observed_bytes",
            "stderr_observed_bytes",
            "stdout_retained_bytes",
            "stderr_retained_bytes",
            "stdout_omitted_bytes",
            "stderr_omitted_bytes",
            "stdout_complete",
            "stderr_complete",
        }
        return execution.model_dump(
            mode="json",
            include={
                "block": True,
                "returncode": True,
                "returncode_scope": True,
                "executable_sha256": True,
                "collector_executable_sha256": True,
                "workload_executable_sha256": True,
                "workload_returncode": True,
                "status": True,
                "failure_code": True,
                "missing_artifact_roles": True,
                "artifact_rejections": True,
                "wall_time_ns": True,
                "containment": True,
                "output_streams": True,
                "console_diagnostics": diagnostic_counts,
                "semantic_oracle": {
                    "returncode": True,
                    "status": True,
                    "failure_code": True,
                    "limit": True,
                    "output_streams": True,
                    "console_diagnostics": diagnostic_counts,
                },
            },
        )

    def query(
        self,
        *,
        evidence_kind: str | None = None,
        capability_id: str | None = None,
        provider_id: str | None = None,
        input_sha256: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 200:
            raise RepositoryError("INVALID_INPUT", "limit must be between 1 and 200")
        if (
            input_sha256 is not None
            and re.fullmatch(LOWERCASE_SHA256_PATTERN, input_sha256) is None
        ):
            raise RepositoryError(
                "INVALID_INPUT", "input_sha256 must be a lowercase SHA-256 digest"
            )
        if created_after is not None and created_after.tzinfo is None:
            raise RepositoryError("INVALID_INPUT", "created_after must include a timezone")
        if created_before is not None and created_before.tzinfo is None:
            raise RepositoryError("INVALID_INPUT", "created_before must include a timezone")
        if (
            created_after is not None
            and created_before is not None
            and created_after > created_before
        ):
            raise RepositoryError("INVALID_INPUT", "created_after must not exceed created_before")
        if not self._require_metadata_or_absent():
            return {"evidence": [], "continuation": None, "inventory_digest": _empty_digest()}
        self._validate_repository()
        evidence_root = self.root / "evidence" / "sha256"
        self._validate_inventory_layout(evidence_root)
        inventory = sorted(evidence_root.glob("*/*/manifest.json"))
        inventory_ids = [path.parent.name for path in inventory]
        inventory_digest = hashlib.sha256("\n".join(inventory_ids).encode()).hexdigest()
        query_digest = self._query_digest(
            evidence_kind=evidence_kind,
            capability_id=capability_id,
            provider_id=provider_id,
            input_sha256=input_sha256,
            created_after=created_after,
            created_before=created_before,
        )
        offset = self._decode_cursor(cursor, inventory_digest, query_digest)
        if cursor is not None and offset >= len(inventory):
            raise RepositoryError(
                "INVALID_INPUT", "Repository query continuation is beyond the inventory."
            )
        matches: list[dict[str, Any]] = []
        next_offset: int | None = None
        for index, path in enumerate(inventory[offset:], offset):
            manifest = self._validate_evidence(path.parent)
            body = manifest.body
            if not self._matches(
                body,
                evidence_kind=evidence_kind,
                capability_id=capability_id,
                provider_id=provider_id,
                input_sha256=input_sha256,
                created_after=created_after,
                created_before=created_before,
            ):
                continue
            if len(matches) == limit:
                next_offset = index
                break
            matches.append(self._summary(manifest))
        continuation = (
            self._encode_cursor(inventory_digest, query_digest, next_offset)
            if next_offset is not None
            else None
        )
        return {
            "evidence": matches,
            "continuation": continuation,
            "inventory_digest": inventory_digest,
        }

    def cleanup_abandoned_staging(self) -> None:
        """Remove only staging owned by process identities proven dead."""

        self._validate_repository()
        staging = self.root / ".staging"
        for owner in staging.iterdir():
            if owner.name == self.session_id or not owner.is_dir():
                continue
            pid_text = owner.name.split("-", 1)[0]
            if not pid_text.isdigit():
                continue
            try:
                os.kill(int(pid_text), 0)
            except ProcessLookupError:
                shutil.rmtree(owner)
            except (PermissionError, OSError):
                continue

    def _publish_artifact(self, artifact: NativeSource) -> dict[str, Any]:
        if not artifact.path.is_file():
            raise RepositoryError(
                "UNSUPPORTED_FORMAT", "Only regular native artifact files can be preserved."
            )
        destination = self._artifact_path(artifact.sha256)
        metadata = {
            "format_version": REPOSITORY_FORMAT,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
        if destination.exists():
            self._validate_artifact(destination, metadata)
        else:
            stage = self._new_stage("artifact")
            try:
                try:
                    copy_verified_file(artifact, stage / "payload")
                except RuntimeFailure as error:
                    raise RepositoryError(error.code, error.message) from error
                with (stage / "payload").open("rb") as stream:
                    os.fsync(stream.fileno())
                self._write_file(stage / "artifact.json", canonical_bytes(metadata))
                self._validate_artifact(stage, metadata)
                self._publish_directory(stage, destination)
                if stage.exists():
                    self._validate_artifact(destination, metadata)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        return {
            **metadata,
            "role": artifact.role,
            "format": artifact.format,
            "producer": artifact.producer,
        }

    def _publish_directory(self, stage: Path, destination: Path) -> None:
        """Rename a validated stage; a remaining stage means another publisher won."""
        self._assert_no_symlink_path(destination.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._assert_no_symlink_path(destination.parent)
        _fsync_directory(stage)
        try:
            stage.rename(destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
        _fsync_directory(destination.parent)

    def _new_stage(self, kind: str) -> Path:
        root = self.root / ".staging" / self.session_id
        self._assert_no_symlink_path(root)
        root.mkdir(parents=True, exist_ok=True)
        self._assert_no_symlink_path(root)
        return Path(tempfile.mkdtemp(prefix=f"{kind}-{uuid4().hex}-", dir=root))

    def _read_document[M: BaseModel](self, path: Path, model: type[M]) -> M:
        self._assert_no_symlink_path(path)
        try:
            value = json.loads(path.read_bytes())
            header = VersionHeader.model_validate(value)
            if header.format_version != REPOSITORY_FORMAT:
                raise RepositoryError(
                    "UNSUPPORTED_REPOSITORY_FORMAT", "The evidence format is unsupported."
                )
            return model.model_validate(value)
        except (OSError, ValueError) as exc:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence metadata is unreadable or invalid."
            ) from exc

    def _validate_repository(self) -> None:
        self._read_document(self.root / "repository.json", RepositoryMetadata)
        self._validate_repository_layout()

    def _validate_artifact(self, path: Path, expected: Mapping[str, Any]) -> None:
        self._assert_no_symlink_path(path)
        try:
            entries = {item.name for item in path.iterdir()}
            if entries != {"artifact.json", "payload"} or any(
                item.is_symlink() for item in path.iterdir()
            ):
                raise RepositoryError("REPOSITORY_CORRUPTION", "Artifact bundle layout is invalid.")
            metadata = self._read_document(path / "artifact.json", ArtifactMetadata).model_dump(
                mode="json"
            )
            digest, size = sha256_file(path / "payload", max_bytes=expected["size_bytes"])
        except RepositoryError:
            raise
        except (OSError, RuntimeFailure) as exc:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Artifact bundle is incomplete."
            ) from exc
        if metadata != expected or digest != expected["sha256"] or size != expected["size_bytes"]:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Artifact bundle digest mismatch.")

    def _parse_evidence(
        self, path: Path, expected: Mapping[str, Any] | None = None
    ) -> EvidenceManifest:
        manifest = self._read_document(path / "manifest.json", EvidenceManifest)
        serialized = manifest.model_dump(mode="json", exclude_unset=True)
        try:
            evidence_id = hashlib.sha256(canonical_bytes(serialized["body"])).hexdigest()
        except ValueError as exc:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence is not canonical JSON."
            ) from exc
        if evidence_id != manifest.evidence_id or (expected is None and evidence_id != path.name):
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence identity does not match its body."
            )
        if expected is not None and serialized != expected:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Existing evidence bundle differs.")
        return manifest

    def _validate_evidence(
        self, path: Path, expected: Mapping[str, Any] | None = None
    ) -> EvidenceManifest:
        manifest = self._parse_evidence(path, expected)
        for item in manifest.body.data_files:
            try:
                data_path = path / item.path
                self._assert_no_symlink_path(data_path)
                digest, size = sha256_file(data_path, max_bytes=item.size_bytes)
            except RepositoryError:
                raise
            except (OSError, RuntimeFailure) as exc:
                raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence data is missing.") from exc
            if (digest, size) != (item.sha256, item.size_bytes):
                raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence data digest mismatch.")
        for artifact in manifest.body.artifacts:
            self._validate_artifact(
                self._artifact_path(artifact.sha256),
                {
                    "format_version": REPOSITORY_FORMAT,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                },
            )
        return manifest

    def _validate_repository_layout(self) -> None:
        for path in (
            self.root,
            self.root / "artifacts",
            self.root / "artifacts" / "sha256",
            self.root / "evidence",
            self.root / "evidence" / "sha256",
            self.root / ".staging",
        ):
            self._assert_no_symlink_path(path)
            if not path.is_dir():
                raise RepositoryError(
                    "REPOSITORY_CORRUPTION", "Evidence repository layout is incomplete."
                )

    def _validate_inventory_layout(self, evidence_root: Path) -> None:
        for prefix in evidence_root.iterdir():
            self._assert_no_symlink_path(prefix)
            if (
                not prefix.is_dir()
                or len(prefix.name) != 2
                or any(character not in "0123456789abcdef" for character in prefix.name)
            ):
                raise RepositoryError(
                    "REPOSITORY_CORRUPTION", "Evidence inventory prefix is invalid."
                )
            for bundle in prefix.iterdir():
                self._assert_no_symlink_path(bundle)
                if (
                    not bundle.is_dir()
                    or not _is_digest(bundle.name)
                    or bundle.name[:2] != prefix.name
                    or not (bundle / "manifest.json").is_file()
                ):
                    raise RepositoryError(
                        "REPOSITORY_CORRUPTION", "Evidence inventory bundle is invalid."
                    )

    def _assert_no_symlink_path(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Repository path escapes the evidence data directory."
            ) from exc
        current = self.root
        if current.is_symlink():
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Repository paths must not contain symlinks."
            )
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise RepositoryError(
                    "REPOSITORY_CORRUPTION", "Repository paths must not contain symlinks."
                )

    @staticmethod
    def _matches(
        body: ManifestBody,
        *,
        evidence_kind: str | None,
        capability_id: str | None,
        provider_id: str | None,
        input_sha256: str | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> bool:
        if evidence_kind is not None and body.evidence_kind != evidence_kind:
            return False
        if capability_id is not None and body.capability_id != capability_id:
            return False
        if provider_id is not None and body.provider.id != provider_id:
            return False
        if input_sha256 is not None and all(item.sha256 != input_sha256 for item in body.inputs):
            return False
        created_at = datetime.fromisoformat(body.episode.created_at)
        return not (
            (created_after is not None and created_at < created_after)
            or (created_before is not None and created_at > created_before)
        )

    @staticmethod
    def _summary(manifest: EvidenceManifest) -> dict[str, Any]:
        body = manifest.body
        return {
            "evidence_id": manifest.evidence_id,
            "uri": f"flameox://evidence/{manifest.evidence_id}",
            "evidence_kind": body.evidence_kind,
            "capability_id": body.capability_id,
            "provider": body.provider.model_dump(mode="json"),
            "created_at": body.episode.created_at,
            "coverage": body.coverage.model_dump(mode="json"),
            "limitations": body.limitations,
        }

    @staticmethod
    def _query_digest(
        *,
        evidence_kind: str | None,
        capability_id: str | None,
        provider_id: str | None,
        input_sha256: str | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> str:
        filters = {
            "evidence_kind": evidence_kind,
            "capability_id": capability_id,
            "provider_id": provider_id,
            "input_sha256": input_sha256,
            "created_after": created_after.isoformat() if created_after is not None else None,
            "created_before": created_before.isoformat() if created_before is not None else None,
        }
        return hashlib.sha256(canonical_bytes(filters)).hexdigest()

    @staticmethod
    def _encode_cursor(inventory_digest: str, query_digest: str, offset: int) -> str:
        value = canonical_bytes(
            {"inventory": inventory_digest, "query": query_digest, "offset": offset}
        )
        return base64.urlsafe_b64encode(value).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str | None, inventory_digest: str, query_digest: str) -> int:
        if cursor is None:
            return 0
        try:
            padding = "=" * (-len(cursor) % 4)
            value = json.loads(base64.urlsafe_b64decode(cursor + padding))
            if value["inventory"] != inventory_digest:
                raise RepositoryError("INVALID_INPUT", "Repository query continuation is stale.")
            if value["query"] != query_digest:
                raise RepositoryError(
                    "INVALID_INPUT", "Repository query continuation does not match its filters."
                )
            offset = value["offset"]
            if type(offset) is not int or offset < 0:
                raise ValueError
            return offset
        except RepositoryError:
            raise
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "INVALID_INPUT", "Repository query continuation is invalid."
            ) from exc

    def _artifact_path(self, digest: str) -> Path:
        return self.root / "artifacts" / "sha256" / digest[:2] / digest

    def _evidence_path(self, evidence_id: str) -> Path:
        return self.root / "evidence" / "sha256" / evidence_id[:2] / evidence_id

    @staticmethod
    def _validate_id(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RepositoryError(
                "INVALID_INPUT", "evidence_id must be a lowercase SHA-256 digest."
            )

    def _write_file(self, path: Path, content: bytes) -> None:
        temporary_parent = path.parent
        if path == self.root / "repository.json":
            temporary_parent = self.root / ".staging" / self.session_id
            self._assert_no_symlink_path(temporary_parent)
            temporary_parent.mkdir(parents=True, exist_ok=True)
            self._assert_no_symlink_path(temporary_parent)
        temporary = temporary_parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _empty_digest() -> str:
    return hashlib.sha256(b"").hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
