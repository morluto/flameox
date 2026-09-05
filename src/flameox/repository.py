"""Lazy, immutable evidence repository for the stateless runtime."""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from flameox.canonical import canonical_bytes
from flameox.runtime_contracts import LOWERCASE_SHA256_PATTERN, CaptureTarget, ExperimentDesign

REPOSITORY_FORMAT = "1"
EVIDENCE_MEDIA_TYPE = "application/vnd.flameox.evidence+json;version=1"
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
class NativeArtifact:
    path: Path
    role: str
    sha256: str
    size_bytes: int
    format: str
    producer: str | None = None


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


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
        artifacts: Iterable[NativeArtifact],
        analysis: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.initialize()
        artifact_refs = [self._publish_artifact(item) for item in artifacts]
        analysis_bytes = canonical_bytes(analysis)
        analysis_digest = hashlib.sha256(analysis_bytes).hexdigest()
        body = dict(manifest_body)
        body["artifacts"] = artifact_refs
        body["data_files"] = [
            {
                "path": "data/analysis.json",
                "sha256": analysis_digest,
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
                self._validate_evidence(destination, expected=manifest)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        return {
            "evidence_id": evidence_id,
            "uri": f"flameox://evidence/{evidence_id}",
            "artifact_count": len(artifact_refs),
        }

    def read(self, evidence_id: str) -> dict[str, Any]:
        self._validate_id(evidence_id)
        if not self._require_metadata_or_absent():
            raise RepositoryError("MISSING_EVIDENCE", f"Evidence {evidence_id} does not exist.")
        self._validate_repository()
        path = self._evidence_path(evidence_id)
        if not path.is_dir():
            raise RepositoryError("MISSING_EVIDENCE", f"Evidence {evidence_id} does not exist.")
        return self._validate_evidence(path)

    def read_agent_projection(self, evidence_id: str) -> dict[str, Any]:
        """Return a bounded MCP-safe view without weakening canonical provenance."""

        manifest = self.read(evidence_id)
        body = manifest["body"]
        logical_artifacts = self.logical_artifacts(body)
        capture_request = body["capture_request"]
        safe_capture: dict[str, Any] | None = None
        if capture_request is not None:
            target = capture_request.get("target", {})
            executions = capture_request.get("executions", [])
            safe_capture = {
                "request_sha256": hashlib.sha256(canonical_bytes(capture_request)).hexdigest(),
                "mode": capture_request.get("mode"),
                "target": {
                    "provider_id": target.get("provider_id"),
                    "argument_count": len(target.get("argv", [])),
                    "environment_override_count": len(target.get("environment", {})),
                },
                "experiment_present": capture_request.get("experiment") is not None,
                "executions": [self._safe_execution_projection(item) for item in executions],
            }
        analysis_request = body["analysis_request"]
        safe_analysis = {
            "request_sha256": hashlib.sha256(canonical_bytes(analysis_request)).hexdigest(),
            "capability_id": analysis_request.get("capability_id"),
            "inputs": [
                {key: value for key, value in item.items() if key in {"sha256", "format"}}
                for item in analysis_request.get("inputs", [])
            ],
            "offset": analysis_request.get("offset"),
            "failure": (
                {"code": analysis_request["failure"].get("code")}
                if isinstance(analysis_request.get("failure"), dict)
                else None
            ),
        }
        limitations = body["limitations"]
        return {
            "format_version": manifest["format_version"],
            "evidence_id": manifest["evidence_id"],
            "analysis_sources": self._analysis_sources(manifest, logical_artifacts),
            "logical_sources": [
                {key: item[key] for key in ("sha256", "size_bytes", "format")}
                | {"source": evidence_source(evidence_id, "logical", index)}
                for index, item in enumerate(logical_artifacts)
            ],
            "body": {
                "evidence_kind": body["evidence_kind"],
                "capability_id": body["capability_id"],
                "provider": body["provider"],
                "inputs": [
                    {key: item[key] for key in ("sha256", "size_bytes", "format")}
                    for item in body["inputs"]
                ],
                "capture_request": safe_capture,
                "analysis_request": safe_analysis,
                "episode": body["episode"],
                "coverage": body["coverage"],
                "limitations": {
                    "count": len(limitations),
                    "sha256": hashlib.sha256(canonical_bytes(limitations)).hexdigest(),
                },
                "artifacts": [
                    {key: item[key] for key in ("sha256", "size_bytes", "format")}
                    | {"source": evidence_source(evidence_id, "artifact", index)}
                    for index, item in enumerate(body["artifacts"])
                ],
                "data_files": [
                    {key: item[key] for key in ("sha256", "size_bytes", "media_type")}
                    for item in body["data_files"]
                ],
            },
        }

    @staticmethod
    def logical_artifacts(body: Mapping[str, Any]) -> list[dict[str, Any]]:
        if "source_layout" in body:
            return cast(list[dict[str, Any]], body["source_layout"]["sources"])
        artifacts = body["artifacts"]
        logical = [item | {"is_directory": False} for item in artifacts if ":" not in item["role"]]
        bundles: dict[str, list[dict[str, Any]]] = {}
        for artifact in artifacts:
            if ":" in artifact["role"]:
                bundles.setdefault(artifact["role"].split(":", 1)[0], []).append(artifact)
        for role, members in bundles.items():
            digest = hashlib.sha256()
            for member in sorted(members, key=lambda item: Path(item["role"].split(":", 1)[1])):
                relative = member["role"].split(":", 1)[1]
                digest.update(relative.encode() + bytes.fromhex(member["sha256"]))
            logical.append(
                {
                    "role": role,
                    "sha256": digest.hexdigest(),
                    "size_bytes": sum(item["size_bytes"] for item in members),
                    "format": members[0]["format"],
                    "producer": members[0].get("producer"),
                    "is_directory": True,
                    "artifact_indices": [
                        index for index, item in enumerate(artifacts) if item in members
                    ],
                }
            )
        return logical

    @staticmethod
    def _analysis_sources(
        manifest: Mapping[str, Any],
        logical_artifacts: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        body = manifest["body"]
        if "source_layout" in body:
            return [
                evidence_source(manifest["evidence_id"], "logical", index)
                for index in body["source_layout"]["analysis_sources"]
            ]
        sources = []
        for item in body["analysis_request"].get("inputs", []):
            matching = [
                (index, artifact)
                for index, artifact in enumerate(logical_artifacts)
                if artifact["sha256"] == item["sha256"]
                and artifact["format"] == item["format"]
                and artifact.get("producer") == item.get("producer")
            ]
            if not matching:
                return []
            # Preserve identity when distinct sources happen to have identical bytes.
            index, _ = next(
                (match for match in matching if match[1]["role"] == item.get("role")),
                matching[0],
            )
            sources.append(evidence_source(manifest["evidence_id"], "logical", index))
        return sources

    @staticmethod
    def _safe_execution_projection(execution: Mapping[str, Any]) -> dict[str, Any]:
        oracle = execution.get("semantic_oracle")
        safe_oracle = None
        if isinstance(oracle, dict):
            safe_oracle = {key: oracle.get(key) for key in ("returncode", "status", "failure_code")}
        return {
            key: execution.get(key)
            for key in (
                "block",
                "returncode",
                "returncode_scope",
                "executable_sha256",
                "workload_returncode",
                "status",
                "failure_code",
                "missing_artifact_roles",
                "wall_time_ns",
                "containment",
            )
        } | {"semantic_oracle": safe_oracle}

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
            body = manifest["body"]
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

    def _publish_artifact(self, artifact: NativeArtifact) -> dict[str, Any]:
        if not artifact.path.is_file():
            raise RepositoryError(
                "UNSUPPORTED_FORMAT", "Only regular native artifact files can be preserved."
            )
        actual_digest, actual_size = sha256_file(artifact.path)
        if (actual_digest, actual_size) != (artifact.sha256, artifact.size_bytes):
            raise RepositoryError(
                "MISSING_OR_CHANGED_INPUT", f"Input changed before preservation: {artifact.path}"
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
                shutil.copyfile(artifact.path, stage / "payload")
                with (stage / "payload").open("rb") as stream:
                    os.fsync(stream.fileno())
                self._write_file(stage / "artifact.json", canonical_bytes(metadata))
                self._validate_artifact(stage, metadata)
                self._publish_directory(stage, destination)
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

    def _validate_repository(self) -> None:
        self._assert_no_symlink_path(self.root / "repository.json")
        try:
            metadata = json.loads((self.root / "repository.json").read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "repository.json is unreadable."
            ) from exc
        if metadata.get("format_version") != REPOSITORY_FORMAT:
            raise RepositoryError(
                "UNSUPPORTED_REPOSITORY_FORMAT", "The evidence repository format is unsupported."
            )
        if set(metadata) != {"format_version", "created_at"}:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "repository.json contains unsupported metadata."
            )
        try:
            created_at = datetime.fromisoformat(str(metadata["created_at"]))
        except (TypeError, ValueError) as exc:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "repository.json creation metadata is invalid."
            ) from exc
        if created_at.tzinfo is None:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "repository.json creation time lacks a timezone."
            )
        self._validate_repository_layout()

    def _validate_artifact(self, path: Path, expected: Mapping[str, Any]) -> None:
        self._assert_no_symlink_path(path)
        try:
            entries = {item.name for item in path.iterdir()}
            if entries != {"artifact.json", "payload"} or any(
                item.is_symlink() for item in path.iterdir()
            ):
                raise RepositoryError("REPOSITORY_CORRUPTION", "Artifact bundle layout is invalid.")
            metadata = json.loads((path / "artifact.json").read_bytes())
            digest, size = sha256_file(path / "payload")
        except RepositoryError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Artifact bundle is incomplete."
            ) from exc
        if metadata != expected or digest != expected["sha256"] or size != expected["size_bytes"]:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Artifact bundle digest mismatch.")

    def _validate_evidence(
        self, path: Path, expected: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self._assert_no_symlink_path(path)
        self._assert_no_symlink_path(path / "manifest.json")
        try:
            manifest = json.loads((path / "manifest.json").read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence manifest is unreadable."
            ) from exc
        if manifest.get("format_version") != REPOSITORY_FORMAT:
            raise RepositoryError(
                "UNSUPPORTED_REPOSITORY_FORMAT", "The evidence manifest format is unsupported."
            )
        if set(manifest) != {"format_version", "evidence_id", "body"}:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence manifest contains unsupported metadata."
            )
        body = manifest.get("body")
        if not isinstance(body, dict):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence manifest body is invalid.")
        self._validate_manifest_body(body)
        evidence_id = hashlib.sha256(canonical_bytes(body)).hexdigest()
        if evidence_id != manifest.get("evidence_id") or (
            expected is None and evidence_id != path.name
        ):
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence identity does not match its body."
            )
        if expected is not None and manifest != expected:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Existing evidence bundle differs.")
        for item in body["data_files"]:
            relative = item.get("path")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence data path is invalid.")
            try:
                data_path = path / relative
                self._assert_no_symlink_path(data_path)
                digest, size = sha256_file(data_path)
            except RepositoryError:
                raise
            except OSError as exc:
                raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence data is missing.") from exc
            if digest != item.get("sha256") or size != item.get("size_bytes"):
                raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence data digest mismatch.")
        artifact_entries = body["artifacts"]
        roles = [item.get("role") for item in artifact_entries if isinstance(item, dict)]
        if len(roles) != len(artifact_entries) or len(set(roles)) != len(roles):
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence artifact roles must be unique."
            )
        for item in artifact_entries:
            digest = item.get("sha256")
            size = item.get("size_bytes")
            role = item.get("role")
            format_name = item.get("format")
            producer = item.get("producer")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(role, str)
                or not role
                or not isinstance(format_name, str)
                or not format_name
                or (producer is not None and not isinstance(producer, str))
            ):
                raise RepositoryError(
                    "REPOSITORY_CORRUPTION", "Evidence artifact reference is invalid."
                )
            self._validate_artifact(
                self._artifact_path(digest),
                {
                    "format_version": REPOSITORY_FORMAT,
                    "sha256": digest,
                    "size_bytes": size,
                },
            )
        return cast(dict[str, Any], manifest)

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
    def _validate_manifest_body(body: Mapping[str, Any]) -> None:
        expected_keys = {
            "evidence_kind",
            "capability_id",
            "provider",
            "inputs",
            "capture_request",
            "analysis_request",
            "episode",
            "coverage",
            "limitations",
            "artifacts",
            "data_files",
        }
        if set(body) not in (expected_keys, expected_keys | {"source_layout"}):
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence manifest body has an invalid shape."
            )
        if not all(
            isinstance(body.get(name), str) and body[name]
            for name in ("evidence_kind", "capability_id")
        ):
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence manifest identity fields are invalid."
            )
        provider = body["provider"]
        if (
            not isinstance(provider, dict)
            or set(provider) != {"id", "version"}
            or not all(isinstance(provider.get(name), str) and provider[name] for name in provider)
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence provider identity is invalid.")
        inputs = body["inputs"]
        if not isinstance(inputs, list) or not inputs:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence inputs are invalid.")
        for item in inputs:
            if (
                not isinstance(item, dict)
                or set(item) != {"sha256", "size_bytes", "format", "role"}
                or not _is_digest(item.get("sha256"))
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] < 0
                or not isinstance(item.get("format"), str)
                or not item["format"]
                or not isinstance(item.get("role"), str)
                or not item["role"]
            ):
                raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence input is invalid.")
        EvidenceRepository._validate_capture_request(body["capture_request"])
        EvidenceRepository._validate_analysis_request(body["analysis_request"])
        episode = body["episode"]
        try:
            created_at = datetime.fromisoformat(episode["created_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence episode is invalid.") from exc
        if set(episode) != {"created_at"} or created_at.tzinfo is None:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence episode is invalid.")
        coverage = body["coverage"]
        if (
            not isinstance(coverage, dict)
            or set(coverage) != {"rows_returned", "rows_observed", "complete"}
            or type(coverage.get("rows_returned")) is not int
            or coverage["rows_returned"] < 0
            or type(coverage.get("rows_observed")) is not int
            or coverage["rows_observed"] < 0
            or type(coverage.get("complete")) is not bool
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence coverage is invalid.")
        limitations = body["limitations"]
        if not isinstance(limitations, list) or any(
            not isinstance(item, str) for item in limitations
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence limitations are invalid.")
        data_files = body["data_files"]
        if not isinstance(data_files, list) or not data_files:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence data files are invalid.")
        for item in data_files:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "sha256", "size_bytes", "media_type"}
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or not _is_digest(item.get("sha256"))
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] < 0
                or not isinstance(item.get("media_type"), str)
                or not item["media_type"]
            ):
                raise RepositoryError(
                    "REPOSITORY_CORRUPTION", "Evidence data file reference is invalid."
                )
        artifacts = body["artifacts"]
        if not isinstance(artifacts, list):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence artifacts are invalid.")
        if "source_layout" in body:
            EvidenceRepository._validate_source_layout(body)

    @staticmethod
    def _validate_source_layout(body: Mapping[str, Any]) -> None:
        layout = body["source_layout"]
        if (
            not isinstance(layout, dict)
            or set(layout) != {"sources", "analysis_sources"}
            or not isinstance(layout["sources"], list)
            or not isinstance(layout["analysis_sources"], list)
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence source layout is invalid.")
        artifacts = body["artifacts"]
        used: set[int] = set()
        roles: set[str] = set()
        for item in layout["sources"]:
            if (
                not isinstance(item, dict)
                or set(item)
                != {
                    "role",
                    "sha256",
                    "size_bytes",
                    "format",
                    "producer",
                    "is_directory",
                    "artifact_indices",
                }
                or not isinstance(item["role"], str)
                or not item["role"]
                or item["role"] in roles
                or not _is_digest(item["sha256"])
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] < 0
                or not isinstance(item["format"], str)
                or not item["format"]
                or (item["producer"] is not None and not isinstance(item["producer"], str))
                or type(item["is_directory"]) is not bool
                or not isinstance(item["artifact_indices"], list)
            ):
                raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence source layout is invalid.")
            roles.add(item["role"])
            members = []
            for index in item["artifact_indices"]:
                if type(index) is not int or not 0 <= index < len(artifacts) or index in used:
                    raise RepositoryError(
                        "REPOSITORY_CORRUPTION", "Evidence membership is invalid."
                    )
                used.add(index)
                member = artifacts[index]
                if (
                    not isinstance(member, dict)
                    or not isinstance(member.get("role"), str)
                    or not _is_digest(member.get("sha256"))
                    or type(member.get("size_bytes")) is not int
                    or member["size_bytes"] < 0
                    or member.get("format") != item["format"]
                    or member.get("producer") != item["producer"]
                ):
                    raise RepositoryError(
                        "REPOSITORY_CORRUPTION", "Evidence membership is invalid."
                    )
                members.append(member)
            if item["is_directory"]:
                prefix = item["role"] + ":"
                digest = hashlib.sha256()
                for member in sorted(
                    members, key=lambda member: Path(member["role"][len(prefix) :])
                ):
                    relative = member["role"][len(prefix) :]
                    path = Path(relative)
                    if (
                        not member["role"].startswith(prefix)
                        or path.is_absolute()
                        or ".." in path.parts
                        or not path.parts
                    ):
                        raise RepositoryError(
                            "REPOSITORY_CORRUPTION", "Evidence membership is invalid."
                        )
                    digest.update(relative.encode() + bytes.fromhex(member["sha256"]))
                expected_digest = digest.hexdigest()
            else:
                if len(members) != 1 or members[0]["role"] != item["role"]:
                    raise RepositoryError(
                        "REPOSITORY_CORRUPTION", "Evidence file membership is invalid."
                    )
                expected_digest = members[0]["sha256"]
            if item["sha256"] != expected_digest or item["size_bytes"] != sum(
                member["size_bytes"] for member in members
            ):
                raise RepositoryError(
                    "REPOSITORY_CORRUPTION", "Evidence source identity is invalid."
                )
        if used != set(range(len(artifacts))):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence membership is incomplete.")
        inputs = body["analysis_request"].get("inputs", [])
        indices = layout["analysis_sources"]
        if indices and len(indices) != len(inputs):
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence analysis source layout is invalid."
            )
        for index, input_item in zip(indices, inputs, strict=False):
            if type(index) is not int or not 0 <= index < len(layout["sources"]):
                raise RepositoryError(
                    "REPOSITORY_CORRUPTION", "Evidence analysis source layout is invalid."
                )
            source = layout["sources"][index]
            if any(
                source.get(key) != input_item.get(key) for key in ("sha256", "format", "producer")
            ):
                raise RepositoryError(
                    "REPOSITORY_CORRUPTION", "Evidence analysis source identity is invalid."
                )

    @staticmethod
    def _validate_capture_request(value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence capture request is invalid.")
        if set(value) != {"target", "mode", "experiment", "executions"}:
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence capture request is invalid.")
        target = value.get("target")
        executions = value.get("executions")
        try:
            CaptureTarget.model_validate(target)
            experiment = value.get("experiment")
            if experiment is not None:
                ExperimentDesign.model_validate(experiment)
        except (TypeError, ValueError) as error:
            raise RepositoryError(
                "REPOSITORY_CORRUPTION", "Evidence capture request is invalid."
            ) from error
        if value.get("mode") not in {"single", "experiment"} or not isinstance(executions, list):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence capture request is invalid.")
        for execution in executions:
            EvidenceRepository._validate_capture_execution(execution)

    @staticmethod
    def _validate_capture_execution(value: Any) -> None:
        fields = {
            "case",
            "block",
            "argv",
            "capture_argv",
            "cwd",
            "returncode",
            "status",
            "failure_code",
            "missing_artifact_roles",
            "semantic_oracle",
            "wall_time_ns",
            "containment",
            "limit",
        }
        attribution_fields = {"returncode_scope", "workload_returncode", "executable_sha256"}
        if not isinstance(value, dict) or set(value) not in (fields, fields | attribution_fields):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence capture request is invalid.")
        if attribution_fields <= set(value) and (
            value["returncode_scope"] not in {"workload", "collector"}
            or not isinstance(value["executable_sha256"], str)
            or re.fullmatch(LOWERCASE_SHA256_PATTERN, value["executable_sha256"]) is None
            or (
                value["workload_returncode"] is not None
                and type(value["workload_returncode"]) is not int
            )
            or (
                value["returncode_scope"] == "collector"
                and value["workload_returncode"] is not None
            )
            or (
                value["returncode_scope"] == "workload"
                and value["workload_returncode"] != value["returncode"]
            )
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Capture exit attribution is invalid.")
        argv_fields = (value.get("argv"), value.get("capture_argv"))
        missing_roles = value.get("missing_artifact_roles")
        scalar_valid = (
            isinstance(value.get("case"), str)
            and type(value.get("block")) is int
            and value["block"] >= 1
            and isinstance(value.get("cwd"), str)
            and Path(value["cwd"]).is_absolute()
            and (value.get("returncode") is None or type(value["returncode"]) is int)
            and value.get("status") in {"pending", "succeeded", "failed"}
            and (value.get("failure_code") is None or isinstance(value["failure_code"], str))
            and (value.get("wall_time_ns") is None or type(value["wall_time_ns"]) is int)
            and isinstance(value.get("containment"), str)
        )
        if (
            not scalar_valid
            or any(
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
                for argv in argv_fields
            )
            or not isinstance(missing_roles, list)
            or any(not isinstance(role, str) or not role for role in missing_roles)
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence capture request is invalid.")
        limit = value.get("limit")
        if limit is not None and (
            not isinstance(limit, dict)
            or set(limit)
            != {"kind", "configured", "observed", "unit", "observation_available", "recovery"}
            or not isinstance(limit.get("kind"), str)
            or not isinstance(limit.get("configured"), int | float)
            or isinstance(limit.get("configured"), bool)
            or (
                limit.get("observed") is not None
                and (
                    not isinstance(limit["observed"], int | float)
                    or isinstance(limit["observed"], bool)
                )
            )
            or not isinstance(limit.get("unit"), str)
            or type(limit.get("observation_available")) is not bool
            or not isinstance(limit.get("recovery"), str)
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence capture request is invalid.")
        oracle = value.get("semantic_oracle")
        if oracle is not None and (
            not isinstance(oracle, dict)
            or set(oracle) != {"argv", "returncode", "status", "failure_code"}
            or not isinstance(oracle.get("argv"), list)
            or not oracle["argv"]
            or any(
                not isinstance(item, str) or not item or "\x00" in item for item in oracle["argv"]
            )
            or (oracle.get("returncode") is not None and type(oracle["returncode"]) is not int)
            or oracle.get("status") not in {"passed", "failed"}
            or (
                oracle.get("failure_code") is not None
                and not isinstance(oracle["failure_code"], str)
            )
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence capture request is invalid.")

    @staticmethod
    def _validate_analysis_request(value: Any) -> None:
        if not isinstance(value, dict) or not isinstance(value.get("capability_id"), str):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence analysis request is invalid.")
        inputs = value.get("inputs", [])
        if not isinstance(inputs, list) or any(
            not isinstance(item, dict)
            or not _is_digest(item.get("sha256"))
            or not isinstance(item.get("format"), str)
            for item in inputs
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence analysis request is invalid.")
        offset = value.get("offset")
        if offset is not None and (type(offset) is not int or offset < 0):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence analysis request is invalid.")
        failure = value.get("failure")
        if failure is not None and (
            not isinstance(failure, dict) or not isinstance(failure.get("code"), str)
        ):
            raise RepositoryError("REPOSITORY_CORRUPTION", "Evidence analysis request is invalid.")

    @staticmethod
    def _matches(
        body: Mapping[str, Any],
        *,
        evidence_kind: str | None,
        capability_id: str | None,
        provider_id: str | None,
        input_sha256: str | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> bool:
        if evidence_kind is not None and body.get("evidence_kind") != evidence_kind:
            return False
        if capability_id is not None and body.get("capability_id") != capability_id:
            return False
        provider = body.get("provider", {})
        if provider_id is not None and provider.get("id") != provider_id:
            return False
        inputs = body.get("inputs", [])
        if input_sha256 is not None and all(item.get("sha256") != input_sha256 for item in inputs):
            return False
        try:
            created_at = datetime.fromisoformat(str(body["episode"]["created_at"]))
        except (KeyError, TypeError, ValueError):
            return False
        return not (
            (created_after is not None and created_at < created_after)
            or (created_before is not None and created_at > created_before)
        )

    @staticmethod
    def _summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
        body = manifest["body"]
        return {
            "evidence_id": manifest["evidence_id"],
            "uri": f"flameox://evidence/{manifest['evidence_id']}",
            "evidence_kind": body.get("evidence_kind"),
            "capability_id": body.get("capability_id"),
            "provider": body.get("provider"),
            "created_at": body.get("episode", {}).get("created_at"),
            "coverage": body.get("coverage"),
            "limitations": body.get("limitations", []),
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
