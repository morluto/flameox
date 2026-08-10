from __future__ import annotations

import json
import mimetypes
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import Field, StringConstraints

from flameox.application.environment import collect_environment
from flameox.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
    source_state_row,
)
from flameox.application.run_rows import run_row
from flameox.application.source import collect_partial_source_state
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import new_id
from flameox.domain.models import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    ImportRunManifest,
    RunManifest,
    Sensitivity,
    ValidationStatus,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactSnapshot, ArtifactStore, RunStore, StoredArtifact, Workspace

if TYPE_CHECKING:
    from flameox.application.otlp import OtlpExtractionResult

_MAX_BUNDLE_MEMBERS = 100


def _verify_declared_integrity(
    member: BundleMember,
    stored: StoredArtifact,
) -> None:
    """Verify imported content against provider-declared integrity.

    Closes the TOCTOU gap between provider preflight and registration:
    if the source changed between the provider's manifest declaration
    and the actual import, the ``StoredArtifact`` digest/size will not
    match and the bundle is rejected before any registration is
    constructed or committed.
    """
    if member.expected_byte_length is not None:
        actual = stored.content.byte_length
        if actual != member.expected_byte_length:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Bundle member {member.path!s} byte length mismatch: "
                f"expected {member.expected_byte_length}, got {actual}.",
            )
    if member.expected_sha256 is not None:
        expected_hex = member.expected_sha256.removeprefix("sha256:").lower()
        actual_hex = stored.content.integrity.sha256.lower()
        if actual_hex != expected_hex:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Bundle member {member.path!s} sha256 mismatch: "
                f"expected {expected_hex}, got {actual_hex}.",
            )


class ImportArtifactRequest(ContractModel):
    path: Path
    kind: ArtifactKind = ArtifactKind.COLLECTOR_METADATA
    media_type: str | None = None
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    role: str = "primary"
    producer: str | None = None
    producer_version: str | None = None
    allow_external_path: bool = False


class ImportResult(ContractModel):
    run: RunManifest
    artifact_id: str
    corpus_commit_id: str


class BundleMember(ContractModel):
    """A single file within a bounded multi-file import bundle.

    ``display_name`` overrides the default basename used in
    ``ArtifactRegistration``.  Multi-file producers like NVBench store
    relative paths (e.g. ``out.json-bin/0.bin``) in their JSON summaries;
    setting ``display_name`` to that relative path lets the extractor
    match sidecars by the exact key the producer declared.

    ``expected_byte_length`` and ``expected_sha256`` bind declared
    integrity from a provider manifest.  When set, the imported
    ``StoredArtifact`` is verified immediately after ``import_path``
    returns and before any registration is constructed, closing the
    TOCTOU gap between provider preflight and registration commit.
    """

    path: Path
    role: str = "primary"
    media_type: str | None = None
    display_name: (
        Annotated[str, StringConstraints(min_length=1, max_length=500, pattern=r"^[^\x00\r\n]+$")]
        | None
    ) = None
    expected_byte_length: Annotated[int, Field(ge=0)] | None = None
    expected_sha256: (
        Annotated[str, StringConstraints(pattern=r"^(?:sha256:)?[0-9a-fA-F]{64}$")] | None
    ) = None


class ImportBundleRequest(ContractModel):
    """A bounded multi-file import that preserves a primary artifact and its sidecars in one run.

    The bundle is capped at 100 total members (primary + sidecars) by a
    dedicated typed-provider limit independent of the generic single-file
    import quota.
    """

    primary: BundleMember
    sidecars: Annotated[tuple[BundleMember, ...], Field(max_length=99)] = ()
    kind: ArtifactKind = ArtifactKind.BENCHMARK_SAMPLES
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    producer: str | None = None
    producer_version: str | None = None
    allow_external_path: bool = False


class ImportBundleResult(ContractModel):
    run: RunManifest
    primary_artifact_id: str
    sidecar_artifact_ids: tuple[str, ...]
    corpus_commit_id: str


class ImportService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.artifacts = ArtifactStore(workspace)
        self.runs = RunStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    @contextmanager
    def _snapshot_provider_document(
        self,
        path: Path,
        *,
        allow_external_path: bool,
        max_bytes: int,
    ) -> Iterator[ArtifactSnapshot]:
        """Copy a provider primary through the canonical no-follow import boundary.

        Provider-specific parsers inspect the immutable returned payload rather
        than opening an untrusted source path themselves. The later bundle
        import remains bound to this snapshot's size and digest.
        """
        allowed_roots = [self.workspace.project_root]
        if allow_external_path:
            allowed_roots.append(path.absolute().parent)
        with self.artifacts.temporary_snapshot(
            path,
            allowed_roots=tuple(allowed_roots),
            max_bytes=min(max_bytes, self.workspace.config.capture.max_artifact_bytes),
        ) as snapshot:
            yield snapshot

    def import_artifact(self, request: ImportArtifactRequest) -> ImportResult:
        if (
            request.kind
            in {
                ArtifactKind.CORE_DUMP,
                ArtifactKind.SOURCE_SNAPSHOT,
                ArtifactKind.INFERENCE_REQUEST_TRACE,
            }
            and request.sensitivity is not Sensitivity.SENSITIVE
        ) or (
            request.kind is ArtifactKind.INFERENCE_RESULT
            and request.producer == "aiperf"
            and request.sensitivity is not Sensitivity.SENSITIVE
        ):
            raise DomainError(
                code=ErrorCode.SENSITIVE_ARTIFACT_REFUSED,
                message=f"{request.kind.value} artifacts must be marked sensitive.",
            )
        environment = collect_environment()
        source_state = collect_partial_source_state(self.workspace)
        run_id = new_id()
        initial = ImportRunManifest(
            run_id=run_id,
            capture_status=CaptureStatus.PENDING,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            collector="import",
        )
        self.runs.create(initial)
        allowed_roots = [self.workspace.project_root]
        if request.allow_external_path:
            allowed_roots.append(request.path.absolute().parent)
        try:
            stored = self.artifacts.import_path(
                request.path,
                allowed_roots=tuple(allowed_roots),
                max_bytes=self.workspace.config.capture.max_artifact_bytes,
            )
        except DomainError as error:
            failed = initial.validated_copy(
                update={
                    "revision": 1,
                    "capture_status": CaptureStatus.FAILED,
                    "limitations": (error.message,),
                }
            )
            self.runs.append(failed, expected_revision=0)
            error.run_id = run_id
            raise

        media_type = request.media_type or mimetypes.guess_type(request.path.name)[0]
        producer = request.producer or self._infer_producer(stored.payload_path, request.kind)
        registration = ArtifactRegistration(
            registration_id=new_id(),
            run_id=run_id,
            artifact_id=stored.content.artifact_id,
            display_name=request.path.name,
            media_type=media_type or "application/octet-stream",
            kind=request.kind,
            role=request.role,
            producer=producer,
            producer_version=request.producer_version,
            sensitivity=request.sensitivity,
        )
        registered = initial.validated_copy(
            update={
                "revision": 1,
                "capture_status": CaptureStatus.REGISTERED,
                "artifacts": (registration,),
            }
        )
        registered = self.runs.append(registered, expected_revision=0)
        published = self.publisher.publish_rows(
            {
                "runs": [run_row(registered)],
                "artifact_registrations": [
                    artifact_registration_row(
                        registration,
                        byte_length=stored.content.byte_length,
                    )
                ],
                "environments": [environment_row(environment)],
                "source_states": [source_state_row(source_state)],
            },
            publisher="flameox.import",
            publisher_version="1",
            input_run_ids=(run_id,),
            input_artifact_ids=(stored.content.artifact_id,),
        )
        return ImportResult(
            run=registered,
            artifact_id=stored.content.artifact_id,
            corpus_commit_id=published.commit.commit_id,
        )

    def _import_provider_bundle(self, request: ImportBundleRequest) -> ImportBundleResult:
        """Import a primary artifact and its bounded sidecars into a single run.

        Enforces a hard 100-member bundle cap independent of the generic
        single-file import quota, preflights every member (containment,
        regular non-linked file, duplicate paths, per-file and total byte
        limits) before any ``ArtifactStore`` import so registration is atomic,
        while allowing distinct bundle members to reference the same immutable content.
        """
        members = (request.primary, *request.sidecars)
        total_files = len(members)
        if total_files > _MAX_BUNDLE_MEMBERS:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Bundle import has {total_files} members but the typed provider "
                f"bundle limit is {_MAX_BUNDLE_MEMBERS}.",
            )
        max_bytes = self.workspace.config.capture.max_artifact_bytes
        max_total_bytes = self.workspace.config.storage.max_staging_bytes
        allowed_roots = [self.workspace.project_root]
        if request.allow_external_path:
            allowed_roots.append(request.primary.path.absolute().parent)
        self._preflight_bundle_members(members, allowed_roots, max_bytes, max_total_bytes)

        environment = collect_environment()
        source_state = collect_partial_source_state(self.workspace)
        run_id = new_id()
        initial = ImportRunManifest(
            run_id=run_id,
            capture_status=CaptureStatus.PENDING,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            collector="import",
        )
        self.runs.create(initial)

        registrations: list[ArtifactRegistration] = []
        artifact_ids: list[str] = []
        registration_rows: list[dict[str, object]] = []
        imported_total_bytes = 0
        try:
            for member in members:
                stored = self.artifacts.import_path(
                    member.path,
                    allowed_roots=tuple(allowed_roots),
                    max_bytes=max_bytes,
                )
                artifact_ids.append(stored.content.artifact_id)
                _verify_declared_integrity(member, stored)
                imported_total_bytes += stored.content.byte_length
                if imported_total_bytes > max_total_bytes:
                    raise DomainError(
                        ErrorCode.ARTIFACT_TOO_LARGE,
                        "Imported provider bundle exceeds the configured total staging limit.",
                    )
                media_type = member.media_type or mimetypes.guess_type(member.path.name)[0]
                registration = ArtifactRegistration(
                    registration_id=new_id(),
                    run_id=run_id,
                    artifact_id=stored.content.artifact_id,
                    display_name=member.display_name or member.path.name,
                    media_type=media_type or "application/octet-stream",
                    kind=request.kind,
                    role=member.role,
                    producer=request.producer or "flameox.import",
                    producer_version=request.producer_version,
                    sensitivity=request.sensitivity,
                )
                registrations.append(registration)
                registration_rows.append(
                    artifact_registration_row(
                        registration,
                        byte_length=stored.content.byte_length,
                    )
                )
        except DomainError as error:
            failed = initial.validated_copy(
                update={
                    "revision": 1,
                    "capture_status": CaptureStatus.FAILED,
                    "limitations": (error.message,),
                }
            )
            self.runs.append(failed, expected_revision=0)
            error.run_id = run_id
            raise

        registered = initial.validated_copy(
            update={
                "revision": 1,
                "capture_status": CaptureStatus.REGISTERED,
                "artifacts": tuple(registrations),
            }
        )
        registered = self.runs.append(registered, expected_revision=0)
        published = self.publisher.publish_rows(
            {
                "runs": [run_row(registered)],
                "artifact_registrations": registration_rows,
                "environments": [environment_row(environment)],
                "source_states": [source_state_row(source_state)],
            },
            publisher="flameox.import",
            publisher_version="1",
            input_run_ids=(run_id,),
            input_artifact_ids=tuple(artifact_ids),
        )
        return ImportBundleResult(
            run=registered,
            primary_artifact_id=artifact_ids[0],
            sidecar_artifact_ids=tuple(artifact_ids[1:]),
            corpus_commit_id=published.commit.commit_id,
        )

    @staticmethod
    def _preflight_bundle_members(
        members: tuple[BundleMember, ...],
        allowed_roots: list[Path],
        max_bytes: int,
        max_total_bytes: int,
    ) -> None:
        """Validate every bundle member before any artifact import.

        Checks containment within allowed roots, regular non-linked file
        status (no symlinks, no hard links), duplicate path detection,
        per-file byte limits (``max_bytes`` = ``max_artifact_bytes``), and
        the aggregate bundle byte limit (``max_total_bytes`` =
        ``max_staging_bytes``).  All checks run before any
        ``ArtifactStore.import_path`` call so that a preflight failure
        leaves no partially-imported artifacts.  The per-artifact
        ``StorageQuota`` enforcement still applies during import.
        """
        seen_paths: set[Path] = set()
        seen_display_names: set[str] = set()
        total_bytes = 0
        for member in members:
            source = member.path.absolute()
            resolved = source.resolve()
            if resolved in seen_paths:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Bundle member {member.path!s} is a duplicate path.",
                )
            seen_paths.add(resolved)
            display_name = member.display_name or member.path.name
            if display_name in seen_display_names:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Bundle member display name {display_name!r} is duplicated.",
                )
            seen_display_names.add(display_name)
            contained = False
            for root in allowed_roots:
                try:
                    relative = resolved.relative_to(root.resolve())
                except ValueError:
                    continue
                if relative.parts:
                    contained = True
                    break
            if not contained:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Bundle member {member.path!s} is outside the allowed roots.",
                )
            try:
                stat_result = source.lstat()
            except OSError as exc:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Bundle member {member.path!s} could not be stat'd.",
                ) from exc
            if not stat.S_ISREG(stat_result.st_mode):
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Bundle member {member.path!s} is not a regular file.",
                )
            if stat_result.st_nlink != 1:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Bundle member {member.path!s} is hard-linked and cannot be imported.",
                )
            if stat_result.st_size > max_bytes:
                raise DomainError(
                    ErrorCode.ARTIFACT_TOO_LARGE,
                    f"Bundle member {member.path!s} exceeds the {max_bytes}-byte limit.",
                )
            total_bytes += stat_result.st_size
        if total_bytes > max_total_bytes:
            raise DomainError(
                ErrorCode.ARTIFACT_TOO_LARGE,
                f"Bundle total {total_bytes} bytes exceeds the "
                f"{max_total_bytes}-byte staging limit.",
            )

    def extract_otlp_trace(
        self, run_id: str, artifact_id: str | None = None
    ) -> OtlpExtractionResult:
        from flameox.application.otlp import OtlpTraceService

        return OtlpTraceService(self.workspace).extract_otlp_trace(run_id, artifact_id)

    @staticmethod
    def _infer_producer(path: Path, kind: ArtifactKind) -> str:
        """Recover common producer identity before analysis routing loses it."""
        if kind is ArtifactKind.OTLP_TRACE:
            return "opentelemetry"
        if kind is not ArtifactKind.EXECUTION_TRACE:
            return "flameox.import"
        try:
            with path.open("rb") as stream:
                raw = stream.read(8 * 1024 * 1024)
        except OSError:
            return "flameox.import"
        lowered = raw.lower()
        if any(
            marker in lowered
            for marker in (
                b'"cpu_op"',
                b'"cuda_runtime"',
                b'"profilerstep',
                b"torch-compiled region",
            )
        ):
            return "torch.profiler"
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return "flameox.import"
        if not isinstance(payload, dict):
            return "flameox.import"
        events = payload.get("traceEvents")
        if not isinstance(events, list):
            return "flameox.import"
        for event in events[:10_000]:
            if not isinstance(event, dict):
                continue
            category = str(event.get("cat", "")).casefold()
            name = str(event.get("name", "")).casefold()
            if (
                "cpu_op" in category
                or "cuda_runtime" in category
                or "profilerstep" in name
                or "torch-compiled region" in name
            ):
                return "torch.profiler"
        return "flameox.import"
