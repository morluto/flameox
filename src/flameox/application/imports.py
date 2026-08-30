from __future__ import annotations

import json
import mimetypes
import stat
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import ijson
from ijson import IncompleteJSONError, JSONError
from pydantic import Field, StringConstraints

from flameox.application.environment import collect_environment
from flameox.application.projections import ProjectionCoordinator
from flameox.application.source import collect_partial_source_state
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import new_id
from flameox.domain.models import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    EnvironmentRecord,
    ImportRunManifest,
    RunManifest,
    RunSemantics,
    Sensitivity,
    SourceState,
    ValidationStatus,
)
from flameox.models import ContractModel
from flameox.storage import ArtifactSnapshot, ArtifactStore, RunStore, StoredArtifact, Workspace

if TYPE_CHECKING:
    from flameox.application.otlp import OtlpExtractionResult

_MAX_BUNDLE_MEMBERS = 100
_PYSPY_PROFILE_MAX_BYTES = 64 * 1024 * 1024
_PYSPY_PROFILE_MAX_EVENTS = 1_000_000
_PYSPY_PROFILE_MAX_THREADS = 4_096
_PYSPY_PROFILE_MAX_STACK_DEPTH = 4_096
_PYSPY_PROFILE_MAX_TEXT_LENGTH = 4_096


class ImportProfile(StrEnum):
    PYSPY_CHROMETRACE = "py-spy-chrometrace"


def _verify_declared_snapshot(member: BundleMember, snapshot: ArtifactSnapshot) -> None:
    if (
        member.expected_byte_length is not None
        and snapshot.byte_length != member.expected_byte_length
    ):
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            f"Bundle member {member.path!s} byte length mismatch: "
            f"expected {member.expected_byte_length}, got {snapshot.byte_length}.",
        )
    if member.expected_sha256 is not None:
        expected_hex = member.expected_sha256.removeprefix("sha256:").lower()
        if snapshot.sha256.lower() != expected_hex:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Bundle member {member.path!s} sha256 mismatch: "
                f"expected {expected_hex}, got {snapshot.sha256.lower()}.",
            )


class ImportArtifactRequest(ContractModel):
    path: Path
    kind: ArtifactKind = ArtifactKind.COLLECTOR_METADATA
    media_type: str | None = None
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    role: str = "primary"
    producer: str | None = None
    producer_version: str | None = None
    profile: ImportProfile | None = None
    allow_external_path: bool = False


class ImportDescriptorRequest(ContractModel):
    """Internal import intent for an exact file descriptor admitted by its producer."""

    descriptor: Annotated[int, Field(ge=0, exclude=True)]
    display_name: Annotated[
        str,
        StringConstraints(min_length=1, max_length=255, pattern=r"^[^/\\\x00\r\n]+$"),
    ]
    kind: ArtifactKind = ArtifactKind.COLLECTOR_METADATA
    media_type: str | None = None
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    role: str = "primary"
    producer: str | None = None
    producer_version: str | None = None


class QualifyArtifactImportRequest(ContractModel):
    run_id: str
    artifact_id: str
    profile: ImportProfile


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
    sidecars: tuple[BundleMember, ...] = ()
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
        self.projections = ProjectionCoordinator(workspace)

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

    @contextmanager
    def snapshot_provider_document(
        self,
        path: Path,
        *,
        allow_external_path: bool,
        max_bytes: int,
    ) -> Iterator[ArtifactSnapshot]:
        """Provide an immutable provider snapshot for a typed import adapter."""
        with self._snapshot_provider_document(
            path,
            allow_external_path=allow_external_path,
            max_bytes=max_bytes,
        ) as snapshot:
            yield snapshot

    def import_snapshot(
        self,
        snapshot: ArtifactSnapshot,
        *,
        kind: ArtifactKind,
        sensitivity: Sensitivity,
        display_name: str,
        media_type: str | None,
        role: str,
        producer: str | None,
        producer_version: str | None,
        semantics: RunSemantics,
        limitations: tuple[str, ...],
        source_state: SourceState,
    ) -> ImportResult:
        """Register one validated immutable provider payload with typed run semantics."""
        return self._import_profiled_snapshot(
            snapshot,
            kind=kind,
            sensitivity=sensitivity,
            display_name=display_name,
            media_type=media_type,
            role=role,
            producer=producer,
            producer_version=producer_version,
            semantics=semantics,
            limitations=limitations,
            terminal_error=None,
            source_state=source_state,
        )

    def import_artifact(self, request: ImportArtifactRequest) -> ImportResult:
        self._validate_import_sensitivity(
            kind=request.kind,
            sensitivity=request.sensitivity,
            producer=request.producer,
        )
        allowed_roots = [self.workspace.project_root]
        if request.allow_external_path:
            allowed_roots.append(request.path.absolute().parent)
        if request.profile is not None:
            with self.artifacts.temporary_snapshot(
                request.path,
                allowed_roots=tuple(allowed_roots),
                max_bytes=min(
                    self.workspace.config.capture.max_artifact_bytes,
                    _PYSPY_PROFILE_MAX_BYTES,
                ),
            ) as snapshot:
                profile_error: DomainError | None = None
                try:
                    semantics, limitations = self._validate_pyspy_import_profile(
                        snapshot.payload_path,
                        kind=request.kind,
                        media_type=request.media_type or mimetypes.guess_type(request.path.name)[0],
                        producer=request.producer,
                        producer_version=request.producer_version,
                    )
                except DomainError as error:
                    profile_error = error
                    semantics = RunSemantics(
                        origin="import",
                        adapter="import",
                        configuration={"attempted_import_profile": request.profile.value},
                        unavailable_fields=("scope",),
                    )
                    limitations = (error.message,)
                return self._import_profiled_snapshot(
                    snapshot,
                    kind=request.kind,
                    sensitivity=request.sensitivity,
                    display_name=request.path.name,
                    media_type=request.media_type,
                    role=request.role,
                    producer=request.producer,
                    producer_version=request.producer_version,
                    semantics=semantics,
                    limitations=limitations,
                    terminal_error=profile_error,
                )
        return self._import_one(
            kind=request.kind,
            sensitivity=request.sensitivity,
            display_name=request.path.name,
            media_type=request.media_type,
            role=request.role,
            producer=request.producer,
            producer_version=request.producer_version,
            store=lambda: self.artifacts.import_path(
                request.path,
                allowed_roots=tuple(allowed_roots),
                max_bytes=self.workspace.config.capture.max_artifact_bytes,
            ),
        )

    def import_descriptor(self, request: ImportDescriptorRequest) -> ImportResult:
        """Register content from an exact descriptor without resolving its source path again."""

        self._validate_import_sensitivity(
            kind=request.kind,
            sensitivity=request.sensitivity,
            producer=request.producer,
        )
        return self._import_one(
            kind=request.kind,
            sensitivity=request.sensitivity,
            display_name=request.display_name,
            media_type=request.media_type,
            role=request.role,
            producer=request.producer,
            producer_version=request.producer_version,
            store=lambda: self.artifacts.import_descriptor(
                request.descriptor,
                display_name=request.display_name,
                max_bytes=self.workspace.config.capture.max_artifact_bytes,
            ),
        )

    def qualify_artifact(self, request: QualifyArtifactImportRequest) -> ImportResult:
        source_run = self.runs.read(request.run_id)
        registrations = [
            item for item in source_run.artifacts if item.artifact_id == request.artifact_id
        ]
        if len(registrations) != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_NOT_FOUND,
                "The source run does not own the requested artifact.",
                run_id=request.run_id,
                details={"artifact_id": request.artifact_id},
            )
        source = registrations[0]
        stored = self.artifacts.get(source.artifact_id)
        profile_error: DomainError | None = None
        try:
            semantics, limitations = self._validate_pyspy_import_profile(
                stored.payload_path,
                kind=source.kind,
                media_type=source.media_type,
                producer=source.producer,
                producer_version=source.producer_version,
            )
        except DomainError as error:
            profile_error = error
            semantics = RunSemantics(
                origin="import",
                adapter="import",
                configuration={"attempted_import_profile": request.profile.value},
                unavailable_fields=("scope",),
            )
            limitations = (error.message,)
        return self._record_profiled_artifact(
            stored,
            kind=source.kind,
            sensitivity=source.sensitivity,
            display_name=source.display_name,
            media_type=source.media_type,
            role=source.role,
            producer=source.producer,
            producer_version=source.producer_version,
            semantics=semantics,
            limitations=limitations,
            terminal_error=profile_error,
        )

    @staticmethod
    def _validate_import_sensitivity(
        *,
        kind: ArtifactKind,
        sensitivity: Sensitivity,
        producer: str | None,
    ) -> None:
        if (
            kind
            in {
                ArtifactKind.CORE_DUMP,
                ArtifactKind.SOURCE_SNAPSHOT,
                ArtifactKind.INFERENCE_REQUEST_TRACE,
                ArtifactKind.METAL_TRACE,
            }
            and sensitivity is not Sensitivity.SENSITIVE
        ) or (
            kind is ArtifactKind.INFERENCE_RESULT
            and producer == "aiperf"
            and sensitivity is not Sensitivity.SENSITIVE
        ):
            raise DomainError(
                code=ErrorCode.SENSITIVE_ARTIFACT_REFUSED,
                message=f"{kind.value} artifacts must be marked sensitive.",
            )

    def _import_one(
        self,
        *,
        kind: ArtifactKind,
        sensitivity: Sensitivity,
        display_name: str,
        media_type: str | None,
        role: str,
        producer: str | None,
        producer_version: str | None,
        store: Callable[[], StoredArtifact],
    ) -> ImportResult:
        environment = collect_environment()
        source_state = collect_partial_source_state(self.workspace)
        run_id = new_id()
        initial = ImportRunManifest(
            run_id=run_id,
            capture_status=CaptureStatus.PENDING,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            semantics=RunSemantics.unavailable(origin="import", adapter="import"),
        )
        self.runs.create(initial)
        try:
            stored = store()
        except DomainError as error:
            failed = initial.validated_copy(
                update={
                    "revision": 1,
                    "capture_status": CaptureStatus.FAILED,
                    "limitations": (error.message,),
                }
            )
            try:
                self.projections.append_run(
                    failed,
                    expected_revision=0,
                    environment=environment,
                    source_state=source_state,
                )
            except Exception as projection_error:
                error.add_note(
                    "The failed import run has a durable projection recovery record: "
                    f"{type(projection_error).__name__}."
                )
            error.run_id = run_id
            raise

        resolved_media_type = media_type or mimetypes.guess_type(display_name)[0]
        resolved_producer = producer or self._infer_producer(stored.payload_path, kind)
        registration = ArtifactRegistration(
            registration_id=new_id(),
            run_id=run_id,
            artifact_id=stored.content.artifact_id,
            display_name=display_name,
            media_type=resolved_media_type or "application/octet-stream",
            kind=kind,
            role=role,
            producer=resolved_producer,
            producer_version=producer_version,
            sensitivity=sensitivity,
        )
        registered = initial.validated_copy(
            update={
                "revision": 1,
                "capture_status": CaptureStatus.REGISTERED,
                "artifacts": (registration,),
            }
        )
        projected = self.projections.append_run(
            registered,
            expected_revision=0,
            environment=environment,
            source_state=source_state,
        )
        return ImportResult(
            run=projected.run,
            artifact_id=stored.content.artifact_id,
            corpus_commit_id=projected.publication.commit.commit_id,
        )

    def _import_profiled_snapshot(
        self,
        snapshot: ArtifactSnapshot,
        *,
        kind: ArtifactKind,
        sensitivity: Sensitivity,
        display_name: str,
        media_type: str | None,
        role: str,
        producer: str | None,
        producer_version: str | None,
        semantics: RunSemantics,
        limitations: tuple[str, ...],
        terminal_error: DomainError | None,
        source_state: SourceState | None = None,
    ) -> ImportResult:
        """Publish immutable bytes before assigning them semantic ownership."""

        environment = collect_environment()
        source_state = source_state or collect_partial_source_state(self.workspace)
        run_id = new_id()
        try:
            stored = self.artifacts.import_snapshot(snapshot, display_name=display_name)
        except DomainError as error:
            failed = ImportRunManifest(
                run_id=run_id,
                capture_status=CaptureStatus.FAILED,
                validation_status=ValidationStatus.NOT_REQUESTED,
                environment_id=environment.environment_id,
                source_state_id=source_state.source_state_id,
                semantics=semantics,
                limitations=(error.message,),
            )
            try:
                self.projections.create_run(
                    failed,
                    environment=environment,
                    source_state=source_state,
                )
            except Exception as projection_error:
                error.add_note(
                    "The failed import run has a durable projection recovery record: "
                    f"{type(projection_error).__name__}."
                )
            error.run_id = run_id
            raise
        return self._record_profiled_artifact(
            stored,
            kind=kind,
            sensitivity=sensitivity,
            display_name=display_name,
            media_type=media_type,
            role=role,
            producer=producer,
            producer_version=producer_version,
            semantics=semantics,
            limitations=limitations,
            terminal_error=terminal_error,
            environment=environment,
            source_state=source_state,
            run_id=run_id,
        )

    def _record_profiled_artifact(
        self,
        stored: StoredArtifact,
        *,
        kind: ArtifactKind,
        sensitivity: Sensitivity,
        display_name: str,
        media_type: str | None,
        role: str,
        producer: str | None,
        producer_version: str | None,
        semantics: RunSemantics,
        limitations: tuple[str, ...],
        terminal_error: DomainError | None,
        environment: EnvironmentRecord | None = None,
        source_state: SourceState | None = None,
        run_id: str | None = None,
    ) -> ImportResult:
        environment = environment or collect_environment()
        source_state = source_state or collect_partial_source_state(self.workspace)
        run_id = run_id or new_id()
        registration = ArtifactRegistration(
            registration_id=new_id(),
            run_id=run_id,
            artifact_id=stored.content.artifact_id,
            display_name=display_name,
            media_type=media_type
            or mimetypes.guess_type(display_name)[0]
            or "application/octet-stream",
            kind=kind,
            role=role,
            producer=producer or self._infer_producer(stored.payload_path, kind),
            producer_version=producer_version,
            sensitivity=sensitivity,
        )
        terminal = ImportRunManifest(
            run_id=run_id,
            capture_status=(
                CaptureStatus.FAILED if terminal_error is not None else CaptureStatus.REGISTERED
            ),
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            semantics=semantics,
            artifacts=(registration,),
            limitations=limitations,
        )
        projected = self.projections.create_run(
            terminal,
            environment=environment,
            source_state=source_state,
        )
        if terminal_error is not None:
            terminal_error.run_id = run_id
            raise terminal_error
        return ImportResult(
            run=projected.run,
            artifact_id=stored.content.artifact_id,
            corpus_commit_id=projected.publication.commit.commit_id,
        )

    def _validate_pyspy_import_profile(
        self,
        path: Path,
        *,
        kind: ArtifactKind,
        media_type: str | None,
        producer: str | None,
        producer_version: str | None,
    ) -> tuple[RunSemantics, tuple[str, ...]]:
        if kind is not ArtifactKind.SAMPLE_PROFILE:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "The py-spy-chrometrace profile requires kind='sample_profile'.",
            )
        if media_type not in {None, "application/json"}:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "The py-spy-chrometrace profile requires JSON media type.",
            )
        if producer not in {None, "py-spy"}:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "The py-spy-chrometrace profile conflicts with the declared producer.",
            )
        self._validate_pyspy_chrometrace(path)
        if producer_version is None:
            limitations = ("Imported py-spy producer version is unavailable.",)
        else:
            limitations = (
                f"Imported py-spy {producer_version} is caller-declared and not independently "
                "verified.",
            )
        return (
            RunSemantics(
                origin="import",
                adapter="py-spy",
                configuration={"import_profile": ImportProfile.PYSPY_CHROMETRACE.value},
                unavailable_fields=("adapter_version", "scope"),
            ),
            limitations,
        )

    def _validate_pyspy_chrometrace(self, path: Path) -> None:
        maximum = min(
            self.workspace.config.storage.max_rows_per_generation,
            _PYSPY_PROFILE_MAX_EVENTS,
        )
        count = 0
        stacks: dict[tuple[int, int], list[tuple[str, str, int | None]]] = {}
        last_timestamp: dict[tuple[int, int], int] = {}
        try:
            with path.open("rb") as stream:
                for count, event in enumerate(ijson.items(stream, "item"), start=1):
                    if count > maximum:
                        raise DomainError(
                            ErrorCode.QUERY_BUDGET_EXCEEDED,
                            "The py-spy Chrome trace exceeds the import-profile event limit.",
                        )
                    if not isinstance(event, dict):
                        raise ValueError("trace event is not an object")
                    args = event.get("args")
                    pid = event.get("pid")
                    tid = event.get("tid")
                    timestamp = event.get("ts")
                    line = args.get("line") if isinstance(args, dict) else None
                    if (
                        event.get("cat") != "py-spy"
                        or event.get("ph") not in {"B", "E"}
                        or not isinstance(event.get("name"), str)
                        or not event["name"]
                        or type(pid) is not int
                        or pid < 0
                        or type(tid) is not int
                        or tid < 0
                        or type(timestamp) is not int
                        or timestamp < 0
                        or not isinstance(args, dict)
                        or not isinstance(args.get("filename"), str)
                        or (line is not None and (type(line) is not int or line < 0))
                    ):
                        raise ValueError("trace event does not match py-spy Chrome trace shape")
                    thread = (pid, tid)
                    if (
                        thread not in last_timestamp
                        and len(last_timestamp) >= _PYSPY_PROFILE_MAX_THREADS
                    ):
                        raise DomainError(
                            ErrorCode.QUERY_BUDGET_EXCEEDED,
                            "The py-spy Chrome trace exceeds the import-profile thread limit.",
                        )
                    if timestamp < last_timestamp.get(thread, 0):
                        raise ValueError("trace timestamps are not monotonic within a thread")
                    last_timestamp[thread] = timestamp
                    frame = (event["name"], args["filename"], line)
                    if (
                        len(frame[0]) > _PYSPY_PROFILE_MAX_TEXT_LENGTH
                        or len(frame[1]) > _PYSPY_PROFILE_MAX_TEXT_LENGTH
                    ):
                        raise DomainError(
                            ErrorCode.QUERY_BUDGET_EXCEEDED,
                            "The py-spy Chrome trace exceeds the import-profile text limit.",
                        )
                    stack = stacks.setdefault(thread, [])
                    if event["ph"] == "B":
                        if len(stack) >= _PYSPY_PROFILE_MAX_STACK_DEPTH:
                            raise DomainError(
                                ErrorCode.QUERY_BUDGET_EXCEEDED,
                                "The py-spy Chrome trace exceeds the import-profile stack limit.",
                            )
                        stack.append(frame)
                    elif not stack or stack.pop() != frame:
                        raise ValueError("trace events do not form balanced per-thread stacks")
                    elif not stack:
                        stacks.pop(thread)
        except DomainError:
            raise
        except (OSError, IncompleteJSONError, JSONError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact does not match the py-spy Chrome trace profile.",
                remediation=("Import it without a profile to preserve only generic semantics.",),
            ) from exc
        if count == 0:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The artifact does not contain any py-spy Chrome trace events.",
                remediation=("Import it without a profile to preserve only generic semantics.",),
            )
        if any(stacks.values()):
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The py-spy Chrome trace contains unclosed stack frames.",
                remediation=("Import it without a profile to preserve only generic semantics.",),
            )

    def import_provider_bundle(self, request: ImportBundleRequest) -> ImportBundleResult:
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
            semantics=RunSemantics.unavailable(origin="import", adapter="import"),
        )
        self.runs.create(initial)

        registrations: list[ArtifactRegistration] = []
        artifact_ids: list[str] = []
        imported_total_bytes = 0
        try:
            with ExitStack() as snapshots:
                verified: list[tuple[BundleMember, ArtifactSnapshot]] = []
                for member in members:
                    snapshot = snapshots.enter_context(
                        self.artifacts.temporary_snapshot(
                            member.path,
                            allowed_roots=tuple(allowed_roots),
                            max_bytes=max_bytes,
                        )
                    )
                    _verify_declared_snapshot(member, snapshot)
                    imported_total_bytes += snapshot.byte_length
                    if imported_total_bytes > max_total_bytes:
                        raise DomainError(
                            ErrorCode.ARTIFACT_TOO_LARGE,
                            "Imported provider bundle exceeds the configured total staging limit.",
                        )
                    verified.append((member, snapshot))

                for member, snapshot in verified:
                    display_name = member.display_name or member.path.name
                    stored = self.artifacts.import_snapshot(
                        snapshot,
                        display_name=display_name,
                    )
                    artifact_ids.append(stored.content.artifact_id)
                    media_type = member.media_type or mimetypes.guess_type(member.path.name)[0]
                    registration = ArtifactRegistration(
                        registration_id=new_id(),
                        run_id=run_id,
                        artifact_id=stored.content.artifact_id,
                        display_name=display_name,
                        media_type=media_type or "application/octet-stream",
                        kind=request.kind,
                        role=member.role,
                        producer=request.producer or "flameox.import",
                        producer_version=request.producer_version,
                        sensitivity=request.sensitivity,
                    )
                    registrations.append(registration)
        except DomainError as error:
            failed = initial.validated_copy(
                update={
                    "revision": 1,
                    "capture_status": (
                        CaptureStatus.REGISTERED if registrations else CaptureStatus.FAILED
                    ),
                    "artifacts": tuple(registrations),
                    "limitations": (error.message,),
                }
            )
            projection_error: Exception | None = None
            try:
                self.projections.append_run(
                    failed,
                    expected_revision=0,
                    environment=environment,
                    source_state=source_state,
                )
            except Exception as caught:
                projection_error = caught
            if registrations:
                error.details = {
                    **error.details,
                    "partial_artifact_ids": tuple(artifact_ids),
                }
            if projection_error is not None:
                error.add_note(
                    "The failed bundle import has a durable projection recovery record: "
                    f"{type(projection_error).__name__}."
                )
            error.run_id = run_id
            raise

        registered = initial.validated_copy(
            update={
                "revision": 1,
                "capture_status": CaptureStatus.REGISTERED,
                "artifacts": tuple(registrations),
            }
        )
        projected = self.projections.append_run(
            registered,
            expected_revision=0,
            environment=environment,
            source_state=source_state,
        )
        return ImportBundleResult(
            run=projected.run,
            primary_artifact_id=artifact_ids[0],
            sidecar_artifact_ids=tuple(artifact_ids[1:]),
            corpus_commit_id=projected.publication.commit.commit_id,
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
