from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from flameox.adapters.kernel_build import (
    KernelBuildArtifact,
    KernelBuildArtifactGroup,
    KernelBuildManifest,
    KernelBuildProducer,
)
from flameox.application.imports import (
    BundleMember,
    ImportBundleRequest,
    ImportService,
)
from flameox.application.pipelines import (
    ArtifactPipelineService,
    PipelineExtractorProfile,
    PipelineStageDeclaration,
    PipelineStageStatus,
    RegisteredPipelineStageDeclaration,
    RegisterPipelineRequest,
)
from flameox.atomic import atomic_write_json
from flameox.domain import (
    AcceleratorIdentityFacet,
    AcceleratorIdentityStatus,
    ArtifactKind,
    CompilerIdentity,
    CompilerTarget,
    CompilerTargetIdentity,
    DomainError,
    ErrorCode,
    RunManifest,
    Sensitivity,
    compiler_identity_id,
    compiler_target_identity_id,
)
from flameox.models import ContractModel
from flameox.storage import Workspace

_MAX_KERNEL_BUILD_MEMBERS = 100
_MAX_KERNEL_BUILD_MANIFEST_BYTES = 1024 * 1024
_MAX_TRITON_COMPILER_EVENTS = 128
_MAX_TRITON_COMPILER_EVENT_BYTES = 64 * 1024
_MAX_PTX_METADATA_BYTES = 1024 * 1024
_PTX_TARGET = re.compile(rb"(?m)^\s*\.target\s+(sm_[0-9]{2,3}a?)\b")
_PTX_VERSION = re.compile(rb"(?m)^\s*\.version\s+([0-9]+\.[0-9]+)\b")

# Triton stage extensions: NVIDIA (ttir/ttgir/llir/ptx/cubin/sass) and AMD
# (amdgcn/hsaco). JSON and .metadata are preserved when declared but are never
# compiler stages.
_TRITON_EXTENSION_MAP: dict[str, tuple[str, str, str]] = {
    ".ttir": ("ttir", "triton-ttir", "text/plain"),
    ".ttgir": ("ttgir", "triton-ttgir", "text/plain"),
    ".llir": ("llir", "llvm-ir", "text/plain"),
    ".ptx": ("ptx", "nvidia-ptx", "text/plain"),
    ".cubin": ("cubin", "nvidia-cubin", "application/octet-stream"),
    ".sass": ("sass", "nvidia-sass", "text/plain"),
    ".amdgcn": ("amdgcn", "amd-amdgcn", "text/plain"),
    ".hsaco": ("hsaco", "amd-hsaco", "application/octet-stream"),
    ".metadata": ("metadata", "triton-metadata", "text/plain"),
    ".json": ("metadata", "triton-metadata", "application/json"),
}

# CuTe DSL CUTE_DSL_KEEP token → retained file extensions.
# Official tokens: ir, ir-debug, ptx, cubin, sass, llvm, all.
_CUTE_TOKEN_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "ir": (".mlir", ".cute_dsl_ir"),
    "ir-debug": (".mlir", ".cute_dsl_ir"),
    "ptx": (".ptx",),
    "cubin": (".cubin",),
    "sass": (".sass",),
    "llvm": (".ll", ".llvm"),
    "all": (
        ".mlir",
        ".cute_dsl_ir",
        ".ptx",
        ".cubin",
        ".sass",
        ".ll",
        ".llvm",
    ),
}

_CUTE_EXTENSION_INFO: dict[str, tuple[str, str, str]] = {
    ".mlir": ("cute_dsl_ir", "cute-dsl-mlir", "text/plain"),
    ".cute_dsl_ir": ("cute_dsl_ir", "cute-dsl-ir", "text/plain"),
    ".ptx": ("ptx", "nvidia-ptx", "text/plain"),
    ".cubin": ("cubin", "nvidia-cubin", "application/octet-stream"),
    ".sass": ("sass", "nvidia-sass", "text/plain"),
    ".ll": ("llvm", "llvm-ir", "text/plain"),
    ".llvm": ("llvm", "llvm-ir", "text/plain"),
}

# The reproducer lives outside dump_dir, in output_root.
_REPRODUCER_MEDIA_TYPE = "text/plain"

_STAGE_PRIORITY: dict[str, int] = {
    ".ttir": 0,
    ".ttgir": 1,
    ".mlir": 0,
    ".cute_dsl_ir": 0,
    ".mlir.debug": 0,
    ".cute_dsl_ir.debug": 0,
    ".llir": 2,
    ".ll": 2,
    ".llvm": 2,
    ".ptx": 3,
    ".amdgcn": 3,
    ".cubin": 4,
    ".hsaco": 4,
    ".sass": 5,
}

_LINEAGE_EXTENSIONS = frozenset(_STAGE_PRIORITY)


@dataclass(frozen=True, slots=True)
class KernelBuildInventoryEntry:
    path: Path
    relative_path: str
    byte_length: int
    sha256: str
    media_type: str


@dataclass(frozen=True, slots=True)
class KernelBuildInventory:
    entries: tuple[KernelBuildInventoryEntry, ...]
    limitations: tuple[str, ...]
    dump_dir: Path | None


class _TritonCompilerEvent(ContractModel):
    cache_hit: bool
    target: CompilerTarget
    triton_version: str


def qualify_triton_compiler_target(
    *,
    events_path: Path,
    native_paths: tuple[Path, ...],
    compiler: CompilerIdentity,
    environment_id: str,
    accelerator: AcceleratorIdentityFacet | None,
    target_intent: object | None,
) -> tuple[CompilerTargetIdentity | None, tuple[str, ...]]:
    """Derive one exact target without inferring identity from Triton's cache.

    Triton's current compilation listener reports the same target metadata for
    fresh compiles and cache hits.  Native PTX directives, when produced, are
    checked against that callback.  The callback does not identify a dump
    group, so mixed targets are deliberately not assigned to individual
    pipelines.
    """

    events, limitations = _triton_compiler_events(events_path)
    if not events:
        return None, limitations
    versions = {event.triton_version for event in events}
    if versions != {compiler.version}:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Triton compiler listener version contradicts the plan-bound distribution.",
        )
    targets = {
        (
            event.target.backend,
            event.target.architecture,
            event.target.warp_size,
        )
        for event in events
    }
    if len(targets) != 1:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Triton listener reported multiple compiler targets without dump-group linkage.",
        )
    backend, architecture, warp_size = next(iter(targets))
    if backend != "cuda" or warp_size != 32:
        return None, (
            *limitations,
            "Triton listener target is not a supported CUDA warp-32 compilation target.",
        )

    ptx_architecture, ptx_version, ptx_limitations = _ptx_target_metadata(native_paths)
    limitations = (*limitations, *ptx_limitations)
    if ptx_architecture is not None and ptx_architecture != architecture:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Emitted PTX target contradicts Triton's compilation listener target.",
        )
    if ptx_architecture is None and any(path.suffix.casefold() == ".ptx" for path in native_paths):
        return None, limitations

    if accelerator is None or accelerator.provider != "cuda":
        return None, (*limitations, "Observed CUDA environment identity was unavailable.")
    if (
        accelerator.status is not AcceleratorIdentityStatus.AVAILABLE
        or accelerator.driver_version is None
        or accelerator.runtime_version is None
        or not accelerator.devices
    ):
        return None, (
            *limitations,
            "Observed CUDA driver, runtime, or device identity was incomplete.",
        )
    observed_architectures = {
        f"sm_{device.compute_capability.replace('.', '')}"
        for device in accelerator.devices
        if device.compute_capability is not None
    }
    if not observed_architectures:
        return None, (*limitations, "Observed CUDA devices did not report compute capability.")

    intent = CompilerTarget.model_validate(target_intent) if target_intent is not None else None
    if intent is not None:
        if (
            intent.backend != backend
            or intent.architecture != architecture
            or (intent.warp_size is not None and intent.warp_size != warp_size)
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Explicit cross-compilation intent contradicts the emitted Triton target.",
            )
    elif len(observed_architectures) != 1 or architecture not in observed_architectures:
        return None, (
            *limitations,
            "Triton target could not be uniquely validated against the observed CUDA device.",
        )
    return (
        CompilerTargetIdentity(
            backend="cuda",
            architecture=architecture,
            warp_size=32,
            ptx_version=ptx_version,
            environment_id=environment_id,
        ),
        limitations,
    )


def _triton_compiler_events(path: Path) -> tuple[tuple[_TritonCompilerEvent, ...], tuple[str, ...]]:
    if not path.is_file():
        return (), ("Triton compiler listener output was unavailable after capture.",)
    events: list[_TritonCompilerEvent] = []
    limitations: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if len(events) == _MAX_TRITON_COMPILER_EVENTS:
                    return (), (
                        "Triton compiler listener exceeded the bounded event limit; target "
                        "identity is unavailable.",
                    )
                if len(line.encode("utf-8")) > _MAX_TRITON_COMPILER_EVENT_BYTES:
                    return (), (
                        "Triton compiler listener emitted an oversized event; target identity "
                        "is unavailable.",
                    )
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict) and set(value) == {"listener_unavailable"}:
                    reason = value["listener_unavailable"]
                    if isinstance(reason, str) and reason:
                        return (), (reason,)
                    return (), ("Triton compiler listener output was invalid.",)
                events.append(_TritonCompilerEvent.model_validate(value))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError):
        return (), ("Triton compiler listener output was invalid or unreadable.",)
    if not events:
        return (), ("Triton did not report a compiler target in the root interpreter.",)
    return tuple(events), tuple(limitations)


def _ptx_target_metadata(
    native_paths: tuple[Path, ...],
) -> tuple[str | None, str | None, tuple[str, ...]]:
    targets: set[str] = set()
    versions: set[str] = set()
    for path in native_paths:
        if path.suffix.casefold() != ".ptx":
            continue
        try:
            with path.open("rb") as stream:
                payload = stream.read(_MAX_PTX_METADATA_BYTES + 1)
        except OSError:
            return None, None, ("Emitted PTX could not be read for target validation.",)
        if len(payload) > _MAX_PTX_METADATA_BYTES:
            return None, None, ("Emitted PTX exceeded the metadata validation bound.",)
        target = _PTX_TARGET.search(payload)
        version = _PTX_VERSION.search(payload)
        if target is None or version is None:
            return None, None, ("Emitted PTX omitted a parseable target or version directive.",)
        targets.add(target.group(1).decode("ascii"))
        versions.add(version.group(1).decode("ascii"))
    if not targets:
        return None, None, ()
    if len(targets) != 1 or len(versions) != 1:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Emitted PTX files disagree on their compilation target or PTX version.",
        )
    return next(iter(targets)), next(iter(versions)), ()


def kernel_build_pipeline_requests(
    manifest: KernelBuildManifest,
    *,
    run_id: str,
    registration_ids_by_path: dict[str, str],
    run: RunManifest | None = None,
) -> tuple[RegisterPipelineRequest, ...]:
    """Build one normalized lineage request for every provider-native dump group."""

    compiler = None
    target = None
    limitations: tuple[str, ...] = ()
    if run is not None:
        compiler = compiler_identity_id(run.compiler_qualification)
        target = compiler_target_identity_id(run.compiler_qualification)
        if compiler is None:
            target = None
            limitations = (
                "Compiler identity is unavailable because the run lacks an exact managed "
                "compiler distribution.",
            )
        elif target is None:
            limitations = ("Compiler target identity was unavailable after capture.",)
    requests: list[RegisterPipelineRequest] = []
    for group in manifest.native_groups:
        stages = _group_pipeline_stages(
            manifest.producer,
            group,
            registration_ids_by_path=registration_ids_by_path,
        )
        requests.append(
            RegisterPipelineRequest(
                run_id=run_id,
                pipeline_name=f"{manifest.producer.value}.compiler",
                producer=manifest.producer,
                compiler_identity_id=compiler,
                target_identity_id=target,
                stages=stages,
                limitations=limitations,
            )
        )
    return tuple(requests)


def _group_pipeline_stages(
    producer: KernelBuildProducer,
    group: KernelBuildArtifactGroup,
    *,
    registration_ids_by_path: dict[str, str],
) -> tuple[PipelineStageDeclaration, ...]:
    """Derive deterministic, within-group compiler lineage from native file kinds."""

    ordered = sorted(
        (artifact for artifact in group.artifacts if _is_lineage_artifact(producer, artifact.path)),
        key=lambda artifact: (
            _STAGE_PRIORITY[_artifact_extension(producer, artifact.path)],
            artifact.path,
        ),
    )
    names: dict[str, int] = {}
    previous: str | None = None
    declarations: list[PipelineStageDeclaration] = []
    for ordinal, artifact in enumerate(ordered):
        format, format_schema = _artifact_format(producer, artifact.path)
        count = names.get(format, 0) + 1
        names[format] = count
        name = format if count == 1 else f"{format}_{count}"
        registration_id = registration_ids_by_path.get(artifact.path)
        if registration_id is None:
            raise ValueError(f"missing registration for {artifact.path!r}")
        declarations.append(
            RegisteredPipelineStageDeclaration(
                name=name,
                ordinal=ordinal,
                predecessor=previous,
                status=PipelineStageStatus.AVAILABLE,
                format=format,
                format_schema=format_schema,
                registration_id=registration_id,
                extractor_profile=(
                    PipelineExtractorProfile.TEXT_LINES_V1
                    if artifact.media_type.startswith("text/")
                    else None
                ),
            )
        )
        previous = name
    return tuple(declarations)


def _artifact_extension(producer: KernelBuildProducer, path: str) -> str:
    extension = next(
        (
            candidate
            for candidate in sorted(_extension_map(producer), key=len, reverse=True)
            if path.casefold().endswith(candidate)
        ),
        None,
    )
    if extension is None:
        raise ValueError(f"unsupported {producer.value} compiler artifact {path!r}")
    return extension


def _artifact_format(producer: KernelBuildProducer, path: str) -> tuple[str, str]:
    format, format_schema, _ = _extension_map(producer)[_artifact_extension(producer, path)]
    return format, format_schema


def _is_lineage_artifact(producer: KernelBuildProducer, path: str) -> bool:
    return _artifact_extension(producer, path) in _LINEAGE_EXTENSIONS


def _extension_map(producer: KernelBuildProducer) -> dict[str, tuple[str, str, str]]:
    return _TRITON_EXTENSION_MAP if producer is KernelBuildProducer.TRITON else _CUTE_EXTENSION_INFO


class KernelBuildCaptureCollector:
    """Inventory allowlisted staged native extensions and emit a kernel-build manifest.

    This collector walks the dump directory produced by the compiler's env-var
    controls, filters to a fixed extension allowlist, enforces hard
    member/byte/containment/symlink bounds, preserves files unchanged, and
    records native dump membership in a strict kernel-build manifest.
    It does not parse or rewrite compiler files or store pipeline lineage.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def collect(
        self,
        *,
        adapter: str,
        dump_dir: Path,
        output_root: Path,
        exit_code: int,
        cute_keep_allowlist: tuple[str, ...] | None = None,
        reproducer_path: Path | None = None,
    ) -> tuple[KernelBuildManifest, Path, tuple[Path, ...], tuple[str, ...]]:
        if adapter == "triton.compiler":
            extension_map = _TRITON_EXTENSION_MAP
        elif adapter == "cute.compiler":
            extension_map = self._cute_extension_map(cute_keep_allowlist)
        else:
            raise DomainError(
                ErrorCode.INTERNAL_ERROR,
                f"Unsupported kernel-build adapter {adapter!r}.",
            )
        inventory = self._inventory(dump_dir, output_root, extension_map)
        native_entries = tuple(inventory.entries)
        limitations: list[str] = list(inventory.limitations)
        # The reproducer file lives outside dump_dir (in output_root) and is
        # inventoried explicitly if it exists.
        reproducer_entry: KernelBuildInventoryEntry | None = None
        if reproducer_path is not None and reproducer_path.is_file():
            reproducer_entry = self._inspect_single_file(
                reproducer_path,
                output_root,
                _REPRODUCER_MEDIA_TYPE,
            )
            if reproducer_entry is not None:
                if len(native_entries) >= _MAX_KERNEL_BUILD_MEMBERS - 1:
                    limitations.append(
                        f"Kernel-build member count is limited to {_MAX_KERNEL_BUILD_MEMBERS} "
                        "including the manifest; the reproducer was skipped."
                    )
                    reproducer_entry = None
                elif (
                    sum(entry.byte_length for entry in native_entries)
                    + reproducer_entry.byte_length
                    > self.workspace.config.storage.max_staging_bytes
                ):
                    limitations.append(
                        "The reproducer would exceed the total kernel-build staging budget; "
                        "it was skipped."
                    )
                    reproducer_entry = None
            else:
                limitations.append(
                    f"Reproducer file {reproducer_path.name!r} was not a valid regular file."
                )
        attachments = (reproducer_entry,) if reproducer_entry is not None else ()
        native_groups = self._native_groups(
            adapter=adapter,
            entries=native_entries,
            dump_dir=dump_dir,
            output_root=output_root,
        )
        if not native_groups and exit_code != 0:
            limitations.append("No allowlisted native artifacts were found after compiler failure.")
        if exit_code == 0 and not native_groups:
            limitations.append(
                "Compiler exited successfully but produced no allowlisted native artifacts."
            )
        manifest = KernelBuildManifest(
            producer=(
                KernelBuildProducer.TRITON
                if adapter == "triton.compiler"
                else KernelBuildProducer.CUTE
            ),
            native_groups=native_groups,
            attachments=tuple(self._manifest_artifact(entry) for entry in attachments),
        )
        manifest_path = output_root / "kernel-build.json"
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
        native_paths = (
            *tuple(entry.path for entry in native_entries),
            *tuple(entry.path for entry in attachments),
        )
        return manifest, manifest_path, native_paths, tuple(dict.fromkeys(limitations))

    @staticmethod
    def _cute_extension_map(
        keep_allowlist: tuple[str, ...] | None,
    ) -> dict[str, tuple[str, str, str]]:
        """Build an extension→info map from the CUTE_DSL_KEEP token allowlist."""
        tokens = keep_allowlist or ("ir", "ptx", "cubin")
        allowed_exts: set[str] = set()
        for token in tokens:
            for ext in _CUTE_TOKEN_EXTENSIONS.get(token, ()):
                allowed_exts.add(ext)
        return {ext: info for ext, info in _CUTE_EXTENSION_INFO.items() if ext in allowed_exts}

    def _inspect_single_file(
        self,
        path: Path,
        output_root: Path,
        media_type: str,
    ) -> KernelBuildInventoryEntry | None:
        """Inspect a single file outside the dump directory (e.g. reproducer)."""
        try:
            relative = path.relative_to(output_root).as_posix()
        except ValueError:
            return None
        contained = PurePosixPath(relative)
        if contained.is_absolute() or ".." in contained.parts:
            return None
        try:
            metadata = path.lstat()
        except OSError:
            return None
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            return None
        if metadata.st_size > self.workspace.config.capture.max_artifact_bytes:
            return None
        try:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError:
            return None
        return KernelBuildInventoryEntry(
            path=path,
            relative_path=relative,
            byte_length=metadata.st_size,
            sha256=digest,
            media_type=media_type,
        )

    def _inventory(
        self,
        dump_dir: Path,
        output_root: Path,
        extension_map: dict[str, tuple[str, str, str]],
    ) -> KernelBuildInventory:
        if not dump_dir.is_dir():
            return KernelBuildInventory(
                entries=(),
                limitations=(f"Dump directory {dump_dir.name!r} was not created.",),
                dump_dir=None,
            )
        limitations: list[str] = []
        raw_entries: list[KernelBuildInventoryEntry] = []
        total_bytes = 0
        max_staging = self.workspace.config.storage.max_staging_bytes
        for path in sorted(dump_dir.rglob("*")):
            if path.is_dir():
                continue
            try:
                relative = path.relative_to(output_root).as_posix()
            except ValueError:
                limitations.append(f"File {path.name!r} is outside the staging root.")
                continue
            contained = PurePosixPath(relative)
            if contained.is_absolute() or ".." in contained.parts:
                limitations.append(f"File {path.name!r} escapes the staging root.")
                continue
            extension = self._matching_extension(path, extension_map)
            if extension is None:
                continue
            try:
                metadata = path.lstat()
            except OSError as exc:
                limitations.append(f"File {path.name!r} could not be inspected: {exc}.")
                continue
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                limitations.append(f"File {path.name!r} is not a regular non-linked file; skipped.")
                continue
            if metadata.st_size > self.workspace.config.capture.max_artifact_bytes:
                limitations.append(
                    f"File {path.name!r} exceeds the per-artifact byte limit; skipped."
                )
                continue
            if total_bytes + metadata.st_size > max_staging:
                limitations.append(
                    "Total kernel-build artifact bytes exceed the staging budget; "
                    "remaining files were skipped."
                )
                break
            if len(raw_entries) >= _MAX_KERNEL_BUILD_MEMBERS - 1:
                limitations.append(
                    f"Kernel-build member count is limited to {_MAX_KERNEL_BUILD_MEMBERS} "
                    "including the manifest; "
                    "remaining files were skipped."
                )
                break
            try:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as exc:
                limitations.append(f"File {path.name!r} could not be read: {exc}.")
                continue
            _, _, media_type = extension_map[extension]
            total_bytes += metadata.st_size
            raw_entries.append(
                KernelBuildInventoryEntry(
                    path=path,
                    relative_path=relative,
                    byte_length=metadata.st_size,
                    sha256=digest,
                    media_type=media_type,
                )
            )
        return KernelBuildInventory(
            entries=tuple(raw_entries),
            limitations=tuple(dict.fromkeys(limitations)),
            dump_dir=dump_dir,
        )

    @staticmethod
    def _matching_extension(
        path: Path,
        extension_map: dict[str, tuple[str, str, str]],
    ) -> str | None:
        name = path.name.lower()
        return next(
            (
                extension
                for extension in sorted(extension_map, key=len, reverse=True)
                if name.endswith(extension)
            ),
            None,
        )

    @staticmethod
    def _manifest_artifact(entry: KernelBuildInventoryEntry) -> KernelBuildArtifact:
        return KernelBuildArtifact(
            path=entry.relative_path,
            byte_length=entry.byte_length,
            sha256=entry.sha256,
            media_type=entry.media_type,
        )

    def _native_groups(
        self,
        *,
        adapter: str,
        entries: tuple[KernelBuildInventoryEntry, ...],
        dump_dir: Path,
        output_root: Path,
    ) -> tuple[KernelBuildArtifactGroup, ...]:
        """Keep each immediate Triton dump parent as one source-hash build group.

        Current Triton creates a dump manager per source hash and writes that
        compilation's ordered stage files into its directory. Nested paths are
        therefore separate groups, never evidence of a cross-group stage order.
        """

        grouped: dict[str, list[KernelBuildInventoryEntry]] = {}
        for entry in entries:
            group_root = entry.path.parent if adapter == "triton.compiler" else dump_dir
            group_path = group_root.relative_to(output_root).as_posix()
            grouped.setdefault(group_path, []).append(entry)
        return tuple(
            KernelBuildArtifactGroup(
                path=group_path,
                artifacts=tuple(
                    self._manifest_artifact(entry)
                    for entry in sorted(entries, key=lambda item: item.relative_path)
                ),
            )
            for group_path, entries in sorted(grouped.items())
        )


class KernelBuildImportResult(ContractModel):
    run: RunManifest
    manifest_artifact_id: str
    pipeline_ids: tuple[str, ...]
    corpus_commit_id: str


class KernelBuildImportService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def import_manifest(
        self,
        manifest_path: Path,
        *,
        sensitivity: Sensitivity = Sensitivity.INTERNAL,
        allow_external_path: bool = False,
    ) -> KernelBuildImportResult:
        importer = ImportService(self.workspace)
        with importer._snapshot_provider_document(
            manifest_path,
            allow_external_path=allow_external_path,
            max_bytes=_MAX_KERNEL_BUILD_MANIFEST_BYTES,
        ) as snapshot:
            manifest, _, _ = self._load_manifest(snapshot.payload_path)
            manifest_bytes = snapshot.byte_length
            manifest_sha256 = snapshot.sha256
        try:
            for group in manifest.native_groups:
                for artifact in group.artifacts:
                    _artifact_format(manifest.producer, artifact.path)
        except ValueError as error:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Kernel-build manifest declares an unsupported compiler artifact.",
                details={"validation_error": str(error)},
            ) from error
        native_paths = {
            artifact.path for group in manifest.native_groups for artifact in group.artifacts
        }
        members: list[BundleMember] = []
        for artifact in manifest.artifacts:
            path = manifest_path.parent / artifact.path
            members.append(
                BundleMember(
                    path=path,
                    role=(
                        "compiler_output"
                        if artifact.path in native_paths
                        else "compiler_attachment"
                    ),
                    media_type=artifact.media_type,
                    display_name=artifact.path,
                    expected_byte_length=artifact.byte_length,
                    expected_sha256=artifact.sha256,
                )
            )
        if len(members) > _MAX_KERNEL_BUILD_MEMBERS - 1:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Kernel-build manifest exceeds the 99 native-artifact bundle limit.",
            )
        imported = importer.import_provider_bundle(
            ImportBundleRequest(
                primary=BundleMember(
                    path=manifest_path,
                    role="kernel_build_manifest",
                    media_type="application/json",
                    expected_byte_length=manifest_bytes,
                    expected_sha256=manifest_sha256,
                ),
                sidecars=tuple(members),
                kind=ArtifactKind.KERNEL_BUILD,
                sensitivity=sensitivity,
                producer=manifest.producer,
                allow_external_path=allow_external_path,
            )
        )
        registrations = imported.run.artifacts[1:]
        artifact_paths = [artifact.path for artifact in manifest.artifacts]
        registration_ids_by_path = {
            path: registration.registration_id
            for path, registration in zip(artifact_paths, registrations, strict=True)
        }
        pipeline_service = ArtifactPipelineService(self.workspace)
        pipelines = tuple(
            pipeline_service.register_imported(request)
            for request in kernel_build_pipeline_requests(
                manifest,
                run_id=imported.run.run_id,
                registration_ids_by_path=registration_ids_by_path,
            )
        )
        return KernelBuildImportResult(
            run=imported.run,
            manifest_artifact_id=imported.primary_artifact_id,
            pipeline_ids=tuple(pipeline.pipeline_id for pipeline in pipelines),
            corpus_commit_id=imported.corpus_commit_id,
        )

    def _load_manifest(self, path: Path) -> tuple[KernelBuildManifest, int, str]:
        try:
            size = path.stat().st_size
            if size > _MAX_KERNEL_BUILD_MANIFEST_BYTES:
                raise DomainError(
                    ErrorCode.ARTIFACT_TOO_LARGE,
                    "Kernel-build manifest exceeds the 1 MiB provider-document limit.",
                )
            raw = path.read_bytes()
            if len(raw) != size:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Kernel-build manifest changed while it was read.",
                    retryable=True,
                )
            payload = json.loads(raw.decode("utf-8"))
            return (
                KernelBuildManifest.model_validate(payload),
                len(raw),
                hashlib.sha256(raw).hexdigest(),
            )
        except DomainError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Kernel-build manifest is missing, malformed, or unsupported.",
                details={"validation_error": str(error)[:2_000]},
            ) from error
