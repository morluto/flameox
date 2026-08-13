from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, assert_never

from pydantic import JsonValue, ValidationError

from flameox.adapters.kernel_build import (
    ArtifactKernelBuildStage,
    ArtifactlessKernelBuildStage,
    KernelBuildArtifact,
    KernelBuildCacheStatus,
    KernelBuildContext,
    KernelBuildManifest,
    KernelBuildManifestV1,
    KernelBuildManifestV2,
    KernelBuildOutcome,
    KernelBuildProducer,
    KernelBuildStage,
    KernelBuildStageStatus,
    KernelBuildTarget,
    parse_kernel_build_manifest,
)
from flameox.application.imports import (
    BundleMember,
    ImportBundleRequest,
    ImportService,
)
from flameox.application.pipelines import (
    ArtifactPipeline,
    ArtifactPipelineService,
    PipelineStageDeclaration,
    RegisteredPipelineStageDeclaration,
    RegisterPipelineRequest,
    UnregisteredPipelineStageDeclaration,
)
from flameox.atomic import atomic_write_json
from flameox.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    RunManifest,
    Sensitivity,
    WorkloadInstance,
    digest_model,
)
from flameox.models import ContractModel
from flameox.storage import Workspace

_MAX_KERNEL_BUILD_MEMBERS = 100
_MAX_KERNEL_BUILD_MANIFEST_BYTES = 1024 * 1024

# Triton dump extensions: NVIDIA (ttir/ttgir/llir/ptx/cubin/sass) and
# AMD (amdgcn/hsaco) targets, plus metadata files emitted alongside IR.
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

# Reproducer file format info (lives outside dump_dir, in output_root).
_REPRODUCER_FORMAT_INFO: tuple[str, str, str] = (
    "reproducer",
    "triton-reproducer-mlir",
    "text/plain",
)


@dataclass(frozen=True, slots=True)
class KernelBuildInventoryEntry:
    path: Path
    relative_path: str
    byte_length: int
    sha256: str
    extension: str
    format: str
    format_schema: str
    media_type: str


@dataclass(frozen=True, slots=True)
class KernelBuildInventory:
    entries: tuple[KernelBuildInventoryEntry, ...]
    limitations: tuple[str, ...]
    dump_dir: Path | None


def managed_kernel_build_context(
    *,
    workload_instance: WorkloadInstance,
    adapter: str,
    producer_version: str | None,
    adapter_options: dict[str, JsonValue],
) -> KernelBuildContext:
    """Bind one compiler invocation to its exact managed workload and protocol inputs."""

    raw_target = adapter_options.get("target")
    target = KernelBuildTarget.model_validate(raw_target) if raw_target is not None else None
    return KernelBuildContext(
        workload_definition_id=workload_instance.workload_definition_id,
        workload_instance_id=workload_instance.workload_instance_id,
        command_digest=digest_model(workload_instance.command.model_dump(mode="json")),
        parameters_digest=digest_model(workload_instance.parameters),
        compiler_identity_id=digest_model(
            {
                "adapter": adapter,
                "producer_version": producer_version or "unknown",
            }
        ),
        build_protocol_id=digest_model(
            {
                "schema_version": "flameox.kernel-build-protocol.v1",
                "adapter": adapter,
                "producer_version": producer_version or "unknown",
                "adapter_options": adapter_options,
            }
        ),
        target=target,
        target_identity_id=(
            digest_model(target.model_dump(mode="json")) if target is not None else None
        ),
    )


def kernel_build_pipeline_request(
    manifest: KernelBuildManifest,
    *,
    run_id: str,
    registration_ids_by_path: dict[str, str],
) -> RegisterPipelineRequest:
    """Translate a parsed provider manifest into the application pipeline contract."""

    declarations: list[PipelineStageDeclaration] = []
    for stage in manifest.stages:
        if isinstance(stage, ArtifactKernelBuildStage):
            registration_id = registration_ids_by_path.get(stage.artifact.path)
            if registration_id is None:
                raise ValueError(f"missing registration for {stage.artifact.path!r}")
            declarations.append(
                RegisteredPipelineStageDeclaration.model_validate(
                    {
                        **stage.model_dump(exclude={"artifact", "elapsed_ns"}),
                        "registration_id": registration_id,
                        "extractor_profile": (
                            "text-lines-v1"
                            if stage.artifact.media_type.startswith("text/")
                            or stage.artifact.media_type == "application/json"
                            else None
                        ),
                    }
                )
            )
        elif isinstance(stage, ArtifactlessKernelBuildStage):
            declarations.append(
                UnregisteredPipelineStageDeclaration.model_validate(
                    stage.model_dump(exclude={"artifact", "elapsed_ns"})
                )
            )
        else:
            assert_never(stage)
    derived: list[str] = []
    if manifest.diagnostics:
        derived.append("compiler diagnostics are preserved in the kernel-build manifest")
    if isinstance(manifest, KernelBuildManifestV1):
        workload_label = manifest.workload_identity
        device_label = manifest.device_identity
        context: KernelBuildContext | None = None
        if device_label is None:
            derived.append("device identity was not supplied by the producer")
    else:
        workload_label = manifest.workload_label
        context = manifest.build_context
        if context.target is None:
            device_label = None
            derived.append("compiler target identity was not supplied by the producer")
        else:
            warp = f":warp{context.target.warp_size}" if context.target.warp_size else ""
            device_label = f"{context.target.backend}:{context.target.architecture}{warp}"
    limitations = list(dict.fromkeys((*derived, *manifest.limitations)))
    if len(limitations) > 20:
        retained = 19 - len(derived)
        limitations = [
            *derived,
            *tuple(dict.fromkeys(manifest.limitations))[:retained],
            "Additional limitations remain available in the kernel-build manifest.",
        ]
    return RegisterPipelineRequest(
        run_id=run_id,
        pipeline_name=f"{manifest.producer.value}.compiler",
        pipeline_schema=manifest.schema_version,
        producer=manifest.producer,
        producer_version=manifest.producer_version,
        workload_identity=workload_label,
        device_identity=device_label,
        workload_definition_id=(context.workload_definition_id if context is not None else None),
        workload_instance_id=(context.workload_instance_id if context is not None else None),
        command_digest=(context.command_digest if context is not None else None),
        parameters_digest=(context.parameters_digest if context is not None else None),
        compiler_identity_id=(context.compiler_identity_id if context is not None else None),
        build_protocol_id=(context.build_protocol_id if context is not None else None),
        target_identity_id=(context.target_identity_id if context is not None else None),
        stages=tuple(declarations),
        limitations=tuple(dict.fromkeys(limitations)),
    )


class KernelBuildCaptureCollector:
    """Inventory allowlisted staged native extensions and emit a kernel-build manifest.

    This collector walks the dump directory produced by the compiler's env-var
    controls, filters to a fixed extension allowlist, enforces hard
    member/byte/containment/symlink bounds, preserves files unchanged, and
    builds a ``flameox.kernel-build.v2`` manifest. It does not parse or rewrite
    compiler files.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def collect(
        self,
        *,
        adapter: str,
        dump_dir: Path,
        output_root: Path,
        workload_label: str,
        build_context: KernelBuildContext,
        exit_code: int,
        producer_version: str | None,
        source_environment: dict[str, str],
        cute_keep_allowlist: tuple[str, ...] | None = None,
        reproducer_path: Path | None = None,
    ) -> tuple[KernelBuildManifestV2, Path, tuple[Path, ...]]:
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
        entries = list(inventory.entries)
        limitations: list[str] = list(inventory.limitations)
        # The reproducer file lives outside dump_dir (in output_root) and is
        # inventoried explicitly if it exists.
        reproducer_entry: KernelBuildInventoryEntry | None = None
        if reproducer_path is not None and reproducer_path.is_file():
            reproducer_entry = self._inspect_single_file(
                reproducer_path,
                output_root,
                _REPRODUCER_FORMAT_INFO,
            )
            if reproducer_entry is not None:
                if len(entries) >= _MAX_KERNEL_BUILD_MEMBERS - 1:
                    limitations.append(
                        f"Kernel-build member count is limited to {_MAX_KERNEL_BUILD_MEMBERS} "
                        "including the manifest; the reproducer was skipped."
                    )
                elif (
                    sum(entry.byte_length for entry in entries) + reproducer_entry.byte_length
                    > self.workspace.config.storage.max_staging_bytes
                ):
                    limitations.append(
                        "The reproducer would exceed the total kernel-build staging budget; "
                        "it was skipped."
                    )
                else:
                    entries.append(reproducer_entry)
            else:
                limitations.append(
                    f"Reproducer file {reproducer_path.name!r} was not a valid regular file."
                )
        entries_tuple = tuple(entries)
        if entries_tuple:
            limitations.append(
                "Compiler artifact predecessor lineage is not declared by the provider output."
            )
        outcome: KernelBuildOutcome
        if exit_code == 0 and entries_tuple:
            outcome = KernelBuildOutcome.SUCCEEDED
        elif exit_code != 0:
            outcome = KernelBuildOutcome.FAILED
        else:
            outcome = KernelBuildOutcome.INCONCLUSIVE
        stages = self._build_stages(entries_tuple, outcome=outcome)
        if not entries_tuple and exit_code != 0:
            limitations.append("No allowlisted native artifacts were found after compiler failure.")
        if outcome is KernelBuildOutcome.INCONCLUSIVE:
            limitations.append(
                "Compiler exited successfully but produced no allowlisted native artifacts."
            )
        if build_context.target is None:
            limitations.append(
                "Compiler target was not declared; exact compatibility cannot be established."
            )
        if producer_version is None or producer_version.casefold() == "unknown":
            limitations.append(
                "Compiler version was unavailable; exact compatibility cannot be established."
            )
        manifest = KernelBuildManifestV2(
            producer=(
                KernelBuildProducer.TRITON
                if adapter == "triton.compiler"
                else KernelBuildProducer.CUTE
            ),
            producer_version=producer_version or "unknown",
            workload_label=workload_label,
            build_context=build_context,
            outcome=outcome,
            cache_status=KernelBuildCacheStatus.UNKNOWN,
            stages=stages,
            source_environment=self._normalized_environment(source_environment, output_root),
            limitations=tuple(dict.fromkeys(limitations)),
        )
        manifest_path = output_root / "kernel-build.json"
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
        native_paths = tuple(entry.path for entry in entries_tuple)
        return manifest, manifest_path, native_paths

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
        format_info: tuple[str, str, str],
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
        fmt, fmt_schema, media_type = format_info
        return KernelBuildInventoryEntry(
            path=path,
            relative_path=relative,
            byte_length=metadata.st_size,
            sha256=digest,
            extension=path.suffix.lower(),
            format=fmt,
            format_schema=fmt_schema,
            media_type=media_type,
        )

    @staticmethod
    def _normalized_environment(environment: dict[str, str], output_root: Path) -> dict[str, str]:
        normalized: dict[str, str] = {}
        resolved_root = output_root.resolve()
        for key, value in environment.items():
            candidate = Path(value)
            if candidate.is_absolute():
                try:
                    relative = candidate.resolve().relative_to(resolved_root).as_posix()
                except (OSError, ValueError):
                    normalized[key] = value
                else:
                    normalized[key] = f"<staging>/{relative}"
            else:
                normalized[key] = value
        return normalized

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
            fmt, fmt_schema, media_type = extension_map[extension]
            total_bytes += metadata.st_size
            raw_entries.append(
                KernelBuildInventoryEntry(
                    path=path,
                    relative_path=relative,
                    byte_length=metadata.st_size,
                    sha256=digest,
                    extension=extension,
                    format=fmt,
                    format_schema=fmt_schema,
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
    def _build_stages(
        entries: tuple[KernelBuildInventoryEntry, ...],
        *,
        outcome: KernelBuildOutcome,
    ) -> tuple[KernelBuildStage, ...]:
        priority = {
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
            ".metadata": 6,
            ".json": 6,
        }
        stages: list[KernelBuildStage] = []
        name_counts: dict[str, int] = {}
        ordered_entries = sorted(
            entries,
            key=lambda entry: (priority.get(entry.extension, 99), entry.relative_path),
        )
        for ordinal, entry in enumerate(ordered_entries):
            base_name = entry.format
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            stage_name = base_name if count == 0 else f"{base_name}_{count + 1}"
            stages.append(
                ArtifactKernelBuildStage(
                    name=stage_name,
                    ordinal=ordinal,
                    predecessor=None,
                    status=KernelBuildStageStatus.AVAILABLE,
                    format=entry.format,
                    format_schema=entry.format_schema,
                    artifact=KernelBuildArtifact(
                        path=entry.relative_path,
                        byte_length=entry.byte_length,
                        sha256=entry.sha256,
                        media_type=entry.media_type,
                    ),
                )
            )
        if not stages:
            empty_status: Literal[
                KernelBuildStageStatus.FAILED,
                KernelBuildStageStatus.UNAVAILABLE,
            ] = (
                KernelBuildStageStatus.FAILED
                if outcome is KernelBuildOutcome.FAILED
                else KernelBuildStageStatus.UNAVAILABLE
            )
            stages.append(
                ArtifactlessKernelBuildStage(
                    name="empty",
                    ordinal=0,
                    status=empty_status,
                    format="unknown",
                    format_schema="unknown",
                )
            )
        elif outcome is KernelBuildOutcome.FAILED:
            stages.append(
                ArtifactlessKernelBuildStage(
                    name="build_failed",
                    ordinal=len(stages),
                    predecessor=None,
                    status=KernelBuildStageStatus.FAILED,
                    format="unknown",
                    format_schema="unknown",
                )
            )
        return tuple(stages)


class KernelBuildImportResult(ContractModel):
    run: RunManifest
    manifest_artifact_id: str
    pipeline: ArtifactPipeline
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
        members: list[BundleMember] = []
        for stage in manifest.stages:
            if stage.artifact is None:
                continue
            path = manifest_path.parent / stage.artifact.path
            members.append(
                BundleMember(
                    path=path,
                    role=stage.artifact.role,
                    media_type=stage.artifact.media_type,
                    display_name=stage.artifact.path,
                    expected_byte_length=stage.artifact.byte_length,
                    expected_sha256=stage.artifact.sha256,
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
                producer_version=manifest.producer_version,
                allow_external_path=allow_external_path,
            )
        )
        registrations = imported.run.artifacts[1:]
        artifact_paths = [
            stage.artifact.path for stage in manifest.stages if stage.artifact is not None
        ]
        registration_ids_by_path = {
            path: registration.registration_id
            for path, registration in zip(artifact_paths, registrations, strict=True)
        }
        pipeline = ArtifactPipelineService(self.workspace).register_imported(
            kernel_build_pipeline_request(
                manifest,
                run_id=imported.run.run_id,
                registration_ids_by_path=registration_ids_by_path,
            )
        )
        return KernelBuildImportResult(
            run=imported.run,
            manifest_artifact_id=imported.primary_artifact_id,
            pipeline=pipeline,
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
                parse_kernel_build_manifest(payload),
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
