"""Provider-defined import for NVBench JSON and declared jsonbin sidecars."""

from __future__ import annotations

import hashlib
import json
import re
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from flameox.adapters.nvbench import NvbenchJsonDocument, NvbenchSummary
from flameox.application.imports import (
    BundleMember,
    ImportBundleRequest,
    ImportBundleResult,
    ImportService,
)
from flameox.domain import ArtifactKind, DomainError, ErrorCode, RunManifest, Sensitivity
from flameox.models import ContractModel
from flameox.storage import Workspace

_FLOAT32_SIZE = 4
_MAX_SIDECARS = 99


class NvbenchImportResult(ContractModel):
    """Result of an NVBench provider-defined bundle import."""

    run: RunManifest
    primary_artifact_id: str
    sidecar_artifact_ids: tuple[str, ...]
    corpus_commit_id: str
    sidecar_count: int


class NvbenchImportService:
    """Import an NVBench JSON document and its provider-declared sidecars.

    The primary JSON is parsed first to discover which sidecar files the
    document references.  Only those files are imported — no arbitrary
    sibling files are accepted.  Each sidecar's ``expected_byte_length``
    is bound to ``declared_size * 4`` (float32) and, if a digest is
    provided, ``expected_sha256`` is bound too.  The ``import_bundle``
    primitive verifies these declarations immediately after import and
    before registration, closing the TOCTOU gap.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def import_json(
        self,
        json_path: Path,
        *,
        sensitivity: Sensitivity = Sensitivity.INTERNAL,
        allow_external_path: bool = False,
        expected_sha256: str | None = None,
    ) -> NvbenchImportResult:
        """Import an NVBench JSON and its declared sidecars as one bundle.

        Parameters
        ----------
        json_path:
            Path to the NVBench ``--json`` output file.
        sensitivity:
            Sensitivity classification for the imported artifacts.
        allow_external_path:
            When true, allow the JSON's parent directory as an import root
            so sidecars outside the project root can be imported.
        expected_sha256:
            Optional declared digest for the primary JSON itself.  When
            set, the imported JSON is verified against this digest.
        """
        if (
            expected_sha256 is not None
            and re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", expected_sha256) is None
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "NVBench expected_sha256 must be a SHA-256 digest.",
            )
        document, document_bytes, document_sha256 = load_nvbench_document_with_integrity(
            json_path,
            max_bytes=self.workspace.config.capture.max_artifact_bytes,
        )
        sidecar_specs = collect_nvbench_sidecar_specs(document)
        sidecar_members: list[BundleMember] = []
        json_dir = json_path.parent
        for spec in sidecar_specs:
            sidecar_path = resolve_nvbench_sidecar_path(spec.filename, json_dir)
            sidecar_members.append(
                BundleMember(
                    path=sidecar_path,
                    role="nvbench_sidecar",
                    display_name=spec.filename,
                    media_type="application/octet-stream",
                    expected_byte_length=spec.byte_length,
                )
            )
        primary_member = BundleMember(
            path=json_path,
            role="primary",
            media_type="application/json",
            display_name=json_path.name,
            expected_byte_length=document_bytes,
            expected_sha256=expected_sha256 or document_sha256,
        )
        imported = ImportService(self.workspace)._import_provider_bundle(
            ImportBundleRequest(
                primary=primary_member,
                sidecars=tuple(sidecar_members),
                kind=ArtifactKind.BENCHMARK_SAMPLES,
                sensitivity=sensitivity,
                producer="nvbench",
                producer_version=self._producer_version(document),
                allow_external_path=allow_external_path,
            )
        )
        return self._build_result(imported, len(sidecar_members))

    @staticmethod
    def _producer_version(document: NvbenchJsonDocument) -> str | None:
        version = document.nvbench_version
        if version is None:
            return None
        return version.string or f"{version.major}.{version.minor}.{version.patch}"

    @staticmethod
    def _build_result(
        imported: ImportBundleResult,
        sidecar_count: int,
    ) -> NvbenchImportResult:
        return NvbenchImportResult(
            run=imported.run,
            primary_artifact_id=imported.primary_artifact_id,
            sidecar_artifact_ids=imported.sidecar_artifact_ids,
            corpus_commit_id=imported.corpus_commit_id,
            sidecar_count=sidecar_count,
        )


@dataclass(frozen=True, slots=True)
class NvbenchSidecarSpec:
    filename: str
    byte_length: int


def _sidecar_spec_from_summary(summary: NvbenchSummary) -> NvbenchSidecarSpec | None:
    """Extract a sidecar specification from a summary entry.

    Returns ``None`` if the summary does not reference a file sidecar.
    Raises ``ARTIFACT_PARSE_FAILED`` if the summary has a known file hint
    (``file/sample_times`` or ``file/sample_freqs``) but is missing the
    filename or has an invalid/missing size, or if the hint starts with
    ``file/`` but is not one of the two known encodings.
    """
    if summary.hint is None or not summary.hint.startswith("file/"):
        return None
    if not summary.is_file_sidecar:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"Unknown NVBench file hint {summary.hint!r}; "
            "only file/sample_times and file/sample_freqs are supported.",
        )
    filename = summary.sidecar_filename
    if filename is None:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench summary with hint {summary.hint!r} is missing the required filename datum.",
        )
    size = summary.sidecar_size
    if size is None:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench summary with hint {summary.hint!r} is missing "
            "a valid decimal-string size datum.",
        )
    return NvbenchSidecarSpec(filename=filename, byte_length=size * _FLOAT32_SIZE)


def collect_nvbench_sidecar_specs(
    document: NvbenchJsonDocument,
) -> list[NvbenchSidecarSpec]:
    """Collect unique sidecar declarations from an NVBench JSON document.

    Shared provider-declared sidecar selector used by both
    ``NvbenchImportService`` (import path) and ``CaptureService``
    (capture path).  Each summary with a ``file/`` hint may declare a
    sidecar filename and size.  Duplicate filenames with **identical**
    sizes are collapsed to the first declaration (NVBench may reference
    the same sidecar from multiple summaries).  Duplicate filenames with
    **different** sizes are rejected as an ambiguous declaration.
    """
    seen: dict[str, int] = {}
    specs: list[NvbenchSidecarSpec] = []
    for bench in document.benchmarks:
        for state in bench.states:
            if state.is_skipped:
                continue
            for summary in state.summaries:
                spec = _sidecar_spec_from_summary(summary)
                if spec is None:
                    continue
                if spec.filename in seen:
                    if seen[spec.filename] != spec.byte_length:
                        raise DomainError(
                            ErrorCode.ARTIFACT_PARSE_FAILED,
                            f"NVBench document declares conflicting sizes "
                            f"for sidecar {spec.filename!r}.",
                        )
                    continue
                seen[spec.filename] = spec.byte_length
                specs.append(spec)
    if len(specs) > _MAX_SIDECARS:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            f"NVBench document declares {len(specs)} sidecars, "
            f"exceeding the {_MAX_SIDECARS}-sidecar limit.",
        )
    return specs


def load_nvbench_sidecar_specs(
    json_path: Path,
    *,
    max_bytes: int,
) -> list[NvbenchSidecarSpec]:
    """Load an NVBench JSON and return its provider-declared sidecar specs.

    Raises ``DomainError(ARTIFACT_PARSE_FAILED)`` if the JSON is missing,
    too large, malformed, or contains unsupported file hints.  Raises
    ``DomainError(ARTIFACT_TOO_LARGE)`` if the file exceeds ``max_bytes``.

    The caller decides how to handle failures:
    - Successful capture: let the exception propagate so
      ``_native_output_manifest_is_valid`` returns ``False`` and the run
      fails with a bounded limitation.
    - Nonzero/partial capture: catch the exception to treat the missing
      sidecar evidence as a bounded limitation/proof gap, not a trigger
      for globbing.
    """
    return collect_nvbench_sidecar_specs(load_nvbench_document(json_path, max_bytes=max_bytes))


def load_nvbench_document(json_path: Path, *, max_bytes: int) -> NvbenchJsonDocument:
    document, _, _ = load_nvbench_document_with_integrity(json_path, max_bytes=max_bytes)
    return document


def load_nvbench_document_with_integrity(
    json_path: Path,
    *,
    max_bytes: int,
) -> tuple[NvbenchJsonDocument, int, str]:
    try:
        size = json_path.stat().st_size
        if size > max_bytes:
            raise DomainError(
                ErrorCode.ARTIFACT_TOO_LARGE,
                "NVBench JSON exceeds the configured per-artifact byte limit.",
            )
        raw = json_path.read_bytes()
        if len(raw) != size:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "NVBench JSON changed while it was read.",
                retryable=True,
            )
        document = NvbenchJsonDocument.model_validate(json.loads(raw.decode("utf-8")))
        return document, len(raw), hashlib.sha256(raw).hexdigest()
    except DomainError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            "The file is not a valid NVBench JSON document.",
            details={"error": str(exc)[:2_000]},
        ) from exc


def resolve_nvbench_sidecar_path(
    filename: str,
    output_root: Path,
) -> Path:
    """Resolve a declared sidecar filename against the output root.

    The filename must be a normalized relative POSIX path (no absolute
    paths, no ``..`` components, no leading ``/``).  The resolved path
    (after symlink expansion) must remain under ``output_root``.

    Returns the **lexical** candidate path (``output_root / filename``),
    not the symlink-resolved target.  This ensures that a symlink
    inside the bundle is preserved for ``lstat`` so that
    ``validate_nvbench_sidecar_file`` can reject it as non-regular.

    Raises ``DomainError(ARTIFACT_PARSE_FAILED)`` if the filename is
    not a safe relative path or escapes the output root.
    """
    relative = PurePosixPath(filename)
    if (
        not filename
        or "\\" in filename
        or relative.is_absolute()
        or filename != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench sidecar filename {filename!r} is not a normalized relative POSIX path.",
        )
    candidate = output_root.joinpath(*relative.parts)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(output_root.resolve())
    except ValueError:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench sidecar filename {filename!r} resolves outside the output root.",
        ) from None
    return candidate


def validate_nvbench_sidecar_file(
    path: Path,
    expected_byte_length: int,
) -> int:
    """Validate a sidecar file on disk before registration.

    Returns the actual byte length on success.  Raises
    ``DomainError(ARTIFACT_PARSE_FAILED)`` if the file is missing, not a
    regular file, is hard-linked, or its byte length does not match the
    declared ``expected_byte_length``.
    """
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench sidecar {path!s} could not be stat'd.",
            details={"error": str(exc)[:2_000]},
        ) from exc
    if not stat_module.S_ISREG(stat_result.st_mode):
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench sidecar {path!s} is not a regular file.",
        )
    if stat_result.st_nlink != 1:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench sidecar {path!s} is hard-linked and cannot be registered.",
        )
    if stat_result.st_size != expected_byte_length:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"NVBench sidecar {path!s} byte length mismatch: "
            f"expected {expected_byte_length}, got {stat_result.st_size}.",
        )
    return stat_result.st_size
