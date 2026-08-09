from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator

from flameox.adapters.torch_profiler import torch_profiler_options
from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel

BoundedFilter = Annotated[
    str,
    StringConstraints(min_length=1, max_length=500, pattern=r"^[^\x00\r\n]+$"),
]


class ComputeSanitizerOptions(ContractModel):
    tool: Literal["memcheck", "racecheck", "initcheck", "synccheck"] = "memcheck"
    launch_skip: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    launch_count: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    target_processes: Literal["application-only", "all"] = "application-only"
    target_processes_filter: BoundedFilter | None = None
    kernel_name: BoundedFilter | None = None
    demangle: Literal["full", "simple", "no"] = "full"
    suppression_file: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = None
    suppression_digest: (
        Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None
    finding_exit_code: Annotated[int, Field(ge=1, le=255)] = 86

    @field_validator("suppression_file")
    @classmethod
    def project_relative_suppression(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("suppression_file must be project-relative and contained")
        return path.as_posix()

    @model_validator(mode="after")
    def suppression_identity_is_complete(self) -> ComputeSanitizerOptions:
        if (self.suppression_file is None) != (self.suppression_digest is None):
            raise ValueError("suppression_file and suppression_digest must be bound together")
        return self


class NvbenchOptions(ContractModel):
    """Bounded capture options for the NVBench CLI (``--json`` and ``--jsonbin``)."""

    enable_jsonbin: bool = True
    stopping_criterion: Annotated[str, StringConstraints(min_length=1, max_length=100)] | None = (
        None
    )
    min_samples: Annotated[int, Field(ge=1, le=1_000_000)] | None = None
    timeout: Annotated[float, Field(gt=0, le=86_400)] | None = None
    devices: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None


_BoundedSubdir = Annotated[
    str,
    StringConstraints(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]


_BoundedFilename = Annotated[
    str,
    StringConstraints(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]


class TritonCompilerOptions(ContractModel):
    """Bounded capture options for the managed triton.compiler env-var adapter.

    ``reproducer_filename`` sets ``TRITON_REPRODUCER_PATH`` to a bounded
    filename inside the output root (not inside the dump directory).  The
    reproducer file is inventoried explicitly if it exists after the run.
    """

    dump_subdir: _BoundedSubdir = "triton-dumps"
    kernel_dump: bool = True
    reproducer_filename: _BoundedFilename | None = None


CuteDslKeepToken = Literal["ir", "ir-debug", "ptx", "cubin", "sass", "llvm", "all"]


class CuteCompilerOptions(ContractModel):
    """Bounded capture options for the managed cute.compiler env-var adapter.

    ``keep_allowlist`` accepts the official ``CUTE_DSL_KEEP`` tokens documented
    by CuTe DSL: ``ir``, ``ir-debug``, ``ptx``, ``cubin``, ``sass``, ``llvm``,
    and ``all``.  The ``all`` token is mutually exclusive with the others.
    """

    dump_subdir: _BoundedSubdir = "cute-dsl-dumps"
    keep_allowlist: Annotated[
        tuple[CuteDslKeepToken, ...],
        Field(min_length=1, max_length=20),
    ] = ("ir", "ptx", "cubin")

    @model_validator(mode="after")
    def unique_and_consistent_allowlist(self) -> CuteCompilerOptions:
        if len(set(self.keep_allowlist)) != len(self.keep_allowlist):
            raise ValueError("CUTE_DSL_KEEP allowlist entries must be unique")
        if "all" in self.keep_allowlist and len(self.keep_allowlist) > 1:
            raise ValueError("CUTE_DSL_KEEP 'all' is mutually exclusive with other tokens")
        if {"ir", "ir-debug"}.issubset(self.keep_allowlist):
            raise ValueError("CUTE_DSL_KEEP 'ir' and 'ir-debug' are mutually exclusive")
        return self


_ADAPTER_OPTION_MODELS: dict[str, type[ContractModel]] = {
    "compute-sanitizer": ComputeSanitizerOptions,
    "cute.compiler": CuteCompilerOptions,
    "nvbench": NvbenchOptions,
    "triton.compiler": TritonCompilerOptions,
}


def adapter_accepts_options(adapter: str) -> bool:
    return adapter == "torch.profiler" or adapter in _ADAPTER_OPTION_MODELS


def _validate_adapter_options(
    adapter: str,
    options: dict[str, object] | None,
) -> ContractModel:
    model = _ADAPTER_OPTION_MODELS[adapter]
    try:
        return model.model_validate(options or {})
    except ValueError as exc:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"Invalid {adapter} capture options.",
            details={"validation_error": str(exc)},
        ) from exc


def bind_adapter_options(
    adapter: str,
    options: dict[str, JsonValue] | None,
    *,
    project_root: Path,
) -> dict[str, JsonValue]:
    if adapter == "torch.profiler":
        selected = torch_profiler_options(cast(dict[str, object] | None, options))
        return cast(dict[str, JsonValue], selected.model_dump(mode="json"))
    if adapter == "compute-sanitizer":
        raw = dict(options or {})
        suppression = raw.get("suppression_file")
        if suppression is not None:
            if not isinstance(suppression, str):
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "Compute Sanitizer suppression_file must be a project-relative string.",
                )
            path = _contained_project_file(project_root, suppression)
            raw["suppression_file"] = path.relative_to(project_root.resolve()).as_posix()
            raw["suppression_digest"] = _sha256(path)
        bound = _validate_adapter_options(adapter, cast(dict[str, object], raw))
        return cast(dict[str, JsonValue], bound.model_dump(mode="json"))
    if adapter in _ADAPTER_OPTION_MODELS:
        bound = _validate_adapter_options(adapter, cast(dict[str, object] | None, options))
        return cast(dict[str, JsonValue], bound.model_dump(mode="json"))
    if options:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"Adapter {adapter!r} does not accept capture options.",
        )
    return {}


def compute_sanitizer_options(options: dict[str, object] | None) -> ComputeSanitizerOptions:
    return cast(ComputeSanitizerOptions, _validate_adapter_options("compute-sanitizer", options))


def compute_sanitizer_suppression_path(
    options: ComputeSanitizerOptions,
    *,
    project_root: Path,
) -> Path | None:
    if options.suppression_file is None:
        return None
    path = _contained_project_file(project_root, options.suppression_file)
    observed_digest = _sha256(path)
    if observed_digest != options.suppression_digest:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "Compute Sanitizer suppression file changed after the capture plan was issued.",
            details={
                "expected_sha256": options.suppression_digest,
                "observed_sha256": observed_digest,
            },
            remediation=("Create a new capture plan after updating the suppression file.",),
        )
    return path


def read_compute_sanitizer_suppression(
    options: ComputeSanitizerOptions,
    *,
    project_root: Path,
) -> bytes | None:
    """Read and verify the plan-bound suppression bytes from a no-follow descriptor."""
    if options.suppression_file is None:
        return None
    path = _contained_project_file(project_root, options.suppression_file)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Compute Sanitizer suppression file must be a regular non-linked file.",
            )
        if metadata.st_size > 1024 * 1024:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Compute Sanitizer suppression file exceeds the 1 MiB limit.",
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(1024 * 1024 + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > 1024 * 1024:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Compute Sanitizer suppression file exceeds the 1 MiB limit.",
        )
    observed_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if observed_digest != options.suppression_digest:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "Compute Sanitizer suppression file changed after the capture plan was issued.",
            details={
                "expected_sha256": options.suppression_digest,
                "observed_sha256": observed_digest,
            },
            remediation=("Create a new capture plan after updating the suppression file.",),
        )
    return payload


def nvbench_options(options: dict[str, object] | None) -> NvbenchOptions:
    return cast(NvbenchOptions, _validate_adapter_options("nvbench", options))


def triton_compiler_options(options: dict[str, object] | None) -> TritonCompilerOptions:
    return cast(TritonCompilerOptions, _validate_adapter_options("triton.compiler", options))


def cute_compiler_options(options: dict[str, object] | None) -> CuteCompilerOptions:
    return cast(CuteCompilerOptions, _validate_adapter_options("cute.compiler", options))


def _contained_project_file(project_root: Path, relative: str) -> Path:
    root = project_root.resolve()
    lexical = Path(relative)
    if lexical.is_absolute() or ".." in lexical.parts:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Compute Sanitizer suppression files must be project-relative.",
        )
    path = (root / lexical).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Compute Sanitizer suppression file escapes the project root.",
        ) from exc
    try:
        metadata = path.stat()
    except OSError as exc:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Compute Sanitizer suppression file is missing.",
        ) from exc
    if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Compute Sanitizer suppression file must be a regular non-linked file.",
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stream = os.fdopen(descriptor, "rb")
    except OSError:
        os.close(descriptor)
        raise
    with stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
