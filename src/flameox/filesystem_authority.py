from __future__ import annotations

import errno
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel


class DirectoryIdentity(ContractModel):
    """Stable identity of a directory opened through the trusted-root boundary."""

    device: Annotated[int, Field(ge=0)]
    inode: Annotated[int, Field(gt=0)]
    guarantee: Literal["posix_dir_fd_no_links"] = "posix_dir_fd_no_links"


class BoundDirectoryReference(ContractModel):
    """Serializable authority reference stored in an immutable operation plan."""

    relative_path: Annotated[
        str,
        StringConstraints(min_length=1, max_length=500, pattern=r"^[A-Za-z0-9._/-]+$"),
    ]
    identity: DirectoryIdentity

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        try:
            _relative_parts(value)
        except DomainError as error:
            raise ValueError(error.message) from error
        return value

    def parts(self) -> tuple[str, ...]:
        return _relative_parts(self.relative_path)


class FileIdentity(ContractModel):
    """Identity and mutation-sensitive metadata for one admitted output file."""

    device: Annotated[int, Field(ge=0)]
    inode: Annotated[int, Field(gt=0)]
    byte_length: Annotated[int, Field(ge=0)]
    modified_ns: int
    links: Literal[1] = 1


class BoundFileReference(ContractModel):
    relative_path: Annotated[
        str,
        StringConstraints(min_length=1, max_length=500, pattern=r"^[A-Za-z0-9._/-]+$"),
    ]
    identity: FileIdentity

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        try:
            _relative_parts(value)
        except DomainError as error:
            raise ValueError(error.message) from error
        return value


class BoundDirectory:
    """An owned directory descriptor; display paths are never used for authority."""

    def __init__(
        self,
        *,
        descriptor: int,
        display_path: Path,
        reference: BoundDirectoryReference,
    ) -> None:
        self._descriptor = descriptor
        self.display_path = display_path
        self.reference = reference
        self._closed = False

    @property
    def descriptor(self) -> int:
        if self._closed:
            raise RuntimeError("bound directory is closed")
        return self._descriptor

    @property
    def identity(self) -> DirectoryIdentity:
        return self.reference.identity

    def child_process_path(self, relative_path: str) -> Path:
        parts = _relative_parts(relative_path)
        return self.child_process_root.joinpath(*parts)

    @property
    def child_process_root(self) -> Path:
        if not Path("/proc/self/fd").is_dir():
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Descriptor-bound child paths require a mounted /proc/self/fd filesystem.",
            )
        return Path(f"/proc/self/fd/{self.descriptor}")

    def absolute_display_path(self, relative_path: str) -> Path:
        return self.display_path.joinpath(*_relative_parts(relative_path))

    def inherited_descriptors(self) -> tuple[int, ...]:
        return (self.descriptor,)

    def ensure_directory(self, relative_path: str, *, mode: int = 0o700) -> None:
        descriptor = self.descriptor
        owned = False
        try:
            for component in _relative_parts(relative_path):
                with suppress(FileExistsError):
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                child = _open_directory_at(
                    descriptor,
                    component,
                    expected_device=self.identity.device,
                )
                if owned:
                    os.close(descriptor)
                descriptor = child
                owned = True
        finally:
            if owned:
                os.close(descriptor)

    @contextmanager
    def open_file(self, reference: str | BoundFileReference) -> Iterator[int]:
        relative_path = (
            reference.relative_path if isinstance(reference, BoundFileReference) else reference
        )
        expected_identity = (
            reference.identity if isinstance(reference, BoundFileReference) else None
        )
        parts = _relative_parts(relative_path)
        parent, final_name = self._open_parent(parts)
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(final_name, flags, dir_fd=parent)
            except OSError as exc:
                raise _authority_error("Could not open a trusted-root file.", exc) from exc
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_dev != self.identity.device
                ):
                    raise DomainError(
                        ErrorCode.EXECUTION_REFUSED,
                        "Trusted-root file target must be a singly linked regular file.",
                    )
                actual_identity = _file_identity(metadata)
                if expected_identity is not None and actual_identity != expected_identity:
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                        "Trusted output file changed after manifest admission.",
                    )
                yield descriptor
            finally:
                os.close(descriptor)
        finally:
            if parent != self.descriptor:
                os.close(parent)

    def write_bytes(self, relative_path: str, payload: bytes, *, mode: int = 0o600) -> None:
        parts = _relative_parts(relative_path)
        parent, final_name = self._open_parent(parts, create=True)
        temporary_name = f".{final_name}.{secrets.token_hex(16)}.tmp"
        temporary: int | None = None
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            temporary = os.open(temporary_name, flags, mode, dir_fd=parent)
            _write_all(temporary, payload)
            os.fsync(temporary)
            os.close(temporary)
            temporary = None
            os.replace(
                temporary_name,
                final_name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
            )
            os.fsync(parent)
        except OSError as exc:
            raise _authority_error(
                "Could not atomically write beneath a trusted root.", exc
            ) from exc
        finally:
            if temporary is not None:
                os.close(temporary)
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent)
            if parent != self.descriptor:
                os.close(parent)

    def read_bytes(self, relative_path: str, *, max_bytes: int) -> bytes:
        with self.open_file(relative_path) as descriptor:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
                if not chunk:
                    return b"".join(chunks)
                total += len(chunk)
                if total > max_bytes:
                    raise DomainError(
                        ErrorCode.ARTIFACT_TOO_LARGE,
                        "Trusted-root file exceeded its read budget.",
                    )
                chunks.append(chunk)

    def admitted_files(
        self,
        basenames: frozenset[str],
        *,
        suffixes: tuple[str, ...] = (),
        max_depth: int = 2,
        max_entries: int = 4_096,
        max_files: int = 32,
    ) -> tuple[BoundFileReference, ...]:
        """Discover only reviewed basenames through held directory descriptors."""

        found: list[BoundFileReference] = []
        observed = 0

        def visit(directory: int, prefix: tuple[str, ...], depth: int) -> None:
            nonlocal observed
            try:
                names = sorted(os.listdir(directory))
            except OSError as exc:
                raise _authority_error(
                    "Could not enumerate a trusted output directory.", exc
                ) from exc
            for name in names:
                observed += 1
                if observed > max_entries:
                    raise DomainError(
                        ErrorCode.QUERY_BUDGET_EXCEEDED,
                        "Trusted output discovery exceeded its entry budget.",
                    )
                if name in {".", ".."} or "/" in name or "\x00" in name:
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                        "Trusted output directory contains an invalid entry name.",
                    )
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                )
                try:
                    child = os.open(name, flags, dir_fd=directory)
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOENT}:
                        raise _authority_error(
                            "Trusted output entry changed or traversed a link.", exc
                        ) from exc
                    raise
                try:
                    metadata = os.fstat(child)
                    relative = (*prefix, name)
                    if stat.S_ISDIR(metadata.st_mode):
                        if metadata.st_dev != self.identity.device:
                            raise DomainError(
                                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                                "Trusted output discovery refused a mounted directory.",
                            )
                        if depth < max_depth:
                            visit(child, relative, depth + 1)
                    elif stat.S_ISREG(metadata.st_mode) and (
                        name in basenames or any(name.endswith(suffix) for suffix in suffixes)
                    ):
                        if metadata.st_nlink != 1:
                            raise DomainError(
                                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                                "Trusted output manifest refused a hard-linked file.",
                            )
                        found.append(
                            BoundFileReference(
                                relative_path=PurePosixPath(*relative).as_posix(),
                                identity=_file_identity(metadata),
                            )
                        )
                        if len(found) > max_files:
                            raise DomainError(
                                ErrorCode.QUERY_BUDGET_EXCEEDED,
                                "Trusted output manifest exceeded its file budget.",
                            )
                finally:
                    os.close(child)

        visit(self.descriptor, (), 0)
        return tuple(found)

    def _open_parent(
        self,
        parts: tuple[str, ...],
        *,
        create: bool = False,
    ) -> tuple[int, str]:
        descriptor = self.descriptor
        owned = False
        try:
            for component in parts[:-1]:
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = _open_directory_at(
                    descriptor,
                    component,
                    expected_device=self.identity.device,
                )
                if owned:
                    os.close(descriptor)
                descriptor = child
                owned = True
            return descriptor, parts[-1]
        except BaseException:
            if owned:
                os.close(descriptor)
            raise

    def close(self) -> None:
        if self._closed:
            return
        os.close(self._descriptor)
        self._closed = True

    def __enter__(self) -> BoundDirectory:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


class TrustedRoot:
    """Strict no-link POSIX root capability for control and staging data."""

    def __init__(self, root: Path) -> None:
        _require_posix_dir_fd_support()
        self.root_path = root.absolute()
        self._descriptor = _open_absolute_directory(self.root_path)
        self.identity = _directory_identity(self._descriptor)
        self._closed = False

    @property
    def descriptor(self) -> int:
        if self._closed:
            raise RuntimeError("trusted root is closed")
        return self._descriptor

    def allocate_directory(self, relative_path: str, *, mode: int = 0o700) -> BoundDirectory:
        parts = _relative_parts(relative_path)
        parent = self._open_components(parts[:-1], create=True)
        try:
            try:
                os.mkdir(parts[-1], mode=mode, dir_fd=parent)
            except OSError as exc:
                raise _authority_error(
                    "Could not exclusively allocate an operation directory.", exc
                ) from exc
            descriptor = _open_directory_at(
                parent,
                parts[-1],
                expected_device=self.identity.device,
            )
        finally:
            if parent != self.descriptor:
                os.close(parent)
        identity = _directory_identity(descriptor)
        reference = BoundDirectoryReference(
            relative_path=PurePosixPath(*parts).as_posix(),
            identity=identity,
        )
        return BoundDirectory(
            descriptor=descriptor,
            display_path=self.root_path.joinpath(*parts),
            reference=reference,
        )

    def open_directory(self, reference: BoundDirectoryReference) -> BoundDirectory:
        try:
            descriptor = self._open_components(reference.parts())
        except DomainError as error:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Trusted operation directory changed after planning.",
                details={"cause": error.message},
            ) from error
        identity = _directory_identity(descriptor)
        if identity != reference.identity:
            os.close(descriptor)
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Trusted operation directory identity changed after planning.",
            )
        return BoundDirectory(
            descriptor=descriptor,
            display_path=self.root_path.joinpath(*reference.parts()),
            reference=reference,
        )

    def remove_directory(self, reference: BoundDirectoryReference) -> None:
        parts = reference.parts()
        parent = self._open_components(parts[:-1])
        try:
            descriptor = _open_directory_at(
                parent,
                parts[-1],
                expected_device=self.identity.device,
            )
            try:
                if _directory_identity(descriptor) != reference.identity:
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                        "Trusted operation directory identity changed before cleanup.",
                    )
                _remove_contents(descriptor)
            finally:
                os.close(descriptor)
            os.rmdir(parts[-1], dir_fd=parent)
            os.fsync(parent)
        except OSError as exc:
            raise _authority_error("Could not remove a trusted operation directory.", exc) from exc
        finally:
            if parent != self.descriptor:
                os.close(parent)

    def _open_components(
        self,
        parts: tuple[str, ...],
        *,
        create: bool = False,
    ) -> int:
        descriptor = os.dup(self.descriptor)
        try:
            for component in parts:
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = _open_directory_at(
                    descriptor,
                    component,
                    expected_device=self.identity.device,
                )
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def close(self) -> None:
        if self._closed:
            return
        os.close(self._descriptor)
        self._closed = True

    def __enter__(self) -> TrustedRoot:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


def _require_posix_dir_fd_support() -> None:
    required = (os.open, os.mkdir, os.unlink, os.rmdir, os.stat)
    if os.name != "posix" or any(function not in os.supports_dir_fd for function in required):
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "This platform cannot provide descriptor-bound trusted-root filesystem operations.",
        )


def _relative_parts(value: str) -> tuple[str, ...]:
    if "\x00" in value or "\\" in value:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Trusted-root paths must be POSIX relative paths.",
        )
    path = PurePosixPath(value)
    parts = path.parts
    normalized = PurePosixPath(*parts).as_posix() if parts else ""
    if (
        path.is_absolute()
        or not parts
        or normalized != value
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Trusted-root paths must contain only non-traversing relative components.",
        )
    return parts


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute():
        raise DomainError(ErrorCode.EXECUTION_REFUSED, "Trusted roots must be absolute paths.")
    descriptor = os.open(
        "/",
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for component in path.parts[1:]:
            child = _open_directory_at(descriptor, component)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_at(parent: int, name: str, *, expected_device: int | None = None) -> int:
    flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as exc:
        raise _authority_error(
            "Trusted-root traversal refused a directory component.", exc
        ) from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Trusted-root traversal encountered a non-directory component.",
        )
    if expected_device is not None and metadata.st_dev != expected_device:
        os.close(descriptor)
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Trusted-root traversal refused a mounted directory.",
        )
    return descriptor


def _directory_identity(descriptor: int) -> DirectoryIdentity:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise DomainError(ErrorCode.EXECUTION_REFUSED, "Bound object is not a directory.")
    return DirectoryIdentity(device=metadata.st_dev, inode=metadata.st_ino)


def _file_identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        byte_length=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        links=1,
    )


def _remove_contents(directory: int) -> None:
    for name in os.listdir(directory):
        child = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory,
        )
        try:
            metadata = os.fstat(child)
            if stat.S_ISDIR(metadata.st_mode):
                _remove_contents(child)
                os.rmdir(name, dir_fd=directory)
            elif stat.S_ISREG(metadata.st_mode):
                os.unlink(name, dir_fd=directory)
            else:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Trusted cleanup encountered an unsupported object type.",
                )
        finally:
            os.close(child)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def _authority_error(message: str, error: OSError) -> DomainError:
    code = (
        ErrorCode.ARTIFACT_INTEGRITY_FAILED
        if error.errno in {errno.ELOOP, errno.ESTALE}
        else ErrorCode.EXECUTION_REFUSED
    )
    return DomainError(code, message, details={"errno": error.errno})
