from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from flameox.runtime_errors import DomainError, ErrorCode


class BoundedFileSystem:
    """Descriptor-bound regular-file access beneath explicit trusted roots."""

    def __init__(self, trusted_roots: tuple[Path, ...]) -> None:
        if not trusted_roots:
            raise ValueError("bounded filesystem requires at least one trusted root")
        self.trusted_roots = tuple(Path(os.path.abspath(root)) for root in trusted_roots)

    @contextmanager
    def open_regular(
        self,
        path: Path,
        *,
        max_bytes: int | None = None,
        require_single_link: bool = False,
    ) -> Iterator[int]:
        descriptor = self.open_descriptor(path)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise DomainError(
                    ErrorCode.EXECUTION_FAILURE,
                    "Trusted-root access requires a regular file.",
                )
            if require_single_link and metadata.st_nlink != 1:
                raise DomainError(
                    ErrorCode.EXECUTION_FAILURE,
                    "Trusted-root access rejects mutable hard-linked files.",
                )
            if max_bytes is not None and metadata.st_size > max_bytes:
                raise DomainError(
                    ErrorCode.LIMIT_EXCEEDED,
                    "Trusted-root file exceeds the configured byte budget.",
                    details={"byte_length": metadata.st_size, "max_bytes": max_bytes},
                )
            yield descriptor
        finally:
            os.close(descriptor)

    def read_bytes(
        self,
        path: Path,
        *,
        max_bytes: int,
        require_single_link: bool = False,
    ) -> bytes:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        with self.open_regular(
            path,
            max_bytes=max_bytes,
            require_single_link=require_single_link,
        ) as descriptor:
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise DomainError(
                ErrorCode.LIMIT_EXCEEDED,
                "Trusted-root file exceeded the configured byte budget while reading.",
                details={"max_bytes": max_bytes},
            )
        return payload

    def open_descriptor(self, path: Path) -> int:
        """Open a no-follow descriptor; the caller owns and must close it."""
        absolute = Path(os.path.abspath(path))
        for root in self.trusted_roots:
            try:
                relative = absolute.relative_to(root)
            except ValueError:
                continue
            if not relative.parts:
                continue
            try:
                return self._open_beneath(root, relative)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
                    raise DomainError(
                        ErrorCode.EXECUTION_FAILURE,
                        "Trusted-root file is missing or contains a symbolic link.",
                    ) from exc
                raise
        raise DomainError(
            ErrorCode.EXECUTION_FAILURE,
            "File is outside the trusted roots.",
        )

    @staticmethod
    def _open_beneath(root: Path, relative: Path) -> int:
        if os.name == "nt":
            return _open_windows_beneath(root, relative)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        source_flags = os.O_RDONLY
        for name in ("O_CLOEXEC", "O_NOFOLLOW"):
            directory_flags |= getattr(os, name, 0)
            source_flags |= getattr(os, name, 0)
        directory_fd = os.open(root, directory_flags)
        try:
            for component in relative.parts[:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            return os.open(relative.parts[-1], source_flags, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)


def _open_windows_beneath(root: Path, relative: Path) -> int:
    candidate = root.joinpath(*relative.parts)
    _reject_windows_reparse_points(root, relative)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    descriptor = os.open(candidate, flags)
    try:
        _reject_windows_reparse_points(root, relative)
        final_path = _windows_final_path(descriptor)
        try:
            final_path.relative_to(root.resolve(strict=True))
        except ValueError as exc:
            raise OSError(
                errno.ELOOP,
                "Opened trusted-root file resolves outside its root.",
            ) from exc
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _reject_windows_reparse_points(root: Path, relative: Path) -> None:
    paths = [root]
    for component in relative.parts:
        paths.append(paths[-1] / component)
    for index, path in enumerate(paths):
        metadata = os.lstat(path)
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
            or bool(getattr(path, "is_junction", lambda: False)())
        ):
            raise OSError(errno.ELOOP, "Trusted-root file contains a reparse point.")
        if index < len(paths) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise OSError(errno.ENOTDIR, "Trusted-root file parent is not a directory.")


def _windows_final_path(descriptor: int) -> Path:
    import ctypes

    msvcrt = __import__("msvcrt")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    get_final_path.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_final_path(msvcrt.get_osfhandle(descriptor), buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        raise OSError(error or errno.EACCES, "Could not verify the opened trusted-root path.")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)
