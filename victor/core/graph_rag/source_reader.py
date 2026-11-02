"""Bounded project-relative source reads without following links or special files.

Descriptor-relative traversal is required. Platforms lacking it fail this source
channel closed; a Windows handle-relative adapter needs separate qualification.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
import errno
import os
from pathlib import Path
import stat


@contextmanager
def project_source_reader(
    root: Path,
) -> Iterator[Callable[[str, int], tuple[bytes, os.stat_result]]]:
    """Pin the selected root before inventory and retain it through acquisition."""
    if os.open not in os.supports_dir_fd or not all(
        hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    ):
        raise OSError(errno.ENOTSUP, "Secure descriptor-relative source reads are unavailable")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield lambda relative, limit: _read_source(descriptor, relative, limit)
    finally:
        os.close(descriptor)


def read_project_source(root: Path, relative: str, limit: int) -> tuple[bytes, os.stat_result]:
    """Read one regular source file; scans must retain project_source_reader."""
    with project_source_reader(root) as read_source:
        return read_source(relative, limit)


def _read_source(root_descriptor: int, relative: str, limit: int) -> tuple[bytes, os.stat_result]:
    path = Path(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {".", ".."} for part in path.parts)
        or limit < 1
    ):
        raise OSError(errno.EINVAL, "Invalid project-relative source path or byte budget")
    with ExitStack() as stack:
        directory = root_descriptor
        for part in path.parts[:-1]:
            directory = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            stack.callback(os.close, directory)
        descriptor = os.open(
            path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        stack.callback(os.close, descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.EINVAL, "Selected source is not a regular file")
        if info.st_size > limit:
            return b"", info
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks), os.fstat(descriptor)
