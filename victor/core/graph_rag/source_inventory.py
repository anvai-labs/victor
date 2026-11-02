"""Inventory the owned source directory, with bounded, callback-free Git I/O."""

from __future__ import annotations

import os
import hashlib
import posixpath
import stat
from pathlib import Path
import shutil
import tempfile
import time

from victor.core.graph_rag.bounded_process import OwnedProcess

INVENTORY_TIMEOUT = 10.0
MAX_INVENTORY_BYTES = 16 * 1024 * 1024
MAX_INVENTORY_RSS = 256 * 1024 * 1024
MAX_WALK_ENTRIES = 200_000
MAX_WALK_DIRECTORIES = 50_000
MAX_WALK_DEPTH = 128
_SKIP = {".git", ".venv", "venv", "node_modules", "build", "dist", ".victor"}


def source_files(descriptor: int, suffixes: set[str]) -> list[str]:
    """List through the pinned directory; reject changed Git context at boundaries.

    Git has its cwd bound to the inherited descriptor. Only an unavailable Git
    executable or Git's explicit not-a-repository result uses filesystem policy;
    timeouts, resource limits and malformed metadata fail this scan closed.
    """
    if os.name == "nt":  # pragma: no cover -- exercised by real Windows qualification.
        from victor.core.graph_rag.windows_inventory import source_files as windows_files

        return windows_files(descriptor, suffixes)
    if os.name != "posix" or not hasattr(os, "fchdir"):
        raise OSError("Descriptor-bound source inventory is unavailable")
    binary = shutil.which("git")
    if binary is None:
        return _walk(descriptor, suffixes)
    binary = str(Path(binary).resolve())
    work_tree, ancestry, marker, context = _git_work_tree(descriptor)
    with tempfile.TemporaryDirectory(prefix="victor-source-inventory-") as directory:
        work = Path(directory)
        arguments = [
            binary,
            "--no-pager",
            *([f"--work-tree={work_tree}"] if work_tree is not None else []),
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ]
        code, _ = OwnedProcess(
            arguments,
            work,
            INVENTORY_TIMEOUT,
            max_rss=MAX_INVENTORY_RSS,
            artifacts={
                work / "backend.log": MAX_INVENTORY_BYTES,
                work / "backend.stderr": 64 * 1024,
            },
            cwd_descriptor=descriptor,
            cwd_ancestry=ancestry,
            cwd_marker=marker,
            cwd_context=context,
            pass_fds=(descriptor,),
            recursive=False,
            separate_stderr=True,
            environment={
                **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
                "LC_ALL": "C",
                "GIT_OPTIONAL_LOCKS": "0",
            },
        ).run_sync()
        out, errors = (work / "backend.log").read_bytes(), (work / "backend.stderr").read_bytes()
        if len(out) > MAX_INVENTORY_BYTES or len(errors) > 64 * 1024:
            raise ValueError("Source inventory exceeded its artifact budget")
        if code != 0:
            if (
                code == 128
                and errors.startswith(b"fatal: not a git repository")
                and work_tree is None
            ):
                return _walk(descriptor, suffixes)
            raise ValueError("Git source inventory failed; coverage is unavailable")
        if out and not out.endswith(b"\0"):
            raise ValueError("Truncated Git source inventory")
        try:
            names = out.decode("utf-8", errors="strict").split("\0")
        except UnicodeError as exc:
            raise ValueError("Source inventory requires UTF-8 paths") from exc
        return sorted({name for name in names if name and Path(name).suffix.lower() in suffixes})


def _git_work_tree(
    descriptor: int,
) -> tuple[
    str | None,
    tuple[tuple[int, int], ...],
    tuple[str, int, int] | None,
    tuple[tuple[str, int | None, int | None, str | None], ...],
]:
    """Locate Git metadata through directory ancestry, ignoring config redirects.

    Keep Git in the owned subdirectory and set its worktree to the discovered
    ancestor via a relative path. Git retains index prefixes and ancestor ignore
    rules; pathname replacements cannot redirect its working directory.
    """
    current = os.dup(descriptor)
    ancestry: list[tuple[int, int]] = []
    try:
        for depth in range(1024):
            info = os.fstat(current)
            ancestry.append((info.st_dev, info.st_ino))
            try:
                metadata = os.stat(".git", dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                return (
                    "/".join([".."] * depth) if depth else ".",
                    tuple(ancestry),
                    (".git", metadata.st_dev, metadata.st_ino),
                    _repository_context(current, metadata),
                )
            before = os.fstat(current)
            parent = os.open("..", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            after = os.fstat(parent)
            os.close(current)
            current = parent
            if (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino):
                return None, tuple(ancestry), None, ()
        raise ValueError("Source inventory exceeded its ancestor budget")
    finally:
        os.close(current)


def _selector(directory: int, path: str, prefix: bytes = b"") -> tuple[str, os.stat_result, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            raise ValueError("Unsupported Git repository selector")
        raw = os.read(descriptor, 4097)
        if len(raw) > 4096 or not raw.startswith(prefix):
            raise ValueError("Invalid Git repository selector")
        target = raw[len(prefix) :].rstrip(b"\r\n").decode("utf-8", errors="strict")
        if not target or "\n" in target or "\r" in target or "\0" in target:
            raise ValueError("Invalid Git repository selector target")
        return target, info, hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


def _repository_context(
    directory: int, metadata: os.stat_result
) -> tuple[tuple[str, int | None, int | None, str | None], ...]:
    entries: list[tuple[str, int | None, int | None, str | None]] = []
    if stat.S_ISDIR(metadata.st_mode):
        git_directory = ".git"
    elif stat.S_ISREG(metadata.st_mode):
        git_directory, info, digest = _selector(directory, ".git", b"gitdir: ")
        entries.append((".git", info.st_dev, info.st_ino, digest))
    else:
        raise ValueError("Unsupported Git repository metadata type")
    target = os.stat(git_directory, dir_fd=directory)
    if not stat.S_ISDIR(target.st_mode):
        raise ValueError("Git repository target is not a directory")
    entries.append((git_directory, target.st_dev, target.st_ino, None))
    common_selector = posixpath.join(git_directory, "commondir")
    try:
        os.stat(common_selector, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        entries.append((common_selector, None, None, None))
    else:
        common, info, digest = _selector(directory, common_selector)
        entries.append((common_selector, info.st_dev, info.st_ino, digest))
        common_path = posixpath.join(git_directory, common)
        target = os.stat(common_path, dir_fd=directory)
        if not stat.S_ISDIR(target.st_mode):
            raise ValueError("Git common repository target is not a directory")
        entries.append((common_path, target.st_dev, target.st_ino, None))
    return tuple(entries)


def _walk(descriptor: int, suffixes: set[str]) -> list[str]:
    files: list[str] = []
    total_bytes = 0
    examined = 0
    directories = 0
    deadline = time.monotonic() + INVENTORY_TIMEOUT

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ValueError("Source inventory exceeded its time budget")

    def visit(directory: int, prefix: str, depth: int) -> None:
        nonlocal total_bytes, examined, directories
        check_time()
        directories += 1
        if directories > MAX_WALK_DIRECTORIES or depth > MAX_WALK_DEPTH:
            raise ValueError("Source inventory exceeded its directory/depth budget")
        children = []
        with os.scandir(directory) as entries:
            for entry in entries:
                check_time()
                examined += 1
                if examined > MAX_WALK_ENTRIES:
                    raise ValueError("Source inventory exceeded its examined-entry budget")
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _SKIP:
                        children.append(entry.name)
                    continue
                if Path(entry.name).suffix.lower() not in suffixes:
                    continue
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                total_bytes += len(relative.encode("utf-8")) + 1
                if total_bytes > MAX_INVENTORY_BYTES:
                    raise ValueError("Source inventory exceeded its path budget")
                files.append(relative)
        for name in sorted(children):
            check_time()
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            try:
                visit(child, f"{prefix}/{name}" if prefix else name, depth + 1)
            finally:
                os.close(child)

    visit(descriptor, "", 0)
    return sorted(files)
