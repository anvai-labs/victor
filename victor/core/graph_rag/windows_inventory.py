"""Bounded Windows source inventory using pinned namespace and Job Objects."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import time

from victor.core.graph_rag.bounded_process import OwnedProcess
from victor.core.graph_rag.source_inventory import (
    INVENTORY_TIMEOUT,
    MAX_WALK_ENTRIES,
    MAX_WALK_DEPTH,
    MAX_WALK_DIRECTORIES,
    MAX_INVENTORY_BYTES,
    MAX_INVENTORY_RSS,
    _SKIP,
)


def source_files(handle: int, suffixes: set[str]) -> list[str]:
    """Bound local handle enumeration and pinned ordinary-Git inventory."""
    from victor.core.graph_rag.windows_source import (
        _Api,
        _DIRECTORY,
        _REPARSE,
        _open,
        entries,
        git_context,
    )

    api = _Api()
    deadline = time.monotonic() + INVENTORY_TIMEOUT
    examined = directories = total_bytes = 0
    files: list[str] = []

    def check() -> None:
        nonlocal examined
        if time.monotonic() > deadline:
            raise ValueError("Source inventory exceeded its time budget")
        examined += 1
        if examined > MAX_WALK_ENTRIES:
            raise ValueError("Source inventory exceeded its examined-entry budget")

    def visit(directory: int, prefix: str, depth: int) -> None:
        nonlocal directories, total_bytes
        check()
        directories += 1
        if directories > MAX_WALK_DIRECTORIES or depth > MAX_WALK_DEPTH:
            raise ValueError("Source inventory exceeded its directory/depth budget")
        children = []
        for name, attributes in entries(api, directory):
            check()
            relative = f"{prefix}/{name}" if prefix else name
            if attributes & _DIRECTORY:
                if name.casefold() not in _SKIP:
                    if attributes & _REPARSE:
                        raise OSError("Source inventory rejects Windows reparse directories")
                    children.append((name, relative))
                continue
            if Path(name).suffix.lower() not in suffixes:
                continue
            total_bytes += len(relative.encode("utf-8")) + 1
            if total_bytes > MAX_INVENTORY_BYTES:
                raise ValueError("Source inventory exceeded its path budget")
            files.append(relative)
        for name, relative in sorted(children):
            check()
            child = _open(api, directory, name, True)
            try:
                visit(child, relative, depth + 1)
            finally:
                api.close(child)

    with git_context(api, handle, check) as context:
        binary = shutil.which("git")
        if context is not None and binary is None:
            raise OSError("Windows Git-backed source inventory requires Git")
        if context is not None:
            cwd, tree, metadata = context
            with tempfile.TemporaryDirectory(prefix="victor-source-inventory-") as directory:
                work = Path(directory)
                code, _ = OwnedProcess(
                    [
                        str(Path(binary).resolve()),
                        "--no-pager",
                        f"--git-dir={metadata}",
                        f"--work-tree={tree}",
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
                    ],
                    work,
                    INVENTORY_TIMEOUT,
                    max_rss=MAX_INVENTORY_RSS,
                    artifacts={
                        work / "backend.log": MAX_INVENTORY_BYTES,
                        work / "backend.stderr": 64 * 1024,
                    },
                    cwd=cwd,
                    recursive=False,
                    separate_stderr=True,
                    environment={
                        **{
                            key: value
                            for key, value in os.environ.items()
                            if not key.startswith("GIT_")
                        },
                        "LC_ALL": "C",
                        "GIT_OPTIONAL_LOCKS": "0",
                        "GIT_COMMON_DIR": metadata,
                    },
                ).run_sync()
                out = (work / "backend.log").read_bytes()
                errors = (work / "backend.stderr").read_bytes()
                if len(out) > MAX_INVENTORY_BYTES or len(errors) > 64 * 1024:
                    raise ValueError("Source inventory exceeded its artifact budget")
                if code != 0:
                    raise ValueError("Git source inventory failed; coverage is unavailable")
                if out and not out.endswith(b"\0"):
                    raise ValueError("Truncated Git source inventory")
                names = out.decode("utf-8", errors="strict").split("\0")
                files = sorted(
                    {name for name in names if name and Path(name).suffix.lower() in suffixes}
                )
        else:
            visit(handle, "", 0)
    return sorted(files)
