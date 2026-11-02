"""Owned external-tool lifetime, output and memory limits for source analysis."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass
from collections.abc import Generator
import math
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, cast

_SUPERVISOR = """import hashlib, json, os, stat, subprocess, sys, time
from pathlib import Path
status = Path(sys.argv[1])
if sys.argv[2] != '-':
    os.fchdir(int(sys.argv[2]))
expected, marker, context = json.loads(sys.argv[3])
def verify_ancestry():
    if not expected:
        return
    current = os.open('.', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, identity in enumerate(expected):
            info = os.fstat(current)
            if [info.st_dev, info.st_ino] != identity:
                raise RuntimeError('Source inventory directory ancestry changed')
            if index + 1 < len(expected) or marker is None:
                try:
                    os.stat('.git', dir_fd=current, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise RuntimeError('Source inventory nearest repository metadata changed')
            if index + 1 < len(expected):
                parent = os.open('..', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                os.close(current)
                current = parent
        if marker is not None:
            info = os.stat(marker[0], dir_fd=current, follow_symlinks=False)
            if [info.st_dev, info.st_ino] != marker[1:]:
                raise RuntimeError('Source inventory repository metadata changed')
        for path, device, inode, digest in context:
            try:
                info = os.stat(path, dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                if device is None:
                    continue
                raise
            if device is None:
                raise RuntimeError('Source inventory repository selector appeared')
            if digest is None:
                info = os.stat(path, dir_fd=current)
                if (info.st_dev, info.st_ino) != (device, inode) or not stat.S_ISDIR(info.st_mode):
                    raise RuntimeError('Source inventory repository target changed')
            else:
                selected = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
                try:
                    info = os.fstat(selected)
                    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (device, inode):
                        raise RuntimeError('Source inventory repository selector changed')
                    if info.st_size > 4096 or hashlib.sha256(os.read(selected, 4097)).hexdigest() != digest:
                        raise RuntimeError('Source inventory repository selector contents changed')
                finally:
                    os.close(selected)
    finally:
        os.close(current)
try:
    verify_ancestry()
    child = subprocess.Popen(sys.argv[4:])
    pid_temporary = status.with_suffix('.pid.tmp')
    pid_temporary.write_text(str(child.pid), encoding='ascii')
    pid_temporary.replace(status.with_suffix('.pid'))
    result = child.wait()
    verify_ancestry()
except (OSError, RuntimeError) as exc:
    print(str(exc), file=sys.stderr)
    result = 125
temporary = status.with_suffix('.tmp')
temporary.write_text(str(result), encoding='ascii')
temporary.replace(status)
while True:
    time.sleep(3600)
"""


@dataclass(frozen=True)
class OwnedProcess:
    """One owned call with synchronous and cancellable asynchronous consumers.

    Direct monitoring is reserved for Git with callbacks disabled. Recursive
    monitoring remains the detector default. The leader is never reaped before
    cleanup, so a reused process-group ID cannot target unrelated processes.
    """

    arguments: list[str]
    work: Path
    timeout: float
    max_rss: int
    artifacts: dict[Path, int]
    cwd: Path | str | None = None
    pass_fds: tuple[int, ...] = ()
    recursive: bool = True
    separate_stderr: bool = False
    environment: dict[str, str] | None = None
    cwd_descriptor: int | None = None
    cwd_ancestry: tuple[tuple[int, int], ...] = ()
    cwd_marker: tuple[str, int, int] | None = None
    cwd_context: tuple[tuple[str, int | None, int | None, str | None], ...] = ()

    def run_sync(self) -> tuple[int, str]:
        updates = self._updates()
        try:
            while True:
                try:
                    delay = next(updates)
                except StopIteration as done:
                    return cast(tuple[int, str], done.value)
                time.sleep(delay)
        finally:
            updates.close()

    async def run(self) -> tuple[int, str]:
        updates = self._updates()
        try:
            while True:
                try:
                    delay = next(updates)
                except StopIteration as done:
                    return cast(tuple[int, str], done.value)
                await asyncio.sleep(delay)
        finally:
            updates.close()

    def _updates(self) -> Generator[float, None, tuple[int, str]]:
        if os.name == "nt":  # pragma: no cover -- exercised by real Windows qualification.
            from victor.core.graph_rag.windows_process import updates

            return (yield from updates(self))
        arguments, work, timeout = self.arguments, self.work, self.timeout
        max_rss, artifacts, cwd = self.max_rss, self.artifacts, self.cwd
        pass_fds, recursive = self.pass_fds, self.recursive
        separate_stderr, environment = self.separate_stderr, self.environment
        cwd_descriptor = self.cwd_descriptor
        import psutil

        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or max_rss < 1
            or any(v < 1 for v in artifacts.values())
        ):
            raise ValueError("Invalid external tool resource budget")
        if cwd_descriptor is not None and cwd_descriptor not in pass_fds:
            raise ValueError("External tool cwd descriptor must be inherited")
        work = work.absolute()
        log_path, error_path = work / "backend.log", work / "backend.stderr"
        status_path = work / "backend.status"
        if any(
            path.exists() or path.is_symlink()
            for path in (
                status_path,
                status_path.with_suffix(".pid"),
                status_path.with_suffix(".pid.tmp"),
                status_path.with_suffix(".tmp"),
            )
        ):
            raise ValueError("External tool requires fresh owned status paths")
        with ExitStack() as stack:
            log = stack.enter_context(log_path.open("wb"))
            errors = stack.enter_context(error_path.open("wb")) if separate_stderr else log
            started = time.monotonic()
            memory_scope = (
                "owned-process-group" if recursive else "reserved-leader-and-declared-child"
            )
            owns_leader = True
            tool_exit_code = None
            observed: list[Any] = []
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _SUPERVISOR,
                    str(status_path),
                    str(cwd_descriptor) if cwd_descriptor is not None else "-",
                    json.dumps((self.cwd_ancestry, self.cwd_marker, self.cwd_context)),
                    *arguments,
                ],
                cwd=cwd if cwd is not None else work,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=errors,
                start_new_session=os.name == "posix",
                pass_fds=pass_fds,
                env=environment,
            )
            try:
                try:
                    parent = psutil.Process(process.pid)
                except psutil.NoSuchProcess as exc:
                    owns_leader = False
                    raise ValueError("External tool ownership was lost") from exc
                except (psutil.AccessDenied, PermissionError) as exc:
                    raise ValueError("External tool status is unavailable") from exc
                while True:
                    if os.name == "posix":
                        try:
                            finished = parent.status() == psutil.STATUS_ZOMBIE
                        except psutil.NoSuchProcess as exc:
                            owns_leader = False
                            raise ValueError("External tool ownership was lost") from exc
                        except (psutil.AccessDenied, PermissionError) as exc:
                            raise ValueError("External tool status is unavailable") from exc
                    else:
                        finished = process.poll() is not None
                    if any(
                        path.exists() and path.stat().st_size > limit
                        for path, limit in artifacts.items()
                    ):
                        raise ValueError("External tool exceeded its artifact budget")
                    if status_path.is_file():
                        if status_path.stat().st_size > 32:
                            raise ValueError("Invalid external tool exit status")
                        tool_exit_code = int(status_path.read_text(encoding="ascii"))
                        break
                    if finished:
                        raise ValueError("External tool supervisor exited without a status")
                    if time.monotonic() - started > timeout:
                        raise ValueError("External tool exceeded its time budget")
                    try:
                        if recursive:
                            try:
                                observed = parent.children(recursive=True)
                            except (psutil.AccessDenied, PermissionError) as exc:
                                raise ValueError(
                                    "External tool process-tree monitoring is unavailable"
                                ) from exc
                        else:
                            pid_path = status_path.with_suffix(".pid")
                            if pid_path.is_file():
                                if pid_path.stat().st_size > 32:
                                    raise ValueError("Invalid external tool child PID")
                                child_pid = int(pid_path.read_text(encoding="ascii"))
                                if child_pid <= 0:
                                    raise ValueError("Invalid external tool child PID")
                                observed = [psutil.Process(child_pid)]
                        if sum(child.memory_info().rss for child in [parent, *observed]) > max_rss:
                            raise ValueError("External tool exceeded its memory budget")
                    except psutil.NoSuchProcess:
                        pass
                    except (psutil.AccessDenied, PermissionError) as exc:
                        raise ValueError("External tool memory status is unavailable") from exc
                    yield 0.02
            finally:
                try:
                    if os.name == "posix" and owns_leader:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif os.name != "posix":
                        for child in observed:
                            try:
                                child.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                                pass
                        if process.poll() is None:
                            process.kill()
                finally:
                    process.wait()
            if tool_exit_code is None:
                raise ValueError("Missing external tool exit status")
            return tool_exit_code, memory_scope
