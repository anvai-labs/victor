"""Windows external-tool ownership through an unnamed, non-breakaway Job Object.

The isolated leader blocks on GO until assignment to the job succeeds. Every
normal descendant inherits membership. Cleanup terminates the job handle rather
than discovered PIDs. RSS is polled through membership-checked process handles;
a separate hard aggregate committed-memory limit uses the same byte budget.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack
import ctypes as ct
import math
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from victor.core.graph_rag.windows_source import _DWORD, _HANDLE, _failure

if TYPE_CHECKING:
    from victor.core.graph_rag.bounded_process import OwnedProcess

_LEADER = """import pathlib, subprocess, sys, time
status = pathlib.Path(sys.argv[1])
if sys.stdin.buffer.readline() != b'GO\\n':
    raise SystemExit(125)
try:
    child = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL)
    result = child.wait()
except OSError as exc:
    print(str(exc), file=sys.stderr)
    result = 125
temporary = status.with_suffix('.tmp')
temporary.write_text(str(result), encoding='ascii')
temporary.replace(status)
while True:
    time.sleep(3600)
"""


class _BasicLimits(ct.Structure):
    _fields_ = [
        ("process_time", ct.c_int64),
        ("job_time", ct.c_int64),
        ("flags", _DWORD),
        ("minimum", ct.c_size_t),
        ("maximum", ct.c_size_t),
        ("active", _DWORD),
        ("affinity", ct.c_size_t),
        ("priority", _DWORD),
        ("scheduling", _DWORD),
    ]


class _IoCounters(ct.Structure):
    _fields_ = [
        (name, ct.c_uint64) for name in ("read", "write", "other", "rbytes", "wbytes", "obytes")
    ]


class _ExtendedLimits(ct.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io", _IoCounters),
        ("process_memory", ct.c_size_t),
        ("job_memory", ct.c_size_t),
        ("peak_process", ct.c_size_t),
        ("peak_job", ct.c_size_t),
    ]


class _Memory(ct.Structure):
    _fields_ = [
        ("size", _DWORD),
        ("faults", _DWORD),
        ("peak_working", ct.c_size_t),
        ("working", ct.c_size_t),
        ("peak_paged", ct.c_size_t),
        ("paged", ct.c_size_t),
        ("peak_nonpaged", ct.c_size_t),
        ("nonpaged", ct.c_size_t),
        ("pagefile", ct.c_size_t),
        ("peak_pagefile", ct.c_size_t),
    ]


class _Job:
    def __init__(self, limit: int) -> None:
        if type(limit) is not int or not 1 <= limit < 1 << (8 * ct.sizeof(ct.c_size_t)):
            raise ValueError("Invalid Windows job memory budget")
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        psapi = ct.WinDLL("psapi", use_last_error=True)

        def bind(dll: Any, name: str, result: Any, arguments: list[Any]) -> Any:
            function = getattr(dll, name)
            function.restype, function.argtypes = result, arguments
            return function

        self.close_handle = bind(kernel, "CloseHandle", ct.c_int, [_HANDLE])
        create = bind(kernel, "CreateJobObjectW", _HANDLE, [_HANDLE, ct.c_wchar_p])
        configure = bind(
            kernel, "SetInformationJobObject", ct.c_int, [_HANDLE, ct.c_int, _HANDLE, _DWORD]
        )
        self.assign = bind(kernel, "AssignProcessToJobObject", ct.c_int, [_HANDLE, _HANDLE])
        self.terminate = bind(kernel, "TerminateJobObject", ct.c_int, [_HANDLE, ct.c_uint])
        self.query = bind(
            kernel,
            "QueryInformationJobObject",
            ct.c_int,
            [_HANDLE, ct.c_int, _HANDLE, _DWORD, ct.POINTER(_DWORD)],
        )
        self.open_process = bind(kernel, "OpenProcess", _HANDLE, [_DWORD, ct.c_int, _DWORD])
        self.member = bind(
            kernel, "IsProcessInJob", ct.c_int, [_HANDLE, _HANDLE, ct.POINTER(ct.c_int)]
        )
        self.memory = bind(
            psapi, "GetProcessMemoryInfo", ct.c_int, [_HANDLE, ct.POINTER(_Memory), _DWORD]
        )
        self.handle = create(None, None)
        if not self.handle:
            raise _failure()
        try:
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
            limits.job_memory = limit
            if not configure(self.handle, 9, ct.byref(limits), ct.sizeof(limits)):
                raise _failure()
        except BaseException:
            self.close_handle(self.handle)
            raise

    def close(self) -> None:
        try:
            if not self.terminate(self.handle, 125):
                raise _failure()
        finally:
            self.close_handle(self.handle)

    def attach(self, process: subprocess.Popen[bytes]) -> None:
        # Popen owns this retained process handle; no PID lookup can retarget it.
        handle = int(process._handle)  # type: ignore[attr-defined]  # Windows Popen owns HANDLE.
        if not self.assign(self.handle, handle):
            raise _failure()

    def rss(self) -> int:
        # JOBOBJECT_BASIC_PROCESS_ID_LIST: two DWORDs then ULONG_PTR IDs.
        buffer = ct.create_string_buffer(8 + ct.sizeof(ct.c_size_t) * 4096)
        returned = _DWORD()
        if not self.query(self.handle, 3, buffer, len(buffer), ct.byref(returned)):
            raise _failure()
        assigned = _DWORD.from_buffer(buffer, 0).value
        count = _DWORD.from_buffer(buffer, 4).value
        if count > 4096 or count < assigned:
            raise ValueError("Windows external-tool job exceeded its process budget")
        total = 0
        for index in range(count):
            pid = ct.c_size_t.from_buffer(buffer, 8 + index * ct.sizeof(ct.c_size_t)).value
            handle = self.open_process(0x1000 | 0x10, False, pid)
            if not handle:
                if ct.get_last_error() == 87:  # Process exited between query and open.
                    continue
                raise _failure()
            try:
                member = ct.c_int()
                if not self.member(handle, self.handle, ct.byref(member)):
                    raise _failure()
                if not member.value:
                    continue  # A reused PID is never measured or terminated.
                memory = _Memory()
                memory.size = ct.sizeof(memory)
                if not self.memory(handle, ct.byref(memory), ct.sizeof(memory)):
                    raise _failure()
                total += memory.working
            finally:
                self.close_handle(handle)
        return total


def updates(job: OwnedProcess) -> Generator[float, None, tuple[int, str]]:
    """Run the owned Windows job with cancellation-safe generator cleanup."""
    if (
        not math.isfinite(job.timeout)
        or job.timeout <= 0
        or job.max_rss < 1
        or any(limit < 1 for limit in job.artifacts.values())
    ):
        raise ValueError("Invalid external tool resource budget")
    if (
        job.cwd_descriptor is not None
        or job.pass_fds
        or job.cwd_ancestry
        or job.cwd_context
        or job.cwd_marker
    ):
        raise ValueError("Windows external tools do not accept POSIX descriptor context")
    work = job.work.absolute()
    status = work / "backend.status"
    if any(
        path.exists() or path.is_symlink()
        for path in (
            status,
            status.with_suffix(".tmp"),
            status.with_suffix(".pid"),
            status.with_suffix(".pid.tmp"),
        )
    ):
        raise ValueError("External tool requires fresh owned status paths")
    with ExitStack() as stack:
        owned = _Job(job.max_rss)
        stack.callback(owned.close)
        log = stack.enter_context((work / "backend.log").open("wb"))
        errors = (
            stack.enter_context((work / "backend.stderr").open("wb"))
            if job.separate_stderr
            else log
        )
        started = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _LEADER, str(status), *job.arguments],
            cwd=job.cwd if job.cwd is not None else work,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=errors,
            env=job.environment,
            close_fds=True,
        )
        try:
            owned.attach(process)
            if process.stdin is None:
                raise ValueError("Windows external-tool leader has no ownership handshake")
            process.stdin.write(b"GO\n")
            process.stdin.close()
            while True:
                if any(
                    path.exists() and path.stat().st_size > limit
                    for path, limit in job.artifacts.items()
                ):
                    raise ValueError("External tool exceeded its artifact budget")
                if status.is_file():
                    if status.stat().st_size > 32:
                        raise ValueError("Invalid external tool exit status")
                    code = int(status.read_text(encoding="ascii"))
                    return code, "owned-windows-job-polled-rss-and-hard-commit"
                if process.poll() is not None:
                    raise ValueError("Windows external-tool leader exited without a status")
                if time.monotonic() - started > job.timeout:
                    raise ValueError("External tool exceeded its time budget")
                if owned.rss() > job.max_rss:
                    raise ValueError("External tool exceeded its memory budget")
                yield 0.02
        finally:
            # Job teardown happens before wait; leader never spawns outside it.
            try:
                owned.terminate(owned.handle, 125)
                if process.poll() is None:
                    process.kill()  # Retained Popen HANDLE also covers failed assignment.
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                process.wait()
