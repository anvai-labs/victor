# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared subprocess execution utilities.

This module provides consolidated subprocess execution functions used across
multiple tools (bash, git, docker, cicd, testing, etc.). Consolidating these
utilities reduces code duplication and ensures consistent error handling.

Features:
- Both sync and async subprocess execution
- Command safety checks with dangerous command blocking
- Specialized runners for git, docker, npm, pip
- Proper timeout handling with configurable defaults
- Output capture and structured result format
- Error categorization and logging
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from victor.tools.sandbox import SandboxBackend

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes and Enums
# =============================================================================


class CommandErrorType(Enum):
    """Types of command execution errors."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"  # Command not found
    PERMISSION_DENIED = "permission_denied"
    WORKING_DIR_NOT_FOUND = "working_dir_not_found"
    DANGEROUS_COMMAND = "dangerous_command"
    EXECUTION_ERROR = "execution_error"
    UNKNOWN = "unknown"


@dataclass
class CommandResult:
    """Result of command execution."""

    success: bool
    stdout: str
    stderr: str
    return_code: int
    error_type: CommandErrorType
    error_message: Optional[str] = None
    command: Optional[str] = None
    working_dir: Optional[str] = None
    duration_ms: Optional[float] = None
    truncated: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "command": self.command,
            "working_dir": self.working_dir,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
        }


# =============================================================================
# Safety Configuration
# =============================================================================


# Commands that should never be executed
# Consolidated dangerous command detection — single source of truth.
from victor.security.command_safety import (  # noqa: E402
    DANGEROUS_COMMANDS,
    DANGEROUS_PATTERNS,
    is_dangerous_command,
)

# =============================================================================
# Subprocess Resource Limits
# =============================================================================


def create_resource_limit_preexec(
    max_memory_mb: int = 512,
    max_cpu_seconds: int = 300,
    max_file_descriptors: int = 1024,
) -> "Optional[Callable[[], None]]":
    """Create a ``preexec_fn`` that applies POSIX resource limits.

    Returns ``None`` on non-POSIX platforms (Windows) where
    ``resource.setrlimit`` is unavailable.
    """
    try:
        import resource as _resource  # POSIX only
    except ImportError:
        return None

    def _apply_limits() -> None:
        try:
            mem_bytes = max_memory_mb * 1024 * 1024
            _resource.setrlimit(_resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass  # macOS may not enforce RLIMIT_AS
        try:
            _resource.setrlimit(
                _resource.RLIMIT_CPU,
                (max_cpu_seconds, max_cpu_seconds + 10),
            )
        except (ValueError, OSError):
            pass
        try:
            _resource.setrlimit(
                _resource.RLIMIT_NOFILE,
                (max_file_descriptors, max_file_descriptors),
            )
        except (ValueError, OSError):
            pass

    return _apply_limits


def _truncate_output(text: str, max_bytes: int) -> Tuple[str, bool]:
    """Truncate output to *max_bytes* with a marker. Returns (text, truncated).

    DEPRECATED: Use _truncate_output_by_lines() for line-based truncation.
    This is kept for backward compatibility.
    """
    if max_bytes <= 0 or len(text.encode("utf-8", errors="replace")) <= max_bytes:
        return text, False
    truncated = text.encode("utf-8", errors="replace")[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + "\n... [output truncated]", True


def _truncate_output_by_lines(
    text: str,
    max_lines: Optional[int],
    max_bytes: Optional[int] = None,
    stream_name: str = "output",
) -> Tuple[str, bool, int]:
    """Truncate output by lines with byte limit fallback.

    Args:
        text: Text to truncate
        max_lines: Maximum lines to keep (None=unlimited)
        max_bytes: Maximum bytes to keep (None=use tool settings default ONLY in unlimited mode)
        stream_name: Name of stream (for error messages)

    Returns:
        Tuple of (truncated_text, was_truncated, line_count)
    """
    # Default byte limit if not specified
    # Priority: use tool settings in unlimited mode, otherwise use 1MB safety default
    if max_bytes is None:
        if max_lines is None:
            # True unlimited mode - use tool settings (higher limit for accuracy-first)
            from victor.config.tool_settings import get_tool_settings

            tool_settings = get_tool_settings()
            # Convert MB to bytes
            max_bytes = int(tool_settings.tool_output_byte_limit_mb * 1024 * 1024)
        else:
            # Line limit is set, but no byte limit - use 1MB safety default
            max_bytes = 1024 * 1024  # 1MB safety default when line-limited

    # Handle unlimited
    if max_lines is None or max_lines <= 0:
        # Still enforce byte limit (safety)
        text_bytes = len(text.encode("utf-8", errors="replace"))
        if text_bytes > max_bytes:
            truncated = text.encode("utf-8", errors="replace")[:max_bytes].decode(
                "utf-8", errors="ignore"
            )
            return (
                truncated + f"\n... [{stream_name} truncated: {text_bytes}→{max_bytes} bytes]",
                True,
                text.count("\n") + 1,
            )
        return text, False, text.count("\n") + 1

    # Split into lines (preserving line endings)
    lines = text.splitlines(keepends=True)
    line_count = len(lines)

    # Check if truncation needed
    if line_count <= max_lines:
        # Within line limit, but check byte limit (if specified)
        if max_bytes is not None:
            text_bytes = len(text.encode("utf-8", errors="replace"))
            if text_bytes > max_bytes:
                truncated = text.encode("utf-8", errors="replace")[:max_bytes].decode(
                    "utf-8", errors="ignore"
                )
                # Recount lines after byte truncation
                new_line_count = truncated.count("\n") + 1
                return (
                    truncated + f"\n... [{stream_name} truncated: {text_bytes}→{max_bytes} bytes]",
                    True,
                    new_line_count,
                )
        return text, False, line_count

    # Truncate by lines, keeping head AND tail. Build/test tooling puts its
    # diagnostic payload (failure summaries, error counts, exit reasons) at the
    # end of the stream, so head-only truncation discards exactly the content
    # the model needs most.
    head_count = max(1, max_lines * 7 // 10)
    tail_count = max(0, max_lines - head_count)
    if tail_count > 0:
        omitted = line_count - head_count - tail_count
        truncated = (
            "".join(lines[:head_count])
            + f"... [{stream_name}: {omitted} middle lines omitted] ...\n"
            + "".join(lines[-tail_count:])
        )
    else:
        truncated = "".join(lines[:max_lines])

    # Enforce byte limit on truncated text (if specified)
    if max_bytes is not None:
        truncated_bytes = len(truncated.encode("utf-8", errors="replace"))
        if truncated_bytes > max_bytes:
            # Byte limit exceeded, truncate further
            truncated = truncated.encode("utf-8", errors="replace")[:max_bytes].decode(
                "utf-8", errors="ignore"
            )
            # Recount lines after byte truncation
            new_line_count = truncated.count("\n") + 1
            return (
                truncated
                + f"\n... [{stream_name} truncated: {line_count}→{new_line_count} lines, {truncated_bytes}→{max_bytes} bytes]",
                True,
                new_line_count,
            )

    return (
        truncated + f"\n... [{stream_name} truncated: {line_count}→{max_lines} lines]",
        True,
        max_lines,
    )


# =============================================================================
# Tool Availability Checking
# =============================================================================


def is_tool_available(tool_name: str) -> bool:
    """Check if a command-line tool is available.

    Args:
        tool_name: Name of the tool to check (e.g., 'git', 'docker').

    Returns:
        True if the tool is available, False otherwise.
    """
    return shutil.which(tool_name) is not None


def check_git_available() -> bool:
    """Check if git is available."""
    if not is_tool_available("git"):
        return False
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_docker_available() -> bool:
    """Check if docker is available."""
    if not is_tool_available("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_npm_available() -> bool:
    """Check if npm is available."""
    if not is_tool_available("npm"):
        return False
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_pip_available() -> bool:
    """Check if pip is available."""
    try:
        result = subprocess.run(
            ["pip", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# =============================================================================
# Synchronous Execution
# =============================================================================


def _resolve_default_sandbox() -> "SandboxBackend":
    """Resolve the sandbox backend from global settings (fail-open to none).

    Reads ``settings.sandbox`` (off by default). Any failure degrades to no
    sandbox so subprocess execution is never broken by sandbox resolution.
    """
    from victor.tools.sandbox import NoneSandbox, resolve_sandbox_backend

    try:
        from victor.config.settings import get_settings

        sandbox_settings = getattr(get_settings(), "sandbox", None)
        return resolve_sandbox_backend(sandbox_settings)
    except Exception:  # pragma: no cover - defensive: never break execution
        return NoneSandbox()


def _apply_sandbox(
    sandbox: "Optional[SandboxBackend]",
    args: Union[str, List[str]],
    shell: bool,
    working_dir: Optional[Union[str, Path]],
) -> Tuple[Union[str, List[str]], bool]:
    """Wrap ``args`` with the sandbox launcher if one is active.

    Returns ``(exec_args, exec_shell)``. When no sandbox is active the inputs
    are returned unchanged (zero behavior change with sandboxing disabled).
    A wrapped command always runs as an argv list with ``shell=False``.
    """
    backend = sandbox if sandbox is not None else _resolve_default_sandbox()
    if backend.type_name == "none":
        return args, shell

    cwd_path = Path(working_dir) if working_dir else None
    if shell or isinstance(args, str):
        cmd_str = args if isinstance(args, str) else " ".join(args)
        return backend.wrap_argv(["/bin/sh", "-c", cmd_str], cwd_path), False
    return backend.wrap_argv(list(args), cwd_path), False


def run_command(
    args: Union[str, List[str]],
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 60,
    check_dangerous: bool = True,
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
    preexec_fn: Optional[Callable[[], None]] = None,
    max_output_bytes: int = 0,
    sandbox: "Optional[SandboxBackend]" = None,
) -> CommandResult:
    """Execute a command synchronously and return structured result.

    Args:
        args: Command to execute. Either a string (requires shell=True) or list of args.
        working_dir: Working directory for command execution.
        timeout: Timeout in seconds (default: 60).
        check_dangerous: Whether to check for dangerous commands (default: True).
        env: Environment variables to set.
        shell: Whether to use shell execution (default: False).
        preexec_fn: Callable invoked in the child process before exec (POSIX only).
        max_output_bytes: Truncate stdout/stderr beyond this many bytes (0 = no limit).
        sandbox: Optional sandbox backend. When None, resolved from settings
            (off by default). A non-none backend wraps the command argv.

    Returns:
        CommandResult with execution details.
    """
    import time

    start_time = time.time()

    # Convert args to string for safety check
    cmd_str = args if isinstance(args, str) else " ".join(args)

    # Safety check
    if check_dangerous and is_dangerous_command(cmd_str):
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.DANGEROUS_COMMAND,
            error_message=f"Dangerous command blocked: {cmd_str}",
            command=cmd_str,
        )

    # Validate working directory
    if working_dir:
        working_dir = Path(working_dir)
        if not working_dir.exists():
            return CommandResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error_type=CommandErrorType.WORKING_DIR_NOT_FOUND,
                error_message=f"Working directory not found: {working_dir}",
                command=cmd_str,
                working_dir=str(working_dir),
            )

    # Apply OS sandbox if active (no-op when disabled; default off).
    exec_args, exec_shell = _apply_sandbox(sandbox, args, shell, working_dir)

    try:
        result = subprocess.run(
            exec_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
            env=env,
            shell=exec_shell,  # nosec B602
            preexec_fn=preexec_fn,  # nosec B603
        )

        duration_ms = (time.time() - start_time) * 1000

        stdout_text = result.stdout
        stderr_text = result.stderr
        was_truncated = False

        # Truncate output with separate limits for stdout and stderr
        if max_output_bytes > 0:
            # Use the simpler _truncate_output for byte-based truncation
            # (matches test expectations)
            stdout_text, t1 = _truncate_output(stdout_text, max_output_bytes)
            stderr_text, t2 = _truncate_output(stderr_text, max_output_bytes)
            was_truncated = t1 or t2
        elif max_output_bytes == 0:
            pass  # No limit (unlimited output)
        else:
            pass  # Negative limit means unlimited

        return CommandResult(
            success=result.returncode == 0,
            stdout=stdout_text,
            stderr=stderr_text,
            return_code=result.returncode,
            error_type=(
                CommandErrorType.SUCCESS
                if result.returncode == 0
                else CommandErrorType.EXECUTION_ERROR
            ),
            error_message=stderr_text if result.returncode != 0 else None,
            command=cmd_str,
            working_dir=str(working_dir) if working_dir else None,
            duration_ms=duration_ms,
            truncated=was_truncated,
        )

    except subprocess.TimeoutExpired:
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.TIMEOUT,
            error_message=f"Command timed out after {timeout} seconds",
            command=cmd_str,
            working_dir=str(working_dir) if working_dir else None,
        )

    except FileNotFoundError as e:
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.NOT_FOUND,
            error_message=f"Command not found: {e}",
            command=cmd_str,
        )

    except PermissionError as e:
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.PERMISSION_DENIED,
            error_message=f"Permission denied: {e}",
            command=cmd_str,
        )

    except Exception as e:
        logger.exception("Unexpected error executing command: %s", cmd_str)
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.UNKNOWN,
            error_message=str(e),
            command=cmd_str,
        )


# =============================================================================
# Asynchronous Execution
# =============================================================================


def _process_group_kwargs() -> Dict[str, Any]:
    """``start_new_session=True`` on POSIX so a timeout/cap kill can reach
    every process a spawned shell command starts (e.g. ``sleep 100 &``), not
    just the direct child — that child would otherwise be orphaned and keep
    running past the timeout. A no-op dict on platforms without
    ``os.killpg``/``os.getpgid`` (co-design review item 15)."""
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        return {"start_new_session": True}
    return {}


def kill_process_group(process: "asyncio.subprocess.Process") -> None:
    """Kill ``process`` and, on POSIX, its whole process group.

    Falls back to killing just the direct child when ``os.killpg``/
    ``os.getpgid`` are unavailable (non-POSIX) or the group lookup fails
    (process already reaped). Never raises.

    Checks ``process.returncode`` first, mirroring the guard
    ``subprocess.Popen.send_signal()`` itself applies before signaling
    (bpo-38630): once asyncio has observed the child exit, its pid may
    already have been recycled by the OS for an unrelated process, and
    ``killpg``/``kill`` on a stale pid would land on that.
    """
    if process.returncode is not None:
        return
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


async def run_managed_process(
    command: Optional[str] = None,
    *,
    argv: Optional[Sequence[str]] = None,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    preexec_fn: Optional[Callable[[], None]] = None,
    timeout: int = 60,
    max_output_bytes: int = 0,
    on_chunk: Optional[Callable[[bool, bytes], None]] = None,
) -> Tuple[bytes, bytes, int, bool, bool]:
    """Spawn a command and run it to completion or ``timeout``, whichever
    comes first. The single runner underneath every async subprocess call in
    this module (co-design review item 15).

    Exactly one of ``command`` (run through the shell) or ``argv`` (exec'd
    directly, no shell) must be given.

    stdout/stderr are read incrementally into caller-owned buffers as they
    arrive — never via a single ``communicate()`` — so whatever was produced
    before a timeout or a ``max_output_bytes`` cap survives the kill instead
    of being discarded.

    Returns:
        ``(stdout, stderr, return_code, timed_out, capped)``. ``stdout``/
        ``stderr`` are the FULL output on a clean exit, or PARTIAL output
        (everything read before the kill) when ``timed_out`` or ``capped``
        is True. ``return_code`` is 0 for a killed process (no exit was
        observed to report). On POSIX, both timeout and cap kills reach the
        whole process group, not just the direct child.
    """
    if (command is None) == (argv is None):
        raise ValueError("run_managed_process requires exactly one of command or argv")

    spawn_kwargs: Dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd,
        "env": env,
    }
    if preexec_fn is not None:
        spawn_kwargs["preexec_fn"] = preexec_fn
    spawn_kwargs.update(_process_group_kwargs())

    if argv is not None:
        process = await asyncio.create_subprocess_exec(*argv, **spawn_kwargs)
    else:
        process = await asyncio.create_subprocess_shell(command, **spawn_kwargs)

    out_buf = bytearray()
    err_buf = bytearray()
    capped = False

    async def _drain(stream: Any, buf: bytearray, is_stderr: bool) -> None:
        nonlocal capped
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            buf.extend(chunk)
            if on_chunk is not None:
                on_chunk(is_stderr, chunk)
            if max_output_bytes > 0 and len(buf) >= max_output_bytes:
                capped = True
                kill_process_group(process)
                return

    drain = asyncio.gather(
        _drain(process.stdout, out_buf, False),
        _drain(process.stderr, err_buf, True),
    )

    timed_out = False
    try:
        await asyncio.wait_for(drain, timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        kill_process_group(process)
    except BaseException:
        # Guarantee cleanup no matter WHY the drain failed — not just on the
        # expected timeout path. Without this, an unexpected exception from
        # a future on_chunk callback (or anywhere else in _drain) would skip
        # the kill entirely and leave `await process.wait()` below blocking
        # until the process exits naturally: the exact unbounded-hang bug
        # this runner exists to eliminate, reachable through a different
        # door. Re-raised after cleanup — this is not error handling, only
        # cleanup ordering.
        kill_process_group(process)
        raise
    finally:
        await process.wait()

    return bytes(out_buf), bytes(err_buf), (process.returncode or 0), timed_out, capped


async def run_command_async(
    command: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 60,
    check_dangerous: bool = True,
    env: Optional[Dict[str, str]] = None,
    preexec_fn: Optional[Callable[[], None]] = None,
    max_output_bytes: int = 0,
    sandbox: "Optional[SandboxBackend]" = None,
) -> CommandResult:
    """Execute a shell command asynchronously and return structured result.

    Args:
        command: Shell command to execute.
        working_dir: Working directory for command execution.
        timeout: Timeout in seconds (default: 60).
        check_dangerous: Whether to check for dangerous commands (default: True).
        env: Environment variables to set.
        preexec_fn: Callable invoked in the child process before exec (POSIX only).
        max_output_bytes: Truncate stdout/stderr beyond this many bytes (0 = no limit).
        sandbox: Optional sandbox backend. When None, resolved from settings
            (off by default). A non-none backend wraps the command argv and
            runs it via exec instead of the shell.

    Returns:
        CommandResult with execution details.
    """
    import time

    start_time = time.time()

    # Safety check
    if check_dangerous and is_dangerous_command(command):
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.DANGEROUS_COMMAND,
            error_message=f"Dangerous command blocked: {command}",
            command=command,
        )

    # Validate working directory
    cwd = None
    if working_dir:
        cwd = Path(working_dir)
        if not cwd.exists():
            return CommandResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error_type=CommandErrorType.WORKING_DIR_NOT_FOUND,
                error_message=f"Working directory not found: {working_dir}",
                command=command,
                working_dir=str(working_dir),
            )

    # Apply OS sandbox if active (no-op when disabled; default off).
    backend = sandbox if sandbox is not None else _resolve_default_sandbox()

    try:
        run_kwargs: Dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "preexec_fn": preexec_fn,
            "timeout": timeout,
            "max_output_bytes": max_output_bytes,
        }
        if backend.type_name != "none":
            wrapped = backend.wrap_argv(["/bin/sh", "-c", command], cwd)
            logger.debug("Executing command via %s sandbox: %.200s", backend.type_name, command)
            stdout, stderr, return_code, timed_out, capped = await run_managed_process(
                argv=wrapped, **run_kwargs
            )
        else:
            logger.debug("Executing command via shell: %.200s", command)
            stdout, stderr, return_code, timed_out, capped = await run_managed_process(
                command=command, **run_kwargs
            )

        duration_ms = (time.time() - start_time) * 1000
        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        was_truncated = capped
        if max_output_bytes > 0:
            stdout_str, t1, _ = _truncate_output_by_lines(
                stdout_str, None, max_bytes=max_output_bytes, stream_name="stdout"
            )
            stderr_str, t2, _ = _truncate_output_by_lines(
                stderr_str, None, max_bytes=max_output_bytes, stream_name="stderr"
            )
            was_truncated = was_truncated or t1 or t2

        if timed_out:
            return CommandResult(
                success=False,
                stdout=stdout_str,
                stderr=stderr_str,
                return_code=-1,
                error_type=CommandErrorType.TIMEOUT,
                error_message=f"Command timed out after {timeout} seconds",
                command=command,
                working_dir=str(working_dir) if working_dir else None,
                duration_ms=duration_ms,
                truncated=was_truncated,
            )

        return CommandResult(
            success=return_code == 0,
            stdout=stdout_str,
            stderr=stderr_str,
            return_code=return_code,
            error_type=(
                CommandErrorType.SUCCESS if return_code == 0 else CommandErrorType.EXECUTION_ERROR
            ),
            error_message=stderr_str if return_code != 0 else None,
            command=command,
            working_dir=str(working_dir) if working_dir else None,
            duration_ms=duration_ms,
            truncated=was_truncated,
        )

    except Exception as e:
        logger.exception("Unexpected error executing async command: %s", command)
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.UNKNOWN,
            error_message=str(e),
            command=command,
        )


# =============================================================================
# Specialized Runners
# =============================================================================


def run_git(
    *args: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Execute a git command.

    Args:
        *args: Git command arguments (e.g., "status", "-s").
        working_dir: Repository directory.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_git_available():
        return False, "", "Git is not available"

    result = run_command(
        ["git", *args],
        working_dir=working_dir,
        timeout=timeout,
        check_dangerous=False,  # Git commands are generally safe
    )

    return result.success, result.stdout, result.stderr


def run_docker(
    *args: str,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute a docker command.

    Args:
        *args: Docker command arguments.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_docker_available():
        return False, "", "Docker is not available"

    result = run_command(
        ["docker", *args],
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


def run_npm(
    *args: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute an npm command.

    Args:
        *args: npm command arguments.
        working_dir: Project directory.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_npm_available():
        return False, "", "npm is not available"

    result = run_command(
        ["npm", *args],
        working_dir=working_dir,
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


def run_pip(
    *args: str,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute a pip command.

    Args:
        *args: pip command arguments.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    result = run_command(
        ["pip", *args],
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


async def run_git_async(
    *args: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Execute a git command asynchronously.

    Args:
        *args: Git command arguments.
        working_dir: Repository directory.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_git_available():
        return False, "", "Git is not available"

    command = "git " + " ".join(args)
    result = await run_command_async(
        command,
        working_dir=working_dir,
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


async def run_docker_async(
    *args: str,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute a docker command asynchronously.

    Args:
        *args: Docker command arguments.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_docker_available():
        return False, "", "Docker is not available"

    command = "docker " + " ".join(args)
    result = await run_command_async(
        command,
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


async def run_npm_async(
    *args: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute an npm command asynchronously.

    Args:
        *args: npm command arguments.
        working_dir: Project directory.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_npm_available():
        return False, "", "npm is not available"

    command = "npm " + " ".join(args)
    result = await run_command_async(
        command,
        working_dir=working_dir,
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


async def run_pip_async(
    *args: str,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute a pip command asynchronously.

    Args:
        *args: pip command arguments.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    command = "pip " + " ".join(args)
    result = await run_command_async(
        command,
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


# =============================================================================
# Utility Functions
# =============================================================================


def parse_git_status(stdout: str) -> Dict[str, List[str]]:
    """Parse git status output into categorized files.

    Args:
        stdout: Output from 'git status --porcelain'.

    Returns:
        Dictionary with 'staged', 'modified', 'untracked' file lists.
    """
    result: Dict[str, List[str]] = {
        "staged": [],
        "modified": [],
        "untracked": [],
        "deleted": [],
    }

    for line in stdout.strip().split("\n"):
        if not line:
            continue

        status = line[:2]
        filename = line[3:].strip()

        if status[0] in ("A", "M", "D", "R"):
            result["staged"].append(filename)
        if status[1] == "M":
            result["modified"].append(filename)
        if status == "??":
            result["untracked"].append(filename)
        if status[1] == "D" or status[0] == "D":
            result["deleted"].append(filename)

    return result


def parse_docker_ps(stdout: str) -> List[Dict[str, str]]:
    """Parse docker ps output into container information.

    Args:
        stdout: Output from 'docker ps'.

    Returns:
        List of container info dictionaries.
    """
    containers = []
    lines = stdout.strip().split("\n")

    if len(lines) <= 1:
        return containers

    # Skip header line, parse data lines
    for line in lines[1:]:
        if not line.strip():
            continue

        # Simple parsing - assumes standard docker ps format
        parts = line.split()
        if len(parts) >= 2:
            containers.append(
                {
                    "id": parts[0],
                    "image": parts[1] if len(parts) > 1 else "",
                    "status": " ".join(parts[4:-2]) if len(parts) > 4 else "",
                    "name": parts[-1] if len(parts) > 0 else "",
                }
            )

    return containers
