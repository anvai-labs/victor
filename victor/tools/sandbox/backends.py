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

"""OS-level sandbox backends for subprocess-spawning tools.

Victor runs most tools in-process, so isolation only applies to the tools that
spawn OS subprocesses (shell/git/docker/test/cicd). A :class:`SandboxBackend`
wraps a command's argv with a platform launcher:

* :class:`BubblewrapSandbox` — Linux ``bwrap`` (mount-namespace isolation).
* :class:`SeatbeltSandbox` — macOS ``sandbox-exec`` (SBPL profile).
* :class:`NoneSandbox` — passthrough (no isolation).

:func:`resolve_sandbox_backend` selects a backend from :class:`SandboxSettings`.
It is **fail-open**: if the requested backend's binary is missing or the
platform is unsupported, it logs a one-time warning and returns
:class:`NoneSandbox` so tools keep working (the feature is opt-in and off by
default).

Cross-learning from Omnigent (``../omnigent/omnigent/inner/bwrap_sandbox.py``
and ``seatbelt_sandbox.py``). This is an MVP: the seatbelt profile uses an
``(allow default)`` base with narrow denies (vs Omnigent's stricter
deny-default), and bwrap omits the seccomp/egress hardening.
"""

from __future__ import annotations

import logging
import platform
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# System paths bound read-only inside the bwrap mount namespace so common
# interpreters/tools (sh, git, python, …) and their libraries remain reachable.
_BWRAP_SYSTEM_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt")

# Temp dirs that must stay writable for tools that use them (macOS uses
# /var/folders; /private/tmp is the real path behind /tmp).
_SEATBELT_WRITABLE_TEMP = ("/tmp", "/private/tmp", "/var/folders")

# Guard so the fail-open warning is emitted at most once per process.
_warned_unavailable = False


class SandboxBackend(ABC):
    """Wraps a command argv with an OS isolation launcher."""

    #: Stable backend identifier (e.g. "none", "bwrap", "seatbelt").
    type_name: str = "sandbox"

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can actually be used on the current host."""
        raise NotImplementedError

    @abstractmethod
    def wrap_argv(self, argv: List[str], cwd: Optional[Path] = None) -> List[str]:
        """Return ``argv`` wrapped by the launcher (or unchanged for none)."""
        raise NotImplementedError


class NoneSandbox(SandboxBackend):
    """No isolation — returns the command unchanged."""

    type_name = "none"

    def available(self) -> bool:
        """Always available."""
        return True

    def wrap_argv(self, argv: List[str], cwd: Optional[Path] = None) -> List[str]:
        """Passthrough."""
        return list(argv)


class BubblewrapSandbox(SandboxBackend):
    """Linux ``bwrap`` mount-namespace sandbox.

    System paths are bound read-only; the working directory is bound writable
    (workspace-only mode). Optional PID/IPC and network namespace isolation are
    driven by the corresponding :class:`SandboxSettings` flags.
    """

    type_name = "bwrap"

    def __init__(
        self,
        *,
        workspace_only: bool = True,
        network_isolation: bool = False,
        namespace_restrictions: bool = True,
        allowed_mounts: Optional[List[str]] = None,
    ) -> None:
        """Configure the bwrap policy from sandbox settings."""
        self._workspace_only = workspace_only
        self._network_isolation = network_isolation
        self._namespace_restrictions = namespace_restrictions
        self._allowed_mounts = list(allowed_mounts or [])

    def available(self) -> bool:
        """True on Linux with the ``bwrap`` binary on PATH."""
        return platform.system() == "Linux" and shutil.which("bwrap") is not None

    def wrap_argv(self, argv: List[str], cwd: Optional[Path] = None) -> List[str]:
        """Build the ``bwrap`` argv that wraps ``argv``."""
        flags: List[str] = ["bwrap", "--die-with-parent"]

        # System roots: read-only (--ro-bind-try tolerates missing paths).
        for root in _BWRAP_SYSTEM_ROOTS:
            flags += ["--ro-bind-try", root, root]

        # Minimal /proc, /dev and a private /tmp.
        flags += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]

        # Working directory: writable in workspace-only mode, else read-only.
        if cwd is not None:
            cwd_str = str(Path(cwd))
            bind = "--bind" if self._workspace_only else "--ro-bind"
            flags += [bind, cwd_str, cwd_str]

        # Extra writable mounts the operator allow-listed.
        for mount in self._allowed_mounts:
            flags += ["--bind-try", mount, mount]

        if self._namespace_restrictions:
            flags += ["--unshare-pid", "--unshare-ipc"]
        if self._network_isolation:
            flags += ["--unshare-net"]

        flags.append("--")
        return flags + list(argv)


class SeatbeltSandbox(SandboxBackend):
    """macOS ``sandbox-exec`` (Seatbelt/SBPL) sandbox.

    Uses a deliberately permissive ``(allow default)`` base so tools that read
    ``/usr``/``$HOME`` keep working, then narrows the two controls the settings
    express: workspace-only writes and optional network isolation.
    """

    type_name = "seatbelt"

    def __init__(
        self,
        *,
        workspace_only: bool = True,
        network_isolation: bool = False,
    ) -> None:
        """Configure the seatbelt policy from sandbox settings."""
        self._workspace_only = workspace_only
        self._network_isolation = network_isolation

    def available(self) -> bool:
        """True on macOS with the ``sandbox-exec`` binary on PATH."""
        return platform.system() == "Darwin" and shutil.which("sandbox-exec") is not None

    def build_profile(self, cwd: Optional[Path] = None) -> str:
        """Build the SBPL profile string for the configured policy."""
        lines = ["(version 1)", "(allow default)"]

        if self._workspace_only:
            # Deny all writes, then re-allow the workspace and temp dirs.
            lines.append("(deny file-write*)")
            if cwd is not None:
                lines.append(f'(allow file-write* (subpath "{Path(cwd)}"))')
            for tmp in _SEATBELT_WRITABLE_TEMP:
                lines.append(f'(allow file-write* (subpath "{tmp}"))')

        if self._network_isolation:
            lines.append("(deny network*)")

        return "\n".join(lines)

    def wrap_argv(self, argv: List[str], cwd: Optional[Path] = None) -> List[str]:
        """Build the ``sandbox-exec -p <profile>`` argv that wraps ``argv``."""
        profile = self.build_profile(cwd)
        return ["sandbox-exec", "-p", profile] + list(argv)


def _warn_once(message: str) -> None:
    """Emit the fail-open warning at most once per process."""
    global _warned_unavailable
    if not _warned_unavailable:
        logger.warning(message)
        _warned_unavailable = True


def resolve_sandbox_backend(sandbox_settings: Any) -> SandboxBackend:
    """Select a sandbox backend from settings (fail-open to :class:`NoneSandbox`).

    Args:
        sandbox_settings: A ``SandboxSettings``-like object (or None).

    Returns:
        A platform-appropriate backend when enabled and available; otherwise
        :class:`NoneSandbox` (with a one-time warning if a sandbox was requested
        but is unavailable).
    """
    if sandbox_settings is None or not getattr(sandbox_settings, "sandbox_enabled", False):
        return NoneSandbox()

    workspace_only = getattr(sandbox_settings, "sandbox_filesystem_mode", "workspace-only") == (
        "workspace-only"
    )
    network_isolation = bool(getattr(sandbox_settings, "sandbox_network_isolation", False))
    namespace_restrictions = bool(getattr(sandbox_settings, "sandbox_namespace_restrictions", True))
    allowed_mounts = list(getattr(sandbox_settings, "sandbox_allowed_mounts", []) or [])

    system = platform.system()
    if system == "Linux":
        backend: SandboxBackend = BubblewrapSandbox(
            workspace_only=workspace_only,
            network_isolation=network_isolation,
            namespace_restrictions=namespace_restrictions,
            allowed_mounts=allowed_mounts,
        )
    elif system == "Darwin":
        backend = SeatbeltSandbox(
            workspace_only=workspace_only,
            network_isolation=network_isolation,
        )
    else:
        _warn_once(
            f"Sandbox enabled but unsupported platform '{system}'; "
            "running tools UNSANDBOXED (fail-open)."
        )
        return NoneSandbox()

    if not backend.available():
        _warn_once(
            f"Sandbox enabled but the '{backend.type_name}' backend is unavailable "
            f"(binary missing on {system}); running tools UNSANDBOXED (fail-open). "
            "Install the sandbox binary to enable isolation."
        )
        return NoneSandbox()

    return backend


__all__ = [
    "SandboxBackend",
    "NoneSandbox",
    "BubblewrapSandbox",
    "SeatbeltSandbox",
    "resolve_sandbox_backend",
]
