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

* :class:`BubblewrapSandbox` — Linux ``bwrap`` (mount-namespace isolation with
  a classic-BPF seccomp deny filter, network-namespace egress deny and
  ``--new-session`` tty detachment by default).
* :class:`SeatbeltSandbox` — macOS ``sandbox-exec`` (SBPL profile, deny-default
  with explicit allow rules).
* :class:`NoneSandbox` — passthrough (no isolation).

:func:`resolve_sandbox_backend` selects a backend from :class:`SandboxSettings`.
With the ``preferred`` policy it is **fail-open**: if the requested backend's
binary is missing or the platform is unsupported, it logs a one-time warning
and returns :class:`NoneSandbox` so tools keep working. With the ``required``
policy it is **fail-closed**: an unavailable backend raises
:class:`SandboxRequiredUnavailableError` and the caller must refuse dispatch
(zero execution).

Backends that need to hand open file descriptors to the wrapped process
(bwrap's ``--seccomp`` filter fd) expose them via :meth:`SandboxBackend.take_pass_fds`
— callers must forward them as ``pass_fds`` to the spawn call and close them
afterwards.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
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


# -- seccomp deny filter (classic BPF) ---------------------------------------

# AUDIT_ARCH_* constants from linux/audit.h.
_AUDIT_ARCH_X86_64 = 0xC000003E
_AUDIT_ARCH_AARCH64 = 0xC00000B7

# SECCOMP_RET_* action flags from linux/seccomp.h.
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000
_EPERM = 1

# Syscall numbers DENIED inside the sandbox, per architecture. The list
# targets sandbox escape and kernel-integrity surfaces: mounts, namespace
# manipulation, module loading, kexec, ptrace-style inspection, BPF, the
# keyring API and handle-based name lookup. Everything not listed is ALLOWED
# (deny-list posture — a workspace shell still needs the vast majority of
# syscalls). Unknown architectures fail closed (all syscalls denied).
_SECCOMP_DENY_SYSCALLS: dict = {
    "x86_64": (
        165,  # mount
        166,  # umount2
        203,  # pivot_root
        272,  # unshare
        308,  # setns
        321,  # bpf
        101,  # ptrace
        298,  # perf_event_open
        246,  # kexec_load
        294,  # kexec_file_load
        175,  # init_module
        313,  # finit_module
        176,  # delete_module
        304,  # open_by_handle_at
        303,  # name_to_handle_at
        250,  # keyctl
        248,  # add_key
        249,  # request_key
        169,  # reboot
        167,  # swapon
        168,  # swapoff
        310,  # process_vm_readv
        311,  # process_vm_writev
    ),
    "aarch64": (
        40,  # mount
        39,  # umount2
        41,  # pivot_root
        266,  # unshare
        268,  # setns
        280,  # bpf
        117,  # ptrace
        241,  # perf_event_open
        104,  # kexec_load
        294,  # kexec_file_load
        105,  # init_module
        313,  # finit_module
        106,  # delete_module
        265,  # open_by_handle_at
        264,  # name_to_handle_at
        219,  # keyctl
        217,  # add_key
        218,  # request_key
        142,  # reboot
        224,  # swapon
        225,  # swapoff
        270,  # process_vm_readv
        271,  # process_vm_writev
    ),
}

# Classic-BPF instruction classes (linux/filter.h).
_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_RET_K = 0x06


def build_seccomp_deny_program() -> bytes:
    """Build the classic-BPF deny-list program handed to ``bwrap --seccomp``.

    Layout (jump offsets are relative to the NEXT instruction)::

        0        ld [4]                    (arch)
        1        jeq X86_64   -> x86 section, fall -> 2
        2        jeq AARCH64  -> arm section, fall -> 3
        3        ret ERRNO(EPERM)         (unknown arch fails CLOSED)
        4..      x86: ld [0]; jeq nr -> deny; ... ; deny ret; allow ret
        ...      arm: same shape

    Returns:
        The packed ``sock_filter`` program as bytes (one instruction = 8
        bytes: u16 code, u8 jt, u8 jf, u32 k).
    """

    def _ld(off: int):
        return (_BPF_LD_W_ABS, 0, 0, off)

    def _jeq(k: int, jt: int, jf: int):
        return (_BPF_JMP_JEQ_K, jt, jf, k)

    def _ret(k: int):
        return (_BPF_RET_K, 0, 0, k)

    deny_x86 = _SECCOMP_DENY_SYSCALLS["x86_64"]
    deny_arm = _SECCOMP_DENY_SYSCALLS["aarch64"]
    n1, n2 = len(deny_x86), len(deny_arm)
    ret_errno = _ret(_SECCOMP_RET_ERRNO | _EPERM)
    ret_allow = _ret(_SECCOMP_RET_ALLOW)

    insns: List[Any] = []

    # Arch dispatch (indices 0-3).
    insns.append(_ld(4))  # 0: arch
    insns.append(_jeq(_AUDIT_ARCH_X86_64, jt=2, jf=1))  # 1: -> x86 ld (idx 4); jf -> 2
    insns.append(_jeq(_AUDIT_ARCH_AARCH64, jt=4 + n1, jf=0))  # 2: -> arm ld (idx 7+n1); jf -> 3
    insns.append(ret_errno)  # 3: unknown arch fail-closed

    # x86_64 section: ld nr (idx 4), n1 jeqs (5..4+n1), deny (5+n1), allow (6+n1).
    insns.append(_ld(0))  # 4: syscall nr
    for i, nr in enumerate(deny_x86):
        insns.append(_jeq(nr, jt=n1 - 1 - i, jf=1))
    insns.append(ret_errno)  # 5+n1
    insns.append(ret_allow)  # 6+n1

    # aarch64 section: ld nr (idx 7+n1), n2 jeqs, deny, allow.
    insns.append(_ld(0))  # 7+n1
    for i, nr in enumerate(deny_arm):
        insns.append(_jeq(nr, jt=n2 - 1 - i, jf=1))
    insns.append(ret_errno)
    insns.append(ret_allow)

    return b"".join(struct.pack("<HBBI", code, jt, jf, k) for code, jt, jf, k in insns)


def _open_seccomp_fd() -> int:
    """Load :func:`build_seccomp_deny_program` into an inherited read fd.

    ``bwrap --seccomp FD`` reads the BPF program from the open descriptor.
    The write end is closed immediately so the child observes EOF; the read
    end stays open in the parent until exec (the caller must forward it via
    ``pass_fds``) and is closed by the executor afterwards.
    """
    r, w = os.pipe()
    try:
        os.set_inheritable(r, True)
        os.write(w, build_seccomp_deny_program())
    finally:
        os.close(w)
    return r


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

    def take_pass_fds(self) -> List[int]:
        """Return and clear fds that must be inherited by the wrapped argv.

        Callers forward these as ``pass_fds`` to the spawn call and close
        them once the child is running. Base backends need none.
        """
        return []


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
    (workspace-only mode). PID/IPC and network namespace isolation are on by
    default (ADR-035: egress deny is the deliverable, not a dead knob), as is
    a classic-BPF seccomp deny filter over sandbox-escape syscalls and
    ``--new-session`` tty detachment. The seccomp filter fd must be inherited
    by the wrapped process: callers collect it via :meth:`take_pass_fds`.
    """

    type_name = "bwrap"

    def __init__(
        self,
        *,
        workspace_only: bool = True,
        network_isolation: bool = True,
        namespace_restrictions: bool = True,
        seccomp: bool = True,
        allowed_mounts: Optional[List[str]] = None,
    ) -> None:
        """Configure the bwrap policy from sandbox settings."""
        self._workspace_only = workspace_only
        self._network_isolation = network_isolation
        self._namespace_restrictions = namespace_restrictions
        self._seccomp = seccomp
        self._allowed_mounts = list(allowed_mounts or [])
        self._pending_fds: List[int] = []

    def available(self) -> bool:
        """True on Linux with the ``bwrap`` binary on PATH."""
        return platform.system() == "Linux" and shutil.which("bwrap") is not None

    def take_pass_fds(self) -> List[int]:
        """Return and clear the seccomp filter fds awaiting inheritance."""
        fds, self._pending_fds = self._pending_fds, []
        return fds

    def wrap_argv(self, argv: List[str], cwd: Optional[Path] = None) -> List[str]:
        """Build the ``bwrap`` argv that wraps ``argv``."""
        flags: List[str] = ["bwrap", "--die-with-parent", "--new-session"]

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
        if self._seccomp:
            fd = _open_seccomp_fd()
            self._pending_fds.append(fd)
            flags += ["--seccomp", str(fd)]

        flags.append("--")
        return flags + list(argv)


class SeatbeltSandbox(SandboxBackend):
    """macOS ``sandbox-exec`` (Seatbelt/SBPL) sandbox.

    Deny-default profile (ADR-035): unmatched operations DENY rather than
    allow. The allowlist keeps tools functional — process fork/exec, broad
    file reads, workspace+temp writes, mach lookups and local sockets — then
    scopes the two controls the settings express: workspace-only writes and
    network access (denied by default).
    """

    type_name = "seatbelt"

    def __init__(
        self,
        *,
        workspace_only: bool = True,
        network_isolation: bool = True,
    ) -> None:
        """Configure the seatbelt policy from sandbox settings."""
        self._workspace_only = workspace_only
        self._network_isolation = network_isolation

    def available(self) -> bool:
        """True on macOS with the ``sandbox-exec`` binary on PATH."""
        return platform.system() == "Darwin" and shutil.which("sandbox-exec") is not None

    def build_profile(self, cwd: Optional[Path] = None) -> str:
        """Build the SBPL profile string for the configured policy."""
        lines = [
            "(version 1)",
            "(deny default)",
            # Process lifecycle and broad reads keep tools functional.
            "(allow process-fork)",
            '(allow process-exec* (subpath "/"))',
            '(allow file-read* (subpath "/"))',
            # Mach/XPC lookups and local sockets are required by common tools.
            "(allow mach-lookup)",
            "(allow system-socket)",
        ]

        if self._workspace_only:
            # Writes are scoped to the workspace and temp dirs.
            if cwd is not None:
                lines.append(f'(allow file-write* (subpath "{Path(cwd)}"))')
            for tmp in _SEATBELT_WRITABLE_TEMP:
                lines.append(f'(allow file-write* (subpath "{tmp}"))')
        else:
            lines.append("(allow file-write*)")

        if self._network_isolation:
            lines.append("(deny network*)")
        else:
            lines.append("(allow network*)")

        return "\n".join(lines)

    def wrap_argv(self, argv: List[str], cwd: Optional[Path] = None) -> List[str]:
        """Build the ``sandbox-exec -p <profile>`` argv that wraps ``argv``."""
        profile = self.build_profile(cwd)
        return ["sandbox-exec", "-p", profile] + list(argv)


class PolicyUnavailable(RuntimeError):
    """The sandbox policy could not be established (settings read failure,
    malformed configuration).

    Unknowable policy is NOT affirmative disabled mode (NEXT-01): launch
    owners must refuse governed execution instead of degrading open. An
    explicit disabled/preferred decision remains available only after a
    successful settings read.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _warn_once(message: str) -> None:
    """Emit the fail-open warning at most once per process."""
    global _warned_unavailable
    if not _warned_unavailable:
        logger.warning(message)
        _warned_unavailable = True


class SandboxRequiredUnavailableError(RuntimeError):
    """No backend can enforce a ``required`` containment policy.

    Raised by :func:`resolve_sandbox_backend` when the policy is mandatory
    but the platform is unsupported or the backend binary is missing.
    Callers must treat this as a dispatch refusal (zero execution), never
    as permission to run unsandboxed.
    """

    def __init__(self, reason: str, message: str) -> None:
        """Record a stable reason code alongside the human message."""
        super().__init__(message)
        self.reason = reason


def effective_sandbox_policy(sandbox_settings: Any) -> str:
    """Resolve the effective policy from settings.

    An explicit (non-``None``) ``sandbox_policy`` wins; ``None``/absent
    derives from the legacy ``sandbox_enabled`` bool, which maps to
    ``preferred`` when true (its historical fail-open semantics). Missing
    settings mean disabled.
    """
    if sandbox_settings is None:
        return "disabled"
    policy = getattr(sandbox_settings, "sandbox_policy", None)
    if policy in ("disabled", "preferred", "required"):
        return policy
    if bool(getattr(sandbox_settings, "sandbox_enabled", False)):
        return "preferred"
    return "disabled"


def resolve_sandbox_backend(sandbox_settings: Any) -> SandboxBackend:
    """Select a sandbox backend from settings.

    Policy semantics:

    * ``disabled`` — :class:`NoneSandbox`, silently (historical default).
    * ``preferred`` — fail-open: unavailable backend/platform degrades to
      :class:`NoneSandbox` with a one-time DEGRADED warning.
    * ``required`` — fail-closed: raises
      :class:`SandboxRequiredUnavailableError`; dispatch must refuse to
      execute rather than run unsandboxed.

    Args:
        sandbox_settings: A ``SandboxSettings``-like object (or None).

    Returns:
        A platform-appropriate backend when isolation is requested and
        enforceable; otherwise :class:`NoneSandbox`.

    Raises:
        SandboxRequiredUnavailableError: policy is ``required`` and no
            backend can enforce it.
    """
    policy = effective_sandbox_policy(sandbox_settings)
    if policy == "disabled":
        return NoneSandbox()

    workspace_only = getattr(sandbox_settings, "sandbox_filesystem_mode", "workspace-only") == (
        "workspace-only"
    )
    network_isolation = bool(getattr(sandbox_settings, "sandbox_network_isolation", True))
    namespace_restrictions = bool(getattr(sandbox_settings, "sandbox_namespace_restrictions", True))
    seccomp = bool(getattr(sandbox_settings, "sandbox_seccomp", True))
    allowed_mounts = list(getattr(sandbox_settings, "sandbox_allowed_mounts", []) or [])

    system = platform.system()
    if system == "Linux":
        backend: SandboxBackend = BubblewrapSandbox(
            workspace_only=workspace_only,
            network_isolation=network_isolation,
            namespace_restrictions=namespace_restrictions,
            seccomp=seccomp,
            allowed_mounts=allowed_mounts,
        )
    elif system == "Darwin":
        backend = SeatbeltSandbox(
            workspace_only=workspace_only,
            network_isolation=network_isolation,
        )
    else:
        if policy == "required":
            raise SandboxRequiredUnavailableError(
                "unsupported_platform",
                f"Sandbox policy is 'required' but platform '{system}' has no "
                "enforcing backend; refusing execution (fail-closed).",
            )
        _warn_once(
            f"Sandbox enabled but unsupported platform '{system}'; "
            "running tools UNSANDBOXED (fail-open). [DEGRADED]"
        )
        return NoneSandbox()

    if not backend.available():
        if policy == "required":
            raise SandboxRequiredUnavailableError(
                "binary_missing",
                f"Sandbox policy is 'required' but the '{backend.type_name}' backend is "
                f"unavailable (binary missing on {system}); refusing execution "
                "(fail-closed). Install the sandbox binary to enable isolation.",
            )
        _warn_once(
            f"Sandbox enabled but the '{backend.type_name}' backend is unavailable "
            f"(binary missing on {system}); running tools UNSANDBOXED (fail-open). "
            "Install the sandbox binary to enable isolation. [DEGRADED]"
        )
        return NoneSandbox()

    return backend


@dataclass(frozen=True)
class RequiredLaunchDecision:
    """Resolved decision for a required-policy launch (HB-02/03, ADR-L02).

    ``required_active`` is False unless the effective policy is required.
    ``refuse`` marks a required policy that cannot be enforced (typed
    resolver failures and generic resolver/configuration errors alike) —
    callers must refuse before any spawn and never degrade to no sandbox.
    """

    required_active: bool = False
    backend: Optional["SandboxBackend"] = None
    refuse: bool = False
    reason: str = ""


def resolve_required_launch(sandbox_settings: Any) -> RequiredLaunchDecision:
    """Resolve whether a required policy can reach an enforcing launch.

    Consolidates the fail-closed rule shared by the canonical subprocess
    executor, the workflow sandbox executor and direct-spawn routes: under a
    KNOWN required policy, any resolver or configuration failure refuses
    (never degrades); other policies return a non-active decision and keep
    their historical semantics.
    """
    try:
        policy = effective_sandbox_policy(sandbox_settings)
    except Exception:
        return RequiredLaunchDecision()
    if policy != "required":
        return RequiredLaunchDecision()
    try:
        backend: SandboxBackend = resolve_sandbox_backend(sandbox_settings)
    except SandboxRequiredUnavailableError as exc:
        return RequiredLaunchDecision(refuse=True, reason=str(exc))
    except Exception as exc:
        return RequiredLaunchDecision(
            refuse=True,
            reason=(
                "Sandbox policy is 'required' but backend resolution failed "
                f"({exc}); refusing execution (fail-closed)."
            ),
        )
    return RequiredLaunchDecision(required_active=True, backend=backend)


__all__ = [
    "SandboxBackend",
    "NoneSandbox",
    "BubblewrapSandbox",
    "SeatbeltSandbox",
    "SandboxRequiredUnavailableError",
    "PolicyUnavailable",
    "RequiredLaunchDecision",
    "build_seccomp_deny_program",
    "effective_sandbox_policy",
    "resolve_required_launch",
    "resolve_sandbox_backend",
]
