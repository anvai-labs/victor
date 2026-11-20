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

from victor.config.secure_paths import get_victor_dir

logger = logging.getLogger(__name__)

# Temp dirs that must stay writable for tools that use them (macOS uses
# /var/folders; /private/tmp is the real path behind /tmp).
_SEATBELT_WRITABLE_TEMP = ("/tmp", "/private/tmp", "/var/folders")

# Warn-once bookkeeping, per category ("preferred_degraded",
# "opt_out_disabled") so one category's warning can never permanently
# suppress another's in a long-lived process.
_warned_categories: set = set()


def _prepare_victor_state_root() -> Path:
    """Return the validated, absolute Victor user-state root.

    The state root is created on demand so a first-run sandbox can open the global database
    and profile files. Failure is raised here, before command launch, rather than becoming a
    silent no-write mount.
    """
    root = get_victor_dir().expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot prepare writable Victor state root {root}: {exc}") from exc
    if not root.is_dir():
        raise RuntimeError(f"Victor state root is not a directory: {root}")
    return root


def _absolute_mount_path(path: Path | str) -> Path:
    """Normalize an operator or workspace path without creating it."""
    return Path(path).expanduser().resolve()


# -- seccomp deny filter (classic BPF) ---------------------------------------

# AUDIT_ARCH_* constants from linux/audit.h.
_AUDIT_ARCH_X86_64 = 0xC000003E
_AUDIT_ARCH_AARCH64 = 0xC00000B7

# SECCOMP_RET_* action flags from linux/seccomp.h.
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000
_EPERM = 1

# Syscall numbers DENIED inside the sandbox, per architecture. Derived from
# the Kubernetes/containerd RuntimeDefault seccomp posture (the same baseline
# the Omnigent reference adopts), restricted to what the kABI-fixed syscall
# tables actually define per arch. Targets: mounts (legacy AND the new
# mount API), namespace manipulation, module loading, kexec, ptrace-style
# inspection, BPF, kernel observability, clock setters, NUMA policy, the
# keyring API and handle-based name lookup. Everything not listed is ALLOWED
# (deny-list posture — a workspace shell still needs the vast majority of
# syscalls). Unknown architectures fail closed (all syscalls denied), which
# also covers the compat-ABI seccomp footgun (int $0x80 / x32): unlisted
# ABIs get EPERM across the board rather than translated allow rules.
# Numbers verified against asm/unistd_64.h (x86_64) and the preprocessed
# asm-generic table (aarch64) — several commonly cited values differ per
# arch (e.g. kexec_file_load is 320 on x86_64 but 294 on aarch64).
_SECCOMP_DENY_SYSCALLS: dict = {
    "x86_64": (
        165,  # mount
        166,  # umount2
        155,  # pivot_root
        161,  # chroot
        428,  # open_tree
        429,  # move_mount
        430,  # fsopen
        431,  # fsconfig
        432,  # fsmount
        433,  # fspick
        442,  # mount_setattr
        272,  # unshare
        308,  # setns
        175,  # init_module
        313,  # finit_module
        176,  # delete_module
        174,  # create_module
        178,  # query_module
        177,  # get_kernel_syms
        246,  # kexec_load
        320,  # kexec_file_load
        304,  # open_by_handle_at
        303,  # name_to_handle_at
        180,  # nfsservctl
        139,  # sysfs
        156,  # _sysctl
        136,  # ustat
        134,  # uselib
        179,  # quotactl
        443,  # quotactl_fd
        321,  # bpf
        101,  # ptrace
        298,  # perf_event_open
        323,  # userfaultfd
        312,  # kcmp
        212,  # lookup_dcookie
        300,  # fanotify_init
        227,  # clock_settime
        164,  # settimeofday
        103,  # syslog (kernel ring buffer)
        169,  # reboot
        163,  # acct
        167,  # swapon
        168,  # swapoff
        237,  # mbind
        256,  # migrate_pages
        279,  # move_pages
        238,  # set_mempolicy
        239,  # get_mempolicy
        450,  # set_mempolicy_home_node
        172,  # iopl
        173,  # ioperm
        170,  # sethostname
        171,  # setdomainname
        153,  # vhangup
        438,  # pidfd_getfd
        440,  # process_madvise
        248,  # add_key
        249,  # request_key
        250,  # keyctl
        310,  # process_vm_readv
        311,  # process_vm_writev
    ),
    "aarch64": (
        40,  # mount
        39,  # umount2
        41,  # pivot_root
        51,  # chroot
        428,  # open_tree
        429,  # move_mount
        430,  # fsopen
        431,  # fsconfig
        432,  # fsmount
        433,  # fspick
        442,  # mount_setattr
        97,  # unshare
        268,  # setns
        105,  # init_module
        273,  # finit_module
        106,  # delete_module
        104,  # kexec_load
        294,  # kexec_file_load
        265,  # open_by_handle_at
        264,  # name_to_handle_at
        219,  # keyctl
        217,  # add_key
        218,  # request_key
        280,  # bpf
        117,  # ptrace
        241,  # perf_event_open
        282,  # userfaultfd
        272,  # kcmp
        18,  # lookup_dcookie
        262,  # fanotify_init
        112,  # clock_settime
        170,  # settimeofday
        142,  # reboot
        116,  # syslog (kernel ring buffer)
        224,  # swapon
        225,  # swapoff
        89,  # acct
        235,  # mbind
        238,  # migrate_pages
        239,  # move_pages
        237,  # set_mempolicy
        236,  # get_mempolicy
        450,  # set_mempolicy_home_node
        161,  # sethostname
        162,  # setdomainname
        58,  # vhangup
        438,  # pidfd_getfd
        440,  # process_madvise
        270,  # process_vm_readv
        271,  # process_vm_writev
        60,  # quotactl
        443,  # quotactl_fd
    ),
}

# clone() flags that create namespaces. clone() itself must stay allowed
# (threads), so it is denied only when its flag argument carries one of
# these — the argument-filtered rule Omnigent layers on top of the baseline.
_CLONE_NEW_FLAG_MASK = (
    0x00020000  # CLONE_NEWNS
    | 0x02000000  # CLONE_NEWCGROUP
    | 0x04000000  # CLONE_NEWUTS
    | 0x08000000  # CLONE_NEWIPC
    | 0x10000000  # CLONE_NEWUSER
    | 0x20000000  # CLONE_NEWPID
    | 0x40000000  # CLONE_NEWNET
)
# clone3() takes its flags through a struct pointer that cBPF cannot
# inspect; glibc falls back to clone() for thread creation, and kernel
# seccomp + no-new-privs remain in force for it.
_CLONE_NR = {"x86_64": 56, "aarch64": 220}

# seccomp_data offsets: nr=0, arch=4, args[0]=16.
_SECCOMP_DATA_NR = 0
_SECCOMP_DATA_ARCH = 4
_SECCOMP_DATA_ARG0 = 16

# Classic-BPF instruction classes (linux/filter.h).
_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_ALU_AND_K = 0x54
_BPF_RET_K = 0x06

# Per-arch section size: ld nr + n denies + clone jeq + ld args + and + jeq
# + deny ret + allow ret.
_SECTION_FIXED = 7


def _emit_arch_section(insns: List[Any], deny_nrs: tuple, clone_nr: int) -> None:
    """Append one architecture section (deny list + clone flag filter).

    Layout (offsets computed from the running instruction index, never
    hand-counted)::

        ld [0]                          (syscall nr)
        jeq nr -> deny ; ...            (n deny rules)
        jeq clone_nr -> flag-check      (jf -> allow)
        ld [16]                         (args[0] = clone flags)
        and _CLONE_NEW_FLAG_MASK
        jeq 0 -> allow                  (jf -> deny: namespace flags present)
        ret ERRNO(EPERM)                (deny)
        ret ALLOW
    """
    base = len(insns)
    n = len(deny_nrs)
    deny_idx = base + 1 + n + 4
    allow_idx = deny_idx + 1
    clone_jeq_idx = base + 1 + n

    insns.append((_BPF_LD_W_ABS, 0, 0, _SECCOMP_DATA_NR))
    for i, nr in enumerate(deny_nrs):
        jt = deny_idx - (base + 2 + i)
        # jf=0: cBPF offsets are relative to pc+1, so fall-through to the
        # next deny check is offset 0 (a value of 1 silently SKIPS a check).
        insns.append((_BPF_JMP_JEQ_K, jt, 0, nr))
    # clone(): allow unless the flags request a namespace. jt=0 lands on the
    # args load at clone_jeq_idx+1 (offsets are relative to pc+1).
    insns.append((_BPF_JMP_JEQ_K, 0, allow_idx - (clone_jeq_idx + 1), clone_nr))
    insns.append((_BPF_LD_W_ABS, 0, 0, _SECCOMP_DATA_ARG0))
    insns.append((_BPF_ALU_AND_K, 0, 0, _CLONE_NEW_FLAG_MASK))
    # Flag test sits at clone_jeq_idx+3, so offsets are relative to +4.
    insns.append(
        (
            _BPF_JMP_JEQ_K,
            allow_idx - (clone_jeq_idx + 4),
            deny_idx - (clone_jeq_idx + 4),
            0,
        )
    )
    insns.append((_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | _EPERM))
    insns.append((_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW))


def build_seccomp_deny_program() -> bytes:
    """Build the classic-BPF deny-list program handed to ``bwrap --seccomp``.

    Layout::

        0        ld [4]                    (arch)
        1        jeq X86_64   -> x86 section, fall -> 2
        2        jeq AARCH64  -> arm section, fall -> 3
        3        ret ERRNO(EPERM)         (unknown arch fails CLOSED)
        4..      per-arch sections from :func:`_emit_arch_section`

    Returns:
        The packed ``sock_filter`` program as bytes (one instruction = 8
        bytes: u16 code, u8 jt, u8 jf, u32 k).
    """
    n1 = len(_SECCOMP_DENY_SYSCALLS["x86_64"])
    insns: List[Any] = [
        (_BPF_LD_W_ABS, 0, 0, _SECCOMP_DATA_ARCH),  # 0: arch
        # 1: -> x86 ld (idx 4); jf -> 2 (arm check)
        (_BPF_JMP_JEQ_K, 2, 0, _AUDIT_ARCH_X86_64),
        # 2: -> arm ld (idx 4 + n1 + _SECTION_FIXED); jf -> 3
        (_BPF_JMP_JEQ_K, 4 + n1 + _SECTION_FIXED - 3, 0, _AUDIT_ARCH_AARCH64),
        (_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | _EPERM),  # 3: unknown arch
    ]
    _emit_arch_section(insns, _SECCOMP_DENY_SYSCALLS["x86_64"], _CLONE_NR["x86_64"])
    _emit_arch_section(insns, _SECCOMP_DENY_SYSCALLS["aarch64"], _CLONE_NR["aarch64"])

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

    The host root starts read-only; workspace-only mode adds writable mounts for
    the working directory and Victor user state. PID/IPC and network namespace isolation are on by
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
        self._state_root: Optional[Path] = None
        self._pending_fds: List[int] = []

    def _writable_state_root(self) -> Path:
        """Resolve/create the default user-state mount exactly once."""
        if self._state_root is None:
            self._state_root = _prepare_victor_state_root()
        return self._state_root

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

        # Start from a read-only host root. Writable policy is opt-in below through the
        # workspace, Victor state, operator mounts, and the private tmpfs.
        flags += ["--ro-bind", "/", "/"]

        # Minimal /proc, /dev and a private /tmp.
        flags += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]

        # Workspace is writable only in workspace-only mode. The Victor user-state root is an
        # additional default in that mode; operator allow-lists remain independent.
        writable_mounts: dict[Path, str] = {}
        if cwd is not None:
            cwd_path = _absolute_mount_path(cwd)
            if self._workspace_only:
                writable_mounts[cwd_path] = "--bind"
            else:
                flags += ["--ro-bind", str(cwd_path), str(cwd_path)]

        if self._workspace_only:
            writable_mounts[self._writable_state_root()] = "--bind"
        for mount in self._allowed_mounts:
            if str(mount).strip():
                writable_mounts.setdefault(_absolute_mount_path(mount), "--bind-try")

        # Mount broad roots before nested roots. This preserves a workspace nested beneath the
        # state root and a state root nested beneath the workspace without either mount masking
        # the more specific exception.
        for mount, bind in sorted(writable_mounts.items(), key=lambda item: len(item[0].parts)):
            flags += [bind, str(mount), str(mount)]

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
        self._state_root: Optional[Path] = None

    def _writable_state_root(self) -> Path:
        """Resolve/create the default user-state exception exactly once."""
        if self._state_root is None:
            self._state_root = _prepare_victor_state_root()
        return self._state_root

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
            # Writes are scoped to the workspace, Victor state, and temp dirs.
            if cwd is not None:
                lines.append(f'(allow file-write* (subpath "{_absolute_mount_path(cwd)}"))')
            lines.append(f'(allow file-write* (subpath "{self._writable_state_root()}"))')
            for tmp in _SEATBELT_WRITABLE_TEMP:
                lines.append(f'(allow file-write* (subpath "{tmp}"))')
            # /dev/null is a black hole, not an exfiltration surface — tools
            # legitimately target it (curl -o /dev/null, discard redirects
            # opened by the wrapped process itself). Live-qualified on
            # macOS 26.2/arm64: without this literal, curl fails with
            # rc=56 "Failure writing output" under the deny-default floor.
            lines.append('(allow file-write* (literal "/dev/null"))')
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


def _warn_once(category: str, message: str) -> None:
    """Emit ``message`` once per process for ``category``."""
    if category not in _warned_categories:
        logger.warning(message)
        _warned_categories.add(category)


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
    """Resolve the effective policy from settings (ADR-035, fail-closed).

    Resolution order:

    1. Explicit ``sandbox_policy`` in settings (``disabled`` /
       ``preferred`` / ``required``) wins.
    2. The ``VICTOR_SANDBOX_POLICY`` environment variable (same values;
       invalid values are ignored).
    3. **Default: ``required``.** Shell execution is wrapped by an
       available sandbox or the caller refuses dispatch (fail-closed).
       The historical ``sandbox_enabled`` bool no longer maps to the
       fail-open ``preferred`` — that mode is now explicit opt-in only.
    """
    if sandbox_settings is not None:
        policy = getattr(sandbox_settings, "sandbox_policy", None)
        if isinstance(policy, str):
            policy = policy.strip().lower()
            if policy in ("disabled", "preferred", "required"):
                return policy
    import os

    env_policy = os.environ.get("VICTOR_SANDBOX_POLICY", "").strip().lower()
    if env_policy in ("disabled", "preferred", "required"):
        return env_policy
    return "required"


def resolve_sandbox_backend(sandbox_settings: Any) -> SandboxBackend:
    """Select a sandbox backend from settings.

    Policy semantics:

    * ``disabled`` — :class:`NoneSandbox` with a one-time opt-out warning
      (explicit opt-out only; the default posture is ``required``).
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
        # Explicit opt-out (ADR-035 decision 2): unsandboxed execution is
        # now always a deliberate choice — warn once so sessions know.
        _warn_once(
            "opt_out_disabled",
            "Sandbox policy is 'disabled' (explicit opt-out via settings or "
            "VICTOR_SANDBOX_POLICY): tools are running UNSANDBOXED. "
            "Set sandbox_policy to 'required' or unset the override to "
            "restore isolation.",
        )
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
                "enforcing backend; refusing execution (fail-closed). "
                "Escape hatch: set sandbox_policy: 'disabled' in settings or "
                "VICTOR_SANDBOX_POLICY=disabled in the environment.",
            )
        _warn_once(
            "preferred_degraded",
            f"Sandbox enabled but unsupported platform '{system}'; "
            "running tools UNSANDBOXED (fail-open). [DEGRADED]",
        )
        return NoneSandbox()

    if not backend.available():
        if policy == "required":
            raise SandboxRequiredUnavailableError(
                "binary_missing",
                f"Sandbox policy is 'required' but the '{backend.type_name}' backend is "
                f"unavailable (binary missing on {system}); refusing execution "
                "(fail-closed). Install the sandbox binary to enable isolation, "
                "or set sandbox_policy: 'disabled' in settings / "
                "VICTOR_SANDBOX_POLICY=disabled in the environment to opt out.",
            )
        _warn_once(
            "preferred_degraded",
            f"Sandbox enabled but the '{backend.type_name}' backend is unavailable "
            f"(binary missing on {system}); running tools UNSANDBOXED (fail-open). "
            "Install the sandbox binary to enable isolation. [DEGRADED]",
        )
        return NoneSandbox()

    return backend


@dataclass(frozen=True)
class PolicyResolution:
    """Tri-state sandbox-policy resolution (NEXT-01, Codex finding 20).

    * ``known`` — the effective policy resolved to one of the three
      configured values; ``policy`` is valid.
    * ``unknown`` — evaluation raised (unreadable policy field, faulty
      accessor). Unknown is NOT affirmative disabled mode: launch owners
      must REFUSE before launch, never fall through to a passthrough
      backend.
    """

    state: str  # "known" | "unknown"
    policy: str = ""

    @property
    def known(self) -> bool:
        return self.state == "known"

    @property
    def unknown(self) -> bool:
        return self.state == "unknown"


def resolve_effective_policy(sandbox_settings: Any) -> PolicyResolution:
    """Evaluate the effective policy WITHOUT inventing a default on failure.

    :func:`effective_sandbox_policy` (which this wraps) returns the
    fail-closed ``required`` default only for a SUCCESSFUL read; a raising
    settings accessor must surface as ``unknown`` — an unknowable policy is
    not permission to proceed.
    """
    try:
        return PolicyResolution(state="known", policy=effective_sandbox_policy(sandbox_settings))
    except Exception as exc:  # noqa: BLE001 - any evaluation fault is "unknown"
        _warn_once(
            "policy_unknown",
            f"Sandbox policy evaluation failed ({exc}); policy is UNKNOWN — "
            "governed launches must refuse (fail-closed).",
        )
        return PolicyResolution(state="unknown")


@dataclass(frozen=True)
class RequiredLaunchDecision:
    """Resolved decision for a required-policy launch (HB-02/03, ADR-L02).

    ``required_active`` is False unless the effective policy is required.
    ``refuse`` marks a required policy that cannot be enforced (typed
    resolver failures and generic resolver/configuration errors alike) —
    callers must refuse before any spawn and never degrade to no sandbox.
    ``refuse`` ALSO marks an UNKNOWN policy (evaluation fault): unknowable
    policy refuses before launch instead of returning an inactive,
    non-refusing decision (Codex finding 20).
    """

    required_active: bool = False
    backend: Optional["SandboxBackend"] = None
    refuse: bool = False
    reason: str = ""


def resolve_required_launch(sandbox_settings: Any) -> RequiredLaunchDecision:
    """Resolve whether a required policy can reach an enforcing launch.

    Consolidates the fail-closed rule shared by the canonical subprocess
    executor, the workflow sandbox executor and direct-spawn routes:

    * UNKNOWN policy (evaluation raised) — refuse before launch.
    * KNOWN ``required`` policy — any resolver or configuration failure
      refuses (never degrades).
    * KNOWN non-required policy (explicit ``disabled`` / ``preferred``) —
      non-active decision with the historical semantics.
    """
    resolution = resolve_effective_policy(sandbox_settings)
    if resolution.unknown:
        return RequiredLaunchDecision(
            refuse=True,
            reason=(
                "Sandbox policy could not be established (policy evaluation "
                "failed); refusing execution before launch (fail-closed)."
            ),
        )
    policy = resolution.policy
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


def _effective_policy_from_settings(*, on_error: str) -> str:
    """Effective policy from the global settings read; ``on_error`` when
    the read fails. Shared by the observability surface and the executor's
    HB-02 guard so there is exactly one settings-read derivation."""
    try:
        from victor.config.settings import get_settings

        return effective_sandbox_policy(getattr(get_settings(), "sandbox", None))
    except Exception:
        return on_error


def force_unsandboxed_for_session() -> tuple[bool, str]:
    """CLI ``--dangerously-unsandboxed``: explicit per-invocation opt-out.

    An explicit CLI request is the strongest signal, so this overrides BOTH
    the environment and any settings-file policy for the process lifetime.
    The ADR-035 opt-out warning still fires (deliberate unsandboxed
    execution must stay visible). Returns ``(applied, message)`` — on a
    settings failure the fail-closed posture stays in effect (the safe
    direction; never degrade on error).
    """
    try:
        from victor.config.sandbox_settings import SandboxSettings
        from victor.config.settings import get_settings

        settings = get_settings()
        if getattr(settings, "sandbox", None) is None:
            settings.sandbox = SandboxSettings(sandbox_policy="disabled")
        else:
            settings.sandbox.sandbox_policy = "disabled"
    except Exception as exc:
        return (
            False,
            f"could not apply the opt-out ({exc}); the fail-closed posture " "stays in effect",
        )
    _warn_once(
        "opt_out_disabled",
        "Sandbox policy is 'disabled' (--dangerously-unsandboxed): tools are "
        "running UNSANDBOXED for this session.",
    )
    return True, "tools will run UNSANDBOXED for this session"


def current_sandbox_posture() -> str:
    """Effective sandbox posture from global settings (observability seam).

    Used by the ``/system/status`` surface and doctor-style reporting so
    operators can see the ADR-035 posture a session is actually running
    under (``required`` / ``preferred`` / ``disabled``). A failed settings
    read reports ``unknown`` — never mask an unknowable-policy state
    (NEXT-01) as if enforcement were healthy.
    """
    return _effective_policy_from_settings(on_error="unknown")


__all__ = [
    "SandboxBackend",
    "NoneSandbox",
    "BubblewrapSandbox",
    "SeatbeltSandbox",
    "SandboxRequiredUnavailableError",
    "PolicyUnavailable",
    "PolicyResolution",
    "RequiredLaunchDecision",
    "build_seccomp_deny_program",
    "_effective_policy_from_settings",
    "current_sandbox_posture",
    "force_unsandboxed_for_session",
    "effective_sandbox_policy",
    "resolve_effective_policy",
    "resolve_required_launch",
    "resolve_sandbox_backend",
]
