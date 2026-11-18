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

"""FOU-S1a hardening tests: bwrap seccomp + default egress deny; Seatbelt
deny-default allowlist; pass_fds plumbing; live bwrap qualification.

ADR-035: a fail-closed posture over MVP backends would only relocate the
false sense of security, so the backends are hardened first.
"""

import platform
import shutil
import struct
from pathlib import Path


from victor.config.sandbox_settings import SandboxSettings
from victor.tools.sandbox import backends as sb
from victor.tools.sandbox import (
    BubblewrapSandbox,
    SeatbeltSandbox,
)


def _bwrap_available() -> bool:
    return platform.system() == "Linux" and shutil.which("bwrap") is not None


# -- Settings defaults -------------------------------------------------------


class TestSettingsDefaults:
    def test_network_isolation_defaults_on(self):
        """ADR-035: unshare-network egress deny is default-on."""
        assert SandboxSettings().sandbox_network_isolation is True

    def test_seccomp_defaults_on(self):
        """ADR-035: the bwrap seccomp filter is default-on."""
        assert SandboxSettings().sandbox_seccomp is True

    def test_resolver_reads_network_default_on(self, monkeypatch):
        """An object with no network attribute still resolves isolation ON."""
        from types import SimpleNamespace

        monkeypatch.setattr(sb.BubblewrapSandbox, "available", lambda self: True)
        settings = SimpleNamespace(sandbox_policy="required")
        backend = sb.resolve_sandbox_backend(settings)
        assert isinstance(backend, BubblewrapSandbox)
        assert backend._network_isolation is True


# -- bwrap hardening ---------------------------------------------------------


class TestBubblewrapHardening:
    def test_default_wraps_network_isolated(self):
        argv = BubblewrapSandbox().wrap_argv(["/bin/echo", "hi"], cwd=None)
        assert "--unshare-net" in argv

    def test_network_flag_still_can_be_disabled(self):
        argv = BubblewrapSandbox(network_isolation=False).wrap_argv(["/bin/echo"], cwd=None)
        assert "--unshare-net" not in argv

    def test_wraps_new_session(self):
        """--new-session detaches from the controlling tty (anti-TIOCSTI)."""
        argv = BubblewrapSandbox().wrap_argv(["/bin/echo"], cwd=None)
        assert "--new-session" in argv

    def test_default_wraps_seccomp(self):
        sandbox = BubblewrapSandbox()
        argv = sandbox.wrap_argv(["/bin/echo"], cwd=None)
        idx = argv.index("--seccomp")
        fd = int(argv[idx + 1])
        assert fd > 2
        # The BPF program must be reachable through the reported pass fd.
        assert sandbox.take_pass_fds() == [fd]

    def test_seccomp_disabled_via_flag(self):
        sandbox = BubblewrapSandbox(seccomp=False)
        argv = sandbox.wrap_argv(["/bin/echo"], cwd=None)
        assert "--seccomp" not in argv
        assert sandbox.take_pass_fds() == []

    def test_resolver_threads_settings_seccomp(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(sb.BubblewrapSandbox, "available", lambda self: True)
        settings = SimpleNamespace(
            sandbox_policy="required",
            sandbox_seccomp=False,
            sandbox_network_isolation=True,
        )
        backend = sb.resolve_sandbox_backend(settings)
        argv = backend.wrap_argv(["/bin/echo"], cwd=None)
        assert "--seccomp" not in argv
        assert "--unshare-net" in argv


# -- seccomp BPF program -----------------------------------------------------


def _decode(program: bytes):
    """Decode a classic-BPF program into (code, jt, jf, k) tuples."""
    assert len(program) % 8 == 0
    n = len(program) // 8
    flat = struct.unpack("<" + "HBBI" * n, program)
    return [tuple(flat[i * 4 : i * 4 + 4]) for i in range(n)]


_X86, _ARM = 0xC000003E, 0xC00000B7


class _Emulator:
    """Evaluate the seccomp program exactly as the kernel would."""

    def __init__(self, program: bytes):
        self.insns = _decode(program)

    def run(self, arch: int, nr: int, arg0: int = 0) -> str:
        acc = 0
        i = 0
        while True:
            code, jt, jf, k = self.insns[i]
            if code == 0x20:  # LD
                if k == sb._SECCOMP_DATA_ARCH:
                    acc = arch
                elif k == sb._SECCOMP_DATA_NR:
                    acc = nr
                else:
                    acc = arg0
                i += 1
                continue
            if code == 0x54:  # AND
                acc &= k
                i += 1
                continue
            if code == 0x15:  # JEQ
                # First two jeqs compare the loaded arch; later jeqs compare
                # whatever was loaded last (nr, then clone flags).
                take = jt if acc == k else jf
                i = i + 1 + take
                continue
            if code == 0x06:  # RET
                return "ALLOW" if k == 0x7FFF0000 else "EPERM"


class TestSeccompProgram:
    def setup_method(self):
        self.program = sb.build_seccomp_deny_program()
        self.vm = _Emulator(self.program)

    def test_deny_syscalls_x86_64(self):
        """Verified-number escape/integrity set must ALL deny on x86_64."""
        for nr in (
            165,  # mount
            166,  # umount2
            155,  # pivot_root
            161,  # chroot
            430,  # fsopen (new mount API)
            272,  # unshare
            308,  # setns
            321,  # bpf
            101,  # ptrace
            298,  # perf_event_open
            246,  # kexec_load
            320,  # kexec_file_load
            313,  # finit_module
            304,  # open_by_handle_at
            250,  # keyctl
            312,  # kcmp
            323,  # userfaultfd
            227,  # clock_settime
            438,  # pidfd_getfd
            440,  # process_madvise
            170,  # sethostname
        ):
            verdict = self.vm.run(_X86, nr)
            assert verdict == "EPERM", f"x86_64 syscall {nr}: {verdict}"

    def test_deny_syscalls_aarch64(self):
        """aarch64 uses the generic table — several numbers DIFFER from x86."""
        for nr in (
            97,  # unshare
            40,  # mount
            51,  # chroot
            268,  # setns
            273,  # finit_module
            294,  # kexec_file_load (320 on x86_64!)
            280,  # bpf
            117,  # ptrace
            58,  # vhangup
            18,  # lookup_dcookie
            262,  # fanotify_init
            112,  # clock_settime
        ):
            verdict = self.vm.run(_ARM, nr)
            assert verdict == "EPERM", f"aarch64 syscall {nr}: {verdict}"

    def test_benign_syscalls_stay_allowed(self):
        assert self.vm.run(_X86, 0) == "ALLOW"  # read
        assert self.vm.run(_X86, 1) == "ALLOW"  # write
        assert self.vm.run(_X86, 59) == "ALLOW"  # execve
        assert self.vm.run(_ARM, 63) == "ALLOW"  # read (arm)
        assert self.vm.run(_ARM, 64) == "ALLOW"  # write (arm)

    def test_no_collateral_denies(self):
        """Numbers previously denied by mistake must NOT deny."""
        assert self.vm.run(_X86, 203) == "ALLOW"  # was wrongly denied (reserved)
        assert self.vm.run(_X86, 294) == "ALLOW"  # was wrongly denied (reserved)
        assert self.vm.run(_ARM, 266) == "ALLOW"  # clock_adjtime (was collateral)

    def test_clone_flag_filter(self):
        """clone() stays allowed for threads but denies namespace creation."""
        assert self.vm.run(_X86, 56, arg0=0x00000000) == "ALLOW"  # plain thread
        assert self.vm.run(_ARM, 220, arg0=0x00000000) == "ALLOW"
        for flag in (0x10000000, 0x40000000, 0x00020000, 0x20000000):  # USER/NET/NS/PID
            assert self.vm.run(_X86, 56, arg0=flag) == "EPERM"
            assert self.vm.run(_ARM, 220, arg0=flag) == "EPERM"

    def test_unknown_arch_fails_closed(self):
        """Compat ABIs (int $0x80 / x32) get EPERM across the board."""
        for arch in (0x40000003, 0x40000002, 0x80000000):
            assert self.vm.run(arch, 0) == "EPERM"

    def test_within_bpf_max_insns(self):
        assert len(self.program) // 8 <= 4096


# -- Seatbelt deny-default ---------------------------------------------------


class TestSeatbeltDenyDefault:
    def test_profile_has_no_allow_default(self):
        profile = SeatbeltSandbox(network_isolation=True).build_profile(None)
        assert "(allow default)" not in profile
        assert "(deny default)" in profile

    def test_profile_allows_read_and_exec(self):
        profile = SeatbeltSandbox().build_profile(None)
        assert "(allow process-exec*" in profile
        assert "(allow file-read*" in profile

    def test_profile_scopes_writes_to_workspace_and_temp(self):
        profile = SeatbeltSandbox().build_profile(Path("/work/ws"))
        assert '(allow file-write* (subpath "/work/ws"))' in profile
        for tmp in sb._SEATBELT_WRITABLE_TEMP:
            assert f'(allow file-write* (subpath "{tmp}"))' in profile

    def test_profile_denies_network_by_default(self):
        profile = SeatbeltSandbox().build_profile(None)
        assert "(deny network*)" in profile

    def test_profile_allows_network_when_isolation_off(self):
        profile = SeatbeltSandbox(network_isolation=False).build_profile(None)
        assert "(deny network*)" not in profile
        assert "(allow network*)" in profile

    def test_profile_no_blanket_write_allow(self):
        profile = SeatbeltSandbox().build_profile(None)
        assert "(allow file-write*)" not in profile

    def test_resolver_darwin_gets_hardened_profile(self, monkeypatch):
        monkeypatch.setattr(sb.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(sb.SeatbeltSandbox, "available", lambda self: True)
        backend = sb.resolve_sandbox_backend(SandboxSettings(sandbox_policy="required"))
        assert isinstance(backend, SeatbeltSandbox)
        profile = backend.build_profile(None)
        assert "(deny default)" in profile
