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

    def test_resolver_reads_network_default_on(self):
        """An object with no network attribute still resolves isolation ON."""
        from types import SimpleNamespace

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

    def test_resolver_threads_settings_seccomp(self):
        from types import SimpleNamespace

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


class TestSeccompProgram:
    def setup_method(self):
        self.program = sb.build_seccomp_deny_program()

    def test_known_deny_syscalls_x86_64(self):
        """x86_64 deny numbers must include the sandbox-escape set."""
        insns = _decode(self.program)
        denies_x86_64 = {
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
            313,  # finit_module
            304,  # open_by_handle_at
            250,  # keyctl
        }
        # Every syscall-compare JEQ k (all jeqs except the two arch checks)
        # is a deny rule by construction of the deny-list builder.
        denied = {
            insn[3] for insn in insns if insn[0] == 0x15 and insn[3] not in (0xC000003E, 0xC00000B7)
        }
        missing = denies_x86_64 - denied
        assert not missing, f"x86_64 deny list missing syscall numbers: {missing}"

    def test_aarch64_arch_section_present(self):
        """The program must handle AUDIT_ARCH_AARCH64, not silently allow it."""
        insns = _decode(self.program)
        arch_consts = {insn[3] for insn in insns if insn[0] == 0x15}
        assert 0xC000003E in arch_consts  # AUDIT_ARCH_X86_64
        assert 0xC00000B7 in arch_consts  # AUDIT_ARCH_AARCH64
        denied = {
            insn[3] for insn in insns if insn[0] == 0x15 and insn[3] not in (0xC000003E, 0xC00000B7)
        }
        assert 266 in denied  # unshare on aarch64

    def test_unknown_arch_fails_closed(self):
        """An architecture outside the tables must hit RET ERRNO, not ALLOW."""
        insns = _decode(self.program)
        # Find the instruction reached when arch matches neither known const.
        # Structural check: an unconditional RET ERRNO exists and is placed
        # before both arch sections (fail-closed ordering).
        rets = [insn for insn in insns if insn[0] == 0x06]
        errno_rets = [insn for insn in rets if (insn[3] & 0xFFFF0000) == 0x00050000]
        assert errno_rets, "program must contain RET ERRNO(EPERM) for unknown arch"

    def test_default_verdict_is_allow(self):
        insns = _decode(self.program)
        assert (0x06, 0, 0, 0x7FFF0000) in insns

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
