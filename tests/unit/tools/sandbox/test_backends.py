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

"""Tests for sandbox backend argv/profile construction and resolution."""

from pathlib import Path

import pytest

from victor.tools.sandbox import backends as sb
from victor.tools.sandbox import (
    BubblewrapSandbox,
    NoneSandbox,
    SeatbeltSandbox,
    resolve_sandbox_backend,
)


class _Settings:
    def __init__(self, **kw):
        self.sandbox_enabled = kw.get("sandbox_enabled", True)
        self.sandbox_filesystem_mode = kw.get("sandbox_filesystem_mode", "workspace-only")
        self.sandbox_namespace_restrictions = kw.get("sandbox_namespace_restrictions", True)
        self.sandbox_network_isolation = kw.get("sandbox_network_isolation", False)
        self.sandbox_allowed_mounts = kw.get("sandbox_allowed_mounts", [])


# -- NoneSandbox ------------------------------------------------------------


def test_none_sandbox_is_passthrough():
    s = NoneSandbox()
    assert s.available() is True
    assert s.type_name == "none"
    assert s.wrap_argv(["echo", "hi"], Path("/x")) == ["echo", "hi"]


# -- BubblewrapSandbox ------------------------------------------------------


def test_bwrap_argv_structure():
    b = BubblewrapSandbox()
    argv = b.wrap_argv(["/bin/sh", "-c", "echo hi"], Path("/work"))
    assert argv[0] == "bwrap"
    assert "--die-with-parent" in argv
    # system roots bound read-only
    assert "--ro-bind-try" in argv and "/usr" in argv
    # cwd bound writable in workspace-only mode
    joined = " ".join(argv)
    assert "--bind /work /work" in joined
    # command appended after the -- separator
    assert argv[-3:] == ["/bin/sh", "-c", "echo hi"]
    assert "--" in argv


def test_bwrap_network_and_namespace_toggles():
    on = BubblewrapSandbox(network_isolation=True, namespace_restrictions=True)
    argv = on.wrap_argv(["x"], Path("/w"))
    assert "--unshare-net" in argv
    assert "--unshare-pid" in argv and "--unshare-ipc" in argv

    off = BubblewrapSandbox(network_isolation=False, namespace_restrictions=False)
    argv2 = off.wrap_argv(["x"], Path("/w"))
    assert "--unshare-net" not in argv2
    assert "--unshare-pid" not in argv2


def test_bwrap_allowed_mounts_and_readonly_workspace():
    b = BubblewrapSandbox(workspace_only=False, allowed_mounts=["/data", "/cache"])
    argv = b.wrap_argv(["x"], Path("/w"))
    joined = " ".join(argv)
    # non-workspace mode binds cwd read-only
    assert "--ro-bind /w /w" in joined
    assert "--bind-try /data /data" in joined
    assert "--bind-try /cache /cache" in joined


# -- SeatbeltSandbox --------------------------------------------------------


def test_seatbelt_argv_and_profile():
    s = SeatbeltSandbox(network_isolation=True)
    argv = s.wrap_argv(["echo", "hi"], Path("/work"))
    assert argv[0] == "sandbox-exec"
    assert argv[1] == "-p"
    assert argv[-2:] == ["echo", "hi"]
    profile = argv[2]
    assert "(version 1)" in profile
    assert '(allow file-write* (subpath "/work"))' in profile
    assert "(deny network*)" in profile


def test_seatbelt_no_network_deny_when_disabled():
    s = SeatbeltSandbox(network_isolation=False)
    profile = s.build_profile(Path("/work"))
    assert "(deny network*)" not in profile


def test_seatbelt_full_fs_mode_has_no_write_deny():
    s = SeatbeltSandbox(workspace_only=False)
    profile = s.build_profile(Path("/work"))
    assert "(deny file-write*)" not in profile


# -- resolve_sandbox_backend ------------------------------------------------


def test_resolve_disabled_returns_none():
    assert resolve_sandbox_backend(_Settings(sandbox_enabled=False)).type_name == "none"
    assert resolve_sandbox_backend(None).type_name == "none"


def test_resolve_linux_with_bwrap(monkeypatch):
    monkeypatch.setattr(sb.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/bin/bwrap")
    backend = resolve_sandbox_backend(_Settings())
    assert backend.type_name == "bwrap"


def test_resolve_darwin_with_seatbelt(monkeypatch):
    monkeypatch.setattr(sb.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    backend = resolve_sandbox_backend(_Settings())
    assert backend.type_name == "seatbelt"


def test_resolve_fail_open_when_binary_missing(monkeypatch, caplog):
    monkeypatch.setattr(sb.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    monkeypatch.setattr(sb, "_warned_unavailable", False)
    with caplog.at_level("WARNING"):
        backend = resolve_sandbox_backend(_Settings())
    assert backend.type_name == "none"
    assert any("UNSANDBOXED" in r.message for r in caplog.records)


def test_resolve_fail_open_unsupported_platform(monkeypatch):
    monkeypatch.setattr(sb.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sb, "_warned_unavailable", False)
    assert resolve_sandbox_backend(_Settings()).type_name == "none"


def test_warn_once_only(monkeypatch, caplog):
    monkeypatch.setattr(sb.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    monkeypatch.setattr(sb, "_warned_unavailable", False)
    with caplog.at_level("WARNING"):
        resolve_sandbox_backend(_Settings())
        resolve_sandbox_backend(_Settings())
    warnings = [r for r in caplog.records if "UNSANDBOXED" in r.message]
    assert len(warnings) == 1
