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

"""Git route spawn gate: required policy follows the actual launch (R2-01)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient

from victor.integrations.api import fastapi_server


def _create_server(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    return fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
    )


def _install_required(monkeypatch, backend_available: bool) -> None:
    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(
        sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy="required")
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
    import victor.tools.sandbox.backends as sb_mod

    monkeypatch.setattr(sb_mod.platform, "system", lambda: "Linux")
    if backend_available:
        monkeypatch.setattr(sb_mod.shutil, "which", lambda name: "/usr/bin/bwrap")
    else:
        monkeypatch.setattr(sb_mod.shutil, "which", lambda name: None)


def _trap_git_spawn(monkeypatch, launched: list, *, returncode: int = 0, stdout: str = ""):
    """Trap subprocess.run at the OS boundary and record launched argv."""

    class _FakeCompleted:
        def __init__(self) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    real_run = subprocess.run

    def _trap(argv, *args, **kwargs):
        launched.append([str(a) for a in argv])
        if launched and launched[-1][0] == "bwrap":
            # The wrapper intercepts the launch; simulate the inner git call
            # having succeeded so routes can complete.
            return _FakeCompleted()
        return real_run(argv, *args, **kwargs) if argv[0] != "git" else _FakeCompleted()

    monkeypatch.setattr(subprocess, "run", _trap)


def test_git_commit_refused_when_required_unenforceable(monkeypatch, tmp_path):
    """Required policy with no enforcing backend: /git/commit must refuse
    before ANY spawn (no `git add`, no `git commit`)."""
    server = _create_server(monkeypatch, tmp_path)
    _install_required(monkeypatch, backend_available=False)
    launched: list = []
    _trap_git_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    response = client.post("/git/commit", json={"message": "probe"})
    body = response.json()
    assert launched == [], f"spawn attempted under unenforceable required: {launched}"
    assert body.get("success") is False
    assert "required" in body.get("error", "").lower()


def test_git_commit_launch_wrapped_when_required_enforceable(monkeypatch, tmp_path):
    """Required policy with an enforcing backend: the git launches carry the
    backend wrapper on the actual argv."""
    server = _create_server(monkeypatch, tmp_path)
    _install_required(monkeypatch, backend_available=True)
    launched: list = []
    _trap_git_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    response = client.post("/git/commit", json={"message": "probe"})
    assert launched, "expected a spawn attempt to inspect"
    assert launched[0][0] == "bwrap", f"raw argv launched: {launched[0]}"
    assert any("commit" in a for argv in launched for a in argv)


def test_git_status_refused_when_required_unenforceable(monkeypatch, tmp_path):
    """Read helpers are gated too: `git diff` can execute repo-configured
    drivers, so a fixed argv is not proof of containment."""
    server = _create_server(monkeypatch, tmp_path)
    _install_required(monkeypatch, backend_available=False)
    launched: list = []
    _trap_git_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    response = client.get("/git/status")
    assert launched == []
    assert "required" in response.json().get("error", "").lower()


def test_git_status_raw_exec_on_disabled_policy(monkeypatch, tmp_path):
    """Default (no sandbox configured) keeps the historical raw exec."""
    server = _create_server(monkeypatch, tmp_path)
    launched: list = []
    _trap_git_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    response = client.get("/git/status")
    assert launched and launched[0][0] == "git"
    assert response.status_code == 200
