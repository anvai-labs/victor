# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Terminal execute route policy.

First tests for /terminal/execute: workspace containment, and the
server-side dangerous-command policy. Previously ``require_approval`` was
caller-attested — a client could send ``require_approval=false`` (or omit
it) and the server executed blocklisted commands (co-design review U7-F2).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from victor.integrations.api import fastapi_server


class _FakeOrchestrator:
    async def chat(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(content="ok", tool_calls=[])

    async def graceful_shutdown(self) -> None:
        return None


def _create_server(monkeypatch, tmp_path: Path):
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


def _execute(client: TestClient, command: str, **overrides):
    payload = {"command": command, "require_approval": False, **overrides}
    return client.post("/terminal/execute", json=payload)


def test_dangerous_command_pending_even_when_caller_declines_approval(monkeypatch, tmp_path):
    """Route-blocklisted command (chmod -R 777 — route list, not the request
    model's list) + require_approval=false (attested bypass attempt) →
    pending, never executed."""
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    client = TestClient(server.app)

    response = _execute(client, "chmod -R 777 /tmp/victor-route-policy-probe")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["is_dangerous"] is True
    assert not body.get("output")  # nothing ran
    assert body.get("exit_code") is None


def test_dangerous_command_pending_with_approval_requested(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    client = TestClient(server.app)

    response = _execute(client, "sudo rm -rf /tmp/victor-probe-dir", require_approval=True)
    assert response.status_code == 422  # request-model validator rejects it outright


def test_benign_command_executes(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    client = TestClient(server.app)

    response = _execute(client, "echo hello-terminal-test")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("completed", "success", "running"), body
    assert "hello-terminal-test" in body.get("output", "")


def test_working_dir_outside_workspace_rejected(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    client = TestClient(server.app)

    response = _execute(client, "echo x", working_dir=str(tmp_path.parent))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "within workspace" in body.get("output", "")


def test_sibling_prefix_working_dir_rejected(monkeypatch, tmp_path):
    """Negative: string-prefix containment admitted a sibling directory
    (/ws/victor-evil beside workspace /ws/victor). Found by adversarial
    review."""
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    client = TestClient(server.app)

    sibling = tmp_path.parent / (tmp_path.name + "-evil")
    response = _execute(client, "pwd", working_dir=str(sibling))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "within workspace" in body.get("output", "")


# ---------------------------------------------------------------------------
# Direct-spawn convergence: /terminal/execute follows the required policy
# ---------------------------------------------------------------------------


def _install_required(monkeypatch, backend_available: bool) -> None:
    from types import SimpleNamespace

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


def _trap_terminal_spawn(monkeypatch, launched: list) -> None:
    import victor.integrations.api.routes.terminal_routes as tr

    async def _trap(*args, **kwargs):
        launched.append([str(a) for a in args])
        raise RuntimeError("synthetic spawn trap")

    monkeypatch.setattr(tr.asyncio, "create_subprocess_exec", _trap)


def test_required_policy_wraps_terminal_launch(monkeypatch, tmp_path):
    """Under required policy with an enforcing backend, the terminal route's
    actual launch carries the backend wrapper, not the raw argv."""
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    _install_required(monkeypatch, backend_available=True)
    launched: list = []
    _trap_terminal_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    response = _execute(client, "echo hi")
    assert launched, "expected a spawn attempt to inspect"
    assert launched[0][0] == "bwrap", f"raw argv launched: {launched[0]}"
    assert "synthetic-command" not in launched[0]


def test_required_policy_refuses_terminal_launch_when_unenforceable(monkeypatch, tmp_path):
    """Under required policy with no enforcing backend, the terminal route
    refuses before any spawn."""
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    _install_required(monkeypatch, backend_available=False)
    launched: list = []
    _trap_terminal_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    response = _execute(client, "echo hi")
    body = response.json()
    assert launched == [], "no spawn may be attempted under unenforceable required"
    assert body["status"] == "failed"
    assert "required" in body.get("output", "").lower()


def test_disabled_policy_keeps_terminal_launch_raw(monkeypatch, tmp_path):
    """Default (no sandbox configured) keeps the historical raw exec."""
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    launched: list = []
    _trap_terminal_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    response = _execute(client, "echo hi")
    assert launched and launched[0][0] != "bwrap"


def test_terminal_refuses_when_policy_unreadable(monkeypatch, tmp_path):
    """NEXT-01: unreadable settings refuse the terminal launch (zero spawn)."""
    import victor.config.settings as cs

    def _boom():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(cs, "get_settings", _boom)
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    launched: list = []
    _trap_terminal_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    response = _execute(client, "echo hi")
    assert launched == []
    body = response.json()
    assert body["status"] == "failed"
    assert "policy" in body.get("output", "").lower()


# ---------------------------------------------------------------------------
# FEP-0044: direct terminal launches consume request-admitted policy
# ---------------------------------------------------------------------------


def _install_globals(monkeypatch, policy: str) -> None:
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(sandbox=SandboxSettings(sandbox_policy=policy))
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)


def _admitted_snapshot(policy: str):
    from victor.runtime.policy_snapshot import RunPolicySnapshot

    return RunPolicySnapshot.from_settings(_install_snapshot_settings(policy))


def _install_snapshot_settings(policy: str):
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings

    return SimpleNamespace(sandbox=SandboxSettings(sandbox_policy=policy))


def test_admitted_required_terminal_beats_reloaded_disabled(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    _install_required(monkeypatch, backend_available=True)
    _install_globals(monkeypatch, "disabled")
    monkeypatch.setattr(
        "victor.runtime.policy_snapshot.current_run_policy_snapshot",
        lambda: _admitted_snapshot("required"),
    )
    launched: list = []
    _trap_terminal_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    _execute(client, "echo hi")

    assert launched and launched[0][0] == "bwrap", f"admitted policy lost: {launched}"


def test_admitted_disabled_terminal_is_not_tightened_by_reloaded_required(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path)
    server._orchestrator = _FakeOrchestrator()
    _install_globals(monkeypatch, "required")
    _install_required(monkeypatch, backend_available=False)
    monkeypatch.setattr(
        "victor.runtime.policy_snapshot.current_run_policy_snapshot",
        lambda: _admitted_snapshot("disabled"),
    )
    launched: list = []
    _trap_terminal_spawn(monkeypatch, launched)
    client = TestClient(server.app)

    _execute(client, "echo hi")

    assert launched and launched[0][0] != "bwrap", f"admitted policy tightened: {launched}"
