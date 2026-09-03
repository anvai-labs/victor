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
