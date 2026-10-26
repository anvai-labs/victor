# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Remote-mode wiring in the chat app (VAS-14: remote Chainlit speaks the API).

``chainlit`` is faked via ``sys.modules`` (same pattern as test_app_rendering).
These pin the three integration facts: the env var selects the HTTP transport,
remote sessions never half-arm approvals, and the render loop drives the SAME
RenderAction contract from wire dicts (including member lanes).
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


class _FakeMessage:
    sent: List["_FakeMessage"] = []

    def __init__(self, content: str = "", actions: Optional[list] = None) -> None:
        self.content = content
        self.actions = actions
        _FakeMessage.sent.append(self)

    async def send(self) -> "_FakeMessage":
        return self

    async def stream_token(self, token: str) -> None:
        self.content += token

    async def update(self) -> None:
        return None

    async def remove(self) -> None:
        return None


class _FakeStep:
    created: List["_FakeStep"] = []

    def __init__(self, name: Optional[str] = None, type: Optional[str] = None) -> None:
        self.name = name
        self.type = type
        self.input: Any = None
        self.output: Any = None
        self.is_error = False
        _FakeStep.created.append(self)

    async def __aenter__(self) -> "_FakeStep":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def update(self) -> None:
        return None


class _FakeUserSession:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value


class _WireClient:
    """Stands in for RemoteChatClient inside the render loop: is_remote + wire dicts."""

    is_remote = True
    provider_name = "remote"

    def __init__(self, wire_events: List[dict]) -> None:
        self._events = wire_events

    async def stream(self, message: str):
        for event in self._events:
            yield event

    def get_available_providers(self) -> list:
        return []

    def get_last_turn_cost(self):
        return None

    async def close(self) -> None:
        return None


@pytest.fixture
def app_module(monkeypatch):
    _FakeMessage.sent.clear()
    _FakeStep.created.clear()
    fake_cl = SimpleNamespace(
        on_chat_start=lambda f: f,
        on_message=lambda f: f,
        on_chat_end=lambda f: f,
        on_settings_update=lambda f: f,
        action_callback=lambda name: (lambda f: f),
        Message=_FakeMessage,
        Step=_FakeStep,
        Action=lambda **kwargs: SimpleNamespace(**kwargs),
        ChatSettings=lambda inputs: SimpleNamespace(send=lambda: None),
        user_session=_FakeUserSession(),
    )
    monkeypatch.setitem(sys.modules, "chainlit", fake_cl)
    monkeypatch.setitem(
        sys.modules,
        "chainlit.input_widget",
        SimpleNamespace(
            Select=lambda **k: SimpleNamespace(**k),
            Switch=lambda **k: SimpleNamespace(**k),
            TextInput=lambda **k: SimpleNamespace(**k),
        ),
    )
    sys.modules.pop("victor.ui.chat_app.app", None)
    module = importlib.import_module("victor.ui.chat_app.app")
    return module, fake_cl


async def test_remote_env_builds_remote_client_with_approvals_off(app_module, monkeypatch):
    module, _ = app_module
    monkeypatch.setenv("VICTOR_UI_REMOTE_URL", "http://127.0.0.1:8757")
    monkeypatch.setenv("VICTOR_UI_REMOTE_API_KEY", "sk-remote")

    client = module._configured_client(module._session_config("default", approval=True))
    assert getattr(client, "is_remote", False) is True

    config = module._session_config("default", approval=True)
    assert config.tool_approval.enabled is False, "remote mode must never half-arm approvals"


async def test_remote_env_absent_builds_embedded_client(app_module, monkeypatch):
    module, _ = app_module
    monkeypatch.delenv("VICTOR_UI_REMOTE_URL", raising=False)
    client = module._configured_client(module._session_config("default"))
    assert getattr(client, "is_remote", False) is False


async def test_render_loop_drives_wire_events_through_the_same_actions(app_module):
    module, fake_cl = app_module
    client = _WireClient(
        [
            {"event": "thinking", "content": "pondering"},
            {"event": "content", "content": "Working on it."},
            {
                "event": "tool_call",
                "tool": "read",
                "call_id": "c1",
                "arguments": {"path": "x.py"},
            },
            {
                "event": "tool_result",
                "tool": "read",
                "call_id": "c1",
                "result": "file contents",
                "success": True,
                "elapsed_ms": 12,
            },
            {"event": "stream_end"},
        ]
    )
    fake_cl.user_session.set(module._CLIENT_KEY, client)

    await module._run_turn(client, "read x.py")

    texts = [m.content for m in _FakeMessage.sent]
    assert any("Working on it." in t for t in texts)
    tool_steps = [s for s in _FakeStep.created if "read" in (s.name or "")]
    assert tool_steps, "wire tool_call/tool_result must render a tool step"
    assert "file contents" in (tool_steps[-1].output or "")


async def test_render_loop_renders_member_lanes_from_wire_events(app_module):
    module, fake_cl = app_module
    client = _WireClient(
        [
            {"event": "member_start", "member_id": "researcher", "content": "starting"},
            {"event": "member_completed", "member_id": "researcher", "content": "done"},
            {"event": "stream_end"},
        ]
    )
    fake_cl.user_session.set(module._CLIENT_KEY, client)

    await module._run_turn(client, "go")

    joined = "\n".join(m.content for m in _FakeMessage.sent)
    assert "researcher" in joined
    assert "starting" in joined
    assert "done" in joined
