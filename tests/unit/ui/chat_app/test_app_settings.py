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

"""PR4 product: ChatSettings client-rebuild + best-effort session-restore seam.

Chainlit (incl. chainlit.input_widget) is faked via sys.modules so app.py imports without the
chat-ui extra, extended here with ChatSettings/input_widget/on_settings_update.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


class _FakeMessage:
    created: List["_FakeMessage"] = []

    def __init__(self, content: str = "", author: Optional[str] = None, **_: Any) -> None:
        self.content = content
        self.author = author
        _FakeMessage.created.append(self)

    async def send(self) -> "_FakeMessage":
        return self

    async def remove(self) -> None:
        return None

    async def stream_token(self, token: str) -> None:
        self.content += token

    async def update(self) -> None:
        return None


class _FakeChatSettings:
    created: List["_FakeChatSettings"] = []

    def __init__(self, inputs: list) -> None:
        self.inputs = inputs
        _FakeChatSettings.created.append(self)

    async def send(self) -> None:
        return None


class _FakeUserSession:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value


def _widget(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


@pytest.fixture
def app_module(monkeypatch):
    _FakeMessage.created.clear()
    _FakeChatSettings.created.clear()
    fake_cl = SimpleNamespace(
        on_chat_start=lambda f: f,
        on_message=lambda f: f,
        on_chat_end=lambda f: f,
        on_settings_update=lambda f: f,
        action_callback=lambda name: (lambda f: f),
        Message=_FakeMessage,
        Step=lambda **k: SimpleNamespace(**k),
        Action=lambda **k: SimpleNamespace(**k),
        ChatSettings=_FakeChatSettings,
        user_session=_FakeUserSession(),
    )
    fake_widgets = SimpleNamespace(
        Select=lambda **k: _widget(**k),
        Switch=lambda **k: _widget(**k),
        TextInput=lambda **k: _widget(**k),
    )
    monkeypatch.setitem(sys.modules, "chainlit", fake_cl)
    monkeypatch.setitem(sys.modules, "chainlit.input_widget", fake_widgets)
    sys.modules.pop("victor.ui.chat_app.app", None)
    module = importlib.import_module("victor.ui.chat_app.app")
    return module, fake_cl


class _Msg:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class _ClientWithHistory:
    def __init__(self) -> None:
        self.provider_name = "ollama"
        self.model = "qwen"

    async def get_messages(self, limit=None, role=None):
        return [_Msg("user", "earlier question"), _Msg("assistant", "earlier answer")]

    def get_available_providers(self):
        return ["ollama", "anthropic"]


class _FreshClient:
    def __init__(self) -> None:
        self.provider_name = "ollama"
        self.model = ""

    async def get_messages(self, limit=None, role=None):
        raise RuntimeError("VictorClient not initialized")  # fresh session

    def get_available_providers(self):
        return ["ollama"]


async def test_on_chat_start_restores_prior_history(app_module):
    module, fake_cl = app_module
    fake_cl.user_session.set(module._CLIENT_KEY, _ClientWithHistory())

    await module.on_chat_start()

    rendered = [(m.author, m.content) for m in _FakeMessage.created]
    assert ("You", "earlier question") in rendered
    assert ("Victor", "earlier answer") in rendered
    # When history is restored, the generic greeting is skipped.
    assert not any("is ready" in m.content for m in _FakeMessage.created)
    # Settings panel is still offered.
    assert _FakeChatSettings.created


async def test_on_chat_start_greets_when_no_history(app_module):
    module, fake_cl = app_module
    fake_cl.user_session.set(module._CLIENT_KEY, _FreshClient())

    await module.on_chat_start()

    assert any("is ready" in m.content for m in _FakeMessage.created)
    assert _FakeChatSettings.created  # settings always offered


async def test_settings_update_rebuilds_client_with_new_config(app_module, monkeypatch):
    module, fake_cl = app_module

    class _OldClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    captured = {}

    class _RecordingClient:
        def __init__(self, config) -> None:
            captured["config"] = config

    old = _OldClient()
    fake_cl.user_session.set(module._CLIENT_KEY, old)
    monkeypatch.setattr(module, "VictorClient", _RecordingClient)

    await module.on_settings_update(
        {
            "provider": "anthropic",
            "profile": "review",
            "model": "claude-opus",
            "tool_approval": False,
        }
    )

    assert old.closed  # old client released
    cfg = captured["config"]
    assert cfg.agent_profile == "review"
    assert cfg.provider_override.provider == "anthropic"
    assert cfg.provider_override.model == "claude-opus"
    assert cfg.tool_approval.enabled is False
    # New client is stored for the session.
    assert isinstance(fake_cl.user_session.get(module._CLIENT_KEY), _RecordingClient)


async def test_settings_update_keeps_approval_tools_when_enabled(app_module, monkeypatch):
    module, fake_cl = app_module
    captured = {}
    monkeypatch.setattr(
        module, "VictorClient", lambda config: captured.setdefault("config", config)
    )

    await module.on_settings_update({"provider": "ollama", "tool_approval": True})

    cfg = captured["config"]
    assert cfg.tool_approval.enabled is True
    assert cfg.tool_approval.ask_on_tools  # non-empty when approval is on
