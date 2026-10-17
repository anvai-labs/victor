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
from unittest.mock import AsyncMock
import asyncio

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


def _select(**kwargs: Any) -> SimpleNamespace:
    # Chainlit 2 Select resolves initial_value/index and overwrites initial.
    values = kwargs.get("values", [])
    index = kwargs.get("initial_index")
    kwargs["initial"] = values[index] if index is not None else kwargs.get("initial_value")
    return _widget(**kwargs)


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
        Select=_select,
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

        def set_approval_handler(self, handler):
            captured["approval_handler"] = handler

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
    assert captured["approval_handler"] is module.chainlit_approval_handler
    assert any("new conversation" in m.content for m in _FakeMessage.created)


async def test_settings_update_keeps_approval_tools_when_enabled(app_module, monkeypatch):
    module, fake_cl = app_module
    captured = {}

    def client_factory(config):
        captured["config"] = config
        return SimpleNamespace(set_approval_handler=lambda handler: None)

    monkeypatch.setattr(module, "VictorClient", client_factory)

    await module.on_settings_update({"provider": "ollama", "tool_approval": True})

    cfg = captured["config"]
    assert cfg.tool_approval.enabled is True
    assert cfg.tool_approval.ask_on_tools  # non-empty when approval is on


async def test_settings_show_launch_profile_without_inventing_overrides(app_module, monkeypatch):
    module, fake_cl = app_module
    monkeypatch.setenv(module._PROFILE_ENV, "subscription")
    monkeypatch.setattr(module, "_profile_names", lambda: ["default", "subscription"])
    # Active provider/model are resolved values, not user-selected overrides.
    client = _ClientWithHistory()
    await module._send_settings(client)
    controls = {widget.id: widget for widget in _FakeChatSettings.created[-1].inputs}
    assert controls["profile"].initial == "subscription"
    assert controls["provider"].initial == module._PROFILE_DEFAULT
    assert controls["model"].initial == ""


async def test_unchanged_settings_preserve_client_and_conversation(app_module, monkeypatch):
    module, fake_cl = app_module
    monkeypatch.setenv(module._PROFILE_ENV, "subscription")
    client = _ClientWithHistory()
    fake_cl.user_session.set(module._CLIENT_KEY, client)
    await module._send_settings(client)
    settings = {widget.id: widget.initial for widget in _FakeChatSettings.created[-1].inputs}
    monkeypatch.setattr(
        module, "VictorClient", lambda config: pytest.fail("rebuilt unchanged client")
    )
    await module.on_settings_update(settings)
    assert fake_cl.user_session.get(module._CLIENT_KEY) is client


async def test_invalid_settings_do_not_close_existing_client(app_module, monkeypatch):
    module, fake_cl = app_module
    old = SimpleNamespace(close=AsyncMock())
    fake_cl.user_session.set(module._CLIENT_KEY, old)

    def reject(config):
        raise ValueError("invalid configuration")

    monkeypatch.setattr(module, "VictorClient", reject)
    with pytest.raises(ValueError, match="invalid configuration"):
        await module.on_settings_update({"provider": "anthropic"})
    old.close.assert_not_called()
    assert fake_cl.user_session.get(module._CLIENT_KEY) is old


async def test_failed_close_does_not_publish_replacement(app_module, monkeypatch):
    module, fake_cl = app_module
    old = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("cleanup failed")))
    fake_cl.user_session.set(module._CLIENT_KEY, old)
    config = module._current_config()
    monkeypatch.setattr(
        module,
        "VictorClient",
        lambda config: SimpleNamespace(set_approval_handler=lambda handler: None),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await module.on_settings_update({"provider": "anthropic"})
    assert fake_cl.user_session.get(module._CLIENT_KEY) is old
    assert module._current_config() == config


async def test_settings_transition_serializes_turn_admission(app_module, monkeypatch):
    module, fake_cl = app_module
    closing, release = asyncio.Event(), asyncio.Event()

    async def close():
        closing.set()
        await release.wait()

    old = SimpleNamespace(close=close)
    new = SimpleNamespace(set_approval_handler=lambda handler: None)
    fake_cl.user_session.set(module._CLIENT_KEY, old)
    monkeypatch.setattr(module, "VictorClient", lambda config: new)
    seen = []

    async def turn(client, content):
        seen.append(client)

    monkeypatch.setattr(module, "_run_turn", turn)
    change = asyncio.create_task(module.on_settings_update({"provider": "anthropic"}))
    await asyncio.wait_for(closing.wait(), 1)
    start = asyncio.create_task(module._start_turn("question"))
    await asyncio.sleep(0)
    assert seen == []
    release.set()
    await asyncio.wait_for(asyncio.gather(change, start), 1)
    assert seen == [new]


def test_ui_approval_policy_covers_canonical_mutating_file_tools(app_module):
    from victor.framework.policies.builtins import AskOnToolsPolicy

    module, _ = app_module
    policy = AskOnToolsPolicy(module._session_config(None).tool_approval.ask_on_tools)
    assert policy.applies_to("write")
    assert policy.applies_to("edit")
    assert policy.applies_to("shell")
    assert not policy.applies_to("read")


@pytest.mark.parametrize("value", ["false", 0, None])
async def test_malformed_approval_setting_preserves_client(app_module, value):
    module, fake_cl = app_module
    old = SimpleNamespace(close=AsyncMock())
    fake_cl.user_session.set(module._CLIENT_KEY, old)
    with pytest.raises(ValueError, match="approval must be a boolean"):
        await module.on_settings_update({"tool_approval": value})
    old.close.assert_not_called()
