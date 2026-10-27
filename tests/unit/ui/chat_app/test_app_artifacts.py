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

"""Inline image artifacts in the chat app (VAS-14 artifact channel).

``chainlit`` is faked via ``sys.modules`` (same pattern as the sibling suites);
the fake captures ``cl.Image`` constructions so the test asserts real decoding
and element attachment, not just that a branch executed.
"""

from __future__ import annotations

import base64
import importlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


class _FakeMessage:
    sent: List["_FakeMessage"] = []

    def __init__(self, content: str = "", actions: Optional[list] = None, elements=None) -> None:
        self.content = content
        self.actions = actions
        self.elements = elements or []
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


class _FakeImage:
    constructed: List["_FakeImage"] = []

    def __init__(self, name=None, content=b"", media_type="image/png", **kwargs) -> None:
        self.name = name
        self.content = content
        self.media_type = media_type
        _FakeImage.constructed.append(self)


class _FakeUserSession:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value


class _ToolClient:
    def __init__(self, events: list) -> None:
        self._events = events

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
    _FakeImage.constructed.clear()
    fake_cl = SimpleNamespace(
        on_chat_start=lambda f: f,
        on_message=lambda f: f,
        on_chat_end=lambda f: f,
        on_settings_update=lambda f: f,
        action_callback=lambda name: (lambda f: f),
        Message=_FakeMessage,
        Step=_FakeStep,
        Image=_FakeImage,
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


async def test_tool_images_render_as_inline_elements_and_placeholder(app_module):
    module, fake_cl = app_module
    b64 = base64.b64encode(b"\x89PNG-fake").decode()
    client = _ToolClient(
        [
            SimpleNamespace(
                event_type="tool_call",
                tool_name="mcp_browser_screenshot",
                metadata={"arguments": {"target": "e1_1"}},
            ),
            SimpleNamespace(
                event_type="tool_result",
                tool_name="mcp_browser_screenshot",
                content="screenshot captured",
                result=None,
                metadata={
                    "tool_result": {
                        "success": True,
                        "arguments": {"target": "e1_1"},
                        "images": [{"mime_type": "image/png", "b64": b64}],
                    }
                },
            ),
        ]
    )
    fake_cl.user_session.set(module._CLIENT_KEY, client)

    await module._run_turn(client, "screenshot")

    assert len(_FakeImage.constructed) == 1
    image = _FakeImage.constructed[0]
    assert image.name == "mcp_browser_screenshot-1"
    assert image.content == b"\x89PNG-fake"
    assert image.media_type == "image/png"
    artifact_messages = [m for m in _FakeMessage.sent if "image artifact" in m.content]
    assert artifact_messages, "an inline artifact message must follow the tool step"
    assert artifact_messages[0].elements == [image]
    tool_steps = [s for s in _FakeStep.created if "screenshot" in (s.name or "").lower()]
    assert tool_steps and "🖼️" in tool_steps[-1].output, "step keeps a text placeholder too"


async def test_undecodable_image_payload_renders_empty_not_fatal(app_module):
    module, fake_cl = app_module
    client = _ToolClient(
        [
            SimpleNamespace(
                event_type="tool_result",
                tool_name="mcp_browser_screenshot",
                content="captured",
                result=None,
                metadata={
                    "tool_result": {
                        "success": True,
                        "arguments": {},
                        "images": [{"mime_type": "image/png", "b64": "!!not-base64!!"}],
                    }
                },
            )
        ]
    )
    fake_cl.user_session.set(module._CLIENT_KEY, client)

    await module._run_turn(client, "screenshot")

    # base64.b64decode(validate=False) tolerates junk; either way the turn must
    # complete and the artifact message must exist — never a crashed render loop.
    assert any("image artifact" in m.content for m in _FakeMessage.sent)
