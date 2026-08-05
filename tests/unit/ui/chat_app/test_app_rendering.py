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

"""Chat UI ``on_message`` tool-step rendering.

``chainlit`` is faked via ``sys.modules`` so the app module imports without the
optional ``chat-ui`` extra; we then drive ``on_message`` with mapped tool events
and assert the rendered step carries duration + follow-up telemetry.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


@dataclass
class _FakeEvent:
    event_type: Any
    content: Optional[str] = None
    tool_name: Optional[str] = None
    result: Any = None
    metadata: Optional[Dict[str, Any]] = None


class _FakeMessage:
    def __init__(self, content: str = "", actions: Optional[list] = None) -> None:
        self.content = content
        self.actions = actions or []
        self.tokens: List[str] = []

    async def send(self) -> "_FakeMessage":
        return self

    async def stream_token(self, token: str) -> None:
        self.tokens.append(token)

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


class _FakeUserSession:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value


class _FakeClient:
    def __init__(self, events: List[_FakeEvent]) -> None:
        self._events = events

    async def stream(self, message: str):
        for event in self._events:
            yield event

    async def close(self) -> None:
        return None


@pytest.fixture
def app_module(monkeypatch):
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


async def test_tool_step_shows_duration_and_follow_ups(app_module):
    module, fake_cl = app_module
    events = [
        _FakeEvent("tool_call", tool_name="read", metadata={"arguments": {"path": "x.py"}}),
        _FakeEvent(
            "tool_result",
            tool_name="read",
            result={
                "success": True,
                "elapsed": 0.012,
                "original_result": "the real file output",
                "arguments": {"path": "x.py"},
                "follow_up_suggestions": [{"suggestion": "Read the imported module"}],
            },
        ),
    ]
    client = _FakeClient(events)
    fake_cl.user_session.set(module._CLIENT_KEY, client)

    await module.on_message(_FakeMessage(content="read x.py"))

    # The child tool step is the one carrying the tool's arguments/output (the
    # "🔧 tools" group step has no input).
    step = next(s for s in _FakeStep.created if s.type == "tool" and s.input == {"path": "x.py"})
    # Duration is appended to the step label (e.g. "read(path='x.py') · 12ms").
    assert "·" in (step.name or "")
    assert "ms" in (step.name or "")
    # Real output is rendered (via the markdown presenter), not the "/expand" placeholder.
    assert "the real file output" in (step.output or "")
    # Follow-up suggestions are grouped with the tool step that produced them.
    assert "Next steps" in (step.output or "")
    assert "Read the imported module" in (step.output or "")


def test_format_follow_ups_is_defensive(app_module):
    module, _ = app_module
    assert module._format_follow_ups([]) == ""
    assert module._format_follow_ups([{"nope": 1}]) == ""
    out = module._format_follow_ups([{"command": "ls -la"}, "not-a-dict"])
    assert "ls -la" in out
