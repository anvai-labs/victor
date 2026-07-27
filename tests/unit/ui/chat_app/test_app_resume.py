# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Cross-visit resume UX in the Chainlit app: picker + resume action callback."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


class _FakeMessage:
    created: List["_FakeMessage"] = []

    def __init__(self, content: str = "", actions: Optional[list] = None, author: str = "") -> None:
        self.content = content
        self.actions = actions or []
        self.author = author
        _FakeMessage.created.append(self)

    async def send(self) -> "_FakeMessage":
        return self


class _FakeAction:
    def __init__(
        self, name: str = "", payload: Optional[Dict[str, Any]] = None, label: str = "", **_: Any
    ) -> None:
        self.name = name
        self.payload = payload or {}
        self.label = label


class _FakeUserSession:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value


async def _noop_async(*_a, **_k) -> None:
    return None


@pytest.fixture
def app_module(monkeypatch):
    _FakeMessage.created.clear()
    fake_cl = SimpleNamespace(
        on_chat_start=lambda f: f,
        on_message=lambda f: f,
        on_chat_end=lambda f: f,
        on_settings_update=lambda f: f,
        action_callback=lambda name: (lambda f: f),
        Message=_FakeMessage,
        Action=_FakeAction,
        ChatSettings=lambda inputs: SimpleNamespace(send=_noop_async),
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


class _FakeClient:
    """Client exposing the resume seam VictorClient provides."""

    def __init__(self, sessions=None, history=None, resume_result=...):
        self._sessions = sessions or []
        self._history = history or []
        self._resume_result = {"title": "arithmetic"} if resume_result is ... else resume_result
        self.resumed_id = None
        self.initialized = False

    def list_recent_sessions(self, limit=10):
        return self._sessions[:limit]

    async def initialize(self):
        self.initialized = True

    async def resume_session(self, session_id):
        self.resumed_id = session_id
        return self._resume_result

    async def get_messages(self, limit=None, role=None):
        return self._history


class TestResumeActions:
    def test_picker_built_from_recent_sessions(self, app_module):
        module, _ = app_module
        client = _FakeClient(
            sessions=[
                {"session_id": "s1", "title": "first", "message_count": 4},
                {"session_id": "s2", "title": None, "message_count": 0},
            ]
        )
        actions = module._resume_actions(client)
        assert len(actions) == 2
        assert actions[0].payload == {"session_id": "s1"}
        assert "first" in actions[0].label
        # Missing title falls back gracefully.
        assert "Untitled" in actions[1].label

    def test_picker_skips_sessions_without_id(self, app_module):
        module, _ = app_module
        client = _FakeClient(sessions=[{"title": "no id"}])
        assert module._resume_actions(client) == []

    def test_picker_survives_store_failure(self, app_module):
        module, _ = app_module

        class _Boom:
            def list_recent_sessions(self, limit=10):
                raise RuntimeError("db down")

        assert module._resume_actions(_Boom()) == []


class TestResumeActionCallback:
    async def test_resume_hydrates_and_replays(self, app_module, monkeypatch):
        module, fake_cl = app_module
        from victor.providers.base import Message

        client = _FakeClient(
            history=[Message(role="user", content="2+2?"), Message(role="assistant", content="4")]
        )
        # The callback fetches the session's client via _get_client.
        fake_cl.user_session.set(module._CLIENT_KEY, client)

        action = _FakeAction(name="resume_session", payload={"session_id": "s9"})
        await module._on_resume_session(action)

        assert client.initialized is True
        assert client.resumed_id == "s9"
        # Confirmation + replayed turns are sent.
        contents = " ".join(m.content for m in _FakeMessage.created)
        assert "Resumed" in contents
        assert "2+2?" in contents and "4" in contents

    async def test_resume_missing_session_reports(self, app_module):
        module, fake_cl = app_module
        client = _FakeClient(resume_result=None)
        fake_cl.user_session.set(module._CLIENT_KEY, client)

        await module._on_resume_session(_FakeAction(payload={"session_id": "gone"}))
        contents = " ".join(m.content for m in _FakeMessage.created)
        assert "could not be found" in contents

    async def test_resume_no_payload_is_noop(self, app_module):
        module, _ = app_module
        before = len(_FakeMessage.created)
        await module._on_resume_session(_FakeAction(payload={}))
        assert len(_FakeMessage.created) == before
