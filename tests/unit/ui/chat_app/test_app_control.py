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

"""PR3 control/resilience: stop-a-turn, close-guard, and in-chat error recovery.

The Chainlit dependency is faked via ``sys.modules`` so app.py imports without the chat-ui
extra (same pattern as test_app_rendering.py), extended here with Action/action_callback.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


class _FakeMessage:
    created: List["_FakeMessage"] = []

    def __init__(self, content: str = "", actions: Optional[list] = None) -> None:
        self.content = content
        self.actions = actions or []
        self.removed = False
        _FakeMessage.created.append(self)

    async def send(self) -> "_FakeMessage":
        return self

    async def stream_token(self, token: str) -> None:
        self.content += token

    async def update(self) -> None:
        return None

    async def remove(self) -> None:
        self.removed = True


class _FakeStep:
    def __init__(self, name: Optional[str] = None, type: Optional[str] = None) -> None:
        self.name = name
        self.type = type
        self.input: Any = None
        self.output: Any = None
        self.is_error = False

    async def __aenter__(self) -> "_FakeStep":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeAction:
    def __init__(
        self,
        name: str = "",
        payload: Optional[Dict[str, Any]] = None,
        label: str = "",
        **_: Any,
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


async def _noop_async() -> None:
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
        Step=_FakeStep,
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


class _BlockingClient:
    """Streams nothing and blocks until cancelled — to exercise stop/close."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = False
        self.provider_name = "ollama"

    async def stream(self, content: str):
        self.entered.set()
        await asyncio.sleep(3600)  # cancellable block
        yield  # pragma: no cover

    async def close(self) -> None:
        self.closed = True

    def get_last_turn_cost(self) -> Dict[str, Any]:
        return {}

    def get_available_providers(self) -> List[str]:
        return ["ollama", "anthropic"]


class _RecordingClient:
    def __init__(self) -> None:
        self.last_content: Optional[str] = None
        self.provider_name = "ollama"

    async def stream(self, content: str):
        self.last_content = content
        return
        yield  # pragma: no cover  (make this an async generator)

    async def close(self) -> None:
        return None

    def get_last_turn_cost(self) -> Dict[str, Any]:
        return {}

    def get_available_providers(self) -> List[str]:
        return ["ollama"]


class _FailingClient:
    def __init__(self) -> None:
        self.provider_name = "ollama"

    async def stream(self, content: str):
        raise RuntimeError("kaboom")
        yield  # pragma: no cover

    async def close(self) -> None:
        return None

    def get_last_turn_cost(self) -> Dict[str, Any]:
        return {}

    def get_available_providers(self) -> List[str]:
        return ["ollama", "anthropic"]


@pytest.mark.parametrize("blocked_phase", ["provider", "render"])
async def test_native_stop_cancels_and_drains_the_turn(app_module, monkeypatch, blocked_phase):
    module, fake_cl = app_module
    client = _BlockingClient()
    entered = asyncio.Event()
    owners = []
    finalized = []

    async def generate():
        owners.append(asyncio.current_task())
        try:
            if blocked_phase == "provider":
                entered.set()
                await asyncio.sleep(3600)
            yield SimpleNamespace(event_type="content", content="text", metadata={})
        finally:
            finalized.append(asyncio.current_task())

    async def render(self, token):
        entered.set()
        await asyncio.sleep(3600)

    # Retain the generator so GC cannot hide a missing close in the UI owner.
    stream = generate()
    monkeypatch.setattr(client, "stream", lambda content: stream)
    monkeypatch.setattr(_FakeMessage, "stream_token", render)
    fake_cl.user_session.set(module._CLIENT_KEY, client)

    start = asyncio.create_task(module._start_turn("hi"))
    await asyncio.wait_for(entered.wait(), 1)
    # Chainlit's native stop cancels the on_message task, which owns _start_turn.
    start.cancel()
    await asyncio.wait_for(start, 1)

    assert any("stopped" in m.content.lower() for m in _FakeMessage.created)
    assert fake_cl.user_session.get(module._TASK_KEY) is None  # cleared
    assert not any(a.name == "stop_turn" for m in _FakeMessage.created for a in m.actions)
    assert finalized == owners


async def test_second_turn_cannot_replace_active_turn(app_module):
    module, fake_cl = app_module
    client = _BlockingClient()
    fake_cl.user_session.set(module._CLIENT_KEY, client)
    start = asyncio.create_task(module._start_turn("first"))
    await asyncio.wait_for(client.entered.wait(), 1)
    active = fake_cl.user_session.get(module._TASK_KEY)
    try:
        await asyncio.wait_for(module._start_turn("second"), 0.2)
        assert fake_cl.user_session.get(module._TASK_KEY) is active
        assert not active.done()
        assert any("already running" in m.content for m in _FakeMessage.created)
    finally:
        start.cancel()
        await start


async def test_on_chat_end_drains_active_turn_before_close(app_module):
    module, fake_cl = app_module
    client = _BlockingClient()
    fake_cl.user_session.set(module._CLIENT_KEY, client)
    inner = asyncio.create_task(asyncio.sleep(3600))
    fake_cl.user_session.set(module._TASK_KEY, inner)

    await module.on_chat_end()

    assert inner.cancelled() or inner.done()  # turn drained
    assert client.closed  # client released after the drain
    assert fake_cl.user_session.get(module._TASK_KEY) is None
    assert fake_cl.user_session.get(module._CLIENT_KEY) is None


@pytest.mark.parametrize("operation", ["turn", "settings", "resume"])
@pytest.mark.parametrize("queued", [False, True])
async def test_disconnected_session_rejects_queued_work(app_module, operation, queued):
    module, fake_cl = app_module
    client = _BlockingClient()
    fake_cl.user_session.set(module._CLIENT_KEY, client)

    async def work():
        if operation == "turn":
            await module._start_turn("late message")
        elif operation == "settings":
            await module.on_settings_update({"profile": "different"})
        else:
            await module._on_resume_session(_FakeAction(payload={"session_id": "old"}))

    if queued:
        # A settings transition owns the lock; work queues before disconnect.
        lock = module._session_lock()
        await lock.acquire()
        pending = asyncio.create_task(work())
        await asyncio.sleep(0)
        disconnect = asyncio.create_task(module.on_chat_end())
        await asyncio.sleep(0)
        lock.release()
        results = await asyncio.wait_for(
            asyncio.gather(pending, disconnect, return_exceptions=True), 1
        )
        assert isinstance(results[0], RuntimeError)
        assert "session is closed" in str(results[0])
        assert results[1] is None
    else:
        await module.on_chat_end()
        with pytest.raises(RuntimeError, match="session is closed"):
            await work()
    assert not client.entered.is_set()


async def test_retry_action_reruns_with_original_message(app_module):
    module, fake_cl = app_module
    client = _RecordingClient()
    fake_cl.user_session.set(module._CLIENT_KEY, client)

    await module._on_retry_turn(_FakeAction("retry_turn", payload={"content": "redo this"}))

    assert client.last_content == "redo this"


async def test_error_recovery_renders_friendly_message_retry_and_hint(app_module):
    module, fake_cl = app_module
    client = _FailingClient()

    await module._run_turn(client, "boom")

    err = next(m for m in _FakeMessage.created if "⚠️" in m.content)
    # Retry action carries the original message for re-run.
    assert any(getattr(a, "name", "") == "retry_turn" for a in err.actions)
    assert err.actions[0].payload.get("content") == "boom"
    # Provider-switch hint suggests the other available provider.
    assert "anthropic" in err.content
