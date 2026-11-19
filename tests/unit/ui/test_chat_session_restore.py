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

"""Regression: ``--sessionid`` resume must not die on a stale module import.

``_restore_session_into_agent`` imports lazily (function-level), so the
collect-check cannot see the import graph — the module path itself must be
exercised. The historical bug imported the removed
``victor.agent.conversation_state`` module, which crashed every
``victor chat --sessionid <id>`` invocation.
"""

from __future__ import annotations

from typing import Any

import pytest

from victor.ui.commands.chat import _restore_session_into_agent


class _FakeConsole:
    def print(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - trivial
        pass


def _session_data(with_conversation_state: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "metadata": {"title": "Fix the flux capacitor", "message_count": 2},
        "conversation": {
            "system_prompt": "You are Victor.",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        },
    }
    if with_conversation_state:
        data["conversation_state"] = {"state": {"stage": "INITIAL"}}
    return data


@pytest.fixture()
def _fake_store(monkeypatch: Any) -> None:
    from victor.agent.conversation import store as store_module

    class _FakeStore:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def load_session(self, session_id: str) -> dict[str, Any] | None:
            if session_id == "s_missing":
                return None
            return _session_data(with_conversation_state=(session_id == "s_with_state"))

    monkeypatch.setattr(store_module, "ConversationStore", _FakeStore)


def test_restore_session_happy_path(_fake_store: None) -> None:
    agent: dict[str, Any] = {}

    class _Agent:
        pass

    agent_obj = _Agent()
    message = _restore_session_into_agent(agent_obj, "s_abc123", _FakeConsole())  # type: ignore[arg-type]

    assert message is not None
    assert "Resumed session" in message
    assert "Fix the flux capacitor" in message
    assert agent_obj.active_session_id == "s_abc123"
    assert agent_obj.conversation is not None
    assert len(agent_obj.conversation._messages) == 2


def test_restore_session_restores_conversation_state(_fake_store: None) -> None:
    agent_obj: Any = type("_Agent", (), {})()

    message = _restore_session_into_agent(agent_obj, "s_with_state", _FakeConsole())

    assert message is not None
    assert agent_obj.conversation_state is not None


def test_restore_session_unknown_id_raises_value_error(_fake_store: None) -> None:
    agent_obj: Any = type("_Agent", (), {})()

    with pytest.raises(ValueError, match="Session not found"):
        _restore_session_into_agent(agent_obj, "s_missing", _FakeConsole())


def test_restore_session_noop_without_id(_fake_store: None) -> None:
    agent_obj: Any = type("_Agent", (), {})()

    assert _restore_session_into_agent(agent_obj, None, _FakeConsole()) is None
