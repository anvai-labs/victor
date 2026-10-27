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

"""UX-02 qualification items: progress delivery, terminal outcomes, mode controls.

``chainlit`` is faked via ``sys.modules`` (sibling-suite pattern). These pin the
three UI facts the audit called out: plan progress reaches the browser as ONE
updating status line, a failed terminal outcome renders a durable summary card,
and the settings panel now owns explicit mode/reasoning/planning controls.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from victor.ui.chat_app.event_mapping import RenderKind, map_event


class _FakeMessage:
    sent: List["_FakeMessage"] = []

    def __init__(self, content: str = "", actions: Optional[list] = None, elements=None) -> None:
        self.content = content
        self.actions = actions
        self.elements = elements or []
        self.removed = False
        self.updates = 0
        _FakeMessage.sent.append(self)

    async def send(self) -> "_FakeMessage":
        return self

    async def stream_token(self, token: str) -> None:
        self.content += token

    async def update(self) -> None:
        self.updates += 1

    async def remove(self) -> None:
        self.removed = True


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


class _TurnClient:
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


SETTINGS_PANELS: list[list] = []


@pytest.fixture
def app_module(monkeypatch):
    _FakeMessage.sent.clear()
    _FakeStep.created.clear()
    SETTINGS_PANELS.clear()

    class _Panel:
        def __init__(self, inputs):
            self.inputs = inputs

        async def send(self):
            SETTINGS_PANELS.append(self.inputs)
            return self

    fake_cl = SimpleNamespace(
        on_chat_start=lambda f: f,
        on_message=lambda f: f,
        on_chat_end=lambda f: f,
        on_settings_update=lambda f: f,
        action_callback=lambda name: (lambda f: f),
        Message=_FakeMessage,
        Step=_FakeStep,
        Image=lambda **k: SimpleNamespace(**k),
        Action=lambda **kwargs: SimpleNamespace(**kwargs),
        ChatSettings=_Panel,
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


class TestProgressAndOutcomeMapping:
    def test_progress_and_milestone_map_to_one_kind(self):
        progress = map_event(
            SimpleNamespace(event_type="progress", content="step 2 of 7", metadata={})
        )
        milestone = map_event(
            SimpleNamespace(event_type="milestone", content="plan approved", metadata={})
        )
        assert progress.kind is RenderKind.PROGRESS and progress.text == "step 2 of 7"
        assert milestone.kind is RenderKind.PROGRESS
        assert milestone.metadata["milestone"] is True
        assert progress.metadata["milestone"] is False

    def test_progress_falls_back_to_metadata_message(self):
        action = map_event(
            SimpleNamespace(event_type="progress", content=None, metadata={"message": "indexing"})
        )
        assert action.kind is RenderKind.PROGRESS and action.text == "indexing"

    def test_failed_stream_end_maps_to_outcome_clean_end_ignored(self):
        failed = map_event(
            SimpleNamespace(
                event_type="stream_end", content=None, metadata={"error": "upstream 502"}
            )
        )
        loop_failed = map_event(
            SimpleNamespace(
                event_type="stream_end", content=None, metadata={"agentic_loop_error": "exhausted"}
            )
        )
        no_error_failed = map_event(
            SimpleNamespace(event_type="stream_end", content=None, metadata={}, success=False)
        )
        clean = map_event(SimpleNamespace(event_type="stream_end", content=None, metadata={}))
        assert failed.kind is RenderKind.OUTCOME and failed.success is False
        assert failed.text == "upstream 502"
        assert loop_failed.kind is RenderKind.OUTCOME and loop_failed.text == "exhausted"
        assert no_error_failed.kind is RenderKind.OUTCOME
        assert clean.kind is RenderKind.IGNORE

    def test_real_factory_events_flow_to_progress(self):
        """Drive the REAL producers: factory event -> _to_stream_event -> map_event.

        A prior round mapped fabricated shapes while real milestone events carry
        their text on ``event.milestone`` and content stays empty — the UI would
        have shown nothing (UX-02's original complaint).
        """
        from victor.framework.client import _to_stream_event
        from victor.framework.events import milestone_event, progress_event

        milestone = map_event(_to_stream_event(milestone_event("step 2 of 7", metadata={})))
        progress = map_event(_to_stream_event(progress_event(0.5)))
        assert milestone.kind is RenderKind.PROGRESS and milestone.text == "step 2 of 7"
        assert milestone.metadata["milestone"] is True
        assert progress.kind is RenderKind.PROGRESS and progress.text == "50% complete"
        assert progress.metadata["milestone"] is False


class TestTurnRendering:
    async def test_progress_updates_one_status_line_and_clears_it(self, app_module):
        module, fake_cl = app_module
        client = _TurnClient(
            [
                SimpleNamespace(event_type="progress", content="step 1", metadata={}),
                SimpleNamespace(event_type="progress", content="step 2", metadata={}),
                SimpleNamespace(event_type="milestone", content="plan approved", metadata={}),
                SimpleNamespace(event_type="content", content="done", metadata={}),
            ]
        )
        fake_cl.user_session.set(module._CLIENT_KEY, client)

        await module._run_turn(client, "go")

        progress_msgs = [m for m in _FakeMessage.sent if m.content.startswith(("⏳", "🎯"))]
        assert (
            len(progress_msgs) == 1
        ), "progress must be ONE updating message, not a stream of them"
        msg = progress_msgs[0]
        # The single line carried step 1 -> step 2 -> the milestone (content mutates).
        assert "plan approved" in msg.content and msg.updates >= 2
        assert msg.removed, "the status line is removed when the turn ends"

    async def test_failed_terminal_outcome_renders_a_durable_card(self, app_module):
        module, fake_cl = app_module
        client = _TurnClient(
            [
                SimpleNamespace(event_type="content", content="partial work", metadata={}),
                SimpleNamespace(
                    event_type="stream_end", content=None, metadata={"error": "upstream 502"}
                ),
            ]
        )
        fake_cl.user_session.set(module._CLIENT_KEY, client)

        await module._run_turn(client, "do it")

        cards = [m for m in _FakeMessage.sent if "Turn ended" in m.content]
        assert len(cards) == 1
        assert "upstream 502" in cards[0].content
        assert not cards[0].removed, "terminal outcomes are durable, not transient status"


class TestModeControls:
    def test_session_config_accepts_mode_reasoning_planning(self, app_module, monkeypatch):
        module, _ = app_module
        monkeypatch.delenv("VICTOR_UI_REMOTE_URL", raising=False)
        config = module._session_config("default", mode="plan", show_reasoning=True, planning=True)
        assert config.mode == "plan"
        assert config.show_reasoning is True
        assert config.planning_enabled is True

    def test_session_config_defaults_preserve_unset_behavior(self, app_module, monkeypatch):
        module, _ = app_module
        monkeypatch.delenv("VICTOR_UI_REMOTE_URL", raising=False)
        config = module._session_config("default")
        assert config.mode is None
        assert config.show_reasoning is False
        assert config.planning_enabled is None

    def test_explicit_build_round_trips_equal_to_default(self, app_module, monkeypatch):
        """Panel 'build' must equal the unset default or unchanged-confirm rebuilds."""
        module, _ = app_module
        monkeypatch.delenv("VICTOR_UI_REMOTE_URL", raising=False)
        default_config = module._session_config("default")
        build_config = module._session_config("default", mode="build")
        assert build_config == default_config

    def test_bogus_mode_is_rejected_loudly(self, app_module, monkeypatch):
        module, _ = app_module
        monkeypatch.delenv("VICTOR_UI_REMOTE_URL", raising=False)
        with pytest.raises(ValueError, match="Unknown mode"):
            module._session_config("default", mode="vibes")

    async def test_settings_panel_carries_mode_controls(self, app_module):
        module, fake_cl = app_module
        client = _TurnClient([])
        fake_cl.user_session.set(module._CLIENT_KEY, client)
        fake_cl.user_session.set(module._CONFIG_KEY, module._session_config("default"))

        await module._send_settings(client)

        assert SETTINGS_PANELS, "the settings panel must be sent"
        ids = {getattr(w, "id", None) for w in SETTINGS_PANELS[-1]}
        assert {"mode", "show_reasoning", "planning", "tool_approval"} <= ids
