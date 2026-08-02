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

"""Tests for the Chainlit-free Victor-event -> render-action mapping.

These run without the optional ``chat-ui`` extra installed (no ``chainlit`` import).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from victor.ui.chat_app.event_mapping import (
    RenderKind,
    history_messages,
    map_event,
    provider_switch_hint,
)


@dataclass
class FakeEvent:
    """Mimics the public surface of a ``VictorClient.stream()`` event."""

    event_type: Any
    content: Optional[str] = None
    tool_name: Optional[str] = None
    result: Any = None
    metadata: Optional[Dict[str, Any]] = None


def test_content_event_maps_to_token() -> None:
    action = map_event(FakeEvent(event_type="content", content="hello"))
    assert action.kind is RenderKind.TOKEN
    assert action.text == "hello"


def test_thinking_event_prefers_content_then_metadata() -> None:
    assert map_event(FakeEvent("thinking", content="step 1")).text == "step 1"
    from_meta = map_event(FakeEvent("thinking", metadata={"reasoning_content": "deep"}))
    assert from_meta.kind is RenderKind.THINKING
    assert from_meta.text == "deep"


def test_tool_call_carries_name_and_arguments() -> None:
    action = map_event(
        FakeEvent("tool_call", tool_name="grep", metadata={"arguments": {"q": "foo"}})
    )
    assert action.kind is RenderKind.TOOL_START
    assert action.tool_name == "grep"
    assert action.metadata["arguments"] == {"q": "foo"}


def test_tool_result_from_dict_result_payload() -> None:
    action = map_event(
        FakeEvent(
            "tool_result",
            tool_name="grep",
            result={"result": "3 matches", "success": True, "arguments": {"q": "foo"}},
        )
    )
    assert action.kind is RenderKind.TOOL_END
    assert action.tool_name == "grep"
    assert action.text == "3 matches"
    assert action.success is True
    assert action.metadata["arguments"] == {"q": "foo"}


def test_tool_result_content_overrides_payload_result() -> None:
    action = map_event(
        FakeEvent("tool_result", tool_name="bash", content="exit 0", result={"success": True})
    )
    assert action.text == "exit 0"


def test_tool_error_marks_unsuccessful() -> None:
    action = map_event(FakeEvent("tool_error", tool_name="bash", content="boom"))
    assert action.kind is RenderKind.TOOL_END
    assert action.success is False
    assert action.text == "boom"


def test_error_event_uses_content_then_metadata() -> None:
    assert map_event(FakeEvent("error", content="bad")).text == "bad"
    from_meta = map_event(FakeEvent("error", metadata={"error": "timeout"}))
    assert from_meta.kind is RenderKind.ERROR
    assert from_meta.text == "timeout"


def test_error_event_falls_back_to_default_message() -> None:
    action = map_event(FakeEvent("error"))
    assert action.kind is RenderKind.ERROR
    assert action.text == "Unknown streaming error"


def test_lifecycle_and_unknown_events_are_ignored() -> None:
    for et in ("stream_start", "stream_end", "totally_new_event"):
        assert map_event(FakeEvent(et)).kind is RenderKind.IGNORE


def test_enum_like_event_type_is_normalized() -> None:
    class _EnumLike:
        value = "content"

        def __str__(self) -> str:  # not used once .value is read
            return "EventType.CONTENT"

    assert map_event(FakeEvent(_EnumLike(), content="x")).kind is RenderKind.TOKEN


def test_dotted_repr_event_type_is_normalized() -> None:
    # event_type given as a bare repr string "EventType.TOOL_CALL"
    action = map_event(FakeEvent("EventType.TOOL_CALL", tool_name="t"))
    assert action.kind is RenderKind.TOOL_START


def test_missing_tool_name_defaults_to_tool() -> None:
    assert map_event(FakeEvent("tool_call")).tool_name == "tool"
    assert map_event(FakeEvent("tool_result", content="x")).tool_name == "tool"


def test_tool_result_carries_telemetry_fields() -> None:
    action = map_event(
        FakeEvent(
            "tool_result",
            tool_name="read",
            result={
                "success": True,
                "elapsed": 0.012,
                "was_pruned": True,
                "follow_up_suggestions": [{"tool": "ls", "suggestion": "Check the dir"}],
                "arguments": {"path": "x.py"},
            },
        )
    )
    assert action.kind is RenderKind.TOOL_END
    assert action.elapsed == 0.012
    assert action.was_pruned is True
    assert action.follow_up_suggestions == [{"tool": "ls", "suggestion": "Check the dir"}]


def test_tool_result_prefers_original_result_over_placeholder() -> None:
    action = map_event(
        FakeEvent(
            "tool_result",
            tool_name="read",
            content="Tool completed successfully. Use /expand …",
            result={
                "success": True,
                "result": "Tool completed successfully. Use /expand …",
                "original_result": "the real file contents",
            },
        )
    )
    assert action.text == "the real file contents"


def test_tool_result_telemetry_from_nested_metadata_fallback() -> None:
    # Un-flattened event: payload nested under metadata["tool_result"].
    action = map_event(
        FakeEvent(
            "tool_result",
            tool_name="grep",
            metadata={"tool_result": {"success": True, "elapsed": 1.5, "result": "ok"}},
        )
    )
    assert action.kind is RenderKind.TOOL_END
    assert action.elapsed == 1.5
    assert action.text == "ok"


def test_tool_result_defaults_when_no_telemetry() -> None:
    action = map_event(FakeEvent("tool_result", tool_name="grep", result={"success": True}))
    assert action.elapsed == 0.0
    assert action.was_pruned is False
    assert action.follow_up_suggestions is None


def test_provider_switch_hint_lists_others_excluding_current() -> None:
    hint = provider_switch_hint("ollama", ["ollama", "anthropic", "openai"])
    assert "anthropic" in hint and "openai" in hint
    assert "ollama" not in hint  # current provider is excluded


def test_provider_switch_hint_empty_when_no_alternative() -> None:
    assert provider_switch_hint("ollama", ["ollama"]) == ""
    assert provider_switch_hint("ollama", []) == ""
    assert provider_switch_hint(None, []) == ""


def test_history_messages_keeps_user_and_assistant_with_content() -> None:
    msgs = [
        FakeEvent("ignored"),  # has no role/content -> skipped
        _RoleMsg("user", "hello"),
        _RoleMsg("assistant", "hi there"),
        _RoleMsg("system", "you are…"),  # internal -> skipped
        _RoleMsg("assistant", "   "),  # empty -> skipped
    ]
    assert history_messages(msgs) == [("You", "hello"), ("Victor", "hi there")]


def test_history_messages_handles_dicts_and_enum_roles() -> None:
    enum_role = type("Role", (), {"value": "assistant"})()
    msgs = [
        {"role": "user", "content": "from a dict"},
        _RoleMsg(enum_role, "from an enum role"),
    ]
    assert history_messages(msgs) == [
        ("You", "from a dict"),
        ("Victor", "from an enum role"),
    ]


def test_history_messages_empty_input() -> None:
    assert history_messages([]) == []
    assert history_messages(None) == []


class _RoleMsg:
    def __init__(self, role, content: str) -> None:
        self.role = role
        self.content = content
