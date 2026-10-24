"""Context-assembly extraction (self-analysis round 2, W3).

The orchestrator methods are now facades over
victor.agent.conversation.context_assembly — same names, same patch
targets, implementation moved verbatim. These tests pin the moved logic
against duck-typed orchestrator stand-ins (no real orchestrator needed).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from victor.agent.conversation.context_assembly import (
    assembled_messages_for_call,
    conversation_message_count,
    session_tools,
)
from victor.core.provider_messages import Message


class _FakePromptBuilder:
    def ensure_system_prompt_current(self):
        pass


def _orchestrator_with_messages(messages):
    return SimpleNamespace(
        messages=messages,
        _get_prompt_builder_runtime=lambda: _FakePromptBuilder(),
        _get_max_context_chars=lambda: 100_000,
        _context_assembler=None,
        _kv_optimization_enabled=False,
        _prompt_pipeline=None,
    )


def test_conversation_message_count_duck_types_history_backends():
    assert conversation_message_count(SimpleNamespace(conversation=None)) == 0
    assert (
        conversation_message_count(
            SimpleNamespace(conversation=SimpleNamespace(messages=[1, 2, 3]))
        )
        == 3
    )
    # No messages attr — fall through to a callable message_count().
    backend = SimpleNamespace(message_count=lambda: 7)
    assert conversation_message_count(SimpleNamespace(conversation=backend)) == 7
    broken = SimpleNamespace(message_count=lambda: None)  # int(None) raises TypeError
    assert conversation_message_count(SimpleNamespace(conversation=broken)) == 0


def test_assembled_messages_normalizes_leaked_dicts_and_strings():
    orchestrator = _orchestrator_with_messages(
        [
            Message(role="user", content="hello"),
            {"role": "assistant", "content": "hi", "bogus": "dropped"},
            "bare string",
            42,  # dropped entirely
        ]
    )
    orchestrator._kv_prefix_fingerprint = lambda: "abc123"
    result = assembled_messages_for_call(orchestrator)
    assert [m.role for m in result] == ["user", "assistant", "assistant"]
    assert result[1].content == "hi"
    assert not hasattr(result[1], "bogus") or result[1].model_dump().get("bogus") is None
    assert result[2].content == "bare string"
    assert len(result) == 3  # the 42 vanished


def test_assembled_messages_uses_assembler_budget_when_present():
    assembled = [Message(role="system", content="s"), Message(role="user", content="q")]
    assembler = SimpleNamespace(assemble=lambda msgs, max_chars, current_query=None: assembled)
    orchestrator = _orchestrator_with_messages([Message(role="user", content="long history")])
    orchestrator._context_assembler = assembler
    result = assembled_messages_for_call(orchestrator, "q")
    assert [m.content for m in result] == ["s", "q"]


def test_assembled_messages_records_prefix_tracker_when_kv_enabled():
    recorded = {}

    class _Tracker:
        def record(self, messages, tools=None):
            recorded["messages"] = list(messages)
            recorded["tools"] = tools

    orchestrator = _orchestrator_with_messages([Message(role="user", content="q")])
    orchestrator._kv_optimization_enabled = True
    orchestrator._kv_prefix_fingerprint = lambda: "abc123"
    orchestrator._last_sorted_tool_names = ["t1"]
    with mock.patch("victor.agent.kv_prefix_observability.PrefixCallTracker", _Tracker):
        result = assembled_messages_for_call(orchestrator)
    assert recorded["tools"] == ["t1"]
    assert recorded["messages"] == result


def test_session_tools_locking_and_cache_disabled_path():
    tool = SimpleNamespace(name="search")
    orchestrator = SimpleNamespace(
        _cache_optimization_enabled=False,
    )
    assert session_tools(orchestrator) is None

    def _list_tools(only_enabled=True):
        return [tool]

    enabled = SimpleNamespace(
        _cache_optimization_enabled=True,
        _session_tools=None,
        get_enabled_tools=lambda: {"search"},
        tools=SimpleNamespace(list_tools=_list_tools),
    )
    locked = session_tools(enabled)
    assert [t.name for t in locked] == ["search"]
    assert enabled._session_tools is locked  # cached for the session
    assert session_tools(enabled) is locked
