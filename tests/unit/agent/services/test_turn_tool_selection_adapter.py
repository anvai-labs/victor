# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Split-context mapping and session ownership for the shared selection entry."""

from types import SimpleNamespace

import pytest

from victor.agent.action_authorizer import ActionIntent
from victor.agent.services.turn_tool_selection_adapter import TurnToolSelectionAdapter


def make_adapter(owner=None):
    chat = SimpleNamespace(settings=None, conversation="conversation", messages=[])
    tool = SimpleNamespace(
        tool_selector=SimpleNamespace(tools="registry"),
        use_semantic_selection=False,
        observed_files={"x"},
    )
    provider = SimpleNamespace(provider="provider")
    executor = SimpleNamespace(
        _chat_context=chat,
        _tool_context=tool,
        _provider_context=provider,
        _resolve_orchestrator=lambda: owner,
    )
    return TurnToolSelectionAdapter(executor), chat, tool


def test_maps_distinct_contexts_and_updates_live_values():
    adapter, chat, tool = make_adapter()
    assert adapter.settings is None
    assert adapter.conversation == "conversation"
    assert adapter.messages is chat.messages
    assert adapter.tool_selector is tool.tool_selector
    assert adapter.tools == "registry"
    assert adapter.observed_files == {"x"}
    assert adapter.provider == "provider"
    chat.messages = ["next"]
    assert adapter.messages == ["next"]
    assert adapter._model_supports_tool_calls() is True


def test_turn_binding_uses_owner_intent_then_clears_invalid_intent():
    owner = SimpleNamespace(_current_intent=ActionIntent.READ_ONLY)
    adapter, _, _ = make_adapter(owner)
    adapter.bind_turn("inspect app.py", None)
    assert adapter._current_intent == ActionIntent.READ_ONLY
    adapter.bind_turn("edit app.py", "write_allowed")
    assert adapter._current_intent == ActionIntent.WRITE_ALLOWED
    assert owner._current_intent == ActionIntent.READ_ONLY
    adapter.bind_turn("inspect app.py", "unknown")
    assert adapter._current_intent is None
    assert adapter._current_user_message == "inspect app.py"


@pytest.mark.parametrize("with_owner", [False, True])
def test_session_state_persists_on_its_owner(with_owner):
    owner = SimpleNamespace() if with_owner else None
    adapter, _, _ = make_adapter(owner)
    tools = ["read"]
    reranker = object()
    adapter._kv_frozen_tools = tools
    adapter._e3tir_reranker = reranker
    adapter.bind_turn("another turn", None)
    assert adapter._kv_frozen_tools is tools
    assert adapter._e3tir_reranker is reranker
    other, _, _ = make_adapter(owner)
    if with_owner:
        assert other._kv_frozen_tools is tools
        assert other._e3tir_reranker is reranker
    else:
        assert getattr(other, "_kv_frozen_tools", None) is None
        assert getattr(other, "_e3tir_reranker", None) is None


def test_standalone_planner_is_cached_and_uses_canonical_filter():
    adapter, _, _ = make_adapter()
    planner = adapter._tool_planner
    assert adapter._tool_planner is planner
    tools = [{"name": "read"}, {"name": "write"}]
    assert planner.filter_tools_by_intent(
        tools, ActionIntent.READ_ONLY, user_message="inspect the implementation"
    ) == [{"name": "read"}]
