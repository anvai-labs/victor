# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""FEP-0034: the primary agent and benchmark transport use the shared entry.

Real schema construction and concrete context hosts keep these checks sensitive
to curation, registry changes, intent, gates, and persistent session state.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from victor.agent.action_authorizer import ActionIntent
from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime
from victor.agent.services.turn_execution_runtime import TurnExecutor


def make_executor(*, curated=None, owner=True):
    definitions = {
        name: SimpleNamespace(
            name=name, description=f"Use {name}", parameters={"type": "object", "properties": {}}
        )
        for name in ("read", "write", "code_search", "ls")
    }
    registry = SimpleNamespace(
        get=lambda name: definitions.get(name),
        list_tools=lambda **kwargs: list(definitions.values()),
    )
    selector = SimpleNamespace(
        tools=registry,
        _enabled_tools=curated,
        select_tools=AsyncMock(return_value=[definitions["read"], definitions["write"]]),
        prioritize_by_stage=Mock(side_effect=lambda message, tools: tools),
    )
    host = SimpleNamespace(
        settings=None,
        tool_selector=selector,
        use_semantic_selection=True,
        conversation=SimpleNamespace(message_count=lambda: 3),
        messages=[{"role": "user", "content": "inspect app.py"}],
        observed_files=set(),
        tools=registry,
        provider=SimpleNamespace(supports_tools=lambda: True),
        _model_supports_tool_calls=lambda: True,
        _tool_skip_mode=Mock(return_value="tools"),
        tool_registrar=SimpleNamespace(ensure_tools_for_query=Mock()),
        _current_intent=None,
        _current_user_message="",
        _e3tir_reranker=None,
    )
    from victor.agent.services.tool_planning_runtime import ToolPlanner

    host._tool_planner = ToolPlanner(host.tool_registrar, host.settings)
    executor = TurnExecutor(host, host, host, None)
    executor._orchestrator = host if owner else None
    return executor, host, definitions


@pytest.fixture(autouse=True)
def selection_env(monkeypatch):
    monkeypatch.setenv("VICTOR_TOOL_SELECTION", "1")


@pytest.mark.asyncio
async def test_executor_delegates_to_shared_entry(monkeypatch):
    executor, host, _ = make_executor()
    select = AsyncMock(return_value=["sentinel"])
    monkeypatch.setattr(ToolSelectionRuntime, "select_tools_for_turn", select)
    assert await executor._select_tools_for_turn("inspect app.py", "read_only") == ["sentinel"]
    select.assert_awaited_once_with("inspect app.py", goals=None)
    adapter = executor._turn_tool_selection_adapter
    assert adapter._current_intent == ActionIntent.READ_ONLY
    assert adapter._current_user_message == "inspect app.py"
    assert adapter.tool_selector is host.tool_selector
    host.tool_selector.select_tools.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("pruning", ["0", "1"])
@pytest.mark.parametrize("curated", [None, {"read", "write"}])
async def test_shared_entry_supply_and_trace_parity(monkeypatch, pruning, curated):
    monkeypatch.setenv("VICTOR_TOOL_SELECTION", pruning)
    executor, host, _ = make_executor(curated=curated)
    emitted = []
    monkeypatch.setattr(
        "victor.agent.services.tool_selection_runtime._emit_tool_supply_trace", emitted.append
    )
    actual = await executor._select_tools_for_turn("inspect app.py", "read_only")
    host._current_user_message = "inspect app.py"
    host._current_intent = ActionIntent.READ_ONLY
    expected = await ToolSelectionRuntime(host).select_tools_for_turn("inspect app.py", goals=None)
    assert actual == expected
    assert emitted[0].to_payload() == emitted[1].to_payload()
    assert len(emitted) == 2
    assert emitted[0].dispatched
    assert host.tool_registrar.ensure_tools_for_query.call_count == (0 if curated else 2)
    if curated:
        assert {t.name for t in actual} == curated
        host.tool_selector.select_tools.assert_not_called()
        host.tool_selector.prioritize_by_stage.assert_not_called()
    elif pruning == "0":
        assert {t.name for t in actual} == {"read", "write", "code_search", "ls"}
        host.tool_selector.select_tools.assert_not_called()
    else:
        assert [t.name for t in actual] == ["read"]
        assert [s.name for s in emitted[0].stages] == [
            "stage_priority",
            "intent_filter",
            "ensure_write",
            "explicit_db",
            "e3tir_rerank",
            "kv_strategy",
            "kv_sort",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pruning", ["0", "1"])
@pytest.mark.parametrize("curated,expected", [({"gh", "read"}, ["read"]), ({"missing"}, [])])
async def test_curated_supply_is_authoritative_before_hydration(
    monkeypatch, pruning, curated, expected
):
    monkeypatch.setenv("VICTOR_TOOL_SELECTION", pruning)
    executor, host, definitions = make_executor(curated=curated)
    emitted = []
    monkeypatch.setattr(
        "victor.agent.services.tool_selection_runtime._emit_tool_supply_trace", emitted.append
    )

    def hydrate(message):
        definitions["gh"] = SimpleNamespace(name="gh", description="GitHub", parameters={})

    host.tool_registrar.ensure_tools_for_query.side_effect = hydrate
    first = await executor._select_tools_for_turn("open a pull request on github")
    second = await executor._select_tools_for_turn("inspect the pull request")
    assert [t.name for t in first] == expected
    assert second == first
    if expected:
        assert second is first
    host.tool_registrar.ensure_tools_for_query.assert_not_called()
    host.tool_selector.select_tools.assert_not_called()
    assert len(emitted) == 2
    assert list(emitted[0].dispatched) == expected
    assert list(emitted[0].candidates) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate,pruning,expected",
    [
        ("skip", "0", None),
        ("read_core", "1", {"read", "code_search", "ls"}),
        ("read_core", "0", {"read", "write", "code_search", "ls"}),
    ],
)
async def test_necessity_and_pruning_gates(monkeypatch, gate, pruning, expected):
    monkeypatch.setenv("VICTOR_TOOL_SELECTION", pruning)
    executor, host, _ = make_executor()
    host._tool_skip_mode.return_value = gate
    actual = await executor._select_tools_for_turn("explain this concept")
    assert (None if actual is None else {t.name for t in actual}) == expected
    host.tool_selector.select_tools.assert_not_called()
    host.tool_registrar.ensure_tools_for_query.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_support,model_support", [(False, True), (True, False)])
async def test_capability_gate(provider_support, model_support):
    executor, host, _ = make_executor()
    host.provider.supports_tools = lambda: provider_support
    host._model_supports_tool_calls = lambda: model_support
    assert await executor._select_tools_for_turn("inspect app.py") is None
    host.tool_selector.select_tools.assert_not_called()


@pytest.mark.asyncio
async def test_standalone_context_uses_canonical_intent_filter():
    executor, host, _ = make_executor(owner=False)
    del host._tool_planner
    result = await executor._select_tools_for_turn("inspect the app.py implementation", "read_only")
    assert [tool.name for tool in result] == ["read"]
    assert executor._turn_tool_selection_adapter._fallback_planner is not None


@pytest.mark.asyncio
async def test_service_pruning_override_applies_to_executor(monkeypatch):
    monkeypatch.setenv("VICTOR_TOOL_SELECTION", "0")
    executor, host, _ = make_executor()
    host._tool_service = SimpleNamespace(_config=SimpleNamespace(tool_selection_enabled=True))
    result = await executor._select_tools_for_turn("inspect the app.py implementation", "read_only")
    assert [tool.name for tool in result] == ["read"]
    host.tool_selector.select_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_state_survives_turns_and_is_shared_with_owner():
    executor, host, definitions = make_executor()
    host._kv_optimization_enabled = True
    host.settings = SimpleNamespace(kv_tool_strategy="session_stable")
    first = await executor._select_tools_for_turn("inspect app.py", "read_only")
    adapter = executor._turn_tool_selection_adapter
    assert host._kv_frozen_tools == first
    host.tool_selector.select_tools.return_value = [definitions["write"], definitions["read"]]
    second = await executor._select_tools_for_turn("edit app.py", "write_allowed")
    assert executor._turn_tool_selection_adapter is adapter
    assert [tool.name for tool in second] == ["read"]
    assert adapter._current_user_message == "edit app.py"
    assert adapter._current_intent == ActionIntent.WRITE_ALLOWED


@pytest.mark.asyncio
async def test_overlapping_turns_keep_their_own_intents():
    executor, host, definitions = make_executor()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def select(message, **kwargs):
        if message == "inspect app.py":
            entered.set()
            await release.wait()
        return [definitions["read"], definitions["write"]]

    host.tool_selector.select_tools.side_effect = select
    first = asyncio.create_task(executor._select_tools_for_turn("inspect app.py", "read_only"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    second = asyncio.create_task(executor._select_tools_for_turn("edit app.py", "write_allowed"))
    try:
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        read_tools, write_tools = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
        assert [tool.name for tool in read_tools] == ["read"]
        assert {tool.name for tool in write_tools} == {"read", "write"}
    finally:
        release.set()
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_holder", [False, True])
async def test_selection_lock_recovers_after_cancellation(cancel_holder):
    executor, host, definitions = make_executor()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def select(message, **kwargs):
        if message == "inspect app.py":
            entered.set()
            await release.wait()
        return [definitions["read"], definitions["write"]]

    host.tool_selector.select_tools.side_effect = select
    holder = asyncio.create_task(executor._select_tools_for_turn("inspect app.py", "read_only"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    waiter = asyncio.create_task(executor._select_tools_for_turn("edit app.py", "write_allowed"))
    try:
        await asyncio.sleep(0)
        cancelled = holder if cancel_holder else waiter
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        survivor = waiter if cancel_holder else holder
        tools = await asyncio.wait_for(survivor, timeout=2)
        expected = {"read", "write"} if cancel_holder else {"read"}
        assert {tool.name for tool in tools} == expected
        follow_up = await asyncio.wait_for(
            executor._select_tools_for_turn("inspect the module", "read_only"), timeout=2
        )
        assert [tool.name for tool in follow_up] == ["read"]
    finally:
        release.set()
        for task in (holder, waiter):
            task.cancel()
        await asyncio.gather(holder, waiter, return_exceptions=True)
