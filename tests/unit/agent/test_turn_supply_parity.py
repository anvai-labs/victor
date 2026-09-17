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

"""Stage C parity: the agentic-loop transport must demand-hydrate.

``TurnExecutionRuntime._select_tools_for_turn`` is the selection transport
headless runs and benchmarks exercise. Before the Stage C parity fix it ran
its own gate order with no demand hydration, so mention-wired tools (graph,
gh) measured differently than the chat transports served them.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.turn_execution_runtime import TurnExecutor


def _make_runtime(user_message: str = "open a pull request on github"):
    chat_context = MagicMock()
    chat_context.messages = []
    chat_context.conversation.message_count.return_value = 0
    tool_context = MagicMock()
    tool_context.tool_selector.select_tools = AsyncMock(return_value=[])
    tool_context.tool_selector.prioritize_by_stage.side_effect = lambda _m, tools: tools
    runtime = TurnExecutor(
        chat_context=chat_context,
        tool_context=tool_context,
        provider_context=MagicMock(),
        execution_provider=MagicMock(),
    )
    return runtime, tool_context


@pytest.mark.asyncio
async def test_agentic_transport_hydrates_demand_tools(monkeypatch: pytest.MonkeyPatch):
    """Hydration runs before selection with the turn's user message."""
    runtime, tool_context = _make_runtime()
    hydrated: list[tuple[Any, str]] = []
    monkeypatch.setattr(
        "victor.agent.services.tool_selection_runtime.hydrate_demand_tools",
        lambda host, text: hydrated.append((host, text)),
    )

    await runtime._select_tools_for_turn("open a pull request on github")

    assert len(hydrated) == 1
    host, text = hydrated[0]
    assert text == "open a pull request on github"
    # The registrar handle resolves through the orchestrator facade.
    assert getattr(host, "tool_registrar", None) is not None


@pytest.mark.asyncio
async def test_hydration_runs_before_pruning_gate(monkeypatch: pytest.MonkeyPatch):
    """Hydration happens even on pruning-disabled turns (the default)."""
    runtime, _ = _make_runtime()
    seen = []
    monkeypatch.setattr(
        "victor.agent.services.tool_selection_runtime.hydrate_demand_tools",
        lambda host, text: seen.append(text),
    )
    await runtime._select_tools_for_turn("summarize this repository")
    assert seen == ["summarize this repository"]


def test_agentic_transport_emits_tool_supply_trace(monkeypatch: pytest.MonkeyPatch):
    """Stage 8 parity: the agentic transport emits the same supply trace the
    chat transports do, so benchmark runs are queryable like served sessions."""

    emitted = []
    monkeypatch.setattr(
        "victor.agent.services.tool_selection_runtime._emit_tool_supply_trace",
        lambda trace: emitted.append(trace),
    )
    runtime, tool_context = _make_runtime()
    tool_context.tool_selector.select_tools = AsyncMock(return_value=[])
    tool_context.tool_selector.prioritize_by_stage.side_effect = lambda _m, tools: tools

    asyncio.run(runtime._select_tools_for_turn("fix the bug"))

    assert len(emitted) == 1
    assert tuple(emitted[0].dispatched) == ()
    assert tuple(emitted[0].candidates) == ()


def test_agentic_transport_emits_skipped_trace_on_stable_definitions(
    monkeypatch: pytest.MonkeyPatch,
):
    """Pruning-off turns that resolve stable definitions record a skip (the
    supply was the byte-stable definition set, selected by nothing)."""

    emitted = []
    monkeypatch.setattr(
        "victor.agent.services.tool_selection_runtime._emit_tool_supply_trace",
        lambda trace: emitted.append(trace),
    )
    monkeypatch.setattr(
        "victor.agent.tool_selection.stable_definitions.stable_all_definitions",
        lambda _selector: ["read", "edit"],
    )
    # Force the pruning-enabled branch so the stable-definitions skip runs.
    monkeypatch.setattr(
        "victor.agent.services.turn_execution_runtime.is_tool_selection_enabled",
        lambda _settings: False,
    )
    runtime, _ = _make_runtime()

    asyncio.run(runtime._select_tools_for_turn("fix the bug"))

    assert len(emitted) == 1
    assert emitted[0].skipped is True
    assert emitted[0].skip_reason == "pruning_disabled_stable_definitions"


@pytest.mark.parametrize("curated_name,expected", [("read", ("read",)), ("missing", ())])
def test_agentic_transport_curated_set_emits_finalize(
    monkeypatch: pytest.MonkeyPatch, curated_name, expected
):
    """A curated toolset reaches the LLM unchanged AND still emits its
    finalized supply trace (the early-return path must not skip telemetry)."""
    emitted = []
    monkeypatch.setattr(
        "victor.agent.services.tool_selection_runtime._emit_tool_supply_trace",
        lambda trace: emitted.append(trace),
    )
    runtime, tool_context = _make_runtime()
    definitions = {"read": SimpleNamespace(name="read", description="Read a file", parameters={})}
    tool_context.tool_selector = SimpleNamespace(
        _enabled_tools={curated_name}, tools=SimpleNamespace(get=definitions.get)
    )

    asyncio.run(runtime._select_tools_for_turn("fix the bug"))

    assert len(emitted) == 1
    assert tuple(emitted[0].dispatched) == expected
    assert tuple(emitted[0].candidates) == expected
