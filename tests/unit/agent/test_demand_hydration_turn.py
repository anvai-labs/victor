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

"""Regression: the live per-turn tool-supply path must demand-hydrate.

``ToolSelectionRuntime.select_tools_for_turn`` is the single live choke point
(turn executor + chat stream executor both route through it). Demand tools
(graph, gh) are wired register-on-mention via
``ToolRegistrar.ensure_tools_for_query``, but that hook was only reachable
from ``ToolService.select_tools`` — a method with zero callers — so
mention-wired tools never appeared in chat sessions (the live probe that
caught it: a session asked to use ``gh`` reported "No dedicated gh tool
exists in this session's toolset" and fell back to shell).
"""

from typing import Optional
from unittest.mock import MagicMock

import pytest

from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime


def _make_runtime(message: Optional[str] = "open a pull request on github"):
    """Runtime mock that short-circuits at the capability gate (no LLM needed).

    The capability gate sits after demand hydration in the pipeline, so the
    early ``None`` return proves hydration ran first without exercising any
    selection machinery. The registrar handle mirrors the real assembly
    (``component_assembler`` sets ``orchestrator.tool_registrar``; ``tools``
    is the plain registry).
    """
    runtime = MagicMock()
    runtime._current_user_message = message
    runtime.provider.supports_tools.return_value = False
    runtime._model_supports_tool_calls.return_value = False
    runtime.tool_selector._enabled_tools = None  # no curated bypass
    runtime.tools = None
    runtime.tool_registrar.ensure_tools_for_query.return_value = 1
    return runtime


@pytest.mark.asyncio
async def test_hydration_runs_before_supply_decision():
    runtime = _make_runtime()
    result = await ToolSelectionRuntime(runtime).select_tools_for_turn("ignored", None)
    assert result is None  # capability gate short-circuits
    runtime.tool_registrar.ensure_tools_for_query.assert_called_once_with(
        "open a pull request on github"
    )


@pytest.mark.asyncio
async def test_hydration_falls_back_to_context_msg():
    runtime = _make_runtime(message=None)
    await ToolSelectionRuntime(runtime).select_tools_for_turn("gh pr list please", None)
    runtime.tool_registrar.ensure_tools_for_query.assert_called_once_with("gh pr list please")


@pytest.mark.asyncio
async def test_tools_attr_fallback_for_older_hosts():
    """Hosts without ``tool_registrar`` hydrate via a registrar-ish ``tools``."""
    runtime = _make_runtime()
    runtime.tool_registrar = None
    runtime.tools = MagicMock()
    await ToolSelectionRuntime(runtime).select_tools_for_turn("ignored", None)
    # anchor comes from _current_user_message even when .tools supplies the hook
    runtime.tools.ensure_tools_for_query.assert_called_once_with("open a pull request on github")


@pytest.mark.asyncio
async def test_host_without_registrar_is_tolerated():
    runtime = _make_runtime()
    runtime.tool_registrar = None
    result = await ToolSelectionRuntime(runtime).select_tools_for_turn("hello", None)
    assert result is None  # no raise; gate still short-circuits


@pytest.mark.asyncio
async def test_hydration_failure_does_not_break_the_turn():
    runtime = _make_runtime()
    runtime.tool_registrar.ensure_tools_for_query.side_effect = RuntimeError("boom")
    result = await ToolSelectionRuntime(runtime).select_tools_for_turn("x", None)
    assert result is None  # swallowed; selection continues
