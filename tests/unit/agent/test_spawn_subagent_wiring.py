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

"""Dormancy tests for the FOU-A1 spawn_subagent production wiring.

The FEP-0041 lesson (re-learned at FOU-A2c): a shipped-but-unwired tool
ships nothing. These tests pin the WIRING itself:

- constructing ``AgentOrchestrator`` the way production does (composition
  root for CLI chat, TUI, and the API path via ``AgentFactory``) must bind
  the subagent substrate runner into the session's tool-executor context,
  with the ``spawn_subagent_tool_budget`` setting resolved alongside it;
- the session registry must expose the ``spawn_subagent`` tool to the LLM
  (dynamic discovery registration, not just module existence);
- ``subagent_orchestration_enabled=false`` must degrade to the runner
  returning None through the REAL construction path — the tool then refuses,
  the session never breaks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings


@pytest.fixture
def mock_provider():
    """Mock provider, mirroring the orchestrator-core unit fixtures."""
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.supports_tools.return_value = True
    provider.get_context_window.return_value = 100000
    provider.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    return provider


def _production_settings(**overrides) -> Settings:
    """Settings the way a real (offline-safe) session resolves them."""
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
        use_mcp_tools=False,
        **overrides,
    )


def _build_orchestrator(settings: Settings, mock_provider) -> AgentOrchestrator:
    """Construct the chat composition root exactly as production does."""
    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        return AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )


# =============================================================================
# The dormancy test: the PRODUCTION path wires the tool
# =============================================================================


async def test_production_construction_wires_spawn_runner(mock_provider):
    """Real construction binds the runner + budget into the executor context
    and registers the tool in the session registry — placement, not just
    existence."""
    orch = _build_orchestrator(_production_settings(), mock_provider)

    # 1. The binding happened through the real constructor, not by the test.
    runner = orch.tool_executor.context.get("spawn_subagent_runner")
    assert callable(runner)
    # Budget key is always present (None = substrate role default).
    assert "spawn_subagent_budget" in orch.tool_executor.context
    assert orch.tool_executor.context["spawn_subagent_budget"] is None
    assert orch._subagent_orchestration_enabled is True

    # 2. The LLM can actually reach the tool: the real turn path
    # (ToolService tool supply) hydrates the full catalog through
    # registrar.get_all_tools() — lazy startup registers only the bootstrap
    # set at construction, so assert through the same call production makes.
    all_tools = orch.tool_registrar.get_all_tools()
    assert any(getattr(t, "name", "") == "spawn_subagent" for t in all_tools)
    assert orch.tools.get("spawn_subagent") is not None

    # 3. The runner resolves the REAL lazy subagent orchestrator (the member
    # execution substrate), not a stub.
    assert orch.subagent_orchestrator is not None
    from victor.agent.subagents.orchestrator import SubAgentOrchestrator

    assert isinstance(orch.subagent_orchestrator, SubAgentOrchestrator)


async def test_spawn_tool_budget_setting_reaches_executor_context(mock_provider):
    """ToolSettings.spawn_subagent_tool_budget flows through the live
    _tool_setting resolution into the executor context the tool reads."""
    settings = _production_settings()
    from victor.config.tool_settings import ToolSettings

    settings.tools = ToolSettings(spawn_subagent_tool_budget=7)
    orch = _build_orchestrator(settings, mock_provider)

    assert orch.tool_executor.context["spawn_subagent_budget"] == 7


async def test_spawn_runner_refuses_depth_recursion_through_real_path(mock_provider):
    """Through the REAL wiring, a runner invocation constructs the member
    with can_spawn_subagents=False — the substrate-level depth-1 guard."""
    orch = _build_orchestrator(_production_settings(), mock_provider)
    runner = orch.tool_executor.context["spawn_subagent_runner"]

    spawn_spy = AsyncMock(return_value=MagicMock(to_dict=lambda: {"success": True}))
    orch.subagent_orchestrator.spawn = spawn_spy

    # The tool layer already strips spawn_subagent; the runner must not
    # re-authorize it and must hard-code can_spawn_subagents=False.
    await runner(
        prompt="task",
        tools=["read", "spawn_subagent"],
        model=None,
        tool_budget=None,
    )

    kwargs = spawn_spy.await_args.kwargs
    assert kwargs["can_spawn_subagents"] is False
    assert "spawn_subagent" not in (kwargs["allowed_tools"] or [])


# =============================================================================
# Negative: disabled orchestration degrades to a runner that refuses
# =============================================================================


async def test_disabled_orchestration_runner_returns_none(mock_provider):
    """subagent_orchestration_enabled=false → no substrate, runner returns
    None through the real construction path (the tool maps that to a
    structured refusal; session start is unaffected)."""
    settings = _production_settings(subagent_orchestration_enabled=False)
    orch = _build_orchestrator(settings, mock_provider)

    assert orch._subagent_orchestration_enabled is False
    assert orch.subagent_orchestrator is None

    runner = orch.tool_executor.context.get("spawn_subagent_runner")
    assert callable(runner)  # still bound — refusal is the runner's job
    assert await runner(prompt="task", tools=None, model=None, tool_budget=None) is None
