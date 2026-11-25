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

"""Dormancy/wiring tests for FOU-A1 phase-2: background subagents.

The FEP-0041 lesson, applied to phase 2: background mode + the
``subagent_status`` / ``subagent_result`` companions ship nothing unless the
REAL composition root wires them. These tests pin, through the production
``AgentOrchestrator`` construction and its live ``ToolExecutor``:

- the per-session ``SubAgentRunRegistry`` is bound into the executor context
  (same instance the substrate runner detaches into);
- the companion tools are discoverable through the session registry;
- ``spawn_subagent(background=True)`` executed through the REAL tool executor
  returns immediately, registers a running run, and the companions find it
  through the production executor context;
- the terminal notification reaches an event callback when wired;
- background children pass through the SAME depth-1 enforcement and
  intersection as foreground (spawn tool absent from the child tool set,
  ``can_spawn_subagents=False``);
- multiple background children + one foreground interleave.

Only the substrate's ``SubAgentOrchestrator.spawn`` is stubbed — no provider,
database, or network is touched.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.subagents.run_registry import SubAgentRunRegistry
from victor.config.settings import Settings

# SubAgentResult.to_dict() shapes by prompt, as the real substrate returns.
_RESULT_BY_PROMPT = {
    "bg task": {
        "success": True,
        "summary": "bg done",
        "details": {"role": "researcher"},
        "tool_calls_used": 2,
        "context_size": 1000,
        "duration_seconds": 0.5,
        "error": None,
    },
    "fg task": {
        "success": True,
        "summary": "fg done",
        "details": {"role": "researcher"},
        "tool_calls_used": 1,
        "context_size": 1000,
        "duration_seconds": 0.1,
        "error": None,
    },
}


def _production_settings(**overrides) -> Settings:
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
        use_mcp_tools=False,
        **overrides,
    )


def _build_orchestrator(settings: Settings, mock_provider) -> AgentOrchestrator:
    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        return AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.supports_tools.return_value = True
    provider.get_context_window.return_value = 100000
    provider.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    return provider


@pytest.fixture
async def orch(mock_provider):
    orch = _build_orchestrator(_production_settings(), mock_provider)
    # The production turn path hydrates the lazy session catalog through
    # registrar.get_all_tools() (same call as the phase-1 wiring test);
    # mirror it so executor-level tests see the companions placed in the
    # session registry the way a real turn supplies them.
    orch.tool_registrar.get_all_tools()
    yield orch
    # Detached background tasks from these tests must not leak into others.
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _spawn_stub_with_gates(gates: dict):
    """Stub SubAgentOrchestrator.spawn: blocks on the gate keyed by the task
    prompt, then returns the mapped SubAgentResult mock."""

    async def _spawn(**kwargs):
        prompt = kwargs["task"]
        gate = gates.get(prompt)
        if gate is not None:
            await asyncio.wait_for(gate.wait(), timeout=5.0)
        payload = _RESULT_BY_PROMPT.get(prompt, {"success": True, "summary": prompt})
        return MagicMock(
            to_dict=lambda: {"details": {"child_session_id": kwargs["child_session_id"]}, **payload}
        )

    return AsyncMock(side_effect=_spawn)


async def _wait_terminal(registry: SubAgentRunRegistry, subagent_id: str, timeout: float = 5.0):
    async def _poll():
        while True:
            record = registry.get(subagent_id)
            if record is not None and record.is_terminal:
                return record
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(_poll(), timeout=timeout)


async def _wait_spawn_called(spy, count: int = 1, timeout: float = 5.0):
    async def _poll():
        while spy.await_count < count:
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(_poll(), timeout=timeout)


# =============================================================================
# The dormancy test: the PRODUCTION path wires the registry + companions
# =============================================================================


async def test_production_construction_binds_run_registry(mock_provider):
    """Real construction binds ONE per-session run registry into the executor
    context — the same instance the substrate runner detaches into."""
    orch = _build_orchestrator(_production_settings(), mock_provider)
    orch.tool_registrar.get_all_tools()  # production turn-path hydration

    bound = orch.tool_executor.context.get("subagent_run_registry")
    assert isinstance(bound, SubAgentRunRegistry)
    assert orch.subagent_run_registry is bound


async def test_companion_tools_reachable_through_session_registry(mock_provider):
    """The LLM can reach both companions: registered through the same dynamic
    discovery path the spawn tool uses (placement, not module existence)."""
    orch = _build_orchestrator(_production_settings(), mock_provider)

    all_tools = orch.tool_registrar.get_all_tools()
    names = {getattr(t, "name", "") for t in all_tools}
    assert "subagent_status" in names
    assert "subagent_result" in names
    assert orch.tools.get("subagent_status") is not None
    assert orch.tools.get("subagent_result") is not None


# =============================================================================
# End-to-end: spawn(background=true) through the REAL tool executor
# =============================================================================


async def test_background_spawn_returns_immediately_and_companions_track_it(orch):
    gates = {"bg task": asyncio.Event()}
    orch.subagent_orchestrator.spawn = _spawn_stub_with_gates(gates)
    registry = orch.tool_executor.context["subagent_run_registry"]

    handle_result = await orch.tool_executor.execute(
        "spawn_subagent", {"prompt": "bg task", "background": True}
    )
    assert handle_result.success, handle_result.error
    handle = handle_result.result

    # Immediate structured handle — the child is still running.
    assert handle["success"] is True
    assert handle["background"] is True
    assert handle["status"] == "running"
    assert "-spawn-" in handle["child_session_id"]
    subagent_id = handle["subagent_id"]

    # The tool call did not block on child completion: the substrate spawn is
    # in flight, and status/result through the SAME executor context see it.
    record = registry.get(subagent_id)
    assert record is not None and record.status.value == "running"
    status = await orch.tool_executor.execute("subagent_status", {"subagent_id": subagent_id})
    assert status.result["found"] is True
    assert status.result["status"] == "running"

    running_result = await orch.tool_executor.execute(
        "subagent_result", {"subagent_id": subagent_id}
    )
    assert running_result.result["result_ready"] is False
    assert "still running" in running_result.result["message"]

    # Terminal transition surfaces through the companions.
    gates["bg task"].set()
    finished = await _wait_terminal(registry, subagent_id)
    assert finished.status.value == "completed"

    terminal_result = await orch.tool_executor.execute(
        "subagent_result", {"subagent_id": subagent_id}
    )
    payload = terminal_result.result
    assert payload["result_ready"] is True
    assert payload["result"]["success"] is True
    assert payload["result"]["content"] == "bg done"
    assert payload["result"]["child_session_id"] == handle["child_session_id"]


async def test_background_terminal_notification_observed_through_real_path(orch):
    events: list = []
    gates = {"bg task": asyncio.Event()}
    orch.subagent_orchestrator.spawn = _spawn_stub_with_gates(gates)
    registry = orch.tool_executor.context["subagent_run_registry"]
    registry.set_event_callback(lambda event, data: events.append((event, data)))

    handle = (
        await orch.tool_executor.execute(
            "spawn_subagent", {"prompt": "bg task", "background": True}
        )
    ).result

    gates["bg task"].set()
    await _wait_terminal(registry, handle["subagent_id"])

    assert events, "terminal notification never fired through the real wiring"
    event, data = events[0]
    assert event == "subagent_completed"
    assert data["subagent_id"] == handle["subagent_id"]
    assert data["summary"] == "bg done"
    assert data["child_session_id"] == handle["child_session_id"]


# =============================================================================
# Depth/enforcement: background children get the SAME treatment
# =============================================================================


async def test_background_child_tool_set_intersects_and_excludes_spawn(orch):
    """Same compute_child_tool_set intersection and can_spawn_subagents=False
    as the foreground runner — a background child cannot spawn (depth 1)."""
    captured: list = []

    async def _spawn(**kwargs):
        captured.append(kwargs)
        return MagicMock(to_dict=lambda: {"success": True, "summary": "done", "details": {}})

    orch.subagent_orchestrator.spawn = AsyncMock(side_effect=_spawn)
    runner = orch.tool_executor.context["spawn_subagent_runner"]

    handle = await runner(
        prompt="bg task",
        tools=["read", "spawn_subagent", "grep"],  # 'grep' not parent-admitted here
        model=None,
        tool_budget=None,
        background=True,
    )

    assert handle["success"] is True and handle["background"] is True
    await _wait_spawn_called(orch.subagent_orchestrator.spawn)
    kwargs = captured[0]
    assert kwargs["can_spawn_subagents"] is False
    assert "spawn_subagent" not in (kwargs["allowed_tools"] or [])
    assert kwargs["allowed_tools"] == ["read"]
    assert kwargs["child_session_id"] == handle["child_session_id"]

    await _wait_terminal(orch.subagent_run_registry, handle["subagent_id"])


async def test_disabled_orchestration_background_refuses_through_real_path(mock_provider):
    """subagent_orchestration_enabled=false: background mode refuses (the
    session never breaks), while the still-bound registry answers not-found."""
    orch = _build_orchestrator(
        _production_settings(subagent_orchestration_enabled=False), mock_provider
    )
    orch.tool_registrar.get_all_tools()  # production turn-path hydration

    result = await orch.tool_executor.execute(
        "spawn_subagent", {"prompt": "bg task", "background": True}
    )
    assert result.result["success"] is False
    assert "orchestration" in result.result["error"].lower()

    status = await orch.tool_executor.execute("subagent_status", {"subagent_id": "nope"})
    assert status.result["success"] is False
    assert status.result["found"] is False


async def test_status_hallucinated_id_not_found_through_production_executor(orch):
    result = await orch.tool_executor.execute(
        "subagent_status", {"subagent_id": "subagent-fabricated"}
    )
    # The TOOL call resolves (structured not-found payload), distinct from an
    # executor-level failure: the fabricated id simply matches nothing.
    assert result.result is not None
    assert result.result["success"] is False
    assert result.result["found"] is False
    assert "unknown subagent id" in result.result["error"].lower()


# =============================================================================
# Concurrency: multiple background children + one foreground, interleaved
# =============================================================================


async def test_multiple_background_children_and_one_foreground_interleave(orch):
    gates = {
        "bg a": asyncio.Event(),
        "bg b": asyncio.Event(),
        "bg c": asyncio.Event(),
        "fg task": asyncio.Event(),
    }
    orch.subagent_orchestrator.spawn = _spawn_stub_with_gates(gates)
    registry = orch.tool_executor.context["subagent_run_registry"]

    handles = {}
    for prompt in ("bg a", "bg b", "bg c"):
        result = await orch.tool_executor.execute(
            "spawn_subagent", {"prompt": prompt, "background": True}
        )
        handles[prompt] = result.result
    assert all(h["status"] == "running" for h in handles.values())
    assert len({h["subagent_id"] for h in handles.values()}) == 3
    assert len(registry) == 3

    # The foreground spawn BLOCKS on its child (phase-1 semantics) while the
    # three background children run concurrently.
    fg_task = asyncio.create_task(
        orch.tool_executor.execute("spawn_subagent", {"prompt": "fg task"})
    )
    await asyncio.sleep(0.05)
    assert not fg_task.done()  # foreground is still awaiting its substrate

    # Complete the background children one at a time.
    for prompt in ("bg a", "bg b", "bg c"):
        gates[prompt].set()
        record = await _wait_terminal(registry, handles[prompt]["subagent_id"])
        assert record.status.value == "completed"

    # Then release the foreground child and map its result as before.
    gates["fg task"].set()
    fg_result = await asyncio.wait_for(fg_task, timeout=5.0)
    assert fg_result.result["success"] is True
    assert fg_result.result["content"] == "fg done"

    assert len(registry) == 3  # foreground spawn never entered the registry
