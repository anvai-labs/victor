"""Tests for ContextService compatibility with dict-based message history."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.context_service import (
    ContextService,
    ContextServiceConfig,
    compact_context_if_recommended,
)
from victor.agent.runtime.context import AgentRuntimeContext
from victor.agent.services.context_service import ContextServiceRegistry


@pytest.mark.asyncio
async def test_compact_context_if_recommended_uses_service_policy():
    service = SimpleNamespace(
        get_compaction_recommendation=MagicMock(return_value={"should_compact": True}),
        compact_context=AsyncMock(return_value=3),
    )

    result = await compact_context_if_recommended(
        service,
        strategy="semantic",
        min_messages=4,
    )

    assert result.handled is True
    assert result.should_compact is True
    assert result.messages_removed == 3
    assert result.recommendation == {"should_compact": True}
    service.get_compaction_recommendation.assert_called_once()
    service.compact_context.assert_awaited_once_with(strategy="semantic", min_messages=4)


@pytest.mark.asyncio
async def test_compact_context_if_recommended_noop_still_handles_policy():
    service = SimpleNamespace(
        get_compaction_recommendation=MagicMock(return_value={"should_compact": False}),
        compact_context=AsyncMock(return_value=3),
    )

    result = await compact_context_if_recommended(service, strategy="tiered")

    assert result.handled is True
    assert result.should_compact is False
    assert result.messages_removed == 0
    service.get_compaction_recommendation.assert_called_once()
    service.compact_context.assert_not_called()


@pytest.mark.asyncio
async def test_compact_context_if_recommended_ignores_non_service_objects():
    result = await compact_context_if_recommended(SimpleNamespace(), strategy="tiered")

    assert result.handled is False
    assert result.should_compact is False
    assert result.messages_removed == 0


@pytest.mark.asyncio
async def test_get_context_metrics_counts_dict_messages_added_via_keyword_fields():
    service = ContextService(ContextServiceConfig(max_tokens=1000))

    service.add_message(role="system", content="system prompt")
    service.add_message(role="user", content="hello")
    service.add_message({"role": "assistant", "content": "done"})

    metrics = await service.get_context_metrics()

    assert metrics.message_count == 3
    assert metrics.user_message_count == 1
    assert metrics.assistant_message_count == 1
    assert metrics.system_prompt_tokens == service.estimate_tokens("system prompt")


def test_get_messages_filters_dict_messages_by_role():
    service = ContextService(ContextServiceConfig())

    service.add_messages(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "done"},
        ]
    )

    assert service.get_messages(role="assistant") == [{"role": "assistant", "content": "done"}]


def test_clear_messages_retains_system_dict_messages():
    service = ContextService(ContextServiceConfig())

    service.add_messages(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ]
    )

    service.clear_messages(retain_system=True)

    assert service.get_messages() == [{"role": "system", "content": "system prompt"}]


def test_get_context_size_counts_dict_messages():
    service = ContextService(ContextServiceConfig(max_tokens=1000))

    service.add_messages(
        [
            {"role": "user", "content": "x" * 40},
            {"role": "assistant", "content": "y" * 80},
        ]
    )

    assert service.get_context_size() == 30


@pytest.mark.asyncio
async def test_prepare_for_tool_output_injection_compacts_and_tracks_saved_tokens():
    service = ContextService(
        ContextServiceConfig(max_tokens=100, default_compaction_strategy="tiered")
    )
    service.add_messages(
        [
            {"role": "user", "content": "x" * 40},
            {"role": "assistant", "content": "y" * 40},
            {"role": "user", "content": "z" * 40},
            {"role": "assistant", "content": "w" * 40},
            {"role": "user", "content": "q" * 40},
            {"role": "assistant", "content": "r" * 40},
            {"role": "user", "content": "s" * 40},
            {"role": "assistant", "content": "t" * 40},
        ]
    )

    result = await service.prepare_for_tool_output_injection(
        18,
        provider_name="ollama",
        model_name="local-model",
        task_type="analysis",
        min_messages=6,
        default_strategy="tiered",
    )

    assert result["should_compact"] is True
    assert result["compacted"] is True
    assert result["messages_removed"] == 2
    assert result["saved_tokens"] == 20
    assert result["strategy"] == "tiered"
    assert result["policy_reason"] == "tool_output_exceeds_remaining_budget"
    assert service.get_performance_metrics()["last_compaction_saved_tokens"] == 20


def test_context_service_registry_scopes_context_by_agent_session():
    registry = ContextServiceRegistry(ContextServiceConfig(max_tokens=100))
    root = AgentRuntimeContext(
        agent_id="root",
        display_name="Root",
        role="manager",
        session_id="session_root",
    )
    child = root.derive_child(
        agent_id="child",
        display_name="Child",
        role="researcher",
        member_id="member_1",
        team_id="team_1",
    )

    root_context = registry.get_or_create(root)
    child_context = registry.get_or_create(child)
    root_context.add_message(role="user", content="root")
    child_context.add_message(role="user", content="child")

    assert root_context is registry.get_or_create(root)
    assert root_context is not child_context
    assert root_context.get_messages() == [{"role": "user", "content": "root"}]
    assert child_context.get_messages() == [{"role": "user", "content": "child"}]


@pytest.mark.asyncio
async def test_context_service_registry_compacts_only_target_agent_session():
    registry = ContextServiceRegistry(
        ContextServiceConfig(max_tokens=20, overflow_threshold_percent=50.0)
    )
    root = AgentRuntimeContext(
        agent_id="root",
        display_name="Root",
        role="manager",
        session_id="session_root",
    )
    child = root.derive_child(
        agent_id="child",
        display_name="Child",
        role="researcher",
        member_id="member_1",
        team_id="team_1",
    )
    root_context = registry.get_or_create(root)
    child_context = registry.get_or_create(child)
    for index in range(8):
        root_context.add_message(role="user", content=f"root {index}")
        child_context.add_message(role="user", content="x" * 20)

    result = await registry.compact_if_needed(child, min_messages=4)

    assert result["compacted"] is True
    assert result["agent_id"] == "child"
    assert result["session_id"] == child.session_id
    assert len(child_context.get_messages()) == 4
    assert len(root_context.get_messages()) == 8


# =============================================================================
# Wave A3: task_complexity threading tests
# =============================================================================


@pytest.mark.asyncio
async def test_compact_context_if_recommended_accepts_task_complexity_param():
    """compact_context_if_recommended() must accept a task_complexity parameter."""
    import inspect

    from victor.agent.services.context_service import compact_context_if_recommended

    sig = inspect.signature(compact_context_if_recommended)
    assert "task_complexity" in sig.parameters, (
        "compact_context_if_recommended() must accept a task_complexity: Optional[str] = None "
        "keyword argument so TurnExecutor can pass high-complexity context to the compactor."
    )


@pytest.mark.asyncio
async def test_high_complexity_propagated_to_context_service():
    """task_complexity='high' should be forwarded when the service accepts it."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from victor.agent.services.context_service import compact_context_if_recommended

    received: list = []

    def recording_recommendation(*args, **kwargs):
        received.append(("recommendation", args, kwargs))
        return {
            "should_compact": True,
            "task_complexity": kwargs.get("task_complexity"),
        }

    compact_mock = AsyncMock(return_value=2)

    service = SimpleNamespace(
        get_compaction_recommendation=recording_recommendation,
        compact_context=compact_mock,
    )

    result = await compact_context_if_recommended(
        service, strategy="tiered", task_complexity="high"
    )

    assert result.handled is True
    assert result.should_compact is True


@pytest.mark.asyncio
async def test_none_complexity_is_noop_for_context_service():
    """task_complexity=None must not alter behavior relative to omitting the argument."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from victor.agent.services.context_service import compact_context_if_recommended

    service = SimpleNamespace(
        get_compaction_recommendation=MagicMock(return_value={"should_compact": False}),
        compact_context=AsyncMock(return_value=0),
    )

    result_with_none = await compact_context_if_recommended(
        service, strategy="tiered", task_complexity=None
    )
    result_without = await compact_context_if_recommended(service, strategy="tiered")

    assert result_with_none.handled == result_without.handled
    assert result_with_none.should_compact == result_without.should_compact


@pytest.mark.asyncio
async def test_turn_executor_passes_task_complexity_to_compaction():
    """Turn execution runtime should pass task_complexity to compact_context_if_recommended."""
    import inspect

    from victor.agent.services import turn_execution_runtime

    source = inspect.getsource(turn_execution_runtime)
    assert "task_complexity" in source, (
        "turn_execution_runtime.py must pass task_complexity to compact_context_if_recommended(). "
        "Check _check_context_service_compaction() — it should forward "
        "task_classification.complexity.value."
    )
