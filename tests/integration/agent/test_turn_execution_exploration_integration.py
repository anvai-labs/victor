# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integration coverage for TurnExecutor exploration ownership."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.conversation.types import MESSAGE_SOURCE_METADATA_KEY, MessageSource
from victor.agent.coordinators.exploration_state_passed import (
    ExplorationStatePassedCoordinator,
)
from victor.agent.services.exploration_runtime import ExplorationResult
from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.framework.task.protocols import TaskComplexity


@pytest.mark.asyncio
async def test_turn_executor_uses_shared_state_passed_exploration_coordinator() -> None:
    chat_context = MagicMock()
    chat_context.settings = MagicMock()
    chat_context.add_message = MagicMock()
    chat_context.conversation = MagicMock()

    executor = TurnExecutor(
        chat_context=chat_context,
        tool_context=MagicMock(),
        provider_context=MagicMock(provider_name="anthropic", model="claude-test"),
        execution_provider=MagicMock(),
    )

    exploration = ExplorationStatePassedCoordinator()
    exploration._inner.explore_parallel = AsyncMock(
        return_value=ExplorationResult(
            file_paths=["victor/agent/services/exploration_runtime.py"],
            summary="shared state-passed exploration",
            duration_seconds=0.5,
            tool_calls=2,
        )
    )

    orchestrator = SimpleNamespace(
        _orchestration_facade=SimpleNamespace(exploration_state_passed=exploration),
        messages=[],
        session_id="integration-session",
        conversation_stage="initial",
        settings=MagicMock(),
        model="claude-test",
        provider_name="anthropic",
        max_tokens=4096,
        temperature=0.2,
        conversation_state={},
        session_state={},
        observed_files=[],
        _capabilities={},
        add_message=MagicMock(),
    )
    executor._orchestrator = orchestrator
    task_classification = SimpleNamespace(complexity=TaskComplexity.COMPLEX)

    with (
        patch(
            "victor.config.settings.load_settings",
            return_value=SimpleNamespace(pipeline=SimpleNamespace(parallel_exploration=True)),
        ),
        patch(
            "victor.config.settings.get_project_paths",
            return_value=SimpleNamespace(project_root="/tmp/project"),
        ),
        patch(
            "victor.agent.budget.resource_calculator.calculate_exploration_budget",
            return_value=SimpleNamespace(
                max_parallel_agents=2,
                tool_budget_per_agent=3,
                exploration_timeout=5,
            ),
        ),
        patch(
            "victor.agent.coordinators.factory_support.create_exploration_coordinator",
            side_effect=AssertionError("direct exploration helper should not be used"),
        ),
    ):
        explored = await executor._run_parallel_exploration(
            "review the shared exploration coordinator path",
            task_classification,
        )

    assert explored is True
    assert executor._exploration_coordinator is exploration
    assert orchestrator.conversation_state["explored_files"] == [
        "victor/agent/services/exploration_runtime.py"
    ]
    assert (
        orchestrator.conversation_state["exploration_summary"] == "shared state-passed exploration"
    )
    chat_context.add_message.assert_called_once_with(
        "user",
        "[Parallel exploration results]\nshared state-passed exploration",
        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_GUIDANCE.value},
    )
