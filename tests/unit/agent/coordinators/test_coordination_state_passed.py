# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.coordinators.coordination_state_passed import (
    CoordinationStatePassedCoordinator,
)
from victor.agent.coordinators.state_context import ContextSnapshot


@pytest.mark.asyncio
async def test_suggest_returns_transitions_for_coordination_payload():
    coordination_runtime = MagicMock()
    suggestion = SimpleNamespace(
        primary_team=SimpleNamespace(team_name="feature_team", confidence=0.9),
        primary_workflow=SimpleNamespace(workflow_name="feature_implementation", confidence=0.8),
        team_recommendations=[MagicMock()],
        workflow_recommendations=[MagicMock()],
    )
    coordination_runtime.suggest_for_task.return_value = suggestion
    coordination_runtime.serialize_suggestion.return_value = {"action": "auto_spawn"}

    coordinator = CoordinationStatePassedCoordinator(coordination_runtime=coordination_runtime)
    snapshot = ContextSnapshot(
        messages=(),
        session_id="session-1",
        conversation_stage="initial",
        settings=MagicMock(),
        model="test-model",
        provider="test-provider",
        max_tokens=4096,
        temperature=0.7,
        conversation_state={"task_complexity": "high"},
        session_state={},
        observed_files=(),
        capabilities={},
    )

    result = await coordinator.suggest(snapshot, task_type="feature", mode="build")

    coordination_runtime.suggest_for_task.assert_called_once_with(
        task_type="feature",
        complexity="high",
        mode="build",
        coordination_advisor=None,
        vertical_context=None,
    )
    assert result.metadata == {"action": "auto_spawn"}
    assert result.confidence == 0.9
    assert len(result.transitions.transitions) == 3


@pytest.mark.asyncio
async def test_suggest_defaults_complexity_when_context_has_none():
    coordination_runtime = MagicMock()
    coordination_runtime.suggest_for_task.return_value = SimpleNamespace(
        primary_team=None,
        primary_workflow=None,
        team_recommendations=[],
        workflow_recommendations=[],
    )
    coordination_runtime.serialize_suggestion.return_value = {"action": "none"}

    coordinator = CoordinationStatePassedCoordinator(coordination_runtime=coordination_runtime)
    snapshot = ContextSnapshot(
        messages=(),
        session_id="session-1",
        conversation_stage="initial",
        settings=MagicMock(),
        model="test-model",
        provider="test-provider",
        max_tokens=4096,
        temperature=0.7,
        conversation_state={},
        session_state={},
        observed_files=(),
        capabilities={},
    )

    await coordinator.suggest(snapshot, task_type="research")

    coordination_runtime.suggest_for_task.assert_called_once_with(
        task_type="research",
        complexity="medium",
        mode="build",
        coordination_advisor=None,
        vertical_context=None,
    )
