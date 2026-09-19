"""Typed planning capability behavior for the FEP-0031 chat boundary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.chat_planning import ChatPlanning


async def test_planning_capability_delegates_without_exposing_runtime_host():
    guidance = SimpleNamespace(
        apply_intent_guard=MagicMock(),
        apply_task_guidance=MagicMock(),
        classify_task_keywords=MagicMock(return_value={"task_type": "analysis"}),
        current_intent=MagicMock(return_value="write_allowed"),
    )
    planner = SimpleNamespace(
        infer_goals_from_message=MagicMock(return_value=["inspect"]),
        plan_tools=MagicMock(return_value=["read"]),
    )
    selection = SimpleNamespace(
        select_tools_for_turn=AsyncMock(return_value=["read-definition"]),
    )
    planning = ChatPlanning(
        guidance=guidance,
        planner=planner,
        selection=selection,
    )

    planning.apply_intent_guard("inspect app.py")
    assert planning.classify_task_keywords("inspect app.py") == {"task_type": "analysis"}
    assert planning.current_intent() == "write_allowed"
    planning.apply_task_guidance(
        user_message="inspect app.py",
        unified_task_type="analyze",
        is_analysis_task=True,
        is_action_task=False,
        needs_execution=False,
        max_exploration_iterations=8,
    )

    assert planning.infer_goals("inspect app.py") == ["inspect"]
    assert planning.plan_tools(["inspect"], ["query"]) == ["read"]
    assert await planning.select_tools("inspect app.py", ["inspect"], planned_tools=["read"]) == [
        "read-definition"
    ]
    guidance.apply_intent_guard.assert_called_once_with("inspect app.py")
    guidance.apply_task_guidance.assert_called_once()
    selection.select_tools_for_turn.assert_awaited_once_with(
        "inspect app.py", ["inspect"], planned_tools=["read"]
    )
    assert not hasattr(planning, "runtime_host")
    assert not hasattr(planning, "orchestrator")


async def test_missing_planning_dependencies_fail_before_turn_execution():
    planning = ChatPlanning()

    with pytest.raises(TypeError, match="task-guidance"):
        planning.apply_intent_guard("inspect app.py")
    with pytest.raises(TypeError, match="task-guidance"):
        planning.current_intent()
    with pytest.raises(TypeError, match="tool planner"):
        planning.infer_goals("inspect app.py")
    with pytest.raises(TypeError, match="tool-selection"):
        await planning.select_tools("inspect app.py", [])
