"""Tests for sub-agent delegation in planning — Layer 5 of agentic execution quality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.planning.base import (
    ExecutionPlan,
    PlanResult,
    PlanStep,
    StepResult,
    StepStatus,
    StepType,
)


def _make_step(
    step_type: StepType = StepType.RESEARCH,
    sub_agent_role: str = None,
    step_id: str = "step-1",
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description="Research the auth module",
        step_type=step_type,
        depends_on=[],
        estimated_tool_calls=5,
        requires_approval=False,
        sub_agent_role=sub_agent_role,
        status=StepStatus.PENDING,
        result=None,
        context={},
    )


def _make_planner(with_sub_agent_orchestrator=True):
    from victor.agent.planning.autonomous import AutonomousPlanner

    mock_orchestrator = MagicMock()
    mock_orchestrator.chat = AsyncMock(return_value=MagicMock(content="done"))

    mock_sub_agent = None
    if with_sub_agent_orchestrator:
        mock_sub_agent = MagicMock()
        mock_sub_agent.spawn = AsyncMock(
            return_value=MagicMock(
                success=True,
                summary="Research complete",
                tool_calls_used=3,
                duration_seconds=1.5,
            )
        )

    planner = AutonomousPlanner(
        orchestrator=mock_orchestrator,
        sub_agent_orchestrator=mock_sub_agent,
    )
    return planner, mock_orchestrator, mock_sub_agent


class TestShouldDelegate:
    def test_should_delegate_research_with_orchestrator(self):
        planner, _, _ = _make_planner(with_sub_agent_orchestrator=True)
        step = _make_step(StepType.RESEARCH)
        assert planner._should_delegate_step(step) is True

    def test_should_not_delegate_without_orchestrator(self):
        planner, _, _ = _make_planner(with_sub_agent_orchestrator=False)
        step = _make_step(StepType.RESEARCH)
        assert planner._should_delegate_step(step) is False

    def test_should_not_delegate_implementation(self):
        planner, _, _ = _make_planner(with_sub_agent_orchestrator=True)
        step = _make_step(StepType.IMPLEMENTATION)
        assert planner._should_delegate_step(step) is False

    def test_should_delegate_with_sub_agent_role(self):
        planner, _, _ = _make_planner(with_sub_agent_orchestrator=True)
        step = _make_step(StepType.IMPLEMENTATION, sub_agent_role="researcher")
        assert planner._should_delegate_step(step) is True


class TestStepDelegation:
    @pytest.mark.asyncio
    async def test_sequential_delegates_research_steps(self):
        planner, mock_orch, mock_sub = _make_planner(with_sub_agent_orchestrator=True)
        step = _make_step(StepType.RESEARCH, step_id="s1")
        plan = ExecutionPlan(id="plan-1", goal="test", steps=[step])
        result = PlanResult(
            plan_id="plan-1",
            success=False,
            steps_completed=0,
            steps_failed=0,
            total_tool_calls=0,
            total_duration=0.0,
            step_results={},
        )

        await planner._execute_sequential(plan, result, auto_approve=True, progress_callback=None)
        mock_sub.spawn.assert_called_once()

    @pytest.mark.asyncio
    async def test_sequential_runs_non_research_locally(self):
        planner, mock_orch, mock_sub = _make_planner(with_sub_agent_orchestrator=True)
        step = _make_step(StepType.IMPLEMENTATION, step_id="s1")
        plan = ExecutionPlan(id="plan-1", goal="test", steps=[step])
        result = PlanResult(
            plan_id="plan-1",
            success=False,
            steps_completed=0,
            steps_failed=0,
            total_tool_calls=0,
            total_duration=0.0,
            step_results={},
        )

        await planner._execute_sequential(plan, result, auto_approve=True, progress_callback=None)
        mock_orch.chat.assert_called_once()
        mock_sub.spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_delegation_result_wrapped_as_step_result(self):
        planner, _, mock_sub = _make_planner(with_sub_agent_orchestrator=True)
        step = _make_step(StepType.RESEARCH)
        result = await planner._execute_step_via_subagent(step)
        assert isinstance(result, StepResult)
        assert result.success is True
        assert "Research complete" in result.output

    @pytest.mark.asyncio
    async def test_delegation_preserves_step_allowed_tools(self):
        planner, _, mock_sub = _make_planner(with_sub_agent_orchestrator=True)
        step = _make_step(StepType.RESEARCH)
        step.allowed_tools = ["grep", "shell"]

        result = await planner._execute_step_via_subagent(step)

        assert result.success is True
        assert mock_sub.spawn.await_args.kwargs["allowed_tools"] == [
            "read",
            "ls",
            "grep",
            "shell",
        ]

    @pytest.mark.asyncio
    async def test_delegation_failure_marks_step_failed(self):
        planner, _, mock_sub = _make_planner(with_sub_agent_orchestrator=True)
        mock_sub.spawn = AsyncMock(side_effect=RuntimeError("spawn failed"))
        step = _make_step(StepType.RESEARCH)
        result = await planner._execute_step_via_subagent(step)
        assert result.success is False
        assert "spawn failed" in result.error
