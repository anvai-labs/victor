from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.benchmark.agent import BenchmarkAgent, BenchmarkAgentConfig
from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.task import TaskResult


class _FakeFrameworkAgent:
    def __init__(
        self,
        result: TaskResult,
        orchestrator: MagicMock,
        *,
        events: list[AgentExecutionEvent] | None = None,
    ):
        self._event_callback = None
        self._events = list(events or [])

        async def _run(*args, **kwargs):
            if self._event_callback is not None:
                for event in self._events:
                    self._event_callback(event)
            return result

        def _subscribe(_pattern, callback):
            self._event_callback = callback
            return lambda: None

        self.run = AsyncMock(side_effect=_run)
        self.subscribe_to_events = MagicMock(side_effect=_subscribe)
        self.get_orchestrator = MagicMock(return_value=orchestrator)


@pytest.mark.asyncio
async def test_execute_task_captures_task_report_metrics() -> None:
    orchestrator = MagicMock()
    orchestrator.get_last_task_report.return_value = {
        "task_id": "bench-1",
        "api_prompt_tokens": 40,
        "api_completion_tokens": 18,
        "api_total_tokens": 58,
        "request_count": 3,
        "cache_read_tokens": 12,
        "cache_hit_rate": 0.25,
        "tool_schema_tokens": 144,
        "compaction_saved_tokens": 33,
        "compaction_messages_removed": 2,
        "total_cost_usd": 0.0021,
        "metadata": {
            "continuation_ledger": "Intent: fix regression; Plan: inspect tests; patch parser",
        },
    }
    framework_result = TaskResult(
        content="patched",
        success=True,
        tool_calls=[{"name": "read"}],
        metadata={
            "tokens_input": 40,
            "tokens_output": 18,
            "turns": 3,
            "cached_tokens": 12,
            "reasoning_tokens": 5,
            "cost_usd_micros": 2100,
            "cache_hit_rate": 0.25,
            "tool_schema_tokens": 144,
            "compaction_saved_tokens": 33,
            "compaction_messages_removed": 2,
            "task_report": orchestrator.get_last_task_report.return_value,
        },
    )
    agent = BenchmarkAgent(
        _FakeFrameworkAgent(framework_result, orchestrator),
        BenchmarkAgentConfig(),
    )
    task = BenchmarkTask(
        task_id="bench-1",
        benchmark=BenchmarkType.CUSTOM,
        description="Fix the regression",
        prompt="fix the parser regression",
    )

    trace = await agent.execute_task(task)

    assert trace.tokens_input == 40
    assert trace.tokens_output == 18
    assert trace.cached_tokens == 12
    assert trace.reasoning_tokens == 5
    assert trace.cost_usd_micros == 2100
    assert trace.cache_hit_rate == pytest.approx(0.25)
    assert trace.tool_schema_tokens == 144
    assert trace.compaction_saved_tokens == 33
    assert trace.compaction_messages_removed == 2
    assert trace.task_report["metadata"]["continuation_ledger"].startswith("Intent:")
    assert trace.build_result_metadata()["task_report"]["task_id"] == "bench-1"


@pytest.mark.asyncio
async def test_execute_task_preserves_workspace_feedback_for_benchmark_exports() -> None:
    orchestrator = MagicMock()
    orchestrator.get_last_task_report.return_value = None
    framework_result = TaskResult(
        content="needs retry",
        success=False,
        tool_calls=[{"name": "read"}],
        metadata={
            "worktree_plan": {
                "team_name": "feature_team",
                "formation": "parallel",
                "assignments": [{"member_id": "tester", "claimed_paths": ["tests"]}],
            },
            "workspace_isolation_policy": {
                "mode": "delegate",
                "worktree_isolation": True,
                "materialize_worktrees": True,
                "cleanup_worktrees": False,
            },
            "delegate_follow_up_contract": {
                "next_action": "fix_validation",
                "preserve_worktrees": False,
                "workspace_isolation_diagnostics": [
                    {
                        "operation": "materialize",
                        "reason": "branch_exists",
                        "message": "branch already exists",
                    }
                ],
                "approval_contract": {
                    "required": True,
                    "reason": "validation_failed",
                    "recommended_action": "approve_retry",
                    "target_member_ids": ["tester"],
                },
            },
        },
    )
    agent = BenchmarkAgent(
        _FakeFrameworkAgent(framework_result, orchestrator),
        BenchmarkAgentConfig(),
    )
    task = BenchmarkTask(
        task_id="bench-workspace",
        benchmark=BenchmarkType.CUSTOM,
        description="Validate delegate work",
        prompt="run delegated validation",
    )

    trace = await agent.execute_task(task)

    summary = trace.team_feedback_summary
    metadata = trace.build_result_metadata()
    assert summary is not None
    assert summary["team_name"] == "feature_team"
    assert summary["workspace_policy_mode"] == "delegate"
    assert summary["workspace_policy_materialize_worktrees"] is True
    assert summary["workspace_diagnostic_count"] == 1
    assert summary["workspace_diagnostic_reasons"] == {"branch_exists": 1}
    assert metadata["team_feedback_summary"]["workspace_diagnostic_operations"] == {
        "materialize": 1
    }


@pytest.mark.asyncio
async def test_execute_task_tracks_time_to_first_tool_call_and_edit(
    monkeypatch,
) -> None:
    orchestrator = MagicMock()
    orchestrator.get_last_task_report.return_value = None
    framework_result = TaskResult(
        content="patched",
        success=True,
        tool_calls=[{"name": "read"}, {"name": "edit"}],
        metadata={},
    )
    agent = BenchmarkAgent(
        _FakeFrameworkAgent(
            framework_result,
            orchestrator,
            events=[
                AgentExecutionEvent(type=EventType.TOOL_CALL, tool_name="read"),
                AgentExecutionEvent(type=EventType.TOOL_CALL, tool_name="edit"),
            ],
        ),
        BenchmarkAgentConfig(),
    )
    task = BenchmarkTask(
        task_id="bench-2",
        benchmark=BenchmarkType.CUSTOM,
        description="Fix the regression",
        prompt="fix the parser regression",
    )

    timestamps = iter([100.0, 100.5, 101.25, 102.0])
    monkeypatch.setattr(
        "victor.benchmark.agent.time", SimpleNamespace(time=lambda: next(timestamps))
    )

    trace = await agent.execute_task(task)

    assert trace.time_to_first_tool_call_seconds == pytest.approx(0.5)
    assert trace.time_to_first_edit_seconds == pytest.approx(1.25)
    assert trace.first_edit_tool_name == "edit"
    assert trace.to_dict()["time_to_first_edit_seconds"] == pytest.approx(1.25)
    metadata = trace.build_result_metadata()
    assert metadata["time_to_first_tool_call_seconds"] == pytest.approx(0.5)
    assert metadata["time_to_first_edit_seconds"] == pytest.approx(1.25)
