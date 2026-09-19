"""Turn-frame ownership contracts for the canonical chat service."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.chat_turn_runtime import ChatTurnRuntime


class _State:
    provider_name = "anthropic"
    model = "claude-sonnet"
    stream_context = None
    unified_tracker = None
    task_type_candidates = (None, None)
    context_service = None

    def __init__(self) -> None:
        self.metrics = MagicMock()
        self.metrics.start_task_report.return_value = "task-1"
        self.metrics.get_last_tool_strategy_event.return_value = {"tool_tokens": 144}
        self.metrics.finish_task_report.return_value = {"task_id": "task-1"}
        self.apply_skill_for_turn = MagicMock()
        self.activate_constraints = MagicMock()
        self.deactivate_constraints = MagicMock()
        self.assign_turn_credit = MagicMock()


def test_turn_runtime_owns_setup_and_teardown_as_one_immutable_capability() -> None:
    state = _State()
    runtime = ChatTurnRuntime(state)
    constraints = object()

    runtime.enter("inspect app.py", stream=True, constraints=constraints, vertical="review")
    runtime.exit("inspect app.py", stream=True, constraints=constraints, vertical="review")

    state.apply_skill_for_turn.assert_called_once_with("inspect app.py")
    state.activate_constraints.assert_called_once_with(constraints, "review")
    state.deactivate_constraints.assert_called_once_with()
    state.assign_turn_credit.assert_called_once_with()
    assert not hasattr(runtime, "__dict__")
    with pytest.raises(FrozenInstanceError):
        runtime.state = _State()


def test_turn_runtime_skips_stream_only_and_optional_scope_work() -> None:
    state = _State()
    runtime = ChatTurnRuntime(state)

    runtime.enter("hello", stream=False)
    runtime.exit("hello", stream=False)

    state.apply_skill_for_turn.assert_not_called()
    state.activate_constraints.assert_not_called()
    state.deactivate_constraints.assert_not_called()
    state.assign_turn_credit.assert_called_once_with()


def test_turn_runtime_builds_start_and_finish_report_metadata() -> None:
    state = _State()
    state.stream_context = SimpleNamespace(
        unified_task_type=SimpleNamespace(value="edit"),
        task_intent="Fix the parser regression",
        plan_steps=["Inspect tests", "Patch parser"],
        intent_log=[],
        resume_summary="Verify the patch",
        degraded_resume_state=True,
        build_continuation_ledger=MagicMock(return_value="Intent: fix parser"),
        compaction_summary="summarized",
        compaction_occurred=True,
        last_compaction_turn=4,
        compaction_message_removed_count=2,
        last_compaction_strategy="hybrid",
        last_compaction_reason="pre_tool_output",
        last_compaction_policy_reason="tool_output_exceeds_remaining_budget",
    )
    state.context_service = MagicMock()
    state.context_service.get_performance_metrics.return_value = {}
    runtime = ChatTurnRuntime(state)

    assert (
        runtime.start_task_report(
            "Fix the parser regression", stream=True, metadata={"source": "test"}
        )
        == "task-1"
    )
    assert runtime.finish_task_report(
        True,
        user_message="Fix the parser regression",
        stream=True,
        metadata={"source": "test"},
    ) == {"task_id": "task-1"}

    start = state.metrics.start_task_report.call_args.kwargs
    assert start["task_type"] == "edit"
    assert start["metadata"] == {
        "stream": True,
        "provider": "anthropic",
        "model": "claude-sonnet",
        "source": "test",
    }

    finish = state.metrics.finish_task_report.call_args.kwargs
    assert finish["task_type"] == "edit"
    assert finish["tool_schema_tokens"] == 144
    assert finish["metadata"]["continuation_ledger"] == "Intent: fix parser"
    assert finish["metadata"]["degraded_resume_state"] is True
    assert finish["compaction"]["occurred"] is True
