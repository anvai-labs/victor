# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.session_cost_tracker import SessionCostTracker
from victor.agent.services.metrics_service import AgentMetricsService


def _make_metrics_coordinator() -> AgentMetricsService:
    return AgentMetricsService(
        metrics_collector=MagicMock(),
        session_cost_tracker=MagicMock(),
        cumulative_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )


def test_emit_tool_strategy_event_uses_provider_capability_for_metrics_labels(caplog):
    coordinator = _make_metrics_coordinator()
    provider = MagicMock()
    provider.name = "openai"
    provider.supports_prompt_caching.return_value = True
    tools = [SimpleNamespace(name="read"), SimpleNamespace(name="git_diff")]

    with caplog.at_level("DEBUG"):
        coordinator.emit_tool_strategy_event(
            strategy="session_lock",
            tool_count=2,
            tool_tokens=512,
            context_window=32768,
            provider=provider,
            model="test-model",
            reason="cache_discount_or_large_context",
            tools=tools,
            v2_enabled=True,
        )

    assert "category=caching" in caplog.text
    assert 'victor_tool_strategy_v2_enabled{provider="openai"} 1' in caplog.text
    assert 'victor_tool_tier_count{provider="openai",tier="FULL"} 1' in caplog.text


def test_emit_tool_strategy_event_falls_back_to_provider_name_for_strings(caplog):
    coordinator = _make_metrics_coordinator()

    with caplog.at_level("INFO"):
        coordinator.emit_tool_strategy_event(
            strategy="semantic_selection",
            tool_count=1,
            tool_tokens=64,
            context_window=8192,
            provider="ollama",
            model="test-model",
            reason="small_context_window",
            tools=[SimpleNamespace(name="read")],
            v2_enabled=False,
        )

    assert "provider=ollama, category=local" in caplog.text


def test_emit_tool_strategy_event_records_last_event():
    coordinator = _make_metrics_coordinator()

    coordinator.emit_tool_strategy_event(
        strategy="semantic_selection",
        tool_count=1,
        tool_tokens=64,
        context_window=8192,
        provider="ollama",
        model="test-model",
        reason="small_context_window",
        tools=[SimpleNamespace(name="read")],
        v2_enabled=False,
    )

    assert coordinator.get_last_tool_strategy_event()["tool_tokens"] == 64


def test_task_report_captures_token_deltas_and_success_average():
    cumulative = {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "cached_tokens": 10,
    }
    tracker = SessionCostTracker(provider="unknown", model="test-model")
    coordinator = AgentMetricsService(
        metrics_collector=MagicMock(),
        session_cost_tracker=tracker,
        cumulative_token_usage=cumulative,
    )

    coordinator.start_task_report(
        "Implement canonical runtime path",
        task_type="edit",
        metadata={"mode": "build"},
    )
    cumulative["prompt_tokens"] += 60
    cumulative["completion_tokens"] += 30
    cumulative["total_tokens"] += 90
    cumulative["cached_tokens"] += 20
    tracker.record_request(
        prompt_tokens=60,
        completion_tokens=30,
        cache_read_tokens=20,
        cache_write_tokens=5,
        duration_seconds=1.0,
    )

    report = coordinator.finish_task_report(
        True,
        compaction={
            "occurred": True,
            "saved_tokens": 45,
            "messages_removed": 3,
            "reason": "pre_tool_output",
            "policy_reason": "tool_output_exceeds_remaining_budget",
        },
        tool_schema_tokens=256,
    )

    assert report["task_type"] == "edit"
    assert report["api_total_tokens"] == 90
    assert report["api_prompt_tokens"] == 60
    assert report["api_completion_tokens"] == 30
    assert report["cache_read_tokens"] == 20
    assert report["cache_write_tokens"] == 5
    assert report["cache_hit_rate"] == pytest.approx(0.25)
    assert report["compaction_saved_tokens"] == 45
    assert report["compaction_messages_removed"] == 3
    assert report["tool_schema_tokens"] == 256
    assert report["tokens_per_successful_task"] == pytest.approx(90.0)
    assert report["metadata"]["compaction_reason"] == "pre_tool_output"
    assert report["metadata"]["compaction_policy_reason"] == "tool_output_exceeds_remaining_budget"
    assert coordinator.get_last_task_report()["task_id"] == report["task_id"]


def test_task_report_promotes_workspace_policy_and_diagnostics():
    coordinator = AgentMetricsService(
        metrics_collector=MagicMock(),
        session_cost_tracker=SessionCostTracker(provider="unknown", model="test-model"),
        cumulative_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )

    coordinator.start_task_report("Delegate isolated work", metadata={"mode": "delegate"})

    report = coordinator.finish_task_report(
        False,
        metadata={
            "workspace_isolation_policy": {
                "mode": "delegate",
                "worktree_isolation": True,
                "materialize_worktrees": True,
                "dry_run_worktrees": False,
                "auto_merge_worktrees": False,
                "cleanup_worktrees": False,
            },
            "workspace_isolation_diagnostics": [
                {
                    "operation": "materialize",
                    "reason": "branch_exists",
                    "message": "branch already exists",
                    "details": {"member_id": "tester"},
                }
            ],
        },
    )

    metadata = report["metadata"]
    assert metadata["workspace_policy_mode"] == "delegate"
    assert metadata["workspace_policy_materialize_worktrees"] is True
    assert metadata["workspace_policy_cleanup_worktrees"] is False
    assert metadata["workspace_isolation_diagnostic_count"] == 1
    assert metadata["workspace_isolation_diagnostic_reasons"] == {"branch_exists": 1}
    assert metadata["workspace_isolation_diagnostic_operations"] == {"materialize": 1}


def test_task_report_history_respects_limit():
    tracker = SessionCostTracker(provider="unknown", model="test-model")
    coordinator = AgentMetricsService(
        metrics_collector=MagicMock(),
        session_cost_tracker=tracker,
        cumulative_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )

    for index in range(3):
        coordinator.start_task_report(f"task-{index}")
        coordinator.finish_task_report(True)

    history = coordinator.get_task_report_history(limit=2)

    assert len(history) == 2
    assert history[0]["description"] == "task-1"
    assert history[1]["description"] == "task-2"
