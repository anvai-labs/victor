# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for GEPA v2 service, tier manager, and strategy adapter."""

from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.config.gepa_settings import GEPAModelSpec, GEPASettings
from victor.framework.rl.gepa_service import GEPAService
from victor.framework.rl.gepa_tier_manager import GEPATierManager


class TestGEPAService:
    def _make_service(self, response_text="Improved prompt text here."):
        provider = MagicMock()
        response = MagicMock()
        response.content = response_text

        async def mock_chat(**kwargs):
            return response

        provider.chat = mock_chat
        return GEPAService(
            provider=provider,
            model="test-model",
            tier="balanced",
            max_prompt_chars=1500,
        )

    def test_reflect_returns_result(self):
        service = self._make_service("- Fix file paths\n- Add context")
        result = service.reflect("traces here", "TEST_SECTION", "current text")
        assert "Fix file paths" in result

    def test_mutate_returns_result(self):
        service = self._make_service("Better prompt guidance.")
        result = service.mutate("old text", "reflection", "TEST", max_chars=1500)
        assert result == "Better prompt guidance."

    def test_mutate_hard_truncates(self):
        long_text = "x" * 2000
        service = self._make_service(long_text)
        result = service.mutate("old", "reflection", "TEST", max_chars=100)
        assert len(result) <= 100

    def test_mutate_fallback_on_failure(self):
        provider = MagicMock()

        async def mock_chat(**kwargs):
            raise RuntimeError("API error")

        provider.chat = mock_chat
        service = GEPAService(provider=provider, model="test", tier="balanced")
        result = service.mutate("original text", "reflection", "TEST")
        assert result == "original text"

    def test_merge_returns_result(self):
        service = self._make_service("Merged: best of both.")
        result = service.merge("prompt A", "prompt B", "TEST", max_chars=1500)
        assert "Merged" in result

    def test_get_tier(self):
        service = self._make_service()
        assert service.get_tier() == "balanced"

    def test_strip_thinking_blocks(self):
        text = "<think>internal reasoning</think>Actual output here."
        result = GEPAService._strip_thinking(text)
        assert "internal reasoning" not in result
        assert "Actual output" in result


class TestGEPATierManager:
    def _make_config(self, **overrides):
        defaults = {
            "default_tier": "balanced",
            "auto_tier_switch": True,
            "convergence_window": 3,
            "convergence_threshold": 0.02,
            "max_prompt_chars": 1500,
        }
        defaults.update(overrides)
        return GEPASettings(**defaults)

    def test_initial_tier(self):
        config = self._make_config()
        mgr = GEPATierManager(config)
        assert mgr.get_current_tier() == "balanced"

    def test_force_tier(self):
        config = self._make_config()
        mgr = GEPATierManager(config)
        mgr.force_tier("performance")
        assert mgr.get_current_tier() == "performance"

    def test_force_tier_invalid(self):
        config = self._make_config()
        mgr = GEPATierManager(config)
        with pytest.raises(ValueError):
            mgr.force_tier("invalid")

    def test_downgrade_on_convergence(self):
        config = self._make_config(convergence_window=3, convergence_threshold=0.05)
        mgr = GEPATierManager(config)
        assert mgr.get_current_tier() == "balanced"

        # Record 3 small deltas → convergence → downgrade
        for _ in range(3):
            mgr.record_evolution_delta("TEST", 0.01)
        assert mgr.get_current_tier() == "economic"

    def test_upgrade_on_regression(self):
        config = self._make_config(default_tier="economic")
        mgr = GEPATierManager(config)
        assert mgr.get_current_tier() == "economic"

        # Large negative delta → upgrade
        mgr.record_evolution_delta("TEST", -0.2)
        assert mgr.get_current_tier() == "balanced"

    def test_no_switch_when_auto_disabled(self):
        config = self._make_config(auto_tier_switch=False)
        mgr = GEPATierManager(config)
        for _ in range(10):
            mgr.record_evolution_delta("TEST", 0.001)
        assert mgr.get_current_tier() == "balanced"

    def test_metrics(self):
        config = self._make_config()
        mgr = GEPATierManager(config)
        mgr.record_evolution_delta("SEC1", 0.1)
        metrics = mgr.get_metrics()
        assert metrics["current_tier"] == "balanced"
        assert metrics["total_evolutions"] == 1
        assert "SEC1" in metrics["convergence_windows"]


class TestGEPAStrategyAdapter:
    def test_format_traces_as_asi_with_details(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy
        from victor.framework.rl.learners.prompt_optimizer import (
            ExecutionTrace,
            ToolCallTrace,
        )

        mock_mgr = MagicMock()
        adapter = GEPAServiceStrategy(mock_mgr)

        detail = ToolCallTrace(
            tool_name="edit",
            arguments_summary="path='foo.py'",
            reasoning_before="I need to fix the import",
            success=False,
            error_detail="old_str not found",
            duration_ms=45.0,
        )
        trace = ExecutionTrace(
            session_id="s1",
            task_type="action",
            provider="ollama",
            model="qwen3",
            tool_calls=1,
            tool_failures={"edit_mismatch": 1},
            success=False,
            completion_score=0.3,
            tokens_used=500,
            tool_call_details=[detail],
        )

        result = adapter._format_traces_as_asi([trace])
        assert "edit" in result
        assert "FAILED" in result
        assert "old_str not found" in result
        assert "fix the import" in result

    def test_format_traces_fallback_for_v1(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy
        from victor.framework.rl.learners.prompt_optimizer import ExecutionTrace

        mock_mgr = MagicMock()
        adapter = GEPAServiceStrategy(mock_mgr)

        trace = ExecutionTrace(
            session_id="s2",
            task_type="action",
            provider="ollama",
            model="qwen3",
            tool_calls=5,
            tool_failures={"file_not_found": 2},
            success=False,
            completion_score=0.4,
            tokens_used=800,
        )

        result = adapter._format_traces_as_asi([trace])
        assert "Tool calls: 5" in result
        assert "file_not_found" in result

    def test_categorize_tool(self):
        from victor.framework.rl.gepa_strategy_adapter import categorize_tool

        assert categorize_tool("read") == "exploration"
        assert categorize_tool("edit") == "mutation"
        assert categorize_tool("bash") == "execution"
        assert categorize_tool("list_directory") == "exploration"
        assert categorize_tool("create_file") == "mutation"
        assert categorize_tool("execute_bash") == "execution"
        assert categorize_tool("graph") == "analysis"
        assert categorize_tool("unknown_tool") == "general"
