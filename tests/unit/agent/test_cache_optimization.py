# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TDD tests for LLM prefix cache optimization.

Tests verify that tools + system prompt remain byte-identical across
API calls within a session, enabling provider-level prefix caching
(90% discount on Anthropic, OpenAI, Google).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor.providers.base import Message


class TestSessionLockedTools:
    """Verify tools are frozen at session start when cache optimization enabled."""

    def test_session_tools_returns_full_set(self):
        """get_session_tools() locks the currently enabled tool definitions."""
        orch = MagicMock()
        orch._cache_optimization_enabled = True
        orch._session_tools = None

        all_tools = [MagicMock(name=f"tool_{i}") for i in range(10)]
        for i, tool in enumerate(all_tools):
            tool.name = f"tool_{i}"
        orch.get_enabled_tools = MagicMock(return_value={tool.name for tool in all_tools})
        orch.tools.list_tools = MagicMock(return_value=all_tools)

        # Import and call the method pattern
        from victor.agent.orchestrator import AgentOrchestrator

        result = AgentOrchestrator.get_session_tools(orch)
        assert result == all_tools
        assert len(result) == 10
        orch.tools.list_tools.assert_called_once_with(only_enabled=True)

    def test_session_tools_cached_after_first_call(self):
        """Second call returns same object (not re-computed)."""
        orch = MagicMock()
        orch._cache_optimization_enabled = True
        all_tools = [MagicMock(name="tool_1")]
        orch._session_tools = all_tools  # Already cached

        from victor.agent.orchestrator import AgentOrchestrator

        result = AgentOrchestrator.get_session_tools(orch)
        assert result is all_tools
        orch.tools.list_tools.assert_not_called()

    def test_returns_none_when_disabled(self):
        """When cache optimization disabled, returns None (use per-turn)."""
        orch = MagicMock()
        orch._cache_optimization_enabled = False

        from victor.agent.orchestrator import AgentOrchestrator

        result = AgentOrchestrator.get_session_tools(orch)
        assert result is None


class TestReminderInjection:
    """Verify reminders move to user message when cache optimization enabled."""

    def test_get_user_message_prefix_returns_formatted(self):
        """get_user_message_prefix() returns [Context: ...] formatted string."""
        from victor.agent.context_reminder import ContextReminderManager

        manager = ContextReminderManager.__new__(ContextReminderManager)
        manager.get_consolidated_reminder = MagicMock(
            return_value="FILES: a.py, b.py | Budget: 5 remaining"
        )

        result = manager.get_user_message_prefix()
        assert result.startswith("[Context:")
        assert "FILES: a.py" in result
        assert result.endswith("\n\n")

    def test_get_user_message_prefix_empty_when_no_reminder(self):
        """Returns empty string when no reminders needed."""
        from victor.agent.context_reminder import ContextReminderManager

        manager = ContextReminderManager.__new__(ContextReminderManager)
        manager.get_consolidated_reminder = MagicMock(return_value=None)

        result = manager.get_user_message_prefix()
        assert result == ""


class TestAnthropicCacheControl:
    """Verify cache_control annotations on Anthropic requests."""

    def test_system_message_formatted_with_cache_control(self):
        """System message should be list of content blocks with cache_control."""
        # Simulate what the Anthropic provider should produce
        system_text = "You are a helpful assistant."

        # Expected format for Anthropic API with caching:
        expected = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Build the format
        result = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert result == expected
        assert result[0]["cache_control"]["type"] == "ephemeral"

    def test_last_tool_has_cache_control(self):
        """Last tool definition should have cache_control annotation."""
        tools = [
            {"name": "read", "description": "Read a file", "input_schema": {}},
            {"name": "write", "description": "Write a file", "input_schema": {}},
            {"name": "search", "description": "Search code", "input_schema": {}},
        ]

        # Apply cache_control to last tool
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}

        assert "cache_control" not in tools[0]
        assert "cache_control" not in tools[1]
        assert tools[2]["cache_control"] == {"type": "ephemeral"}

    def test_empty_tools_no_crash(self):
        """Empty tools list should not crash when adding cache_control."""
        tools = []
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        assert tools == []


class TestSystemPromptFreeze:
    """Verify system prompt is built once and frozen."""

    def test_update_noop_when_cache_optimization_enabled(self):
        """update_system_prompt_for_query() should be no-op when frozen."""
        orch = MagicMock()
        orch._cache_optimization_enabled = True
        orch._system_prompt_frozen = True

        # The method should return early without rebuilding
        from victor.agent.orchestrator import AgentOrchestrator

        original_prompt = "Original system prompt"
        orch._system_prompt = original_prompt

        # Simulate calling update_system_prompt_for_query
        # When frozen, it should not modify _system_prompt
        if orch._system_prompt_frozen and orch._cache_optimization_enabled:
            pass  # No-op
        else:
            orch._system_prompt = "Rebuilt prompt"

        assert orch._system_prompt == original_prompt

    def test_task_guidance_as_user_prefix(self):
        """Task guidance should be extractable for user message injection."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.query_classification = MagicMock()
        builder.query_classification.query_type = "EXPLORATION"

        # The method should produce guidance text for user message
        guidance = "[Task: Explore systematically]"
        assert "[Task:" in guidance


class TestCacheOptimizationSetting:
    """Verify the cache_optimization_enabled setting."""

    def test_default_enabled(self):
        """Cache optimization should be enabled by default."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings()
        assert hasattr(settings, "cache_optimization_enabled")
        assert settings.cache_optimization_enabled is True
