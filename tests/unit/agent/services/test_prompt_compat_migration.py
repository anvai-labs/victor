# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TDD tests for prompt_compat migration to canonical API.

Tests that:
1. PromptRuntimeAdapter from prompt_runtime has all functionality from PromptCoordinator
2. PromptRuntimeContext has all fields from TaskContext
3. PromptRuntimeConfig has all fields from PromptCoordinatorConfig
4. system_prompt_policy can use PromptRuntimeContext directly
5. Existing imports can be updated to use canonical locations
"""

from __future__ import annotations

import pytest

from victor.agent.services.prompt_runtime import (
    PromptRuntimeAdapter,
    PromptRuntimeConfig,
    PromptRuntimeContext,
)
from victor.agent.system_prompt_policy import SystemPromptPolicy


class TestPromptRuntimeContextParity:
    """Verify PromptRuntimeContext has all TaskContext fields."""

    def test_message_field(self):
        """TaskContext.message -> PromptRuntimeContext.message"""
        context = PromptRuntimeContext(message="test message")
        assert context.message == "test message"

    def test_task_type_field(self):
        """TaskContext.task_type -> PromptRuntimeContext.task_type"""
        context = PromptRuntimeContext(message="test", task_type="bugfix")
        assert context.task_type == "bugfix"

    def test_complexity_field(self):
        """TaskContext.complexity -> PromptRuntimeContext.complexity"""
        context = PromptRuntimeContext(message="test", complexity="high")
        assert context.complexity == "high"

    def test_stage_field(self):
        """TaskContext.stage -> PromptRuntimeContext.stage"""
        context = PromptRuntimeContext(message="test", stage="execution")
        assert context.stage == "execution"

    def test_model_field(self):
        """TaskContext.model -> PromptRuntimeContext.model"""
        context = PromptRuntimeContext(message="test", model="claude-opus-4")
        assert context.model == "claude-opus-4"

    def test_provider_field(self):
        """TaskContext.provider -> PromptRuntimeContext.provider"""
        context = PromptRuntimeContext(message="test", provider="anthropic")
        assert context.provider == "anthropic"

    def test_additional_context_field(self):
        """TaskContext.additional_context -> PromptRuntimeContext.additional_context"""
        context = PromptRuntimeContext(message="test", additional_context={"project": "victor"})
        assert context.additional_context == {"project": "victor"}

    def test_default_values_match(self):
        """Default values match between TaskContext and PromptRuntimeContext."""
        context = PromptRuntimeContext(message="test")
        assert context.task_type == "unknown"
        assert context.complexity == "medium"
        assert context.stage is None
        assert context.model is None
        assert context.provider is None


class TestPromptRuntimeConfigParity:
    """Verify PromptRuntimeConfig has all PromptCoordinatorConfig fields."""

    def test_default_grounding_mode(self):
        """PromptCoordinatorConfig.default_grounding_mode -> PromptRuntimeConfig.default_grounding_mode"""
        config = PromptRuntimeConfig()
        assert config.default_grounding_mode == "minimal"

    def test_enable_task_hints(self):
        """PromptCoordinatorConfig.enable_task_hints -> PromptRuntimeConfig.enable_task_hints"""
        config = PromptRuntimeConfig()
        assert config.enable_task_hints is True

    def test_enable_vertical_sections(self):
        """PromptCoordinatorConfig.enable_vertical_sections -> PromptRuntimeConfig.enable_vertical_sections"""
        config = PromptRuntimeConfig()
        assert config.enable_vertical_sections is True

    def test_enable_safety_rules(self):
        """PromptCoordinatorConfig.enable_safety_rules -> PromptRuntimeConfig.enable_safety_rules"""
        config = PromptRuntimeConfig()
        assert config.enable_safety_rules is True

    def test_max_context_tokens(self):
        """PromptCoordinatorConfig.max_context_tokens -> PromptRuntimeConfig.max_context_tokens"""
        config = PromptRuntimeConfig()
        assert config.max_context_tokens == 2000

    def test_custom_config(self):
        """Custom config values work the same way."""
        config = PromptRuntimeConfig(
            default_grounding_mode="extended",
            enable_task_hints=False,
            max_context_tokens=4000,
        )
        assert config.default_grounding_mode == "extended"
        assert config.enable_task_hints is False
        assert config.max_context_tokens == 4000


class TestPromptRuntimeAdapterParity:
    """Verify PromptRuntimeAdapter has all PromptCoordinator methods."""

    def test_build_system_prompt_method(self):
        """PromptCoordinator.build_system_prompt -> PromptRuntimeAdapter.build_system_prompt"""
        adapter = PromptRuntimeAdapter()
        context = PromptRuntimeContext(message="test")
        prompt = adapter.build_system_prompt(context)
        assert isinstance(prompt, str)
        assert "Victor" in prompt  # default identity

    def test_add_task_hint_method(self):
        """PromptCoordinator.add_task_hint -> PromptRuntimeAdapter.add_task_hint"""
        adapter = PromptRuntimeAdapter()
        adapter.add_task_hint("edit", "Read the file first")
        assert adapter.get_task_hint("edit") == "Read the file first"

    def test_get_task_hint_method(self):
        """PromptCoordinator.get_task_hint -> PromptRuntimeAdapter.get_task_hint"""
        adapter = PromptRuntimeAdapter()
        adapter.add_task_hint("debug", "Check logs")
        assert adapter.get_task_hint("debug") == "Check logs"
        assert adapter.get_task_hint("unknown") is None

    def test_remove_task_hint_method(self):
        """PromptCoordinator.remove_task_hint -> PromptRuntimeAdapter.remove_task_hint"""
        adapter = PromptRuntimeAdapter()
        adapter.add_task_hint("edit", "Hint")
        adapter.remove_task_hint("edit")
        assert adapter.get_task_hint("edit") is None

    def test_add_section_method(self):
        """PromptCoordinator.add_section -> PromptRuntimeAdapter.add_section"""
        adapter = PromptRuntimeAdapter()
        adapter.add_section("custom", "Custom content")
        assert "custom" in adapter._additional_sections

    def test_remove_section_method(self):
        """PromptCoordinator.remove_section -> PromptRuntimeAdapter.remove_section"""
        adapter = PromptRuntimeAdapter()
        adapter.add_section("custom", "Content")
        adapter.remove_section("custom")
        assert "custom" not in adapter._additional_sections

    def test_add_safety_rule_method(self):
        """PromptCoordinator.add_safety_rule -> PromptRuntimeAdapter.add_safety_rule"""
        adapter = PromptRuntimeAdapter()
        adapter.add_safety_rule("Never expose credentials")
        assert "Never expose credentials" in adapter._safety_rules

    def test_clear_safety_rules_method(self):
        """PromptCoordinator.clear_safety_rules -> PromptRuntimeAdapter.clear_safety_rules"""
        adapter = PromptRuntimeAdapter()
        adapter.add_safety_rule("Rule 1")
        adapter.add_safety_rule("Rule 2")
        adapter.clear_safety_rules()
        assert len(adapter._safety_rules) == 0

    def test_set_grounding_mode_method(self):
        """PromptCoordinator.set_grounding_mode -> PromptRuntimeAdapter.set_grounding_mode"""
        adapter = PromptRuntimeAdapter()
        adapter.set_grounding_mode("extended")
        assert adapter._config.default_grounding_mode == "extended"

    def test_set_base_identity_method(self):
        """PromptCoordinator.set_base_identity -> PromptRuntimeAdapter.set_base_identity"""
        adapter = PromptRuntimeAdapter()
        adapter.set_base_identity("New identity")
        assert adapter._base_identity == "New identity"

    def test_get_all_task_hints_method(self):
        """PromptCoordinator.get_all_task_hints -> PromptRuntimeAdapter.get_all_task_hints"""
        adapter = PromptRuntimeAdapter()
        adapter.add_task_hint("edit", "Hint 1")
        adapter.add_task_hint("debug", "Hint 2")
        hints = adapter.get_all_task_hints()
        assert len(hints) == 2
        assert hints["edit"] == "Hint 1"

    def test_clear_method(self):
        """PromptCoordinator.clear -> PromptRuntimeAdapter.clear"""
        adapter = PromptRuntimeAdapter()
        adapter.add_task_hint("edit", "Hint")
        adapter.add_section("custom", "Content")
        adapter.add_safety_rule("Rule")
        adapter.clear()
        assert adapter._task_hints == {}
        assert adapter._additional_sections == {}
        assert adapter._safety_rules == []

    def test_vertical_context_property(self):
        """PromptCoordinator.vertical_context -> PromptRuntimeAdapter.vertical_context"""
        adapter = PromptRuntimeAdapter()
        from unittest.mock import MagicMock

        mock_context = MagicMock()
        adapter.vertical_context = mock_context
        assert adapter.vertical_context == mock_context


class TestSystemPromptPolicyUsesPromptRuntimeContext:
    """Verify SystemPromptPolicy works with PromptRuntimeContext directly."""

    def test_policy_enforce_with_prompt_runtime_context(self):
        """SystemPromptPolicy.enforce accepts PromptRuntimeContext."""
        from victor.framework.prompt_builder import PromptBuilder

        policy = SystemPromptPolicy()
        builder = PromptBuilder()
        context = PromptRuntimeContext(
            message="test",
            task_type="bugfix",
            stage="execution",
            model="claude-opus-4",
            provider="anthropic",
        )

        # Should not raise
        policy.enforce(builder, context)
        result = builder.build()
        assert "Victor" in result

    def test_policy_build_fallback_with_prompt_runtime_context(self):
        """SystemPromptPolicy.build_fallback_prompt accepts PromptRuntimeContext."""
        policy = SystemPromptPolicy()
        context = PromptRuntimeContext(
            message="Fix the bug",
            task_type="debug",
        )

        fallback = policy.build_fallback_prompt(context)
        assert "Fix the bug" in fallback
        assert "debug" in fallback


class TestMigrationPath:
    """Tests for the actual migration path."""

    def test_prompt_runtime_context_is_replacement(self):
        """PromptRuntimeContext is the direct replacement for TaskContext."""
        # Both are dataclasses with same fields
        context = PromptRuntimeContext(
            message="test",
            task_type="bugfix",
            complexity="high",
        )
        assert context.message == "test"
        assert context.task_type == "bugfix"
        assert context.complexity == "high"

    def test_prompt_runtime_adapter_is_replacement(self):
        """PromptRuntimeAdapter is the direct replacement for PromptCoordinator."""
        adapter = PromptRuntimeAdapter(base_identity="You are Victor.")
        assert adapter._base_identity == "You are Victor."
        assert adapter._config.default_grounding_mode == "minimal"
