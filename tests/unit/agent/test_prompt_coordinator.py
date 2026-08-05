# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PromptCoordinator.

Tests the prompt coordination functionality including:
- System prompt building
- Task hint management
- Vertical integration
"""

import pytest
from unittest.mock import MagicMock

# MIGRATION: Import from canonical prompt_runtime instead of deprecated prompt_compat
from victor.agent.services.prompt_runtime import (
    PromptRuntimeAdapter as PromptCoordinator,
    PromptRuntimeConfig as PromptCoordinatorConfig,
    PromptRuntimeContext as TaskContext,
    create_prompt_coordinator,
)
from victor.framework.prompt_builder import PromptBuilder


class TestPromptCoordinatorConfig:
    """Tests for PromptCoordinatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PromptCoordinatorConfig()

        assert config.default_grounding_mode == "minimal"
        assert config.enable_task_hints is True
        assert config.enable_vertical_sections is True
        assert config.enable_safety_rules is True
        assert config.max_context_tokens == 2000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PromptCoordinatorConfig(
            default_grounding_mode="extended",
            enable_task_hints=False,
            max_context_tokens=4000,
        )

        assert config.default_grounding_mode == "extended"
        assert config.enable_task_hints is False
        assert config.max_context_tokens == 4000


class TestTaskContext:
    """Tests for TaskContext."""

    def test_default_context(self):
        """Test default context values."""
        context = TaskContext(message="test message")

        assert context.message == "test message"
        assert context.task_type == "unknown"
        assert context.complexity == "medium"
        assert context.stage is None
        assert context.model is None
        assert context.provider is None

    def test_custom_context(self):
        """Test custom context values."""
        context = TaskContext(
            message="fix the authentication bug",
            task_type="bugfix",
            complexity="high",
            stage="execution",
            model="claude-opus-4",
            provider="anthropic",
            additional_context={"project": "victor"},
        )

        assert context.message == "fix the authentication bug"
        assert context.task_type == "bugfix"
        assert context.model == "claude-opus-4"
        assert context.additional_context["project"] == "victor"


class TestPromptCoordinator:
    """Tests for PromptCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with default config."""
        return PromptCoordinator()

    @pytest.fixture
    def coordinator_with_identity(self):
        """Create coordinator with base identity."""
        return PromptCoordinator(base_identity="You are Victor, an AI coding assistant.")

    def test_init_default(self, coordinator):
        """Test default initialization."""
        assert coordinator._config.default_grounding_mode == "minimal"
        assert coordinator._base_identity is None
        assert coordinator._task_hints == {}

    def test_init_with_identity(self, coordinator_with_identity):
        """Test initialization with base identity."""
        assert coordinator_with_identity._base_identity == "You are Victor, an AI coding assistant."

    def test_build_system_prompt_empty(self, coordinator):
        """Test building empty prompt."""
        context = TaskContext(message="test")
        prompt = coordinator.build_system_prompt(context)

        # Should have grounding rules at minimum
        assert isinstance(prompt, str)
        assert "Victor" in prompt  # default identity injected by policy

    def test_build_system_prompt_with_identity(self, coordinator_with_identity):
        """Test building prompt with identity."""
        context = TaskContext(message="test")
        prompt = coordinator_with_identity.build_system_prompt(context)

        assert "Victor" in prompt

    def test_add_task_hint(self, coordinator):
        """Test adding task hints."""
        coordinator.add_task_hint("edit", "Read the file first before editing")

        assert "edit" in coordinator._task_hints
        assert coordinator._task_hints["edit"] == "Read the file first before editing"

    def test_add_task_hint_lowercase(self, coordinator):
        """Test that task hints are normalized to lowercase."""
        coordinator.add_task_hint("EDIT", "Read the file first")

        assert "edit" in coordinator._task_hints

    def test_get_task_hint(self, coordinator):
        """Test getting task hints."""
        coordinator.add_task_hint("debug", "Check logs first")

        assert coordinator.get_task_hint("debug") == "Check logs first"
        assert coordinator.get_task_hint("DEBUG") == "Check logs first"
        assert coordinator.get_task_hint("unknown") is None

    def test_remove_task_hint(self, coordinator):
        """Test removing task hints."""
        coordinator.add_task_hint("edit", "Test hint")
        coordinator.remove_task_hint("edit")

        assert coordinator.get_task_hint("edit") is None

    def test_add_section(self, coordinator):
        """Test adding runtime sections."""
        coordinator.add_section("custom", "Custom section content")

        assert "custom" in coordinator._additional_sections

    def test_remove_section(self, coordinator):
        """Test removing runtime sections."""
        coordinator.add_section("custom", "Content")
        coordinator.remove_section("custom")

        assert "custom" not in coordinator._additional_sections

    def test_add_safety_rule(self, coordinator):
        """Test adding safety rules."""
        coordinator.add_safety_rule("Never expose credentials")
        coordinator.add_safety_rule("Confirm before destructive operations")

        assert len(coordinator._safety_rules) == 2
        assert "Never expose credentials" in coordinator._safety_rules

    def test_add_safety_rule_no_duplicates(self, coordinator):
        """Test that duplicate safety rules are not added."""
        coordinator.add_safety_rule("Never expose credentials")
        coordinator.add_safety_rule("Never expose credentials")

        assert len(coordinator._safety_rules) == 1

    def test_clear_safety_rules(self, coordinator):
        """Test clearing safety rules."""
        coordinator.add_safety_rule("Rule 1")
        coordinator.add_safety_rule("Rule 2")
        coordinator.clear_safety_rules()

        assert len(coordinator._safety_rules) == 0

    def test_set_grounding_mode_minimal(self, coordinator):
        """Test setting minimal grounding mode."""
        coordinator.set_grounding_mode("minimal")

        assert coordinator._config.default_grounding_mode == "minimal"

    def test_set_grounding_mode_extended(self, coordinator):
        """Test setting extended grounding mode."""
        coordinator.set_grounding_mode("extended")

        assert coordinator._config.default_grounding_mode == "extended"

    def test_set_grounding_mode_invalid(self, coordinator):
        """Test setting invalid grounding mode."""
        original = coordinator._config.default_grounding_mode
        coordinator.set_grounding_mode("invalid")

        # Should keep original
        assert coordinator._config.default_grounding_mode == original

    def test_set_base_identity(self, coordinator):
        """Test setting base identity."""
        coordinator.set_base_identity("New identity")

        assert coordinator._base_identity == "New identity"

    def test_get_all_task_hints(self, coordinator):
        """Test getting all task hints."""
        coordinator.add_task_hint("edit", "Hint 1")
        coordinator.add_task_hint("debug", "Hint 2")

        hints = coordinator.get_all_task_hints()

        assert len(hints) == 2
        assert hints["edit"] == "Hint 1"
        assert hints["debug"] == "Hint 2"

    def test_clear(self, coordinator):
        """Test clearing all state."""
        coordinator.add_task_hint("edit", "Hint")
        coordinator.add_section("custom", "Content")
        coordinator.add_safety_rule("Rule")

        coordinator.clear()

        assert coordinator._task_hints == {}
        assert coordinator._additional_sections == {}
        assert coordinator._safety_rules == []

    def test_vertical_context_property(self, coordinator):
        """Test vertical context property getter/setter."""
        mock_context = MagicMock()
        coordinator.vertical_context = mock_context

        assert coordinator.vertical_context == mock_context

    def test_build_prompt_with_context(self, coordinator):
        """Test building prompt with additional context."""
        context = TaskContext(
            message="test",
            additional_context={"project": "victor", "language": "python"},
        )

        prompt = coordinator.build_system_prompt(context)

        assert isinstance(prompt, str)
        assert "project: victor" in prompt.lower()

    def test_on_prompt_built_callback(self):
        """Test prompt built callback."""
        callback_calls = []

        def on_built(prompt, context):
            callback_calls.append((prompt, context))

        coordinator = PromptCoordinator(on_prompt_built=on_built)
        context = TaskContext(message="test")
        coordinator.build_system_prompt(context)

        assert len(callback_calls) == 1
        assert callback_calls[0][1] == context

    def test_build_system_prompt_respects_context_budget(self):
        """Ensure context strings are truncated to the configured budget."""
        config = PromptCoordinatorConfig(max_context_tokens=20)
        coordinator = PromptCoordinator(config=config)
        context = TaskContext(message="test", additional_context={"long": "x" * 100})

        prompt = coordinator.build_system_prompt(context)

        assert "x" * 21 not in prompt

    def test_build_system_prompt_fallback_on_error(self, monkeypatch):
        """PromptCoordinator should fall back when builder raises."""

        def boom(_self):
            raise RuntimeError("boom")

        monkeypatch.setattr("victor.framework.prompt_builder.PromptBuilder.build", boom)

        coordinator = PromptCoordinator()
        prompt = coordinator.build_system_prompt(TaskContext(message="fail gracefully"))

        assert "fail gracefully" in prompt
        assert "Victor" in prompt


class TestCreatePromptCoordinator:
    """Tests for create_prompt_coordinator factory function."""

    def test_create_basic(self):
        """Test basic factory creation."""
        coordinator = create_prompt_coordinator()

        assert isinstance(coordinator, PromptCoordinator)
        assert coordinator._config.default_grounding_mode == "minimal"

    def test_create_with_config(self):
        """Test factory creation with config."""
        config = PromptCoordinatorConfig(default_grounding_mode="extended")

        coordinator = create_prompt_coordinator(config=config)

        assert coordinator._config.default_grounding_mode == "extended"

    def test_create_with_identity(self):
        """Test factory creation with base identity."""
        coordinator = create_prompt_coordinator(base_identity="You are a helpful assistant.")

        assert coordinator._base_identity == "You are a helpful assistant."

    def test_create_with_builder(self):
        """Test factory creation with custom builder."""
        builder = PromptBuilder()
        coordinator = create_prompt_coordinator(prompt_builder=builder)

        assert coordinator._builder is builder
