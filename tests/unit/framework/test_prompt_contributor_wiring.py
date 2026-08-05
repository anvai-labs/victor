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

"""Tests for PromptContributorAdapter wiring in step handlers.

Workstream C: OpenAI Codex feedback fix - PromptContributorAdapter exists
but was not wired into step handlers (dead code). These tests verify the
adapter is properly integrated.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

from victor.agent.vertical_context import VerticalContext, create_vertical_context
from victor.core.verticals.prompt_adapter import (
    PromptContributorAdapter,
    CompositePromptContributor,
    create_prompt_adapter,
)
from victor.core.verticals.protocols import TaskTypeHint
from victor.core.verticals.protocols.prompt_provider import PromptSectionContribution
from victor.framework.step_handlers import PromptStepHandler

# =============================================================================
# Test Fixtures
# =============================================================================


class MockContributorLegacyFormat:
    """Legacy format contributor (dict-based hints)."""

    def get_task_type_hints(self) -> Dict[str, dict]:
        """Return hints in legacy dict format."""
        return {
            "edit": {"hint": "Read file first", "tool_budget": 5},
            "search": {"hint": "Use grep for searching"},
        }

    def get_system_prompt_section(self) -> str:
        return "Legacy prompt section."

    def get_grounding_rules(self) -> str:
        return "Ground responses on file content."

    def get_priority(self) -> int:
        return 50


class MockContributorStringFormat:
    """String format contributor (simple string hints)."""

    def get_task_type_hints(self) -> Dict[str, str]:
        """Return hints in simple string format."""
        return {
            "refactor": "Preserve existing tests",
            "fix": "Verify fix with test",
        }

    def get_system_prompt_section(self) -> str:
        return "String format section."

    def get_grounding_rules(self) -> str:
        return ""

    def get_priority(self) -> int:
        return 60


class MockContributorProtocolFormat:
    """Protocol-compliant contributor (TaskTypeHint objects)."""

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Return hints as TaskTypeHint objects."""
        return {
            "implement": TaskTypeHint(
                task_type="implement",
                hint="Write comprehensive tests",
                tool_budget=10,
                priority_tools=["write", "read", "test"],
            ),
        }

    def get_system_prompt_section(self) -> str:
        return "Protocol-compliant section."

    def get_grounding_rules(self) -> str:
        return "Verify with tests."

    def get_priority(self) -> int:
        return 40


class MockContributorNamedSections:
    """Contributor using the named prompt-section metadata path."""

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        return {}

    def get_system_prompt_section(self) -> str:
        return ""

    def get_prompt_section_contributions(self):
        return [
            PromptSectionContribution(
                name="VERTICAL_REVIEW_GUIDANCE",
                text="Prefer API-boundary review before deep implementation changes.",
                aliases={"review_guidance"},
                category="context",
                evolvable=True,
                priority=45,
            )
        ]

    def get_grounding_rules(self) -> str:
        return ""

    def get_priority(self) -> int:
        return 45


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self):
        self._task_hints: Dict[str, Any] = {}
        self._prompt_sections: List[str] = []
        self.prompt_builder = MagicMock()
        self.prompt_builder.set_task_type_hints = MagicMock()
        self.prompt_builder.add_prompt_section = MagicMock()


# =============================================================================
# Test: PromptContributorAdapter is used
# =============================================================================


class TestPromptContributorAdapterWiring:
    """Tests for PromptContributorAdapter wiring in step handlers."""

    def test_prompt_contributor_adapter_is_used(self):
        """PromptContributorAdapter must be wired into step handlers.

        The adapter normalizes different prompt contribution formats
        (dict, string, TaskTypeHint) into a consistent interface.
        This test verifies the adapter is actually used.
        """
        handler = PromptStepHandler()
        orchestrator = MockOrchestrator()
        context = create_vertical_context(name="test")
        result = MagicMock()
        result.prompt_hints_count = 0

        # Create mixed format contributors
        contributors = [
            MockContributorLegacyFormat(),
            MockContributorStringFormat(),
            MockContributorProtocolFormat(),
        ]

        # Apply contributors
        handler.apply_contributors(orchestrator, contributors, context, result)

        # Verify task hints were normalized - all should be TaskTypeHint
        assert len(context.task_hints) > 0
        for task_type, hint in context.task_hints.items():
            assert isinstance(hint, TaskTypeHint), (
                f"Hint for '{task_type}' should be TaskTypeHint, "
                f"got {type(hint).__name__}. PromptContributorAdapter "
                "should normalize all hint formats."
            )

    def test_prompt_contributions_formatted_correctly(self):
        """Prompt contributions must be formatted via adapter.

        The adapter ensures:
        1. Dict format is converted to TaskTypeHint
        2. String format is converted to TaskTypeHint
        3. TaskTypeHint objects are preserved
        4. All hints have consistent structure
        """
        handler = PromptStepHandler()
        orchestrator = MockOrchestrator()
        context = create_vertical_context(name="test")
        result = MagicMock()
        result.prompt_hints_count = 0

        # Test with legacy dict format contributor
        legacy_contributor = MockContributorLegacyFormat()
        handler.apply_contributors(orchestrator, [legacy_contributor], context, result)

        # Verify conversion
        edit_hint = context.get_task_hint("edit")
        assert edit_hint is not None, "edit hint should exist"
        assert isinstance(
            edit_hint, TaskTypeHint
        ), "Dict format should be converted to TaskTypeHint"
        assert edit_hint.hint == "Read file first"
        assert edit_hint.tool_budget == 5

    def test_mixed_format_contributors_normalized(self):
        """Mixed format contributors should all be normalized via adapter."""
        handler = PromptStepHandler()
        orchestrator = MockOrchestrator()
        context = create_vertical_context(name="test")
        result = MagicMock()
        result.prompt_hints_count = 0

        # Mix of different contributor formats
        contributors = [
            MockContributorLegacyFormat(),
            MockContributorStringFormat(),
            MockContributorProtocolFormat(),
        ]

        handler.apply_contributors(orchestrator, contributors, context, result)

        # All hints should be TaskTypeHint instances
        expected_hints = ["edit", "search", "refactor", "fix", "implement"]
        for hint_name in expected_hints:
            hint = context.get_task_hint(hint_name)
            assert hint is not None, f"Hint '{hint_name}' should exist"
            assert isinstance(
                hint, TaskTypeHint
            ), f"Hint '{hint_name}' should be TaskTypeHint after adapter normalization"
            assert hint.task_type == hint_name, f"Hint task_type should be '{hint_name}'"

    def test_adapter_wrap_preserves_existing_contributor(self):
        """Wrapping existing contributor via adapter should preserve data."""
        original = MockContributorProtocolFormat()

        # Wrap in adapter
        adapter = PromptContributorAdapter.wrap(original)

        # Verify preservation
        hints = adapter.get_task_type_hints()
        assert "implement" in hints
        assert hints["implement"].hint == "Write comprehensive tests"
        assert hints["implement"].tool_budget == 10
        assert hints["implement"].priority_tools == ["write", "read", "test"]

    def test_adapter_wrap_uses_named_prompt_sections(self):
        """Adapter.wrap should preserve named contributor prompt text."""
        adapter = PromptContributorAdapter.wrap(MockContributorNamedSections())

        assert (
            adapter.get_system_prompt_section()
            == "Prefer API-boundary review before deep implementation changes."
        )

    def test_adapter_from_dict_normalizes_formats(self):
        """Adapter.from_dict should normalize all hint formats."""
        mixed_hints = {
            # Dict format
            "edit": {"hint": "Edit hint", "tool_budget": 5},
            # String format
            "search": "Search hint",
            # TaskTypeHint format
            "fix": TaskTypeHint(task_type="fix", hint="Fix hint", tool_budget=3),
        }

        adapter = PromptContributorAdapter.from_dict(task_hints=mixed_hints)
        result = adapter.get_task_type_hints()

        # All should be TaskTypeHint
        for task_type in ["edit", "search", "fix"]:
            assert task_type in result
            assert isinstance(result[task_type], TaskTypeHint)

        # Verify specific conversions
        assert result["edit"].hint == "Edit hint"
        assert result["edit"].tool_budget == 5
        assert result["search"].hint == "Search hint"
        assert result["fix"].hint == "Fix hint"


class TestPromptStepHandlerIntegration:
    """Integration tests for PromptStepHandler with adapter."""

    def test_step_handler_uses_adapter_for_normalization(self):
        """Step handler should use adapter to normalize contributor formats."""
        handler = PromptStepHandler()
        orchestrator = MockOrchestrator()
        context = create_vertical_context(name="test")
        result = MagicMock()
        result.prompt_hints_count = 0

        # Legacy contributor with dict hints
        class LegacyDictContributor:
            def get_task_type_hints(self):
                return {"task1": {"hint": "Dict hint"}}

            def get_system_prompt_section(self):
                return ""

            def get_grounding_rules(self):
                return ""

            def get_priority(self):
                return 50

        handler.apply_contributors(orchestrator, [LegacyDictContributor()], context, result)

        # Context should have normalized TaskTypeHint
        hint = context.get_task_hint("task1")
        assert hint is not None
        assert isinstance(hint, TaskTypeHint), (
            "PromptStepHandler should use PromptContributorAdapter to "
            "normalize dict hints to TaskTypeHint"
        )

    def test_prompt_sections_collected(self):
        """Prompt sections should be collected from contributors."""
        handler = PromptStepHandler()
        orchestrator = MockOrchestrator()
        context = create_vertical_context(name="test")
        result = MagicMock()
        result.prompt_hints_count = 0

        contributors = [
            MockContributorLegacyFormat(),
            MockContributorProtocolFormat(),
        ]

        handler.apply_contributors(orchestrator, contributors, context, result)

        # Prompt sections should be in context
        assert len(context.prompt_sections) == 2
        assert "Legacy prompt section." in context.prompt_sections
        assert "Protocol-compliant section." in context.prompt_sections

    def test_named_prompt_sections_are_collected(self):
        """Named contributor sections should flow through the step handler."""
        handler = PromptStepHandler()
        orchestrator = MockOrchestrator()
        context = create_vertical_context(name="test")
        result = MagicMock()
        result.prompt_hints_count = 0

        handler.apply_contributors(orchestrator, [MockContributorNamedSections()], context, result)

        assert context.prompt_sections == [
            "Prefer API-boundary review before deep implementation changes."
        ]

    def test_composite_uses_named_prompt_sections(self):
        """Composite contributors should combine named prompt sections too."""
        composite = CompositePromptContributor(
            [MockContributorNamedSections(), MockContributorLegacyFormat()]
        )

        rendered = composite.get_system_prompt_section()

        assert "Prefer API-boundary review before deep implementation changes." in rendered
        assert "Legacy prompt section." in rendered
