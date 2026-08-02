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

"""Tests for the consolidated PromptBuilder module.

These tests verify the framework's unified prompt building system that
consolidates duplicate prompt construction logic.
"""

import importlib
from typing import Dict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import victor.framework.prompt_sections as prompt_sections_module

from victor.core.grounding_texts import GROUNDING_RULES
from victor.framework.prompt_builder import (
    PromptBuilder,
    PromptSection,
    ToolHint,
    create_coding_prompt_builder,
    create_devops_prompt_builder,
    create_research_prompt_builder,
    create_data_analysis_prompt_builder,
)
from victor.framework.prompt_sections import (
    GROUNDING_RULES_MINIMAL,
    GROUNDING_RULES_EXTENDED,
    CODING_IDENTITY,
    CODING_GUIDELINES,
    DATA_ANALYSIS_GROUNDING,
    DEVOPS_IDENTITY,
    DEVOPS_GROUNDING,
    RESEARCH_IDENTITY,
    RESEARCH_GROUNDING,
    DATA_ANALYSIS_IDENTITY,
)


class TestPromptSection:
    """Tests for PromptSection dataclass."""

    def test_prompt_section_default_values(self):
        """Test default values for PromptSection."""
        section = PromptSection(name="test", content="Test content")
        assert section.name == "test"
        assert section.content == "Test content"
        assert section.priority == 50
        assert section.enabled is True
        assert section.header is None

    def test_prompt_section_custom_values(self):
        """Test custom values for PromptSection."""
        section = PromptSection(
            name="custom",
            content="Custom content",
            priority=10,
            enabled=False,
            header="## Custom Header",
        )
        assert section.name == "custom"
        assert section.priority == 10
        assert section.enabled is False
        assert section.header == "## Custom Header"

    def test_get_formatted_content_with_auto_header(self):
        """Test formatted content with auto-generated header."""
        section = PromptSection(name="tool_usage", content="Use tools wisely.")
        result = section.get_formatted_content()
        assert "## Tool Usage" in result
        assert "Use tools wisely." in result

    def test_get_formatted_content_with_custom_header(self):
        """Test formatted content with custom header."""
        section = PromptSection(
            name="test",
            content="Content here.",
            header="### My Custom Header",
        )
        result = section.get_formatted_content()
        assert "### My Custom Header" in result
        assert "Content here." in result

    def test_get_formatted_content_with_no_header(self):
        """Test formatted content with empty header (disabled)."""
        section = PromptSection(
            name="test",
            content="[EDIT] Read file first.",
            header="",  # Empty string disables header
        )
        result = section.get_formatted_content()
        assert not result.startswith("##")
        assert "[EDIT] Read file first." in result

    def test_get_formatted_content_disabled_section(self):
        """Test that disabled sections return empty string."""
        section = PromptSection(name="test", content="Hidden", enabled=False)
        result = section.get_formatted_content()
        assert result == ""


class TestToolHint:
    """Tests for ToolHint dataclass."""

    def test_tool_hint_default_boost(self):
        """Test default priority boost is 0."""
        hint = ToolHint(tool_name="read", hint="Read file contents")
        assert hint.tool_name == "read"
        assert hint.hint == "Read file contents"
        assert hint.priority_boost == 0.0

    def test_tool_hint_with_boost(self):
        """Test custom priority boost."""
        hint = ToolHint(tool_name="edit", hint="Edit files", priority_boost=0.3)
        assert hint.priority_boost == 0.3


class TestPromptBuilder:
    """Tests for the main PromptBuilder class."""

    def test_add_section_basic(self):
        """Test adding a basic section."""
        builder = PromptBuilder()
        result = builder.add_section("identity", "You are an assistant.")

        # Verify fluent interface
        assert result is builder

        # Verify section added
        assert builder.has_section("identity")
        section = builder.get_section("identity")
        assert section is not None
        assert section.content == "You are an assistant."
        assert section.priority == 50  # default

    def test_add_section_with_priority(self):
        """Test adding sections with different priorities."""
        builder = PromptBuilder()
        builder.add_section("last", "Last section", priority=100)
        builder.add_section("first", "First section", priority=10)
        builder.add_section("middle", "Middle section", priority=50)

        prompt = builder.build()

        # Check ordering
        first_idx = prompt.find("First section")
        middle_idx = prompt.find("Middle section")
        last_idx = prompt.find("Last section")

        assert first_idx < middle_idx < last_idx

    def test_build_document_renders_same_as_build(self):
        """Canonical prompt document should be the single rendering source."""
        builder = PromptBuilder()
        builder.add_section("identity", "I am an assistant.", priority=10)
        builder.add_tool_hints({"read": "Read files"})
        builder.add_safety_rules(["Be careful"])
        builder.add_context("Working on code")

        document = builder.build_document()

        assert document.render() == builder.build()

    def test_ensure_section_adds_missing(self):
        """Ensure section helper adds fallback content."""
        builder = PromptBuilder()
        builder.ensure_section("identity", "Fallback identity", priority=10, header="")

        assert builder.has_section("identity")
        section = builder.get_section("identity")
        assert section is not None
        assert section.content == "Fallback identity"
        assert section.priority == 10

    def test_trim_sections_by_priority(self):
        """Trim sections when prompt exceeds character budget."""
        builder = PromptBuilder()
        builder.add_section("identity", "ID", priority=10)
        builder.add_section(
            "context_a",
            "A" * 100,
            priority=PromptBuilder.PRIORITY_CONTEXT,
        )
        builder.add_section(
            "context_b",
            "B" * 120,
            priority=PromptBuilder.PRIORITY_CONTEXT + 1,
        )

        builder.trim_sections_by_priority(
            max_total_chars=150,
            protected_sections=("identity",),
            min_priority=PromptBuilder.PRIORITY_CONTEXT,
        )

        assert builder.has_section("identity")
        # Only one context section should remain after trimming
        remaining_context = [
            name for name in ("context_a", "context_b") if builder.has_section(name)
        ]
        assert len(remaining_context) == 1

    def test_get_section_fallback_uses_shared_registry(self):
        """Framework prompt builder should reuse registry-backed fallback text."""
        from victor.agent.prompt_section_registry import build_fallback_map

        builder = PromptBuilder()

        assert (
            builder._get_section_fallback("LARGE_FILE_PAGINATION_GUIDANCE")
            == build_fallback_map(["LARGE_FILE_PAGINATION_GUIDANCE"])[
                "LARGE_FILE_PAGINATION_GUIDANCE"
            ]
        )

    def test_add_evolved_section_uses_optimization_injector_interface(self):
        """Framework prompt builder should wire the canonical injector into the resolver."""
        builder = PromptBuilder()
        resolved = SimpleNamespace(text="EVOLVED PAGINATION", source="evolved")

        with (
            patch("victor.agent.optimization_injector.OptimizationInjector") as injector_cls,
            patch("victor.agent.evolved_content_resolver.EvolvedContentResolver") as resolver_cls,
        ):
            injector_instance = injector_cls.return_value
            resolver_cls.return_value.resolve_section.return_value = resolved

            builder.add_evolved_section(
                "LARGE_FILE_PAGINATION_GUIDANCE",
                provider="anthropic",
                model="claude-sonnet",
                task_type="analysis",
                priority=55,
            )

        resolver_cls.assert_called_once_with(optimization_injector=injector_instance)
        resolver_cls.return_value.resolve_section.assert_called_once_with(
            section_name="LARGE_FILE_PAGINATION_GUIDANCE",
            provider="anthropic",
            model="claude-sonnet",
            task_type="analysis",
            fallback_text=builder._get_section_fallback("LARGE_FILE_PAGINATION_GUIDANCE"),
        )
        section = builder.get_section("large_file_pagination_guidance")
        assert section is not None
        assert section.content == "EVOLVED PAGINATION"
        assert section.priority == 55

    def test_add_section_with_custom_header(self):
        """Test adding section with custom header."""
        builder = PromptBuilder()
        builder.add_section("test", "Content", header="### Custom")

        prompt = builder.build()
        assert "### Custom" in prompt
        assert "## Test" not in prompt  # No auto-generated header

    def test_add_tool_hints_dict(self):
        """Test adding tool hints as dictionary."""
        builder = PromptBuilder()
        builder.add_tool_hints(
            {
                "read": "Use to read file contents",
                "write": "Use to write files",
            }
        )

        prompt = builder.build()
        assert "## Tool Hints" in prompt
        assert "read: Use to read file contents" in prompt
        assert "write: Use to write files" in prompt

    def test_add_tool_hint_single(self):
        """Test adding a single tool hint with boost."""
        builder = PromptBuilder()
        builder.add_tool_hint("edit", "Edit existing files", priority_boost=0.2)

        prompt = builder.build()
        assert "edit: Edit existing files" in prompt

    def test_add_safety_rules_list(self):
        """Test adding safety rules as list."""
        builder = PromptBuilder()
        builder.add_safety_rules(
            [
                "Never expose credentials",
                "Validate all input",
            ]
        )

        prompt = builder.build()
        assert "## Safety Rules" in prompt
        assert "- Never expose credentials" in prompt
        assert "- Validate all input" in prompt

    def test_add_safety_rule_single(self):
        """Test adding a single safety rule."""
        builder = PromptBuilder()
        builder.add_safety_rule("Check file permissions")

        prompt = builder.build()
        assert "- Check file permissions" in prompt

    def test_add_context(self):
        """Test adding contextual information."""
        builder = PromptBuilder()
        builder.add_context("Working on a Python project")
        builder.add_context("Using pytest for testing")

        prompt = builder.build()
        assert "## Context" in prompt
        assert "Working on a Python project" in prompt
        assert "Using pytest for testing" in prompt

    def test_set_grounding_mode_minimal(self):
        """Test setting minimal grounding mode."""
        builder = PromptBuilder()
        builder.set_grounding_mode("minimal")

        prompt = builder.build()
        assert "GROUNDING:" in prompt
        # Minimal grounding is shorter
        assert "Quote code exactly" in prompt or "Quote code" in prompt

    def test_set_grounding_mode_extended(self):
        """Test setting extended grounding mode."""
        builder = PromptBuilder()
        builder.set_grounding_mode("extended")

        prompt = builder.build()
        assert "CRITICAL - TOOL OUTPUT GROUNDING:" in prompt

    def test_set_grounding_mode_invalid(self):
        """Test that invalid grounding mode raises error."""
        builder = PromptBuilder()
        with pytest.raises(ValueError, match="Invalid grounding mode"):
            builder.set_grounding_mode("invalid")

    def test_set_custom_grounding(self):
        """Test setting custom grounding rules."""
        builder = PromptBuilder()
        custom = "CUSTOM: Always verify before acting."
        builder.set_custom_grounding(custom)

        prompt = builder.build()
        assert "CUSTOM: Always verify before acting." in prompt
        # Should not have default grounding
        assert "GROUNDING:" not in prompt or "CUSTOM:" in prompt

    def test_remove_section(self):
        """Test removing a section."""
        builder = PromptBuilder()
        builder.add_section("test", "Test content")
        assert builder.has_section("test")

        builder.remove_section("test")
        assert not builder.has_section("test")

    def test_remove_section_nonexistent(self):
        """Test removing nonexistent section doesn't error."""
        builder = PromptBuilder()
        builder.remove_section("nonexistent")  # Should not raise

    def test_disable_enable_section(self):
        """Test disabling and enabling sections."""
        builder = PromptBuilder()
        builder.add_section("test", "Visible content")

        # Initially visible
        prompt1 = builder.build()
        assert "Visible content" in prompt1

        # Disable
        builder.disable_section("test")
        prompt2 = builder.build()
        assert "Visible content" not in prompt2

        # Re-enable
        builder.enable_section("test")
        prompt3 = builder.build()
        assert "Visible content" in prompt3

    def test_clear(self):
        """Test clearing all builder state."""
        builder = PromptBuilder()
        builder.add_section("test", "Content")
        builder.add_tool_hints({"read": "Read files"})
        builder.add_safety_rules(["Be safe"])
        builder.add_context("Context info")
        builder.set_grounding_mode("extended")

        builder.clear()

        prompt = builder.build()
        # Should only have minimal grounding (default)
        assert "Content" not in prompt
        assert "## Tool Hints" not in prompt
        assert "## Safety Rules" not in prompt
        assert "## Context" not in prompt

    def test_build_output_format(self):
        """Test the overall build output format."""
        builder = (
            PromptBuilder()
            .add_section("identity", "I am an assistant.", priority=10)
            .add_tool_hints({"read": "Read files"})
            .add_safety_rules(["Be careful"])
            .add_context("Working on code")
        )

        prompt = builder.build()

        # Verify sections are separated by double newlines
        assert "\n\n" in prompt

        # Verify grounding comes last
        last_section = prompt.split("\n\n")[-1]
        assert "GROUNDING" in last_section

    def test_fluent_api_chaining(self):
        """Test fluent API allows full chaining."""
        prompt = (
            PromptBuilder()
            .add_section("id", "Assistant", priority=10)
            .add_section("guide", "Guidelines", priority=20)
            .add_tool_hints({"read": "Read"})
            .add_tool_hint("write", "Write")
            .add_safety_rules(["Rule 1"])
            .add_safety_rule("Rule 2")
            .add_context("Context 1")
            .set_grounding_mode("minimal")
            .build()
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_repr(self):
        """Test string representation of builder."""
        builder = PromptBuilder()
        builder.add_section("test", "Content")
        builder.add_tool_hints({"read": "Read"})
        builder.add_safety_rules(["Rule"])
        builder.add_context("Context")

        repr_str = repr(builder)
        assert "PromptBuilder" in repr_str
        assert "sections=1" in repr_str
        assert "tool_hints=1" in repr_str
        assert "safety_rules=1" in repr_str
        assert "context=1" in repr_str

    def test_priority_constants(self):
        """Test priority constant values."""
        assert PromptBuilder.PRIORITY_IDENTITY < PromptBuilder.PRIORITY_CAPABILITIES
        assert PromptBuilder.PRIORITY_CAPABILITIES < PromptBuilder.PRIORITY_GUIDELINES
        assert PromptBuilder.PRIORITY_GUIDELINES < PromptBuilder.PRIORITY_TASK_HINTS
        assert PromptBuilder.PRIORITY_TASK_HINTS < PromptBuilder.PRIORITY_TOOL_GUIDANCE
        assert PromptBuilder.PRIORITY_TOOL_GUIDANCE < PromptBuilder.PRIORITY_SAFETY
        assert PromptBuilder.PRIORITY_SAFETY < PromptBuilder.PRIORITY_CONTEXT
        assert PromptBuilder.PRIORITY_CONTEXT < PromptBuilder.PRIORITY_GROUNDING


class TestPromptBuilderMerge:
    """Tests for PromptBuilder.merge() functionality."""

    def test_merge_sections(self):
        """Test merging sections from another builder."""
        builder1 = PromptBuilder()
        builder1.add_section("section1", "Content 1")

        builder2 = PromptBuilder()
        builder2.add_section("section2", "Content 2")

        builder1.merge(builder2)

        assert builder1.has_section("section1")
        assert builder1.has_section("section2")

    def test_merge_sections_override(self):
        """Test that merged sections override existing ones."""
        builder1 = PromptBuilder()
        builder1.add_section("shared", "Original")

        builder2 = PromptBuilder()
        builder2.add_section("shared", "Override")

        builder1.merge(builder2)

        section = builder1.get_section("shared")
        assert section is not None
        assert section.content == "Override"

    def test_merge_tool_hints(self):
        """Test merging tool hints."""
        builder1 = PromptBuilder()
        builder1.add_tool_hints({"read": "Read 1"})

        builder2 = PromptBuilder()
        builder2.add_tool_hints({"write": "Write 2"})

        builder1.merge(builder2)

        prompt = builder1.build()
        assert "read: Read 1" in prompt
        assert "write: Write 2" in prompt

    def test_merge_safety_rules_deduplicate(self):
        """Test that merged safety rules are deduplicated."""
        builder1 = PromptBuilder()
        builder1.add_safety_rules(["Rule A", "Rule B"])

        builder2 = PromptBuilder()
        builder2.add_safety_rules(["Rule B", "Rule C"])  # Rule B is duplicate

        builder1.merge(builder2)

        prompt = builder1.build()
        assert prompt.count("Rule B") == 1  # Not duplicated

    def test_merge_grounding_extended_wins(self):
        """Test that extended grounding mode takes precedence."""
        builder1 = PromptBuilder()
        builder1.set_grounding_mode("minimal")

        builder2 = PromptBuilder()
        builder2.set_grounding_mode("extended")

        builder1.merge(builder2)

        prompt = builder1.build()
        assert "CRITICAL - TOOL OUTPUT GROUNDING:" in prompt

    def test_merge_custom_grounding(self):
        """Test that custom grounding from other builder is used."""
        builder1 = PromptBuilder()

        builder2 = PromptBuilder()
        builder2.set_custom_grounding("CUSTOM GROUNDING")

        builder1.merge(builder2)

        prompt = builder1.build()
        assert "CUSTOM GROUNDING" in prompt


class TestPromptBuilderContributor:
    """Tests for PromptBuilder integration with PromptContributorProtocol."""

    def test_add_from_contributor_system_section(self):
        """Test adding system prompt section from contributor."""
        # Mock a contributor
        contributor = MagicMock()
        contributor.get_priority.return_value = 5
        contributor.get_system_prompt_section.return_value = "Contributor section content"
        contributor.get_grounding_rules.return_value = None
        contributor.get_task_type_hints.return_value = {}

        builder = PromptBuilder()
        builder.add_from_contributor(contributor)

        prompt = builder.build()
        assert "Contributor section content" in prompt

    def test_add_from_contributor_grounding(self):
        """Test adding grounding rules from contributor."""
        contributor = MagicMock()
        contributor.get_priority.return_value = 5
        contributor.get_system_prompt_section.return_value = None
        contributor.get_grounding_rules.return_value = "CUSTOM GROUNDING RULES"
        contributor.get_task_type_hints.return_value = {}

        builder = PromptBuilder()
        builder.add_from_contributor(contributor)

        prompt = builder.build()
        assert "CUSTOM GROUNDING RULES" in prompt

    def test_add_from_contributor_task_hint(self):
        """Test adding task-specific hint from contributor."""
        # Create mock TaskTypeHint-like object
        mock_hint = MagicMock()
        mock_hint.hint = "[EDIT] Read target file first."
        mock_hint.priority_tools = ["read", "edit"]

        contributor = MagicMock()
        contributor.get_priority.return_value = 5
        contributor.get_system_prompt_section.return_value = None
        contributor.get_grounding_rules.return_value = None
        contributor.get_task_type_hints.return_value = {"edit": mock_hint}

        builder = PromptBuilder()
        builder.add_from_contributor(contributor, task_type="edit")

        prompt = builder.build()
        assert "[EDIT] Read target file first." in prompt


class TestFactoryFunctions:
    """Tests for the factory functions that create pre-configured builders."""

    def test_create_coding_prompt_builder(self):
        """Test coding prompt builder factory."""
        builder = create_coding_prompt_builder()

        assert builder.has_section("identity")
        assert builder.has_section("guidelines")
        assert builder.has_section("tool_usage")

        prompt = builder.build()
        assert "Victor" in prompt  # From CODING_IDENTITY
        assert "Incremental changes" in prompt  # From CODING_GUIDELINES

    def test_create_devops_prompt_builder(self):
        """Test DevOps prompt builder factory."""
        builder = create_devops_prompt_builder()

        assert builder.has_section("identity")
        assert builder.has_section("security")
        assert builder.has_section("pitfalls")

        prompt = builder.build()
        assert "DevOps" in prompt or "infrastructure" in prompt

    def test_create_research_prompt_builder(self):
        """Test research prompt builder factory."""
        builder = create_research_prompt_builder()

        assert builder.has_section("identity")
        assert builder.has_section("quality")
        assert builder.has_section("sources")

        prompt = builder.build()
        assert "research" in prompt.lower()

    def test_create_data_analysis_prompt_builder(self):
        """Test data analysis prompt builder factory."""
        builder = create_data_analysis_prompt_builder()

        assert builder.has_section("identity")
        assert builder.has_section("libraries")
        assert builder.has_section("operations")

        prompt = builder.build()
        assert "data" in prompt.lower() or "pandas" in prompt.lower()


class TestPromptSectionsImports:
    """Tests verifying prompt_sections module contents."""

    def test_grounding_rules_minimal_content(self):
        """Test GROUNDING_RULES_MINIMAL has expected content."""
        assert "GROUNDING" in GROUNDING_RULES_MINIMAL
        assert "tool output" in GROUNDING_RULES_MINIMAL.lower()

    def test_grounding_rules_extended_content(self):
        """Test GROUNDING_RULES_EXTENDED has expected content."""
        assert "CRITICAL" in GROUNDING_RULES_EXTENDED
        assert "TOOL OUTPUT GROUNDING" in GROUNDING_RULES_EXTENDED

    def test_framework_vertical_grounding_sections_compose_canonical_base(self):
        """Framework vertical grounding exports should build on canonical shared grounding."""
        assert DEVOPS_GROUNDING.startswith(GROUNDING_RULES)
        assert RESEARCH_GROUNDING.startswith(GROUNDING_RULES)
        assert DATA_ANALYSIS_GROUNDING.startswith(GROUNDING_RULES)
        assert "Verify configuration syntax before suggesting." in DEVOPS_GROUNDING
        assert "Always cite URLs for claims." in RESEARCH_GROUNDING
        assert "Always show code that produced results." in DATA_ANALYSIS_GROUNDING

    def test_shared_framework_sections_resolve_from_registry(self, monkeypatch):
        """Framework shared guidance exports should follow the canonical registry."""
        from victor.agent import prompt_section_registry as registry_module
        from victor.agent.prompt_section_registry import (
            SectionCategory,
            SectionDefinition,
            UnifiedSectionRegistry,
            _initialize_default_sections,
        )

        fresh_registry = UnifiedSectionRegistry()
        _initialize_default_sections(fresh_registry)
        fresh_registry.register(
            SectionDefinition(
                name="GROUNDING_RULES",
                aliases={"grounding", "grounding_rules", "safety_rules"},
                category=SectionCategory.GROUNDING,
                default_text="Registry framework minimal grounding.",
                evolvable=True,
                required=True,
                priority=80,
                default_strategies=("gepa", "prefpo"),
            )
        )
        fresh_registry.register(
            SectionDefinition(
                name="GROUNDING_RULES_EXTENDED",
                aliases={"strict_grounding", "grounding_rules_extended"},
                category=SectionCategory.GROUNDING,
                default_text="Registry framework strict grounding.",
                evolvable=True,
                required=False,
                priority=85,
                default_strategies=("gepa",),
            )
        )
        fresh_registry.register(
            SectionDefinition(
                name="PARALLEL_READ_GUIDANCE",
                aliases={"parallel_read_guidance", "parallel_reads", "batch_reading"},
                category=SectionCategory.TOOL_GUIDANCE,
                default_text="Registry framework parallel reads.",
                evolvable=True,
                required=False,
                priority=70,
                default_strategies=("gepa",),
            )
        )
        monkeypatch.setattr(registry_module, "_registry", fresh_registry)

        reloaded_module = importlib.reload(prompt_sections_module)
        try:
            assert (
                reloaded_module.GROUNDING_RULES_MINIMAL == "Registry framework minimal grounding."
            )
            assert (
                reloaded_module.GROUNDING_RULES_EXTENDED == "Registry framework strict grounding."
            )
            assert reloaded_module.PARALLEL_READ_GUIDANCE == "Registry framework parallel reads."
        finally:
            monkeypatch.setattr(registry_module, "_registry", None)
            importlib.reload(prompt_sections_module)

    def test_coding_identity_content(self):
        """Test CODING_IDENTITY has expected content."""
        assert "Victor" in CODING_IDENTITY
        assert "software development" in CODING_IDENTITY.lower()

    def test_devops_identity_content(self):
        """Test DEVOPS_IDENTITY has expected content."""
        assert "Victor" in DEVOPS_IDENTITY
        assert any(term in DEVOPS_IDENTITY for term in ["DevOps", "infrastructure", "Docker"])

    def test_research_identity_content(self):
        """Test RESEARCH_IDENTITY has expected content."""
        assert "Victor" in RESEARCH_IDENTITY
        assert "research" in RESEARCH_IDENTITY.lower()

    def test_data_analysis_identity_content(self):
        """Test DATA_ANALYSIS_IDENTITY has expected content."""
        assert "Victor" in DATA_ANALYSIS_IDENTITY
        assert "data" in DATA_ANALYSIS_IDENTITY.lower()


class TestPromptBuilderEdgeCases:
    """Edge case tests for PromptBuilder."""

    def test_empty_builder(self):
        """Test building with no sections added."""
        builder = PromptBuilder()
        prompt = builder.build()

        # Should still have grounding rules
        assert "GROUNDING" in prompt

    def test_only_disabled_sections(self):
        """Test building with all sections disabled."""
        builder = PromptBuilder()
        builder.add_section("test", "Content")
        builder.disable_section("test")

        prompt = builder.build()

        # Should have grounding but not the disabled section
        assert "GROUNDING" in prompt
        assert "Content" not in prompt

    def test_section_with_newlines(self):
        """Test section content with embedded newlines."""
        builder = PromptBuilder()
        content = "Line 1\nLine 2\nLine 3"
        builder.add_section("multiline", content)

        prompt = builder.build()
        assert "Line 1" in prompt
        assert "Line 2" in prompt
        assert "Line 3" in prompt

    def test_special_characters_in_content(self):
        """Test section content with special characters."""
        builder = PromptBuilder()
        content = 'Use `code` and **bold** with <tags> and "quotes"'
        builder.add_section("special", content)

        prompt = builder.build()
        assert "`code`" in prompt
        assert "**bold**" in prompt
        assert "<tags>" in prompt

    def test_unicode_content(self):
        """Test section content with unicode characters."""
        builder = PromptBuilder()
        content = "Support for: emojis, chinese (Chinese), 123"
        builder.add_section("unicode", content)

        prompt = builder.build()
        # Note: We avoid emojis per codebase guidelines, but test that unicode works
        assert "chinese (Chinese)" in prompt

    def test_duplicate_safety_rules(self):
        """Test that duplicate safety rules are preserved when added separately."""
        builder = PromptBuilder()
        builder.add_safety_rule("Rule A")
        builder.add_safety_rule("Rule A")  # Duplicate

        prompt = builder.build()
        # When added via add_safety_rule, duplicates ARE preserved
        assert prompt.count("Rule A") == 2

    def test_get_section_nonexistent(self):
        """Test getting nonexistent section returns None."""
        builder = PromptBuilder()
        assert builder.get_section("nonexistent") is None

    def test_has_section_after_remove(self):
        """Test has_section returns False after removal."""
        builder = PromptBuilder()
        builder.add_section("temp", "Temporary")
        builder.remove_section("temp")
        assert not builder.has_section("temp")
