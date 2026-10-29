"""Tests for tool hallucination prevention — Layer 6 of agentic execution quality."""

from unittest.mock import MagicMock, patch


from victor.agent.query_classifier import QueryClassification, QueryType
from victor.framework.task.protocols import TaskComplexity


class TestToolConstraintPrompt:
    """Tests for _get_tool_constraint_section in SystemPromptBuilder."""

    def test_prompt_includes_available_tools_section(self):
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            available_tools=["read_file", "write_file", "shell"],
        )
        section = builder._get_tool_constraint_section()
        assert "read" in section
        assert "write" in section
        assert "shell" in section
        assert "read_file" not in section
        assert "write_file" not in section
        assert "IMPORTANT" in section or "Only use" in section

    def test_prompt_no_tools_when_empty(self):
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            available_tools=[],
        )
        section = builder._get_tool_constraint_section()
        assert section == ""

    def test_prompt_no_tools_when_none(self):
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            available_tools=None,
        )
        section = builder._get_tool_constraint_section()
        assert section == ""

    def test_tool_constraint_combined_with_classification(self):
        from victor.agent.prompt_builder import SystemPromptBuilder

        classification = QueryClassification(
            query_type=QueryType.EXPLORATION,
            complexity=TaskComplexity.COMPLEX,
            should_plan=True,
            should_use_subagents=True,
            continuation_budget_hint=8,
            confidence=0.9,
        )
        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            available_tools=["read_file", "shell"],
            query_classification=classification,
        )
        prompt = builder.build()
        # Both task guidance and tool constraints should be present
        assert "read" in prompt
        assert "systematically" in prompt.lower() or "explore" in prompt.lower()


class TestToolPreFilter:
    """Tests for pre-filtering hallucinated tool names in ToolService."""

    def _make_service(self, known_tools=None):
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        mock_registry = MagicMock(spec=["get_tool_names"])
        mock_registry.get_tool_names.return_value = known_tools or {
            "read_file",
            "write_file",
            "shell",
        }

        return ToolService(
            config=ToolServiceConfig(),
            tool_selector=MagicMock(),
            tool_executor=MagicMock(),
            tool_registrar=mock_registry,
        )

    def test_pre_filter_removes_hallucinated_tools(self):
        service = self._make_service()
        tool_calls = [
            {"name": "read_file", "arguments": {}},
            {"name": "search", "arguments": {}},
            {"name": "run_team", "arguments": {}},
        ]
        valid, filtered = service._pre_filter_tool_calls(tool_calls)
        assert len(valid) == 1
        assert valid[0]["name"] == "read_file"
        assert "search" in filtered
        assert "run_team" in filtered

    def test_pre_filter_preserves_valid_tools(self):
        service = self._make_service()
        tool_calls = [
            {"name": "read_file", "arguments": {}},
            {"name": "write_file", "arguments": {}},
        ]
        valid, filtered = service._pre_filter_tool_calls(tool_calls)
        assert len(valid) == 2
        assert len(filtered) == 0

    def test_pre_filter_logs_filtered_names(self):
        service = self._make_service()
        tool_calls = [
            {"name": "get_stats", "arguments": {}},
        ]
        with patch.object(service._logger, "warning") as mock_warning:
            valid, filtered = service._pre_filter_tool_calls(tool_calls)
            mock_warning.assert_called_once()
            call_args = str(mock_warning.call_args)
            assert "get_stats" in call_args
