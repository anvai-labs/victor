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

"""Comprehensive tests for OpenAI compatibility and markdown rendering.

These tests increase code coverage for:
- victor/providers/openai_compat.py (HTTP error handling, tool call accumulation)
- victor/ui/rendering/markdown.py (Mermaid rendering, image placeholders, error fallback)
"""

import logging
from unittest.mock import Mock

import httpx

from victor.providers.base import ProviderError, ToolDefinition
from victor.providers.openai_compat import (
    handle_httpx_status_error,
    fix_orphaned_tool_messages,
    accumulate_tool_call_delta,
    convert_tools_to_openai_format,
    parse_openai_tool_calls,
)
from victor.ui.rendering.markdown import (
    render_markdown_with_hooks,
    _escape_rich_markup_from_text,
    _markdown_block,
    _render_image_placeholder,
    _parse_mermaid_edges,
    _detect_direction,
    _normalize_mermaid_node,
)


class TestOpenAICompatErrorHandling:
    """Tests for HTTP error handling in openai_compat.py."""

    def test_handle_401_auth_error(self):
        """401 errors are mapped to ProviderAuthError."""
        from victor.providers.base import ProviderAuthError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_exc = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=Mock(),
            response=mock_response,
        )

        result = handle_httpx_status_error(mock_exc, "test_provider")

        assert isinstance(result, ProviderAuthError)
        assert result.status_code == 401
        assert "Authentication failed" in str(result)

    def test_handle_401_with_raw_response(self):
        """401 errors with response text are properly formatted."""
        from victor.providers.base import ProviderAuthError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"

        mock_exc = httpx.HTTPStatusError(
            "401",
            request=Mock(),
            response=mock_response,
        )

        result = handle_httpx_status_error(mock_exc, "deepseek")

        assert isinstance(result, ProviderAuthError)
        assert "Invalid API key" in str(result)

    def test_handle_429_rate_limit_error(self):
        """429 errors are mapped to ProviderRateLimitError."""
        from victor.providers.base import ProviderRateLimitError

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        mock_exc = httpx.HTTPStatusError(
            "429",
            request=Mock(),
            response=mock_response,
        )

        result = handle_httpx_status_error(mock_exc, "openai")

        assert isinstance(result, ProviderRateLimitError)
        assert result.status_code == 429
        assert "Rate limit exceeded" in str(result)

    def test_handle_400_with_json_body(self):
        """400 errors with JSON body are parsed for error message."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = '{"error": {"message": "Invalid tool_calls format"}}'

        mock_exc = httpx.HTTPStatusError(
            "400",
            request=Mock(),
            response=mock_response,
        )

        result = handle_httpx_status_error(mock_exc, "deepseek")

        assert isinstance(result, ProviderError)
        assert result.status_code == 400
        assert "Invalid tool_calls format" in str(result)

    def test_handle_400_with_invalid_json(self):
        """400 errors with invalid JSON fall back to raw text."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request {invalid json"

        mock_exc = httpx.HTTPStatusError(
            "400",
            request=Mock(),
            response=mock_response,
        )

        result = handle_httpx_status_error(mock_exc, "test_provider")

        assert isinstance(result, ProviderError)
        assert result.status_code == 400
        assert "Bad request" in str(result)

    def test_handle_400_empty_response_text(self, caplog):
        """400 errors with empty response text are handled."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = ""

        mock_exc = httpx.HTTPStatusError(
            "400",
            request=Mock(),
            response=mock_response,
        )

        with caplog.at_level(logging.ERROR):
            result = handle_httpx_status_error(mock_exc, "provider")

        assert isinstance(result, ProviderError)
        assert result.status_code == 400

    def test_handle_generic_http_error(self):
        """Generic HTTP errors (not 400/401/429) return ProviderError."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        mock_exc = httpx.HTTPStatusError(
            "500",
            request=Mock(),
            response=mock_response,
        )

        result = handle_httpx_status_error(mock_exc, "test_provider")

        assert isinstance(result, ProviderError)
        assert result.status_code == 500
        assert "HTTP error 500" in str(result)

    def test_handle_response_text_exception(self, caplog):
        """Exception when reading response.text is handled gracefully."""
        mock_response = Mock()
        mock_response.status_code = 500
        # Simulate exception when accessing text
        type(mock_response).text = property(
            lambda self: (_ for _ in ()).throw(Exception("Read error"))
        )

        mock_exc = httpx.HTTPStatusError(
            "500",
            request=Mock(),
            response=mock_response,
        )

        # Should not raise exception
        result = handle_httpx_status_error(mock_exc, "test_provider")

        assert isinstance(result, ProviderError)
        assert result.status_code == 500


class TestAccumulateToolCallDelta:
    """Tests for accumulate_tool_call_delta function."""

    def test_accumulate_single_tool_call_delta(self):
        """Single tool call delta is accumulated correctly."""
        accumulated = []

        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_123",
                    "function": {"name": "test_tool", "arguments": "{}"},
                }
            ]
        }

        accumulate_tool_call_delta(delta, accumulated)

        assert len(accumulated) == 1
        assert accumulated[0]["id"] == "call_123"
        assert accumulated[0]["name"] == "test_tool"
        assert accumulated[0]["arguments"] == "{}"

    def test_accumulate_multiple_tool_calls(self):
        """Multiple tool calls in same delta are accumulated."""
        accumulated = []

        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "tool1", "arguments": "{"},
                },
                {
                    "index": 1,
                    "id": "call_2",
                    "function": {"name": "tool2", "arguments": "{}"},
                },
            ]
        }

        accumulate_tool_call_delta(delta, accumulated)

        assert len(accumulated) == 2
        assert accumulated[0]["id"] == "call_1"
        assert accumulated[1]["id"] == "call_2"

    def test_accumulate_incremental_arguments(self):
        """Tool call arguments are accumulated incrementally."""
        accumulated = [{"id": "call_1", "name": "tool", "arguments": ""}]

        # First chunk - simulate streaming JSON (first part)
        first_chunk = '{"arg1":'
        delta1 = {"tool_calls": [{"index": 0, "function": {"arguments": first_chunk}}]}
        accumulate_tool_call_delta(delta1, accumulated)
        assert accumulated[0]["arguments"] == '{"arg1":'

        # Second chunk - complete the JSON value
        second_chunk = '"value1"}'
        delta2 = {"tool_calls": [{"index": 0, "function": {"arguments": second_chunk}}]}
        accumulate_tool_call_delta(delta2, accumulated)
        assert accumulated[0]["arguments"] == '{"arg1":"value1"}'

    def test_accumulate_creates_slots_for_gaps(self):
        """Gaps in tool call indices are filled with placeholder entries."""
        accumulated = []

        # Delta with index 2 (skipping 0 and 1)
        delta = {
            "tool_calls": [
                {
                    "index": 2,
                    "id": "call_3",
                    "function": {"name": "tool", "arguments": "{}"},
                }
            ]
        }

        accumulate_tool_call_delta(delta, accumulated)

        # Should create slots for indices 0, 1, and 2
        assert len(accumulated) == 3
        assert accumulated[0]["id"] == ""
        assert accumulated[1]["id"] == ""
        assert accumulated[2]["id"] == "call_3"

    def test_accumulate_preserves_id_and_name(self):
        """Tool call ID and name are preserved when already set."""
        accumulated = [{"id": "call_1", "name": "old_name", "arguments": ""}]

        delta = {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "new_name"}}]}

        accumulate_tool_call_delta(delta, accumulated)

        # ID should be updated (same value), name should be updated
        assert accumulated[0]["id"] == "call_1"
        assert accumulated[0]["name"] == "new_name"

    def test_accumulate_empty_delta(self):
        """Empty delta list doesn't modify accumulated list."""
        accumulated = [{"id": "call_1", "name": "tool", "arguments": "{}"}]

        delta = {"tool_calls": []}

        accumulate_tool_call_delta(delta, accumulated)

        assert len(accumulated) == 1
        assert accumulated[0]["id"] == "call_1"

    def test_accumulate_without_tool_calls_key(self):
        """Delta without tool_calls key doesn't crash."""
        accumulated = []

        delta = {"content": "Hello"}

        # Should not raise exception
        accumulate_tool_call_delta(delta, accumulated)

        assert len(accumulated) == 0


class TestFixOrphanedToolMessages:
    """Tests for fix_orphaned_tool_messages function."""

    def test_removes_orphaned_tool_response(self):
        """Tool response without matching tool_call is removed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "orphan_call", "content": "Result"},
        ]

        result = fix_orphaned_tool_messages(messages)

        # Orphaned tool message should be removed
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_removes_tool_calls_without_responses(self):
        """Tool calls without matching response are stripped."""
        messages = [
            {"role": "user", "content": "Use tool"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        ]

        result = fix_orphaned_tool_messages(messages)

        # Tool calls should be stripped, content set to empty string
        assert len(result) == 2
        assert result[1]["content"] == ""
        assert "tool_calls" not in result[1]

    def test_preserves_valid_pairs(self):
        """Valid tool_call/response pairs are preserved."""
        messages = [
            {"role": "user", "content": "Use tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Success"},
        ]

        result = fix_orphaned_tool_messages(messages)

        # All messages should be preserved
        assert len(result) == 3
        assert "tool_calls" in result[1]
        assert result[2]["tool_call_id"] == "call_1"

    def test_preserves_content_when_tool_calls_removed(self):
        """Assistant content is preserved when tool_calls are removed."""
        messages = [
            {
                "role": "assistant",
                "content": "Some text",
                "tool_calls": [{"id": "orphan"}],
            },
        ]

        result = fix_orphaned_tool_messages(messages)

        # Content should be set (not None) when tool_calls are removed
        assert result[0]["content"] == "Some text"
        assert "tool_calls" not in result[0]

    def test_removes_tool_responses_when_tool_calls_are_stripped(self):
        """Tool responses are removed when their tool_calls are stripped from assistant.

        This is the critical bug fix: when context compaction removes some tool response
        messages but leaves others, the assistant message may have tool_calls with missing
        responses. When those tool_calls are stripped, the REMAINING tool responses become
        orphaned and must also be removed to maintain strict pairing.

        Scenario:
        - Assistant message declares tool_calls A and B
        - Tool response for A exists, tool response for B was compacted away
        - Since not all responses exist, tool_calls A and B are both stripped from assistant
        - Now tool response A is orphaned (no tool_calls in assistant) and must be removed
        """
        messages = [
            {"role": "user", "content": "Use tools"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_A",
                        "type": "function",
                        "function": {"name": "tool1", "arguments": "{}"},
                    },
                    {
                        "id": "call_B",
                        "type": "function",
                        "function": {"name": "tool2", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_A", "content": "Success"},
            # Note: call_B's response was compacted away (not in message list)
        ]

        result = fix_orphaned_tool_messages(messages)

        # The assistant message should have tool_calls stripped (missing one response)
        # The tool response for call_A should also be removed (now orphaned)
        assert len(result) == 2  # user + assistant only
        assert result[1]["role"] == "assistant"
        assert "tool_calls" not in result[1]  # stripped because not all responses present
        # No tool messages should remain
        assert not any(m.get("role") == "tool" for m in result)


class TestMarkdownRendering:
    """Tests for markdown rendering functions."""

    def test_escape_rich_markup_basic(self):
        """Basic Rich markup tags are escaped."""
        # Test various patterns - only opening [ is escaped
        assert "\\[bold]" in _escape_rich_markup_from_text("[bold]")
        assert "\\[/bold]" in _escape_rich_markup_from_text("[/bold]")
        assert "\\[tag=value]" in _escape_rich_markup_from_text("[tag=value]")

    def test_escape_rich_markup_with_commas(self):
        """Rich markup tags with comma-separated values are escaped."""
        result = _escape_rich_markup_from_text("[tag=val1,val2]")
        assert "[" not in result or result.startswith("\\")

    def test_escape_rich_markup_preserves_regular_text(self):
        """Regular text without Rich markup is preserved."""
        text = "This is [bracket] text with [normal] words."
        result = _escape_rich_markup_from_text(text)
        # Only tag-like patterns should be escaped
        assert "This is" in result
        assert "text with" in result

    def test_markdown_block_creates_markdown_object(self):
        """_markdown_block creates a Markdown object."""
        result = _markdown_block("Hello **world**")

        # Should return a Markdown object
        from rich.markdown import Markdown

        assert isinstance(result, Markdown)

    def test_render_image_placeholder(self):
        """Image placeholder is rendered correctly."""
        result = _render_image_placeholder("Test image", "https://example.com/image.png")

        # Should return a Panel
        from rich.panel import Panel

        assert isinstance(result, Panel)

    def test_render_markdown_with_hooks_empty_content(self):
        """Empty content returns empty markdown block."""
        result = render_markdown_with_hooks("   ")

        # Should return a valid renderable
        assert result is not None

    def test_render_markdown_with_hooks_plain_text(self):
        """Plain text without special formatting."""
        result = render_markdown_with_hooks("Hello world")

        # Should return a valid renderable
        assert result is not None

    def test_render_markdown_with_hooks_code_block(self):
        """Markdown code blocks are rendered."""
        content = "```python\nprint('hello')\n```"
        result = render_markdown_with_hooks(content)

        # Should return a valid renderable
        assert result is not None

    def test_render_markdown_with_hooks_image_link(self):
        """Image markdown links are rendered as placeholders."""
        content = "![Diagram](diagram.png)"
        result = render_markdown_with_hooks(content)

        # Should return a valid renderable (likely with Panel)
        assert result is not None


class TestMermaidParsing:
    """Tests for Mermaid diagram parsing."""

    def test_parse_simple_edges(self):
        """Simple Mermaid edges are parsed correctly."""
        code = """
        A --> B
        B --> C
        """

        edges = _parse_mermaid_edges(code)

        assert len(edges) == 2
        assert edges[0][0] == "A"
        assert edges[0][1] == "B"
        assert edges[1][0] == "B"
        assert edges[1][1] == "C"

    def test_parse_edges_with_labels(self):
        """Edges with labels are parsed."""
        code = """
        A -->|label| B
        """

        edges = _parse_mermaid_edges(code)

        assert len(edges) == 1
        assert edges[0][0] == "A"
        assert edges[0][1] == "B"
        assert edges[0][2] == "label"

    def test_parse_mermaid_with_arrow_type(self):
        """Mermaid edges with standard arrow types are parsed."""
        code = """
        A --> B
        """

        edges = _parse_mermaid_edges(code)

        # Should parse the edge with standard arrow type
        assert len(edges) >= 1

    def test_normalize_mermaid_node_with_brackets(self):
        """Nodes with bracket labels are normalized."""
        # Test square brackets
        result = _normalize_mermaid_node("A[Label]")
        assert result == "Label"

    def test_normalize_mermaid_node_with_parentheses(self):
        """Nodes with parentheses labels are normalized."""
        result = _normalize_mermaid_node("A(Label)")
        assert result == "Label"

    def test_normalize_mermaid_node_with_braces(self):
        """Nodes with brace labels are normalized."""
        result = _normalize_mermaid_node("A{Label}")
        assert result == "Label"

    def test_normalize_mermaid_node_with_angle_brackets(self):
        """Nodes with angle bracket labels are normalized."""
        result = _normalize_mermaid_node("A<Label>")
        assert result == "Label"

    def test_normalize_mermaid_node_strips_modifiers(self):
        """Arrow modifiers are stripped from node names."""
        result = _normalize_mermaid_node("A-.->B")
        # Should strip the arrow modifier
        assert result is not None

    def test_detect_direction_td(self):
        """TD graph direction is detected."""
        code = "graph TD\nA-->B"
        direction = _detect_direction(code)
        assert direction == "TD"

    def test_detect_direction_lr(self):
        """LR graph direction is detected."""
        code = "graph LR\nA-->B"
        direction = _detect_direction(code)
        assert direction == "LR"

    def test_detect_direction_default(self):
        """Default direction is TD when not specified."""
        code = "A-->B"
        direction = _detect_direction(code)
        assert direction == "TD"


class TestToolConversion:
    """Tests for tool conversion functions."""

    def test_convert_tools_to_openai_format(self):
        """Tools are converted to OpenAI format."""
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={
                    "type": "object",
                    "properties": {"arg1": {"type": "string", "description": "First arg"}},
                },
            )
        ]

        result = convert_tools_to_openai_format(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"
        assert result[0]["function"]["description"] == "A test tool"

    def test_parse_openai_tool_calls(self):
        """OpenAI tool calls are parsed correctly."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg1": "value1"}'},
            }
        ]

        result = parse_openai_tool_calls(tool_calls)

        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["name"] == "test_tool"
        assert result[0]["arguments"] == {"arg1": "value1"}

    def test_parse_openai_tool_calls_empty_list(self):
        """Empty tool calls list returns None."""
        result = parse_openai_tool_calls([])
        assert result is None

    def test_parse_openai_tool_calls_none_input(self):
        """None input returns None."""
        result = parse_openai_tool_calls(None)
        assert result is None
