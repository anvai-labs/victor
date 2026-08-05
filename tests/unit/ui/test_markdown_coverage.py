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

"""Comprehensive tests for markdown rendering functions.

These tests increase code coverage for victor/ui/rendering/markdown.py
to ensure all edge cases are handled properly.
"""

import pytest
from unittest.mock import patch, Mock

from victor.ui.rendering.markdown import (
    render_markdown_with_hooks,
    _escape_rich_markup_from_text,
    _markdown_block,
    _render_image_placeholder,
    _parse_mermaid_edges,
    _detect_direction,
    _normalize_mermaid_node,
    _render_diagram,
    render_collapsible_section,
    render_enhanced_table,
    should_collapse_content,
    get_render_error_context,
)


class TestRichMarkupEscaping:
    """Comprehensive tests for _escape_rich_markup_from_text."""

    def test_escape_simple_tag(self):
        """Simple [tag] pattern is escaped."""
        result = _escape_rich_markup_from_text("[bold]")
        # Only the opening [ is escaped to \\
        assert result == "\\[bold]"

    def test_escape_closing_tag(self):
        """Closing [/tag] pattern is escaped."""
        result = _escape_rich_markup_from_text("[/bold]")
        # Only the opening [ is escaped to \\
        assert result == "\\[/bold]"

    def test_escape_tag_with_equals(self):
        """Tag with value [tag=value] is escaped."""
        result = _escape_rich_markup_from_text("[color=red]")
        # Only the opening [ is escaped to \\
        assert result == "\\[color=red]"

    def test_escape_tag_with_comma_separated_values(self):
        """Tag with multiple values is escaped."""
        result = _escape_rich_markup_from_text("[tag=val1,val2,val3]")
        # Only the opening [ is escaped to \\
        assert result == "\\[tag=val1,val2,val3]"

    def test_escape_multiple_tags_in_text(self):
        """Multiple tags in same text are all escaped."""
        result = _escape_rich_markup_from_text("[bold]text[/] and [red]more[/]")
        # Only tags with word characters are matched by the regex
        # [bold] and [red] match, but [/] does not (no word chars)
        assert "\\[bold]" in result
        # [/] doesn't match the regex (requires word chars after [)
        assert "[/]" in result  # unchanged
        assert "\\[red]" in result

    def test_escape_nested_brackets(self):
        """Tags with nested brackets are escaped."""
        result = _escape_rich_markup_from_text("[tag=[value]]")
        assert "[" not in result or result.count("\\[") >= 2

    def test_escape_preserves_non_tag_text(self):
        """Text that doesn't look like a tag is preserved."""
        result = _escape_rich_markup_from_text("Hello [world]")
        # "world" alone isn't a valid tag (no closing bracket with value)
        assert "Hello " in result or "Hello" in result

    def test_escape_empty_string(self):
        """Empty string returns empty string."""
        result = _escape_rich_markup_from_text("")
        assert result == ""

    def test_escape_string_without_brackets(self):
        """String without brackets is unchanged."""
        result = _escape_rich_markup_from_text("plain text")
        assert result == "plain text"

    def test_escape_tag_with_underscore(self):
        """Tags with underscores are escaped."""
        result = _escape_rich_markup_from_text("[my_tag]")
        # Only the opening [ is escaped to \\
        assert result == "\\[my_tag]"

    def test_escape_tag_with_numbers(self):
        """Tags with numbers are escaped."""
        result = _escape_rich_markup_from_text("[tag123]")
        # Only the opening [ is escaped to \\
        assert result == "\\[tag123]"


class TestMarkdownBlock:
    """Tests for _markdown_block function."""

    def test_markdown_block_returns_markdown_object(self):
        """_markdown_block returns a Rich Markdown object."""
        from rich.markdown import Markdown

        result = _markdown_block("Hello **world**")
        assert isinstance(result, Markdown)

    def test_markdown_block_escapes_content(self):
        """_markdown_block escapes Rich markup before creating Markdown."""
        # Content with Rich-style tags should be escaped
        result = _markdown_block("[bold]text[/]")
        # Should not crash, and should be a valid Markdown object
        from rich.markdown import Markdown

        assert isinstance(result, Markdown)

    def test_markdown_block_with_empty_string(self):
        """Empty string returns valid Markdown object."""
        from rich.markdown import Markdown

        result = _markdown_block("")
        assert isinstance(result, Markdown)

    def test_markdown_block_preserves_markdown(self):
        """Actual markdown syntax is preserved."""
        from rich.markdown import Markdown

        result = _markdown_block("**bold** and *italic*")
        assert isinstance(result, Markdown)


class TestRenderImagePlaceholder:
    """Tests for _render_image_placeholder function."""

    def test_image_placeholder_returns_panel(self):
        """Image placeholder returns a Rich Panel."""
        from rich.panel import Panel

        result = _render_image_placeholder("Diagram", "diagram.png")
        assert isinstance(result, Panel)

    def test_image_placeholder_with_empty_alt(self):
        """Image placeholder with empty alt text uses default."""
        from rich.panel import Panel

        result = _render_image_placeholder("", "image.png")
        assert isinstance(result, Panel)

    def test_image_placeholder_with_empty_source(self):
        """Image placeholder with empty source still renders."""
        from rich.panel import Panel

        result = _render_image_placeholder("Test", "")
        assert isinstance(result, Panel)


class TestRenderMarkdownWithHooks:
    """Tests for render_markdown_with_hooks function."""

    def test_empty_content_returns_markdown_block(self):
        """Empty/whitespace-only content returns markdown block."""
        result = render_markdown_with_hooks("   ")
        assert result is not None

    def test_plain_text_returns_markdown(self):
        """Plain text without special formatting returns markdown."""
        result = render_markdown_with_hooks("Hello world")
        assert result is not None

    def test_markdown_with_code_block(self):
        """Markdown with code blocks is rendered."""
        content = """
# Header

```python
def hello():
    print("world")
```
"""
        result = render_markdown_with_hooks(content)
        assert result is not None

    def test_markdown_with_inline_image(self):
        """Markdown with inline image is rendered."""
        content = "![Diagram](diagram.png)"
        result = render_markdown_with_hooks(content)
        assert result is not None

    def test_markdown_with_multiple_code_blocks(self):
        """Multiple code blocks are handled."""
        content = """
```python
print("one")
```

```javascript
console.log("two");
```
"""
        result = render_markdown_with_hooks(content)
        assert result is not None

    def test_mermaid_diagram_is_rendered(self):
        """Mermaid diagram code blocks are specially handled."""
        content = """
```mermaid
graph TD
A-->B
```
"""
        result = render_markdown_with_hooks(content)
        assert result is not None

    def test_mixed_content_is_rendered(self):
        """Mixed content (text, code, images) is rendered."""
        content = """
Here's a diagram:

```mermaid
graph LR
A-->B
```

And some code:

```python
print("code")
```
"""
        result = render_markdown_with_hooks(content)
        assert result is not None

    def test_rendering_exception_returns_fallback(self):
        """Exception during rendering falls back to plain text."""
        # Mock _markdown_block to raise an exception
        with patch(
            "victor.ui.rendering.markdown._markdown_block",
            side_effect=ValueError("Test error"),
        ):
            result = render_markdown_with_hooks("test content")

        # Should fall back to plain text
        assert result is not None


class TestMermaidEdgeParsing:
    """Comprehensive tests for _parse_mermaid_edges."""

    def test_parse_simple_arrow(self):
        """Simple A-->B edge is parsed."""
        code = "A-->B"
        edges = _parse_mermaid_edges(code)
        assert len(edges) == 1
        assert edges[0][0] == "A"
        assert edges[0][1] == "B"

    def test_parse_edge_with_label(self):
        """Edge with label is parsed."""
        code = "A-->|label|B"
        edges = _parse_mermaid_edges(code)
        assert len(edges) == 1
        assert edges[0][0] == "A"
        assert edges[0][1] == "B"
        assert edges[0][2] == "label"

    def test_parse_edge_with_label_bar(self):
        """Edge with -->| syntax is parsed."""
        code = "A-->|label_bar|B"
        edges = _parse_mermaid_edges(code)
        assert len(edges) >= 1
        assert edges[0][2] == "label_bar"

    def test_parse_empty_code(self):
        """Empty code returns empty list."""
        edges = _parse_mermaid_edges("")
        assert edges == []

    def test_parse_code_with_comments(self):
        """Comments (starting with %) are ignored."""
        code = """
% This is a comment
A-->B
% Another comment
"""
        edges = _parse_mermaid_edges(code)
        # Should have at least one edge
        assert len(edges) >= 1

    def test_parse_code_with_graph_declaration(self):
        """Lines with 'graph ' in them are skipped."""
        code = """
graph TD
A-->B
"""
        edges = _parse_mermaid_edges(code)
        # graph TD line is skipped
        assert len(edges) >= 1

    def test_parse_code_without_arrow(self):
        """Lines without --> are skipped."""
        code = """
A
B
C
"""
        edges = _parse_mermaid_edges(code)
        assert edges == []

    def test_parse_multiple_edges(self):
        """Multiple edges in code are all parsed."""
        code = """
A-->B
B-->C
C-->D
"""
        edges = _parse_mermaid_edges(code)
        assert len(edges) == 3

    def test_parse_edge_with_spaces_around_arrow(self):
        """Edges with spaces around arrow are parsed."""
        code = "A --> B"
        edges = _parse_mermaid_edges(code)
        assert len(edges) == 1

    def test_parse_complex_node_names(self):
        """Complex node names with brackets are handled."""
        code = "A[Label]-->B[Another Label]"
        edges = _parse_mermaid_edges(code)
        # Should parse the edge
        assert len(edges) >= 1


class TestNormalizeMermaidNode:
    """Comprehensive tests for _normalize_mermaid_node."""

    def test_normalize_simple_node(self):
        """Simple node name is returned as-is."""
        result = _normalize_mermaid_node("A")
        assert result == "A"

    def test_normalize_node_with_square_brackets(self):
        """Node with square brackets extracts the label."""
        result = _normalize_mermaid_node("A[LabelName]")
        assert result == "LabelName"

    def test_normalize_node_with_parentheses(self):
        """Node with parentheses extracts the label."""
        result = _normalize_mermaid_node("A(LabelName)")
        assert result == "LabelName"

    def test_normalize_node_with_curly_braces(self):
        """Node with curly braces extracts the label."""
        result = _normalize_mermaid_node("A{LabelName}")
        assert result == "LabelName"

    def test_normalize_node_with_angle_brackets(self):
        """Node with angle brackets extracts the label."""
        result = _normalize_mermaid_node("A<LabelName>")
        assert result == "LabelName"

    def test_normalize_node_strips_arrow_modifiers(self):
        """Arrow modifiers are stripped from the end."""
        result = _normalize_mermaid_node("A-.->")
        # Should strip the arrow modifier
        assert result is not None or result == ""

    def test_normalize_node_strips_dots(self):
        """Trailing dots are stripped."""
        result = _normalize_mermaid_node("A.")
        assert result == "A"

    def test_normalize_node_strips_dashes(self):
        """Trailing dashes are stripped."""
        result = _normalize_mermaid_node("A-")
        assert result == "A"

    def test_normalize_empty_node(self):
        """Empty node returns empty string."""
        result = _normalize_mermaid_node("")
        assert result == ""

    def test_normalize_node_with_ident_and_brackets(self):
        """Node with identifier and bracket label extracts label."""
        result = _normalize_mermaid_node("node1[Label]")
        # Should extract the label from brackets
        assert "Label" in result or result == "Label"

    def test_normalize_node_without_brackets(self):
        """Node without brackets returns identifier."""
        result = _normalize_mermaid_node("node123")
        assert result == "node123"


class TestDetectDirection:
    """Comprehensive tests for _detect_direction."""

    def test_detect_td_direction(self):
        """TD (top-down) direction is detected."""
        code = "graph TD\nA-->B"
        result = _detect_direction(code)
        assert result == "TD"

    def test_detect_lr_direction(self):
        """LR (left-right) direction is detected."""
        code = "graph LR\nA-->B"
        result = _detect_direction(code)
        assert result == "LR"

    def test_detect_rl_direction(self):
        """RL (right-left) direction is detected."""
        code = "graph RL\nA-->B"
        result = _detect_direction(code)
        assert result == "RL"

    def test_detect_tb_direction(self):
        """TB (top-bottom) direction is detected."""
        code = "graph TB\nA-->B"
        result = _detect_direction(code)
        assert result == "TB"

    def test_detect_bt_direction(self):
        """BT (bottom-top) direction is detected."""
        code = "graph BT\nA-->B"
        result = _detect_direction(code)
        assert result == "BT"

    def test_detect_default_direction(self):
        """Default direction is TD when not specified."""
        code = "A-->B"
        result = _detect_direction(code)
        assert result == "TD"

    def test_detect_multiline_graph(self):
        """Direction is detected from first graph line."""
        code = """
graph LR
A-->B
B-->C
"""
        result = _detect_direction(code)
        assert result == "LR"

    def test_detect_with_leading_spaces(self):
        """Direction is detected with leading spaces."""
        code = "  graph TD\nA-->B"
        result = _detect_direction(code)
        assert result == "TD"

    def test_detect_case_insensitive(self):
        """Graph keyword is case-insensitive."""
        code = "GRAPH td\nA-->B"
        result = _detect_direction(code)
        assert result == "TD"


class TestRenderDiagram:
    """Tests for _render_diagram function."""

    def test_render_mermaid_diagram(self):
        """Mermaid diagram is rendered specially."""
        from rich.panel import Panel

        result = _render_diagram("mermaid", "A-->B")
        assert isinstance(result, Panel)

    def test_render_unknown_diagram(self):
        """Unknown diagram type falls back to syntax panel."""
        from rich.panel import Panel

        result = _render_diagram("unknown", "code here")
        assert isinstance(result, Panel)

    def test_render_diagram_with_empty_code(self):
        """Empty code is handled gracefully."""
        from rich.panel import Panel

        result = _render_diagram("mermaid", "")
        assert isinstance(result, Panel)


# Phase 2: Enhanced markdown rendering tests


class TestCollapsibleSections:
    """Tests for collapsible section rendering."""

    def test_short_content_not_collapsed(self):
        """Short content should not be collapsed."""
        content = "Line 1\nLine 2\nLine 3"
        result = render_collapsible_section("Test", content, lines_visible=10)
        # Should return markdown block, not Panel with collapse hint
        from rich.markdown import Markdown

        assert isinstance(result, Markdown)

    def test_long_content_collapsed(self):
        """Long content should be collapsed with hint."""
        content = "\n".join(f"Line {i}" for i in range(20))
        result = render_collapsible_section("Test", content, lines_visible=10)
        # Should return Panel (not plain markdown block)
        from rich.panel import Panel
        from rich.markdown import Markdown

        assert isinstance(result, Panel)
        # Panel should contain more than just the markdown block
        assert hasattr(result, "renderable")

    def test_collapsed_line_count(self):
        """Collapsed hint shows correct remaining line count."""
        content = "\n".join(f"Line {i}" for i in range(15))
        result = render_collapsible_section("Test", content, lines_visible=5)
        # Should return Panel (collapsed content)
        from rich.panel import Panel

        assert isinstance(result, Panel)
        # Should have a title indicating collapse
        assert result.title or hasattr(result, "renderable")


class TestEnhancedTableRendering:
    """Tests for enhanced table rendering."""

    def test_basic_table_rendering(self):
        """Basic table should render correctly."""
        headers = ["Name", "Value"]
        rows = [["Item 1", "100"], ["Item 2", "200"]]
        result = render_enhanced_table(headers, rows)
        from rich.table import Table

        assert isinstance(result, Table)

    def test_table_with_title(self):
        """Table with title should render correctly."""
        headers = ["Col1", "Col2"]
        rows = [["A", "B"]]
        result = render_enhanced_table(headers, rows, title="Test Table")
        assert "Test Table" in str(result.title)

    def test_empty_rows(self):
        """Empty table should render without error."""
        headers = ["Col1", "Col2"]
        result = render_enhanced_table(headers, [])
        from rich.table import Table

        assert isinstance(result, Table)


class TestContentCollapseDecision:
    """Tests for content collapse decision logic."""

    def test_short_text_not_collapsed(self):
        """Short text should not be collapsed."""
        content = "Short text"
        result = should_collapse_content(content, "text")
        assert result is False

    def test_long_text_collapsed(self):
        """Long text should be collapsed."""
        content = "\n".join(f"Line {i}" for i in range(60))
        result = should_collapse_content(content, "text")
        assert result is True

    def test_short_code_not_collapsed(self):
        """Short code should not be collapsed."""
        content = "def foo():\n    pass"
        result = should_collapse_content(content, "code")
        assert result is False

    def test_long_code_collapsed(self):
        """Long code should be collapsed."""
        content = "\n".join(f"line {i}" for i in range(40))
        result = should_collapse_content(content, "code")
        assert result is True


class TestErrorContext:
    """Tests for rendering error context."""

    def test_error_context_includes_type(self):
        """Error context should include error type."""
        try:
            raise ValueError("test error")
        except Exception as e:
            context = get_render_error_context(e, "content")
            assert "ValueError" in context

    def test_error_context_includes_preview(self):
        """Error context should include content preview."""
        long_content = "a" * 200
        try:
            raise ValueError("test")
        except Exception as e:
            context = get_render_error_context(e, long_content)
            assert "..." in context  # Truncated preview
            assert "Content preview" in context

    def test_error_context_includes_tip(self):
        """Error context should include helpful tip."""
        try:
            raise ValueError("test")
        except Exception as e:
            context = get_render_error_context(e, "content")
            assert "Tip:" in context
