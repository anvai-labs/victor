"""Tests for markdown rendering fixes to prevent TUI overflow.

Covers the scoped code block renderer (word_wrap + reduced padding)
and the enhanced table overflow prevention (no_wrap=False, overflow='fold').
"""

import io
import re

from rich.console import Console

from victor.ui.rendering.markdown import _markdown_block, render_enhanced_table


def _render_to_string(renderable, width: int = 80) -> str:
    """Render a Rich renderable to a string at a given width."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=width, height=25)
    console.print(renderable)
    return buf.getvalue()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestPatchedCodeBlockRender:
    """Tests for the patched CodeBlock renderer (word_wrap + reduced padding)."""

    def test_code_block_renders_via_markdown(self) -> None:
        """Code blocks in markdown render without errors."""
        md = _markdown_block("```python\nx = 1\n```")
        output = _render_to_string(md)
        assert "x" in output
        assert "1" in output

    def test_code_block_wraps_long_lines(self) -> None:
        """Long lines in code blocks wrap within the console width."""
        long_line = "x = '" + "a" * 200 + "'"
        md = _markdown_block(f"```python\n{long_line}\n```")
        output = _render_to_string(md, width=60)
        for line in output.splitlines():
            clean = _strip_ansi(line)
            assert len(clean) <= 62, f"Line too wide ({len(clean)}): {clean[:80]}"

    def test_renderer_preserves_full_wrapped_content(self) -> None:
        output = _strip_ansi(
            _render_to_string(_markdown_block("```text\n" + "x" * 200 + "\n```"), width=40)
        )
        assert output.count("x") == 200
        assert all(len(line) <= 40 for line in output.splitlines())


class TestEnhancedTableOverflow:
    """Tests for render_enhanced_table overflow prevention."""

    def test_table_renders(self) -> None:
        """Basic table renders without errors."""
        table = render_enhanced_table(
            ["Symbol", "File"],
            [["BaseTool", "victor/tools/base.py"]],
        )
        output = _render_to_string(table)
        assert "Symbol" in output
        assert "BaseTool" in output

    def test_table_long_content_wraps(self) -> None:
        """Long cell content wraps instead of overflowing."""
        long_desc = "A" * 150
        table = render_enhanced_table(
            ["Name", "Description"],
            [["TestSymbol", long_desc]],
        )
        output = _render_to_string(table, width=60)
        assert "TestSymbol" in output
        for line in output.splitlines():
            clean = _strip_ansi(line)
            assert len(clean) <= 65, f"Line too wide ({len(clean)}): {clean[:80]}"

    def test_table_columns_allow_wrap(self) -> None:
        """Columns are configured with no_wrap=False and overflow='fold'."""
        from rich.table import Table

        table = render_enhanced_table(["H1", "H2"], [["a", "b"]])
        assert isinstance(table, Table)
        for col in table.columns:
            assert col.no_wrap is False
            assert col.overflow == "fold"


def test_importing_victor_renderer_keeps_rich_code_block_behavior_local():
    from rich.markdown import CodeBlock

    assert CodeBlock.__rich_console__.__module__ == "rich.markdown"
