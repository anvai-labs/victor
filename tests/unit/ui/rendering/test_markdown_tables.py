"""Markdown tables must retain every cell when the conversation pane narrows."""

from io import StringIO
from collections import Counter

import pytest
from rich.console import Console
from rich.markdown import Markdown, TableElement

from victor.ui.rendering.markdown import render_markdown_with_hooks


@pytest.mark.parametrize("width", [16, 32, 60, 90, 140])
def test_tables_preserve_all_cells_and_respect_width(width):
    content = (
        "Before table\n\n"
        "| Name | File | Connections | Issue |\n"
        "| :--- | --- | ---: | :---: |\n"
        "| **BaseTool** | `victor/tools/very_long_unbroken_file_name.py` | 43 | Preserve relationships |\n"
        "| Other | [link](https://example.com) | 12 | escaped \\| pipe |\n\n"
        "After table\n"
    )
    output = StringIO()
    console = Console(file=output, width=width, color_system=None)
    console.print(render_markdown_with_hooks(content))
    text = output.getvalue()
    expected = "Name File Connections Issue BaseTool victor/tools/very_long_unbroken_file_name.py 43 Preserve relationships Other link 12 escaped | pipe Before table After table"
    # Wrapped table cells interleave by screen row. Compare all characters,
    # then reconstruct the styled inline-code cell independently of columns.
    assert not (Counter("".join(expected.split())) - Counter("".join(text.split())))
    code_style = console.get_style("markdown.code")
    code = "".join(
        segment.text
        for segment in console.render(render_markdown_with_hooks(content))
        if segment.style and segment.style.bgcolor == code_style.bgcolor
    )
    assert "".join(code.split()) == "victor/tools/very_long_unbroken_file_name.py"
    assert "…" not in text
    assert all(len(line) <= width for line in text.splitlines())
    assert Markdown.elements["table_open"] is TableElement


def test_fenced_table_syntax_stays_code():
    output = StringIO()
    Console(file=output, width=80, color_system=None).print(
        render_markdown_with_hooks("```text\n| A | B |\n| --- | --- |\n| x | y |\n```")
    )
    assert "| --- | --- |" in output.getvalue()
