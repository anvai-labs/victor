# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Regression tests for the markdown escape-removal fix.

`_markdown_block` no longer pre-escapes content with `_escape_rich_markup_from_text`:
`rich.markdown.Markdown` doesn't interpret Rich inline markup, so escaping was
redundant and broke markdown links. These tests pin the corrected behavior.
"""

from io import StringIO

from rich.console import Console

from victor.ui.rendering.markdown import _markdown_block


def _render(md_obj) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=60, color_system="truecolor")
    console.print(md_obj)
    return buf.getvalue()


def test_markdown_link_not_escape_broken():
    # A markdown link must render without a backslash-escaped bracket (the old
    # escape turned [docs] into \[docs] and broke the link).
    out = _render(_markdown_block("see [docs](http://x) here"))
    assert "\\[" not in out
    assert "docs" in out


def test_rich_inline_markup_not_interpreted():
    # Rich [tags] in markdown text must NOT be interpreted as color/style.
    out = _render(_markdown_block("color [red]word[/] now"))
    assert "\x1b[31m" not in out  # no red ANSI


def test_bracket_text_renders_literally():
    # Bracket-text (e.g. a filename in brackets) renders as text, no error.
    out = _render(_markdown_block("err [config.py]: boom"))
    assert "config.py" in out
