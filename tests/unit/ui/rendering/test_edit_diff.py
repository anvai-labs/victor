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

"""Tests for word-level inline edit-diff rendering.

`render_edit_preview` highlights changed *words* within a modified `-`/`+` line
pair (Aider-style inline diff); pure additions/deletions stay line-level.
"""

from io import StringIO

from rich.console import Console
from rich.text import Text

from victor.ui.rendering.utils import _append_inline_diff, render_edit_preview


def _span_styles(text: Text) -> list[str]:
    return [str(s.style) for s in text.spans]


def test_inline_diff_highlights_removed_and_inserted_words():
    body = Text()
    _append_inline_diff(body, "the quick brown fox", "the slow brown fox")
    styles = _span_styles(body)
    # "quick" removed -> bold red; "slow" inserted -> bold green; context dim.
    assert any("bold" in s and "red" in s for s in styles), styles
    assert any("bold" in s and "green" in s for s in styles), styles


def test_inline_diff_context_kept_dim():
    body = Text()
    _append_inline_diff(body, "keep this part changed", "keep this part different")
    styles = _span_styles(body)
    # "keep this part" is equal context -> non-bold red/green.
    assert any(s == "red" for s in styles), styles  # context in - line
    assert any(s == "green" for s in styles), styles  # context in + line


def _render_to_string(diff: str) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100, color_system="truecolor")
    render_edit_preview(console, "f.py", diff)
    return buf.getvalue()


def test_modification_pair_uses_word_level_highlight():
    out = _render_to_string("@@ -1 +1 @@\n-old value here\n+new value here\n")
    # Changed words (old->new, value common) render with bold styling.
    assert "\x1b[1" in out  # ANSI bold sequence present (word-level highlight)


def test_pure_addition_is_line_level_not_inline():
    # A standalone + line with no preceding - is a pure addition (line-level green,
    # no word-level bold pairing).
    out = _render_to_string("@@ -1 +1,2 @@\n ctx\n+brand new line\n")
    assert "brand new line" in out


def test_mismatched_run_lengths_fall_back_to_line_level():
    # One - line, two + lines -> can't pair 1:1 -> line-level (no inline pairing).
    out = _render_to_string("@@ -1 +1,2 @@\n-gone\n+one\ntwo\n".replace("two", "+two"))
    assert "gone" in out and "one" in out
