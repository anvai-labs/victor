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

"""Shared CLI/web markdown presenters (no Chainlit dependency)."""

from __future__ import annotations

from victor.ui.rendering.markdown_presenters import (
    preview_to_markdown,
    strip_rich_markup,
    tool_call_summary,
    tool_result_markdown,
    turn_cost_footer,
)
from victor.ui.rendering.tool_preview import RenderedPreview


def test_strip_rich_markup():
    assert strip_rich_markup("[green]+added[/] [dim]ctx[/]") == "+added ctx"


def test_tool_call_summary_includes_args():
    summary = tool_call_summary("bash", {"command": "ls -la"})
    assert "command" in summary and "ls -la" in summary
    assert tool_call_summary("read", {}) == "read()"


def test_preview_to_markdown_diff_fence_and_header():
    preview = RenderedPreview(
        lines=["+ added", "- removed"],
        header="+1 -1 file.py",
        total_line_count=2,
        syntax_hint="diff",
    )
    md = preview_to_markdown(preview)
    assert "```diff" in md
    assert "+ added" in md and "- removed" in md
    assert "**+1 -1 file.py**" in md


def test_preview_to_markdown_strips_rich_markup_lines():
    preview = RenderedPreview(
        lines=["[green]+ a[/]", "[red]- b[/]"],
        total_line_count=2,
        syntax_hint="diff",
        contains_rich_markup=True,
    )
    md = preview_to_markdown(preview)
    assert "[green]" not in md and "[/]" not in md
    assert "+ a" in md and "- b" in md


def test_preview_to_markdown_truncation_footer():
    preview = RenderedPreview(lines=["line 1"], total_line_count=10, syntax_hint="text")
    md = preview_to_markdown(preview)
    assert "9 more lines" in md


def test_tool_result_markdown_renders_a_fence():
    md = tool_result_markdown("ls", {"path": "."}, "file1\nfile2", success=True)
    assert "```" in md
    assert "file1" in md or "file" in md


def test_tool_result_markdown_marks_failure():
    md = tool_result_markdown("bash", {"command": "false"}, "boom", success=False)
    assert "failed" in md.lower()


def test_tool_result_markdown_never_raises_on_bad_result():
    # A non-string, non-dict result must not crash the presenter.
    md = tool_result_markdown("weird", None, object(), success=True)
    assert isinstance(md, str) and md


def test_turn_cost_footer_renders_tokens_latency_cost():
    footer = turn_cost_footer(
        {
            "api_total_tokens": 1728,
            "api_prompt_tokens": 1675,
            "api_completion_tokens": 53,
            "duration_seconds": 4.2,
            "request_count": 3,
            "total_cost_usd": 0.0012,
            "cache_hit_rate": 0.0,
        }
    )
    assert "1,728 tok" in footer
    assert "1,675↑" in footer and "53↓" in footer
    assert "4.2s" in footer
    assert "3 calls" in footer
    assert "$0.0012" in footer


def test_turn_cost_footer_empty_when_nothing_measured():
    # A turn with no provider call (e.g. a Q&A bypass) must not render a misleading "0 tok".
    assert (
        turn_cost_footer({"api_total_tokens": 0, "duration_seconds": 0.0, "request_count": 0}) == ""
    )
    assert turn_cost_footer({}) == ""
    assert turn_cost_footer(None) == ""


def test_turn_cost_footer_omits_zero_fields_and_singular_call():
    footer = turn_cost_footer(
        {
            "api_total_tokens": 500,
            "api_prompt_tokens": 480,
            "api_completion_tokens": 20,
            "duration_seconds": 1.0,
            "request_count": 1,
            "total_cost_usd": 0.0,
        }
    )
    assert "1 call" in footer and "calls" not in footer  # singular
    assert "💵" not in footer  # zero cost omitted
    assert "cached" not in footer  # zero cache omitted


def test_turn_cost_footer_shows_cache_and_dollar_precision():
    footer = turn_cost_footer(
        {
            "api_total_tokens": 100,
            "api_prompt_tokens": 90,
            "api_completion_tokens": 10,
            "duration_seconds": 2.0,
            "request_count": 1,
            "total_cost_usd": 0.25,
            "cache_hit_rate": 0.9,
        }
    )
    assert "$0.25" in footer  # >= 1c uses 2dp
    assert "90% cached" in footer
