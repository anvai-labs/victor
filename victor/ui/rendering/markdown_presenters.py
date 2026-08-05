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

"""Markdown presenters — the shared seam between the CLI (Rich) and web (Markdown) surfaces.

The terminal renderer turns a tool result into Rich markup; the Chainlit web app needs
**markdown** (code fences, bold) instead. Both, however, can share the format-agnostic
``RenderedPreview`` produced by :class:`~victor.ui.rendering.tool_preview.ToolPreviewRenderer`
and the canonical formatters in :mod:`victor.ui.rendering.utils`. This module converts that
shared data into markdown, so the web UI matches CLI fidelity (syntax-highlighted diffs, file
reads, search results, …) without re-implementing any tool-specific preview logic.

Pure and dependency-light (no Chainlit, no Rich Console), so it is unit-testable directly.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from victor.ui.rendering.tool_preview import RenderedPreview, ToolPreviewRenderer
from victor.ui.rendering.utils import format_tool_args, format_tool_display_name

# Strips Rich markup tags like ``[green]`` / ``[/]`` so preview lines render cleanly in
# markdown (mirrors the inline patterns already used in tool_preview / formatter_aware_preview).
_RICH_TAG = re.compile(r"\[/?[^\]]*\]")

# One shared renderer instance (registry of per-tool preview strategies).
_RENDERER = ToolPreviewRenderer()


def strip_rich_markup(text: str) -> str:
    """Remove Rich markup tags from *text* for safe markdown rendering."""
    return _RICH_TAG.sub("", text)


def tool_call_summary(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    """A compact ``displayName(arg=val, …)`` label for a tool step or approval card."""
    name = format_tool_display_name(tool_name or "tool")
    args = format_tool_args(arguments or {}, max_width=160)
    return f"{name}({args})" if args else f"{name}()"


def preview_to_markdown(preview: RenderedPreview) -> str:
    """Render a format-agnostic ``RenderedPreview`` as markdown (header + fenced body + footer)."""
    parts: List[str] = []
    if preview.header:
        parts.append(f"**{strip_rich_markup(preview.header)}**")
    if preview.lines:
        clean = preview.contains_rich_markup
        body = "\n".join(strip_rich_markup(line) if clean else line for line in preview.lines)
        lang = "diff" if preview.syntax_hint in ("diff", "udiff") else (preview.syntax_hint or "")
        parts.append(f"```{lang}\n{body}\n```")
    remaining = preview.total_line_count - len(preview.lines)
    if remaining > 0:
        parts.append(f"_… {remaining} more line{'s' if remaining != 1 else ''}_")
    return "\n\n".join(parts)


def tool_result_markdown(
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    result: Any,
    *,
    success: bool = True,
    max_lines: int = 12,
) -> str:
    """Full markdown body for a completed tool call — a syntax-highlighted result preview.

    Delegates to the shared ``ToolPreviewRenderer`` so the web UI gets the same per-tool
    preview (diffs, file reads, search results, shell output) the CLI shows. Never raises:
    rendering failures fall back to a plain fenced block so a malformed result can't break chat.
    """
    try:
        preview = _RENDERER.render(tool_name or "tool", arguments or {}, result, max_lines)
        body = preview_to_markdown(preview)
    except Exception:
        text = result if isinstance(result, str) else str(result)
        body = f"```\n{text[:4000]}\n```"
    if not success:
        return f"⚠️ **failed**\n\n{body}" if body else "⚠️ **failed**"
    return body


def turn_cost_footer(report: Optional[Dict[str, Any]]) -> str:
    """Compact per-turn cost/latency footer from a C0 ``TaskExecutionReport`` dict.

    Surfaces the dominant cost term (tokens × round-trips) and latency/$ so the user can see
    what each turn cost — fed by ``VictorClient.get_last_turn_cost()``. Returns ``""`` when the
    record is empty or carries no measured usage (e.g. a turn that made no provider call), so
    callers can skip rendering an empty footer.
    """
    if not isinstance(report, dict) or not report:
        return ""

    total = int(report.get("api_total_tokens", 0) or 0)
    prompt = int(report.get("api_prompt_tokens", 0) or 0)
    completion = int(report.get("api_completion_tokens", 0) or 0)
    duration = float(report.get("duration_seconds", 0.0) or 0.0)
    requests = int(report.get("request_count", 0) or 0)
    cost = float(report.get("total_cost_usd", 0.0) or 0.0)
    cache_hit = float(report.get("cache_hit_rate", 0.0) or 0.0)

    # Nothing measured -> no footer (avoids a misleading "0 tok" line).
    if total == 0 and duration == 0.0 and requests == 0:
        return ""

    parts: List[str] = []
    if duration > 0:
        parts.append(f"⏱ {duration:.1f}s")
    if total > 0:
        parts.append(f"🔢 {total:,} tok ({prompt:,}↑ {completion:,}↓)")
    if requests > 0:
        parts.append(f"🔁 {requests} call{'s' if requests != 1 else ''}")
    if cost > 0:
        # Sub-cent costs still matter over a session; show enough precision to be non-zero.
        parts.append(f"💵 ${cost:.4f}" if cost < 0.01 else f"💵 ${cost:.2f}")
    if cache_hit > 0:
        parts.append(f"⚡ {cache_hit:.0%} cached")

    return "— " + " · ".join(parts) if parts else ""
