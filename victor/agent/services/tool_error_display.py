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

"""Console error display for tool execution results.

Extracted from the ``tool_service`` hotspot (size-ratcheted module). Owns the
"show a tool error on the console, once" concern:

- "not found" errors are deduplicated per tool for the session
- while an interactive live renderer owns console output, nothing is printed
  here at all — the renderer surfaces the failure as its own ✗ error card,
  so a service-level print would be a duplicate
"""

from __future__ import annotations

from typing import Any

from victor.runtime.live_console import live_display_active

__all__ = ["print_tool_error_once"]


def print_tool_error_once(
    ctx: Any,
    tool_name: str,
    error_display: Any,
    *,
    skipped: bool,
    elapsed_ms: float,
) -> None:
    """Print one deduplicated console error line for a failed/skipped tool.

    Args:
        ctx: ToolResultContext carrying console/presentation/shown_tool_errors.
        tool_name: Name of the tool that failed or was skipped.
        error_display: Human-readable error text.
        skipped: Whether the call was skipped rather than executed-and-failed.
        elapsed_ms: Wall-clock milliseconds attributed to the call.
    """
    not_found = "not found" in str(error_display).lower()
    shown_key = f"notfound:{tool_name}" if not_found else None
    if shown_key and shown_key in ctx.shown_tool_errors:
        return
    if shown_key and len(ctx.shown_tool_errors) < 500:
        ctx.shown_tool_errors.add(shown_key)
    if ctx.console and ctx.presentation and not live_display_active():
        prefix = "Tool call skipped" if skipped else "Tool execution failed"
        ctx.console.print(
            f"[red]{ctx.presentation.icon('error', with_color=False)} "
            f"{prefix}: {error_display}[/] "
            f"[dim]({elapsed_ms:.0f}ms)[/dim]"
        )
