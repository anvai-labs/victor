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

"""UI-free symbol registry for status, severity, and tool presentation.

This module is intentionally independent of ``victor.ui`` and configuration.
Callers that need settings-aware behavior should decide whether emojis are
enabled and pass that decision into ``get_symbol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IconSet:
    """Defines emoji and text alternatives for an icon."""

    emoji: str
    text: str
    rich_color: Optional[str] = None

    def get(self, use_emoji: bool = True, with_color: bool = True) -> str:
        """Return the selected symbol, optionally wrapped in Rich color markup."""
        icon = self.emoji if use_emoji else self.text
        if with_color and self.rich_color:
            return f"[{self.rich_color}]{icon}[/]"
        return icon


ICONS = {
    # Status indicators
    "success": IconSet("✓", "+", "green"),
    "error": IconSet("✗", "x", "red"),
    "warning": IconSet("⚠", "!", "yellow"),
    "info": IconSet("ℹ", "i", "blue"),
    # Tool execution
    "running": IconSet("🔧", "*", "cyan"),
    "pending": IconSet("⏳", "...", "dim"),
    "done": IconSet("✔", "OK", "green"),
    # Navigation/UI
    "arrow_right": IconSet("→", "->", None),
    "arrow_left": IconSet("←", "<-", None),
    "bullet": IconSet("•", "-", None),
    # Files/folders
    "file": IconSet("📄", "F", None),
    "folder": IconSet("📁", "D", None),
    # Misc
    "sparkle": IconSet("✨", "*", "yellow"),
    "search": IconSet("🔍", "?", None),
    "chart": IconSet("📊", "#", None),
    "target": IconSet("🎯", "@", None),
    "rocket": IconSet("🚀", "^", None),
    "bolt": IconSet("⚡", "!", "yellow"),
    "bulb": IconSet("💡", "*", "yellow"),
    "note": IconSet("📝", "#", None),
    "refresh": IconSet("🔄", "~", None),
    # Safety/risk level icons
    "risk_high": IconSet("🔴", "!!", "red"),
    "risk_critical": IconSet("⛔", "!!!", "red"),
    "unknown": IconSet("❓", "?", None),
    # Step status icons
    "skipped": IconSet("⏭️", ">>", "dim"),
    "blocked": IconSet("🔒", "[X]", "yellow"),
    # Additional UI icons
    "thinking": IconSet("💭", "...", "dim"),
    "gear": IconSet("⚙️", "[*]", "cyan"),
    "clipboard": IconSet("📋", "[=]", None),
    "stop": IconSet("⛔", "[!]", "red"),
    "clock": IconSet("⏰", "[T]", "yellow"),
    "stop_sign": IconSet("🛑", "[X]", "red"),
    # Severity/complexity indicators
    "level_low": IconSet("🟢", "[L]", "green"),
    "level_medium": IconSet("🟡", "[M]", "yellow"),
    "level_high": IconSet("🟠", "[H]", "yellow"),
    "level_critical": IconSet("🔴", "[!]", "red"),
    "level_info": IconSet("🔵", "[I]", "blue"),
    "level_unknown": IconSet("⚪", "[?]", None),
    "person": IconSet("👤", "[P]", None),
    # Platform/technology icons
    "terraform": IconSet("🏗️", "[TF]", None),
    "docker": IconSet("🐳", "[DK]", None),
    "kubernetes": IconSet("☸️", "[K8]", None),
    # Miscellaneous
    "hint": IconSet("💡", "*", "yellow"),
    "alert": IconSet("🚨", "(!)", "red"),
    "trend_up": IconSet("📈", "^", "green"),
    "trend_down": IconSet("📉", "v", "red"),
}


def get_symbol(
    name: str,
    *,
    use_emoji: bool = True,
    with_color: bool = True,
) -> str:
    """Return a symbol by name without importing settings or UI modules."""
    if name not in ICONS:
        raise KeyError(f"Unknown icon: {name}. Available: {list(ICONS.keys())}")
    return ICONS[name].get(use_emoji=use_emoji, with_color=with_color)


def get_text_symbol(name: str) -> str:
    """Return the plain text fallback for a symbol."""
    return get_symbol(name, use_emoji=False, with_color=False)


__all__ = ["ICONS", "IconSet", "get_symbol", "get_text_symbol"]
