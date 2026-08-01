# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Textual themes for the interactive TUI (ADR-020).

Three registered themes — dark (Dracula-family, matching the REPL palette), a
clean light theme, and a maximal-contrast accessibility theme. ``styles.tcss``
references theme variables (``$background``, ``$surface``, ``$primary``, …) rather
than hard-coded hex, so switching the active theme restyles the whole app. Users
pick one at launch (``victor tui --theme light``) or cycle at runtime (F6).

Pure data + helpers; the only Textual dependency is the ``Theme`` value type.
"""

from __future__ import annotations

from typing import Any, List

from textual.theme import Theme

#: Dark (default) — Dracula family, matches victor/ui/theme.py's REPL palette.
VICTOR_DARK = Theme(
    name="victor-dark",
    primary="#bd93f9",
    secondary="#8be9fd",
    accent="#ffb86c",
    foreground="#f8f8f2",
    background="#282a36",
    surface="#21222c",
    panel="#21222c",
    success="#50fa7b",
    warning="#f1fa8c",
    error="#ff5555",
    dark=True,
)

#: Light — a clean, professional light palette (GitHub-light family).
VICTOR_LIGHT = Theme(
    name="victor-light",
    primary="#6f42c1",
    secondary="#0969da",
    accent="#bc4c00",
    foreground="#1f2328",
    background="#ffffff",
    surface="#f6f8fa",
    panel="#eaeef2",
    success="#1a7f37",
    warning="#9a6700",
    error="#cf222e",
    dark=False,
)

#: High contrast — pure bright colours on black for accessibility.
VICTOR_HIGH_CONTRAST = Theme(
    name="victor-high-contrast",
    primary="#ffffff",
    secondary="#00ffff",
    accent="#ffff00",
    foreground="#ffffff",
    background="#000000",
    surface="#000000",
    panel="#101010",
    success="#00ff00",
    warning="#ffff00",
    error="#ff0000",
    dark=True,
)

#: All TUI themes, in cycle order (F6 advances through these).
THEMES: List[Theme] = [VICTOR_DARK, VICTOR_LIGHT, VICTOR_HIGH_CONTRAST]

#: Canonical default theme name.
DEFAULT_THEME = VICTOR_DARK.name

#: Friendly aliases → registered theme names (accepted by ``--theme``).
_ALIASES = {
    "dark": VICTOR_DARK.name,
    "light": VICTOR_LIGHT.name,
    "high-contrast": VICTOR_HIGH_CONTRAST.name,
    "high_contrast": VICTOR_HIGH_CONTRAST.name,
    "hc": VICTOR_HIGH_CONTRAST.name,
}

_NAMES = {theme.name for theme in THEMES}


def resolve_theme(name: str) -> str:
    """Map a user-supplied name/alias to a registered theme; fall back to default."""
    key = (name or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    if name in _NAMES:
        return name
    return DEFAULT_THEME


def next_theme(current: str) -> str:
    """Return the next theme name after ``current`` in cycle order (wraps)."""
    order = [theme.name for theme in THEMES]
    try:
        index = order.index(current)
    except ValueError:
        return DEFAULT_THEME
    return order[(index + 1) % len(order)]


def register_all(app: Any) -> None:
    """Register every TUI theme on a Textual app."""
    for theme in THEMES:
        app.register_theme(theme)
