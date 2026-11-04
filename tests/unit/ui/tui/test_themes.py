"""Tests for the TUI theme registry and app theme selection/cycling."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict

from textual.theme import Theme

from victor.ui.tui.app import VictorTUIApp
from victor.ui.tui.themes import (
    DEFAULT_THEME,
    THEMES,
    VICTOR_TERMINAL,
    _THEME_SUPPORTS_ANSI,
    next_theme,
    resolve_theme,
)

# ── pure registry ─────────────────────────────────────────────────


def test_all_themes_registered() -> None:
    names = {theme.name for theme in THEMES}
    assert names == {
        "victor-dark",
        "victor-light",
        "victor-high-contrast",
        "victor-terminal",
    }
    assert DEFAULT_THEME == "victor-dark"


def test_resolve_theme_aliases() -> None:
    assert resolve_theme("dark") == "victor-dark"
    assert resolve_theme("light") == "victor-light"
    assert resolve_theme("high-contrast") == "victor-high-contrast"
    assert resolve_theme("high_contrast") == "victor-high-contrast"
    assert resolve_theme("HC") == "victor-high-contrast"
    assert resolve_theme("terminal") == "victor-terminal"
    assert resolve_theme("native") == "victor-terminal"


def test_resolve_theme_passthrough_and_fallback() -> None:
    assert resolve_theme("victor-light") == "victor-light"  # exact name
    assert resolve_theme("nonsense") == DEFAULT_THEME
    assert resolve_theme("") == DEFAULT_THEME


def test_next_theme_cycles_and_wraps() -> None:
    assert next_theme("victor-dark") == "victor-light"
    assert next_theme("victor-light") == "victor-high-contrast"
    assert next_theme("victor-high-contrast") == "victor-terminal"
    assert next_theme("victor-terminal") == "victor-dark"
    assert next_theme("unknown") == DEFAULT_THEME


def test_victor_terminal_surfaces_consistent_across_textual_versions() -> None:
    """The terminal theme's NAME and dark-mode flag are stable regardless of
    which construction path ran (Textual 8.x ansi vs 7.x fallback hex). The
    TUI's theme-cycle (F6), ``--theme terminal`` flag, and ``resolve_theme``
    all rely on the name being ``victor-terminal`` — any drift here breaks
    user-visible behavior.
    """
    assert VICTOR_TERMINAL.name == "victor-terminal"
    assert VICTOR_TERMINAL.dark is True


def test_victor_terminal_fallback_palette_is_valid() -> None:
    """The Textual 7.x fallback palette must construct without error. This is
    the regression pin for the aiserver1 crash: ``Theme.__init__() got an
    unexpected keyword argument 'ansi'`` — the module import failed because
    VICTOR_TERMINAL unconditionally used ansi=True on Textual 7.5. The
    fallback palette reuses VICTOR_DARK's explicit hex colors, which the
    theme's contract says are a degraded-but-functional substitute.
    """
    # The fallback construction is the same one used in the else-branch of
    # themes.py when _THEME_SUPPORTS_ANSI is False. Verifying it builds
    # without error pins the contract even on Textual 8+ (where the
    # ansi-path is taken at import time).
    fallback = Theme(
        name="victor-terminal-fallback-probe",
        primary="#bd93f9",
        secondary="#8be9fd",
        accent="#ffb86c",
        foreground="#f8f8f2",
        background="#282a36",
        surface="#44475a",
        panel="#44475a",
        success="#50fa7b",
        warning="#f1fa8c",
        error="#ff5555",
        dark=True,
    )
    assert fallback.name == "victor-terminal-fallback-probe"
    assert fallback.dark is True
    # The detection flag itself reflects the running Textual's capability.
    # Either value is valid — we just pin that it's a bool.
    assert isinstance(_THEME_SUPPORTS_ANSI, bool)


# ── app integration ───────────────────────────────────────────────


class _FakeClient:
    model = "m"
    provider_name = "p"

    def set_approval_handler(self, handler: Any) -> None:
        pass

    def get_last_turn_cost(self) -> Dict[str, Any]:
        return {}

    async def stream(self, message: str) -> AsyncIterator[Any]:
        return
        yield  # pragma: no cover - makes this an async generator


class _FakeAgent:
    active_session_id = "sess"


def _app(theme: str = "dark") -> VictorTUIApp:
    return VictorTUIApp(client=_FakeClient(), agent=_FakeAgent(), settings=None, theme=theme)


async def test_app_applies_requested_theme() -> None:
    app = _app(theme="light")
    async with app.run_test():
        assert app.theme == "victor-light"


async def test_app_falls_back_to_default_theme() -> None:
    app = _app(theme="bogus")
    async with app.run_test():
        assert app.theme == "victor-dark"


async def test_cycle_theme_action_advances() -> None:
    app = _app(theme="dark")
    async with app.run_test() as pilot:
        assert app.theme == "victor-dark"
        app.action_cycle_theme()
        await pilot.pause()
        assert app.theme == "victor-light"
        app.action_cycle_theme()
        await pilot.pause()
        assert app.theme == "victor-high-contrast"
