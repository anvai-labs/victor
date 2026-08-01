"""Tests for the TUI theme registry and app theme selection/cycling."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict

from victor.ui.tui.app import VictorTUIApp
from victor.ui.tui.themes import (
    DEFAULT_THEME,
    THEMES,
    next_theme,
    resolve_theme,
)

# ── pure registry ─────────────────────────────────────────────────


def test_three_themes_registered() -> None:
    names = {theme.name for theme in THEMES}
    assert names == {"victor-dark", "victor-light", "victor-high-contrast"}
    assert DEFAULT_THEME == "victor-dark"


def test_resolve_theme_aliases() -> None:
    assert resolve_theme("dark") == "victor-dark"
    assert resolve_theme("light") == "victor-light"
    assert resolve_theme("high-contrast") == "victor-high-contrast"
    assert resolve_theme("high_contrast") == "victor-high-contrast"
    assert resolve_theme("HC") == "victor-high-contrast"


def test_resolve_theme_passthrough_and_fallback() -> None:
    assert resolve_theme("victor-light") == "victor-light"  # exact name
    assert resolve_theme("nonsense") == DEFAULT_THEME
    assert resolve_theme("") == DEFAULT_THEME


def test_next_theme_cycles_and_wraps() -> None:
    assert next_theme("victor-dark") == "victor-light"
    assert next_theme("victor-light") == "victor-high-contrast"
    assert next_theme("victor-high-contrast") == "victor-dark"
    assert next_theme("unknown") == DEFAULT_THEME


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
