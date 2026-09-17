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

"""Mouse-capture opt-in flags for the two interactive chat surfaces.

Both default OFF so the terminal's own drag-select/copy works; the flags hand
mouse events to the in-app widgets instead. These tests pin the parsing so the
defaults can't silently drift.
"""

import importlib

import pytest

chat = importlib.import_module("victor.ui.commands.chat")


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "ON", "True", " 1 "])
def test_tui_mouse_optin_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_TUI_MOUSE_SUPPORT", value)
    assert chat._tui_mouse_support_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "garbage", "  "])
def test_tui_mouse_default_off_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_TUI_MOUSE_SUPPORT", value)
    assert chat._tui_mouse_support_enabled() is False


def test_tui_mouse_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VICTOR_TUI_MOUSE_SUPPORT", raising=False)
    assert chat._tui_mouse_support_enabled() is False


def test_cli_mouse_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VICTOR_CHAT_MOUSE_SUPPORT", raising=False)
    assert chat._cli_mouse_support_enabled() is False


def test_cli_mouse_optin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_CHAT_MOUSE_SUPPORT", "1")
    assert chat._cli_mouse_support_enabled() is True


def test_run_tui_app_passes_mouse_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launch path must forward the opt-in to Textual's ``run_async``."""
    import asyncio

    captured: dict = {}

    class FakeApp:
        _bootstrap_result = None  # _run_tui_app returns this after run_async

        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        async def run_async(self, *, mouse: bool = True) -> None:
            captured["mouse"] = mouse

    monkeypatch.setattr(chat, "_tui_mouse_support_enabled", lambda: False)
    monkeypatch.setattr("victor.ui.tui.app.VictorTUIApp", FakeApp)
    asyncio.run(chat._run_tui_app(client=object(), agent=object(), settings=None))
    assert captured["mouse"] is False

    monkeypatch.setattr(chat, "_tui_mouse_support_enabled", lambda: True)
    asyncio.run(chat._run_tui_app(client=object(), agent=object(), settings=None))
    assert captured["mouse"] is True
