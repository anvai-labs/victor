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

"""`victor ui` launcher: profile passthrough via the subprocess env.

Runs without the optional ``chat-ui`` extra — the launcher only probes for chainlit
(``importlib.util.find_spec``) and never imports it, so these tests mock that probe.
"""

from __future__ import annotations

import typer

from victor.ui.commands import ui as uimod


def _capture_launch(monkeypatch, **launch_kwargs):
    captured: dict = {}

    monkeypatch.setattr(uimod, "_chainlit_available", lambda: True)

    def fake_call(cmd, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return 0

    monkeypatch.setattr(uimod.subprocess, "call", fake_call)

    try:
        uimod._launch(host="127.0.0.1", port=8000, headless=True, watch=False, **launch_kwargs)
    except typer.Exit as exc:
        captured["exit_code"] = exc.exit_code
    return captured


def test_profile_is_passed_through_env(monkeypatch):
    captured = _capture_launch(monkeypatch, profile="zai-coding")
    assert captured["env"][uimod._PROFILE_ENV] == "zai-coding"
    assert captured["exit_code"] == 0
    # The app path is run via `chainlit run`.
    assert "chainlit" in captured["cmd"] and "run" in captured["cmd"]


def test_no_profile_env_when_unset(monkeypatch):
    captured = _capture_launch(monkeypatch, profile=None)
    assert uimod._PROFILE_ENV not in captured["env"]


def test_missing_chainlit_exits_with_hint(monkeypatch):
    monkeypatch.setattr(uimod, "_chainlit_available", lambda: False)
    try:
        uimod._launch(
            host="127.0.0.1",
            port=8000,
            headless=True,
            watch=False,
            profile="zai-coding",
        )
        raise AssertionError("expected typer.Exit")
    except typer.Exit as exc:
        assert exc.exit_code == 1
