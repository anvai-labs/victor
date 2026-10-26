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

    def fake_call(cmd, env=None, cwd=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
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


def test_repo_root_chained_when_branded_welcome_exists(monkeypatch):
    """The shipped chainlit.md pins ONLY Chainlit's APP_ROOT — never the cwd.

    cwd is the embedded agent's project context (.victor/project.db, relative
    tool paths); repointing it would silently run the agent against this repo.
    """
    captured = _capture_launch(monkeypatch, profile=None)
    repo_root = uimod._APP_PATH.parents[3]
    assert captured["cwd"] is None, "launcher must not change the process cwd"
    if (repo_root / "chainlit.md").is_file():
        assert captured["env"].get("CHAINLIT_APP_ROOT") == str(repo_root)
    else:  # installed package: nothing to pin, launch unchanged
        assert "CHAINLIT_APP_ROOT" not in captured["env"]


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


def test_packaged_presentation_routes_are_narrow_and_keep_customizations(tmp_path):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from victor.ui.chat_app.presentation import configure_presentation

    app = FastAPI()

    @app.get("/{path:path}")
    def spa(path: str):
        return {"spa": path}

    ui = SimpleNamespace(
        name="Assistant",
        description="",
        custom_css=None,
        logo_file_url=None,
        default_avatar_file_url=None,
        layout="default",
    )
    configure_presentation(app, ui, root_path="/chat")
    configure_presentation(app, ui, root_path="/chat")
    assert ui.name == "Victor" and ui.layout == "wide"
    assert ui.custom_css == "/chat/victor-assets/workspace.css"
    assert sum(getattr(route, "name", "") == "victor-assets" for route in app.routes) == 1
    with TestClient(app) as client:
        css = client.get("/victor-assets/workspace.css")
        assert css.status_code == 200 and "text/css" in css.headers["content-type"]
        assert "--primary:" in css.text
        assert client.get("/victor-assets/victor.svg").status_code == 200
        assert client.get("/victor-assets/missing.css").status_code == 404
        assert client.get("/victor-assets/%2e%2e/app.py").status_code == 404
        assert client.get("/elsewhere").json() == {"spa": "elsewhere"}
    ui.name, ui.custom_css, ui.layout = "Company", "/company.css", "wide"
    ui.logo_file_url = ui.default_avatar_file_url = "/company.svg"
    configure_presentation(app, ui)
    assert (ui.name, ui.custom_css, ui.logo_file_url, ui.default_avatar_file_url) == (
        "Company",
        "/company.css",
        "/company.svg",
        "/company.svg",
    )
