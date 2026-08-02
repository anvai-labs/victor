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

"""``victor ui`` — launch the Chainlit web chat surface.

Shells out to ``chainlit run`` on ``victor/ui/chat_app/app.py`` (Chainlit owns its own
ASGI/uvicorn server). Chainlit is an optional dependency (``victor-ai[chat-ui]``); we fail
with an actionable message when it is absent.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

ui_app = typer.Typer(
    name="ui",
    help="Launch the Victor web chat UI (Chainlit).",
)
console = Console()
logger = logging.getLogger(__name__)

_APP_PATH = Path(__file__).resolve().parents[1] / "chat_app" / "app.py"


@ui_app.callback(invoke_without_command=True)
def ui(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the UI server to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    headless: bool = typer.Option(False, "--headless", help="Run without opening a browser window"),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Auto-reload the app on file changes (development)"
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Agent profile from ~/.victor/profiles.yaml (e.g. 'zai-coding'). Defaults to 'default'.",
    ),
) -> None:
    """Start the Victor web chat UI.

    Examples:
        victor ui                          # open the chat UI in your browser
        victor ui -p 9000                  # custom port
        victor ui --headless               # serve without launching a browser
        victor ui --profile zai-coding     # use a specific agent profile
    """
    if ctx.invoked_subcommand is None:
        _launch(host=host, port=port, headless=headless, watch=watch, profile=profile)


# The chat app (run in a separate `chainlit run` process) reads the profile from this env
# var, since chainlit does not forward CLI args to the app.
_PROFILE_ENV = "VICTOR_UI_PROFILE"


def _launch(
    *, host: str, port: int, headless: bool, watch: bool, profile: Optional[str] = None
) -> None:
    if not _chainlit_available():
        console.print(
            "[red]Chainlit is not installed.[/red] The web chat UI is an optional extra.\n"
            # Escape the brackets so rich prints the extra literally, not as markup.
            r"Install it with:  [bold]pip install 'victor-ai\[chat-ui]'[/bold]"
        )
        raise typer.Exit(code=1)

    cmd = [
        sys.executable,
        "-m",
        "chainlit",
        "run",
        str(_APP_PATH),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if headless:
        cmd.append("--headless")
    if watch:
        cmd.append("--watch")

    env = os.environ.copy()
    if profile:
        env[_PROFILE_ENV] = profile

    profile_note = f" (profile: {profile})" if profile else ""
    console.print(f"[green]Starting Victor chat UI[/green]{profile_note} on http://{host}:{port}")
    raise typer.Exit(code=subprocess.call(cmd, env=env))


def _chainlit_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("chainlit") is not None
