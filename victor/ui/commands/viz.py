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

"""``victor viz`` — the interactive code-graph explorer (ADR-033 phase 1)."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import typer

from victor.core.async_utils import run_sync

viz_app = typer.Typer(name="viz", help="Interactive code-graph visualization (ADR-033).")

try:
    from rich.console import Console

    console = Console()
except ImportError:  # pragma: no cover - rich is a hard dep; defensive only
    console = None


def _print(message: str) -> None:
    if console is not None:
        console.print(message)
    else:
        print(message)


@viz_app.command("serve")
def viz_serve(
    path: str = typer.Option(".", "--path", "-p", help="Workspace root to explore"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8790, "--port", help="Bind port"),
    open_browser: bool = typer.Option(False, "--open", help="Open the browser"),
):
    """Serve the interactive package->module->class explorer."""
    import uvicorn

    from victor.integrations.api.viz_server import create_viz_app

    root = Path(path).resolve()
    if not root.is_dir():
        _print(f"[red]Error:[/red] Path '{path}' is not a directory")
        raise typer.Exit(1)

    app = create_viz_app(root)
    url = f"http://{host}:{port}/"
    _print(f"[green]Victor Code Explorer:[/green] {url} (workspace: {root})")
    _print("[dim]Ctrl+C to stop[/dim]")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")
