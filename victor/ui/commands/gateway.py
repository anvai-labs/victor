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

"""`victor gateway` — the FEP-0020 usage-attribution reverse proxy.

Share one upstream provider key across a team with per-user virtual keys,
token budgets, and per-subject metering. See
``victor/observability/gateway_proxy.py`` for the serving mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()

gateway_app = typer.Typer(
    name="gateway",
    help="Usage-attribution reverse proxy (FEP-0020): one shared upstream key → "
    "per-user virtual keys with budgets + metering.",
)

_CONFIG_HELP = """Path to the gateway config JSON, e.g.:

\b
{
  "host": "0.0.0.0",
  "port": 8600,
  "sink_path": "usage-events.jsonl",
  "virtual_keys": [
    {
      "key_id": "alice-openai",
      "token": "vk-alice-secret",
      "subject_id": "alice",
      "group_id": "platform-team",
      "provider": "openai",
      "upstream_base_url": "https://api.openai.com/v1",
      "upstream_api_key": "sk-the-shared-real-key",
      "budget_tokens": 2000000
    }
  ]
}
"""


@gateway_app.command("serve")
def serve(
    config: Path = typer.Option(..., "--config", "-c", help=_CONFIG_HELP),
    host: Optional[str] = typer.Option(None, "--host", help="Override the config bind host."),
    port: Optional[int] = typer.Option(None, "--port", help="Override the config bind port."),
) -> None:
    """Run the usage-attribution reverse proxy."""
    try:
        import uvicorn

        from victor.observability.gateway_proxy import GatewayConfig, build_gateway_app
        from victor.observability.sandhi_meter import sandhi_available
    except ImportError as exc:
        console.print(f"[red]Error:[/] missing dependency for the usage gateway: {exc}")
        console.print("\nInstall with: [bold]pip install 'victor-ai[gateway]'[/]")
        raise typer.Exit(1)

    if not sandhi_available():
        console.print("[red]Error:[/] sandhi-gateway is not installed.")
        console.print("\nInstall with: [bold]pip install 'victor-ai[gateway]'[/]")
        raise typer.Exit(1)

    if not config.exists():
        console.print(f"[red]Error:[/] config file not found: {config}")
        raise typer.Exit(1)

    try:
        cfg = GatewayConfig(**json.loads(config.read_text()))
    except (ValueError, TypeError) as exc:
        console.print(f"[red]Error:[/] invalid gateway config: {exc}")
        raise typer.Exit(1)

    if host is not None:
        cfg.host = host
    if port is not None:
        cfg.port = port

    if not cfg.virtual_keys:
        console.print("[yellow]Warning:[/] no virtual keys configured — every request will 401.")

    app = build_gateway_app(cfg)
    console.print(
        f"[green]Victor usage gateway[/] on http://{cfg.host}:{cfg.port} "
        f"— {len(cfg.virtual_keys)} virtual key(s); "
        f"sink={'memory' if cfg.sink_path is None else cfg.sink_path}"
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
