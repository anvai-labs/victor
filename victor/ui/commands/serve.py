import logging
import os

import typer
from typing import Dict, Optional
from rich.console import Console
from rich.panel import Panel

from victor.core.async_utils import run_sync
from victor.ui.commands.utils import setup_logging
from victor.workflows.hitl_api import get_default_hitl_db_path

serve_app = typer.Typer(
    name="serve",
    help="Start Victor servers (API, HITL, etc.).",
)
console = Console()
logger = logging.getLogger(__name__)


@serve_app.callback(invoke_without_command=True)
def serve(
    ctx: typer.Context,
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind the server to",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        "-p",
        help="Port to listen on",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (defaults to INFO for serve or VICTOR_LOG_LEVEL env var)",
    ),
    profile: str = typer.Option(
        "default",
        "--profile",
        help="Profile to use for the server",
    ),
    mode: str = typer.Option(
        "build",
        "--mode",
        help="Initial agent mode for API sessions",
    ),
    enable_hitl: bool = typer.Option(
        False,
        "--enable-hitl",
        help="Enable HITL (Human-in-the-Loop) endpoints for workflow approvals",
    ),
    hitl_auth_token: Optional[str] = typer.Option(
        None,
        "--hitl-auth-token",
        help="Bearer token for HITL endpoints (or set HITL_AUTH_TOKEN env var)",
        envvar="HITL_AUTH_TOKEN",
    ),
):
    """Start the Victor API server for IDE integrations.

    The server provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
    and external tool access.

    Examples:
        victor serve                 # Start FastAPI backend
        victor serve -p 8000         # FastAPI on custom port
        victor serve --enable-hitl   # Enable HITL endpoints for approvals
    """
    if ctx.invoked_subcommand is None:
        _serve(host, port, log_level, profile, mode, enable_hitl, hitl_auth_token)


def _serve(
    host: str,
    port: int,
    log_level: Optional[str],
    profile: str,
    mode: str,
    enable_hitl: bool = False,
    hitl_auth_token: Optional[str] = None,
):
    # Normalize log level if provided
    if log_level is not None:
        log_level = log_level.upper()
        if log_level == "WARN":
            log_level = "WARNING"

    from victor.framework.runtime_discovery import CANONICAL_AGENT_MODES

    if mode not in CANONICAL_AGENT_MODES:
        console.print(
            f"[red]Error:[/] Unsupported mode '{mode}'. "
            f"Choose one of: {', '.join(CANONICAL_AGENT_MODES)}"
        )
        raise typer.Exit(1)

    # Use centralized logging config (serve has INFO default in logging_config.yaml)
    setup_logging(command="serve", cli_log_level=log_level)

    backend_info = f"\n[bold]Docs:[/] [cyan]http://{host}:{port}/docs[/]"

    # HITL-specific information
    hitl_info = ""
    if enable_hitl:
        hitl_info = f"\n[bold]HITL UI:[/] [cyan]http://{host}:{port}/hitl/ui[/]"

    console.print(
        Panel(
            f"[bold blue]Victor API Server[/]\n\n"
            f"[bold]Host:[/] [cyan]{host}[/]\n"
            f"[bold]Port:[/] [cyan]{port}[/]\n"
            f"[bold]Backend:[/] [cyan]fastapi[/]\n"
            f"[bold]Profile:[/] [cyan]{profile}[/]\n"
            f"[bold]Mode:[/] [cyan]{mode}[/]"
            f"{backend_info}"
            f"{hitl_info}\n\n"
            f"[dim]Press Ctrl+C to stop[/]",
            title="Victor Server",
            border_style="blue",
        )
    )

    run_sync(_run_fastapi_server(host, port, profile, mode, enable_hitl, hitl_auth_token))


def _api_keys_from_env() -> Dict[str, str]:
    """Parse VICTOR_API_KEYS into the server's api_keys mapping.

    Comma-separated entries of ``key`` or ``key:client_id``; entries without
    an explicit client_id get ``client{i}``. Without this wiring the API-key
    auth gate was unreachable from `victor serve` (adversarial-review
    finding): the command never passed api_keys, so binding 0.0.0.0 served
    an unauthenticated API regardless of the server's support for keys.
    """
    raw = os.environ.get("VICTOR_API_KEYS", "").strip()
    if not raw:
        return {}
    keys: Dict[str, str] = {}
    for i, entry in enumerate(raw.split(",")):
        entry = entry.strip()
        if not entry:
            continue
        key, sep, client_id = entry.partition(":")
        keys[key.strip()] = client_id.strip() if sep and client_id.strip() else f"client{i}"
    return keys


async def _run_fastapi_server(
    host: str,
    port: int,
    profile: str,
    mode: str,
    enable_hitl: bool = False,
    hitl_auth_token: Optional[str] = None,
) -> None:
    """Run the FastAPI server (modern backend with OpenAPI docs)."""
    try:
        from pathlib import Path

        import uvicorn

        from victor.integrations.api.fastapi_server import VictorFastAPIServer
        from victor.framework.session_config import SessionConfig

        api_keys = _api_keys_from_env()
        server = VictorFastAPIServer(
            host=host,
            port=port,
            workspace_root=str(Path.cwd()),
            enable_hitl=enable_hitl,
            hitl_auth_token=hitl_auth_token,
            api_keys=api_keys or None,
            session_config=SessionConfig.from_cli_flags(agent_profile=profile, mode=mode),
        )
        if api_keys:
            console.print(
                f"[green]API-key auth enabled for {len(api_keys)} key(s) "
                f"(VICTOR_API_KEYS); docs endpoints disabled.[/]"
            )

        config = uvicorn.Config(
            server.app,
            host=host,
            port=port,
            log_level="info",
        )
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()

    except ImportError as e:
        console.print(f"[red]Error:[/] Missing dependency for FastAPI server: {e}")
        console.print("\nInstall with: [bold]pip install fastapi uvicorn[/]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


# ============================================================================
# HITL Server Subcommand
# ============================================================================


@serve_app.command(name="hitl")
def serve_hitl(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind the server to (0.0.0.0 for all interfaces)",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port to listen on",
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        "-t",
        help="Bearer token for authentication (or set HITL_AUTH_TOKEN env var)",
        envvar="HITL_AUTH_TOKEN",
    ),
    require_auth: bool = typer.Option(
        False,
        "--require-auth",
        "-a",
        help="Require authentication for API access",
    ),
    persistent: bool = typer.Option(
        False,
        "--persistent",
        help="Use SQLite for persistent storage (survives restarts)",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Path to SQLite database (default: ~/.victor/hitl.db)",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (defaults to INFO)",
    ),
):
    """Start the HITL (Human-in-the-Loop) approval server.

    This server provides a web UI and REST API for remote workflow approvals.
    Use this when deploying workflows to Docker, Kubernetes, Lambda, etc.
    where CLI-based approval isn't possible.

    Examples:
        victor serve hitl                        # Start on 0.0.0.0:8080 (in-memory)
        victor serve hitl --persistent           # Use SQLite storage
        victor serve hitl -p 9000                # Custom port
        victor serve hitl --require-auth -t secret  # With auth

    Access:
        Web UI: http://localhost:8080/hitl/ui
        API:    http://localhost:8080/hitl/requests
        Docs:   http://localhost:8080/docs
    """
    # Normalize log level if provided
    if log_level is not None:
        log_level = log_level.upper()
        if log_level == "WARN":
            log_level = "WARNING"

    setup_logging(command="serve", cli_log_level=log_level)

    auth_info = ""
    if require_auth:
        if auth_token:
            auth_info = "\n[bold]Auth:[/] [green]Enabled[/] (token configured)"
        else:
            auth_info = "\n[bold]Auth:[/] [yellow]Enabled but no token set![/]"
    else:
        auth_info = "\n[bold]Auth:[/] [dim]Disabled[/]"

    storage_info = ""
    if persistent:
        effective_db = db_path or str(get_default_hitl_db_path())
        storage_info = f"\n[bold]Storage:[/] [green]SQLite[/] ({effective_db})"
    else:
        storage_info = "\n[bold]Storage:[/] [dim]In-memory[/]"

    console.print(
        Panel(
            f"[bold blue]Victor HITL Server[/]\n\n"
            f"[bold]Host:[/] [cyan]{host}[/]\n"
            f"[bold]Port:[/] [cyan]{port}[/]"
            f"{auth_info}"
            f"{storage_info}\n\n"
            f"[bold]Web UI:[/] [cyan]http://localhost:{port}/hitl/ui[/]\n"
            f"[bold]API:[/] [cyan]http://localhost:{port}/hitl/requests[/]\n"
            f"[bold]Docs:[/] [cyan]http://localhost:{port}/docs[/]\n\n"
            f"[dim]Press Ctrl+C to stop[/]",
            title="HITL Server",
            border_style="blue",
        )
    )

    run_sync(_run_hitl_server(host, port, require_auth, auth_token, persistent, db_path))


async def _run_hitl_server(
    host: str,
    port: int,
    require_auth: bool,
    auth_token: Optional[str],
    persistent: bool = False,
    db_path: Optional[str] = None,
) -> None:
    """Run the HITL API server."""
    try:
        import uvicorn
        from victor.config.settings import load_settings
        from victor.core.bootstrap import ensure_bootstrapped

        from victor.workflows.hitl_api import (
            HITLStore,
            SQLiteHITLStore,
            create_hitl_app,
        )

        settings = load_settings()
        ensure_bootstrapped(settings)

        # Create appropriate store
        if persistent:
            store = SQLiteHITLStore(db_path=db_path)
            logger.info(f"Using SQLite HITL store: {store.db_path}")
        else:
            store = HITLStore()
            logger.info("Using in-memory HITL store")

        app = create_hitl_app(
            store=store,
            require_auth=require_auth,
            auth_token=auth_token,
        )

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    except ImportError as e:
        console.print(f"[red]Error:[/] Missing dependency for HITL server: {e}")
        console.print("\nInstall with: [bold]pip install fastapi uvicorn[/]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]HITL server stopped[/]")
    except Exception as e:
        logger.exception("HITL server error")
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
