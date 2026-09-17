import typer
import logging
import sys
import os
import inspect
import importlib
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from victor.config.secure_paths import secure_create_file
from victor.config.settings import get_project_paths
from victor.core.async_utils import run_sync
from victor.core.yaml_utils import safe_dump, safe_load
from victor.integrations.mcp.registry import MCPRegistry, MCPServerConfig
from victor.integrations.mcp.server import MCPServer
from victor.tools.registry import ToolRegistry
from victor.ui.commands.utils import setup_logging

mcp_app = typer.Typer(name="mcp", help="Run Victor as an MCP server, or manage MCP client config.")
console = Console()
logger = logging.getLogger(__name__)


@mcp_app.callback(invoke_without_command=True)
def mcp(
    ctx: typer.Context,
    stdio: bool = typer.Option(
        True,
        "--stdio/--no-stdio",
        help="Run in stdio mode (for MCP clients)",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (defaults to WARNING or VICTOR_LOG_LEVEL env var)",
    ),
):
    """Run Victor as an MCP server."""
    if ctx.invoked_subcommand is None:
        _mcp(stdio, log_level)


def _mcp(stdio: bool, log_level: Optional[str]):
    # Configure logging to stderr (stdout is for MCP protocol)
    # Validate and normalize log level
    if log_level is not None:
        log_level = log_level.upper()
        if log_level == "WARN":
            log_level = "WARNING"

    # Use centralized logging config (MCP has specific settings in logging_config.yaml)
    setup_logging(command="mcp", cli_log_level=log_level, stream=sys.stderr)

    if stdio:
        run_sync(_run_mcp_server())
    else:
        console.print("[red]Only stdio mode is currently supported[/]")
        raise typer.Exit(1)


async def _run_mcp_server() -> None:
    """Run MCP server with all registered tools."""
    # Create tool registry
    registry = ToolRegistry()

    # Dynamic tool discovery (same pattern as orchestrator)
    tools_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tools")
    excluded_files = {"__init__.py", "base.py", "decorators.py", "semantic_selector.py"}
    registered_tools_count = 0

    for filename in os.listdir(tools_dir):
        if filename.endswith(".py") and filename not in excluded_files:
            module_name = f"victor.tools.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for _name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                        registry.register(obj)
                        registered_tools_count += 1
            except Exception as e:
                print(
                    f"Warning: Failed to load tools from {module_name}: {e}",
                    file=sys.stderr,
                )

    server = MCPServer(
        name="Victor MCP Server",
        version="1.0.0",
        tool_registry=registry,
    )

    print(
        f"Victor MCP Server starting with {registered_tools_count} tools",
        file=sys.stderr,
    )
    await server.start_stdio_server()


def _parse_env_pairs(pairs: List[str]) -> Dict[str, str]:
    """Parse repeated --env KEY=VALUE options into a dict."""
    result: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected KEY=VALUE, got '{pair}'")
        key, _, value = pair.partition("=")
        if not key:
            raise ValueError(f"empty key in '{pair}'")
        result[key] = value
    return result


def _resolve_mcp_config_path(scope: str, config_dir: Optional[str]) -> Path:
    """Resolve which mcp.yaml to write to.

    --config-dir (mainly for tests) takes precedence over --scope.
    """
    if config_dir is not None:
        return Path(config_dir) / "mcp.yaml"
    paths = get_project_paths()
    return paths.mcp_config if scope == "project" else paths.global_mcp_config


def _load_mcp_servers_list(path: Path) -> List[Dict[str, Any]]:
    """Read the `servers:` list from an mcp.yaml, tolerating a missing file.

    This mirrors the shape MCPRegistry.from_config() actually reads
    (a list of server dicts under `servers:`), not the dict-keyed shape
    McpSettings' docstring implies elsewhere in the codebase - that field
    is not consumed by the real discovery/connect path.
    """
    if not path.exists():
        return []
    data = safe_load(path.read_text()) or {}
    servers = data.get("servers", [])
    return servers if isinstance(servers, list) else []


def _upsert_mcp_server_yaml(path: Path, config: MCPServerConfig) -> None:
    """Write (or update in place) one server entry in an mcp.yaml."""
    servers = _load_mcp_servers_list(path)
    entry = config.model_dump(exclude_defaults=True)

    replaced = False
    for i, existing in enumerate(servers):
        if existing.get("name") == config.name:
            servers[i] = entry
            replaced = True
            break
    if not replaced:
        servers.append(entry)

    content = safe_dump({"servers": servers}, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not secure_create_file(path, content or "", mode=0o600, atomic=True):
        raise RuntimeError(f"Failed to write MCP config to {path}")


async def _validate_mcp_server(
    config: MCPServerConfig, timeout: float
) -> Tuple[bool, int, Optional[str]]:
    """Spawn `config` on a throwaway registry and confirm it connects.

    A fresh MCPRegistry() (never discover_servers()) has nothing else
    registered on it, so connect()/disconnect() here can't disturb any
    other server. Known limitation: MCPRegistry.connect() assigns the
    spawned client to its entry only after the handshake completes: if
    the wait_for timeout below fires in that narrow window, the
    subprocess can leak with no handle left to disconnect() it. This is
    a pre-existing gap in registry.py's connect(), not something this
    command can fully paper over.
    """
    registry = MCPRegistry(health_check_enabled=False)
    registry.register_server(config)
    try:
        success = await asyncio.wait_for(registry.connect(config.name), timeout=timeout)
        if success:
            tool_count = len(registry.get_tools_by_server(config.name))
            return True, tool_count, None
        status = registry.get_server_status(config.name) or {}
        return False, 0, status.get("error") or "connection failed"
    except asyncio.TimeoutError:
        return False, 0, f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a validation failure
        return False, 0, str(exc)
    finally:
        try:
            await registry.disconnect(config.name)
        except Exception as cleanup_exc:  # noqa: BLE001 - best-effort cleanup
            logger.debug(f"Ignoring cleanup error disconnecting '{config.name}': {cleanup_exc}")


@mcp_app.command("add", context_settings={"ignore_unknown_options": True})
def mcp_add(
    name: str = typer.Argument(..., help="Unique name for the MCP server"),
    command: str = typer.Argument(
        ..., help="Executable to launch the stdio MCP server (e.g. 'npx', '/path/to/binary')"
    ),
    args: Optional[List[str]] = typer.Argument(None, help="Arguments passed to the command"),
    env: List[str] = typer.Option(
        [], "--env", "-e", help="Environment variable as KEY=VALUE (repeatable)"
    ),
    scope: str = typer.Option(
        "global",
        "--scope",
        help="Where to persist the config: 'project' (.victor/mcp.yaml) or 'global' (~/.victor/mcp.yaml)",
    ),
    config_dir: Optional[str] = typer.Option(
        None,
        "--config-dir",
        help="Explicit .victor directory to write mcp.yaml into (overrides --scope; mainly for testing)",
    ),
    force: bool = typer.Option(
        False, "--force", help="Persist the server config even if the connectivity check fails"
    ),
    timeout: float = typer.Option(
        15.0,
        "--timeout",
        help="Seconds to wait for the server to connect before failing validation",
    ),
) -> None:
    """Register a stdio MCP server and persist it to mcp.yaml.

    Spawns the server, verifies it responds to the MCP handshake and
    lists tools, then writes (or updates) the entry in the resolved
    mcp.yaml so it is picked up by MCPRegistry.discover_servers() on
    future `victor` runs - closing the gap with `claude mcp add`.

    Only stdio-transport servers are supported: Victor's live MCP
    registry does not implement sse/http transports today, so there is
    no --type or --url option here.
    """
    try:
        env_dict = _parse_env_pairs(env)
    except ValueError as e:
        console.print(f"[red]Invalid --env value:[/] {e}")
        raise typer.Exit(1)

    if scope not in ("project", "global"):
        console.print(f"[red]--scope must be 'project' or 'global', got '{scope}'[/]")
        raise typer.Exit(1)

    target_path = _resolve_mcp_config_path(scope, config_dir)
    full_command = [command, *(args or [])]
    server_config = MCPServerConfig(name=name, command=full_command, env=env_dict)

    ok, tool_count, error = run_sync(_validate_mcp_server(server_config, timeout))

    if ok:
        console.print(f"[green]Connected:[/] {tool_count} tool(s) discovered")
    else:
        console.print(f"[red]Connection failed:[/] {error}")
        if not force:
            console.print("[yellow]Not persisting (use --force to save anyway).[/]")
            raise typer.Exit(1)
        console.print("[yellow]--force set: persisting despite validation failure.[/]")

    _upsert_mcp_server_yaml(target_path, server_config)
    console.print(f"[green]Saved[/] '{name}' to {target_path}")
