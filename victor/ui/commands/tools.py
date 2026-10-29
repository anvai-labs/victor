import typer
from typing import List, Tuple
from rich.console import Console
from rich.table import Table

from victor.core.async_utils import run_sync
from victor.config.settings import load_settings

tools_app = typer.Typer(name="tool", help="Manage Victor's integrated tools.")
console = Console()


def _discover_tools_lightweight() -> List[Tuple[str, str, str]]:
    """Discover tools without full orchestrator initialization.

    Delegates to ToolRegistry.discover_lightweight() for the actual discovery.
    This provides a single source of truth for tool discovery.

    Returns:
        List of tuples: (name, description, cost_tier)
    """
    from victor.tools.registry import ToolRegistry

    return ToolRegistry.discover_lightweight()


@tools_app.command("list")
def list_tools(  # Keep the command name as 'list' for the CLI
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Profile to use from profiles.yaml to initialize agent for tool listing.",
    ),
    lightweight: bool = typer.Option(
        False,
        "--lightweight",
        "-l",
        help="Fast discovery mode without full agent initialization (shows all tools, no enabled/disabled status).",
    ),
) -> None:
    """List all available tools with their descriptions.

    By default, initializes the full agent to show enabled/disabled status.
    Use --lightweight for faster listing without agent initialization.

    Examples:
        victor tools list
        victor tools list --lightweight
        victor tools list -l
    """
    if lightweight:
        _list_tools_lightweight()
    else:
        run_sync(_list_tools_async(profile))


def _list_tools_lightweight() -> None:
    """List tools using lightweight discovery (no agent initialization)."""
    console.print("[dim]Discovering tools (lightweight mode)...[/]")

    try:
        discovered_tools = _discover_tools_lightweight()

        if not discovered_tools:
            console.print("[yellow]No tools found.[/]")
            return

        # Remove duplicates and sort
        unique_tools = {}
        for name, description, cost_tier in discovered_tools:
            if name not in unique_tools:
                unique_tools[name] = (description, cost_tier)

        table = Table(title="Available Tools (Lightweight Discovery)", show_header=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Cost Tier", style="yellow")

        # Sort by name
        for name in sorted(unique_tools.keys()):
            description, cost_tier = unique_tools[name]
            # Take only the first line of description
            short_desc = description.split("\n")[0][:80]
            if len(description.split("\n")[0]) > 80:
                short_desc += "..."
            table.add_row(name, short_desc, cost_tier)

        console.print(table)
        console.print(f"\n[dim]Found {len(unique_tools)} tools[/]")
        console.print("[dim]Note: Use without --lightweight for enabled/disabled status[/]")

    except Exception as e:
        console.print(f"[red]Error discovering tools:[/] {e}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)


async def _list_tools_async(profile: str) -> None:
    """List tools with full agent initialization (shows enabled/disabled status)."""
    # ✅ PROPER: Import the framework client factory and SessionConfig
    from victor.framework.session_config import SessionConfig
    from victor.framework.session_runner import create_victor_client

    console.print(f"[dim]Initializing agent with profile '{profile}' to list tools...[/]")
    agent = None
    client = None
    try:
        # ✅ PROPER: Create the framework client (replaces AgentFactory)
        config = SessionConfig()  # Default config for tools listing
        client = create_victor_client(config)
        agent = await client.initialize()

        # Retrieve tools from the agent's ToolRegistry
        # list_tools(only_enabled=False) gets all registered tools, regardless of current enable/disable status
        all_tools = agent.tools.list_tools(only_enabled=False)

        if not all_tools:
            console.print("[yellow]No tools found.[/]")
            return

        table = Table(title="Available Tools", show_header=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Enabled", style="yellow")

        # Get the current enabled/disabled status for each tool
        tool_states = agent.tools.get_tool_states()

        # Sort tools by name for consistent output
        for tool in sorted(all_tools, key=lambda t: t.name):
            # Take only the first line of the description for brevity in the table
            description = (
                tool.description.split("\\n")[0]
                if tool.description
                else "No description available."
            )
            # Check the actual enabled status from the tool_states map
            enabled_status = (
                "[green]✓ Yes[/]" if tool_states.get(tool.name, False) else "[dim]✗ No[/]"
            )
            table.add_row(tool.name, description, enabled_status)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing tools:[/] {e}")
        # Log full traceback for debugging purposes during development
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)
    finally:
        # Ensure agent resources are properly cleaned up
        if agent:
            await agent.graceful_shutdown()


@tools_app.command("cleanup-containers")
def cleanup_containers(
    include_unlabeled: bool = typer.Option(
        False,
        "--include-unlabeled",
        "-u",
        help="Also remove unlabeled python-slim containers with 'sleep infinity' (legacy cleanup).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be removed without actually removing.",
    ),
) -> None:
    """Clean up orphaned sandbox containers.

    Removes code-executor sandbox containers whose owning Victor process is
    gone (owner labels: host, pid, process start time), plus pre-redesign
    'sleep infinity' sandboxes without owner labels. A live session's sandbox
    is never removed; containers owned by another host expire on their own
    idle TTL.

    Use --include-unlabeled to also clean up very old containers (python-slim
    images running 'sleep infinity') that lack the 'victor.sandbox' label
    entirely. Kubernetes-related containers are excluded.

    Examples:
        victor tools cleanup-containers
        victor tools cleanup-containers --include-unlabeled
        victor tools cleanup-containers --dry-run
    """
    try:
        from victor.core.utils.capability_loader import load_code_executor_module

        _module = load_code_executor_module()
        DOCKER_AVAILABLE = _module.DOCKER_AVAILABLE
        find_orphaned_sandboxes = _module.find_orphaned_sandboxes
    except ImportError:
        console.print(
            "[red]Container cleanup requires the 'victor-coding' package. "
            "Install it (pip install victor-coding) to use this command.[/]"
        )
        raise typer.Exit(1)

    if not DOCKER_AVAILABLE:
        console.print("[red]Docker is not available. Cannot cleanup containers.[/]")
        raise typer.Exit(1)

    try:
        import docker

        client = docker.from_env()

        # First, list what we would clean up
        containers_to_clean = []

        # Labelled sandboxes whose owner is gone (never a live session's)
        for c in find_orphaned_sandboxes(client):
            containers_to_clean.append((c.short_id, c.name, "orphaned", c))

        # Find unlabeled legacy containers if requested
        if include_unlabeled:
            # Kubernetes container name patterns to exclude
            # Note: Docker auto-generates names like "kind_hugle" which should NOT be excluded
            # Only exclude actual k8s infrastructure containers
            k8s_name_patterns = (
                "k8s_",  # Kubernetes pods
                "kube-",  # Kubernetes components
                "kind-",  # kind infrastructure (not kind_)
                "minikube",
                "desktop-",  # Docker Desktop k8s
            )
            all_containers = client.containers.list(all=True)
            for container in all_containers:
                try:
                    container_name = container.name.lower()
                    # Only skip if name starts with a k8s pattern
                    if any(container_name.startswith(p) for p in k8s_name_patterns):
                        continue

                    # Labelled sandboxes are decided by the owner check above only
                    if "victor.sandbox" in (container.labels or {}):
                        continue

                    image_tags = container.image.tags
                    is_python_slim = (
                        any("python" in tag and "slim" in tag for tag in image_tags)
                        if image_tags
                        else False
                    )

                    if not image_tags:
                        cmd = container.attrs.get("Config", {}).get("Cmd", [])
                        if cmd and "sleep" in str(cmd) and "infinity" in str(cmd):
                            is_python_slim = True

                    if is_python_slim:
                        cmd = container.attrs.get("Config", {}).get("Cmd", [])
                        if cmd and "sleep" in str(cmd) and "infinity" in str(cmd):
                            containers_to_clean.append(
                                (
                                    container.short_id,
                                    container.name,
                                    "unlabeled",
                                    container,
                                )
                            )
                except Exception:
                    pass

        if not containers_to_clean:
            console.print("[green]No orphaned containers found.[/]")
            return

        # Display containers
        table = Table(title="Containers to Clean Up", show_header=True)
        table.add_column("Container ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")

        for short_id, name, ctype, _ in containers_to_clean:
            table.add_row(short_id, name, ctype)

        console.print(table)
        console.print(f"\n[bold]Found {len(containers_to_clean)} container(s) to remove.[/]")

        if dry_run:
            console.print("[dim]Dry run - no containers removed.[/]")
            return

        # Confirm and clean
        if not typer.confirm("Remove these containers?"):
            console.print("[dim]Cancelled.[/]")
            return

        cleaned = 0
        for short_id, name, _, container in containers_to_clean:
            try:
                container.remove(force=True)
                console.print(f"[green]Removed:[/] {short_id} ({name})")
                cleaned += 1
            except Exception as e:
                console.print(f"[red]Failed to remove {short_id}:[/] {e}")

        console.print(f"\n[bold green]Cleaned up {cleaned} container(s).[/]")

    except Exception as e:
        console.print(f"[red]Error during cleanup:[/] {e}")
        raise typer.Exit(1)


@tools_app.command("cleanup-images")
def cleanup_images(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be removed without actually removing.",
    ),
) -> None:
    """Clean up dangling Docker images from sandbox operations.

    Removes dangling (untagged) images that may have been left behind
    from previous sandbox container operations. These are typically
    old versions of python-slim images that have been superseded.

    Examples:
        victor tools cleanup-images
        victor tools cleanup-images --dry-run
    """
    try:
        import docker

        client = docker.from_env()

        # Find dangling images
        dangling = client.images.list(filters={"dangling": True})

        if not dangling:
            console.print("[green]No dangling images found.[/]")
            return

        # Display images
        table = Table(title="Dangling Images to Clean Up", show_header=True)
        table.add_column("Image ID", style="cyan")
        table.add_column("Size", style="yellow")
        table.add_column("Created", style="dim")

        total_size = 0
        for img in dangling:
            size_mb = img.attrs.get("Size", 0) / (1024 * 1024)
            total_size += size_mb
            created = img.attrs.get("Created", "unknown")[:19]  # Truncate timestamp
            table.add_row(img.short_id, f"{size_mb:.1f} MB", created)

        console.print(table)
        console.print(
            f"\n[bold]Found {len(dangling)} dangling image(s) ({total_size:.1f} MB total).[/]"
        )

        if dry_run:
            console.print("[dim]Dry run - no images removed.[/]")
            return

        if not typer.confirm("Remove these images?"):
            console.print("[dim]Cancelled.[/]")
            return

        removed = 0
        for img in dangling:
            try:
                client.images.remove(img.id, force=True)
                console.print(f"[green]Removed:[/] {img.short_id}")
                removed += 1
            except Exception as e:
                console.print(f"[red]Failed to remove {img.short_id}:[/] {e}")

        console.print(f"\n[bold green]Cleaned up {removed} image(s).[/]")

    except ImportError:
        console.print("[red]Docker package not installed.[/]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error during cleanup:[/] {e}")
        raise typer.Exit(1)
