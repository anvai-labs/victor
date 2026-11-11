import typer
import os
import time

from rich.console import Console
from victor.ui.console import get_console

from victor.core.async_utils import run_sync
from victor.config.settings import load_settings
from victor.ui.commands.utils import preload_semantic_index

index_app = typer.Typer(name="index", help="Build semantic code search index for the codebase.")
console = get_console()


async def _build_index_async(cwd: str, settings, force: bool) -> bool:
    """Build the semantic index asynchronously."""
    return await preload_semantic_index(cwd, settings, console, force=force)


@index_app.callback(invoke_without_command=True)
def index(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force rebuild index even if up-to-date",
    ),
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Path to codebase root directory",
    ),
):
    """Build semantic code search index for the codebase."""
    if ctx.invoked_subcommand is None:
        settings = load_settings()
        cwd = os.path.abspath(path)

        if not os.path.isdir(cwd):
            console.print(f"[red]Error:[/red] Path '{path}' is not a directory")
            raise typer.Exit(1)

        console.print(f"[dim]Indexing codebase at: {cwd}[/dim]")

        start_time = time.time()
        success = run_sync(_build_index_async(cwd, settings, force))
        elapsed = time.time() - start_time

        if success:
            console.print(f"\n[green]✓ Indexing complete in {elapsed:.1f}s[/green]")
            console.print(
                "[dim]Run [cyan]victor chat --preindex[/cyan] to use pre-loaded index, "
                "or index will load from disk on first semantic search.[/dim]"
            )
        else:
            console.print("\n[red]✗ Indexing failed[/red]")
            raise typer.Exit(1)
