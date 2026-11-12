import importlib
import inspect
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm
from victor.ui.console import get_console

from victor.core.async_utils import run_sync
from victor.storage.cache.embedding_cache_manager import (
    CacheType,
    EmbeddingCacheManager,
)

embeddings_app = typer.Typer(name="embedding", help="Manage Victor embeddings for troubleshooting.")
console = get_console()


@embeddings_app.callback(invoke_without_command=True)
def embeddings(
    ctx: typer.Context,
    stat: bool = typer.Option(
        False, "--stat", "-s", help="Show detailed statistics with timestamps"
    ),
    clear: bool = typer.Option(
        False, "--clear", "-c", help="Clear embeddings (shows preview first)"
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", "-r", help="Clear and trigger rebuild immediately"
    ),
    tool: bool = typer.Option(
        False, "--tool", help="Target: tool embeddings (semantic tool selection)"
    ),
    intent: bool = typer.Option(
        False, "--intent", help="Target: task/intent classifier embeddings"
    ),
    conversation: bool = typer.Option(
        False, "--conversation", help="Target: conversation embeddings"
    ),
    all_embeddings: bool = typer.Option(False, "--all", "-a", help="Target: all embeddings"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Manage Victor embeddings for troubleshooting."""
    if ctx.invoked_subcommand is None:
        # Determine targets based on flags
        targets: list[CacheType] = []
        if tool:
            targets.append(CacheType.TOOL)
        if intent:
            targets.append(CacheType.INTENT)
        if conversation:
            targets.append(CacheType.CONVERSATION)
        if all_embeddings or (not targets and (clear or rebuild)):
            targets = [CacheType.TOOL, CacheType.INTENT, CacheType.CONVERSATION]

        if clear or rebuild:
            _clear_embeddings(targets, rebuild, yes)
        else:
            _show_status(stat, targets, clear, rebuild)


def _show_status(stat: bool, targets: list[CacheType], clear: bool, rebuild: bool):
    manager = EmbeddingCacheManager.get_instance()
    status = manager.get_status()

    console.print("\n[bold]Victor Embedding Status[/]")
    console.print("─" * 70)

    if stat:
        for cache_info in status.caches:
            will_clear = cache_info.cache_type in targets and (clear or rebuild)
            marker = (
                "[red]✗[/]"
                if will_clear and not cache_info.is_empty
                else ("[green]●[/]" if not cache_info.is_empty else "[dim]○[/]")
            )
            suffix = " [red]← will clear[/]" if will_clear and not cache_info.is_empty else ""

            console.print(f"\n  {marker} [bold]{cache_info.name}[/]{suffix}")
            console.print(f"      [dim]Purpose:[/]{cache_info.description}")
            console.print(f"      [dim]Location:[/]{cache_info.path}")
            console.print(f"      [dim]Files:[/]{cache_info.file_count} ({cache_info.size_str})")
            console.print(f"      [dim]Updated:[/]{cache_info.age_str}")

            if cache_info.file_count > 0 and cache_info.file_count <= 5:
                for f in cache_info.files:
                    console.print(f"        [dim]• {f.name} ({f.size_str}, {f.age_str})[/]")
            elif cache_info.file_count > 5:
                for f in cache_info.files[:3]:
                    console.print(f"        [dim]• {f.name} ({f.size_str}, {f.age_str})[/]")
                console.print(f"        [dim]... and {cache_info.file_count - 3} more[/]")
    else:
        for cache_info in status.caches:
            will_clear = cache_info.cache_type in targets and (clear or rebuild)
            marker = (
                "[red]✗[/]"
                if will_clear and not cache_info.is_empty
                else ("[green]●[/]" if not cache_info.is_empty else "[dim]○[/]")
            )
            suffix = " [red]← will clear[/]" if will_clear and not cache_info.is_empty else ""
            age = f" ({cache_info.age_str})" if cache_info.newest else ""
            console.print(
                f"  {marker} {cache_info.name}: {cache_info.file_count} files ({cache_info.size_str}){age}{suffix}"
            )

    console.print("─" * 70)
    console.print(f"  Total: {status.total_files} files ({status.total_size_str})")

    no_flags = not stat and not clear and not rebuild and not targets
    if no_flags:
        console.print("\n[bold]Commands:[/]")
        console.print("  [cyan]victor embeddings --stat[/]          Detailed stats with timestamps")
        console.print("  [cyan]victor embeddings --clear[/]         Clear embeddings")
        console.print("  [cyan]victor embeddings --rebuild[/]       Clear and rebuild immediately")
        console.print("\n[bold]Targets:[/]")
        console.print("  [cyan]--tool[/]         Tool embeddings (semantic tool selection)")
        console.print("  [cyan]--intent[/]       Task/intent classifiers")
        console.print("  [cyan]--conversation[/] Project conversation embeddings")
        console.print("  [cyan]--all[/]          All embeddings (default)")
        console.print("\n[dim]Combine: victor embeddings --rebuild --tool --yes[/]")


def _clear_embeddings(targets: list[CacheType], rebuild: bool, yes: bool):
    manager = EmbeddingCacheManager.get_instance()
    status = manager.get_status()

    target_files = sum(status.get_cache(t).file_count for t in targets)
    target_size = sum(status.get_cache(t).total_size for t in targets)

    if target_files == 0:
        console.print("\n[dim]Nothing to clear - selected embeddings are already empty.[/]")
        return

    target_names = [status.get_cache(t).name for t in targets]
    size_str = f"{target_size / 1024:.1f} KB" if target_size >= 1024 else f"{target_size} B"
    console.print(f"\n[bold yellow]Will clear: {', '.join(target_names)}[/]")
    console.print(f"[bold yellow]{target_files} files ({size_str})[/]")

    if not yes:
        console.print("")
        if not Confirm.ask("[yellow]Proceed?[/]"):  # default=False
            console.print("[dim]Cancelled.[/]")
            return

    console.print("\n[bold]Clearing...[/]")

    def progress_callback(msg: str):
        if msg.startswith("  "):
            if "ERROR" in msg:
                console.print(msg.replace("ERROR", "[red]ERROR[/]"))
            elif ": empty" in msg:
                console.print(f"  [dim]○[/]{msg[1:]}")
            else:
                console.print(f"  [green]✓[/]{msg[1:]}")
        else:
            console.print(f"  [dim]{msg}[/]")

    result = manager.clear(targets, progress_callback)

    size_str = (
        f"{result.cleared_size / 1024:.1f} KB"
        if result.cleared_size >= 1024
        else f"{result.cleared_size} B"
    )
    console.print(f"\n[green]✓ Cleared {result.cleared_files} files ({size_str})[/]")

    if rebuild:
        _rebuild_embeddings(targets, progress_callback)
    else:
        console.print("[dim]Embeddings will auto-rebuild on next 'victor chat'.[/]")


def _rebuild_embeddings(targets: list[CacheType], progress_callback):
    console.print("\n[bold]Rebuilding...[/]")
    manager = EmbeddingCacheManager.get_instance()
    try:
        if CacheType.INTENT in targets:
            phrase_count = manager.rebuild_task_classifiers_sync(progress_callback)
            console.print(f"  [green]✓[/] Task classifiers rebuilt ({phrase_count} phrases)")

        if CacheType.TOOL in targets:
            try:
                console.print("  [dim]Rebuilding tool embeddings...[/]")
                tool_count = _rebuild_tool_embeddings_sync()
                console.print(f"  [green]✓[/] Tool embeddings rebuilt ({tool_count} tools)")
            except Exception as e:
                console.print(f"  [yellow]⚠[/] Tool embeddings: {e}")

        if CacheType.CONVERSATION in targets:
            try:
                console.print("  [dim]Rebuilding conversation embeddings...[/]")
                msg_count = _rebuild_conversation_embeddings_sync()
                console.print(
                    f"  [green]✓[/] Conversation embeddings rebuilt ({msg_count} messages)"
                )
            except Exception as e:
                console.print(f"  [yellow]⚠[/] Conversation embeddings: {e}")

        console.print("\n[green]✓ Rebuild complete![/]")
    except Exception as e:
        console.print(f"\n[yellow]Rebuild skipped: {e}[/]")
        console.print("[dim]Caches will auto-rebuild on next 'victor chat'.[/]")


def _build_tool_registry():
    from victor.tools.registry import ToolRegistry

    registry = ToolRegistry()
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    excluded_files = {"__init__.py", "base.py", "decorators.py", "semantic_selector.py"}

    for module_path in tools_dir.iterdir():
        if module_path.suffix != ".py" or module_path.name in excluded_files:
            continue

        module_name = f"victor.tools.{module_path.stem}"
        try:
            module = importlib.import_module(module_name)
            for _name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                    registry.register(obj)
        except Exception:
            pass

    return registry


async def _rebuild_tool_embeddings_async(registry) -> int:
    from victor.tools.semantic_selector import SemanticToolSelector

    selector = SemanticToolSelector(cache_embeddings=True)
    try:
        await selector.initialize_tool_embeddings(registry)
        return len(registry.list_tools())
    finally:
        await selector.close()


def _rebuild_tool_embeddings_sync() -> int:
    registry = _build_tool_registry()
    return run_sync(_rebuild_tool_embeddings_async(registry))


async def _rebuild_conversation_embeddings_async() -> int:
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
    from victor.storage.embeddings.service import get_embedding_service

    embedding_service = get_embedding_service()
    store = ConversationEmbeddingStore(embedding_service)
    try:
        await store.initialize()
        return await store.rebuild()
    finally:
        await store.close()


def _rebuild_conversation_embeddings_sync() -> int:
    return run_sync(_rebuild_conversation_embeddings_async())
