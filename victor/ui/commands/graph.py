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

"""Graph operations commands for Victor CLI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator, Optional

import typer
from rich.console import Console
from rich.table import Table

from victor.config.settings import get_project_paths, load_settings
from victor.core.async_utils import run_sync
from victor.storage.graph import create_graph_store

graph_app = typer.Typer(name="graph", help="Graph-based code intelligence operations.")
watch_app = typer.Typer(help="Keep the persisted graph incrementally updated.")
console = Console()
graph_app.add_typer(watch_app, name="watch")


@dataclass
class GraphWatchDaemonState:
    """Observed or requested state for a graph watcher daemon."""

    pid_file: Path
    running: bool = False
    pid: Optional[int] = None
    started: bool = False
    stopped: bool = False
    stale_pid_file: bool = False
    stale_pid_removed: bool = False


def _default_graph_watch_manifest_file(project_root: Path) -> Path:
    """Return the manifest file for a project-scoped graph watcher."""
    paths = get_project_paths(project_root)
    return paths.project_victor_dir / "graph-watch.json"


async def _index_async(
    path: str,
    enable_ccg: bool,
    force: bool,
    enable_embeddings: bool = False,
) -> bool:
    """Index codebase into graph store asynchronously.

    Args:
        path: Path to codebase root
        enable_ccg: Whether to build CCG
        force: Force rebuild
        enable_embeddings: Whether to generate node embeddings for semantic search

    Returns:
        True if successful
    """
    from victor.core.graph_rag import GraphIndexConfig, GraphIndexingPipeline
    from victor.core.graph_rag.indexing import run_indexing_with_lock

    root_path = Path(path).resolve()

    if not root_path.is_dir():
        console.print(f"[red]Error:[/red] Path '{path}' is not a directory")
        return False

    console.print(f"[dim]Indexing codebase at: {root_path}[/dim]")

    # Create graph store
    graph_store = create_graph_store("sqlite", root_path)
    await graph_store.initialize()

    # Clear existing if forced
    if force:
        await graph_store.delete_by_repo()
        console.print("[dim]Cleared existing graph data[/dim]")

    # Configure indexing
    config = GraphIndexConfig(
        root_path=root_path,
        enable_ccg=enable_ccg,
        enable_embeddings=enable_embeddings,
    )

    # Build index
    pipeline = GraphIndexingPipeline(graph_store, config)
    stats = await run_indexing_with_lock(root_path, pipeline.index_repository)

    if force or stats.files_processed or stats.files_deleted:
        try:
            from victor.core.graph_rag.query_cache import get_graph_query_cache

            get_graph_query_cache().invalidate_repo(str(root_path))
        except Exception:  # noqa: BLE001 - cache invalidation is best-effort
            pass

    # Print results
    console.print("\n[green]✓ Indexing complete[/green]")
    console.print(f"  Files processed: {stats.files_processed}")
    console.print(f"  Nodes created: {stats.nodes_created}")
    console.print(f"  Edges created: {stats.edges_created}")
    if stats.ccg_nodes_created > 0:
        console.print(f"  CCG nodes: {stats.ccg_nodes_created}")
        console.print(f"  CCG edges: {stats.ccg_edges_created}")
    if stats.embeddings_generated > 0:
        console.print(f"  Embeddings generated: {stats.embeddings_generated}")
    if stats.error_count > 0:
        console.print(f"[yellow]  Warnings: {stats.error_count}[/yellow]")

    return stats.error_count == 0


async def _query_async(
    query: str,
    path: str,
    mode: str,
    max_hops: int,
    max_results: int,
) -> bool:
    """Query code graph using natural language.

    Args:
        query: Natural language query
        path: Path to search within
        mode: Query mode
        max_hops: Maximum hops
        max_results: Maximum results

    Returns:
        True if successful
    """
    from victor.core.graph_rag import MultiHopRetriever, RetrievalConfig

    root_path = Path(path).resolve()

    # Create graph store
    graph_store = create_graph_store("sqlite", root_path)
    await graph_store.initialize()

    # Check if graph exists
    stats = await graph_store.stats()
    if stats.get("nodes", 0) == 0:
        console.print(f"[red]Error:[/red] No graph data found at {path}")
        console.print("[dim]Run 'victor graph index' first[/dim]")
        return False

    # Create retriever
    config = RetrievalConfig(
        seed_count=max_results,
        max_hops=max_hops,
        top_k=max_results,
    )
    retriever = MultiHopRetriever(graph_store, config)

    # Execute query
    result = await retriever.retrieve(query, config)

    # Print results
    console.print(f"\n[green]Query results for:[/green] '{query}'")
    console.print(
        f"[dim]Found {len(result.nodes)} symbols in {result.execution_time_ms:.1f}ms[/dim]\n"
    )

    if not result.nodes:
        console.print("[dim]No matching symbols found[/dim]")
        return True

    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("File", style="blue")
    table.add_column("Line", style="yellow")
    table.add_column("Score", style="magenta")

    for node in result.nodes[:20]:
        line_str = str(node.line) if node.line else "-"
        score_str = f"{result.scores.get(node.node_id, 0):.2f}"
        table.add_row(
            node.name,
            node.type,
            node.file,
            line_str,
            score_str,
        )

    console.print(table)

    if len(result.nodes) > 20:
        console.print(f"[dim]... and {len(result.nodes) - 20} more[/dim]")

    return True


async def _impact_async(
    target: str,
    analysis_type: str,
    max_depth: int,
    path: str,
) -> bool:
    """Analyze impact of code changes.

    Args:
        target: Target symbol or file:line
        analysis_type: forward or backward
        max_depth: Maximum depth
        path: Path to codebase

    Returns:
        True if successful
    """
    from victor.storage.graph.edge_types import EdgeType

    root_path = Path(path).resolve()

    # Create graph store
    graph_store = create_graph_store("sqlite", root_path)
    await graph_store.initialize()

    # Resolve target
    target_node_id = await _resolve_target(target, graph_store)
    if target_node_id is None:
        console.print(f"[red]Error:[/red] Could not resolve target: {target}")
        return False

    # Traverse for impact
    direction = "out" if analysis_type == "forward" else "in"
    edges = await graph_store.get_neighbors(
        target_node_id,
        direction=direction,
        max_depth=max_depth,
    )

    # Collect impacted nodes
    impacted_ids: set[str] = set()
    for edge in edges:
        if direction == "out":
            impacted_ids.add(edge.dst)
        else:
            impacted_ids.add(edge.src)

    # Get node details
    impacted_nodes = []
    for node_id in list(impacted_ids)[:50]:
        node = await graph_store.get_node_by_id(node_id)
        if node:
            impacted_nodes.append(node)

    # Print results
    direction_str = "downstream" if analysis_type == "forward" else "upstream"
    console.print(f"\n[green]Impact Analysis:[/green] {target}")
    console.print(f"[dim]Analysis type: {direction_str} (max depth: {max_depth})[/dim]")
    console.print(f"[dim]Found {len(impacted_nodes)} impacted symbols\n")

    if not impacted_nodes:
        console.print("[dim]No impacted symbols found[/dim]")
        return True

    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("File", style="blue")
    table.add_column("Line", style="yellow")

    for node in impacted_nodes[:20]:
        line_str = str(node.line) if node.line else "-"
        table.add_row(node.name, node.type, node.file, line_str)

    console.print(table)

    if len(impacted_nodes) > 20:
        console.print(f"[dim]... and {len(impacted_nodes) - 20} more[/dim]")

    return True


async def _resolve_target(target: str, graph_store) -> Optional[str]:
    """Resolve a target string to a node ID."""
    from victor.storage.graph.edge_types import EdgeType

    # Check if file:line format
    if ":" in target:
        file_path, line_str = target.rsplit(":", 1)
        try:
            line = int(line_str)
            nodes = await graph_store.get_nodes_by_file(file_path)
            for node in nodes:
                if node.line and node.line <= line <= (node.end_line or line):
                    return node.node_id
        except ValueError:
            pass

    # Search by name
    nodes = await graph_store.find_nodes(name=target)
    if nodes:
        return nodes[0].node_id

    return None


async def _stats_async(path: str) -> bool:
    """Show graph statistics.

    Args:
        path: Path to codebase

    Returns:
        True if successful
    """
    root_path = Path(path).resolve()

    # Create graph store
    graph_store = create_graph_store("sqlite", root_path)
    await graph_store.initialize()

    # Get stats
    stats = await graph_store.stats()

    # Print stats
    console.print(f"\n[green]Graph Statistics:[/green] {path}\n")

    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Nodes", str(stats.get("nodes", 0)))
    table.add_row("Edges", str(stats.get("edges", 0)))
    table.add_row("Indexed files", str(stats.get("indexed_files", 0)))

    console.print(table)

    return True


async def _export_async(
    path: str,
    output: str,
    format: str,
) -> bool:
    """Export graph to file.

    Args:
        path: Path to codebase
        output: Output file path
        format: Export format (dot, json)

    Returns:
        True if successful
    """
    root_path = Path(path).resolve()
    output_path = Path(output)

    # Create graph store
    graph_store = create_graph_store("sqlite", root_path)
    await graph_store.initialize()

    # Get all nodes and edges
    nodes = await graph_store.get_all_nodes()
    edges = await graph_store.get_all_edges()

    if format == "json":
        # Export as JSON
        data = {
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.type,
                    "name": n.name,
                    "file": n.file,
                    "line": n.line,
                    "end_line": n.end_line,
                    "lang": n.lang,
                    "signature": n.signature,
                    "docstring": n.docstring,
                }
                for n in nodes
            ],
            "edges": [
                {
                    "source": e.src,
                    "target": e.dst,
                    "type": e.type,
                    "weight": e.weight,
                }
                for e in edges
            ],
        }

        with output_path.open("w") as f:
            json.dump(data, f, indent=2)

        console.print(
            f"[green]✓ Exported {len(nodes)} nodes and {len(edges)} edges to {output}[/green]"
        )

    elif format == "dot":
        # Export as DOT (GraphViz)
        with output_path.open("w") as f:
            f.write("digraph code_graph {\n")
            f.write("  rankdir=LR;\n\n")

            # Nodes
            for node in nodes:
                label = f"{node.name}\\n{node.type}"
                f.write(f'  "{node.node_id}" [label="{label}"];\n')

            f.write("\n")

            # Edges
            for edge in edges:
                f.write(f'  "{edge.src}" -> "{edge.dst}" [label="{edge.type}"];\n')

            f.write("}\n")

        console.print(
            f"[green]✓ Exported {len(nodes)} nodes and {len(edges)} edges to {output}[/green]"
        )
        console.print("[dim]Render with: dot -Tpng graph.dot -o graph.png[/dim]")

    else:
        console.print(f"[red]Error:[/red] Unsupported format: {format}")
        return False

    return True


# ============================================================================
# CLI Commands
# ============================================================================


@graph_app.command("index")
def graph_index(
    path: str = typer.Option(".", "--path", "-p", help="Path to codebase root"),
    enable_ccg: bool = typer.Option(True, "--ccg/--no-ccg", help="Build Code Context Graph"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild"),
    embeddings: bool = typer.Option(
        False,
        "--embeddings/--no-embeddings",
        help="Generate node embeddings for semantic graph search",
    ),
):
    """Index codebase into graph store."""
    cwd = os.path.abspath(path)

    start_time = time.time()
    success = run_sync(_index_async(cwd, enable_ccg, force, embeddings))
    elapsed = time.time() - start_time

    if success:
        console.print(f"[green]✓ Complete in {elapsed:.1f}s[/green]")
    else:
        raise typer.Exit(1)


@graph_app.command("query")
def graph_query(
    query: str = typer.Argument(..., help="Natural language query"),
    path: str = typer.Option(".", "--path", "-p", help="Path to search within"),
    mode: str = typer.Option("semantic", "--mode", "-m", help="Query mode"),
    max_hops: int = typer.Option(2, "--hops", "-H", help="Maximum hops", min=1, max=3),
    max_results: int = typer.Option(10, "--results", "-r", help="Maximum results", min=1, max=50),
):
    """Query code graph using natural language."""
    cwd = os.path.abspath(path)

    success = run_sync(_query_async(query, cwd, mode, max_hops, max_results))

    if not success:
        raise typer.Exit(1)


@graph_app.command("impact")
def graph_impact(
    target: str = typer.Argument(..., help="Target symbol or file:line"),
    analysis_type: str = typer.Option("forward", "--type", "-t", help="Analysis type"),
    max_depth: int = typer.Option(3, "--depth", "-d", help="Maximum depth", min=1, max=5),
    path: str = typer.Option(".", "--path", "-p", help="Path to codebase"),
):
    """Analyze impact of code changes using CCG."""
    cwd = os.path.abspath(path)

    success = run_sync(_impact_async(target, analysis_type, max_depth, cwd))

    if not success:
        raise typer.Exit(1)


@graph_app.command("stats")
def graph_stats(
    path: str = typer.Option(".", "--path", "-p", help="Path to codebase"),
):
    """Show graph statistics."""
    cwd = os.path.abspath(path)

    success = run_sync(_stats_async(cwd))

    if not success:
        raise typer.Exit(1)


@graph_app.command("export")
def graph_export(
    output: str = typer.Option("graph.json", "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Export format"),
    path: str = typer.Option(".", "--path", "-p", help="Path to codebase"),
):
    """Export graph to file (DOT or JSON)."""
    cwd = os.path.abspath(path)

    success = run_sync(_export_async(cwd, output, format))

    if not success:
        raise typer.Exit(1)


async def _init_context_async(
    path: str,
    task: Optional[str],
    max_symbols: int,
    force: bool,
) -> bool:
    """Generate graph-enhanced init.md.

    Args:
        path: Path to codebase root
        task: Optional task description for context relevance
        max_symbols: Maximum symbols to include
        force: Force overwrite existing file

    Returns:
        True if successful
    """
    from victor.context.project_context import init_victor_md_with_graph

    root_path = Path(path).resolve()

    if not root_path.is_dir():
        console.print(f"[red]Error:[/red] Path '{path}' is not a directory")
        return False

    console.print(f"[dim]Generating graph-enhanced init.md for: {root_path}[/dim]")

    # Check if graph exists
    graph_store = create_graph_store("sqlite", root_path)
    await graph_store.initialize()

    stats = await graph_store.stats()
    node_count = stats.get("node_count", stats.get("nodes", 0))

    if node_count == 0:
        console.print("[yellow]Warning:[/yellow] No graph data found")
        console.print("[dim]Run 'victor graph index' first to build the graph[/dim]")
        console.print("[dim]Proceeding with standard init.md generation...[/dim]")

    await graph_store.close()

    # Generate init.md with graph context
    result_path = await init_victor_md_with_graph(
        root_path=root_path,
        force=force,
        task=task,
        max_symbols=max_symbols,
    )

    if result_path:
        console.print(f"[green]✓ Created {result_path}[/green]")
        if node_count > 0:
            console.print(f"[dim]  Included graph context from {node_count} nodes[/dim]")
        return True
    else:
        console.print("[yellow]init.md already exists[/yellow]")
        console.print("[dim]Use --force to overwrite[/dim]")
        return False


def _default_graph_watch_pid_file(project_root: Path) -> Path:
    """Return the default graph-watch PID file for a project root."""
    paths = get_project_paths(project_root)
    return paths.project_victor_dir / "graph-watch.pid"


def _default_graph_watch_lock_file(project_root: Path) -> Path:
    """Return the startup lock file for a project-scoped graph watcher."""
    paths = get_project_paths(project_root)
    return paths.project_victor_dir / "graph-watch.lock"


def _write_graph_watch_manifest(
    project_root: Path,
    state: GraphWatchDaemonState,
    *,
    enable_ccg: Optional[bool] = None,
    build_now: Optional[bool] = None,
    poll_interval: Optional[float] = None,
    debounce_seconds: Optional[float] = None,
    min_refresh_interval: Optional[float] = None,
    owner: Optional[str] = None,
    last_refresh: Optional[dict[str, Any]] = None,
    manifest_file: Optional[Path] = None,
) -> Path:
    """Persist project-scoped graph watcher state for other sessions."""
    root_path = project_root.resolve()
    resolved_manifest_file = manifest_file or _default_graph_watch_manifest_file(root_path)
    resolved_manifest_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {}
    if resolved_manifest_file.exists():
        try:
            existing_payload = json.loads(resolved_manifest_file.read_text(encoding="utf-8"))
            if isinstance(existing_payload, dict):
                payload.update(existing_payload)
        except (OSError, json.JSONDecodeError):
            payload = {}

    payload.update(
        {
            "project_root": str(root_path),
            "pid_file": str(state.pid_file),
            "lock_file": str(_default_graph_watch_lock_file(root_path)),
            "running": state.running,
            "pid": state.pid,
            "started": state.started,
            "stopped": state.stopped,
            "stale_pid_file": state.stale_pid_file,
            "stale_pid_removed": state.stale_pid_removed,
            "updated_at": time.time(),
        }
    )
    if enable_ccg is not None:
        payload["enable_ccg"] = enable_ccg
    if build_now is not None:
        payload["build_now"] = build_now
    if poll_interval is not None:
        payload["poll_interval"] = poll_interval
    if debounce_seconds is not None:
        payload["debounce_seconds"] = debounce_seconds
    if min_refresh_interval is not None:
        payload["min_refresh_interval"] = min_refresh_interval
    if owner is not None:
        payload["owner"] = owner
    if last_refresh is not None:
        payload["last_refresh"] = last_refresh

    resolved_manifest_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_manifest_file


def _read_graph_watch_manifest(project_root: Path) -> Optional[dict[str, Any]]:
    """Load the graph watch manifest for a project if it exists."""
    manifest_file = _default_graph_watch_manifest_file(project_root.resolve())
    if not manifest_file.exists():
        return None
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _record_graph_watch_refresh(project_root: Path, stats: Any) -> Path:
    """Persist the latest incremental refresh health for graph watch status."""
    root_path = project_root.resolve()
    state = _inspect_graph_watch_daemon(
        _default_graph_watch_pid_file(root_path), remove_stale=False
    )
    return _write_graph_watch_manifest(
        root_path,
        state,
        last_refresh={
            "changed": getattr(stats, "files_processed", 0),
            "deleted": getattr(stats, "files_deleted", 0),
            "unchanged": getattr(stats, "files_unchanged", 0),
            "errors": getattr(stats, "error_count", 0),
            "duration_seconds": getattr(stats, "processing_time_seconds", 0.0),
            "completed_at": time.time(),
        },
    )


def _record_graph_watch_refresh_failure(project_root: Path, exc: Exception) -> Path:
    """Persist the latest incremental refresh failure for graph watch status."""
    from victor.core.indexing.graph_manager import classify_refresh_error

    root_path = project_root.resolve()
    state = _inspect_graph_watch_daemon(
        _default_graph_watch_pid_file(root_path), remove_stale=False
    )
    failure = classify_refresh_error(exc)
    return _write_graph_watch_manifest(
        root_path,
        state,
        last_refresh={
            "changed": 0,
            "deleted": 0,
            "unchanged": 0,
            "errors": 1,
            "duration_seconds": 0.0,
            "completed_at": time.time(),
            "last_error": str(exc),
            "error_type": type(exc).__name__,
            "error_code": failure.get("error_code"),
            "error_category": failure.get("category"),
            "recoverable": failure.get("recoverable"),
            "severity": failure.get("severity"),
            "retry_delay_seconds": failure.get("retry_delay_seconds"),
            "operator_guidance": failure.get("operator_guidance"),
        },
    )


def summarize_graph_watch_startup(
    project_root: Path,
    state: GraphWatchDaemonState,
    *,
    manifest: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Return concise startup messages describing project graph watch state."""
    root_path = project_root.resolve()
    active_manifest = manifest if manifest is not None else _read_graph_watch_manifest(root_path)
    messages: list[str] = []

    if state.stale_pid_removed:
        messages.append("Recovered stale graph watch state before chat startup.")

    if state.running and state.pid is not None:
        if state.started:
            messages.append(f"Graph watch daemon started for this project (PID {state.pid}).")
        else:
            messages.append(f"Graph watch daemon active for this project (PID {state.pid}).")

    if active_manifest and isinstance(active_manifest.get("last_refresh"), dict):
        last_refresh = active_manifest["last_refresh"]
        refresh_label = "Previous refresh" if state.started else "Last refresh"
        messages.append(
            f"{refresh_label}: "
            f"changed={last_refresh.get('changed', 0)}, "
            f"deleted={last_refresh.get('deleted', 0)}, "
            f"unchanged={last_refresh.get('unchanged', 0)}, "
            f"errors={last_refresh.get('errors', 0)}, "
            f"duration={float(last_refresh.get('duration_seconds', 0.0)):.2f}s."
        )

    return messages


def summarize_graph_watch_health(manifest: Optional[dict[str, Any]]) -> tuple[str, str]:
    """Return a compact status-bar summary for graph watch health."""
    if not isinstance(manifest, dict):
        return "Graph: off", "inactive"

    running = bool(manifest.get("running"))
    stale_pid_file = bool(manifest.get("stale_pid_file"))
    last_refresh = manifest.get("last_refresh")

    if stale_pid_file and not running:
        return "Graph: stale", "warning"
    if not running:
        return "Graph: off", "inactive"
    if not isinstance(last_refresh, dict):
        return "Graph: starting", "active"

    changed = int(last_refresh.get("changed", 0) or 0)
    deleted = int(last_refresh.get("deleted", 0) or 0)
    unchanged = int(last_refresh.get("unchanged", 0) or 0)
    errors = int(last_refresh.get("errors", 0) or 0)

    if errors > 0:
        retry_delay_seconds = last_refresh.get("retry_delay_seconds")
        if isinstance(retry_delay_seconds, (int, float)) and retry_delay_seconds > 0:
            return f"Graph: retry {int(round(float(retry_delay_seconds)))}s", "warning"
        return f"Graph: err {errors} c{changed} d{deleted}", "warning"
    return f"Graph: ok c{changed} d{deleted} u{unchanged}", "active"


def _graph_watch_lock_is_stale(lock_file: Path, stale_after_seconds: float) -> bool:
    """Return True when a graph-watch startup lock is old enough to reap."""
    try:
        return (time.time() - lock_file.stat().st_mtime) > stale_after_seconds
    except OSError:
        return False


@contextmanager
def _acquire_graph_watch_startup_lock(
    project_root: Path,
    *,
    lock_file: Optional[Path] = None,
    timeout_seconds: float = 5.0,
    poll_interval: float = 0.05,
    stale_after_seconds: float = 30.0,
) -> Iterator[Path]:
    """Acquire a project-scoped startup lock for graph watcher daemon changes."""
    resolved_lock_file = lock_file or _default_graph_watch_lock_file(project_root)
    resolved_lock_file.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + timeout_seconds
    payload = json.dumps({"pid": os.getpid(), "acquired_at": time.time()})

    while True:
        try:
            fd = os.open(
                resolved_lock_file,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            break
        except FileExistsError:
            if _graph_watch_lock_is_stale(resolved_lock_file, stale_after_seconds):
                with suppress(FileNotFoundError):
                    resolved_lock_file.unlink()
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for graph watch startup lock: {resolved_lock_file}"
                )
            time.sleep(poll_interval)

    try:
        yield resolved_lock_file
    finally:
        with suppress(FileNotFoundError):
            resolved_lock_file.unlink()


def _resolve_graph_watch_pid_file(project_root: Path, pid_file: Optional[Path]) -> Path:
    """Resolve an explicit or default PID file for graph watching."""
    return pid_file if pid_file is not None else _default_graph_watch_pid_file(project_root)


def _inspect_graph_watch_daemon(
    pid_file: Path,
    *,
    remove_stale: bool = False,
) -> GraphWatchDaemonState:
    """Inspect the daemon state represented by a PID file."""
    state = GraphWatchDaemonState(pid_file=pid_file)
    if not pid_file.exists():
        return state

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        state.stale_pid_file = True
        if remove_stale:
            with suppress(FileNotFoundError):
                pid_file.unlink()
            state.stale_pid_removed = True
        return state

    state.pid = pid
    try:
        os.kill(pid, 0)
        state.running = True
    except PermissionError:
        state.running = True
    except ProcessLookupError:
        state.stale_pid_file = True
        if remove_stale:
            with suppress(FileNotFoundError):
                pid_file.unlink()
            state.stale_pid_removed = True

    return state


async def _watch_async(
    path: str,
    enable_ccg: bool,
    poll_interval: float,
    debounce_seconds: float,
    min_refresh_interval: float,
    build_now: bool,
) -> bool:
    """Run foreground graph watching until interrupted."""
    from victor.core.indexing.graph_manager import GraphManager
    from victor.core.indexing.watcher_initializer import stop_file_watchers

    root_path = Path(path).resolve()
    if not root_path.is_dir():
        console.print(f"[red]Error:[/red] Path '{path}' is not a directory")
        return False

    manager = GraphManager.get_instance()

    def _on_refresh_complete(refreshed_root: Path, stats: Any) -> None:
        _record_graph_watch_refresh(refreshed_root, stats)

    def _on_refresh_error(refreshed_root: Path, exc: Exception) -> None:
        _record_graph_watch_refresh_failure(refreshed_root, exc)

    initial_stats = await manager.ensure_background_refresh(
        root_path,
        enable_ccg=enable_ccg,
        poll_interval_seconds=poll_interval,
        debounce_seconds=debounce_seconds,
        min_refresh_interval_seconds=min_refresh_interval,
        build_now=build_now,
        on_refresh_complete=_on_refresh_complete,
        on_refresh_error=_on_refresh_error,
    )

    if initial_stats is not None:
        console.print(
            "[dim]Initial graph refresh: "
            f"{initial_stats.files_processed} changed, "
            f"{initial_stats.files_deleted} deleted, "
            f"{initial_stats.files_unchanged} unchanged[/dim]"
        )

    console.print(f"[green]Watching graph updates for {root_path}[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_event.set()

    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(handled_signal, _request_stop)

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        await manager.stop_background_refresh(root_path)
        await stop_file_watchers([root_path])
        for handled_signal in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.remove_signal_handler(handled_signal)

    return True


def _spawn_watch_daemon(
    pid_file: Path,
    path: str,
    enable_ccg: bool,
    poll_interval: float,
    debounce_seconds: float,
    min_refresh_interval: float,
    build_now: bool,
) -> int:
    """Spawn the graph watcher in a fresh interpreter.

    The chat process imports optional native vector stores such as LanceDB before
    starting the graph watcher. LanceDB is not fork-safe, so daemon startup must
    use subprocess spawn/exec semantics rather than os.fork().
    """
    log_file = pid_file.with_suffix(".log")
    command = [
        sys.executable,
        "-m",
        "victor.ui.cli",
        "graph",
        "watch",
        "start",
        "--path",
        path,
        "--foreground-daemon",
        "--poll-interval",
        str(poll_interval),
        "--debounce",
        str(debounce_seconds),
        "--min-refresh-interval",
        str(min_refresh_interval),
        "--ccg" if enable_ccg else "--no-ccg",
        "--build-now" if build_now else "--no-build-now",
    ]

    with log_file.open("a", buffering=1, encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=path,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    pid_file.write_text(str(process.pid), encoding="utf-8")
    return int(process.pid)


def ensure_graph_watch_daemon(
    project_root: Path,
    *,
    enable_ccg: bool,
    build_now: bool = False,
    pid_file: Optional[Path] = None,
    poll_interval: float = 1.0,
    debounce_seconds: float = 0.3,
    min_refresh_interval: float = 30.0,
    owner: str = "explicit",
) -> GraphWatchDaemonState:
    """Ensure a background graph watcher daemon exists for a project."""
    root_path = project_root.resolve()
    resolved_pid_file = _resolve_graph_watch_pid_file(root_path, pid_file)
    resolved_pid_file.parent.mkdir(parents=True, exist_ok=True)
    with _acquire_graph_watch_startup_lock(root_path):
        state = _inspect_graph_watch_daemon(resolved_pid_file, remove_stale=True)
        if state.running:
            _write_graph_watch_manifest(
                root_path,
                state,
                enable_ccg=enable_ccg,
                build_now=build_now,
                poll_interval=poll_interval,
                debounce_seconds=debounce_seconds,
                min_refresh_interval=min_refresh_interval,
                owner=owner,
            )
            return state

        pid = _spawn_watch_daemon(
            resolved_pid_file,
            str(root_path),
            enable_ccg,
            poll_interval,
            debounce_seconds,
            min_refresh_interval,
            build_now,
        )
        state = GraphWatchDaemonState(
            pid_file=resolved_pid_file,
            running=True,
            pid=pid,
            started=True,
            stale_pid_removed=state.stale_pid_removed,
        )
        _write_graph_watch_manifest(
            root_path,
            state,
            enable_ccg=enable_ccg,
            build_now=build_now,
            poll_interval=poll_interval,
            debounce_seconds=debounce_seconds,
            min_refresh_interval=min_refresh_interval,
            owner=owner,
        )
        return state


def stop_graph_watch_daemon(
    project_root: Path,
    *,
    pid_file: Optional[Path] = None,
) -> GraphWatchDaemonState:
    """Stop the background graph watcher daemon for a project."""
    root_path = project_root.resolve()
    resolved_pid_file = _resolve_graph_watch_pid_file(root_path, pid_file)
    with _acquire_graph_watch_startup_lock(root_path):
        state = _inspect_graph_watch_daemon(resolved_pid_file, remove_stale=False)

        if state.running and state.pid is not None:
            os.kill(state.pid, signal.SIGTERM)
            with suppress(FileNotFoundError):
                resolved_pid_file.unlink()
            state.running = False
            state.stopped = True
            _write_graph_watch_manifest(root_path, state)
            return state

        if state.stale_pid_file:
            with suppress(FileNotFoundError):
                resolved_pid_file.unlink()
            state.stale_pid_removed = True

        _write_graph_watch_manifest(root_path, state)
        return state


@watch_app.command("start")
def graph_watch_start(
    path: str = typer.Option(".", "--path", "-p", help="Path to codebase root"),
    enable_ccg: bool = typer.Option(True, "--ccg/--no-ccg", help="Build Code Context Graph"),
    daemon: bool = typer.Option(False, "--daemon", help="Run watcher as background daemon"),
    foreground_daemon: bool = typer.Option(
        False,
        "--foreground-daemon",
        help="Internal: run daemon worker in the current process",
        hidden=True,
    ),
    pid_file: Optional[Path] = typer.Option(None, "--pid-file", help="PID file for daemon mode"),
    poll_interval: float = typer.Option(
        1.0, "--poll-interval", help="File watcher poll interval in seconds", min=0.1
    ),
    debounce_seconds: float = typer.Option(
        0.3, "--debounce", help="Debounce window before incremental refresh", min=0.0
    ),
    min_refresh_interval: float = typer.Option(
        30.0,
        "--min-refresh-interval",
        help="Minimum seconds between successful graph refresh passes",
        min=0.0,
    ),
    build_now: bool = typer.Option(
        True,
        "--build-now/--no-build-now",
        help="Run one incremental refresh immediately on startup",
    ),
):
    """Watch a project and keep its persisted graph updated incrementally."""
    root_path = Path(path).resolve()

    if foreground_daemon is True:
        success = run_sync(
            _watch_async(
                str(root_path),
                enable_ccg,
                poll_interval,
                debounce_seconds,
                min_refresh_interval,
                build_now,
            )
        )
        if not success:
            raise typer.Exit(1)
        return

    if daemon is True:
        try:
            state = ensure_graph_watch_daemon(
                root_path,
                enable_ccg=enable_ccg,
                build_now=build_now,
                pid_file=pid_file,
                poll_interval=poll_interval,
                debounce_seconds=debounce_seconds,
                min_refresh_interval=min_refresh_interval,
                owner="explicit",
            )
        except (OSError, TimeoutError) as exc:
            console.print(f"[red]Failed to start graph watcher: {exc}[/]")
            raise typer.Exit(1)

        if state.stale_pid_removed:
            console.print("[dim]Recovered stale PID file before starting watcher[/]")
        if state.started and state.pid is not None:
            console.print(f"[green]Graph watcher started (PID {state.pid})[/]")
            console.print(f"[dim]PID file: {state.pid_file}[/]")
        elif state.running and state.pid is not None:
            console.print(f"[yellow]Graph watcher already running (PID {state.pid})[/]")
        return

    success = run_sync(
        _watch_async(
            str(root_path),
            enable_ccg,
            poll_interval,
            debounce_seconds,
            min_refresh_interval,
            build_now,
        )
    )
    if not success:
        raise typer.Exit(1)


@watch_app.command("stop")
def graph_watch_stop(
    path: str = typer.Option(".", "--path", "-p", help="Path to codebase root"),
    pid_file: Optional[Path] = typer.Option(None, "--pid-file", help="PID file for daemon mode"),
):
    """Stop the graph watcher daemon for a project."""
    root_path = Path(path).resolve()
    try:
        state = stop_graph_watch_daemon(root_path, pid_file=pid_file)
    except TimeoutError as exc:
        console.print(f"[red]Failed to stop graph watcher: {exc}[/]")
        raise typer.Exit(1)

    if state.stopped and state.pid is not None:
        console.print(f"[green]Graph watcher stopped (PID {state.pid})[/]")
        return
    if state.stale_pid_removed:
        console.print("[yellow]Graph watcher was not running (stale PID file removed)[/]")
        return

    console.print("[yellow]Graph watcher is not running (no PID file)[/]")
    raise typer.Exit(0)


@watch_app.command("status")
def graph_watch_status(
    path: str = typer.Option(".", "--path", "-p", help="Path to codebase root"),
    pid_file: Optional[Path] = typer.Option(None, "--pid-file", help="PID file for daemon mode"),
):
    """Show graph watcher daemon status for a project."""
    root_path = Path(path).resolve()
    resolved_pid_file = _resolve_graph_watch_pid_file(root_path, pid_file)
    state = _inspect_graph_watch_daemon(resolved_pid_file, remove_stale=False)
    manifest = _read_graph_watch_manifest(root_path)

    console.print(f"\n[green]Graph Watcher Status:[/green] {root_path}\n")
    table = Table(show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    if state.running:
        status_label = "running"
    elif state.stale_pid_file:
        status_label = "stale PID file"
    else:
        status_label = "not running"
    table.add_row("State", status_label)
    table.add_row("Running", "yes" if state.running else "no")
    table.add_row("PID file", str(resolved_pid_file))
    table.add_row("Manifest file", str(_default_graph_watch_manifest_file(root_path)))
    table.add_row("Lock file", str(_default_graph_watch_lock_file(root_path)))
    table.add_row("Log file", str(resolved_pid_file.with_suffix(".log")))
    if manifest and isinstance(manifest.get("owner"), str):
        table.add_row("Owner", str(manifest["owner"]))
    if state.pid is not None:
        table.add_row("PID", str(state.pid))
    if manifest and isinstance(manifest.get("last_refresh"), dict):
        last_refresh = manifest["last_refresh"]
        completed_at = last_refresh.get("completed_at")
        if isinstance(completed_at, (int, float)):
            completed_at_value = datetime.fromtimestamp(completed_at).isoformat(
                sep=" ", timespec="seconds"
            )
        else:
            completed_at_value = "unknown"
        table.add_row("Last refresh", completed_at_value)
        table.add_row(
            "Refresh counts",
            (
                f"changed={last_refresh.get('changed', 0)}, "
                f"deleted={last_refresh.get('deleted', 0)}, "
                f"unchanged={last_refresh.get('unchanged', 0)}"
            ),
        )
        table.add_row("Refresh errors", str(last_refresh.get("errors", 0)))
        duration_seconds = last_refresh.get("duration_seconds")
        if isinstance(duration_seconds, (int, float)):
            table.add_row("Refresh duration", f"{duration_seconds:.2f}s")
        last_error = last_refresh.get("last_error")
        if isinstance(last_error, str) and last_error.strip():
            error_type = last_refresh.get("error_type")
            if isinstance(error_type, str) and error_type.strip():
                table.add_row("Last error", f"{error_type}: {last_error}")
            else:
                table.add_row("Last error", last_error)
        error_category = last_refresh.get("error_category")
        if isinstance(error_category, str) and error_category.strip():
            table.add_row("Error category", error_category)
        recoverable = last_refresh.get("recoverable")
        if isinstance(recoverable, bool):
            table.add_row("Recoverable", "yes" if recoverable else "no")
        retry_delay_seconds = last_refresh.get("retry_delay_seconds")
        if isinstance(retry_delay_seconds, (int, float)) and retry_delay_seconds > 0:
            table.add_row("Retry backoff", f"{retry_delay_seconds:.2f}s")
        operator_guidance = last_refresh.get("operator_guidance")
        if isinstance(operator_guidance, str) and operator_guidance.strip():
            table.add_row("Operator guidance", operator_guidance)
    console.print(table)


@graph_app.command("init-context")
def graph_init_context(
    path: str = typer.Option(".", "--path", "-p", help="Path to codebase root"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Task description for relevance"),
    max_symbols: int = typer.Option(
        50, "--symbols", "-s", help="Max symbols to include", min=10, max=200
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
):
    """Generate graph-enhanced init.md with symbol context."""
    cwd = os.path.abspath(path)

    start_time = time.time()
    success = run_sync(_init_context_async(cwd, task, max_symbols, force))
    elapsed = time.time() - start_time

    if success:
        console.print(f"[green]✓ Complete in {elapsed:.1f}s[/green]")
    else:
        raise typer.Exit(1)
