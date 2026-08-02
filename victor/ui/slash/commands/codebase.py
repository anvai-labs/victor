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

"""Codebase management slash commands: reindex, init."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Awaitable, Callable, cast

from rich.markdown import Markdown
from rich.panel import Panel

from victor.core.utils.capability_loader import load_codebase_analyzer_attr
from victor.ui.commands.init_content import (
    count_architecture_patterns,
    ensure_quality_baseline_section,
)
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command
from victor.config.settings import _resolve_graph_writer_mode

logger = logging.getLogger(__name__)


@register_command
class ReindexCommand(BaseSlashCommand):
    """Reindex codebase for semantic search."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="reindex",
            description="Reindex codebase for semantic search",
            usage="/reindex [--force] [--stats]",
            aliases=["index"],
            category="codebase",
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        import time
        from pathlib import Path

        force = self._has_flag(ctx, "--force", "-f")
        show_stats = self._has_flag(ctx, "--stats", "-s")

        try:
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol
            from victor.config.settings import get_project_paths, load_settings

            _factory = CapabilityRegistry.get_instance().get(CodebaseIndexFactoryProtocol)
            if _factory is None:
                raise ImportError("Codebase indexing not available")

            root = Path.cwd()
            settings = load_settings()
            paths = get_project_paths(root)
            graph_writer_mode = _resolve_graph_writer_mode(settings)
            graph_store_name = getattr(settings, "codebase_graph_store", "sqlite")
            graph_path = getattr(settings, "codebase_graph_path", None)

            embedding_config = {
                "vector_store": getattr(settings, "codebase_vector_store", "lancedb"),
                "embedding_model_type": getattr(
                    settings, "codebase_embedding_provider", "sentence-transformers"
                ),
                "embedding_model_name": getattr(
                    settings, "codebase_embedding_model", "BAAI/bge-small-en-v1.5"
                ),
                "persist_directory": str(paths.embeddings_dir),
            }

            index = _factory.create(
                root_path=str(root),
                use_embeddings=True,
                embedding_config=embedding_config,
                graph_writer_mode=str(graph_writer_mode),
                graph_store_name=graph_store_name,
                graph_path=Path(graph_path) if graph_path else None,
            )

            if show_stats:
                stats = index.get_stats()
                content = (
                    f"[bold]Index Statistics[/]\n\n"
                    f"[bold]Total Files:[/] {stats.get('total_files', 0)}\n"
                    f"[bold]Total Symbols:[/] {stats.get('total_symbols', 0)}\n"
                    f"[bold]Total Lines:[/] {stats.get('total_lines', 0)}\n"
                    f"[bold]Embeddings:[/] {'enabled' if stats.get('embeddings_enabled') else 'disabled'}\n"
                    f"[bold]Indexed:[/] {stats.get('is_indexed', False)}\n"
                )
                ctx.console.print(Panel(content, title="Codebase Index", border_style="cyan"))
                return

            ctx.console.print("[dim]Indexing codebase...[/]")
            start = time.time()

            if force:
                ctx.console.print("[dim]Force reindex: rebuilding from scratch[/]")
                await index.index_codebase()
            else:
                await index.ensure_indexed(auto_reindex=True)

            duration = time.time() - start
            stats = index.get_stats()

            ctx.console.print(
                Panel(
                    f"[green]Indexing complete![/]\n\n"
                    f"[bold]Files Indexed:[/] {stats.get('total_files', 0)}\n"
                    f"[bold]Symbols Found:[/] {stats.get('total_symbols', 0)}\n"
                    f"[bold]Time:[/] {duration:.1f}s",
                    title="Reindex Complete",
                    border_style="green",
                )
            )

        except ImportError as e:
            ctx.console.print(f"[yellow]Indexing not available:[/] {e}")
            ctx.console.print("[dim]Install with: pip install sentence-transformers lancedb[/]")
        except Exception as e:
            ctx.console.print(f"[red]Indexing failed:[/] {e}")
            logger.exception("Reindex error")


@register_command
class InitCommand(BaseSlashCommand):
    """Initialize or update .victor/init.md with codebase analysis."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="init",
            description="Initialize or update .victor/init.md with codebase analysis",
            usage="/init [--index] [--update] [--force] [--deep] [--symlinks]",
            category="codebase",
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        from victor.config.settings import get_project_paths

        force = self._has_flag(ctx, "--force", "-f")
        update = self._has_flag(ctx, "--update", "-u")
        deep = self._has_flag(ctx, "--deep", "-d", "--smart")
        use_index = self._has_flag(ctx, "--index", "-i")
        use_learn = self._has_flag(ctx, "--learn", "-L")
        symlinks = self._has_flag(ctx, "--symlinks", "-l")

        paths = get_project_paths()
        target_path = paths.project_context_file
        paths.project_victor_dir.mkdir(parents=True, exist_ok=True)

        existing_content = None
        if target_path.exists():
            existing_content = target_path.read_text(encoding="utf-8")

            if not force and not update:
                ctx.console.print(f"[yellow]{target_path.name} already exists[/]")
                ctx.console.print("")
                ctx.console.print("[bold]Options:[/]")
                ctx.console.print(
                    "  [cyan]/init --update[/]   Merge new analysis (preserves your edits)"
                )
                ctx.console.print("  [cyan]/init --force[/]    Overwrite completely")
                ctx.console.print(
                    "  [cyan]/init --learn[/]    Enhance with conversation history insights"
                )
                ctx.console.print("  [cyan]/init --deep[/]     LLM-powered analysis (any language)")
                ctx.console.print("  [cyan]/init --symlinks[/] Create CLAUDE.md and other aliases")

                preview = existing_content[:500]
                ctx.console.print(
                    Panel(
                        Markdown(preview + "\n\n..." if len(existing_content) > 500 else preview),
                        title=f"Current {target_path.name}",
                        border_style="dim",
                    )
                )
                return

        if use_learn:
            ctx.console.print("[dim]Analyzing codebase + learning from conversation history...[/]")
        elif use_index:
            ctx.console.print("[dim]Indexing codebase (multi-language symbol analysis)...[/]")
        elif deep:
            ctx.console.print("[dim]Analyzing codebase with LLM (deep mode)...[/]")
        elif update and existing_content:
            ctx.console.print("[dim]Updating codebase analysis (preserving user sections)...[/]")
        else:
            ctx.console.print("[dim]Analyzing codebase (quick mode)...[/]")

        try:
            generate_enhanced_init_md = cast(
                Callable[..., Awaitable[str]],
                load_codebase_analyzer_attr("generate_enhanced_init_md"),
            )
            generate_smart_victor_md = cast(
                Callable[..., str],
                load_codebase_analyzer_attr("generate_smart_victor_md"),
            )
            generate_victor_md_from_index = cast(
                Callable[..., Awaitable[str]],
                load_codebase_analyzer_attr("generate_victor_md_from_index"),
            )
        except ImportError:
            ctx.console.print("[red]Error: codebase_analyzer requires victor-coding vertical.[/]")
            return

        try:
            if deep or use_learn:
                # Pass agent for orchestrator reuse (logging, events, GEPA)
                new_content = await generate_enhanced_init_md(
                    use_llm=deep,
                    include_conversations=use_learn or deep,
                    agent=ctx.agent,
                )
            elif use_index:
                new_content = await generate_victor_md_from_index()
            else:
                new_content = generate_smart_victor_md()

            # Handle update mode
            if update and existing_content:
                content = self.merge_init_content(existing_content, new_content)
                ctx.console.print("[dim]  Merged with existing content[/]")
            else:
                content = new_content

            content = ensure_quality_baseline_section(content)

            # Write the file
            target_path.write_text(content, encoding="utf-8")
            ctx.console.print(f"[green]Created {target_path}[/]")

            # Show what was detected
            lines = content.split("\n")
            component_count = content.count("| `")
            pattern_count = count_architecture_patterns(content)

            ctx.console.print(
                f"[dim]  Lines: {len(lines)}, Components: {component_count}, Patterns: {pattern_count}[/]"
            )

            # Create symlinks if requested
            if symlinks:
                self._create_symlinks(target_path, ctx)

        except ImportError as e:
            ctx.console.print(f"[red]Missing dependency:[/] {e}")
        except Exception as e:
            ctx.console.print(f"[red]Analysis failed:[/] {e}")
            logger.exception("Init error")

    def merge_init_content(self, existing: str, new: str) -> str:
        """Merge new analysis with existing content, preserving user sections.

        This is a public method for use by CLI commands that also need merging
        functionality.
        """
        # Simple merge: keep user-added sections, update auto-generated ones
        # Look for markers like "<!-- AUTO-GENERATED -->" or "## Project Overview"

        # For now, a simple strategy: append new sections that don't exist
        existing_sections = set()
        for line in existing.split("\n"):
            if line.startswith("## "):
                existing_sections.add(line.strip())

        new_lines = []
        current_section = None
        skip_section = False

        for line in new.split("\n"):
            if line.startswith("## "):
                current_section = line.strip()
                skip_section = current_section in existing_sections
                if skip_section:
                    continue

            if not skip_section:
                new_lines.append(line)

        if new_lines:
            return existing + "\n\n" + "\n".join(new_lines)
        return existing

    def _create_symlinks(self, target: Path, ctx: CommandContext) -> None:
        """Create symlinks for other tools (CLAUDE.md, etc.)."""
        aliases = ["CLAUDE.md", "CURSOR.md"]
        for alias in aliases:
            alias_path = target.parent.parent / alias
            if not alias_path.exists():
                try:
                    alias_path.symlink_to(target)
                    ctx.console.print(f"[dim]  Created symlink: {alias}[/]")
                except Exception:
                    # Fall back to copying
                    alias_path.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
                    ctx.console.print(f"[dim]  Created copy: {alias}[/]")
