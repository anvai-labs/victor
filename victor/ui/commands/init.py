import typer
import re
from dataclasses import dataclass
import inspect
from rich.console import Console
from rich.prompt import Confirm, Prompt
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Mapping, Optional, cast
import yaml

from victor.config.settings import (
    get_project_paths,
    VICTOR_CONTEXT_FILE,
    VICTOR_DIR_NAME,
)
from victor.core.async_utils import run_sync
from victor.core.database import get_database, get_project_database
from victor.core.utils.capability_loader import load_codebase_analyzer_attr
from victor.ui.commands.init_content import (
    ensure_architecture_evidence_section,
    count_architecture_patterns,
    ensure_architecture_patterns_section,
    ensure_quality_baseline_section,
)
from victor.tools.common import latest_mtime

init_app = typer.Typer(name="init", help="Initialize project context and configuration.")
console = Console()

INIT_CCG_LOCK_TIMEOUT_SECONDS = 5.0


@dataclass
class _InitProviderAgent:
    """Minimal provider-backed agent for init synthesis.

    This avoids constructing the full orchestration stack when CLI init only
    needs a live provider/model pair for one synthesis request.
    """

    provider: Any
    model: Optional[str]
    provider_name: str
    temperature: float = 0.7
    max_tokens: int = 4096

    async def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


async def _generate_init_content_async(
    *,
    mode: str,
    use_llm: bool = False,
    include_conversations: bool = False,
    on_progress: Optional[Callable] = None,
    force: bool = False,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    graph_context: Optional[dict] = None,
) -> str:
    """Unified async content generator for all init analysis modes.

    Args:
        mode: One of "enhanced" (LLM/learn), "index" (symbol store), "quick" (regex).
        use_llm: Whether to use LLM for deep analysis (enhanced mode).
        include_conversations: Include conversation history insights (enhanced mode).
        on_progress: Progress callback.
        force: Force reindex/regeneration.
        include_dirs: Directories to include in analysis.
        exclude_dirs: Directories to exclude from analysis.
        provider: Override LLM provider (e.g., "anthropic", "openai"). Enhanced mode only.
        model: Override LLM model (e.g., "claude-sonnet-4-20250514"). Enhanced mode only.
        graph_context: Optional graph statistics (CCG, patterns) for LLM synthesis.

    Returns:
        Generated init.md content string.
    """
    if mode == "enhanced":
        generate_enhanced_init_md = cast(
            Callable[..., Awaitable[str]],
            load_codebase_analyzer_attr("generate_enhanced_init_md"),
        )

        # Build agent with provider/model override if specified
        agent = None
        if provider and use_llm:
            agent = await _create_init_agent(provider, model)

        # Try to pass graph_context if the function supports it
        # (victor-coding package may not have this parameter yet)
        try:
            sig = inspect.signature(generate_enhanced_init_md)
            if "graph_context" in sig.parameters:
                return await generate_enhanced_init_md(
                    use_llm=use_llm,
                    include_conversations=include_conversations,
                    on_progress=on_progress,
                    force=force,
                    include_dirs=include_dirs,
                    exclude_dirs=exclude_dirs,
                    agent=agent,
                    graph_context=graph_context,
                )
            else:
                # victor-coding doesn't support graph_context yet, skip it
                return await generate_enhanced_init_md(
                    use_llm=use_llm,
                    include_conversations=include_conversations,
                    on_progress=on_progress,
                    force=force,
                    include_dirs=include_dirs,
                    exclude_dirs=exclude_dirs,
                    agent=agent,
                )
        except TypeError:
            # Fallback if signature check fails
            return await generate_enhanced_init_md(
                use_llm=use_llm,
                include_conversations=include_conversations,
                on_progress=on_progress,
                force=force,
                include_dirs=include_dirs,
                exclude_dirs=exclude_dirs,
                agent=agent,
            )
        finally:
            if agent is not None and hasattr(agent, "close"):
                close_result = agent.close()
                if inspect.isawaitable(close_result):
                    await close_result
    elif mode == "index":
        generate_victor_md_from_index = cast(
            Callable[..., Awaitable[str]],
            load_codebase_analyzer_attr("generate_victor_md_from_index"),
        )
        return await generate_victor_md_from_index(
            force=force,
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs,
        )
    else:
        # "quick" mode — sync regex-based, no LLM
        generate_smart_victor_md = cast(
            Callable[..., str],
            load_codebase_analyzer_attr("generate_smart_victor_md"),
        )
        return generate_smart_victor_md(
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs,
        )


def _generate_init_content(
    *,
    mode: str,
    use_llm: bool = False,
    include_conversations: bool = False,
    on_progress: Optional[Callable] = None,
    force: bool = False,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    graph_context: Optional[dict] = None,
) -> str:
    """Sync wrapper for _generate_init_content_async."""
    if mode == "quick":
        # Quick mode is sync — no need for run_sync
        generate_smart_victor_md = cast(
            Callable[..., str],
            load_codebase_analyzer_attr("generate_smart_victor_md"),
        )
        return generate_smart_victor_md(
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs,
        )
    return run_sync(
        _generate_init_content_async(
            mode=mode,
            use_llm=use_llm,
            include_conversations=include_conversations,
            on_progress=on_progress,
            force=force,
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs,
            provider=provider,
            model=model,
            graph_context=graph_context,
        )
    )


async def _create_init_agent(provider: str, model: Optional[str] = None) -> Any:
    """Create a lightweight provider-backed agent for init synthesis.

    Supports both profile names (e.g., "zai-coding") and bare provider names
    (e.g., "ollama"), but intentionally avoids constructing the full
    Agent/Orchestrator stack. Init synthesis only needs an initialized
    provider/model pair, and the heavier runtime path eagerly opens
    ConversationStore, IntentClassifier, and embedding resources that are not
    required here.

    Args:
        provider: Profile name (e.g., "zai-coding") or bare provider name.
        model: Optional model override.

    Returns:
        Agent instance with an initialized provider.

    Returns:
        Minimal object exposing ``provider``, ``provider_name``, ``model``, and
        ``close()`` for InitSynthesizer reuse.
    """
    from victor.providers.registry import ProviderRegistry

    from victor.framework.init_synthesizer import InitSynthesizer

    bootstrap = InitSynthesizer._resolve_provider_bootstrap(
        provider,
        model,
    )

    provider_instance = ProviderRegistry.create(
        bootstrap.provider_name,
        **bootstrap.provider_init_kwargs,
    )
    if not provider_instance:
        raise RuntimeError(f"Could not create provider {bootstrap.provider_name}")

    return _InitProviderAgent(
        provider=provider_instance,
        model=bootstrap.request_model,
        provider_name=bootstrap.provider_name,
        temperature=bootstrap.temperature,
        max_tokens=bootstrap.max_tokens,
    )


def _run_agentic_synthesis(
    *,
    provider: Optional[str],
    model: Optional[str],
    graph_context: Optional[dict],
    console_: Any,
) -> str:
    """Run tool-driven init.md synthesis via InitSynthesizer.

    Routes through ``synthesize_with_tools`` (which uses the full agentic
    loop) instead of the default one-shot ``synthesize``. Passes the
    pre-gathered ``graph_context`` so the agent's tool calls are anchored
    on measured scale facts and graph stats — without that anchor the
    model wastes early iterations re-deriving counts we already have.

    Slower than one-shot (1-3 min typical, depends on how many tool calls
    the model makes) and more expensive, but produces richer documents
    for under-documented repos. Returns the generated markdown, or an
    empty string on failure (caller treats as "use base content").

    Profile resolution: the CLI's ``-p`` flag can be either a profile name
    from ``~/.victor/profiles.yaml`` (``zai-coding``) or a bare provider
    name (``ollama``). We hand the raw value to
    ``_resolve_provider_bootstrap`` — the same helper the legacy
    one-shot path uses — so the agentic path gets the canonical provider
    name + correct model + profile-specific overrides (coding_plan,
    base_url, etc.) instead of treating ``zai-coding`` as a literal
    provider name (which caused the agent to fall through to the global
    default model and crash with "Unknown Model").
    """
    import asyncio
    from victor.framework.agent import Agent
    from victor.framework.init_synthesizer import InitSynthesizer

    def _is_known_profile_name(name: str) -> bool:
        """True if `name` matches a profile from ~/.victor/profiles.yaml.

        AgentFactory treats `profile=<bare-provider-name>` (e.g. ``ollama``)
        differently from `profile=<real-profile-name>` (e.g. ``zai-coding``).
        Only forward the kwarg when it actually names a profile so we don't
        accidentally crash the factory for bare-provider invocations.
        """
        try:
            from victor.config.settings import load_settings

            settings = load_settings()
            profiles = settings.load_profiles() if hasattr(settings, "load_profiles") else {}
            return isinstance(profiles, dict) and name in profiles
        except Exception:
            return False

    async def _go() -> str:
        bootstrap = InitSynthesizer._resolve_provider_bootstrap(provider, model)
        # Build a real Agent (not the lightweight _InitProviderAgent the
        # one-shot path uses) — synthesize_with_tools needs agent.chat()
        # to drive the agentic loop end-to-end.
        #
        # Pass `profile=` (the original CLI -p value) when it names a real
        # profile so AgentFactory re-applies profile extras like
        # `coding_plan: true` / `base_url`. Forwarding only `provider` +
        # `model` silently drops those extras and the agent ends up talking
        # to the default endpoint (e.g. zai's paas URL instead of the
        # coding-plan URL), which lands in a different — and far stricter —
        # rate-limit pool than `victor chat -p zai-coding` uses.
        agent_kwargs: dict[str, Any] = {
            "provider": bootstrap.provider_name,
            "model": bootstrap.request_model,
            "vertical": "coding",
        }
        if provider and _is_known_profile_name(provider):
            agent_kwargs["profile"] = provider
        if bootstrap.temperature is not None:
            agent_kwargs["temperature"] = bootstrap.temperature
        if bootstrap.max_tokens is not None:
            agent_kwargs["max_tokens"] = bootstrap.max_tokens
        agent = await Agent.create(**agent_kwargs)

        # Init synthesis must produce its deliverable as the assistant's
        # response, not by writing files via tools. The coding vertical's
        # system prompt heavily biases the model toward "use write/edit
        # for new files", which on this task makes the model call
        # `write path=.victor/init.md content=...` and emit
        # `VICTOR_FILE_DONE::` as its final text — at which point init.py
        # writes the *response* (75 chars of marker) over the agent's good
        # file. Disabling write-class tools on the registry removes them
        # from both the function-call schemas and the prompt's live tool
        # listing, so the model has no escape hatch and must return the
        # init.md content as its message.
        try:
            _SYNTHESIS_DISABLED_TOOLS = (
                "write",
                "edit",
                "rename",
                "extract",
                "shell",
                "test",
                "docker",
            )
            _orchestrator = getattr(agent, "_orchestrator", None) or agent
            _tool_registry = getattr(_orchestrator, "tools", None)
            if _tool_registry is not None and hasattr(_tool_registry, "disable_tool"):
                for _t in _SYNTHESIS_DISABLED_TOOLS:
                    try:
                        _tool_registry.disable_tool(_t)
                    except Exception:
                        pass
        except Exception:
            pass

        synthesizer = InitSynthesizer()
        try:
            return await synthesizer.synthesize_with_tools(
                agent=agent,
                graph_context=graph_context,
            )
        finally:
            close = getattr(agent, "close", None)
            if callable(close):
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass

    console_.print(
        "[dim]  Agentic synthesis: agent will call tools to verify facts "
        "before writing (this takes 1-3 min)…[/]"
    )
    try:
        return asyncio.run(_go())
    except Exception as exc:
        console_.print(f"[yellow]![/] Agentic synthesis failed: {exc}")
        return ""


def _gather_project_scale_facts(root: Path) -> dict[str, Any]:
    """Cheap, accurate project-scale numbers for the LLM synthesis prompt.

    Without these, the model invents plausible-sounding counts because the
    prompt only carried graph node/edge totals — observed in proximaDB
    where init.md claimed "6 workspace crates" against an actual 47 and
    "1.6M LOC across 3,642 files" with no source for either figure.

    Returns a dict with:
      - ``files_by_language``: {language → file count} for indexable code
      - ``loc_by_language``: {language → total non-blank lines} for the same
      - ``total_files`` / ``total_loc``: summed across known languages
      - ``cargo_crate_count``: Cargo.toml files under any ``crates/`` dir
      - ``python_packages``: count of ``__init__.py`` files (rough package
        count, Python only)
      - ``js_packages``: count of ``package.json`` files outside node_modules

    Uses ``git ls-files`` when in a repo (respects .gitignore) and falls
    back to filesystem walking with conservative excludes otherwise.
    """
    import subprocess

    # Extension → language map. Kept narrow on purpose: only languages the
    # graph indexer can actually reason about. Listing every extension here
    # would dilute the "totals" the LLM uses to anchor its claims.
    ext_to_lang = {
        ".py": "python",
        ".rs": "rust",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".swift": "swift",
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".cs": "c_sharp",
        ".rb": "ruby",
        ".php": "php",
    }

    # Prefer git for honest exclude semantics; fall back to a manual walk.
    files: list[str] = []
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            files = [line for line in result.stdout.splitlines() if line]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    if not files:
        # No git, or empty repo — fall back to a bounded walk that skips
        # the usual suspects so we don't traverse virtualenvs and caches.
        skip_dirs = {
            ".git",
            "node_modules",
            "target",
            "dist",
            "build",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            "site-packages",
        }
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            try:
                files.append(str(path.relative_to(root)))
            except ValueError:
                continue

    files_by_lang: dict[str, int] = {}
    loc_by_lang: dict[str, int] = {}
    cargo_crates = 0
    python_packages = 0
    js_packages = 0

    for rel in files:
        path = root / rel
        name = Path(rel).name
        # Project-marker counters — independent of language buckets.
        if name == "Cargo.toml" and "crates/" in rel:
            cargo_crates += 1
        elif name == "__init__.py":
            python_packages += 1
        elif name == "package.json" and "node_modules/" not in rel:
            js_packages += 1

        ext = Path(rel).suffix.lower()
        lang = ext_to_lang.get(ext)
        if not lang:
            continue
        files_by_lang[lang] = files_by_lang.get(lang, 0) + 1
        # LOC = non-blank lines. Cheaper than a real lexer and good enough
        # to ground the LLM. Silently skip unreadable files.
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                loc = sum(1 for line in f if line.strip())
            loc_by_lang[lang] = loc_by_lang.get(lang, 0) + loc
        except OSError:
            continue

    return {
        "files_by_language": files_by_lang,
        "loc_by_language": loc_by_lang,
        "total_files": sum(files_by_lang.values()),
        "total_loc": sum(loc_by_lang.values()),
        "cargo_crate_count": cargo_crates,
        "python_packages": python_packages,
        "js_packages": js_packages,
    }


def _format_project_scale_for_prompt(facts: dict[str, Any]) -> str:
    """Render project-scale facts as a hard-anchor section for the LLM.

    The wording is deliberate: it tells the model these are *measured*
    counts and forbids invention. Without an explicit "do not invent"
    clause the model has historically merged real numbers with plausible
    fictions in the same sentence.
    """
    if not facts or not facts.get("total_files"):
        return ""

    lang_lines = []
    by_lang = facts.get("files_by_language", {})
    loc = facts.get("loc_by_language", {})
    for lang, count in sorted(by_lang.items(), key=lambda kv: -kv[1]):
        lang_lines.append(f"  - {lang}: {count:,} files, {loc.get(lang, 0):,} LOC")

    parts = [
        "## Measured Project Scale (DO NOT INVENT — use these numbers verbatim)",
        "",
        f"- **Total source files**: {facts['total_files']:,}",
        f"- **Total LOC** (non-blank, indexable languages only): {facts['total_loc']:,}",
        "- **Files by language**:",
        *lang_lines,
    ]
    if facts.get("cargo_crate_count"):
        parts.append(f"- **Cargo workspace crates**: {facts['cargo_crate_count']}")
    if facts.get("python_packages"):
        parts.append(f"- **Python packages** (``__init__.py`` count): {facts['python_packages']}")
    if facts.get("js_packages"):
        parts.append(f"- **JS/TS sub-packages** (``package.json``): {facts['js_packages']}")
    parts.append("")
    parts.append(
        "When you reference codebase size, crate count, file count, or "
        "language breakdown anywhere in the document, use these exact "
        "numbers. Do not estimate, round, or substitute language names."
    )
    return "\n".join(parts) + "\n"


def _query_edge_type_breakdown(root: Path) -> dict[str, int]:
    """Return ``{edge_type: count}`` for the project graph, sorted desc.

    Used to give users a single concrete picture of what shipped to the DB
    instead of the prior emit-count vs row-count split which routinely
    misled users into thinking work was duplicated. Returns an empty dict
    when the DB is missing or unreadable so the caller can no-op cleanly.
    """
    import sqlite3

    db_path = root / ".victor" / "project.db"
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT type, COUNT(*) FROM graph_edge WHERE type IS NOT NULL "
                "GROUP BY type ORDER BY 2 DESC"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return {}
    return dict(rows)


def _report_ccg_coverage(root: Path, edge_breakdown: dict[str, int], console_) -> None:
    """Surface CCG-language coverage so silent-zero is no longer mistaken
    for success.

    The user asked for ``--ccg`` but if the project's primary languages
    aren't covered by the registered CCG builder, no CFG_/CDG_/DDG_ edges
    materialize. Previously the CLI still printed "CCG index updated"
    which made it look like everything worked. Now we count statement-
    level edges directly from what's in the DB and call it out when zero
    on a CCG run.
    """
    ccg_edge_count = sum(
        c
        for t, c in edge_breakdown.items()
        if t and (t.startswith("CFG_") or t.startswith("CDG") or t.startswith("DDG_"))
    )
    if ccg_edge_count > 0:
        console_.print(
            f"[dim]    → CCG statement-level edges: {ccg_edge_count:,} " f"(CFG/CDG/DDG)[/]"
        )
        return

    # Zero CCG edges with --ccg requested means the registered builder
    # didn't cover any of this project's languages. List what *is* covered
    # so the user can see the gap immediately.
    try:
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import CCGBuilderProtocol

        builder = CapabilityRegistry.get_instance().get(CCGBuilderProtocol)
        supported = []
        if builder is not None and hasattr(builder, "supported_languages"):
            try:
                attr = builder.supported_languages
                supported = list(attr() if callable(attr) else attr)
            except Exception:
                supported = []
        supported_str = ", ".join(sorted(supported)) if supported else "(unknown)"
        console_.print(
            "[yellow]    → CCG produced 0 statement-level edges. "
            f"Builder supports: {supported_str}.[/]"
        )
    except Exception:
        console_.print(
            "[yellow]    → CCG produced 0 statement-level edges (builder not registered).[/]"
        )


def _gather_graph_context(root: Path) -> Optional[dict]:
    """Gather graph statistics for synthesis prompt using COUNT SQL queries.

    Uses GROUP BY aggregation directly on project.db — never loads Python objects
    for individual edges, so it stays fast even with millions of edges.
    """
    import sqlite3

    graph_db_path = root / ".victor" / "project.db"
    if not graph_db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(graph_db_path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM graph_node").fetchone()
            if not row or row[0] == 0:
                return None
            total_nodes: int = row[0]

            total_edges: int = conn.execute("SELECT COUNT(*) FROM graph_edge").fetchone()[0]

            # Edge type distribution via GROUP BY — one table scan, no Python objects
            edge_type_counts: dict = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT type, COUNT(*) FROM graph_edge GROUP BY type"
                ).fetchall()
                if r[0]
            }
        finally:
            conn.close()

        cfg_edges = sum(v for k, v in edge_type_counts.items() if k.startswith("CFG_"))
        cdg_edges = sum(v for k, v in edge_type_counts.items() if k.startswith("CDG"))
        ddg_edges = sum(v for k, v in edge_type_counts.items() if k.startswith("DDG_"))
        ccg_coverage = cfg_edges + cdg_edges + ddg_edges

        # IMPLEMENTS comes from the language handlers — real declared conformance.
        # DECORATES/REGISTERS used to be counted here too, but they only ever
        # existed as synthetic edges the enrichment pass fabricated, and that pass
        # has been removed: it derived IMPLEMENTS from INHERITS plus a "*Protocol"
        # name suffix (adding no information the graph did not already hold), and
        # gated DECORATES/REGISTERS on two hardcoded victor/tools/… node ids, so
        # both were always 0 for any other repository.
        protocol_edges = edge_type_counts.get("IMPLEMENTS", 0)
        inheritance_edges = edge_type_counts.get("INHERITS", 0)
        calls_edges = edge_type_counts.get("CALLS", 0)

        return {
            "project_path": str(root),
            "has_ccg": ccg_coverage > 0,
            "has_graph": total_nodes > 0,
            "stats": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "ccg_edges": ccg_coverage,
                "cfg_edges": cfg_edges,
                "cdg_edges": cdg_edges,
                "ddg_edges": ddg_edges,
            },
            "patterns": {
                "protocol": protocol_edges,
                "inheritance": inheritance_edges,
                "calls": calls_edges,
            },
            "complexity": {
                "ccg_ratio": ccg_coverage / total_edges if total_edges > 0 else 0,
                "avg_branching": total_edges / total_nodes if total_nodes > 0 else 0,
            },
        }
    except Exception as exc:
        console.print(f"[dim]  Failed to gather graph context: {exc}[/]")
        return None


def _format_graph_context_for_prompt(graph_context: dict) -> str:
    """Format graph context for LLM synthesis prompt.

    Uses generic pattern terminology that applies to any project:
    - "Decorator pattern" not "@tool decorator"
    - "Registry pattern" not "tool registry"
    - "Protocol/Interface" not "Victor protocol"
    """
    if not graph_context:
        return ""

    stats = graph_context.get("stats", {})
    patterns = graph_context.get("patterns", {})
    complexity = graph_context.get("complexity", {})

    sections = []

    # CCG section (universal - applies to any codebase)
    if graph_context.get("has_ccg", False):
        ccg_ratio = complexity.get("ccg_ratio", 0) * 100
        sections.append(f"""
## Code Structure Analysis

This project has been analyzed with Code Context Graph (CCG) providing:
- **Control Flow Graph (CFG)**: {stats.get('cfg_edges', 0)} edges showing execution paths
- **Control Dependence Graph (CDG)**: {stats.get('cdg_edges', 0)} edges showing decision dependencies
- **Data Dependence Graph (DDG)**: {stats.get('ddg_edges', 0)} edges showing data flow

CCG Coverage: {ccg_ratio:.1f}% of graph edges have statement-level granularity.

When analyzing, leverage these to:
- Identify control flow complexity hotspots (deep nesting, complex branching)
- Trace data dependencies and side effects
- Understand decision points and their impact
""")

    # Generic design patterns (not Victor-specific)
    pattern_sections = []

    if patterns.get("decorator", 0) > 0:
        pattern_sections.append(f"""
- **Decorator Pattern**: {patterns['decorator']} decorated symbols detected.
  Analyze how decorators modify behavior and cross-cutting concerns.
""")

    if patterns.get("protocol", 0) > 0:
        pattern_sections.append(f"""
- **Protocol/Interface Pattern**: {patterns['protocol']} implementation relationships.
  Map abstraction hierarchies and contract implementations.
""")

    if patterns.get("registry", 0) > 0:
        pattern_sections.append(f"""
- **Registry/Plugin Pattern**: {patterns['registry']} registration relationships.
  Identify extensibility points and plugin architecture.
""")

    if patterns.get("inheritance", 0) > 0:
        pattern_sections.append(f"""
- **Inheritance Pattern**: {patterns['inheritance']} inheritance relationships.
  Analyze class hierarchies and method overrides.
""")

    if pattern_sections:
        sections.append(f"""
## Design Pattern Analysis

The following design patterns have been detected in the codebase:
{''.join(pattern_sections)}

Use these patterns to:
- Understand the architectural approach
- Identify extensibility mechanisms
- Map abstraction layers and contracts
""")

    return "\n".join(sections)


def _ensure_profile_preset(
    profiles_file: Path,
    name: str,
    description: str,
    provider: str = "ollama",
    model: str = "qwen2.5-coder:7b",
) -> Optional[bool]:
    """Add a profile preset if missing. True=added, False=exists, None=error."""
    data: dict = {}
    if profiles_file.exists():
        try:
            data = yaml.safe_load(profiles_file.read_text(encoding="utf-8")) or {}
        except Exception as e:
            console.print(f"[red]Error loading profiles:[/] {e}")
            return None

    profiles = data.get("profiles") or {}
    if name in profiles:
        return False

    profiles[name] = {
        "provider": provider,
        "model": model,
        "temperature": 0.7,
        "max_tokens": 4096,
        "description": description,
    }
    data["profiles"] = profiles

    providers = data.get("providers") or {}
    if provider == "ollama":
        ollama_config = providers.get("ollama") or {}
        if "base_url" not in ollama_config:
            ollama_config["base_url"] = "http://localhost:11434"
        providers["ollama"] = ollama_config
    data["providers"] = providers

    try:
        profiles_file.parent.mkdir(parents=True, exist_ok=True)
        profiles_file.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
    except Exception as e:
        console.print(f"[red]Error saving profiles:[/] {e}")
        return None

    return True


def _ensure_airgapped_env(env_path: Path) -> Optional[bool]:
    """Ensure AIRGAPPED_MODE=true in .env. True=added, False=exists, None=error."""
    content = ""
    if env_path.exists():
        try:
            content = env_path.read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error reading {env_path}:[/] {e}")
            return None
        if re.search(r"^\s*AIRGAPPED_MODE\s*=", content, flags=re.MULTILINE):
            return False

    content = content.rstrip("\n")
    if content:
        content += "\n"
    content += "AIRGAPPED_MODE=true\n"

    try:
        env_path.write_text(content, encoding="utf-8")
    except Exception as e:
        console.print(f"[red]Error saving {env_path}:[/] {e}")
        return None

    return True


@init_app.callback(invoke_without_command=True)
def init(
    ctx: typer.Context,
    update: bool = typer.Option(
        False, "--update", "-u", help="Update existing init.md preserving user edits"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing init.md completely"
    ),
    learn: bool = typer.Option(
        True,
        "--learn/--no-learn",
        "-L",
        help="Include conversation history insights (default: on)",
    ),
    index: bool = typer.Option(False, "--index", "-i", help="Build the symbol store only (no LLM)"),
    embeddings: bool = typer.Option(
        False,
        "--embeddings/--no-embeddings",
        help=(
            "Generate graph embeddings so semantic code search works " "(slower init; default: off)"
        ),
    ),
    deep: bool = typer.Option(
        True, "--deep/--no-deep", "-d", help="Use LLM for deep analysis (default: on)"
    ),
    quick: bool = typer.Option(
        False, "--quick", "-q", help="Fast regex-only analysis (no LLM, no indexing)"
    ),
    ccg: bool = typer.Option(
        True,
        "--ccg/--no-ccg",
        help="Enable Code Context Graph for statement-level analysis (default: on)",
    ),
    symlinks: bool = typer.Option(
        False, "--symlinks", "-l", help="Create CLAUDE.md and other tool aliases"
    ),
    config_only: bool = typer.Option(
        False, "--config", "-c", help="Only setup global config, skip project analysis"
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        "-I",
        help="Use interactive wizard for scoping",
    ),
    local: bool = typer.Option(False, "--local", help="Add a local profile preset (Ollama)"),
    airgapped: bool = typer.Option(
        False, "--airgapped", help="Enable air-gapped mode for this repo"
    ),
    wizard: bool = typer.Option(
        False,
        "--wizard",
        "-w",
        help="Run interactive setup wizard for first-time users",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Override LLM provider for deep analysis (e.g., anthropic, openai, ollama).",
        case_sensitive=False,
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override LLM model for deep analysis (e.g., claude-sonnet-4-20250514).",
    ),
    agentic: bool = typer.Option(
        False,
        "--agentic/--no-agentic",
        help=(
            "Use tool-driven synthesis: the LLM calls overview/ls/read/graph/search "
            "tools to explore the repo before writing init.md. Slower (1-3 min) and "
            "more expensive than the default one-shot synthesis, but produces a "
            "richer document for under-documented repos. Requires --deep."
        ),
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Log level: DEBUG, INFO, WARNING, ERROR (default: WARNING).",
        case_sensitive=False,
    ),
) -> None:
    """Initialize project context and configuration."""
    from victor.ui.commands.utils import setup_logging

    if log_level is not None:
        log_level = log_level.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            console.print(
                f"[red]Invalid log level '{log_level}'. Valid: {', '.join(valid_levels)}[/]"
            )
            raise typer.Exit(1)
        if log_level == "WARN":
            log_level = "WARNING"

    setup_logging(command="init", cli_log_level=log_level)

    # If wizard mode is requested, run the onboarding wizard instead
    if wizard:
        from victor.ui.commands.onboarding import run_onboarding

        exit_code = run_onboarding()
        raise typer.Exit(exit_code)
    if ctx.invoked_subcommand is None:
        paths = get_project_paths()

        # Step 1: Global config setup
        config_dir = paths.global_victor_dir
        config_dir.mkdir(parents=True, exist_ok=True)

        profiles_file = config_dir / "profiles.yaml"
        created_profiles = False
        if not profiles_file.exists():
            console.print(f"[dim]Creating default configuration at {profiles_file}[/]")

            # Create a basic default profile
            default_config = """profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

providers:
  ollama:
    base_url: http://localhost:11434
"""
            profiles_file.write_text(default_config)
            console.print(f"[green]✓[/] Global config created at {config_dir}")
            created_profiles = True
        else:
            console.print(f"[dim]Global config exists at {config_dir}[/]")

        if local:
            add_local = True
            if not created_profiles and interactive:
                add_local = Confirm.ask(
                    "[cyan]Add a local profile preset to profiles.yaml?[/]",
                    default=True,
                )
            if add_local:
                local_profile_result = _ensure_profile_preset(
                    profiles_file,
                    name="local",
                    description="Local Ollama profile",
                )
                if local_profile_result is True:
                    console.print("[green]✓[/] Added local profile preset")
                    console.print("  [dim]Use with:[/] victor chat --profile local")
                    console.print("  [dim]Set default:[/] victor profiles set-default local")
                elif local_profile_result is False:
                    console.print("[dim]Local profile preset already present[/]")
                else:
                    console.print("[yellow]![/] Failed to update profiles.yaml")
            else:
                console.print("[dim]Skipped local profile preset[/]")

        if airgapped:
            airgapped_profile_result = _ensure_profile_preset(
                profiles_file,
                name="airgapped",
                description="Air-gapped local profile",
            )
            if airgapped_profile_result is True:
                console.print("[green]✓[/] Added air-gapped profile preset")
                console.print("  [dim]Use with:[/] victor chat --profile airgapped")
            elif airgapped_profile_result is False:
                console.print("[dim]Air-gapped profile preset already present[/]")
            else:
                console.print("[yellow]![/] Failed to update profiles.yaml")

            env_path = Path.cwd() / ".env"
            env_result = _ensure_airgapped_env(env_path)
            if env_result is True:
                console.print("[green]✓[/] Enabled air-gapped mode in .env")
            elif env_result is False:
                console.print("[dim]AIRGAPPED_MODE already set in .env[/]")
            else:
                console.print(f"[yellow]![/] Failed to update {env_path}")

        # Step 1.5: Initialize databases
        console.print("[dim]Initializing databases...[/]")
        try:
            # Initialize global database (user-level: RL, sessions, signatures)
            global_db = get_database()
            console.print(f"  [green]✓[/] Global database: {global_db.db_path}")

            # Initialize project database (repo-level: graph, conversations, entities)
            project_db = get_project_database()
            console.print(f"  [green]✓[/] Project database: {project_db.db_path}")
        except Exception as e:
            console.print(f"  [yellow]![/] Database initialization warning: {e}")

        if config_only:
            console.print(f"\nEdit {profiles_file} to customize profiles")
            return

        # Step 2: Project codebase analysis
        console.print("")
        target_path = paths.project_context_file
        paths.project_victor_dir.mkdir(parents=True, exist_ok=True)

        existing_content = None
        if target_path.exists():
            existing_content = target_path.read_text(encoding="utf-8")

            if not force and not update and interactive:
                if not Confirm.ask(
                    f"[yellow]{VICTOR_CONTEXT_FILE} already exists. Do you want to overwrite it with new analysis options?[/]"
                ):  # default=False
                    console.print("[dim]Aborted.[/dim]")
                    return
            elif not force and not update:
                console.print(f"[yellow]{VICTOR_CONTEXT_FILE} already exists at {target_path}[/]")
                console.print("")
                console.print("[bold]Options:[/]")
                console.print(
                    "  [cyan]victor init --update[/]   Merge new analysis (preserves your edits)"
                )
                console.print("  [cyan]victor init --force[/]    Overwrite completely")
                console.print(
                    "  [cyan]victor init --learn[/]    Enhance with conversation history insights"
                )
                console.print("  [cyan]victor init --index[/]    Multi-language symbol indexing")
                console.print("  [cyan]victor init --deep[/]     LLM-powered deep analysis")
                console.print(
                    "  [cyan]victor init --no-ccg[/]   Disable Code Context Graph (faster)"
                )
                console.print(
                    "  [cyan]victor init --ccg[/]      Enable Code Context Graph (default)"
                )
                return

        # Handle --quick flag (overrides everything)
        if quick:
            deep = False
            learn = False
            index = False
            ccg = False

        include_dirs: List[str] = []
        exclude_dirs: List[str] = []

        # When --force is used, skip interactive prompts and use "." (current dir)
        if force and interactive:
            interactive = False
            include_dirs = [str(Path.cwd())]
            console.print(
                "[dim]Using current directory (.) for analysis (--force skips prompts)[/]"
            )

        if interactive and not quick:
            console.print("\n[bold]Interactive Project Scoping[/]")

            # Get current working directory (cross-platform)
            cwd = Path.cwd()
            all_dirs = [
                d.name for d in cwd.iterdir() if d.is_dir() and not d.name.startswith(".")
            ]  # noqa
            common_src_dirs = ["src", "app", "lib", "victor", "server", "client"]
            suggested_src = [d for d in common_src_dirs if d in all_dirs]

            if not suggested_src and all_dirs:
                project_dir = cwd.name
                if project_dir in all_dirs:
                    suggested_src.append(project_dir)
                # If no common src dirs found, default to "." (entire project)
                # The user can narrow down if needed

            # Default to "." if no suggestions - scan entire project (cross-platform)
            # "." means current working directory on all platforms
            # Default to scanning the whole project (.) even if we detect likely src dirs;
            # users can narrow scope by entering a comma-separated list.
            default_include = "."

            include_str = Prompt.ask(
                "[cyan]Enter comma-separated source directories to include[/]",
                default=default_include,
            )
            # Normalize "." to current working directory path for downstream processing
            include_dirs = []
            for d in include_str.split(","):
                d = d.strip()
                if d == ".":
                    # Use absolute path for current directory (cross-platform)
                    include_dirs.append(str(cwd))
                else:
                    include_dirs.append(d)

            default_exclude = [
                "__pycache__",
                ".git",
                ".pytest_cache",
                "venv",
                "env",
                ".venv",
                "node_modules",
                ".tox",
                "build",
                "dist",
                "egg-info",
                "htmlcov",
                "htmlcov_lang",
                ".mypy_cache",
                ".ruff_cache",
                # IDE/editor artifacts
                ".vscode-test",
                ".idea",
                # Coverage
                "coverage",
                # Third party / vendor
                "vendor",
                "third_party",
            ]

            exclude_str = Prompt.ask(
                "[cyan]Enter comma-separated directories to exclude[/]",
                default=", ".join(default_exclude),
            )
            exclude_dirs = [d.strip() for d in exclude_str.split(",")]

        # Determine analysis mode
        if quick:
            mode = "quick"
        elif index:
            mode = "index"
        elif deep or learn or ccg:
            mode = "enhanced"
        else:
            mode = "enhanced"  # Default

        # Print status (include provider override if specified)
        provider_note = ""
        if provider and mode == "enhanced" and deep:
            provider_note = f" via {provider}"
            if model:
                provider_note += f"/{model}"

        ccg_note = " + CCG" if ccg and mode not in ("quick", "index") else ""

        if deep and learn:
            console.print(
                f"[dim]Comprehensive analysis: Index → Learn → LLM{provider_note}{ccg_note} (default)...[/]"
            )
        elif deep:
            console.print(
                f"[dim]Analysis: Index → LLM{provider_note}{ccg_note} (no conversation insights)...[/]"
            )
        elif learn:
            console.print(f"[dim]Analysis: Index → Learn{ccg_note} (no LLM)...[/]")
        elif ccg:
            console.print("[dim]Analysis: Index + CCG (no LLM, no conversation)...[/]")
        elif index:
            console.print("[dim]Analysis: Index only (no LLM, no conversation)...[/]")
        else:
            console.print("[dim]Quick regex analysis (no indexing)...[/]")

        try:

            from rich.status import Status as _Status

            _synthesis_status: Optional[_Status] = None

            def on_progress(stage: str, msg: str) -> None:
                if _synthesis_status is not None:
                    _synthesis_status.update(f"[dim]  {msg}[/]")
                else:
                    console.print(f"[dim]  {msg}[/]")

            project_root = Path.cwd()

            # Build graph data BEFORE LLM synthesis so the LLM can leverage it.
            # Order: incremental graph/CCG refresh → LLM synthesis

            # 1. CCG indexing (if enabled and not in quick/index mode)
            if ccg and mode not in ("quick", "index"):
                try:
                    from victor.core.graph_rag import (
                        GraphIndexingPipeline,
                        GraphIndexConfig,
                    )
                    from victor.storage.graph import create_graph_store

                    try:
                        from victor.core.graph_rag.indexing import (
                            run_indexing_with_lock,
                        )
                    except (ImportError, ModuleNotFoundError, AttributeError):
                        run_indexing_with_lock = None

                    ccg_mode = "rebuilding" if force else "updating incrementally"
                    console.print(f"[dim]  {ccg_mode.title()} Code Context Graph...[/]")
                    if force:
                        console.print(
                            "[dim]  → Clearing all existing graph data for clean rebuild[/]"
                        )
                        console.print(
                            "[dim]  → Using optimized bulk load (DELETE + INSERT vs UPSERT)[/]"
                        )
                    import asyncio
                    from rich.progress import (
                        BarColumn,
                        MofNCompleteColumn,
                        Progress,
                        SpinnerColumn,
                        TaskProgressColumn,
                        TextColumn,
                        TimeElapsedColumn,
                    )

                    _ccg_progress: Optional[Any] = None
                    _ccg_task_id: Optional[Any] = None
                    _project_root_str = str(project_root) + "/"

                    def _rel(filename: str) -> str:
                        """Return path relative to project root (strip leading prefix)."""
                        if filename.startswith(_project_root_str):
                            return filename[len(_project_root_str) :]
                        return filename

                    def _on_ccg_status(message: str) -> None:
                        if _ccg_progress is not None and _ccg_task_id is not None:
                            _ccg_progress.update(
                                _ccg_task_id,
                                description=f"[dim]  {message}[/]",
                            )

                    def _on_ccg_progress(done: int, total: int, filename: str) -> None:
                        if _ccg_progress is not None and _ccg_task_id is not None:
                            rel = _rel(filename) if filename else ""
                            # Batch-level update: description shows the last file in the batch
                            desc = f"[cyan]  {rel[-55:]}[/]" if rel else "[dim]  Writing batch…[/]"
                            _ccg_progress.update(
                                _ccg_task_id,
                                completed=done,
                                total=total or done or None,
                                description=desc,
                            )

                    async def _build_ccg_index():
                        # "auto" honors <project>/.victor/graph_backend and falls
                        # back to sqlite. Hardcoding "sqlite" here meant init wrote
                        # to SQLite while the read paths (graph_query_tool,
                        # graph_manager) already resolved through "auto" — a repo
                        # that set the marker got its queries routed to a store
                        # init had never populated.
                        graph_store = create_graph_store("auto", project_path=project_root)
                        config = GraphIndexConfig(
                            root_path=project_root,
                            enable_ccg=True,
                            # Off by default because generating vectors roughly
                            # triples init time, but no longer hardcoded: without
                            # this, `GraphIndexConfig.enable_embeddings` (which
                            # defaults to True) was silently overridden and NO
                            # backend ever produced graph vectors, so semantic
                            # seed->expand had nothing to search after a fresh init.
                            # Opt in with `victor init --embeddings`.
                            enable_embeddings=embeddings,
                            incremental=not force,  # force=True wipes clean, force=False is incremental
                        )
                        pipeline = GraphIndexingPipeline(graph_store, config)
                        if run_indexing_with_lock is not None:
                            stats = await run_indexing_with_lock(
                                project_root,
                                lambda rp=None: pipeline.index_repository(
                                    root_path=rp,
                                    progress_callback=_on_ccg_progress,
                                    status_callback=_on_ccg_status,
                                ),
                                timeout_seconds=INIT_CCG_LOCK_TIMEOUT_SECONDS,
                            )
                        else:
                            stats = await pipeline.index_repository(
                                progress_callback=_on_ccg_progress,
                                status_callback=_on_ccg_status,
                            )
                        db_stats = await graph_store.stats()
                        return stats, db_stats

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("{task.description}"),
                        BarColumn(bar_width=30),
                        MofNCompleteColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        console=console,
                        transient=True,
                    ) as _ccg_progress:
                        _ccg_task_id = _ccg_progress.add_task("[cyan]  Indexing…[/]", total=None)
                        stats, db_stats = asyncio.run(_build_ccg_index())

                    if stats.files_processed or stats.files_deleted:
                        console.print(
                            f"[green]✓[/] Graph index updated "
                            f"({stats.files_processed} changed, {stats.files_deleted} deleted, "
                            f"{stats.files_unchanged} unchanged)"
                        )
                    else:
                        console.print(
                            f"[green]✓[/] Graph already current "
                            f"({stats.files_unchanged} unchanged files reused)"
                        )

                    # Single authoritative line: DB row counts plus per-type
                    # breakdown. Replaces the prior "X created / Y in db" pair
                    # whose first half counted emit-events (1 per call site)
                    # while the second counted unique upserted rows — the gap
                    # could span an order of magnitude on dense codebases and
                    # routinely confused users into thinking work was wasted.
                    edge_breakdown = _query_edge_type_breakdown(project_root)
                    console.print(
                        f"[dim]    → {db_stats['nodes']:,} nodes, "
                        f"{db_stats['edges']:,} edges in database[/]"
                    )
                    if edge_breakdown:
                        formatted = " · ".join(f"{t} {c:,}" for t, c in edge_breakdown.items())
                        console.print(f"[dim]      {formatted}[/]")

                    # CCG-specific surface: if the user asked for CCG (--ccg)
                    # but no statement-level edges materialized, the project's
                    # primary language likely isn't covered by
                    # SUPPORTED_CCG_LANGUAGES in ccg_builder.py. Say so plainly
                    # instead of leaving "CCG index updated" to imply success.
                    _report_ccg_coverage(project_root, edge_breakdown, console)
                except TimeoutError as exc:
                    console.print(f"[yellow]![/] CCG refresh deferred: {exc}")
                    try:
                        from victor.storage.graph import create_graph_store

                        async def _read_ccg_stats():
                            # Must resolve the same way the write path does, or the
                            # reported counts come from a store init never wrote to.
                            graph_store = create_graph_store("auto", project_path=project_root)
                            return await graph_store.stats()

                        db_stats = asyncio.run(_read_ccg_stats())
                        if db_stats.get("nodes") or db_stats.get("edges"):
                            console.print(
                                "[dim]    → Using existing graph snapshot "
                                f"({db_stats['nodes']:,} total nodes, {db_stats['edges']:,} total edges)[/]"
                            )
                    except Exception:
                        pass
                except ImportError:
                    console.print(
                        "[yellow]![/] CCG requires graph dependencies. Run: pip install 'victor-ai[graph]'"
                    )
                except Exception as exc:
                    console.print(f"[yellow]![/] CCG indexing skipped: {exc}")

            # 2. Gather graph context for LLM synthesis (only when using LLM)
            graph_ctx = None
            if deep:  # Only gather for LLM synthesis
                with _Status(
                    "[dim]  Gathering graph context…[/]",
                    console=console,
                    spinner="dots",
                ):
                    graph_ctx = _gather_graph_context(project_root)
                if graph_ctx:
                    # Attach measured project-scale facts so the synthesis
                    # prompt can anchor file/LOC/crate counts on observed
                    # numbers instead of letting the LLM invent them.
                    # (See init_synthesizer._format_graph_context_for_prompt
                    # for the "Measured Project Scale" section that consumes
                    # this.)
                    graph_ctx["scale_facts"] = _gather_project_scale_facts(project_root)
                    n = graph_ctx["stats"]["total_nodes"]
                    e = graph_ctx["stats"]["total_edges"]
                    f = graph_ctx["scale_facts"].get("total_files", 0)
                    console.print(
                        f"[dim]  Graph context: {n:,} nodes, {e:,} edges, " f"{f:,} source files[/]"
                    )

            # 4. LLM synthesis: agentic (tool-driven, slower, more accurate)
            # or one-shot (default, faster). Agentic mode bypasses the
            # CodebaseAnalyzer fast-path because the agent will gather what
            # it needs via tools; the one-shot path keeps the legacy
            # generate_enhanced_init_md flow.
            try:
                # `is True` (not truthiness): when init() is called directly as
                # a Python function (tests, scripts) an omitted Typer option is
                # a truthy OptionInfo object — which must not silently launch a
                # real tool-driven LLM agent.
                if agentic is True and deep:
                    new_content = _run_agentic_synthesis(
                        provider=provider,
                        model=model,
                        graph_context=graph_ctx,
                        console_=console,
                    )
                else:
                    if agentic and not deep:
                        console.print(
                            "[yellow]![/] --agentic requires --deep; "
                            "falling back to one-shot synthesis"
                        )
                    _synth_label = "Analyzing with LLM" if deep else "Analyzing codebase"
                    with _Status(
                        f"[dim]  {_synth_label}…[/]",
                        console=console,
                        spinner="dots",
                    ) as _s:
                        _synthesis_status = _s
                        new_content = _generate_init_content(
                            mode=mode,
                            use_llm=deep,
                            include_conversations=learn,
                            on_progress=on_progress,
                            force=force,
                            include_dirs=include_dirs or None,
                            exclude_dirs=exclude_dirs or None,
                            provider=provider,
                            model=model,
                            graph_context=graph_ctx,
                        )
                    _synthesis_status = None
            except ImportError:
                console.print("[red]Error: codebase_analyzer requires victor-coding vertical.[/]")
                return

            if update and existing_content:
                from victor.ui.slash.commands.codebase import InitCommand

                # Use InitCommand's public merge function
                init_cmd = InitCommand()
                content = init_cmd.merge_init_content(existing_content, new_content)
                console.print("[dim]  Merged with existing content[/]")
            else:
                content = new_content

            content = ensure_architecture_patterns_section(content, graph_ctx)
            content = ensure_architecture_evidence_section(content, graph_ctx)
            content = ensure_quality_baseline_section(content)

            target_path.write_text(content, encoding="utf-8")
            console.print(f"[green]✓[/] Created {target_path}")

            component_count = content.count("| `")
            pattern_count = count_architecture_patterns(content)
            console.print(f"[dim]  - Detected {component_count} key components[/]")
            console.print(f"[dim]  - Found {pattern_count} architecture patterns[/]")

            if symlinks:
                try:
                    CONTEXT_FILE_ALIASES = cast(
                        Mapping[str, str],
                        load_codebase_analyzer_attr("CONTEXT_FILE_ALIASES"),
                    )
                    create_context_symlinks = cast(
                        Callable[[], dict[str, str]],
                        load_codebase_analyzer_attr("create_context_symlinks"),
                    )
                except ImportError:
                    console.print(
                        "[red]Error: codebase_analyzer requires victor-coding vertical.[/]"
                    )
                    return

                console.print("\n[dim]Creating symlinks for other AI tools...[/]")
                results = create_context_symlinks()

                for alias, status in results.items():
                    tool_name = CONTEXT_FILE_ALIASES.get(alias, "Unknown")
                    if status == "created":
                        console.print(
                            f"  [green]✓[/] {alias} -> {VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE} ({tool_name})"
                        )
                    elif status == "exists":
                        console.print(f"  [dim]○[/] {alias} (already linked)")
                    elif status == "exists_file":
                        console.print(f"  [yellow]![/] {alias} (file exists, not a symlink)")

            console.print(f"\n[dim]Review and customize {target_path} as needed.[/]")

            if interactive and created_profiles:
                if Confirm.ask("[cyan]Show a 2-minute first-run guide?[/]", default=True):
                    console.print("\n[bold]First Run Guide[/]")
                    console.print("1) Start chat: [cyan]victor chat[/]")
                    console.print(
                        '2) Try a repo overview: "Summarize this repo and list risky areas."'
                    )
                    console.print(
                        '3) Try a one-shot: [cyan]victor "write tests for src/utils.py"[/]'
                    )
                    console.print("4) Switch models in [cyan]~/.victor/profiles.yaml[/]")
                    console.print("See [cyan]docs/guides/FIRST_RUN.md[/] for more.")

        except Exception as e:
            console.print(f"[red]Failed to create {VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE}:[/] {e}")
            import traceback

            traceback.print_exc()
