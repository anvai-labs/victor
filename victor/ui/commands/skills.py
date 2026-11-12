"""Skill management CLI commands.

Provides ``victor skill`` subcommands for listing, inspecting,
searching, creating, and removing skills.

Usage:
    victor skill list
    victor skill list --category coding
    victor skill info debug_test
    victor skill search "refactor"
    victor skill create my_skill --description "..." --tools "read,grep"
    victor skill remove my_skill
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table
from victor.ui.console import get_console

logger = logging.getLogger(__name__)

skills_app = typer.Typer(name="skill", help="Manage and discover skills")
console = get_console()


def _get_skills_dir() -> str:
    """Return the user skills directory (~/.victor/skills/)."""
    from victor.framework.skills import USER_SKILLS_DIR

    return USER_SKILLS_DIR


def _build_registry():
    """Build a SkillRegistry populated from all sources.

    Delegates to SkillRegistry.from_all_sources() for a single source of truth.
    """
    from victor.framework.skills import SkillRegistry

    return SkillRegistry.from_all_sources()


def _render_skills_table(skills, title: str = "Skills") -> None:
    """Render a list of skills as a rich table."""
    from victor.ui.rendering.table_builder import create_skill_table

    if not skills:
        console.print("[dim]No skills found.[/]")
        return

    table = create_skill_table(title=title, show_tools=True)

    for skill in sorted(skills, key=lambda s: s.name):
        tool_count = len(skill.required_tools) + len(skill.optional_tools)
        table.add_row(
            skill.name,
            skill.category,
            skill.description,
            f"{tool_count} tools",
        )

    console.print(table)


@skills_app.command("list")
def list_skills(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (e.g., 'coding', 'devops', 'analysis')",
    ),
) -> None:
    """List all discovered skills."""
    registry = _build_registry()

    if category:
        skills = registry.search(category=category)
        title = f"Skills (category: {category})"
    else:
        skills = registry.list_all()
        title = "Skills"

    _render_skills_table(skills, title=title)


@skills_app.command("info")
def skill_info(
    name: str = typer.Argument(help="Skill name to inspect"),
) -> None:
    """Show detailed information about a skill."""
    registry = _build_registry()
    skill = registry.get_optional(name)

    if skill is None:
        console.print(f"[red]Skill '{name}' not found.[/]")
        return

    console.print(f"[bold cyan]{skill.name}[/] v{skill.version}")
    console.print(f"[dim]Category:[/] {skill.category}")
    console.print(f"[dim]Description:[/] {skill.description}")
    console.print()

    if skill.required_tools:
        console.print("[bold]Required tools:[/]")
        for tool in skill.required_tools:
            console.print(f"  - {tool}")

    if skill.optional_tools:
        console.print("[bold]Optional tools:[/]")
        for tool in skill.optional_tools:
            console.print(f"  - {tool}")

    if skill.tags:
        console.print(f"[dim]Tags:[/] {', '.join(sorted(skill.tags))}")

    console.print(f"[dim]Max tool calls:[/] {skill.max_tool_calls}")

    if skill.prompt_fragment:
        console.print()
        console.print("[bold]Prompt fragment:[/]")
        console.print(f"  {skill.prompt_fragment[:200]}")


@skills_app.command("search")
def search_skills(
    query: str = typer.Argument(help="Search query (matches name, description, tags)"),
) -> None:
    """Search skills by keyword."""
    registry = _build_registry()
    results = registry.search(query=query)
    _render_skills_table(results, title=f"Search: '{query}'")


@skills_app.command("stats")
def skill_stats() -> None:
    """Show skill selection analytics (from current or last session)."""
    try:
        from victor.framework.skill_analytics import SkillAnalytics

        # Try to load from a running session — in CLI context this
        # creates a fresh instance showing the analytics pattern
        analytics = SkillAnalytics()

        # In a real session, analytics would be populated.
        # For CLI demonstration, show the schema
        all_stats = analytics.get_all_stats()
        global_stats = analytics.get_global_stats()

        if not all_stats:
            console.print("[dim]No skill analytics recorded yet.[/]")
            console.print(
                "[dim]Analytics are collected during 'victor chat' sessions "
                "when skills are auto-selected.[/]"
            )
            return

        table = Table(title="Skill Selection Analytics")
        table.add_column("Skill", style="cyan")
        table.add_column("Selections", justify="right")
        table.add_column("Avg Score", justify="right")

        for stat in all_stats:
            table.add_row(
                stat["name"],
                str(stat["count"]),
                f"{stat['avg_score']:.2f}",
            )

        console.print(table)
        console.print()
        console.print(
            f"[dim]Total matches: {global_stats['total_matches']} | "
            f"Misses: {global_stats['total_misses']} | "
            f"Miss rate: {global_stats['miss_rate']:.0%} | "
            f"Multi-skill: {global_stats['multi_skill_count']}[/]"
        )
    except Exception as e:
        console.print(f"[red]Error loading analytics:[/] {e}")


@skills_app.command("create")
def create_skill(
    name: str = typer.Argument(help="Skill name (e.g., analyze_logs)"),
    description: str = typer.Option(..., "--description", "-d", help="What this skill does"),
    prompt: str = typer.Option(
        ..., "--prompt", "-p", help="Prompt fragment injected when skill is active"
    ),
    tools: str = typer.Option(
        ...,
        "--tools",
        "-t",
        help="Comma-separated required tools (e.g., read,grep,shell)",
    ),
    category: str = typer.Option("custom", "--category", "-c", help="Skill category"),
    optional_tools: Optional[str] = typer.Option(
        None, "--optional-tools", help="Comma-separated optional tools"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    max_tool_calls: int = typer.Option(20, "--max-tool-calls", help="Max tool calls"),
) -> None:
    """Create a new user skill (saved as YAML in ~/.victor/skills/)."""
    skills_dir = _get_skills_dir()
    os.makedirs(skills_dir, exist_ok=True)

    skill_data = {
        "name": name,
        "description": description,
        "category": category,
        "prompt_fragment": prompt,
        "required_tools": [t.strip() for t in tools.split(",") if t.strip()],
    }

    if optional_tools:
        skill_data["optional_tools"] = [t.strip() for t in optional_tools.split(",") if t.strip()]

    if tags:
        skill_data["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    skill_data["max_tool_calls"] = max_tool_calls

    filepath = os.path.join(skills_dir, f"{name}.yaml")
    with open(filepath, "w") as f:
        yaml.dump(skill_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created skill:[/] {name}")
    console.print(f"[dim]Saved to:[/] {filepath}")
    console.print()
    console.print("The skill will appear in [cyan]victor skill list[/] immediately.")


@skills_app.command("remove")
def remove_skill(
    name: str = typer.Argument(help="Name of the user skill to remove"),
) -> None:
    """Remove a user-created skill (deletes the YAML file from ~/.victor/skills/)."""
    skills_dir = _get_skills_dir()

    for ext in (".yaml", ".yml"):
        filepath = os.path.join(skills_dir, f"{name}{ext}")
        if os.path.exists(filepath):
            os.remove(filepath)
            console.print(f"[green]Removed skill:[/] {name}")
            console.print(f"[dim]Deleted:[/] {filepath}")
            return

    console.print(f"[yellow]User skill '{name}' not found in {skills_dir}[/]")


@skills_app.command("preview")
def preview_skill(
    name: str = typer.Argument(help="Skill name to preview"),
    task: str = typer.Option(
        "Describe what you would do.",
        "--task",
        "-t",
        help="Example task to show in context",
    ),
) -> None:
    """Preview how a skill configures the agent (dry run — no LLM call)."""
    registry = _build_registry()
    skill = registry.get_optional(name)

    if skill is None:
        console.print(f"[red]Skill '{name}' not found.[/]")
        raise typer.Exit(1)

    console.print(f"[bold]Skill Preview:[/] [cyan]{skill.name}[/]")
    console.print()

    # Show the composed system prompt
    console.print("[bold green]System Prompt (injected):[/]")
    console.print("─" * 60)
    console.print(skill.prompt_fragment)
    console.print("─" * 60)
    console.print()

    # Show tool configuration
    console.print("[bold green]Tools activated:[/]")
    for tool in skill.required_tools:
        console.print(f"  [cyan]{tool}[/] (required)")
    for tool in skill.optional_tools:
        console.print(f"  [dim]{tool}[/] (optional)")
    console.print()

    # Show constraints
    console.print(f"[bold green]Constraints:[/] max {skill.max_tool_calls} tool calls")
    console.print()

    # Show what the user message would look like
    console.print("[bold green]User message:[/]")
    console.print(f"  {task}")
    console.print()
    console.print("[dim]To run this skill for real: " f'victor skill run {name} -t "{task}"[/]')


@skills_app.command("run")
def run_skill(
    name: str = typer.Argument(help="Skill name to activate"),
    task: str = typer.Option(..., "--task", "-t", help="The task for the agent to perform"),
    profile: str = typer.Option("default", "--profile", "-p", help="Profile from profiles.yaml"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override provider (e.g., anthropic, ollama, openai)"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model"),
    thinking: bool = typer.Option(False, "--thinking", help="Enable extended thinking"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show skill injection details"),
) -> None:
    """Run the agent with a skill active — its prompt + tools are injected.

    Uses the same provider/model/profile resolution as 'victor chat'.
    The skill's prompt fragment is prepended to the system prompt and
    its tool budget is enforced.

    Examples:
        victor skill run debug_test_failure -t "Fix the failing test in test_auth.py"
        victor skill run security_audit -t "Audit src/" --provider anthropic
        victor skill run explore_codebase -t "How does routing work?" --profile fast
    """
    from victor.core.async_utils import run_sync

    registry = _build_registry()
    skill = registry.get_optional(name)

    if skill is None:
        console.print(f"[red]Skill '{name}' not found.[/]")
        console.print("[dim]Run 'victor skill list' to see available skills.[/]")
        raise typer.Exit(1)

    if verbose:
        console.print(f"[bold]Activating skill:[/] [cyan]{skill.name}[/]")
        console.print(f"[dim]Category:[/] {skill.category}")
        console.print(f"[dim]Tools:[/] {', '.join(skill.required_tools)}")
        if skill.optional_tools:
            console.print(f"[dim]Optional:[/] {', '.join(skill.optional_tools)}")
        console.print(f"[dim]Max tool calls:[/] {skill.max_tool_calls}")
        console.print()

    async def _run():
        from victor.core.verticals import get_vertical

        from victor.framework.session_config import SessionConfig
        from victor.framework.session_runner import create_victor_client

        # Compose the skill system prompt
        skill_system_prompt = (
            f"You are an expert assistant with the '{skill.name}' skill active.\n"
            f"Description: {skill.description}\n\n"
            f"Follow these steps:\n{skill.prompt_fragment}\n\n"
            f"Constraint: Use at most {skill.max_tool_calls} tool calls."
        )

        # Resolve vertical from skill category if a matching vertical exists
        vertical_class = None
        try:
            vertical_class = get_vertical(skill.category)
        except Exception:
            pass  # No matching vertical — run without one

        # ✅ PROPER: Create SessionConfig with all overrides
        config = SessionConfig.from_cli_flags(
            provider=provider,
            model=model,
            tool_budget=skill.max_tool_calls,
            thinking=thinking,
            profile=profile,
            vertical=vertical_class.__name__ if vertical_class else None,
            enable_observability=False,
        )

        client = create_victor_client(config)
        agent = await client.initialize()

        # Inject skill prompt into the agent's system prompt
        if hasattr(agent, "_system_prompt"):
            agent._system_prompt = skill_system_prompt + "\n\n" + (agent._system_prompt or "")
        elif hasattr(agent, "system_prompt"):
            agent.system_prompt = skill_system_prompt + "\n\n" + (agent.system_prompt or "")

        console.print(f"[bold green]Running skill:[/] [cyan]{skill.name}[/]")
        console.print(f"[bold green]Task:[/] {task}")
        console.print("─" * 60)

        response = await agent.chat(task)
        content = response.content if hasattr(response, "content") else str(response)

        console.print()
        console.print("─" * 60)
        console.print("[bold green]Result:[/]")
        console.print(content)

        usage = getattr(response, "usage", None)
        if usage:
            console.print()
            console.print(
                f"[dim]Tokens: {usage.get('input_tokens', '?')} in "
                f"/ {usage.get('output_tokens', '?')} out[/]"
            )

    try:
        run_sync(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        console.print(
            "[dim]Check your API key and provider settings. "
            "Run 'victor keys check' for diagnostics.[/]"
        )
