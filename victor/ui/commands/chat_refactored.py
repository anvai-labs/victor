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

"""
Refactored chat command using the framework client factory and SessionConfig.

This demonstrates the proper architectural pattern:
- Uses the framework client creation seam instead of AgentFactory
- Uses SessionConfig for CLI/runtime overrides (not settings mutation)
- No direct imports of AgentOrchestrator or FrameworkShim
- Proper service-oriented architecture
"""

import typer
from typing import Optional, Any
from rich.console import Console

from victor.framework.session_config import SessionConfig
from victor.framework.session_runner import create_victor_client

app = typer.Typer()
console = Console()


@app.command()
def chat(
    message: str = typer.Argument(None, help="Message to send"),
    provider: str = typer.Option(None, help="LLM provider"),
    model: Optional[str] = typer.Option(None, help="Model identifier"),
    tool_budget: Optional[int] = typer.Option(None, help="Tool call budget"),
    max_iterations: Optional[int] = typer.Option(None, help="Maximum iterations"),
    enable_smart_routing: bool = typer.Option(False, help="Enable smart routing"),
    routing_profile: str = typer.Option("balanced", help="Routing profile"),
    tool_preview: bool = typer.Option(True, help="Show tool output previews"),
    enable_pruning: bool = typer.Option(False, help="Enable tool output pruning"),
    profile: str = typer.Option("default", help="Configuration profile"),
    vertical: Optional[str] = typer.Option(None, help="Vertical to use"),
    thinking: bool = typer.Option(False, help="Enable extended thinking"),
    show_reasoning: bool = typer.Option(False, help="Show LLM reasoning"),
    mode: Optional[str] = typer.Option(None, help="Agent mode"),
    enable_planning: bool = typer.Option(None, help="Enable structured planning"),
    planning_model: Optional[str] = typer.Option(None, help="Planning model"),
    # Compaction parameters
    compaction_threshold: Optional[float] = typer.Option(None, help="Compaction threshold"),
    adaptive_threshold: Optional[bool] = typer.Option(None, help="Adaptive compaction"),
    compaction_min_threshold: Optional[float] = typer.Option(None, help="Min adaptive threshold"),
    compaction_max_threshold: Optional[float] = typer.Option(None, help="Max adaptive threshold"),
):
    """Start interactive chat or send a one-shot message.

    This refactored version demonstrates the proper architectural pattern:
    - Uses the framework client seam (NOT AgentFactory or AgentOrchestrator)
    - Uses SessionConfig for CLI overrides (NOT settings mutations)
    - No FrameworkShim (deprecated)
    """
    import asyncio

    if message:
        # One-shot mode
        asyncio.run(
            _run_oneshot(
                message=message,
                provider=provider,
                model=model,
                tool_budget=tool_budget,
                max_iterations=max_iterations,
                enable_smart_routing=enable_smart_routing,
                routing_profile=routing_profile,
                tool_preview=tool_preview,
                enable_pruning=enable_pruning,
                profile=profile,
                vertical=vertical,
                thinking=thinking,
                show_reasoning=show_reasoning,
                mode=mode,
                enable_planning=enable_planning,
                planning_model=planning_model,
                compaction_threshold=compaction_threshold,
                adaptive_threshold=adaptive_threshold,
                compaction_min_threshold=compaction_min_threshold,
                compaction_max_threshold=compaction_max_threshold,
            )
        )
    else:
        # Interactive mode
        asyncio.run(
            _run_interactive(
                provider=provider,
                model=model,
                tool_budget=tool_budget,
                max_iterations=max_iterations,
                enable_smart_routing=enable_smart_routing,
                routing_profile=routing_profile,
                tool_preview=tool_preview,
                enable_pruning=enable_pruning,
                profile=profile,
                vertical=vertical,
                thinking=thinking,
                show_reasoning=show_reasoning,
                mode=mode,
                enable_planning=enable_planning,
                planning_model=planning_model,
                compaction_threshold=compaction_threshold,
                adaptive_threshold=adaptive_threshold,
                compaction_min_threshold=compaction_min_threshold,
                compaction_max_threshold=compaction_max_threshold,
            )
        )


async def _run_oneshot(
    message: str,
    provider: Optional[str],
    model: Optional[str],
    tool_budget: Optional[int],
    max_iterations: Optional[int],
    enable_smart_routing: bool,
    routing_profile: str,
    tool_preview: bool,
    enable_pruning: bool,
    profile: str,
    vertical: Optional[str],
    thinking: bool,
    show_reasoning: bool,
    mode: Optional[str],
    enable_planning: Optional[bool],
    planning_model: Optional[str],
    compaction_threshold: Optional[float],
    adaptive_threshold: Optional[bool],
    compaction_min_threshold: Optional[float],
    compaction_max_threshold: Optional[float],
) -> None:
    """Run a single message using the framework client seam.

    This demonstrates the PROPER pattern:
    1. Create SessionConfig from CLI flags
    2. Create the canonical framework client
    3. Use client.chat() or client.stream()
    4. NO settings mutations, NO AgentFactory, NO orchestrator access
    """
    # ✅ STEP 1: Create SessionConfig from CLI flags
    config = SessionConfig.from_cli_flags(
        tool_budget=tool_budget,
        max_iterations=max_iterations,
        compaction_threshold=compaction_threshold,
        adaptive_threshold=adaptive_threshold,
        compaction_min_threshold=compaction_min_threshold,
        compaction_max_threshold=compaction_max_threshold,
        enable_smart_routing=enable_smart_routing,
        routing_profile=routing_profile,
        fallback_chain=None,
        tool_preview=tool_preview,
        enable_pruning=enable_pruning,
        planning_enabled=enable_planning,
        planning_model=planning_model,
        mode=mode,
        show_reasoning=show_reasoning,
    )

    # ✅ STEP 2: Create the canonical framework client
    client = create_victor_client(config)

    # ✅ STEP 3: Use client methods (NOT orchestrator directly)
    try:
        result = await client.chat(message)

        # Display result
        console.print("[green]Response:[/]")
        console.print(result.content)

        if result.metadata:
            console.print(f"\n[dim]Metadata: {result.metadata}[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)
    finally:
        await client.close()


async def _run_interactive(
    provider: Optional[str],
    model: Optional[str],
    tool_budget: Optional[int],
    max_iterations: Optional[int],
    enable_smart_routing: bool,
    routing_profile: str,
    tool_preview: bool,
    enable_pruning: bool,
    profile: str,
    vertical: Optional[str],
    thinking: bool,
    show_reasoning: bool,
    mode: Optional[str],
    enable_planning: Optional[bool],
    planning_model: Optional[str],
    compaction_threshold: Optional[float],
    adaptive_threshold: Optional[bool],
    compaction_min_threshold: Optional[float],
    compaction_max_threshold: Optional[float],
) -> None:
    """Run interactive mode using the framework client seam.

    This demonstrates the PROPER pattern:
    1. Create SessionConfig from CLI flags
    2. Create the canonical framework client
    3. Use client.stream() for responses
    4. NO settings mutations, NO AgentFactory, NO orchestrator access
    """
    # ✅ STEP 1: Create SessionConfig from CLI flags
    config = SessionConfig.from_cli_flags(
        tool_budget=tool_budget,
        max_iterations=max_iterations,
        compaction_threshold=compaction_threshold,
        adaptive_threshold=adaptive_threshold,
        compaction_min_threshold=compaction_min_threshold,
        compaction_max_threshold=compaction_max_threshold,
        enable_smart_routing=enable_smart_routing,
        routing_profile=routing_profile,
        fallback_chain=None,
        tool_preview=tool_preview,
        enable_pruning=enable_pruning,
        planning_enabled=enable_planning,
        planning_model=planning_model,
        mode=mode,
        show_reasoning=show_reasoning,
    )

    # ✅ STEP 2: Create the canonical framework client
    client = create_victor_client(config)

    console.print("[bold green]Victor Chat[/] [dim](Ctrl+D to exit)[/]")
    console.print("")

    try:
        while True:
            # Read user input
            try:
                user_input = typer.prompt("You")
            except EOFError:
                break

            if not user_input.strip():
                continue

            # ✅ STEP 3: Use VictorClient.stream() for responses
            console.print("[bold]Victor:[/]", end=" ")

            async for event in client.stream(user_input):
                if event.event_type == "content":
                    console.print(event.content, end="")
                elif event.event_type == "thinking":
                    console.print(f"[dim yellow]Thinking: {event.content}[/]", end=" ")
                elif event.event_type == "tool_call":
                    console.print(f"\n[dim]→ {event.tool_name}[/]", end="")
                elif event.event_type == "error":
                    console.print(f"\n[red]Error: {event.content}[/]")

            console.print("")  # New line after response

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        raise typer.Exit(1)
    finally:
        await client.close()
        console.print("[dim]Session closed[/]")


if __name__ == "__main__":
    app()
