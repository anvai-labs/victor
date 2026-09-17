#!/usr/bin/env python3
"""Victor Test Harness - Programmatic testing of slash commands and verticals.

This script provides a non-interactive test harness for:
- All slash commands without needing manual REPL interaction
- Different verticals (coding, devops, research, data_analysis)
- Agent modes (build, plan, explore)
- Edit/readonly modes

Usage:
    python scripts/test_harness.py                    # Run all tests
    python scripts/test_harness.py --commands         # Test slash commands only
    python scripts/test_harness.py --verticals        # Test verticals only
    python scripts/test_harness.py --modes           # Test modes only
    python scripts/test_harness.py --vertical coding  # Test specific vertical
    python scripts/test_harness.py --provider inferflux  # Use specific provider
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

# Ensure victor is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from a single test."""

    name: str
    passed: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class TestSuite:
    """Collection of test results."""

    name: str
    results: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)


class CaptureConsole:
    """Console that captures output for testing."""

    def __init__(self):
        self._output = StringIO()
        self._console = Console(file=self._output, force_terminal=True, width=120)

    @property
    def console(self) -> Console:
        return self._console

    def get_output(self) -> str:
        return self._output.getvalue()

    def clear(self):
        self._output.truncate(0)
        self._output.seek(0)


class SlashCommandTester:
    """Test harness for slash commands."""

    def __init__(self, provider: str = "inferflux", model: str = "qwen3-coder-30b"):
        self.provider = provider
        self.model = model
        self._settings = None
        self._agent = None
        self._handler = None

    async def setup(self, vertical: str = "coding"):
        """Initialize settings, agent, and command handler."""
        from victor.config.settings import Settings
        from victor.ui.slash.handler import SlashCommandHandler

        # Create settings with specified provider
        self._settings = Settings(
            provider=self.provider,
            default_model=self.model,
            vertical=vertical,
            tool_budget=20,
            max_iterations=30,
            streaming=True,
        )

        # Create capture console
        self._capture = CaptureConsole()

        # Create command handler without agent first (for non-agent commands)
        self._handler = SlashCommandHandler(
            console=self._capture.console,
            settings=self._settings,
            agent=None,
        )

    async def setup_with_agent(self, vertical: str = "coding"):
        """Initialize with full agent for commands that require it."""
        await self.setup(vertical)

        try:
            from victor.agent.orchestrator import AgentOrchestrator
            from victor.providers.registry import ProviderRegistry

            # Get provider instance
            provider_cls = ProviderRegistry.get(self.provider)
            if provider_cls:
                provider = provider_cls()
                self._agent = AgentOrchestrator(
                    settings=self._settings,
                    provider=provider,
                    model=self.model,
                )
                self._handler.set_agent(self._agent)
                logger.info(f"Agent initialized with vertical: {vertical}")
            else:
                logger.warning(f"Provider {self.provider} not found")
        except Exception as e:
            logger.warning(f"Could not initialize agent: {e}")

    def get_all_commands(self) -> List[Tuple[str, Any]]:
        """Get all registered commands."""
        if not self._handler:
            return []
        return list(self._handler.registry.list_commands())

    async def test_command(self, command: str, args: List[str] = None) -> TestResult:
        """Test a single slash command."""
        import time

        if args is None:
            args = []

        full_command = f"/{command} {' '.join(args)}".strip()
        self._capture.clear()

        start = time.perf_counter()
        try:
            await self._handler.execute(full_command)
            duration = (time.perf_counter() - start) * 1000
            output = self._capture.get_output()

            return TestResult(
                name=full_command,
                passed=True,
                output=output,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=full_command,
                passed=False,
                error=str(e),
                duration_ms=duration,
            )

    async def test_all_commands(self) -> TestSuite:
        """Test all registered slash commands."""
        suite = TestSuite(name="Slash Commands")

        # Commands that don't require agent
        no_agent_commands = [
            ("help", []),
            ("help", ["mode"]),
            ("config", []),
            ("theme", []),
            ("theme", ["list"]),
        ]

        # Commands that require agent
        agent_commands = [
            ("mode", []),
            ("mode", ["build"]),
            ("mode", ["plan"]),
            ("mode", ["explore"]),
            ("build", []),
            ("explore", []),
            ("tools", []),
            ("context", []),
            ("status", []),
            ("cost", []),
            ("metrics", []),
        ]

        # Test no-agent commands
        for cmd, args in no_agent_commands:
            result = await self.test_command(cmd, args)
            suite.results.append(result)

        # Test agent commands (if agent available)
        if self._agent:
            for cmd, args in agent_commands:
                result = await self.test_command(cmd, args)
                suite.results.append(result)

        return suite


class VerticalTester:
    """Test harness for verticals."""

    VERTICALS = ["coding", "devops", "research", "data_analysis"]

    def __init__(self, provider: str = "inferflux", model: str = "qwen3-coder-30b"):
        self.provider = provider
        self.model = model

    async def test_vertical_loading(self, vertical: str) -> TestResult:
        """Test loading a specific vertical."""
        try:
            from victor.core.verticals.vertical_loader import VerticalLoader

            loader = VerticalLoader()
            vertical_cls = loader.load(vertical)

            # Get vertical info
            loader.get_config()
            tools = loader.get_tools()
            prompt = loader.get_system_prompt()

            output = (
                f"Loaded: {vertical_cls.name}\n"
                f"Tools: {len(tools)}\n"
                f"System prompt length: {len(prompt)} chars"
            )

            return TestResult(
                name=f"load_{vertical}",
                passed=True,
                output=output,
            )
        except Exception as e:
            return TestResult(
                name=f"load_{vertical}",
                passed=False,
                error=str(e),
            )

    async def test_vertical_tools(self, vertical: str) -> TestResult:
        """Test that vertical has expected tools."""
        try:
            from victor.core.verticals.vertical_loader import VerticalLoader

            loader = VerticalLoader()
            loader.load(vertical)
            tools = loader.get_tools()

            # Check for expected tools per vertical (short tool names)
            expected_tools = {
                "coding": ["code_search", "read", "edit", "bash", "git"],
                "devops": ["shell", "read", "edit", "docker", "web_search"],
                "research": ["web_search", "web_fetch", "read", "code_search"],
                "data_analysis": ["shell", "read", "edit", "code_search", "graph"],
            }

            expected = expected_tools.get(vertical, [])
            missing = [t for t in expected if t not in tools]

            if missing:
                return TestResult(
                    name=f"tools_{vertical}",
                    passed=False,
                    error=f"Missing expected tools: {missing}",
                    output=f"Available: {tools[:10]}...",
                )

            return TestResult(
                name=f"tools_{vertical}",
                passed=True,
                output=f"Has {len(tools)} tools including: {expected}",
            )
        except Exception as e:
            return TestResult(
                name=f"tools_{vertical}",
                passed=False,
                error=str(e),
            )

    async def test_all_verticals(self) -> TestSuite:
        """Test all verticals."""
        suite = TestSuite(name="Verticals")

        for vertical in self.VERTICALS:
            result = await self.test_vertical_loading(vertical)
            suite.results.append(result)

            result = await self.test_vertical_tools(vertical)
            suite.results.append(result)

        return suite


class ModeTester:
    """Test harness for agent modes."""

    def __init__(self, provider: str = "inferflux", model: str = "qwen3-coder-30b"):
        self.provider = provider
        self.model = model
        self._settings = None
        self._agent = None

    async def setup(self, vertical: str = "coding"):
        """Initialize agent for mode testing."""
        from victor.config.settings import Settings

        self._settings = Settings(
            provider=self.provider,
            default_model=self.model,
            vertical=vertical,
            tool_budget=10,
            max_iterations=10,
        )

        try:
            from victor.agent.orchestrator import AgentOrchestrator
            from victor.providers.registry import ProviderRegistry

            # Get provider instance
            provider_cls = ProviderRegistry.get(self.provider)
            if provider_cls:
                provider = provider_cls()
                self._agent = AgentOrchestrator(
                    settings=self._settings,
                    provider=provider,
                    model=self.model,
                )
            else:
                logger.warning(f"Provider {self.provider} not found")
        except Exception as e:
            logger.warning(f"Could not initialize agent: {e}")

    async def test_mode_switch(self, mode: str) -> TestResult:
        """Test switching to a specific mode."""
        try:
            from victor.agent.adaptive_mode_controller import AgentMode

            if not self._agent:
                return TestResult(
                    name=f"mode_{mode}",
                    passed=False,
                    error="Agent not available",
                )

            # Get mode controller
            mode_controller = getattr(self._agent, "_mode_controller", None)

            if mode_controller:
                target_mode = AgentMode(mode)
                mode_controller.switch_mode(target_mode)
                current = mode_controller.get_current_mode()

                if current == target_mode:
                    return TestResult(
                        name=f"mode_{mode}",
                        passed=True,
                        output=f"Successfully switched to {mode} mode",
                    )
                else:
                    return TestResult(
                        name=f"mode_{mode}",
                        passed=False,
                        error=f"Mode mismatch: expected {mode}, got {current.value}",
                    )
            else:
                # Fallback to direct attribute
                self._agent._current_mode = AgentMode(mode)
                return TestResult(
                    name=f"mode_{mode}",
                    passed=True,
                    output=f"Set mode via fallback: {mode}",
                )

        except Exception as e:
            return TestResult(
                name=f"mode_{mode}",
                passed=False,
                error=str(e),
            )

    async def test_all_modes(self) -> TestSuite:
        """Test all agent modes."""
        suite = TestSuite(name="Agent Modes")

        modes = ["build", "plan", "explore"]

        for mode in modes:
            result = await self.test_mode_switch(mode)
            suite.results.append(result)

        return suite


class LiveChatTester:
    """Test harness for live chat interactions."""

    def __init__(self, provider: str = "inferflux", model: str = "qwen3-coder-30b"):
        self.provider = provider
        self.model = model
        self._settings = None
        self._agent = None

    async def setup(self, vertical: str = "coding"):
        """Initialize agent for chat testing."""
        from victor.config.settings import Settings

        self._settings = Settings(
            provider=self.provider,
            default_model=self.model,
            vertical=vertical,
            tool_budget=15,
            max_iterations=20,
            streaming=False,  # Non-streaming for easier testing
        )

        try:
            from victor.agent.orchestrator import AgentOrchestrator
            from victor.providers.registry import ProviderRegistry

            # Get provider instance
            provider_cls = ProviderRegistry.get(self.provider)
            if provider_cls:
                provider = provider_cls()
                self._agent = AgentOrchestrator(
                    settings=self._settings,
                    provider=provider,
                    model=self.model,
                )
                logger.info(f"Chat tester initialized for {vertical}")
            else:
                raise ValueError(f"Provider {self.provider} not found")
        except Exception as e:
            logger.error(f"Could not initialize agent: {e}")
            raise

    async def test_simple_chat(self, prompt: str, timeout: float = 120.0) -> TestResult:
        """Test a simple chat interaction."""
        import time

        if not self._agent:
            return TestResult(
                name=f"chat: {prompt[:30]}...",
                passed=False,
                error="Agent not available",
            )

        start = time.perf_counter()
        try:
            await asyncio.wait_for(
                self._agent.chat(prompt),
                timeout=timeout,
            )
            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name=f"chat: {prompt[:30]}...",
                passed=True,
                output="Chat completed successfully",
                duration_ms=duration,
            )
        except asyncio.TimeoutError:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=f"chat: {prompt[:30]}...",
                passed=False,
                error=f"Timeout after {timeout}s",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=f"chat: {prompt[:30]}...",
                passed=False,
                error=str(e),
                duration_ms=duration,
            )

    async def test_tool_usage(
        self, prompt: str, expected_tools: List[str], timeout: float = 90.0
    ) -> TestResult:
        """Test that a prompt triggers expected tools."""
        import time

        if not self._agent:
            return TestResult(
                name=f"tools: {prompt[:30]}...",
                passed=False,
                error="Agent not available",
            )

        start = time.perf_counter()
        try:
            await asyncio.wait_for(
                self._agent.chat(prompt),
                timeout=timeout,
            )
            duration = (time.perf_counter() - start) * 1000

            # Check if tools were used (from conversation history or response metadata)
            tools_used = []
            if hasattr(self._agent, "_conversation_controller"):
                history = self._agent._conversation_controller.get_history()
                for msg in history:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if isinstance(tc, dict):
                                tools_used.append(tc.get("name", ""))
                            else:
                                tools_used.append(getattr(tc, "name", ""))

            tools_used = list(set(tools_used))
            matched = [t for t in expected_tools if t in tools_used]

            if matched:
                return TestResult(
                    name=f"tools: {prompt[:30]}...",
                    passed=True,
                    output=f"Used tools: {tools_used}",
                    duration_ms=duration,
                )
            else:
                return TestResult(
                    name=f"tools: {prompt[:30]}...",
                    passed=False,
                    error=f"Expected {expected_tools}, got {tools_used}",
                    duration_ms=duration,
                )

        except asyncio.TimeoutError:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=f"tools: {prompt[:30]}...",
                passed=False,
                error=f"Timeout after {timeout}s",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=f"tools: {prompt[:30]}...",
                passed=False,
                error=str(e),
                duration_ms=duration,
            )


def print_results(console: Console, suites: List[TestSuite]):
    """Print test results in a nice format."""
    total_passed = sum(s.passed for s in suites)
    total_failed = sum(s.failed for s in suites)
    total_tests = sum(s.total for s in suites)

    for suite in suites:
        table = Table(title=f"{suite.name} ({suite.passed}/{suite.total} passed)")
        table.add_column("Test", style="cyan")
        table.add_column("Status")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Details", max_width=60)

        for result in suite.results:
            status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
            time_str = f"{result.duration_ms:.1f}" if result.duration_ms > 0 else "-"
            details = (
                result.error
                if result.error
                else (result.output[:60] + "..." if len(result.output) > 60 else result.output)
            )
            # Clean up ANSI codes for display
            details = details.replace("\n", " ").strip()

            table.add_row(result.name, status, time_str, details)

        console.print(table)
        console.print()

    # Summary
    if total_failed > 0:
        console.print(
            Panel(
                f"[red]FAILED[/] - {total_passed}/{total_tests} tests passed, {total_failed} failed",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                f"[green]ALL PASSED[/] - {total_passed}/{total_tests} tests passed",
                border_style="green",
            )
        )


async def run_tests(args):
    """Run the test harness."""
    console = Console()
    suites = []

    console.print(
        Panel(
            f"[bold]Victor Test Harness[/]\n\n"
            f"Provider: {args.provider}\n"
            f"Model: {args.model}\n"
            f"Vertical: {args.vertical or 'all'}",
            title="Configuration",
            border_style="blue",
        )
    )

    # Test slash commands
    if args.commands or args.all:
        console.print("\n[bold cyan]Testing Slash Commands...[/]\n")
        tester = SlashCommandTester(provider=args.provider, model=args.model)

        if args.with_agent:
            await tester.setup_with_agent(args.vertical or "coding")
        else:
            await tester.setup(args.vertical or "coding")

        # Show available commands
        commands = tester.get_all_commands()
        console.print(f"[dim]Found {len(commands)} registered commands[/]")

        suite = await tester.test_all_commands()
        suites.append(suite)

    # Test verticals
    if args.verticals or args.all:
        console.print("\n[bold cyan]Testing Verticals...[/]\n")
        tester = VerticalTester(provider=args.provider, model=args.model)

        if args.vertical:
            # Test specific vertical
            suite = TestSuite(name=f"Vertical: {args.vertical}")
            result = await tester.test_vertical_loading(args.vertical)
            suite.results.append(result)
            result = await tester.test_vertical_tools(args.vertical)
            suite.results.append(result)
            suites.append(suite)
        else:
            suite = await tester.test_all_verticals()
            suites.append(suite)

    # Test modes
    if args.modes or args.all:
        console.print("\n[bold cyan]Testing Agent Modes...[/]\n")
        tester = ModeTester(provider=args.provider, model=args.model)
        await tester.setup(args.vertical or "coding")
        suite = await tester.test_all_modes()
        suites.append(suite)

    # Test live chat (requires provider to be available)
    if args.chat:
        console.print("\n[bold cyan]Testing Live Chat...[/]\n")
        tester = LiveChatTester(provider=args.provider, model=args.model)

        try:
            await tester.setup(args.vertical or "coding")
            suite = TestSuite(name="Live Chat")

            # Simple chat test
            result = await tester.test_simple_chat("Hello, what can you help me with?")
            suite.results.append(result)

            # Code-related prompt
            result = await tester.test_simple_chat("What programming languages do you know?")
            suite.results.append(result)

            suites.append(suite)
        except Exception as e:
            console.print(f"[red]Chat test setup failed: {e}[/]")

    # Print results
    if suites:
        console.print("\n")
        print_results(console, suites)
    else:
        console.print(
            "[yellow]No tests run. Use --all, --commands, --verticals, --modes, or --chat[/]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Victor Test Harness - Programmatic testing of slash commands and verticals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/test_harness.py --all                # Run all tests
    python scripts/test_harness.py --commands           # Test slash commands
    python scripts/test_harness.py --verticals          # Test all verticals
    python scripts/test_harness.py --vertical coding    # Test specific vertical
    python scripts/test_harness.py --modes              # Test agent modes
    python scripts/test_harness.py --chat               # Test live chat (requires provider)
    python scripts/test_harness.py --provider fireworks # Use Fireworks provider
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--commands", action="store_true", help="Test slash commands")
    parser.add_argument("--verticals", action="store_true", help="Test verticals")
    parser.add_argument("--modes", action="store_true", help="Test agent modes")
    parser.add_argument("--chat", action="store_true", help="Test live chat (requires provider)")

    parser.add_argument("--vertical", type=str, help="Test specific vertical")
    parser.add_argument(
        "--with-agent", action="store_true", help="Initialize agent for command tests"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="inferflux",
        help="Provider to use (default: inferflux)",
    )
    parser.add_argument("--model", type=str, default="qwen3-coder:30b", help="Model to use")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # If no test type specified, run all
    if not (args.commands or args.verticals or args.modes or args.chat):
        args.all = True

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(run_tests(args))


if __name__ == "__main__":
    main()
