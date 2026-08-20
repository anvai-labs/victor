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

"""Tests for the modular slash command system (victor.ui.slash)."""

from datetime import datetime
import io
import logging
from pathlib import Path
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, AsyncMock, call, patch

from rich.console import Console

from victor.ui.slash import (
    SlashCommandHandler,
    CommandMetadata,
    CommandContext,
    BaseSlashCommand,
    get_command_registry,
    register_command,
)
from victor.ui.slash.registry import CommandRegistry


class TestCommandMetadata:
    """Tests for CommandMetadata dataclass."""

    def test_basic_metadata(self):
        """Test creating basic command metadata."""
        meta = CommandMetadata(
            name="test",
            description="A test command",
            usage="/test",
        )
        assert meta.name == "test"
        assert meta.description == "A test command"
        assert meta.usage == "/test"
        assert meta.aliases == []
        assert meta.category == "general"

    def test_metadata_with_aliases(self):
        """Test metadata with aliases."""
        meta = CommandMetadata(
            name="help",
            description="Show help",
            usage="/help [command]",
            aliases=["?", "h"],
        )
        assert meta.aliases == ["?", "h"]

    def test_metadata_with_category(self):
        """Test metadata with category."""
        meta = CommandMetadata(
            name="save",
            description="Save session",
            usage="/save [name]",
            category="session",
        )
        assert meta.category == "session"


class TestCommandContext:
    """Tests for CommandContext."""

    def test_context_creation(self):
        """Test creating a command context."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        agent = MagicMock()

        ctx = CommandContext(
            console=console,
            settings=settings,
            agent=agent,
            args=["arg1", "arg2"],
        )

        assert ctx.console is console
        assert ctx.settings is settings
        assert ctx.agent is agent
        assert ctx.args == ["arg1", "arg2"]

    def test_context_without_agent(self):
        """Test context without agent."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        ctx = CommandContext(
            console=console,
            settings=settings,
        )

        assert ctx.agent is None
        assert ctx.args == []


class TestSlashCommandHandlerInit:
    """Tests for SlashCommandHandler initialization."""

    def test_init_with_defaults(self):
        """Test handler initialization with defaults."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.console is console
        assert handler.settings is settings
        assert handler.agent is None

    def test_init_with_agent(self):
        """Test handler initialization with agent."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        agent = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings, agent=agent)

        assert handler.agent is agent

    def test_set_agent(self):
        """Test setting agent after initialization."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        agent = MagicMock()
        handler.set_agent(agent)

        assert handler.agent is agent


class TestSlashCommandHandlerDefaultCommands:
    """Tests for default command registration."""

    def test_help_command_registered(self):
        """Test help command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.registry.has("help")
        assert handler.registry.has("?")  # alias

    def test_model_command_registered(self):
        """Test model command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.registry.has("model")
        assert handler.registry.has("models")  # alias

    def test_clear_command_registered(self):
        """Test clear command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.registry.has("clear")
        assert handler.registry.has("reset")  # alias

    def test_tools_command_registered(self):
        """Test tools command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.registry.has("tools")

    def test_exit_command_registered(self):
        """Test exit command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.registry.has("exit")
        assert handler.registry.has("quit")  # alias


class TestSlashCommandHandlerIsCommand:
    """Tests for is_command method."""

    def test_is_command_true(self):
        """Test is_command returns True for slash commands."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.is_command("/help") is True
        assert handler.is_command("/model gpt-4") is True
        assert handler.is_command("  /clear") is True

    def test_is_command_false(self):
        """Test is_command returns False for non-commands."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.is_command("hello") is False
        assert handler.is_command("help me") is False
        assert handler.is_command("") is False


class TestSlashCommandHandlerParseCommand:
    """Tests for parse_command method."""

    def test_parse_simple_command(self):
        """Test parsing simple command without args."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        name, args = handler.parse_command("/help")
        assert name == "help"
        assert args == []

    def test_parse_command_with_args(self):
        """Test parsing command with arguments."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        name, args = handler.parse_command("/model gpt-4")
        assert name == "model"
        assert args == ["gpt-4"]

    def test_parse_command_multiple_args(self):
        """Test parsing command with multiple arguments."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        name, args = handler.parse_command("/save my_session --force")
        assert name == "save"
        assert args == ["my_session", "--force"]

    def test_parse_command_with_whitespace(self):
        """Test parsing command with extra whitespace."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        name, args = handler.parse_command("  /help  ")
        assert name == "help"


class TestSlashCommandHandlerExecute:
    """Tests for execute method."""

    @pytest.mark.asyncio
    async def test_execute_known_command(self):
        """Test executing a known command."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        result = await handler.execute("/help")
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_unknown_command(self):
        """Test executing an unknown command still returns True (handled)."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        result = await handler.execute("/nonexistent")
        # Unknown commands are still "handled" (error message shown)
        assert result is True
        assert "unknown" in stdout.getvalue().lower() or "nonexistent" in stdout.getvalue().lower()

    @pytest.mark.asyncio
    async def test_execute_command_by_alias(self):
        """Test executing command using alias."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        result = await handler.execute("/?")  # Alias for help
        assert result is True

    def test_execute_sync_runs_async_execute(self):
        """execute_sync bridges async execution from sync contexts."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)
        handler.execute = AsyncMock(return_value=True)

        result = handler.execute_sync("/help")

        assert result is True
        handler.execute.assert_awaited_once_with("/help")

    @pytest.mark.asyncio
    async def test_execute_sync_rejects_async_context(self):
        """execute_sync rejects nested event-loop usage."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)
        handler.execute = AsyncMock(return_value=True)

        with pytest.raises(RuntimeError, match="Cannot use run_sync\\(\\)"):
            handler.execute_sync("/help")


class TestSlashCommandHandlerHelp:
    """Tests for help command."""

    @pytest.mark.asyncio
    async def test_help_shows_commands(self):
        """Test help command shows available commands."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/help")
        output = stdout.getvalue()

        assert "help" in output.lower() or "commands" in output.lower()

    @pytest.mark.asyncio
    async def test_help_specific_command(self):
        """Test help for specific command."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/help model")
        output = stdout.getvalue()

        # Should show model command info
        assert "model" in output.lower()

    @pytest.mark.asyncio
    async def test_help_graph_tool_modes(self):
        """Test help can describe graph tool call modes."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/help graph")
        output = stdout.getvalue().lower()

        assert "callers" in output
        assert "callees" in output
        assert "trace" in output


class TestSlashCommandHandlerClear:
    """Tests for clear command."""

    @pytest.mark.asyncio
    async def test_clear_with_agent(self):
        """Test clear command with agent."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        agent = MagicMock()
        agent.reset_conversation = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings, agent=agent)

        await handler.execute("/clear")
        agent.reset_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_without_agent(self):
        """Test clear command without agent shows warning."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/clear")
        # Should show "no active session" warning
        output = stdout.getvalue()
        assert "session" in output.lower() or len(output) > 0


class TestSlashCommandHandlerStatus:
    """Tests for status command."""

    @pytest.mark.asyncio
    async def test_status_without_agent(self):
        """Test status command without agent shows warning."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/status")
        output = stdout.getvalue()
        # Should show some output (warning or error)
        assert len(output) > 0

    @pytest.mark.asyncio
    async def test_status_with_agent(self):
        """Test status command with agent shows info."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        settings.tools.tool_call_budget = 25

        agent = MagicMock()
        agent.provider_name = "anthropic"
        agent.model = "claude-3"
        agent.conversation = MagicMock()
        agent.conversation.message_count = MagicMock(return_value=10)

        handler = SlashCommandHandler(console=console, settings=settings, agent=agent)

        await handler.execute("/status")
        output = stdout.getvalue()

        # Should show some status info
        assert len(output) > 0


class TestSlashCommandHandlerExit:
    """Tests for exit command."""

    @pytest.mark.asyncio
    async def test_exit_raises_exception(self):
        """Test exit command raises SystemExit."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        with pytest.raises(SystemExit):
            await handler.execute("/exit")


class TestAllCommandsRegistered:
    """Tests to verify all expected slash commands are registered."""

    # Complete list of expected commands
    EXPECTED_COMMANDS = [
        "help",
        "init",
        "model",
        "profile",
        "provider",
        "clear",
        "context",
        "lmstudio",
        "tools",
        "status",
        "config",
        "save",
        "load",
        "sessions",
        "compact",
        "mcp",
        "review",
        "bug",
        "exit",
        "undo",
        "redo",
        "theme",
        "changes",
        "cost",
        "approvals",
        "resume",
        "plan",
        "search",
        "copy",
        "directory",
        "snapshots",
        "commit",
        "mode",
        "build",
        "explore",
        "reindex",
        "metrics",
        "gain",
        "serialization",
        "learning",
        "mlstats",
        "bayesian",
    ]

    # Expected aliases mapping
    EXPECTED_ALIASES = {
        "help": ["?"],
        "model": ["models", "switch"],
        "profile": ["profiles"],
        "provider": ["providers"],
        "clear": ["reset"],
        "context": ["ctx", "memory"],
        "lmstudio": ["lm"],
        "status": ["info"],
        "config": ["settings"],
        "compact": ["summarize"],
        "mcp": ["servers"],
        "bug": ["issue", "feedback"],
        "exit": ["quit"],
        "theme": ["dark", "light"],
        "changes": ["diff"],
        "cost": ["usage"],
        "approvals": ["safety"],
        "search": ["web"],
        "directory": ["cd"],
        "snapshots": ["snap"],
        "commit": ["ci"],
        "mode": ["m"],
        "reindex": ["index"],
        "metrics": ["perf", "performance"],
        "gain": ["savings", "condensation"],
        "serialization": ["serialize", "ser"],
        "learning": ["qlearn", "rl"],
        "mlstats": ["ml", "analytics"],
        "bayesian": ["bayes"],
    }

    def test_all_commands_registered(self):
        """Test that all expected commands are registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        for cmd_name in self.EXPECTED_COMMANDS:
            assert handler.registry.has(cmd_name), f"Command '{cmd_name}' not registered"

    def test_all_aliases_registered(self):
        """Test that all expected aliases are registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        for cmd_name, aliases in self.EXPECTED_ALIASES.items():
            for alias in aliases:
                assert handler.registry.has(
                    alias
                ), f"Alias '{alias}' for '{cmd_name}' not registered"

    def test_command_count(self):
        """Test we have at least 40 commands."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        command_count = len(list(handler.registry.list_commands()))
        assert command_count >= 40, f"Expected at least 40 commands, got {command_count}"

    def test_each_command_has_description(self):
        """Test that each command has a non-empty description."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        for cmd_name in self.EXPECTED_COMMANDS:
            cmd = handler.registry.get(cmd_name)
            assert cmd is not None, f"Command '{cmd_name}' not found"
            assert cmd.metadata.description, f"Command '{cmd_name}' has empty description"


class TestSlashCommandCategories:
    """Tests for different categories of slash commands."""

    def test_session_commands(self):
        """Test session-related commands exist."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        session_commands = ["save", "load", "sessions", "resume"]
        for cmd in session_commands:
            assert handler.registry.has(cmd), f"Session command '{cmd}' missing"

    def test_mode_commands(self):
        """Test mode-related commands exist."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        mode_commands = ["mode", "build", "explore", "plan"]
        for cmd in mode_commands:
            assert handler.registry.has(cmd), f"Mode command '{cmd}' missing"

    def test_utility_commands(self):
        """Test utility commands exist."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        util_commands = ["copy", "theme", "directory", "search"]
        for cmd in util_commands:
            assert handler.registry.has(cmd), f"Utility command '{cmd}' missing"

    def test_metrics_commands(self):
        """Test metrics/stats commands exist."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        metrics_commands = ["cost", "metrics", "mlstats", "learning", "bayesian"]
        for cmd in metrics_commands:
            assert handler.registry.has(cmd), f"Metrics command '{cmd}' missing"


class TestCommandRegistry:
    """Tests for the command registry."""

    def test_registry_singleton(self):
        """Test that get_command_registry returns the same instance."""
        registry1 = get_command_registry()
        registry2 = get_command_registry()
        assert registry1 is registry2

    def test_registry_categories(self):
        """Test registry has expected categories."""
        registry = get_command_registry()
        categories = registry.categories()

        expected = [
            "system",
            "session",
            "model",
            "tools",
            "mode",
            "metrics",
            "navigation",
            "codebase",
        ]
        for cat in expected:
            assert cat in categories, f"Category '{cat}' not found"

    def test_registry_list_by_category(self):
        """Test listing commands by category."""
        registry = get_command_registry()

        system_commands = registry.list_by_category("system")
        assert len(system_commands) > 0

        # Help should be in system category
        help_in_system = any(c.metadata.name == "help" for c in system_commands)
        assert help_in_system, "Help command not in system category"

    def test_duplicate_registration_logs_debug_not_warning(self, caplog):
        """Duplicate slash command registration is expected during reloads."""

        class DuplicateCommand(BaseSlashCommand):
            @property
            def metadata(self):
                return CommandMetadata(
                    name="duplicate",
                    description="duplicate command",
                    usage="/duplicate",
                )

        registry = CommandRegistry()
        command = DuplicateCommand()
        registry.register(command)

        with caplog.at_level(logging.DEBUG, logger="victor.ui.slash.registry"):
            registry.register(command)

        duplicate_records = [
            record for record in caplog.records if "already registered" in record.message
        ]
        assert duplicate_records
        assert all(record.levelno == logging.DEBUG for record in duplicate_records)


class TestBaseSlashCommand:
    """Tests for BaseSlashCommand helper methods."""

    def test_require_agent_with_agent(self):
        """Test _require_agent returns True when agent present."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        agent = MagicMock()

        ctx = CommandContext(console=console, settings=settings, agent=agent)

        cmd = BaseSlashCommand()
        assert cmd._require_agent(ctx) is True

    def test_require_agent_without_agent(self):
        """Test _require_agent returns False when no agent."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()

        ctx = CommandContext(console=console, settings=settings, agent=None)

        cmd = BaseSlashCommand()
        assert cmd._require_agent(ctx) is False
        assert "session" in stdout.getvalue().lower()

    def test_has_flag(self):
        """Test _has_flag method."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        ctx = CommandContext(
            console=console,
            settings=settings,
            args=["--force", "-v", "other"],
        )

        cmd = BaseSlashCommand()
        assert cmd._has_flag(ctx, "--force") is True
        assert cmd._has_flag(ctx, "-f", "--force") is True
        assert cmd._has_flag(ctx, "-v") is True
        assert cmd._has_flag(ctx, "--verbose") is False

    def test_get_arg(self):
        """Test _get_arg method."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        ctx = CommandContext(
            console=console,
            settings=settings,
            args=["first", "second", "third"],
        )

        cmd = BaseSlashCommand()
        assert cmd._get_arg(ctx, 0) == "first"
        assert cmd._get_arg(ctx, 1) == "second"
        assert cmd._get_arg(ctx, 5) is None
        assert cmd._get_arg(ctx, 5, "default") == "default"

    def test_parse_int_arg(self):
        """Test _parse_int_arg method."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        ctx = CommandContext(
            console=console,
            settings=settings,
            args=["42", "not_a_number"],
        )

        cmd = BaseSlashCommand()
        assert cmd._parse_int_arg(ctx, 0) == 42
        assert cmd._parse_int_arg(ctx, 1) == 0  # default for invalid
        assert cmd._parse_int_arg(ctx, 1, default=10) == 10
        assert cmd._parse_int_arg(ctx, 5, default=99) == 99


class TestLearningCommandUnified:
    """Tests to verify the /learning command is unified (not duplicated)."""

    def test_learning_command_exists(self):
        """Test learning command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.registry.has("learning")
        assert handler.registry.has("rl")  # alias
        assert handler.registry.has("qlearn")  # alias

    def test_learning_in_metrics_category(self):
        """Test learning command is in metrics category."""
        registry = get_command_registry()
        cmd = registry.get("learning")

        assert cmd is not None
        assert cmd.metadata.category == "metrics"

    @pytest.mark.asyncio
    async def test_learning_shows_stats(self):
        """Test learning command shows stats without errors."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        # Should not raise an error
        await handler.execute("/learning")
        output = stdout.getvalue()

        # Should show some output
        assert len(output) > 0

    def test_learning_prefers_runtime_intelligence_capability_for_stats(self):
        """Learning stats should resolve the canonical runtime-intelligence capability first."""
        from victor.ui.slash.commands.metrics import LearningCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        stats = SimpleNamespace(
            session_duration=12.5,
            total_requests=4,
            enhanced_requests=3,
            quality_validations=2,
            avg_quality_score=0.85,
            avg_grounding_score=0.75,
        )
        integration = SimpleNamespace(
            get_stats=lambda: stats,
            mode_controller=SimpleNamespace(
                get_session_stats=lambda: {
                    "profile_name": "coding",
                    "total_reward": 1.25,
                    "mode_transitions": 2,
                    "exploration_rate": 0.15,
                    "modes_visited": ["READING", "ACTING"],
                }
            ),
        )

        def get_capability_value(name: str):
            if name == "runtime_intelligence_integration":
                return integration
            raise AssertionError(f"Unexpected capability lookup: {name}")

        agent = SimpleNamespace(
            runtime_intelligence_integration=None,
            get_capability_value=MagicMock(side_effect=get_capability_value),
            provider_name="ollama",
        )
        ctx = CommandContext(console=console, settings=settings, agent=agent)

        with patch("victor.framework.rl.coordinator.get_rl_coordinator") as mock_get_coordinator:
            mock_get_coordinator.return_value.get_learner.return_value = None
            LearningCommand().execute(ctx)

        output = stdout.getvalue()
        assert "Runtime-Intelligence:" in output
        assert "Intelligent Pipeline:" not in output
        assert agent.get_capability_value.call_args_list == [
            call("runtime_intelligence_integration")
        ]

    def test_learning_reset_prefers_runtime_intelligence_capability(self):
        """Learning reset should resolve the canonical runtime-intelligence capability first."""
        from victor.ui.slash.commands.metrics import LearningCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        integration = MagicMock()
        learner = MagicMock()

        def get_capability_value(name: str):
            if name == "runtime_intelligence_integration":
                return integration
            raise AssertionError(f"Unexpected capability lookup: {name}")

        agent = SimpleNamespace(
            runtime_intelligence_integration=None,
            get_capability_value=MagicMock(side_effect=get_capability_value),
            provider_name="ollama",
        )
        ctx = CommandContext(console=console, settings=settings, agent=agent, args=["reset"])

        with patch("victor.framework.rl.coordinator.get_rl_coordinator") as mock_get_coordinator:
            mock_get_coordinator.return_value.get_learner.return_value = learner
            LearningCommand().execute(ctx)

        integration.reset_session.assert_called_once_with()
        learner.reset.assert_called_once_with()
        assert agent.get_capability_value.call_args_list == [
            call("runtime_intelligence_integration")
        ]


# =============================================================================
# SYSTEM COMMANDS TESTS
# =============================================================================


class TestSystemCommands:
    """Tests for system slash commands (help, config, status, exit, clear)."""

    def test_help_command_metadata(self):
        """Test HelpCommand metadata."""
        from victor.ui.slash.commands.system import HelpCommand

        cmd = HelpCommand()
        meta = cmd.metadata

        assert meta.name == "help"
        assert "?" in meta.aliases
        assert meta.category == "system"

    def test_handler_get_help_supports_graph_tool(self):
        """Test programmatic help includes embedded graph tool guidance."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        output = handler.get_help("graph").lower()

        assert "callers" in output
        assert "callees" in output
        assert "trace" in output

    def test_config_command_metadata(self):
        """Test ConfigCommand metadata."""
        from victor.ui.slash.commands.system import ConfigCommand

        cmd = ConfigCommand()
        meta = cmd.metadata

        assert meta.name == "config"
        assert meta.category == "system"

    def test_status_command_metadata(self):
        """Test StatusCommand metadata."""
        from victor.ui.slash.commands.system import StatusCommand

        cmd = StatusCommand()
        meta = cmd.metadata

        assert meta.name == "status"
        assert meta.category == "system"

    def test_exit_command_metadata(self):
        """Test ExitCommand metadata."""
        from victor.ui.slash.commands.system import ExitCommand

        cmd = ExitCommand()
        meta = cmd.metadata

        assert meta.name == "exit"
        assert "quit" in meta.aliases

    def test_clear_command_metadata(self):
        """Test ClearCommand metadata."""
        from victor.ui.slash.commands.system import ClearCommand

        cmd = ClearCommand()
        meta = cmd.metadata

        assert meta.name == "clear"
        assert "reset" in meta.aliases

    @pytest.mark.asyncio
    async def test_config_command_execution(self):
        """Test ConfigCommand execution."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        settings.provider.default_provider = "anthropic"
        settings.provider.default_model = "claude-3-5-sonnet"
        settings.ollama_base_url = "http://localhost:11434"
        settings.airgapped_mode = False
        settings.use_semantic_tool_selection = True
        settings.search.unified_embedding_model = "all-MiniLM-L6-v2"
        settings.codebase_graph_store = "sqlite"
        settings.graph_enabled = True

        handler = SlashCommandHandler(console=console, settings=settings)
        await handler.execute("/config")

        output = stdout.getvalue()
        assert "anthropic" in output or "Provider" in output


# =============================================================================
# MODE COMMANDS TESTS
# =============================================================================


class TestModeCommands:
    """Tests for mode slash commands."""

    def test_mode_command_metadata(self):
        """Test ModeCommand metadata."""
        from victor.ui.slash.commands.mode import ModeCommand

        cmd = ModeCommand()
        meta = cmd.metadata

        assert meta.name == "mode"
        assert meta.category in ["mode", "system", "general"]

    def test_build_command_metadata(self):
        """Test BuildCommand metadata."""
        from victor.ui.slash.commands.mode import BuildCommand

        cmd = BuildCommand()
        meta = cmd.metadata

        assert meta.name == "build"
        assert meta.category in ["mode", "system", "general"]

    def test_plan_command_metadata(self):
        """Test PlanCommand metadata."""
        from victor.ui.slash.commands.mode import PlanCommand

        cmd = PlanCommand()
        meta = cmd.metadata

        assert meta.name == "plan"
        assert meta.category in ["mode", "system", "general"]

    def test_explore_command_metadata(self):
        """Test ExploreCommand metadata."""
        from victor.ui.slash.commands.mode import ExploreCommand

        cmd = ExploreCommand()
        meta = cmd.metadata

        assert meta.name == "explore"
        assert meta.category in ["mode", "system", "general"]

    def test_review_command_metadata(self):
        """Test ReviewCommand metadata."""
        from victor.ui.slash.commands.mode import ReviewCommand

        cmd = ReviewCommand()
        meta = cmd.metadata

        assert meta.name == "review"
        assert meta.category in ["mode", "system", "general"]

    def test_delegate_command_metadata(self):
        """Test DelegateCommand metadata."""
        from victor.ui.slash.commands.mode import DelegateCommand

        cmd = DelegateCommand()
        meta = cmd.metadata

        assert meta.name == "delegate"
        assert meta.category in ["mode", "system", "general"]

    def test_mode_command_switches_canonical_controller_and_refreshes_prompt(self):
        """Mode command should use the canonical AgentMode enum and refresh prompt."""
        from victor.agent.mode_controller import AgentMode, AgentModeController
        from victor.ui.slash.commands.mode import ModeCommand

        console = Console(file=io.StringIO())
        controller = AgentModeController()
        agent = SimpleNamespace(
            mode_controller=controller,
            refresh_system_prompt=MagicMock(),
        )
        ctx = CommandContext(
            console=console,
            settings=MagicMock(),
            agent=agent,
            args=["plan"],
        )

        ModeCommand().execute(ctx)

        assert controller.current_mode == AgentMode.PLAN
        assert controller.config.name == "Plan"
        agent.refresh_system_prompt.assert_called_once()

    def test_mode_command_switches_to_review_mode(self):
        """Mode command should support the review-focused coding mode."""
        from victor.agent.mode_controller import AgentMode, AgentModeController
        from victor.ui.slash.commands.mode import ModeCommand

        console = Console(file=io.StringIO())
        controller = AgentModeController()
        agent = SimpleNamespace(
            mode_controller=controller,
            refresh_system_prompt=MagicMock(),
        )
        ctx = CommandContext(
            console=console,
            settings=MagicMock(),
            agent=agent,
            args=["review"],
        )

        ModeCommand().execute(ctx)

        assert controller.current_mode == AgentMode.REVIEW
        assert controller.config.name == "Review"
        agent.refresh_system_prompt.assert_called_once()


# =============================================================================
# MODEL COMMANDS TESTS
# =============================================================================


class TestModelCommands:
    """Tests for model slash commands."""

    def test_model_command_metadata(self):
        """Test ModelCommand metadata."""
        from victor.ui.slash.commands.model import ModelCommand

        cmd = ModelCommand()
        meta = cmd.metadata

        assert meta.name == "model"
        assert "models" in meta.aliases
        assert meta.category in ["model", "system", "general"]

    def test_provider_command_metadata(self):
        """Test ProviderCommand metadata."""
        from victor.ui.slash.commands.model import ProviderCommand

        cmd = ProviderCommand()
        meta = cmd.metadata

        assert meta.name == "provider"
        assert meta.category in ["model", "system", "general"]

    @pytest.mark.asyncio
    async def test_model_command_awaits_switch_model(self):
        """ModelCommand uses the async model-switch API."""
        from victor.ui.slash.commands.model import ModelCommand

        console = Console(file=io.StringIO())
        settings = MagicMock()
        agent = MagicMock()
        agent.switch_model = AsyncMock(return_value=True)
        agent.get_current_provider_info.return_value = {
            "native_tool_calls": True,
            "thinking_mode": False,
        }

        ctx = CommandContext(
            console=console,
            settings=settings,
            agent=agent,
            args=["claude-3.7-sonnet"],
        )

        await ModelCommand().execute(ctx)

        agent.switch_model.assert_awaited_once_with("claude-3.7-sonnet")


class TestCheckpointCommands:
    """Tests for checkpoint slash commands."""

    def test_checkpoint_command_metadata(self):
        """CheckpointCommand is registered as an async command."""
        from victor.ui.slash.commands.checkpoint import CheckpointCommand

        cmd = CheckpointCommand()
        meta = cmd.metadata

        assert meta.name == "checkpoint"
        assert meta.is_async is True

    @pytest.mark.asyncio
    async def test_checkpoint_save_awaits_agent_api(self):
        """Checkpoint save uses the async agent API directly."""
        from victor.ui.slash.commands.checkpoint import CheckpointCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        agent = MagicMock()
        agent.checkpoint_manager = MagicMock()
        agent.save_checkpoint = AsyncMock(return_value="ckpt_1234567890")

        ctx = CommandContext(
            console=console,
            settings=settings,
            agent=agent,
            args=["save", "before", "refactor"],
        )

        await CheckpointCommand().execute(ctx)

        agent.save_checkpoint.assert_awaited_once_with(description="before refactor")
        assert "Checkpoint saved successfully" in stdout.getvalue()

    @pytest.mark.asyncio
    async def test_checkpoint_list_awaits_manager_api(self):
        """Checkpoint list uses the async checkpoint-manager API directly."""
        from victor.ui.slash.commands.checkpoint import CheckpointCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        checkpoint = MagicMock()
        checkpoint.checkpoint_id = "ckpt_1234567890abcdef"
        checkpoint.timestamp = datetime(2026, 3, 27, 12, 34)
        checkpoint.stage = "ACTIVE"
        checkpoint.tool_count = 3
        checkpoint.message_count = 7
        checkpoint.description = "snapshot"

        checkpoint_manager = MagicMock()
        checkpoint_manager.list_checkpoints = AsyncMock(return_value=[checkpoint])

        agent = MagicMock()
        agent.checkpoint_manager = checkpoint_manager
        agent._memory_session_id = "session-123"

        ctx = CommandContext(
            console=console,
            settings=settings,
            agent=agent,
            args=["list", "5"],
        )

        await CheckpointCommand().execute(ctx)

        checkpoint_manager.list_checkpoints.assert_awaited_once_with("session-123", limit=5)
        assert "Checkpoints" in stdout.getvalue()


# =============================================================================
# NAVIGATION COMMANDS TESTS
# =============================================================================


class TestNavigationCommands:
    """Tests for navigation slash commands."""

    def test_directory_command_metadata(self):
        """Test DirectoryCommand metadata."""
        from victor.ui.slash.commands.navigation import DirectoryCommand

        cmd = DirectoryCommand()
        meta = cmd.metadata

        assert meta.name == "directory"
        assert meta.category in ["navigation", "filesystem", "general"]

    def test_changes_command_metadata(self):
        """Test ChangesCommand metadata."""
        from victor.ui.slash.commands.navigation import ChangesCommand

        cmd = ChangesCommand()
        meta = cmd.metadata

        assert meta.name == "changes"

    def test_undo_command_metadata(self):
        """Test UndoCommand metadata."""
        from victor.ui.slash.commands.navigation import UndoCommand

        cmd = UndoCommand()
        meta = cmd.metadata

        assert meta.name == "undo"

    def test_redo_command_metadata(self):
        """Test RedoCommand metadata."""
        from victor.ui.slash.commands.navigation import RedoCommand

        cmd = RedoCommand()
        meta = cmd.metadata

        assert meta.name == "redo"

    def test_history_command_metadata(self):
        """Test HistoryCommand metadata."""
        from victor.ui.slash.commands.navigation import HistoryCommand

        cmd = HistoryCommand()
        meta = cmd.metadata

        assert meta.name == "filehistory"

    def test_snapshots_command_metadata(self):
        """Test SnapshotsCommand metadata."""
        from victor.ui.slash.commands.navigation import SnapshotsCommand

        cmd = SnapshotsCommand()
        meta = cmd.metadata

        assert meta.name == "snapshots"

    def test_commit_command_metadata(self):
        """Test CommitCommand metadata."""
        from victor.ui.slash.commands.navigation import CommitCommand

        cmd = CommitCommand()
        meta = cmd.metadata

        assert meta.name == "commit"

    def test_copy_command_metadata(self):
        """Test CopyCommand metadata."""
        from victor.ui.slash.commands.navigation import CopyCommand

        cmd = CopyCommand()
        meta = cmd.metadata

        assert meta.name == "copy"


# =============================================================================
# SESSION COMMANDS TESTS
# =============================================================================


class TestSessionCommands:
    """Tests for session slash commands."""

    def test_save_command_metadata(self):
        """Test SaveCommand metadata."""
        from victor.ui.slash.commands.session import SaveCommand

        cmd = SaveCommand()
        meta = cmd.metadata

        assert meta.name == "save"
        assert meta.category in ["session", "general"]

    def test_save_command_surfaces_preview_counts(self):
        """Test SaveCommand shows saved message and preview counts."""
        from victor.ui.slash.commands.session import SaveCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        settings.current_profile = "default"
        agent = MagicMock()
        agent.active_session_id = None
        agent.model = "claude-sonnet-4-20250514"
        agent.provider_name = "anthropic"
        agent.conversation = MagicMock()
        agent.conversation.message_count.return_value = 3
        agent.conversation.preview_messages = [
            {
                "role": "assistant",
                "content": "diff",
                "metadata": {"preview_path": "app.py"},
            }
        ]

        store = MagicMock()
        store.save_session.return_value = "test-session-123"

        ctx = CommandContext(
            console=console, settings=settings, agent=agent, args=["Preview Session"]
        )

        with patch("victor.agent.conversation.store.ConversationStore", return_value=store):
            SaveCommand().execute(ctx)

        output = stdout.getvalue()
        assert "Session Saved" in output
        assert "test-session-123" in output
        assert "Messages:" in output
        assert "Previews:" in output
        assert "Preview Files:" in output
        assert "Preview Session" in output
        assert "app.py" in output
        assert "3" in output
        assert "1" in output

    def test_load_command_metadata(self):
        """Test LoadCommand metadata."""
        from victor.ui.slash.commands.session import LoadCommand

        cmd = LoadCommand()
        meta = cmd.metadata

        assert meta.name == "load"
        assert meta.category in ["session", "general"]

    def test_sessions_command_metadata(self):
        """Test SessionsCommand metadata."""
        from victor.ui.slash.commands.session import SessionsCommand

        cmd = SessionsCommand()
        meta = cmd.metadata

        assert meta.name == "sessions"

    def test_sessions_command_displays_preview_counts(self):
        """Test SessionsCommand shows preview counts in the listing table."""
        from victor.ui.slash.commands.session import SessionsCommand
        from victor.agent.conversation.store import ConversationSession

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        store = MagicMock()
        store.list_sessions.return_value = [
            ConversationSession(
                session_id="test-session-123",
                title="Preview Session",
                model="claude-sonnet-4-20250514",
                provider="anthropic",
                messages=[MagicMock(), MagicMock()],
                preview_messages=[MagicMock()],
                created_at=datetime.fromisoformat("2026-03-19T10:00:00"),
            )
        ]

        ctx = CommandContext(console=console, settings=settings, args=[])

        with patch("victor.agent.conversation.store.ConversationStore", return_value=store):
            SessionsCommand().execute(ctx)

        output = stdout.getvalue()
        assert "Saved Sessions" in output
        assert "test-session-123" in output
        assert "Preview Session" in output
        assert "claude-sonnet-4-20250514" in output
        assert "1" in output

    def test_load_command_surfaces_preview_counts(self):
        """Test LoadCommand shows preview counts from persisted preview messages."""
        from victor.ui.slash.commands.session import LoadCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        agent = MagicMock()
        session = MagicMock()
        session.conversation = {
            "messages": [{"role": "user", "content": "hello"}],
            "preview_messages": [
                {
                    "role": "assistant",
                    "content": "diff",
                    "metadata": {"preview_path": "app.py"},
                }
            ],
        }
        session.conversation_state = None
        session.metadata = SimpleNamespace(
            title="Preview Session",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            message_count=2,
            created_at="2026-03-19T10:00:00",
        )

        ctx = CommandContext(
            console=console,
            settings=settings,
            agent=agent,
            args=["test-session-123"],
        )

        with patch("victor.agent.session.get_session_manager") as get_session_manager:
            get_session_manager.return_value.load_session.return_value = session
            LoadCommand().execute(ctx)

        output = stdout.getvalue()
        assert "Session Loaded" in output
        assert "Previews:" in output
        assert "Preview Files:" in output
        assert "Preview Session" in output
        assert "app.py" in output
        assert "1" in output

    def test_resume_command_surfaces_resume_summary(self):
        """Test ResumeCommand shows SessionContextLinker resume summary."""
        from victor.agent.session_context_linker import SessionResumeContext
        from victor.ui.slash.commands.session import ResumeCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        agent = MagicMock()
        agent.conversation_controller = None

        store = MagicMock()
        store.load_session.return_value = {
            "metadata": {
                "session_id": "test-session-123",
                "title": "Preview Session",
                "model": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "message_count": 2,
                "created_at": "2026-03-19T10:00:00",
            },
            "conversation": {
                "messages": [{"role": "user", "content": "hello"}],
                "preview_messages": [
                    {
                        "role": "assistant",
                        "content": "diff",
                        "metadata": {"preview_path": "app.py"},
                    }
                ],
            },
        }
        linker_instance = MagicMock()
        linker_instance.build_resume_context.return_value = SessionResumeContext(
            resume_summary="[Resumed session. previews: app.py.]"
        )

        ctx = CommandContext(
            console=console, settings=settings, agent=agent, args=["test-session-123"]
        )

        with (
            patch("victor.agent.conversation.store.ConversationStore", return_value=store),
            patch(
                "victor.agent.session_context_linker.SessionContextLinker",
                return_value=linker_instance,
            ),
        ):
            ResumeCommand().execute(ctx)

        output = stdout.getvalue()
        assert "Session Resumed" in output
        assert "Previews:" in output
        assert "Preview Files:" in output
        assert "Resume:" in output
        assert "app.py" in output
        assert "1" in output

    def test_resume_command_selection_displays_preview_counts(self):
        """Test interactive ResumeCommand listing shows preview counts."""
        from victor.ui.slash.commands.session import ResumeCommand
        from victor.agent.conversation.store import ConversationSession

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False, width=160)
        settings = MagicMock()
        agent = MagicMock()
        store = MagicMock()
        store.list_sessions.return_value = [
            ConversationSession(
                session_id="test-session-123",
                title="Preview Session",
                model="claude-sonnet-4-20250514",
                provider="anthropic",
                messages=[MagicMock(), MagicMock()],
                preview_messages=[MagicMock()],
                created_at=datetime.fromisoformat("2026-03-19T10:00:00"),
            )
        ]

        ctx = CommandContext(console=console, settings=settings, agent=agent, args=[])

        with patch("victor.agent.conversation.store.ConversationStore", return_value=store):
            ResumeCommand().execute(ctx)

        output = stdout.getvalue()
        assert "Recent Sessions" in output
        assert "Previews" in output
        assert "test-session-123" in output
        assert "Preview Session" in output
        assert "/resume <session_id>" in output


# =============================================================================
# TOOLS COMMANDS TESTS
# =============================================================================


class TestToolsCommands:
    """Tests for tools slash commands."""

    def test_tools_command_metadata(self):
        """Test ToolsCommand metadata."""
        from victor.ui.slash.commands.tools import ToolsCommand

        cmd = ToolsCommand()
        meta = cmd.metadata

        assert meta.name == "tools"
        assert meta.category in ["tools", "system", "general"]

    def test_context_command_metadata(self):
        """Test ContextCommand metadata."""
        from victor.ui.slash.commands.tools import ContextCommand

        cmd = ContextCommand()
        meta = cmd.metadata

        assert meta.name == "context"

    def test_lmstudio_command_metadata(self):
        """Test LMStudioCommand metadata."""
        from victor.ui.slash.commands.tools import LMStudioCommand

        cmd = LMStudioCommand()
        meta = cmd.metadata

        assert meta.name == "lmstudio"

    def test_mcp_command_metadata(self):
        """Test MCPCommand metadata."""
        from victor.ui.slash.commands.tools import MCPCommand

        cmd = MCPCommand()
        meta = cmd.metadata

        assert meta.name == "mcp"

    def test_review_command_metadata(self):
        """Test ReviewCommand metadata."""
        from victor.ui.slash.commands.tools import ReviewCommand

        cmd = ReviewCommand()
        meta = cmd.metadata

        assert meta.name == "review"

    @pytest.mark.asyncio
    async def test_tools_command_lists_tools(self):
        """Test ToolsCommand lists available tools."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)
        await handler.execute("/tools")

        output = stdout.getvalue()
        # Should output something (tool list or message)
        assert len(output) > 0


# =============================================================================
# METRICS COMMANDS TESTS
# =============================================================================


class TestMetricsCommands:
    """Tests for metrics slash commands."""

    def test_metrics_command_metadata(self):
        """Test MetricsCommand metadata."""
        from victor.ui.slash.commands.metrics import MetricsCommand

        cmd = MetricsCommand()
        meta = cmd.metadata

        assert meta.name == "metrics"
        assert meta.category in ["metrics", "system", "general"]

    def test_cost_command_metadata(self):
        """Test CostCommand metadata."""
        from victor.ui.slash.commands.metrics import CostCommand

        cmd = CostCommand()
        meta = cmd.metadata

        assert meta.name == "cost"

    def test_serialization_command_metadata(self):
        """Test SerializationCommand metadata."""
        from victor.ui.slash.commands.metrics import SerializationCommand

        cmd = SerializationCommand()
        meta = cmd.metadata

        assert meta.name == "serialization"

    def test_learning_command_metadata(self):
        """Test LearningCommand metadata."""
        from victor.ui.slash.commands.metrics import LearningCommand

        cmd = LearningCommand()
        meta = cmd.metadata

        assert meta.name == "learning"
        assert "rl" in meta.aliases or "qlearn" in meta.aliases

    def test_mlstats_command_metadata(self):
        """Test MLStatsCommand metadata."""
        from victor.ui.slash.commands.metrics import MLStatsCommand

        cmd = MLStatsCommand()
        meta = cmd.metadata

        assert meta.name == "mlstats"

    def test_bayesian_command_metadata(self):
        """Test BayesianCommand metadata."""
        from victor.ui.slash.commands.bayesian import BayesianCommand

        cmd = BayesianCommand()
        meta = cmd.metadata

        assert meta.name == "bayesian"
        assert "bayes" in meta.aliases
        assert meta.category == "metrics"

    def test_bayesian_command_shows_summary(self):
        """Test BayesianCommand renders summary output."""
        from victor.ui.slash.commands.bayesian import BayesianCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        service = MagicMock()
        service.render_summary.return_value = "Bayesian summary body"

        with patch(
            "victor.ui.slash.commands.bayesian.get_bayesian_monitoring_service",
            return_value=service,
        ):
            BayesianCommand().execute(
                CommandContext(
                    console=console,
                    settings=settings,
                    args=["summary", "--days", "14"],
                )
            )

        output = stdout.getvalue()
        assert "Bayesian summary body" in output
        service.render_summary.assert_called_once_with(14)


# =============================================================================
# CODEBASE COMMANDS TESTS
# =============================================================================


class TestCodebaseCommands:
    """Tests for codebase slash commands."""

    def test_reindex_command_metadata(self):
        """Test ReindexCommand metadata."""
        from victor.ui.slash.commands.codebase import ReindexCommand

        cmd = ReindexCommand()
        meta = cmd.metadata

        assert meta.name == "reindex"
        assert meta.category in ["codebase", "code", "general"]

    def test_init_command_metadata(self):
        """Test InitCommand metadata."""
        from victor.ui.slash.commands.codebase import InitCommand

        cmd = InitCommand()
        meta = cmd.metadata

        assert meta.name == "init"

    @pytest.mark.asyncio
    async def test_reindex_command_passes_graph_writer_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """ReindexCommand should pass graph settings through to CodebaseIndexFactory.create."""
        from victor.ui.slash.commands.codebase import ReindexCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        graph_path = tmp_path / "graph.sqlite"
        settings = SimpleNamespace(
            codebase_graph_writer_mode="off",
            codebase_graph_store="sqlite",
            codebase_graph_path=str(graph_path),
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
        )
        paths = SimpleNamespace(embeddings_dir=tmp_path / ".victor" / "embeddings")
        mock_index = MagicMock()
        mock_index.ensure_indexed = AsyncMock()
        mock_index.get_stats.return_value = {
            "total_files": 3,
            "total_symbols": 11,
            "total_lines": 420,
            "embeddings_enabled": True,
            "is_indexed": True,
        }
        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_index
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_factory

        monkeypatch.setattr("victor.config.settings.load_settings", lambda: settings)
        monkeypatch.setattr(
            "victor.config.settings.get_project_paths",
            lambda root=None: paths,
        )
        monkeypatch.setattr(
            "victor.core.capability_registry.CapabilityRegistry.get_instance",
            lambda: mock_registry,
        )
        monkeypatch.chdir(tmp_path)

        ctx = CommandContext(console=console, settings=settings, args=[])
        await ReindexCommand().execute(ctx)

        mock_factory.create.assert_called_once_with(
            root_path=str(tmp_path),
            use_embeddings=True,
            embedding_config={
                "vector_store": "lancedb",
                "embedding_model_type": "sentence-transformers",
                "embedding_model_name": "BAAI/bge-small-en-v1.5",
                "persist_directory": str(paths.embeddings_dir),
            },
            graph_writer_mode="off",
            graph_store_name="sqlite",
            graph_path=graph_path,
        )
        mock_index.ensure_indexed.assert_awaited_once_with(auto_reindex=True)

    @pytest.mark.asyncio
    async def test_init_command_quick_mode_writes_generated_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """InitCommand executes the quick-mode analyzer path and writes init.md."""
        from victor.ui.slash.commands.codebase import InitCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        target_path = tmp_path / ".victor" / "init.md"
        project_victor_dir = target_path.parent

        monkeypatch.setattr(
            "victor.config.settings.get_project_paths",
            lambda: SimpleNamespace(
                project_context_file=target_path,
                project_victor_dir=project_victor_dir,
            ),
        )

        generated = "# Generated init\n"

        def load_attr(name: str):
            if name == "generate_smart_victor_md":
                return lambda: generated
            if name == "generate_enhanced_init_md":
                return AsyncMock(return_value="# deep init\n")
            if name == "generate_victor_md_from_index":
                return AsyncMock(return_value="# index init\n")
            raise AssertionError(f"unexpected analyzer export requested: {name}")

        monkeypatch.setattr(
            "victor.ui.slash.commands.codebase.load_codebase_analyzer_attr", load_attr
        )

        ctx = CommandContext(console=console, settings=settings, args=[])

        await InitCommand().execute(ctx)

        content = target_path.read_text(encoding="utf-8")
        assert content.startswith(generated)
        assert "## Repository Working Agreements" in content
        assert "Preserve user work in git" in content
        assert "Created" in stdout.getvalue()

    @pytest.mark.asyncio
    async def test_init_command_reports_architecture_pattern_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """InitCommand should count markdown architecture bullets accurately."""
        from victor.ui.slash.commands.codebase import InitCommand

        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        target_path = tmp_path / ".victor" / "init.md"
        project_victor_dir = target_path.parent

        monkeypatch.setattr(
            "victor.config.settings.get_project_paths",
            lambda: SimpleNamespace(
                project_context_file=target_path,
                project_victor_dir=project_victor_dir,
            ),
        )

        generated = """# init.md

## Architecture Patterns

- **Facade**: One entry point
- **Registry**: Shared extension lookup
"""

        def load_attr(name: str):
            if name == "generate_smart_victor_md":
                return lambda: generated
            if name == "generate_enhanced_init_md":
                return AsyncMock(return_value="# deep init\n")
            if name == "generate_victor_md_from_index":
                return AsyncMock(return_value="# index init\n")
            raise AssertionError(f"unexpected analyzer export requested: {name}")

        monkeypatch.setattr(
            "victor.ui.slash.commands.codebase.load_codebase_analyzer_attr", load_attr
        )

        ctx = CommandContext(console=console, settings=settings, args=[])

        await InitCommand().execute(ctx)

        assert "Patterns: 2" in stdout.getvalue()
