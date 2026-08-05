# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for /entities slash command."""

import pytest
from unittest.mock import MagicMock

from victor.ui.slash.commands.entities import EntitiesCommand, entities_command
from victor.ui.slash.protocol import CommandContext, CommandMetadata


@pytest.fixture
def command():
    """Create command instance."""
    return EntitiesCommand()


@pytest.fixture
def mock_console():
    """Create mock console."""
    console = MagicMock()
    console.print = MagicMock()
    return console


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    return MagicMock()


@pytest.fixture
def context(mock_console, mock_settings):
    """Create command context."""
    return CommandContext(
        console=mock_console,
        settings=mock_settings,
        agent=None,
        args=[],
    )


class TestEntitiesCommandMetadata:
    """Tests for command metadata."""

    def test_command_name(self, command):
        """Test command name."""
        assert command.metadata.name == "entities"

    def test_command_description(self, command):
        """Test command description exists."""
        assert command.metadata.description is not None
        assert len(command.metadata.description) > 0

    def test_command_usage(self, command):
        """Test command usage."""
        assert "/entities" in command.metadata.usage

    def test_command_category(self, command):
        """Test command category."""
        assert command.metadata.category == "entities"

    def test_requires_agent(self, command):
        """Test requires_agent flag."""
        assert command.metadata.requires_agent is False


class TestEntitiesHelp:
    """Tests for help subcommand."""

    def test_help_no_args(self, command, context):
        """Test help displayed when no args."""
        command.execute(context)

        # Should print help panel
        context.console.print.assert_called()

    def test_help_explicit(self, command, context):
        """Test explicit help subcommand."""
        context.args = ["help"]
        command.execute(context)

        context.console.print.assert_called()


class TestEntitiesList:
    """Tests for list subcommand."""

    def test_list_no_memory(self, command, context):
        """Test listing entities without memory."""
        context.args = ["list"]
        command.execute(context)

        # Should print warning
        call_args = [str(call) for call in context.console.print.call_args_list]
        assert any(
            "No entity memory" in str(call) or "memory" in str(call).lower() for call in call_args
        )

    def test_list_with_type_filter(self, command, context):
        """Test listing with type filter."""
        context.args = ["list", "class"]
        command.execute(context)

        context.console.print.assert_called()


class TestEntitiesSearch:
    """Tests for search subcommand."""

    def test_search_no_query(self, command, context):
        """Test search without query."""
        context.args = ["search"]
        command.execute(context)

        # Should print usage
        call_args = [str(call) for call in context.console.print.call_args_list]
        assert any("Usage" in str(call) for call in call_args)

    def test_search_with_query(self, command, context):
        """Test search with query."""
        context.args = ["search", "auth"]
        command.execute(context)

        call_args = [str(call) for call in context.console.print.call_args_list]
        assert any("auth" in str(call).lower() for call in call_args)


class TestEntitiesShow:
    """Tests for show subcommand."""

    def test_show_no_name(self, command, context):
        """Test show without name."""
        context.args = ["show"]
        command.execute(context)

        call_args = [str(call) for call in context.console.print.call_args_list]
        assert any("Usage" in str(call) for call in call_args)

    def test_show_with_name(self, command, context):
        """Test show with name."""
        context.args = ["show", "TestClass"]
        command.execute(context)

        context.console.print.assert_called()


class TestEntitiesRelated:
    """Tests for related subcommand."""

    def test_related_no_name(self, command, context):
        """Test related without name."""
        context.args = ["related"]
        command.execute(context)

        call_args = [str(call) for call in context.console.print.call_args_list]
        assert any("Usage" in str(call) for call in call_args)

    def test_related_with_name(self, command, context):
        """Test related with name."""
        context.args = ["related", "TestClass"]
        command.execute(context)

        context.console.print.assert_called()


class TestEntitiesStats:
    """Tests for stats subcommand."""

    def test_show_stats_no_memory(self, command, context):
        """Test showing stats without memory."""
        context.args = ["stats"]
        command.execute(context)

        # Should print table
        context.console.print.assert_called()


class TestEntitiesClear:
    """Tests for clear subcommand."""

    def test_clear_no_memory(self, command, context):
        """Test clear when no entity memory."""
        context.args = ["clear"]
        command.execute(context)

        call_args = [str(call) for call in context.console.print.call_args_list]
        assert any("No entity memory" in str(call) for call in call_args)


class TestUnknownSubcommand:
    """Tests for unknown subcommand handling."""

    def test_unknown_subcommand(self, command, context):
        """Test unknown subcommand."""
        context.args = ["unknown"]
        command.execute(context)

        call_args = [str(call) for call in context.console.print.call_args_list]
        assert any("Unknown subcommand" in str(call) for call in call_args)


class TestGlobalCommandInstance:
    """Tests for the global command instance."""

    def test_global_instance_exists(self):
        """Test global command instance exists."""
        assert entities_command is not None
        assert isinstance(entities_command, EntitiesCommand)

    def test_global_instance_name(self):
        """Test global instance has correct name."""
        assert entities_command.metadata.name == "entities"


class TestWithAgentContext:
    """Tests with mock agent context."""

    def test_list_with_agent_no_entity_memory(self, command, mock_console, mock_settings):
        """Test listing with agent that has no entity memory."""
        mock_agent = MagicMock()
        mock_agent.entity_memory = None
        del mock_agent.entity_memory  # Remove attribute

        context = CommandContext(
            console=mock_console,
            settings=mock_settings,
            agent=mock_agent,
            args=["list"],
        )

        command.execute(context)
        mock_console.print.assert_called()

    def test_stats_with_agent(self, command, mock_console, mock_settings):
        """Test stats with agent context."""
        mock_agent = MagicMock()
        mock_agent.entity_memory = MagicMock()

        context = CommandContext(
            console=mock_console,
            settings=mock_settings,
            agent=mock_agent,
            args=["stats"],
        )

        command.execute(context)
        mock_console.print.assert_called()
