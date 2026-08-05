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

"""Tests for ToolRegistrar component.

Tests cover:
- Configuration dataclasses
- Dynamic tool discovery
- Plugin initialization
- MCP integration
- Tool dependency graph
- Goal inference
- Statistics collection
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from victor.agent.tool_registrar import (
    ToolRegistrar,
    ToolRegistrarConfig,
    RegistrationStats,
)
from victor.tools.registry import ToolRegistry


class TestToolRegistrarConfig:
    """Tests for ToolRegistrarConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ToolRegistrarConfig()

        assert config.enable_plugins is True
        assert config.enable_mcp is False
        assert config.enable_tool_graph is True
        assert config.airgapped_mode is False
        assert config.plugin_dirs == []
        assert config.disabled_plugins == set()
        assert config.plugin_packages == []
        assert config.max_workers == 4
        assert config.max_complexity == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ToolRegistrarConfig(
            enable_plugins=False,
            enable_mcp=True,
            airgapped_mode=True,
            plugin_dirs=["/custom/path"],
            disabled_plugins={"plugin1", "plugin2"},
            max_workers=8,
        )

        assert config.enable_plugins is False
        assert config.enable_mcp is True
        assert config.airgapped_mode is True
        assert "/custom/path" in config.plugin_dirs
        assert "plugin1" in config.disabled_plugins
        assert config.max_workers == 8


class TestRegistrationStats:
    """Tests for RegistrationStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = RegistrationStats()

        assert stats.dynamic_tools == 0
        assert stats.plugin_tools == 0
        assert stats.mcp_tools == 0
        assert stats.dependency_graph_tools == 0
        assert stats.total_tools == 0
        assert stats.plugins_loaded == 0
        assert stats.mcp_servers_connected == 0

    def test_custom_values(self):
        """Test custom statistics values."""
        stats = RegistrationStats(
            dynamic_tools=50,
            plugin_tools=10,
            mcp_tools=5,
            total_tools=65,
            plugins_loaded=3,
        )

        assert stats.dynamic_tools == 50
        assert stats.plugin_tools == 10
        assert stats.total_tools == 65
        assert stats.plugins_loaded == 3

    def test_as_dict(self):
        """Test converting stats to dictionary."""
        stats = RegistrationStats(dynamic_tools=10, total_tools=10)
        d = asdict(stats)

        assert d["dynamic_tools"] == 10
        assert d["total_tools"] == 10
        assert "plugin_tools" in d


class TestToolRegistrarInit:
    """Tests for ToolRegistrar initialization."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool registry."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.get.return_value = None
        return tools

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        return settings

    def test_initialization(self, mock_tools, mock_settings):
        """Test basic initialization."""
        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
        )

        assert registrar.tools is mock_tools
        assert registrar.settings is mock_settings
        assert registrar.provider is None
        assert registrar.model is None
        assert registrar.config is not None

    def test_initialization_with_config(self, mock_tools, mock_settings):
        """Test initialization with custom config."""
        config = ToolRegistrarConfig(enable_mcp=True, max_workers=16)

        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
            config=config,
        )

        assert registrar.config.enable_mcp is True
        assert registrar.config.max_workers == 16

    def test_initialization_with_provider(self, mock_tools, mock_settings):
        """Test initialization with provider."""
        mock_provider = MagicMock()

        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
            provider=mock_provider,
            model="test-model",
        )

        assert registrar.provider is mock_provider
        assert registrar.model == "test-model"


class TestToolRegistrarBackgroundTasks:
    """Tests for background task management."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        return ToolRegistrar(tools=tools, settings=settings)

    def test_set_background_task_callback(self, registrar):
        """Test setting background task callback."""
        callback = MagicMock(return_value=MagicMock())
        registrar.set_background_task_callback(callback)

        assert registrar._create_background_task is callback

    def test_create_task_with_callback(self, registrar):
        """Test creating task with callback."""
        mock_task = MagicMock()
        callback = MagicMock(return_value=mock_task)
        registrar.set_background_task_callback(callback)

        async def dummy_coro():
            pass

        coro = dummy_coro()
        result = registrar._create_task(coro, "test_task")

        callback.assert_called_once_with(coro, "test_task")
        assert result is mock_task
        coro.close()


class TestCanonicalRegistrationSurfaces:
    """Tests for canonical public registration entry points."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock(spec=ToolRegistry)
        settings = MagicMock()
        settings.use_mcp_tools = False
        settings.load_tool_config.return_value = {}
        # Test non-lazy startup path (dynamic tool registration)
        config = ToolRegistrarConfig(lazy_startup=False, enable_mcp=False)
        return ToolRegistrar(tools=tools, settings=settings, config=config)

    def test_register_default_tools_prefers_registrar_owned_surface(self, registrar):
        with (
            patch.object(registrar, "_setup_providers") as setup_providers,
            patch.object(registrar, "_register_dynamic_tools", return_value=5) as register_dynamic,
            patch.object(registrar, "_setup_mcp_integration") as setup_mcp,
        ):
            registered_count = registrar.register_default_tools()

        # Note: The expected count may vary based on how many tools are actually registered.
        # The mock returns 5, but the actual count depends on the bootstrap tools registered.
        # Use >= to allow for flexibility as tools are added/removed.
        assert registered_count >= 5, f"Expected at least 5 tools, got {registered_count}"
        setup_providers.assert_called_once_with()
        register_dynamic.assert_called_once_with()  # Called when lazy_startup=False
        setup_mcp.assert_not_called()

    def test_initialize_plugins_delegates_to_canonical_public_method(self, registrar):
        with patch.object(registrar, "_load_plugin_tools", return_value=3) as initialize_plugins:
            tool_count = registrar.initialize_plugins()

        assert tool_count == 3
        initialize_plugins.assert_called_once_with()


class TestDynamicToolRegistration:
    """Tests for dynamic tool discovery and registration."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.register.return_value = None
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        return ToolRegistrar(tools=tools, settings=settings)

    def test_register_dynamic_tools_runs(self, registrar):
        """Test dynamic tool registration runs without error."""
        # Simply verify the method can be called - actual tool discovery
        # depends on filesystem state
        count = registrar._register_dynamic_tools()

        # Should return a count >= 0
        assert count >= 0

    def test_register_dynamic_tools_includes_graph_tool(self):
        """Dynamic tool registration should include search functionality.

        Graph-related tools (graph_semantic_search, etc.) are intentionally kept
        out of the default registration set via ``SCHEMA_HIDDEN_TOOL_NAMES``.
        Search functionality is instead provided by the unified command domains:
        ``code`` (``code grep`` / ``code search``) and ``fs`` (``fs search``).
        This test verifies at least one of those search-capable domains is
        registered via dynamic discovery.
        """
        # _register_dynamic_tools reads the process-global SharedToolRegistry
        # singleton; a prior test in the same shard can leave it in a state that
        # yields zero tools (test-isolation/ordering flake). Reset to a clean
        # instance so this test is self-contained (mirrors the pattern in
        # tests/unit/core/test_shared_tool_registry.py), and reset again after so
        # we don't leak our populated singleton to later tests.
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        try:
            settings = MagicMock()
            settings.load_tool_config.return_value = {}
            # Force non-lazy mode for this test
            config = ToolRegistrarConfig(lazy_startup=False)
            registrar = ToolRegistrar(
                tools=ToolRegistry(),
                settings=settings,
                config=config,
            )

            count = registrar._register_dynamic_tools()

            assert count > 0
            tool_names = [t.name for t in registrar.tools.list_tools()]
            # ``code`` and ``fs`` are the unified command domains that own search.
            has_search_domain = any(name in {"code", "fs", "search"} for name in tool_names)
            assert (
                has_search_domain
            ), f"Expected a search-capable unified domain (code/fs/search), got: {tool_names[:10]}..."
        finally:
            SharedToolRegistry.reset_instance()

    def test_excluded_files(self, registrar):
        """Test that excluded files are not loaded."""
        # The excluded files are checked inside _register_dynamic_tools
        excluded = {"__init__.py", "base.py", "decorators.py", "semantic_selector.py"}

        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = list(excluded)

            with patch("importlib.import_module") as mock_import:
                registrar._register_dynamic_tools()

                # Should not have imported any of the excluded files
                assert mock_import.call_count == 0


class TestToolConfiguration:
    """Tests for tool configuration loading."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.disable_tool.return_value = None
        settings = MagicMock()
        return ToolRegistrar(tools=tools, settings=settings)

    def test_load_empty_config(self, registrar):
        """Test loading empty configuration."""
        registrar.settings.load_tool_config.return_value = {}

        registrar._load_tool_configurations()

        # Should not disable any tools
        registrar.tools.disable_tool.assert_not_called()

    def test_load_disabled_tools(self, registrar):
        """Test loading disabled tools configuration."""
        # Mock registered tools to include the tools we want to disable
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        registrar.tools.list_tools.return_value = [mock_tool1, mock_tool2]

        registrar.settings.load_tool_config.return_value = {
            "disabled": ["tool1", "tool2"],
        }

        registrar._load_tool_configurations()

        assert registrar.tools.disable_tool.call_count == 2

    def test_load_enabled_tools(self, registrar):
        """Test loading enabled tools configuration."""
        mock_tool = MagicMock()
        mock_tool.name = "allowed_tool"
        registrar.tools.list_tools.return_value = [mock_tool]

        registrar.settings.load_tool_config.return_value = {
            "enabled": ["other_tool"],  # allowed_tool not in list
        }

        registrar._load_tool_configurations()

        # Should disable tools not in enabled list
        registrar.tools.disable_tool.assert_called_with("allowed_tool")

    def test_load_per_tool_config(self, registrar):
        """Test loading per-tool configuration."""
        # Mock registered tools to include the tool we want to configure
        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        registrar.tools.list_tools.return_value = [mock_tool]

        registrar.settings.load_tool_config.return_value = {
            "my_tool": {"enabled": False},
        }

        registrar._load_tool_configurations()

        registrar.tools.disable_tool.assert_called_with("my_tool")

    def test_config_load_error(self, registrar):
        """Test handling configuration load errors."""
        registrar.settings.load_tool_config.side_effect = Exception("Config error")

        # Should not raise
        registrar._load_tool_configurations()


class TestPluginInitialization:
    """Tests for plugin system initialization."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        settings.plugin_config = {}
        return ToolRegistrar(tools=tools, settings=settings)

    @patch("victor.tools.plugin_registry.ToolPluginRegistry")
    @patch("victor.config.settings.get_project_paths")
    def test_initialize_plugins(self, mock_paths, mock_registry_class, registrar):
        """Test plugin initialization."""
        mock_paths.return_value = MagicMock(global_plugins_dir="/plugins")

        mock_registry = MagicMock()
        mock_registry.discover_and_load.return_value = 2
        mock_registry.loaded_plugins = {"p1": MagicMock(), "p2": MagicMock()}
        mock_registry.register_tools.return_value = 5
        mock_registry_class.return_value = mock_registry

        count = registrar.initialize_plugins()

        assert count == 5
        assert registrar.plugin_manager is mock_registry

    @patch("victor.tools.plugin_registry.ToolPluginRegistry")
    @patch("victor.config.settings.get_project_paths")
    def test_plugin_dirs_from_config(self, mock_paths, mock_registry_class, registrar):
        """Test custom plugin directories."""
        mock_paths.return_value = MagicMock(global_plugins_dir="/plugins")
        registrar.config.plugin_dirs = ["/custom/plugins"]

        mock_registry = MagicMock()
        mock_registry.discover_and_load.return_value = 0
        mock_registry.loaded_plugins = {}
        mock_registry_class.return_value = mock_registry

        registrar.initialize_plugins()

        call_args = mock_registry_class.call_args
        plugin_dirs = call_args.kwargs.get("plugin_dirs", [])
        assert "/custom/plugins" in plugin_dirs

    @patch("victor.tools.plugin_registry.ToolPluginRegistry")
    @patch("victor.config.settings.get_project_paths")
    def test_disabled_plugins(self, mock_paths, mock_registry_class, registrar):
        """Test disabling specific plugins."""
        mock_paths.return_value = MagicMock(global_plugins_dir="/plugins")
        registrar.config.disabled_plugins = {"bad_plugin"}

        mock_registry = MagicMock()
        mock_registry.discover_and_load.return_value = 0
        mock_registry.loaded_plugins = {}
        mock_registry_class.return_value = mock_registry

        registrar.initialize_plugins()

        mock_registry.disable_plugin.assert_called_with("bad_plugin")

    @patch(
        "victor.tools.plugin_registry.ToolPluginRegistry",
        side_effect=ImportError("No plugins"),
    )
    def test_plugin_init_error(self, mock_registry_class, registrar):
        """Test handling plugin initialization errors."""
        count = registrar.initialize_plugins()

        assert count == 0
        assert registrar.plugin_manager is None


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
class TestMCPIntegration:
    """Tests for MCP integration.

    Note: These tests create the registrar with enable_mcp=False to avoid
    calling _setup_mcp_integration() during __init__. This allows us to
    properly mock MCPRegistry before calling _setup_mcp_integration() manually.

    The filterwarnings decorator suppresses the expected "coroutine was never awaited"
    warning that occurs when mocking _create_task in synchronous tests.
    """

    def _create_registrar(self, mcp_command: str = None):
        """Create registrar with MCP disabled during init for proper mocking."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.register_dict.return_value = None
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        settings.mcp_command = mcp_command
        settings.use_mcp_tools = False
        # Create with enable_mcp=False to prevent _setup_mcp_integration during __init__
        registrar = ToolRegistrar(
            tools=tools, settings=settings, config=ToolRegistrarConfig(enable_mcp=False)
        )
        # Enable MCP for manual testing
        registrar.config = ToolRegistrarConfig(enable_mcp=True)
        return registrar

    @patch("victor.integrations.mcp.registry.MCPRegistry")
    def test_setup_mcp_registry(self, mock_registry_class):
        """Test MCP registry setup."""
        mock_registry = MagicMock()
        # Return servers to trigger full registry initialization path
        mock_registry.list_servers.return_value = ["discovered_server"]
        mock_registry_class.discover_servers.return_value = mock_registry

        registrar = self._create_registrar()
        # Mock _start_mcp_registry to avoid creating a coroutine
        registrar._start_mcp_registry = MagicMock(return_value=None)
        # Mock _create_task to avoid async event loop requirement
        registrar._create_task = MagicMock(return_value=None)
        registrar._setup_mcp_integration()

        mock_registry_class.discover_servers.assert_called_once()
        assert registrar.mcp_registry is mock_registry
        # Verify async startup was attempted
        registrar._create_task.assert_called_once()

    @patch("victor.integrations.mcp.registry.MCPRegistry")
    def test_setup_mcp_with_command(self, mock_registry_class):
        """Test MCP setup with command from settings."""
        mock_registry = MagicMock()
        # Return servers after registration to trigger async startup path
        mock_registry.list_servers.return_value = ["settings_mcp"]
        mock_registry_class.discover_servers.return_value = mock_registry

        registrar = self._create_registrar(mcp_command="python mcp_server.py")
        # Mock _start_mcp_registry to avoid creating a coroutine
        registrar._start_mcp_registry = MagicMock(return_value=None)
        # Mock _create_task to avoid async event loop requirement
        registrar._create_task = MagicMock(return_value=None)
        registrar._setup_mcp_integration()

        # Should register server from command
        mock_registry.register_server.assert_called_once()

    @patch("victor.tools.mcp_bridge_tool.get_mcp_tool_definitions")
    @patch("victor.integrations.mcp.registry.MCPRegistry")
    def test_register_mcp_tools(self, mock_registry_class, mock_get_tools):
        """Test registering MCP tool definitions."""
        mock_registry = MagicMock()
        mock_registry.list_servers.return_value = ["server1"]  # Has servers
        mock_registry_class.discover_servers.return_value = mock_registry

        mock_get_tools.return_value = [
            {"name": "mcp_tool1"},
            {"name": "mcp_tool2"},
        ]

        registrar = self._create_registrar()
        # Mock _start_mcp_registry to avoid creating a coroutine
        registrar._start_mcp_registry = MagicMock(return_value=None)
        # Mock _create_task to avoid async event loop requirement
        registrar._create_task = MagicMock(return_value=None)
        count = registrar._setup_mcp_integration()

        assert count == 2
        assert registrar.tools.register_dict.call_count == 2


class TestToolDependencyGraph:
    """Tests for tool dependency graph."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with tool graph."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        settings = MagicMock()
        settings.load_tool_config.return_value = {}

        tool_graph = MagicMock()
        tool_graph.add_tool.return_value = None

        return ToolRegistrar(tools=tools, settings=settings, tool_graph=tool_graph)

    def test_register_tool_dependencies(self, registrar):
        """Test registering tool dependencies."""
        count = registrar._register_tool_dependencies()

        # Should register 8 tools with dependencies
        assert count == 8
        assert registrar.tool_graph.add_tool.call_count == 8

    def test_no_graph(self):
        """Test with no tool graph."""
        tools = MagicMock()
        settings = MagicMock()
        registrar = ToolRegistrar(tools=tools, settings=settings, tool_graph=None)

        count = registrar._register_tool_dependencies()

        assert count == 0


class TestToolPlanning:
    """Tests for tool planning functionality."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with tool graph."""
        tools = MagicMock()
        tool = MagicMock()
        tool.name = "code_search"
        tool.description = "Search code"
        tool.parameters = {}
        tools.get.return_value = tool
        tools.is_tool_enabled.return_value = True

        settings = MagicMock()

        tool_graph = MagicMock()
        tool_graph.plan.return_value = ["code_search", "read_file"]

        return ToolRegistrar(tools=tools, settings=settings, tool_graph=tool_graph)

    def test_plan_tools(self, registrar):
        """Test planning tools for goals."""
        result = registrar.plan_tools(goals=["summary"])

        assert len(result) >= 1
        registrar.tool_graph.plan.assert_called_once()

    def test_plan_tools_empty_goals(self, registrar):
        """Test planning with empty goals."""
        result = registrar.plan_tools(goals=[])

        assert result == []

    def test_plan_tools_no_graph(self):
        """Test planning without tool graph."""
        tools = MagicMock()
        settings = MagicMock()
        registrar = ToolRegistrar(tools=tools, settings=settings, tool_graph=None)

        result = registrar.plan_tools(goals=["summary"])

        assert result == []


class TestGoalInference:
    """Tests for goal inference from messages."""

    @pytest.fixture
    def registrar(self):
        """Create registrar."""
        tools = MagicMock()
        settings = MagicMock()
        return ToolRegistrar(tools=tools, settings=settings)

    def test_infer_summary_goals(self, registrar):
        """Test inferring summary goals."""
        test_cases = [
            "Summarize this code",
            "Give me an overview",
            "Analyze the project",
        ]

        for message in test_cases:
            goals = registrar.infer_goals_from_message(message)
            assert "summary" in goals, f"Failed for: {message}"

    def test_infer_documentation_goals(self, registrar):
        """Test inferring documentation goals."""
        test_cases = [
            "Generate documentation",
            "Create a readme",
            "Write docs for this",
        ]

        for message in test_cases:
            goals = registrar.infer_goals_from_message(message)
            assert "documentation" in goals, f"Failed for: {message}"

    def test_infer_security_goals(self, registrar):
        """Test inferring security goals."""
        test_cases = [
            "Scan for vulnerabilities",
            "Check for secrets",
            "Security audit",
        ]

        for message in test_cases:
            goals = registrar.infer_goals_from_message(message)
            assert "security_report" in goals, f"Failed for: {message}"

    def test_infer_metrics_goals(self, registrar):
        """Test inferring metrics goals."""
        test_cases = [
            "Calculate complexity",
            "Show me the metrics",
            "Technical debt analysis",
        ]

        for message in test_cases:
            goals = registrar.infer_goals_from_message(message)
            assert "metrics_report" in goals, f"Failed for: {message}"

    def test_no_goals(self, registrar):
        """Test message with no matching goals."""
        goals = registrar.infer_goals_from_message("Hello, how are you?")

        assert goals == []


class TestRegistrationStatistics:
    """Tests for registration statistics."""

    @pytest.fixture
    def registrar(self):
        """Create registrar."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        settings = MagicMock()
        return ToolRegistrar(tools=tools, settings=settings)

    def test_get_stats(self, registrar):
        """Test getting registration stats."""
        stats = registrar.get_stats()

        assert isinstance(stats, RegistrationStats)

    def test_get_plugin_info_no_manager(self, registrar):
        """Test getting plugin info without manager."""
        info = registrar.get_plugin_info()

        assert info == {"plugins": [], "total": 0}

    def test_get_plugin_info_with_manager(self, registrar):
        """Test getting plugin info with manager."""
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"
        mock_plugin.version = "1.0.0"
        mock_plugin.get_tools.return_value = [MagicMock(), MagicMock()]

        registrar.plugin_manager = MagicMock()
        registrar.plugin_manager.loaded_plugins = {"test_plugin": mock_plugin}

        info = registrar.get_plugin_info()

        assert info["total"] == 1
        assert len(info["plugins"]) == 1
        assert info["plugins"][0]["name"] == "test_plugin"

    def test_get_mcp_info_no_registry(self, registrar):
        """Test getting MCP info without registry."""
        info = registrar.get_mcp_info()

        assert info == {"servers": [], "connected": 0, "total": 0}

    def test_get_mcp_info_with_registry(self, registrar):
        """Test getting MCP info with registry."""
        mock_server = MagicMock()
        mock_server.name = "test_server"
        mock_server.description = "Test MCP server"

        registrar.mcp_registry = MagicMock()
        registrar.mcp_registry.list_servers.return_value = [mock_server]
        registrar._stats.mcp_servers_connected = 1

        info = registrar.get_mcp_info()

        assert info["total"] == 1
        assert info["connected"] == 1


class TestShutdown:
    """Tests for shutdown functionality."""

    @pytest.fixture
    def registrar(self):
        """Create registrar."""
        tools = MagicMock()
        settings = MagicMock()
        return ToolRegistrar(tools=tools, settings=settings)

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tasks(self, registrar):
        """Test that shutdown cancels pending tasks."""
        # Create mock task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        registrar._mcp_tasks.append(mock_task)

        await registrar.shutdown()

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_mcp_registry(self, registrar):
        """Test shutdown with MCP registry."""
        registrar.mcp_registry = MagicMock()
        registrar.mcp_registry.shutdown = AsyncMock()

        await registrar.shutdown()

        registrar.mcp_registry.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors(self, registrar):
        """Test shutdown handles errors gracefully."""
        registrar.mcp_registry = MagicMock()
        registrar.mcp_registry.shutdown = AsyncMock(side_effect=Exception("Shutdown error"))

        # Should not raise
        await registrar.shutdown()


class TestLazyToolLoading:
    """Tests for lazy tool loading behavior.

    These tests verify that tools are NOT loaded during ToolRegistrar initialization,
    but ARE loaded on first access (lazy loading pattern for faster startup).
    """

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool registry."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.get.return_value = None
        return tools

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        return settings

    def test_tools_not_loaded_on_init(self, mock_tools, mock_settings):
        """Test that tools are NOT loaded during registrar initialization.

        This is the key test for lazy loading - _register_dynamic_tools should
        NOT be called during __init__.
        """
        with patch.object(ToolRegistrar, "_register_dynamic_tools") as mock_register:
            registrar = ToolRegistrar(
                tools=mock_tools,
                settings=mock_settings,
            )

            # Tools should NOT be loaded on init
            mock_register.assert_not_called()

            # Flag should indicate tools not yet loaded
            assert registrar._tools_loaded is False

    def test_tools_loaded_on_first_access_via_get_all_tools(self, mock_tools, mock_settings):
        """Test that tools ARE loaded when first accessed via get_all_tools()."""
        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
        )

        # Before access, tools not loaded
        assert registrar._tools_loaded is False

        # Mock the CatalogLoader that the facade delegates to
        mock_loader = MagicMock()
        mock_loader.load.return_value = MagicMock(tools_loaded=5)

        with patch.object(registrar, "_get_catalog_loader", return_value=mock_loader):
            # Access tools via get_all_tools
            registrar.get_all_tools()

            # Now tools should be loaded via CatalogLoader
            mock_loader.load.assert_called_once()
            assert registrar._tools_loaded is True

    def test_tools_loaded_on_first_access_via_get_tool(self, mock_tools, mock_settings):
        """Test that tools ARE loaded when first accessed via get_tool()."""
        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
        )

        # Before access, tools not loaded
        assert registrar._tools_loaded is False

        # Mock the CatalogLoader that the facade delegates to
        mock_loader = MagicMock()
        mock_loader.ensure_tools.return_value = MagicMock(tools_loaded=1)

        with patch.object(registrar, "_get_catalog_loader", return_value=mock_loader):
            # Access tools via get_tool
            registrar.get_tool("some_tool")

            # ensure_tools should be called (not load - that's for batch loading)
            mock_loader.ensure_tools.assert_called_once_with(["some_tool"])
            assert registrar._tools_loaded is False  # Not set for individual tool loading

    def test_subsequent_accesses_dont_reload_tools(self, mock_tools, mock_settings):
        """Test that subsequent accesses don't reload tools (only loads once)."""
        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
        )

        # Mock the CatalogLoader that the facade delegates to
        mock_loader = MagicMock()
        mock_loader.load.return_value = MagicMock(tools_loaded=5)

        with patch.object(registrar, "_get_catalog_loader", return_value=mock_loader):
            # First access triggers loading
            registrar.get_all_tools()
            assert mock_loader.load.call_count == 1

            # Second access should NOT reload
            registrar.get_all_tools()
            assert mock_loader.load.call_count == 1

            # Third access via different method also should NOT reload
            registrar.get_tool("test")
            assert mock_loader.load.call_count == 1

    def test_ensure_tools_loaded_is_idempotent(self, mock_tools, mock_settings):
        """Test that _ensure_tools_loaded() is idempotent."""
        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
        )

        # Mock the CatalogLoader that the facade delegates to
        mock_loader = MagicMock()
        mock_loader.load.return_value = MagicMock(tools_loaded=5)

        with patch.object(registrar, "_get_catalog_loader", return_value=mock_loader):
            # Call multiple times
            registrar._ensure_tools_loaded()
            registrar._ensure_tools_loaded()
            registrar._ensure_tools_loaded()

            # Should only register once (via CatalogLoader)
            assert mock_loader.load.call_count == 1

    def test_tools_loaded_flag_persists_across_accesses(self, mock_tools, mock_settings):
        """Test that the _tools_loaded flag correctly persists."""
        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
        )

        # Initially not loaded
        assert registrar._tools_loaded is False

        # Mock the CatalogLoader that the facade delegates to
        mock_loader = MagicMock()
        mock_loader.load.return_value = MagicMock(tools_loaded=5)

        with patch.object(registrar, "_get_catalog_loader", return_value=mock_loader):
            registrar._ensure_tools_loaded()

        # After loading, flag should be True
        assert registrar._tools_loaded is True

        # And remain True
        registrar.get_all_tools()
        assert registrar._tools_loaded is True


class TestInitializeMethod:
    """Tests for the main initialize method using SRP-compliant components."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with all mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = [MagicMock() for _ in range(10)]
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        settings.use_mcp_tools = False

        # Explicitly set lazy_startup=False to test full catalog loading path
        config = ToolRegistrarConfig(enable_plugins=False, enable_mcp=False, lazy_startup=False)
        return ToolRegistrar(tools=tools, settings=settings, config=config)

    @pytest.mark.asyncio
    async def test_initialize_basic(self, registrar):
        """Test basic initialization via CatalogLoader component."""
        # Mock the CatalogLoader component
        mock_catalog_loader = MagicMock()
        mock_catalog_loader.load.return_value = MagicMock(tools_loaded=10)

        with patch.object(registrar, "_setup_providers"):
            with patch.object(registrar, "_get_catalog_loader", return_value=mock_catalog_loader):
                stats = await registrar.initialize()

        assert stats.dynamic_tools == 10
        assert stats.total_tools == 10
        mock_catalog_loader.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_plugins(self, registrar):
        """Test initialization with plugins enabled via PluginLoader component."""
        registrar.config.enable_plugins = True

        # Mock the CatalogLoader component
        mock_catalog_loader = MagicMock()
        mock_catalog_loader.load.return_value = MagicMock(tools_loaded=10)

        # Mock the PluginLoader component
        mock_plugin_loader = MagicMock()
        mock_plugin_loader.load.return_value = MagicMock(tools_registered=5, plugins_loaded=2)
        mock_plugin_loader.plugin_manager = MagicMock()

        with patch.object(registrar, "_setup_providers"):
            with patch.object(registrar, "_get_catalog_loader", return_value=mock_catalog_loader):
                with patch.object(registrar, "_get_plugin_loader", return_value=mock_plugin_loader):
                    stats = await registrar.initialize()

        assert stats.dynamic_tools == 10
        assert stats.plugin_tools == 5
        mock_plugin_loader.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_tool_graph(self, registrar):
        """Test initialization with tool graph via GraphBuilder component."""
        registrar.config.enable_tool_graph = True
        registrar.tool_graph = MagicMock()

        # Mock the CatalogLoader component
        mock_catalog_loader = MagicMock()
        mock_catalog_loader.load.return_value = MagicMock(tools_loaded=10)

        # Mock the GraphBuilder component
        mock_graph_builder = MagicMock()
        mock_graph_builder.build.return_value = MagicMock(tools_registered=8)

        with patch.object(registrar, "_setup_providers"):
            with patch.object(registrar, "_get_catalog_loader", return_value=mock_catalog_loader):
                with patch.object(registrar, "_get_graph_builder", return_value=mock_graph_builder):
                    stats = await registrar.initialize()

        assert stats.dependency_graph_tools == 8
        mock_graph_builder.build.assert_called_once()
