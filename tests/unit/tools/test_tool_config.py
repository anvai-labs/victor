"""Tests for Tool Configuration Externalization.

Phase 7.5: Tests for ToolConfigurator, ToolConfig, and ToolConfigBuilder.
"""

import pytest
from unittest.mock import MagicMock

from victor.framework.tool_config import (
    AirgappedFilter,
    CostTierFilter,
    SecurityFilter,
    ToolCategory,
    ToolConfig,
    ToolConfigBuilder,
    ToolConfigEntry,
    ToolConfigMode,
    ToolConfigResult,
    ToolConfigurator,
    configure_tools,
    configure_tools_from_toolset,
    get_tool_configurator,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator with tools."""
    orchestrator = MagicMock()
    orchestrator.tools = {
        "read": MagicMock(),
        "write": MagicMock(),
        "edit": MagicMock(),
        "shell": MagicMock(),
        "git": MagicMock(),
        "web_search": MagicMock(),
        "web_fetch": MagicMock(),
        "docker": MagicMock(),
    }
    # Set up protocol methods instead of private attribute
    orchestrator._enabled_tools = set()
    orchestrator.get_enabled_tools = MagicMock(side_effect=lambda: orchestrator._enabled_tools)
    orchestrator.set_enabled_tools = MagicMock(
        side_effect=lambda tools: setattr(orchestrator, "_enabled_tools", tools)
    )
    return orchestrator


@pytest.fixture
def mock_toolset():
    """Create a mock ToolSet."""
    toolset = MagicMock()
    toolset.get_tool_names.return_value = ["read", "write", "edit"]
    return toolset


# =============================================================================
# ToolConfigEntry Tests
# =============================================================================


class TestToolConfigEntry:
    """Tests for ToolConfigEntry dataclass."""

    def test_default_values(self):
        """Test default entry values."""
        entry = ToolConfigEntry(name="test_tool")
        assert entry.name == "test_tool"
        assert entry.enabled
        assert entry.category == ToolCategory.CUSTOM
        assert entry.priority == 0
        assert entry.cost_tier == "low"

    def test_custom_values(self):
        """Test custom entry values."""
        entry = ToolConfigEntry(
            name="my_tool",
            enabled=False,
            category=ToolCategory.CORE,
            priority=10,
            cost_tier="high",
        )
        assert entry.name == "my_tool"
        assert not entry.enabled
        assert entry.category == ToolCategory.CORE
        assert entry.priority == 10
        assert entry.cost_tier == "high"

    def test_to_dict(self):
        """Test to_dict conversion."""
        entry = ToolConfigEntry(
            name="test_tool",
            category=ToolCategory.GIT,
        )
        d = entry.to_dict()
        assert d["name"] == "test_tool"
        assert d["category"] == "git"
        assert d["enabled"]


# =============================================================================
# ToolConfigResult Tests
# =============================================================================


class TestToolConfigResult:
    """Tests for ToolConfigResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolConfigResult(
            success=True,
            enabled_tools={"read", "write"},
            disabled_tools={"shell"},
        )
        assert result.success
        assert "read" in result.enabled_tools
        assert "shell" in result.disabled_tools
        assert result.errors == []

    def test_result_with_errors(self):
        """Test result with errors."""
        result = ToolConfigResult(
            success=False,
            enabled_tools=set(),
            disabled_tools=set(),
            errors=["Failed to configure"],
        )
        assert not result.success
        assert "Failed to configure" in result.errors


# =============================================================================
# ToolConfigMode Tests
# =============================================================================


class TestToolConfigMode:
    """Tests for ToolConfigMode enum."""

    def test_mode_values(self):
        """Test mode values."""
        assert ToolConfigMode.REPLACE.value == "replace"
        assert ToolConfigMode.EXTEND.value == "extend"
        assert ToolConfigMode.RESTRICT.value == "restrict"
        assert ToolConfigMode.FILTER.value == "filter"


# =============================================================================
# ToolCategory Tests
# =============================================================================


class TestToolCategory:
    """Tests for ToolCategory enum."""

    def test_category_values(self):
        """Test category values."""
        assert ToolCategory.CORE.value == "core"
        assert ToolCategory.FILESYSTEM.value == "filesystem"
        assert ToolCategory.GIT.value == "git"
        assert ToolCategory.WEB.value == "web"
        assert ToolCategory.CUSTOM.value == "custom"


# =============================================================================
# ToolConfigurator Tests
# =============================================================================


class TestToolConfigurator:
    """Tests for ToolConfigurator class."""

    def test_create_configurator(self):
        """Test configurator creation."""
        configurator = ToolConfigurator()
        assert configurator is not None

    def test_get_tool_configurator(self):
        """Test factory function."""
        configurator = get_tool_configurator()
        assert isinstance(configurator, ToolConfigurator)

    def test_configure_from_toolset(self, mock_orchestrator, mock_toolset):
        """Test configuration from toolset."""
        configurator = ToolConfigurator()
        result = configurator.configure_from_toolset(
            mock_orchestrator,
            mock_toolset,
        )

        assert result.success
        assert "read" in result.enabled_tools
        assert "write" in result.enabled_tools
        assert "edit" in result.enabled_tools

    def test_configure_replace_mode(self, mock_orchestrator):
        """Test replace configuration mode."""
        configurator = ToolConfigurator()
        result = configurator.configure(
            mock_orchestrator,
            ["read", "write"],
            mode=ToolConfigMode.REPLACE,
        )

        assert result.success
        assert result.enabled_tools == {"read", "write"}

    def test_configure_extend_mode(self, mock_orchestrator):
        """Test extend configuration mode."""
        # Set initial tools using protocol method
        mock_orchestrator._enabled_tools = {"read"}

        configurator = ToolConfigurator()
        result = configurator.configure(
            mock_orchestrator,
            ["write"],
            mode=ToolConfigMode.EXTEND,
        )

        assert result.success
        assert "read" in result.enabled_tools
        assert "write" in result.enabled_tools

    def test_configure_restrict_mode(self, mock_orchestrator):
        """Test restrict configuration mode."""
        # Set initial tools using protocol method
        mock_orchestrator._enabled_tools = {"read", "write", "edit", "shell"}

        configurator = ToolConfigurator()
        result = configurator.configure(
            mock_orchestrator,
            ["read", "write"],
            mode=ToolConfigMode.RESTRICT,
        )

        assert result.success
        assert result.enabled_tools == {"read", "write"}
        assert "edit" not in result.enabled_tools
        assert "shell" not in result.enabled_tools

    def test_configure_with_invalid_tools(self, mock_orchestrator):
        """Test configuration with invalid tools."""
        configurator = ToolConfigurator()
        result = configurator.configure(
            mock_orchestrator,
            ["read", "nonexistent_tool"],
            mode=ToolConfigMode.REPLACE,
        )

        assert result.success
        assert "read" in result.enabled_tools
        assert "nonexistent_tool" not in result.enabled_tools
        assert len(result.warnings) > 0

    def test_configure_sets_enabled_tools(self, mock_orchestrator):
        """Test that configuration applies enabled tools correctly."""
        configurator = ToolConfigurator()
        result = configurator.configure(
            mock_orchestrator,
            ["read", "write"],
            mode=ToolConfigMode.REPLACE,
        )

        # Verify tools were configured correctly
        assert result.success
        assert "read" in result.enabled_tools
        assert "write" in result.enabled_tools


class TestToolConfiguratorFilters:
    """Tests for ToolConfigurator filter functionality."""

    def test_add_filter(self, mock_orchestrator):
        """Test adding a filter."""
        configurator = ToolConfigurator()
        mock_filter = MagicMock()
        mock_filter.filter.return_value = {"read"}

        configurator.add_filter(mock_filter)
        configurator.configure(
            mock_orchestrator,
            ["read", "write"],
            mode=ToolConfigMode.FILTER,
        )

        mock_filter.filter.assert_called()

    def test_remove_filter(self):
        """Test removing a filter."""
        configurator = ToolConfigurator()
        mock_filter = MagicMock()

        configurator.add_filter(mock_filter)
        configurator.remove_filter(mock_filter)

        assert mock_filter not in configurator._filters


class TestToolConfiguratorHooks:
    """Tests for ToolConfigurator hook functionality."""

    def test_pre_configure_hook(self, mock_orchestrator):
        """Test pre-configure hook."""
        configurator = ToolConfigurator()
        hook_called = []

        def my_hook(orch, tools, mode):
            hook_called.append((orch, tools, mode))

        configurator.add_hook("pre_configure", my_hook)
        configurator.configure(mock_orchestrator, ["read"], ToolConfigMode.REPLACE)

        assert len(hook_called) == 1
        assert hook_called[0][0] is mock_orchestrator

    def test_post_configure_hook(self, mock_orchestrator):
        """Test post-configure hook."""
        configurator = ToolConfigurator()
        hook_called = []

        def my_hook(orch, enabled, disabled):
            hook_called.append((orch, enabled, disabled))

        configurator.add_hook("post_configure", my_hook)
        configurator.configure(mock_orchestrator, ["read"], ToolConfigMode.REPLACE)

        assert len(hook_called) == 1

    def test_hook_unsubscribe(self, mock_orchestrator):
        """Test hook unsubscribe."""
        configurator = ToolConfigurator()
        hook_called = []

        def my_hook(orch, tools, mode):
            hook_called.append(True)

        unsubscribe = configurator.add_hook("pre_configure", my_hook)
        unsubscribe()

        configurator.configure(mock_orchestrator, ["read"], ToolConfigMode.REPLACE)
        assert len(hook_called) == 0


# =============================================================================
# ToolConfig Tests
# =============================================================================


class TestToolConfig:
    """Tests for ToolConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ToolConfig()
        assert config.enabled_tools == set()
        assert config.disabled_tools == set()
        assert config.mode == ToolConfigMode.REPLACE

    def test_custom_config(self):
        """Test custom configuration."""
        config = ToolConfig(
            enabled_tools={"read", "write"},
            disabled_tools={"shell"},
            mode=ToolConfigMode.RESTRICT,
        )
        assert config.enabled_tools == {"read", "write"}
        assert config.disabled_tools == {"shell"}
        assert config.mode == ToolConfigMode.RESTRICT

    def test_with_enabled(self):
        """Test with_enabled creates new config."""
        config = ToolConfig(enabled_tools={"read"})
        new_config = config.with_enabled("write", "edit")

        assert "write" in new_config.enabled_tools
        assert "edit" in new_config.enabled_tools
        # Original unchanged
        assert "write" not in config.enabled_tools

    def test_with_disabled(self):
        """Test with_disabled creates new config."""
        config = ToolConfig(disabled_tools={"shell"})
        new_config = config.with_disabled("rm", "docker")

        assert "rm" in new_config.disabled_tools
        assert "docker" in new_config.disabled_tools
        # Original unchanged
        assert "rm" not in config.disabled_tools


# =============================================================================
# ToolConfigBuilder Tests
# =============================================================================


class TestToolConfigBuilder:
    """Tests for ToolConfigBuilder class."""

    def test_builder_creation(self):
        """Test builder creation."""
        builder = ToolConfigBuilder()
        assert builder is not None

    def test_mode_chain(self):
        """Test mode chaining."""
        builder = ToolConfigBuilder().mode(ToolConfigMode.EXTEND)
        config = builder.build()
        assert config.mode == ToolConfigMode.EXTEND

    def test_enable_tools_chain(self):
        """Test enable_tools chaining."""
        builder = ToolConfigBuilder().enable_tools("read", "write")
        config = builder.build()
        assert "read" in config.enabled_tools
        assert "write" in config.enabled_tools

    def test_disable_tools_chain(self):
        """Test disable_tools chaining."""
        builder = ToolConfigBuilder().disable_tools("shell", "rm")
        config = builder.build()
        assert "shell" in config.disabled_tools
        assert "rm" in config.disabled_tools

    def test_enable_category_chain(self):
        """Test enable_category chaining."""
        builder = ToolConfigBuilder().enable_category(ToolCategory.CORE)
        config = builder.build()
        assert ToolCategory.CORE in config.categories
        # Core tools should be enabled
        assert "read" in config.enabled_tools

    def test_disable_category_chain(self):
        """Test disable_category chaining."""
        builder = ToolConfigBuilder().disable_category(ToolCategory.WEB)
        config = builder.build()
        assert "web_search" in config.disabled_tools
        assert "web_fetch" in config.disabled_tools

    def test_metadata_chain(self):
        """Test metadata chaining."""
        builder = ToolConfigBuilder().metadata("key", "value")
        config = builder.build()
        assert config.metadata["key"] == "value"

    def test_full_chain(self):
        """Test full chained configuration."""
        config = (
            ToolConfigBuilder()
            .mode(ToolConfigMode.RESTRICT)
            .enable_category(ToolCategory.CORE)
            .enable_category(ToolCategory.GIT)
            .enable_tools("custom_tool")
            .disable_tools("shell")
            .metadata("purpose", "testing")
            .build()
        )

        assert config.mode == ToolConfigMode.RESTRICT
        assert ToolCategory.CORE in config.categories
        assert ToolCategory.GIT in config.categories
        assert "custom_tool" in config.enabled_tools
        assert "shell" in config.disabled_tools
        assert config.metadata["purpose"] == "testing"


# =============================================================================
# Apply Config Tests
# =============================================================================


class TestApplyConfig:
    """Tests for applying ToolConfig to orchestrator."""

    def test_apply_config(self, mock_orchestrator):
        """Test applying config to orchestrator."""
        config = ToolConfig(
            enabled_tools={"read", "write"},
            disabled_tools={"shell"},
            mode=ToolConfigMode.REPLACE,
        )

        configurator = ToolConfigurator()
        result = configurator.apply_config(mock_orchestrator, config)

        assert result.success
        assert "read" in result.enabled_tools
        assert "write" in result.enabled_tools
        assert "shell" not in result.enabled_tools


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_configure_tools_from_toolset(self, mock_orchestrator, mock_toolset):
        """Test configure_tools_from_toolset function."""
        result = configure_tools_from_toolset(mock_orchestrator, mock_toolset)
        assert result.success
        assert "read" in result.enabled_tools

    def test_configure_tools(self, mock_orchestrator):
        """Test configure_tools function."""
        result = configure_tools(
            mock_orchestrator,
            ["read", "write"],
            ToolConfigMode.REPLACE,
        )
        assert result.success
        assert result.enabled_tools == {"read", "write"}


# =============================================================================
# Built-in Filters Tests
# =============================================================================


class TestAirgappedFilter:
    """Tests for AirgappedFilter."""

    def test_removes_canonical_http_and_alias(self):
        assert AirgappedFilter().filter({"read", "http", "http_request"}, {}) == {"read"}

    def test_removes_network_tools(self):
        """Test that network tools are removed."""
        filter = AirgappedFilter()
        available = {"read", "write", "web_search", "web_fetch", "slack"}
        filtered = filter.filter(available, {})

        assert "read" in filtered
        assert "write" in filtered
        assert "web_search" not in filtered
        assert "web_fetch" not in filtered
        assert "slack" not in filtered


class TestCostTierFilter:
    """Tests for CostTierFilter."""

    def test_filter_creation(self):
        """Test filter creation."""
        filter = CostTierFilter(max_tier="low")
        assert filter.max_tier == "low"

    def test_filter_returns_all(self):
        """Test filter returns all tools (placeholder behavior)."""
        filter = CostTierFilter(max_tier="medium")
        available = {"read", "write", "web_search"}
        filtered = filter.filter(available, {})
        assert filtered == available


class TestSecurityFilter:
    """Tests for SecurityFilter."""

    def test_removes_dangerous_tools(self):
        """Test that dangerous tools are removed."""
        filter = SecurityFilter(allow_dangerous=False)
        available = {"read", "write", "shell", "rm", "database"}
        filtered = filter.filter(available, {})

        assert "read" in filtered
        assert "write" in filtered
        assert "shell" not in filtered
        assert "rm" not in filtered
        assert "database" not in filtered

    def test_allows_dangerous_when_enabled(self):
        """Test that dangerous tools are allowed when enabled."""
        filter = SecurityFilter(allow_dangerous=True)
        available = {"read", "write", "shell", "rm"}
        filtered = filter.filter(available, {})

        assert filtered == available


# =============================================================================
# Integration with ToolConfigurator Tests
# =============================================================================


class TestFilterIntegration:
    """Tests for filter integration with ToolConfigurator."""

    def test_airgapped_filter_integration(self, mock_orchestrator):
        """Test airgapped filter with configurator."""
        configurator = ToolConfigurator()
        configurator.add_filter(AirgappedFilter())

        result = configurator.configure(
            mock_orchestrator,
            ["read", "write", "web_search", "web_fetch"],
            mode=ToolConfigMode.FILTER,
        )

        assert result.success
        assert "read" in result.enabled_tools
        assert "write" in result.enabled_tools
        assert "web_search" not in result.enabled_tools
        assert "web_fetch" not in result.enabled_tools

    def test_security_filter_integration(self, mock_orchestrator):
        """Test security filter with configurator."""
        configurator = ToolConfigurator()
        configurator.add_filter(SecurityFilter(allow_dangerous=False))

        result = configurator.configure(
            mock_orchestrator,
            ["read", "write", "shell"],
            mode=ToolConfigMode.FILTER,
        )

        assert result.success
        assert "read" in result.enabled_tools
        assert "write" in result.enabled_tools
        assert "shell" not in result.enabled_tools
