"""Unit tests for agent creation strategies.

Tests the Strategy pattern implementation for agent creation across
different modes (CLI, one-shot).

Test Coverage:
- AgentCreationContext dataclass
- FrameworkStrategy creation
- LegacyStrategy creation
- AgentCreationFactory
- Edge cases and error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.creation_strategies import (
    AgentCreationStrategy,
    FrameworkStrategy,
    LegacyStrategy,
    AgentCreationFactory,
    AgentCreationContext,
)


class TestAgentCreationContext:
    """Test AgentCreationContext dataclass."""

    def test_init_with_minimal_params(self):
        """Test initialization with minimal required parameters."""
        from victor.config.settings import Settings

        settings = MagicMock(spec=Settings)
        context = AgentCreationContext(
            settings=settings,
            profile="default",
        )

        assert context.settings is settings
        assert context.profile == "default"
        assert context.provider is None
        assert context.model is None
        assert context.thinking is False
        assert context.vertical is None
        assert context.enable_observability is True
        assert context.tool_budget is None
        assert context.max_iterations is None

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        from victor.config.settings import Settings

        settings = MagicMock(spec=Settings)
        vertical = MagicMock()

        context = AgentCreationContext(
            settings=settings,
            profile="test-profile",
            provider="anthropic",
            model="claude-3-5-sonnet",
            thinking=True,
            vertical=vertical,
            enable_observability=False,
            session_id="test-session-123",
            tool_budget=100,
            max_iterations=50,
            mode="build",
            metadata={"key": "value"},
        )

        assert context.settings is settings
        assert context.profile == "test-profile"
        assert context.provider == "anthropic"
        assert context.model == "claude-3-5-sonnet"
        assert context.thinking is True
        assert context.vertical is vertical
        assert context.enable_observability is False
        assert context.session_id == "test-session-123"
        assert context.tool_budget == 100
        assert context.max_iterations == 50
        assert context.mode == "build"
        assert context.metadata == {"key": "value"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from victor.config.settings import Settings

        settings = MagicMock(spec=Settings)
        vertical = MagicMock(__name__="TestVertical")

        context = AgentCreationContext(
            settings=settings,
            profile="test",
            provider="test-provider",
            model="test-model",
            vertical=vertical,
        )

        result = context.to_dict()

        assert result["profile"] == "test"
        assert result["provider"] == "test-provider"
        assert result["model"] == "test-model"
        assert result["vertical"] == "TestVertical"


class TestFrameworkStrategy:
    """Test FrameworkStrategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Return FrameworkStrategy instance."""
        return FrameworkStrategy()

    @pytest.fixture
    def mock_context(self):
        """Return mock AgentCreationContext."""
        settings = MagicMock()
        settings.provider.default_temperature = 0.7
        settings.provider.default_max_tokens = 4096

        return AgentCreationContext(
            settings=settings,
            profile="test-profile",
            provider="anthropic",
            model="claude-3-5-sonnet",
            thinking=False,
        )

    @pytest.mark.asyncio
    async def test_create_agent_success(self, strategy, mock_context):
        """Test successful agent creation via AgentFactory."""
        mock_agent = MagicMock()
        mock_agent.observability = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            result = await strategy.create_agent(mock_context)

        assert result is mock_agent
        mock_factory.create.assert_called_once_with()
        mock_agent.observability.on_session_start.assert_called_once_with(
            {"mode": "unknown", "vertical": None, "thinking": False}
        )

    @pytest.mark.asyncio
    async def test_create_agent_applies_overrides(self, strategy, mock_context):
        """Test that mode override still applies on the canonical path."""
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.observability = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_context.tool_budget = 100
        mock_context.max_iterations = 50
        mock_context.mode = "plan"

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.agent.mode_controller.get_mode_controller") as mock_get_controller:
                mock_controller = MagicMock()
                mock_get_controller.return_value = mock_controller

                result = await strategy.create_agent(mock_context)

        assert result is mock_agent
        mock_factory.create.assert_called_once_with()
        mock_controller.switch_mode.assert_called_once()
        mock_agent.unified_tracker.set_tool_budget.assert_not_called()
        mock_agent.unified_tracker.set_max_iterations.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_agent_with_mode_override(self, strategy, mock_context):
        """Test that mode override is applied."""
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.observability = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        mock_context.mode = "plan"

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.agent.mode_controller.get_mode_controller") as mock_get_controller:
                mock_controller = MagicMock()
                mock_get_controller.return_value = mock_controller

                await strategy.create_agent(mock_context)

                mock_controller.switch_mode.assert_called_once()

    def test_supports_observability(self, strategy):
        """Test that FrameworkStrategy supports observability."""
        assert strategy.supports_observability() is True

    def test_supports_verticals(self, strategy):
        """Test that FrameworkStrategy supports verticals."""
        assert strategy.supports_verticals() is True


class TestLegacyStrategy:
    """Test LegacyStrategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Return LegacyStrategy instance."""
        return LegacyStrategy()

    @pytest.fixture
    def mock_context(self):
        """Return mock AgentCreationContext."""
        from victor.config.settings import Settings

        settings = MagicMock(spec=Settings)

        return AgentCreationContext(
            settings=settings,
            profile="test-profile",
            provider="anthropic",
            model="claude-3-5-sonnet",
            thinking=False,
        )

    @pytest.mark.asyncio
    async def test_create_agent_success(self, strategy, mock_context):
        """Test successful agent creation with direct from_settings."""
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        with patch("victor.agent.orchestrator.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator_class.from_settings = AsyncMock(return_value=mock_agent)

            result = await strategy.create_agent(mock_context)

            assert result is mock_agent
            mock_orchestrator_class.from_settings.assert_called_once_with(
                mock_context.settings,
                mock_context.profile,
                thinking=False,
            )

    def test_supports_observability(self, strategy):
        """Test that LegacyStrategy does NOT support observability."""
        assert strategy.supports_observability() is False

    def test_supports_verticals(self, strategy):
        """Test that LegacyStrategy does NOT support verticals."""
        assert strategy.supports_verticals() is False


class TestAgentCreationFactory:
    """Test AgentCreationFactory."""

    @pytest.fixture
    def factory(self):
        """Return factory with default FrameworkStrategy."""
        return AgentCreationFactory()

    @pytest.fixture
    def mock_context(self):
        """Return mock AgentCreationContext."""
        from victor.config.settings import Settings

        settings = MagicMock(spec=Settings)

        return AgentCreationContext(
            settings=settings,
            profile="test-profile",
        )

    @pytest.mark.asyncio
    async def test_create_agent_with_default_strategy(self, factory, mock_context):
        """Test agent creation with default (Framework) strategy."""
        mock_agent = MagicMock()
        mock_agent.observability = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            result = await factory.create_agent(mock_context)

        assert result is mock_agent

    @pytest.mark.asyncio
    async def test_create_agent_with_explicit_strategy(self, factory, mock_context):
        """Test agent creation with explicit strategy override."""
        strategy = LegacyStrategy()
        mock_agent = MagicMock()

        with patch("victor.agent.orchestrator.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator_class.from_settings = AsyncMock(return_value=mock_agent)

            result = await factory.create_agent(mock_context, strategy=strategy)

            assert result is mock_agent

    def test_init_with_custom_default_strategy(self):
        """Test factory initialization with custom default strategy."""
        custom_strategy = LegacyStrategy()
        factory = AgentCreationFactory(default_strategy=custom_strategy)

        assert factory._default_strategy is custom_strategy

    @pytest.mark.asyncio
    async def test_create_context_from_cli_args(self, factory):
        """Test AgentCreationContext creation from CLI arguments."""
        from victor.config.settings import Settings

        settings = MagicMock(spec=Settings)

        context = AgentCreationFactory.create_context_from_cli_args(
            settings=settings,
            profile="test-profile",
            provider="anthropic",
            model="claude-3-5-sonnet",
            thinking=True,
            mode="plan",
            tool_budget=100,
            max_iterations=50,
            vertical="benchmark",  # Use benchmark which is a built-in vertical
            enable_observability=False,
            session_id="test-123",
            custom_key="custom_value",
        )

        assert context.settings is settings
        assert context.profile == "test-profile"
        assert context.provider == "anthropic"
        assert context.model == "claude-3-5-sonnet"
        assert context.thinking is True
        assert context.mode == "plan"
        assert context.tool_budget == 100
        assert context.max_iterations == 50
        assert context.vertical is not None  # benchmark is a built-in vertical
        assert context.enable_observability is False
        assert context.session_id == "test-123"
        assert context.metadata == {"custom_key": "custom_value"}

    @pytest.mark.asyncio
    async def test_create_context_from_cli_args_with_none_values(self, factory):
        """Test context creation with None optional values."""
        from victor.config.settings import Settings

        settings = MagicMock(spec=Settings)

        context = AgentCreationFactory.create_context_from_cli_args(
            settings=settings,
            profile="test-profile",
        )

        assert context.settings is settings
        assert context.profile == "test-profile"
        assert context.provider is None
        assert context.model is None
        assert context.thinking is False
        assert context.mode is None
        assert context.tool_budget is None
        assert context.max_iterations is None
        assert context.vertical is None
        assert context.enable_observability is True
        assert context.session_id is None
        assert context.metadata == {}


class TestStrategyInterface:
    """Test that AgentCreationStrategy is a proper abstract interface."""

    def test_cannot_instantiate_abstract_strategy(self):
        """Test that abstract strategy cannot be instantiated."""
        with pytest.raises(TypeError):
            AgentCreationStrategy()

    def test_strategy_requires_create_agent(self):
        """Test that strategy requires create_agent method."""
        assert hasattr(AgentCreationStrategy, "create_agent")
        assert hasattr(AgentCreationStrategy, "supports_observability")
        assert hasattr(AgentCreationStrategy, "supports_verticals")


class TestStrategyLogging:
    """Test logging behavior of strategies."""

    @pytest.mark.asyncio
    async def test_framework_strategy_logs_creation(self, caplog):
        """Test that FrameworkStrategy logs agent creation."""
        import logging

        caplog.set_level(logging.INFO)

        factory = AgentCreationFactory()
        mock_context = MagicMock()
        mock_context.settings = MagicMock()
        mock_context.profile = "test"
        mock_context.vertical = None  # No vertical to avoid __name__ error

        mock_agent = MagicMock()
        mock_agent.observability = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            await factory.create_agent(mock_context)

        # Verify logging occurred from the factory
        assert "Creating agent" in caplog.text
        assert "FrameworkStrategy" in caplog.text
