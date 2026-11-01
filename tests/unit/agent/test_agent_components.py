"""Tests for Agent Component Decomposition - Builder/Session/Bridge pattern.

Phase 7.4: Tests for AgentBuilder, AgentSession, and AgentBridge components.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.agent_components import (
    AgentBridge,
    AgentBuilder,
    AgentBuildOptions,
    AgentSession,
    BridgeConfiguration,
    BuilderPreset,
    SessionContext,
    SessionLifecycleHooks,
    SessionMetrics,
    SessionState,
    create_bridge,
    create_builder,
    create_session,
)
from victor.agent.config import UnifiedAgentConfig
from victor.framework.config import AgentConfig
from victor.core.errors import AgentError, ConfigurationError
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.task import TaskResult
from victor.framework.tools import ToolSet

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create a mock Agent instance."""
    agent = MagicMock()
    agent._orchestrator = MagicMock()
    agent._state_observers = []
    agent.event_bus = None
    agent._builder_metadata = {}
    agent._presets_applied = []

    # Mock state
    mock_state = MagicMock()
    mock_state.stage = MagicMock(value="initial")
    agent._state = mock_state
    agent.state = mock_state

    # Mock async methods
    async def mock_run(prompt, context=None):
        return TaskResult(content="Mock response", success=True)

    async def mock_stream(prompt, context=None):
        yield AgentExecutionEvent(type=EventType.CONTENT, content="Mock ")
        yield AgentExecutionEvent(type=EventType.CONTENT, content="response")

    async def mock_reset():
        pass

    agent.run = mock_run
    agent.stream = mock_stream
    agent.reset = mock_reset

    return agent


@pytest.fixture
def mock_orchestrator():
    """Create a mock AgentOrchestrator."""
    orchestrator = MagicMock()
    orchestrator.messages = []
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "anthropic"
    return orchestrator


# =============================================================================
# AgentBuildOptions Tests
# =============================================================================


class TestAgentBuildOptions:
    """Tests for AgentBuildOptions dataclass."""

    def test_default_values(self):
        """Test default option values."""
        options = AgentBuildOptions()
        assert options.provider == "anthropic"
        assert options.model is None
        assert options.temperature == 0.7
        assert options.max_tokens == 4096
        assert options.tools is None
        assert not options.thinking
        assert not options.airgapped
        assert options.enable_observability

    def test_custom_values(self):
        """Test custom option values."""
        options = AgentBuildOptions(
            provider="openai",
            model="gpt-4",
            temperature=0.5,
            thinking=True,
        )
        assert options.provider == "openai"
        assert options.model == "gpt-4"
        assert options.temperature == 0.5
        assert options.thinking

    def test_legacy_config_is_normalized(self):
        """Legacy AgentConfig inputs should be stored canonically."""
        # AgentConfig itself is deprecated; assert the deprecation fires so the
        # legacy path stays properly marked while we verify normalization.
        with pytest.warns(DeprecationWarning):
            options = AgentBuildOptions(config=AgentConfig.high_budget())

        assert isinstance(options.config, UnifiedAgentConfig)
        # 5x'd 2026-10-31: stamina budgets (was 200/100)
        assert options.config.tool_budget == 1000
        assert options.config.max_iterations == 500


# =============================================================================
# AgentBuilder Tests
# =============================================================================


class TestAgentBuilder:
    """Tests for AgentBuilder class."""

    def test_create_builder(self):
        """Test builder creation."""
        builder = create_builder()
        assert isinstance(builder, AgentBuilder)

    def test_from_options(self):
        """Test creating builder from options."""
        options = AgentBuildOptions(provider="openai", model="gpt-4")
        builder = AgentBuilder.from_options(options)
        assert builder._options.provider == "openai"
        assert builder._options.model == "gpt-4"

    def test_provider_chain(self):
        """Test fluent provider setting."""
        builder = AgentBuilder().provider("openai")
        assert builder._options.provider == "openai"
        assert isinstance(builder, AgentBuilder)  # Returns self

    def test_model_chain(self):
        """Test fluent model setting."""
        builder = AgentBuilder().model("gpt-4-turbo")
        assert builder._options.model == "gpt-4-turbo"

    def test_temperature_chain(self):
        """Test fluent temperature setting."""
        builder = AgentBuilder().temperature(0.5)
        assert builder._options.temperature == 0.5

    def test_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ConfigurationError):
            AgentBuilder().temperature(3.0)

        with pytest.raises(ConfigurationError):
            AgentBuilder().temperature(-0.1)

    def test_max_tokens_chain(self):
        """Test fluent max_tokens setting."""
        builder = AgentBuilder().max_tokens(8192)
        assert builder._options.max_tokens == 8192

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        with pytest.raises(ConfigurationError):
            AgentBuilder().max_tokens(0)

        with pytest.raises(ConfigurationError):
            AgentBuilder().max_tokens(-100)

    def test_tools_chain(self):
        """Test fluent tools setting."""
        builder = AgentBuilder().tools(["filesystem", "git"])
        assert builder._options.tools == ["filesystem", "git"]

    def test_thinking_chain(self):
        """Test fluent thinking setting."""
        builder = AgentBuilder().thinking(True)
        assert builder._options.thinking

        builder2 = AgentBuilder().thinking(False)
        assert not builder2._options.thinking

    def test_airgapped_chain(self):
        """Test fluent airgapped setting."""
        builder = AgentBuilder().airgapped(True)
        assert builder._options.airgapped

    def test_profile_chain(self):
        """Test fluent profile setting."""
        builder = AgentBuilder().profile("production")
        assert builder._options.profile == "production"

    def test_workspace_chain(self):
        """Test fluent workspace setting."""
        builder = AgentBuilder().workspace("/tmp/project")
        assert builder._options.workspace == "/tmp/project"

    def test_config_chain(self):
        """Test fluent config setting."""
        config = UnifiedAgentConfig.high_budget()
        builder = AgentBuilder().config(config)
        assert isinstance(builder._options.config, UnifiedAgentConfig)
        assert builder._options.config.tool_budget == config.tool_budget
        assert builder._options.config.max_iterations == config.max_iterations

    def test_system_prompt_chain(self):
        """Test fluent system prompt setting."""
        builder = AgentBuilder().system_prompt("You are a helpful assistant")
        assert builder._options.custom_system_prompt == "You are a helpful assistant"

    def test_with_observability_chain(self):
        """Test fluent observability setting."""
        builder = AgentBuilder().with_observability(True)
        assert builder._options.enable_observability

        builder2 = AgentBuilder().with_observability(False)
        assert not builder2._options.enable_observability

    def test_session_id_chain(self):
        """Test fluent session_id setting."""
        builder = AgentBuilder().session_id("my-session-123")
        assert builder._options.session_id == "my-session-123"

    def test_metadata_chain(self):
        """Test fluent metadata setting."""
        builder = AgentBuilder().metadata("key1", "value1").metadata("key2", 123)
        assert builder._options.metadata["key1"] == "value1"
        assert builder._options.metadata["key2"] == 123

    def test_full_chain(self):
        """Test full fluent chain."""
        builder = (
            AgentBuilder()
            .provider("openai")
            .model("gpt-4")
            .temperature(0.8)
            .max_tokens(4096)
            .thinking(True)
            .with_observability(True)
            .metadata("purpose", "testing")
        )

        opts = builder.get_options()
        assert opts.provider == "openai"
        assert opts.model == "gpt-4"
        assert opts.temperature == 0.8
        assert opts.max_tokens == 4096
        assert opts.thinking
        assert opts.enable_observability
        assert opts.metadata["purpose"] == "testing"

    def test_get_options_returns_copy(self):
        """Test that get_options returns a copy."""
        builder = AgentBuilder().provider("anthropic")
        opts1 = builder.get_options()
        opts2 = builder.get_options()
        assert opts1 is not opts2
        assert opts1.provider == opts2.provider


class TestBuilderPresets:
    """Tests for builder preset configurations."""

    def test_preset_default(self):
        """Test DEFAULT preset."""
        builder = AgentBuilder().preset(BuilderPreset.DEFAULT)
        assert builder._options.tools is not None
        assert isinstance(builder._options.config, UnifiedAgentConfig)
        assert builder._options.config.mode == "foreground"
        assert BuilderPreset.DEFAULT in builder._presets_applied

    def test_preset_minimal(self):
        """Test MINIMAL preset."""
        builder = AgentBuilder().preset(BuilderPreset.MINIMAL)
        assert not builder._options.enable_observability
        assert isinstance(builder._options.config, UnifiedAgentConfig)
        assert builder._options.config.tool_budget == 75  # 5x'd 2026-10-31 (was 15)

    def test_preset_high_budget(self):
        """Test HIGH_BUDGET preset."""
        builder = AgentBuilder().preset(BuilderPreset.HIGH_BUDGET)
        assert isinstance(builder._options.config, UnifiedAgentConfig)
        # High budget config should have higher limits
        assert builder._options.config.tool_budget > UnifiedAgentConfig.minimal().tool_budget

    def test_preset_airgapped(self):
        """Test AIRGAPPED preset."""
        builder = AgentBuilder().preset(BuilderPreset.AIRGAPPED)
        assert builder._options.airgapped
        assert isinstance(builder._options.config, UnifiedAgentConfig)
        assert builder._options.config.enable_semantic_search is False
        assert builder._options.config.enable_analytics is False

    def test_multiple_presets(self):
        """Test applying multiple presets."""
        builder = AgentBuilder().preset(BuilderPreset.DEFAULT).preset(BuilderPreset.HIGH_BUDGET)
        assert len(builder._presets_applied) == 2
        assert BuilderPreset.DEFAULT in builder._presets_applied
        assert BuilderPreset.HIGH_BUDGET in builder._presets_applied


class TestBuilderToolPresets:
    """Tests for builder tool preset methods."""

    def test_default_tools(self):
        """Test default_tools method."""
        builder = AgentBuilder().default_tools()
        assert builder._options.tools is not None
        assert isinstance(builder._options.tools, ToolSet)

    def test_minimal_tools(self):
        """Test minimal_tools method."""
        builder = AgentBuilder().minimal_tools()
        assert builder._options.tools is not None
        assert isinstance(builder._options.tools, ToolSet)

    def test_full_tools(self):
        """Test full_tools method."""
        builder = AgentBuilder().full_tools()
        assert builder._options.tools is not None
        assert isinstance(builder._options.tools, ToolSet)

    def test_airgapped_tools(self):
        """Test airgapped_tools method."""
        builder = AgentBuilder().airgapped_tools()
        assert builder._options.tools is not None
        assert isinstance(builder._options.tools, ToolSet)


class TestBuilderStateHooks:
    """Tests for builder state hook methods."""

    def test_on_enter_stage(self):
        """Test on_enter_stage hook registration."""
        callback = MagicMock()
        builder = AgentBuilder().on_enter_stage(callback)
        assert builder._options.state_hooks is not None
        assert builder._options.state_hooks["on_enter"] is callback

    def test_on_exit_stage(self):
        """Test on_exit_stage hook registration."""
        callback = MagicMock()
        builder = AgentBuilder().on_exit_stage(callback)
        assert builder._options.state_hooks is not None
        assert builder._options.state_hooks["on_exit"] is callback

    def test_on_transition(self):
        """Test on_transition hook registration."""
        callback = MagicMock()
        builder = AgentBuilder().on_transition(callback)
        assert builder._options.state_hooks is not None
        assert builder._options.state_hooks["on_transition"] is callback

    def test_multiple_hooks(self):
        """Test registering multiple hooks."""
        enter_cb = MagicMock()
        exit_cb = MagicMock()
        trans_cb = MagicMock()

        builder = (
            AgentBuilder().on_enter_stage(enter_cb).on_exit_stage(exit_cb).on_transition(trans_cb)
        )

        assert builder._options.state_hooks["on_enter"] is enter_cb
        assert builder._options.state_hooks["on_exit"] is exit_cb
        assert builder._options.state_hooks["on_transition"] is trans_cb


# =============================================================================
# AgentSession Tests
# =============================================================================


class TestAgentSession:
    """Tests for AgentSession class."""

    def test_session_creation(self, mock_agent):
        """Test session creation."""
        session = AgentSession(mock_agent, "Hello")
        assert session._initial_prompt == "Hello"
        assert session.turn_count == 0
        assert session.state == SessionState.IDLE
        assert not session._initialized

    def test_session_with_id(self, mock_agent):
        """Test session with custom ID."""
        session = AgentSession(mock_agent, "Hello", session_id="test-123")
        assert session.session_id == "test-123"

    def test_session_with_metadata(self, mock_agent):
        """Test session with metadata."""
        session = AgentSession(mock_agent, "Hello", metadata={"purpose": "testing"})
        assert session._context.metadata["purpose"] == "testing"

    @pytest.mark.asyncio
    async def test_send_first_turn(self, mock_agent):
        """Test first send uses initial prompt."""
        session = AgentSession(mock_agent, "Initial prompt")

        await session.send("Second message")

        # First turn should use initial prompt, not the send message
        assert session._initialized
        assert session.turn_count == 1
        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_send_subsequent_turns(self, mock_agent):
        """Test subsequent sends use provided message."""
        session = AgentSession(mock_agent, "Initial")

        await session.send("First call")
        await session.send("Second call")

        assert session.turn_count == 2

    @pytest.mark.asyncio
    async def test_send_with_context(self, mock_agent):
        """Test send with context."""
        session = AgentSession(mock_agent, "Hello")

        result = await session.send_with_context(
            "Fix bug",
            context={"error": "IndexError"},
        )

        assert result is not None
        assert len(session.turns) == 1
        assert session.turns[0]["context"] == {"error": "IndexError"}

    @pytest.mark.asyncio
    async def test_send_uses_message_when_initial_prompt_not_provided(self, mock_agent):
        """Deferred sessions should use the first send() message as the opening turn."""
        session = AgentSession(mock_agent)

        await session.send("Deferred first turn")

        assert session.turn_count == 1
        assert session.turns[0]["prompt"] == "Deferred first turn"

    @pytest.mark.asyncio
    async def test_stream(self, mock_agent):
        """Test streaming."""
        session = AgentSession(mock_agent, "Hello")

        events = []
        async for event in session.stream("Test"):
            events.append(event)

        assert len(events) >= 1
        assert session.turn_count == 1

    @pytest.mark.asyncio
    async def test_turns_tracking(self, mock_agent):
        """Test turn history tracking."""
        session = AgentSession(mock_agent, "Hello")

        await session.send("First")
        await session.send("Second")

        turns = session.turns
        assert len(turns) == 2
        assert turns[0]["turn"] == 1
        assert turns[1]["turn"] == 2
        assert "duration" in turns[0]

    def test_pause_resume(self, mock_agent):
        """Test pause and resume."""
        session = AgentSession(mock_agent, "Hello")
        session._state = SessionState.ACTIVE

        session.pause()
        assert session.state == SessionState.PAUSED

        session.resume()
        assert session.state == SessionState.ACTIVE

    def test_pause_only_from_active(self, mock_agent):
        """Test pause only works from active state."""
        session = AgentSession(mock_agent, "Hello")
        session._state = SessionState.IDLE

        session.pause()
        assert session.state == SessionState.IDLE  # Unchanged

    def test_resume_only_from_paused(self, mock_agent):
        """Test resume only works from paused state."""
        session = AgentSession(mock_agent, "Hello")
        session._state = SessionState.IDLE

        session.resume()
        assert session.state == SessionState.IDLE  # Unchanged

    @pytest.mark.asyncio
    async def test_close(self, mock_agent):
        """Test session close."""
        session = AgentSession(mock_agent, "Hello")

        await session.close()
        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_send_after_close_raises(self, mock_agent):
        """Test send after close raises error."""
        session = AgentSession(mock_agent, "Hello")
        await session.close()

        with pytest.raises(AgentError, match="closed"):
            await session.send("Test")

    @pytest.mark.asyncio
    async def test_reset(self, mock_agent):
        """Test session reset."""
        session = AgentSession(mock_agent, "Hello")
        await session.send("First")

        await session.reset()

        assert session.turn_count == 0
        assert not session._initialized
        assert session.state == SessionState.IDLE
        assert len(session._turns) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_agent):
        """Test session as context manager."""
        async with AgentSession(mock_agent, "Hello") as session:
            await session.send("Test")
            assert session.turn_count == 1

        assert session.state == SessionState.CLOSED

    def test_repr(self, mock_agent):
        """Test session repr."""
        session = AgentSession(mock_agent, "Hello")
        repr_str = repr(session)
        assert "AgentSession" in repr_str
        assert "turns=0" in repr_str
        assert "idle" in repr_str


class TestAgentSessionHistory:
    """Tests for AgentSession history functionality."""

    def test_history_empty(self, mock_agent):
        """Test empty history."""
        session = AgentSession(mock_agent, "Hello")
        assert session.history == []

    def test_history_with_messages(self, mock_agent):
        """Test history with messages."""
        # Set up mock messages
        mock_message = MagicMock()
        mock_message.role = "user"
        mock_message.content = "Hello"
        mock_agent._orchestrator.messages = [mock_message]

        session = AgentSession(mock_agent, "Hello")
        history = session.history

        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"


# =============================================================================
# AgentBridge Tests
# =============================================================================


class TestBridgeConfiguration:
    """Tests for BridgeConfiguration."""

    def test_default_config(self):
        """Test default configuration."""
        config = BridgeConfiguration()
        assert config.enable_observability
        assert config.enable_metrics
        assert config.auto_forward_events

    def test_custom_config(self):
        """Test custom configuration."""
        config = BridgeConfiguration(
            auto_forward_events=False,
        )
        assert not config.auto_forward_events


class TestAgentBridge:
    """Tests for AgentBridge class."""

    def test_bridge_creation(self, mock_agent):
        """Test bridge creation."""
        bridge = AgentBridge(mock_agent)
        assert not bridge.connected
        assert bridge.session_id is None
        assert bridge.agent is mock_agent

    def test_bridge_with_config(self, mock_agent):
        """Test bridge with custom config."""
        config = BridgeConfiguration()
        bridge = AgentBridge(mock_agent, config)
        assert bridge._config.enable_observability is True

    @pytest.mark.asyncio
    async def test_connect(self, mock_agent):
        """Test connect."""
        bridge = AgentBridge(mock_agent)

        session_id = await bridge.connect()

        assert bridge.connected
        assert session_id is not None
        assert bridge.session_id == session_id

    @pytest.mark.asyncio
    async def test_connect_already_connected_raises(self, mock_agent):
        """Test connect when already connected raises error."""
        bridge = AgentBridge(mock_agent)
        await bridge.connect()

        with pytest.raises(AgentError, match="already connected"):
            await bridge.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_agent):
        """Test disconnect."""
        bridge = AgentBridge(mock_agent)
        await bridge.connect()

        await bridge.disconnect()

        assert not bridge.connected
        assert bridge.session_id is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, mock_agent):
        """Test disconnect when not connected is safe."""
        bridge = AgentBridge(mock_agent)
        await bridge.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_agent):
        """Test bridge as context manager."""
        async with AgentBridge(mock_agent) as bridge:
            assert bridge.connected

        assert not bridge.connected

    def test_repr(self, mock_agent):
        """Test bridge repr."""
        bridge = AgentBridge(mock_agent)
        repr_str = repr(bridge)
        assert "AgentBridge" in repr_str
        assert "disconnected" in repr_str


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_builder(self):
        """Test create_builder function."""
        builder = create_builder()
        assert isinstance(builder, AgentBuilder)

    @pytest.mark.asyncio
    async def test_create_session_context_manager(self, mock_agent):
        """Test create_session context manager."""
        async with create_session(mock_agent, "Hello") as session:
            assert isinstance(session, AgentSession)
            await session.send("Test")

        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_create_bridge_context_manager(self, mock_agent):
        """Test create_bridge context manager."""
        async with create_bridge(mock_agent) as bridge:
            assert isinstance(bridge, AgentBridge)
            assert bridge.connected

        assert not bridge.connected


# =============================================================================
# SessionContext Tests
# =============================================================================


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_session_context_creation(self):
        """Test SessionContext creation."""
        ctx = SessionContext(
            session_id="test-123",
            created_at=1234567890.0,
            metadata={"key": "value"},
        )
        assert ctx.session_id == "test-123"
        assert ctx.created_at == 1234567890.0
        assert ctx.metadata["key"] == "value"

    def test_session_context_default_metadata(self):
        """Test SessionContext default metadata."""
        ctx = SessionContext(session_id="test", created_at=0)
        assert ctx.metadata == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentBuilderIntegration:
    """Integration tests for AgentBuilder with Agent.create()."""

    @pytest.mark.asyncio
    async def test_build_with_mock_create(self):
        """Test build calls Agent.create with correct options."""
        mock_agent = MagicMock()
        mock_agent.get_orchestrator.return_value = MagicMock()
        with patch(
            "victor.framework.Agent.create", new=AsyncMock(return_value=mock_agent)
        ) as create:

            builder = (
                AgentBuilder().provider("openai").model("gpt-4").temperature(0.5).thinking(True)
            )

            await builder.build()

            # Verify create was called with correct options
            create.assert_called_once()
            call_kwargs = create.call_args.kwargs
            assert call_kwargs.get("provider") == "openai" or create.call_args.args[0] == "openai"


# =============================================================================
# DI Container Integration Tests (Phase 8.2)
# =============================================================================


class TestAgentBuilderContainerIntegration:
    """Tests for AgentBuilder integration with ServiceContainer."""

    def test_builder_creation_without_container(self):
        """Test builder creation without container."""
        builder = AgentBuilder()
        assert builder._container is None
        assert builder.has_container is False

    def test_builder_creation_with_container(self):
        """Test builder creation with container."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        builder = AgentBuilder(container=container)

        assert builder._container is container
        assert builder.has_container is True

    def test_from_container_factory(self):
        """Test from_container class method."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        builder = AgentBuilder.from_container(container)

        assert builder._container is container
        assert builder.has_container is True

    def test_with_container_method(self):
        """Test with_container fluent method."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        builder = AgentBuilder().with_container(container)

        assert builder._container is container
        assert builder.has_container is True

    def test_from_options_with_container(self):
        """Test from_options with container parameter."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        options = AgentBuildOptions(provider="openai")
        builder = AgentBuilder.from_options(options, container=container)

        assert builder._container is container
        assert builder._options.provider == "openai"

    def test_add_tool_filter(self):
        """Test add_tool_filter method."""
        mock_filter = MagicMock()
        builder = AgentBuilder().add_tool_filter(mock_filter)

        assert mock_filter in builder._tool_filters
        assert len(builder._tool_filters) == 1

    def test_add_multiple_tool_filters(self):
        """Test adding multiple tool filters."""
        filter1 = MagicMock()
        filter2 = MagicMock()

        builder = AgentBuilder().add_tool_filter(filter1).add_tool_filter(filter2)

        assert len(builder._tool_filters) == 2
        assert filter1 in builder._tool_filters
        assert filter2 in builder._tool_filters

    def test_create_builder_function_without_container(self):
        """Test create_builder factory without container."""
        builder = create_builder()
        assert builder._container is None

    def test_create_builder_function_with_container(self):
        """Test create_builder factory with container."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        builder = create_builder(container=container)

        assert builder._container is container

    def test_get_tool_configurator_without_container(self):
        """Test _get_tool_configurator without container."""
        builder = AgentBuilder()
        configurator = builder._get_tool_configurator()

        assert configurator is None

    def test_get_tool_configurator_with_container(self):
        """Test _get_tool_configurator with configured container."""
        from victor.core.container import ServiceContainer
        from victor.framework.service_provider import (
            FrameworkServiceProvider,
        )

        container = ServiceContainer()
        provider = FrameworkServiceProvider()
        provider.register_services(container)

        builder = AgentBuilder(container=container)
        configurator = builder._get_tool_configurator()

        assert configurator is not None
        assert hasattr(configurator, "configure")

    def test_get_event_registry_without_container(self):
        """Test _get_event_registry without container."""
        builder = AgentBuilder()
        registry = builder._get_event_registry()

        assert registry is None

    def test_get_event_registry_with_container(self):
        """Test _get_event_registry with configured container."""
        from victor.core.container import ServiceContainer
        from victor.framework.service_provider import (
            FrameworkServiceProvider,
        )

        container = ServiceContainer()
        provider = FrameworkServiceProvider()
        provider.register_services(container)

        builder = AgentBuilder(container=container)
        registry = builder._get_event_registry()

        assert registry is not None
        assert hasattr(registry, "get_converter")

    @pytest.mark.asyncio
    async def test_build_stores_container_reference(self):
        """Test that build stores container reference on agent."""
        from victor.core.container import ServiceContainer
        from victor.framework.service_provider import FrameworkServiceProvider

        container = ServiceContainer()
        provider = FrameworkServiceProvider()
        provider.register_services(container)

        mock_agent = MagicMock()
        mock_agent.get_orchestrator.return_value = MagicMock()
        with patch("victor.framework.Agent.create", new=AsyncMock(return_value=mock_agent)):

            builder = AgentBuilder(container=container)
            await builder.build()

            # Verify container is stored on agent
            assert mock_agent._container is container

    @pytest.mark.asyncio
    async def test_build_with_tool_filters_uses_container(self):
        """Test that tool filters are applied via container configurator."""
        from victor.core.container import ServiceContainer
        from victor.framework.service_provider import (
            FrameworkServiceProvider,
            ToolConfiguratorService,
        )

        container = ServiceContainer()
        provider = FrameworkServiceProvider()
        provider.register_services(container)

        mock_filter = MagicMock()

        mock_agent = MagicMock()
        mock_orchestrator = MagicMock()
        mock_agent.get_orchestrator.return_value = mock_orchestrator
        with patch("victor.framework.Agent.create", new=AsyncMock(return_value=mock_agent)):

            # Get the configurator to verify filters are added
            configurator = container.get(ToolConfiguratorService)
            original_add_filter = configurator.add_filter

            filters_added = []

            def track_add_filter(f):
                filters_added.append(f)
                return original_add_filter(f)

            configurator.add_filter = track_add_filter

            builder = (
                AgentBuilder(container=container)
                .tools(["read", "write"])
                .add_tool_filter(mock_filter)
            )
            await builder.build()

            # Verify filter was added
            assert mock_filter in filters_added

    @pytest.mark.asyncio
    async def test_build_with_tool_filters_fallback(self):
        """Test that tool filters work without container (fallback)."""
        mock_filter = MagicMock()

        mock_agent = MagicMock()
        mock_orchestrator = MagicMock()
        mock_agent.get_orchestrator.return_value = mock_orchestrator
        with patch("victor.framework.Agent.create", new=AsyncMock(return_value=mock_agent)):

            # Patch at the source module where get_tool_configurator is defined
            with patch(
                "victor.framework.tool_config.get_tool_configurator"
            ) as mock_get_configurator:
                mock_configurator = MagicMock()
                mock_get_configurator.return_value = mock_configurator

                builder = AgentBuilder().tools(["read"]).add_tool_filter(mock_filter)
                await builder.build()

                # Verify fallback configurator was used
                mock_get_configurator.assert_called_once()
                mock_configurator.add_filter.assert_called_with(mock_filter)


class TestAgentBuilderChaining:
    """Test fluent API method chaining returns builder."""

    def test_with_container_returns_builder(self):
        """Test with_container returns builder for chaining."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        builder = AgentBuilder()
        result = builder.with_container(container)

        assert result is builder

    def test_add_tool_filter_returns_builder(self):
        """Test add_tool_filter returns builder for chaining."""
        builder = AgentBuilder()
        result = builder.add_tool_filter(MagicMock())

        assert result is builder

    def test_full_chain_with_container(self):
        """Test full method chain with container."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        mock_filter = MagicMock()

        builder = (
            AgentBuilder()
            .with_container(container)
            .provider("anthropic")
            .model("claude-3")
            .tools(["read", "write"])
            .add_tool_filter(mock_filter)
            .thinking(True)
            .airgapped(True)
        )

        assert builder._container is container
        assert mock_filter in builder._tool_filters
        assert builder._options.provider == "anthropic"
        assert builder._options.thinking is True


# =============================================================================
# Phase 8.3: Session Lifecycle Management Tests
# =============================================================================


class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_metrics_default_values(self):
        """Test default values for SessionMetrics."""
        metrics = SessionMetrics()

        assert metrics.total_turns == 0
        assert metrics.total_duration == 0.0
        assert metrics.total_tool_calls == 0
        assert metrics.successful_turns == 0
        assert metrics.failed_turns == 0
        assert metrics.average_turn_duration == 0.0

    def test_metrics_update_successful_turn(self):
        """Test updating metrics with successful turn data."""
        metrics = SessionMetrics()

        turn_data = {
            "duration": 2.5,
            "tool_count": 3,
            "success": True,
        }
        metrics.update(turn_data)

        assert metrics.total_turns == 1
        assert metrics.total_duration == 2.5
        assert metrics.total_tool_calls == 3
        assert metrics.successful_turns == 1
        assert metrics.failed_turns == 0
        assert metrics.average_turn_duration == 2.5

    def test_metrics_update_failed_turn(self):
        """Test updating metrics with failed turn data."""
        metrics = SessionMetrics()

        turn_data = {
            "duration": 1.0,
            "tool_count": 1,
            "success": False,
        }
        metrics.update(turn_data)

        assert metrics.total_turns == 1
        assert metrics.successful_turns == 0
        assert metrics.failed_turns == 1

    def test_metrics_update_multiple_turns(self):
        """Test updating metrics with multiple turns."""
        metrics = SessionMetrics()

        metrics.update({"duration": 2.0, "tool_count": 2, "success": True})
        metrics.update({"duration": 4.0, "tool_count": 3, "success": True})
        metrics.update({"duration": 3.0, "tool_count": 1, "success": False})

        assert metrics.total_turns == 3
        assert metrics.total_duration == 9.0
        assert metrics.total_tool_calls == 6
        assert metrics.successful_turns == 2
        assert metrics.failed_turns == 1
        assert metrics.average_turn_duration == 3.0

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = SessionMetrics()
        metrics.update({"duration": 2.0, "tool_count": 2, "success": True})

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["total_turns"] == 1
        assert result["total_duration"] == 2.0
        assert result["total_tool_calls"] == 2
        assert result["successful_turns"] == 1
        assert result["failed_turns"] == 0
        assert result["average_turn_duration"] == 2.0


class TestSessionLifecycleHooks:
    """Tests for SessionLifecycleHooks dataclass."""

    def test_hooks_default_none(self):
        """Test that all hooks default to None."""
        hooks = SessionLifecycleHooks()

        assert hooks.on_start is None
        assert hooks.on_turn_start is None
        assert hooks.on_turn_end is None
        assert hooks.on_error is None
        assert hooks.on_pause is None
        assert hooks.on_resume is None
        assert hooks.on_close is None

    def test_hooks_with_callbacks(self):
        """Test creating hooks with callbacks."""
        on_start = MagicMock()
        on_close = MagicMock()

        hooks = SessionLifecycleHooks(
            on_start=on_start,
            on_close=on_close,
        )

        assert hooks.on_start is on_start
        assert hooks.on_close is on_close


class TestSessionLifecycleManagement:
    """Tests for AgentSession lifecycle management (Phase 8.3)."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock Agent."""
        agent = MagicMock()
        agent._orchestrator = MagicMock()
        agent._orchestrator.messages = []
        agent.run = AsyncMock(
            return_value=TaskResult(
                content="Test response",
                success=True,
                tool_calls=[{"name": "test_tool"}],
            )
        )
        agent.stream = AsyncMock(return_value=iter([]))
        agent.reset = AsyncMock()
        return agent

    def test_session_with_hooks(self, mock_agent):
        """Test creating session with lifecycle hooks."""
        hooks = SessionLifecycleHooks(
            on_start=MagicMock(),
            on_close=MagicMock(),
        )

        session = AgentSession(mock_agent, "Test prompt", hooks=hooks)

        assert session.hooks is hooks
        assert session._hooks is hooks

    def test_session_metrics_property(self, mock_agent):
        """Test session metrics property."""
        session = AgentSession(mock_agent, "Test prompt")

        metrics = session.metrics

        assert isinstance(metrics, SessionMetrics)
        assert metrics.total_turns == 0

    def test_session_scope_property_without_container(self, mock_agent):
        """Test scope property without container."""
        session = AgentSession(mock_agent, "Test prompt")

        assert session.scope is None

    def test_session_scope_property_with_container(self, mock_agent):
        """Test scope property with container."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        session = AgentSession(mock_agent, "Test prompt", container=container)

        assert session.scope is not None

    @pytest.mark.asyncio
    async def test_on_start_hook_called_on_first_send(self, mock_agent):
        """Test on_start hook is called on first send."""
        on_start = MagicMock()
        hooks = SessionLifecycleHooks(on_start=on_start)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        await session.send("message")

        on_start.assert_called_once_with(session)

    @pytest.mark.asyncio
    async def test_on_start_hook_called_once(self, mock_agent):
        """Test on_start hook is only called once."""
        on_start = MagicMock()
        hooks = SessionLifecycleHooks(on_start=on_start)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        await session.send("first")
        await session.send("second")

        assert on_start.call_count == 1

    @pytest.mark.asyncio
    async def test_on_turn_start_hook_called(self, mock_agent):
        """Test on_turn_start hook is called before each turn."""
        on_turn_start = MagicMock()
        hooks = SessionLifecycleHooks(on_turn_start=on_turn_start)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        await session.send("message")

        on_turn_start.assert_called()

    @pytest.mark.asyncio
    async def test_on_turn_end_hook_called(self, mock_agent):
        """Test on_turn_end hook is called after each turn."""
        on_turn_end = MagicMock()
        hooks = SessionLifecycleHooks(on_turn_end=on_turn_end)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        await session.send("message")

        on_turn_end.assert_called_once()
        # Verify it received the session and a TaskResult
        call_args = on_turn_end.call_args[0]
        assert call_args[0] is session
        assert isinstance(call_args[1], TaskResult)

    @pytest.mark.asyncio
    async def test_on_error_hook_called_on_exception(self, mock_agent):
        """Test on_error hook is called when exception occurs."""
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Test error"))
        on_error = MagicMock()
        hooks = SessionLifecycleHooks(on_error=on_error)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)

        with pytest.raises(RuntimeError):
            await session.send("message")

        on_error.assert_called_once()
        call_args = on_error.call_args[0]
        assert call_args[0] is session
        assert isinstance(call_args[1], RuntimeError)

    def test_on_pause_hook_called(self, mock_agent):
        """Test on_pause hook is called when session is paused."""
        on_pause = MagicMock()
        hooks = SessionLifecycleHooks(on_pause=on_pause)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        session._state = SessionState.ACTIVE
        session.pause()

        on_pause.assert_called_once_with(session)

    def test_on_pause_hook_not_called_if_not_active(self, mock_agent):
        """Test on_pause hook is not called if session not active."""
        on_pause = MagicMock()
        hooks = SessionLifecycleHooks(on_pause=on_pause)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        session.pause()  # Session is IDLE, not ACTIVE

        on_pause.assert_not_called()

    def test_on_resume_hook_called(self, mock_agent):
        """Test on_resume hook is called when session is resumed."""
        on_resume = MagicMock()
        hooks = SessionLifecycleHooks(on_resume=on_resume)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        session._state = SessionState.PAUSED
        session.resume()

        on_resume.assert_called_once_with(session)

    def test_on_resume_hook_not_called_if_not_paused(self, mock_agent):
        """Test on_resume hook is not called if session not paused."""
        on_resume = MagicMock()
        hooks = SessionLifecycleHooks(on_resume=on_resume)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        session.resume()  # Session is IDLE, not PAUSED

        on_resume.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_close_hook_called_with_metrics(self, mock_agent):
        """Test on_close hook is called with metrics when session closes."""
        on_close = MagicMock()
        hooks = SessionLifecycleHooks(on_close=on_close)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        await session.close()

        on_close.assert_called_once()
        call_args = on_close.call_args[0]
        assert call_args[0] is session
        assert isinstance(call_args[1], SessionMetrics)

    @pytest.mark.asyncio
    async def test_close_disposes_scoped_container(self, mock_agent):
        """Test close disposes scoped container."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        session = AgentSession(mock_agent, "Initial prompt", container=container)

        scope = session.scope
        assert scope is not None

        await session.close()

        assert session._scope is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_agent):
        """Test calling close multiple times is safe."""
        on_close = MagicMock()
        hooks = SessionLifecycleHooks(on_close=on_close)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        await session.close()
        await session.close()  # Should not raise

        # on_close should only be called once
        on_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_invokes_close(self, mock_agent):
        """Test reset invokes close hook."""
        on_close = MagicMock()
        hooks = SessionLifecycleHooks(on_close=on_close)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        await session.reset()

        on_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_resets_metrics(self, mock_agent):
        """Test reset resets session metrics."""
        session = AgentSession(mock_agent, "Initial prompt")

        # Add some data
        await session.send("message")

        # Reset
        await session.reset()

        assert session.metrics.total_turns == 0
        assert session.turn_count == 0
        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_reset_recreates_scope(self, mock_agent):
        """Test reset recreates scoped container."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        session = AgentSession(mock_agent, "Initial prompt", container=container)

        original_scope = session.scope
        await session.reset()

        # Should have a new scope
        assert session.scope is not None
        assert session.scope is not original_scope

    @pytest.mark.asyncio
    async def test_metrics_updated_after_send(self, mock_agent):
        """Test metrics are updated after send."""
        session = AgentSession(mock_agent, "Initial prompt")
        await session.send("message")

        assert session.metrics.total_turns == 1
        assert session.metrics.successful_turns == 1

    def test_hook_errors_logged_not_raised(self, mock_agent):
        """Test that hook errors are logged but not raised."""
        on_pause = MagicMock(side_effect=RuntimeError("Hook error"))
        hooks = SessionLifecycleHooks(on_pause=on_pause)

        session = AgentSession(mock_agent, "Initial prompt", hooks=hooks)
        session._state = SessionState.ACTIVE

        # Should not raise
        session.pause()

        assert session.state == SessionState.PAUSED


class TestSessionLifecycleIntegration:
    """Integration tests for session lifecycle."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock Agent."""
        agent = MagicMock()
        agent._orchestrator = MagicMock()
        agent._orchestrator.messages = []
        agent.run = AsyncMock(
            return_value=TaskResult(
                content="Response",
                success=True,
                tool_calls=[],
            )
        )
        agent.reset = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_full_lifecycle_flow(self, mock_agent):
        """Test full session lifecycle with all hooks."""
        events = []

        hooks = SessionLifecycleHooks(
            on_start=lambda s: events.append("start"),
            on_turn_start=lambda s, m: events.append(f"turn_start:{m[:10]}"),
            on_turn_end=lambda s, r: events.append("turn_end"),
            on_pause=lambda s: events.append("pause"),
            on_resume=lambda s: events.append("resume"),
            on_close=lambda s, m: events.append(f"close:{m.total_turns}"),
        )

        session = AgentSession(mock_agent, "Initial", hooks=hooks)

        # First turn (triggers on_start)
        await session.send("message1")

        # Second turn
        await session.send("message2")

        # Pause/Resume
        session._state = SessionState.ACTIVE
        session.pause()
        session.resume()

        # Close
        await session.close()

        assert events == [
            "start",
            "turn_start:Initial",
            "turn_end",
            "turn_start:message2",
            "turn_end",
            "pause",
            "resume",
            "close:2",
        ]

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, mock_agent):
        """Test lifecycle with context manager."""
        on_close = MagicMock()
        hooks = SessionLifecycleHooks(on_close=on_close)

        async with create_session(mock_agent, "Test", hooks=hooks) as session:
            await session.send("message")

        # on_close should be called when exiting context
        on_close.assert_called_once()
