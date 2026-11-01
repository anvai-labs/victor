from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.agent import Agent
from victor.framework.agent_components import AgentSession
from victor.framework.client import VictorClient
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.session_config import SessionConfig


@pytest.mark.asyncio
async def test_victor_client_ensure_initialized_uses_provider_defaults() -> None:
    config = SessionConfig.from_cli_flags(tool_budget=4)
    client = VictorClient(config, container=object())
    settings = SimpleNamespace(
        provider=SimpleNamespace(
            default_provider="ollama",
            default_model="mistral-tools:7b-instruct",
        )
    )
    mock_agent = object()

    with (
        patch("victor.config.settings.load_settings", return_value=settings),
        patch.object(Agent, "create", new=AsyncMock(return_value=mock_agent)) as create,
    ):
        agent = await client._ensure_initialized()

    assert agent is mock_agent
    create.assert_awaited_once_with(
        profile=None,  # agent_profile from SessionConfig (defaults to None)
        provider="ollama",
        model="mistral-tools:7b-instruct",
        session_config=config,
    )


@pytest.mark.asyncio
async def test_victor_client_does_not_clobber_profile_with_settings_defaults() -> None:
    config = SessionConfig.from_cli_flags(agent_profile="zai-coding")
    client = VictorClient(config, container=object())
    settings = SimpleNamespace(
        provider=SimpleNamespace(
            default_provider="ollama",
            default_model="qwen3.5:27b-q4_K_M",
        )
    )
    mock_agent = object()

    with (
        patch("victor.config.settings.load_settings", return_value=settings),
        patch.object(Agent, "create", new=AsyncMock(return_value=mock_agent)) as create,
    ):
        agent = await client._ensure_initialized()

    assert agent is mock_agent
    create.assert_awaited_once_with(
        profile="zai-coding",
        provider=None,
        model=None,
        session_config=config,
    )


def test_session_config_normalizes_provider_override_and_applies_endpoint() -> None:
    config = SessionConfig.from_cli_flags(
        provider="OLLAMA",
        endpoint="http://localhost:11434",
        provider_timeout=180,
        observability_logging=True,
        auto_skill_enabled=False,
        one_shot_mode=True,
    )
    settings = SimpleNamespace(
        provider=SimpleNamespace(
            default_provider="anthropic",
            default_model="claude-sonnet",
            ollama_base_url="http://old-host:11434",
            timeout=60,
        ),
        tool_settings=SimpleNamespace(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=False,
            tool_output_pruning_safe_only=True,
        ),
        observability=SimpleNamespace(enable_observability_logging=False),
        automation=SimpleNamespace(one_shot_mode=False),
        enable_observability_logging=False,
        skill_auto_select_enabled=True,
        one_shot_mode=False,
    )

    config.apply_to_settings(settings)

    assert config.provider_override.provider == "ollama"
    assert config.provider_override.model == "qwen2.5-coder:7b"
    assert config.provider_override.timeout == 180
    assert settings.provider.default_provider == "ollama"
    assert settings.provider.default_model == "qwen2.5-coder:7b"
    assert settings.provider.ollama_base_url == "http://localhost:11434"
    assert settings.provider.timeout == 180
    assert settings.observability.enable_observability_logging is True
    assert settings.enable_observability_logging is True
    assert settings.skill_auto_select_enabled is False
    assert settings.automation.one_shot_mode is True
    assert settings.one_shot_mode is True


def test_session_config_rejects_endpoint_for_cloud_provider() -> None:
    with pytest.raises(ValueError, match="--endpoint is only supported for local providers"):
        SessionConfig.from_cli_flags(provider="openai", endpoint="https://custom.example")


def test_victor_client_bootstrap_container_uses_bootstrap_factory_result() -> None:
    client = VictorClient(SessionConfig())
    sentinel_container = object()

    with patch(
        "victor.core.bootstrap.bootstrap_container",
        return_value=sentinel_container,
    ) as bootstrap_container:
        result = client._bootstrap_container()

    assert result is sentinel_container
    bootstrap_container.assert_called_once_with()


@pytest.mark.asyncio
async def test_agent_public_api_for_runtime_configuration() -> None:
    """Test Agent exposes runtime configuration methods for CLI integration."""
    from victor.framework.agent import Agent

    # Mock orchestrator with unified_tracker and provider
    mock_tracker = MagicMock()
    mock_provider = MagicMock()
    mock_provider.supports_streaming.return_value = True

    mock_orchestrator = MagicMock()
    mock_orchestrator.unified_tracker = mock_tracker
    mock_orchestrator.provider = mock_provider
    # Make the mock pass Agent's type validation
    mock_orchestrator.__class__.__name__ = "AgentOrchestrator"

    # Create agent via from_orchestrator (escape hatch)
    agent = Agent.from_orchestrator(mock_orchestrator)

    # Test set_tool_budget
    agent.set_tool_budget(50, user_override=True)
    mock_tracker.set_tool_budget.assert_called_once_with(50, user_override=True)

    # Test set_max_iterations
    agent.set_max_iterations(20, user_override=True)
    mock_tracker.set_max_iterations.assert_called_once_with(20, user_override=True)

    # Test supports_streaming
    assert agent.supports_streaming() is True
    mock_provider.supports_streaming.assert_called_once()


@pytest.mark.asyncio
async def test_agent_public_api_delegates_to_orchestrator_internals() -> None:
    """Test Agent methods correctly delegate to orchestrator internals."""
    from victor.framework.agent import Agent

    # Mock orchestrator with unified_tracker and provider
    mock_tracker = MagicMock()
    mock_provider = MagicMock()
    mock_provider.supports_streaming.return_value = False

    mock_orchestrator = MagicMock()
    mock_orchestrator.unified_tracker = mock_tracker
    mock_orchestrator.provider = mock_provider
    mock_orchestrator.__class__.__name__ = "AgentOrchestrator"

    agent = Agent.from_orchestrator(mock_orchestrator)

    # Test that methods delegate correctly
    agent.set_tool_budget(100)
    mock_tracker.set_tool_budget.assert_called_once_with(100, user_override=False)

    agent.set_max_iterations(30)
    mock_tracker.set_max_iterations.assert_called_once_with(30, user_override=False)

    assert agent.supports_streaming() is False
    mock_provider.supports_streaming.assert_called_once()


@pytest.mark.asyncio
async def test_victor_client_stream_preserves_framework_event_contract() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())

    class _FakeAgent:
        async def stream(self, _message: str):
            yield AgentExecutionEvent(type=EventType.CONTENT, content="READY")
            yield AgentExecutionEvent(
                type=EventType.TOOL_RESULT,
                tool_name="database",
                arguments={"sql": "select 1"},
                result="row",
                metadata={"elapsed": 0.02},
            )

    client._agent = _FakeAgent()

    events = [event async for event in client.stream("ping")]

    assert events[0].type == EventType.CONTENT
    assert events[0].event_type == "content"
    assert events[0].content == "READY"

    assert events[1].type == EventType.TOOL_RESULT
    assert events[1].event_type == "tool_result"
    assert events[1].tool_name == "database"
    assert events[1].arguments == {"sql": "select 1"}
    assert events[1].result["result"] == "row"
    assert events[1].result["arguments"] == {"sql": "select 1"}


@pytest.mark.asyncio
async def test_victor_client_stream_preserves_error_event_message() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())

    class _FakeAgent:
        async def stream(self, _message: str):
            yield AgentExecutionEvent(
                type=EventType.ERROR,
                error="provider rejected tool schema",
                recoverable=False,
            )

    client._agent = _FakeAgent()

    events = [event async for event in client.stream("ping")]

    assert len(events) == 1
    assert events[0].event_type == "error"
    assert events[0].success is False
    assert events[0].content == "provider rejected tool schema"
    assert events[0].metadata["error"] == "provider rejected tool schema"
    assert events[0].metadata["recoverable"] is False


@pytest.mark.asyncio
async def test_victor_client_stream_chat_renders_error_events_as_content() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())

    class _FakeAgent:
        async def stream(self, _message: str):
            yield AgentExecutionEvent(
                type=EventType.ERROR,
                error="provider rejected tool schema",
                recoverable=False,
            )

    client._agent = _FakeAgent()

    chunks = [chunk async for chunk in client.stream_chat("ping")]

    assert len(chunks) == 1
    assert chunks[0].content == "Error: provider rejected tool schema\n"
    assert chunks[0].metadata["error"] == "provider rejected tool schema"


@pytest.mark.asyncio
async def test_victor_client_stream_delegates_to_shared_stream_helper() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())
    orchestrator = MagicMock()
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=MagicMock()))

    class _FakeAgent:
        def __init__(self):
            self.execution_context = execution_context

        def get_orchestrator(self):
            return orchestrator

    client._agent = _FakeAgent()
    client._context = execution_context

    async def _fake_stream_message_events(**kwargs):
        assert kwargs["orchestrator"] is orchestrator
        assert kwargs["execution_context"] is execution_context
        assert kwargs["user_message"] == "ping"
        yield AgentExecutionEvent(type=EventType.CONTENT, content="READY")

    with patch(
        "victor.framework.client.stream_message_events",
        _fake_stream_message_events,
    ):
        events = [event async for event in client.stream("ping")]

    assert len(events) == 1
    assert events[0].event_type == "content"
    assert events[0].content == "READY"


@pytest.mark.asyncio
async def test_victor_client_create_session_reuses_agent_session_and_closes_service() -> None:
    config = SessionConfig.from_cli_flags(tool_budget=4)
    client = VictorClient(config, container=object())

    mock_orchestrator = MagicMock()
    mock_orchestrator.__class__.__name__ = "AgentOrchestrator"
    mock_orchestrator.provider = MagicMock()
    mock_orchestrator.provider.name = "test-provider"
    mock_orchestrator.model = "test-model"
    mock_orchestrator.messages = []
    mock_orchestrator.get_stage = MagicMock(return_value=MagicMock(value="INITIAL"))
    mock_orchestrator.get_tool_calls_count = MagicMock(return_value=0)
    mock_orchestrator.get_tool_budget = MagicMock(return_value=50)
    mock_orchestrator.get_observed_files = MagicMock(return_value=set())
    mock_orchestrator.get_modified_files = MagicMock(return_value=set())
    mock_orchestrator.get_message_count = MagicMock(return_value=0)
    mock_orchestrator.is_streaming = MagicMock(return_value=False)
    mock_orchestrator.get_iteration_count = MagicMock(return_value=0)
    mock_orchestrator.get_max_iterations = MagicMock(return_value=25)
    mock_orchestrator.reset = MagicMock()
    mock_orchestrator.close = AsyncMock()

    from victor.providers.base import CompletionResponse

    mock_orchestrator.chat = AsyncMock(
        return_value=CompletionResponse(
            content="Session response",
            role="assistant",
            tool_calls=[],
            model="test-model",
        )
    )

    agent = Agent.from_orchestrator(mock_orchestrator)
    session_service = SimpleNamespace(
        create_session=AsyncMock(return_value="session-123"),
        close_session=AsyncMock(return_value=True),
    )
    chat_service = SimpleNamespace(
        reset_conversation=MagicMock(),
        chat=AsyncMock(
            return_value=CompletionResponse(
                content="Session response",
                role="assistant",
                tool_calls=[],
                model="test-model",
            )
        ),
    )
    execution_context = SimpleNamespace(
        services=SimpleNamespace(session=session_service, chat=chat_service)
    )

    client._agent = agent
    client._context = execution_context
    agent._context = execution_context

    session = await client.create_session()

    assert isinstance(session, AgentSession)
    await session.send("Deferred hello")
    chat_service.chat.assert_awaited_once_with("Deferred hello")
    mock_orchestrator.chat.assert_not_awaited()
    chat_service.reset_conversation.assert_called_once_with()
    session_service.create_session.assert_awaited_once()

    await session.close()
    session_service.close_session.assert_awaited_once_with("session-123")


@pytest.mark.parametrize("success", [True, False])
async def test_client_preserves_terminal_loop_verdict(success):
    client = VictorClient(SessionConfig(), container=object())
    metadata = {"agentic_loop_iterations": 2, "agentic_loop_success": success}
    error = None if success else "verification failed"

    class Agent:
        async def stream(self, message):
            yield AgentExecutionEvent(
                type=EventType.STREAM_END, success=success, error=error, metadata=metadata
            )

    client._agent = Agent()
    events = [event async for event in client.stream("q")]
    assert len(events) == 1
    assert events[0].event_type == EventType.STREAM_END
    assert events[0].success is success
    assert events[0].metadata["agentic_loop_iterations"] == 2
    if not success:
        assert events[0].metadata["error"] == error


def test_get_available_providers_returns_static_surface():
    """The static list IS the behavior (self-analysis round 1): the historic
    dynamic lookup imported `list_available_providers` — a symbol defined
    nowhere in the codebase — so the try-path always excepted here."""
    from victor.framework.client import VictorClient

    client = VictorClient.__new__(VictorClient)
    assert client.get_available_providers() == ["anthropic", "openai", "ollama", "google"]


def test_context_window_returns_none_when_no_agent():
    """context_window() returns None when no agent is initialized."""
    config = SessionConfig.from_cli_flags()
    client = VictorClient(config, container=object())
    assert client.context_window() is None


def test_context_window_returns_none_when_no_provider():
    """context_window() returns None when orchestrator has no provider."""
    config = SessionConfig.from_cli_flags()
    client = VictorClient(config, container=object())
    client._agent = SimpleNamespace(_orchestrator=SimpleNamespace(provider=None))
    assert client.context_window() is None


def test_context_window_returns_none_when_no_model():
    """context_window() returns None when model is not set."""
    config = SessionConfig.from_cli_flags()
    client = VictorClient(config, container=object())
    mock_provider = MagicMock()
    mock_provider.context_window = MagicMock(return_value=200000)
    client._agent = SimpleNamespace(
        _orchestrator=SimpleNamespace(provider=mock_provider, model=None)
    )
    assert client.context_window() is None


def test_context_window_returns_value_from_provider():
    """context_window() returns the provider's context window for the model."""
    config = SessionConfig.from_cli_flags()
    client = VictorClient(config, container=object())
    mock_provider = MagicMock()
    mock_provider.context_window = MagicMock(return_value=200000)
    client._agent = SimpleNamespace(
        _orchestrator=SimpleNamespace(provider=mock_provider, model="claude-3-sonnet")
    )
    assert client.context_window() == 200000
    mock_provider.context_window.assert_called_once_with("claude-3-sonnet")


def test_context_window_returns_none_on_exception():
    """context_window() returns None when provider raises an exception."""
    config = SessionConfig.from_cli_flags()
    client = VictorClient(config, container=object())
    mock_provider = MagicMock()
    mock_provider.context_window = MagicMock(side_effect=ValueError("unknown model"))
    client._agent = SimpleNamespace(
        _orchestrator=SimpleNamespace(provider=mock_provider, model="unknown-model")
    )
    assert client.context_window() is None
