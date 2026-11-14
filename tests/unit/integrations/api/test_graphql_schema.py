"""Tests for GraphQL schema integration."""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from victor import __version__

import pytest

strawberry = pytest.importorskip("strawberry")

from victor.integrations.api.graphql_schema import (
    AgentEventType,
    ChatResponseType,
    HealthType,
    create_graphql_schema,
)
from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType, EventBroadcaster
import victor.integrations.api.graphql_schema as graphql_schema_module


@pytest.fixture
def mock_server():
    """Create a mock VictorFastAPIServer."""
    server = MagicMock()
    server.workspace_root = "/test/workspace"
    server._orchestrator = None
    server._event_bridge = None

    # Mock _get_orchestrator
    mock_orchestrator = MagicMock()
    mock_orchestrator.provider = MagicMock()
    mock_orchestrator.provider.name = "anthropic"
    mock_orchestrator.provider.model = "claude-3-sonnet"
    mock_orchestrator.adaptive_controller = None
    mock_orchestrator.chat = AsyncMock(return_value=MagicMock(content="Hello!", tool_calls=[]))
    mock_orchestrator.reset_conversation = MagicMock()
    server._get_orchestrator = AsyncMock(return_value=mock_orchestrator)

    # Mock tool category helper
    server._get_tool_category = MagicMock(return_value="general")

    return server


@pytest.fixture
def schema(mock_server):
    """Create a test GraphQL schema."""
    return create_graphql_schema(mock_server)


class TestSchemaCreation:
    """Tests for GraphQL schema creation."""

    def test_schema_creation(self, schema):
        """Schema should be a valid strawberry Schema."""
        assert isinstance(schema, strawberry.Schema)

    def test_schema_has_query_type(self, schema):
        """Schema should have query type."""
        # Introspection check — schema should have query via internal _schema
        assert schema._schema is not None
        assert schema._schema.query_type is not None

    def test_schema_introspection(self, schema):
        """Introspection query should succeed."""
        result = schema.execute_sync("""
            {
                __schema {
                    queryType { name }
                    mutationType { name }
                    subscriptionType { name }
                }
            }
            """)
        assert result.errors is None
        data = result.data["__schema"]
        assert data["queryType"]["name"] == "Query"
        assert data["mutationType"]["name"] == "Mutation"
        assert data["subscriptionType"]["name"] == "Subscription"


class TestHealthQuery:
    """Tests for health query."""

    @pytest.mark.asyncio
    async def test_health_query(self, schema):
        """Health query should return status and version."""
        result = await schema.execute("{ health { status version } }")
        assert result.errors is None
        assert result.data["health"]["status"] == "healthy"
        assert result.data["health"]["version"] == __version__


class TestStatusQuery:
    """Tests for status query."""

    @pytest.mark.asyncio
    async def test_status_query(self, schema):
        """Status query should return server status."""
        result = await schema.execute("{ status { connected mode provider model workspace } }")
        assert result.errors is None
        data = result.data["status"]
        assert data["connected"] is True
        assert data["provider"] == "anthropic"
        assert data["model"] == "claude-3-sonnet"
        assert data["workspace"] == "/test/workspace"


class TestToolsQuery:
    """Tests for tools query."""

    @pytest.mark.asyncio
    async def test_tools_query(self, schema):
        """Tools query should return list of ToolInfoType."""
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"

        with patch("victor.tools.registry.ToolRegistry") as MockRegistry:
            MockRegistry.return_value.list_tools.return_value = [mock_tool]
            result = await schema.execute("{ tools { name description category } }")

        assert result.errors is None
        tools = result.data["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"
        assert tools[0]["description"] == "Read a file"


class TestChatMutation:
    """Tests for chat mutation."""

    @pytest.mark.asyncio
    async def test_chat_mutation(self, schema):
        """Chat mutation should return ChatResponseType."""
        result = await schema.execute("""
            mutation {
                chat(messages: [{role: "user", content: "Hello"}]) {
                    role
                    content
                    toolCalls
                }
            }
            """)
        assert result.errors is None
        data = result.data["chat"]
        assert data["role"] == "assistant"
        assert data["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_empty_messages(self, schema):
        """Chat with empty messages should return empty response."""
        result = await schema.execute("""
            mutation {
                chat(messages: []) {
                    role
                    content
                }
            }
            """)
        assert result.errors is None
        assert result.data["chat"]["content"] == ""


class TestSwitchModelMutation:
    """Tests for switch_model mutation."""

    @pytest.mark.asyncio
    async def test_switch_model(self, schema):
        """Switch model should return success boolean."""
        with patch("victor.agent.model_switcher.get_model_switcher") as mock_get:
            mock_switcher = MagicMock()
            mock_get.return_value = mock_switcher

            result = await schema.execute("""
                mutation {
                    switchModel(provider: "anthropic", model: "claude-3-opus")
                }
                """)

        assert result.errors is None
        assert result.data["switchModel"] is True


class TestResetConversationMutation:
    """Tests for reset_conversation mutation."""

    @pytest.mark.asyncio
    async def test_reset_conversation(self, schema, mock_server):
        """Reset conversation should return success."""
        mock_server._orchestrator = MagicMock()
        result = await schema.execute("""
            mutation {
                resetConversation
            }
            """)
        assert result.errors is None
        assert result.data["resetConversation"] is True


class TestGraphQLDisabledWithoutStrawberry:
    """Test graceful fallback when strawberry is not installed."""

    def test_graphql_disabled_without_strawberry(self):
        """Server should start without GraphQL when strawberry is unavailable."""
        with patch.dict("sys.modules", {"strawberry": None, "strawberry.fastapi": None}):
            # The import guard in fastapi_server.py catches ImportError
            # so the server should still initialize. We verify the pattern.
            try:
                from victor.integrations.api.graphql_schema import (
                    create_graphql_schema,  # noqa: F401
                )
            except (ImportError, ModuleNotFoundError):
                pass  # Expected when strawberry is not available


class TestGraphQLTypes:
    """Tests for GraphQL type definitions."""

    def test_health_type_fields(self):
        """HealthType should have status and version fields."""
        h = HealthType(status="ok", version="1.0")
        assert h.status == "ok"
        assert h.version == "1.0"

    def test_agent_event_type_defaults(self):
        """AgentEventType should have sensible defaults."""
        e = AgentEventType(type="content", content="hello")
        assert e.tool_name is None
        assert e.error is None

    def test_chat_response_type(self):
        """ChatResponseType should hold role, content, and optional tool_calls."""
        r = ChatResponseType(role="assistant", content="hi", tool_calls=None)
        assert r.role == "assistant"
        assert r.tool_calls is None


class TestGraphQLSubscriptionQueueBounds:
    """Prove a stalled GraphQL consumer cannot retain events without limit."""

    async def test_slow_subscriber_drops_oldest_without_blocking_fast_subscriber(
        self, mock_server, monkeypatch
    ):
        monkeypatch.setattr(graphql_schema_module, "GRAPHQL_SUBSCRIPTION_QUEUE_SIZE", 2)
        EventBroadcaster.reset_instance()
        broadcaster = EventBroadcaster()
        mock_server._event_bridge = MagicMock()
        mock_server._event_bridge._broadcaster = broadcaster
        schema = create_graphql_schema(mock_server)
        query = """
            subscription {
              agentEvents { type content error timestamp }
            }
        """

        slow_iterator = await schema.subscribe(query)
        fast_iterator = await schema.subscribe(query)
        slow_first = asyncio.create_task(slow_iterator.__anext__())
        fast_first = asyncio.create_task(fast_iterator.__anext__())
        for _ in range(1000):
            if broadcaster.client_count == 2:
                break
            await asyncio.sleep(0)
        assert broadcaster.client_count == 2

        async def receive(iterator, count):
            values = []
            for _ in range(count):
                result = await asyncio.wait_for(iterator.__anext__(), timeout=2)
                assert result.errors is None
                values.append(result.data["agentEvents"]["content"])
            return values

        class PayloadEvent(BridgeEvent):
            def to_dict(self) -> dict[str, Any]:
                return {
                    "id": self.id,
                    "type": self.type.value,
                    **self.data,
                    "timestamp": self.timestamp,
                }

        await broadcaster._delivery._send_to_clients(
            PayloadEvent(
                type=BridgeEventType.NOTIFICATION,
                data={"content": "one", "timestamp": time.time()},
            )
        )
        for first in (slow_first, fast_first):
            first_result = await asyncio.wait_for(first, timeout=2)
            assert first_result.errors is None
            assert first_result.data["agentEvents"]["content"] == "one"

        fast_receiver = asyncio.create_task(receive(fast_iterator, 3))

        # This is the same per-client delivery boundary used by the engine after
        # its bounded sender queue. The resolver-local queue is the final
        # slow-consumer retention point and must therefore also be bounded.
        for value in ("two", "three", "four"):
            await broadcaster._delivery._send_to_clients(
                PayloadEvent(
                    type=BridgeEventType.NOTIFICATION,
                    data={"content": value, "timestamp": time.time()},
                )
            )
            # Let the actively consuming fast subscriber drain its queue; the
            # stalled slow subscriber remains the only retention pressure point.
            await asyncio.sleep(0)

        assert await fast_receiver == ["two", "three", "four"]
        slow_next = await asyncio.wait_for(slow_iterator.__anext__(), timeout=2)
        assert slow_next.errors is None
        assert slow_next.data["agentEvents"]["content"] == "three"
        assert broadcaster.get_reliability_dashboard()["queue_dropped_events"] >= 1

        await slow_iterator.aclose()
        await fast_iterator.aclose()
        assert broadcaster.client_count == 0
