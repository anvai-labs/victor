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

"""Integration tests for multi-agent communication.

These tests verify the messaging and communication infrastructure
for multi-agent teams, including:
- Broadcast messaging to all team members
- Direct messaging between specific agents
- Message routing and delivery
- Response aggregation
- Message bus functionality

Tests exercise the real implementations in:
- victor/agent/teams/communication.py
- victor/framework/team_coordinator.py
- victor/framework/agent_protocols.py
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pytest

from victor.framework.agent_protocols import (
    AgentCapability,
    AgentMessage,
    IAgentRole,
    MessageType,
    TeamFormation,
)
from victor.teams import (
    UnifiedTeamCoordinator,
    AgentMessage as TeamAgentMessage,
    MessageType as TeamMessageType,
    TeamMessageBus,
    TeamSharedMemory,
)

# =============================================================================
# Test Fixtures and Mock Implementations
# =============================================================================


@dataclass
class CommunicationTestRole:
    """Role for communication testing."""

    name: str = "comm_test_role"
    capabilities: Set[AgentCapability] = field(
        default_factory=lambda: {AgentCapability.COMMUNICATE}
    )
    allowed_tools: Set[str] = field(default_factory=set)
    tool_budget: int = 10

    def get_system_prompt_section(self) -> str:
        return f"You are a {self.name} for communication testing."


class CommunicationTestAgent:
    """Test agent focused on communication capabilities."""

    def __init__(
        self,
        agent_id: str,
        response_delay: float = 0.0,
        response_content: Optional[str] = None,
    ):
        self._id = agent_id
        self._role = CommunicationTestRole(name=agent_id)
        self._persona = None
        self.response_delay = response_delay
        self.response_content = response_content

        # Tracking
        self.received_messages: List[AgentMessage] = []
        self.sent_responses: List[AgentMessage] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> IAgentRole:
        return self._role

    @property
    def persona(self):
        return self._persona

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        return f"Task executed by {self.id}"

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and respond to a message."""
        self.received_messages.append(message)

        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        content = self.response_content or f"Response from {self.id}: {message.content}"
        response = AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content=content,
            message_type=MessageType.RESULT,
            data={"original_message_type": message.message_type.value},
        )
        self.sent_responses.append(response)
        return response


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator() -> UnifiedTeamCoordinator:
    """Create a fresh coordinator."""
    return UnifiedTeamCoordinator()


@pytest.fixture
def message_bus() -> TeamMessageBus:
    """Create a message bus for testing."""
    return TeamMessageBus("test_team")


@pytest.fixture
def shared_memory() -> TeamSharedMemory:
    """Create shared memory for testing."""
    return TeamSharedMemory()


@pytest.fixture
def three_comm_agents() -> List[CommunicationTestAgent]:
    """Create three agents for communication tests."""
    return [
        CommunicationTestAgent("agent_1"),
        CommunicationTestAgent("agent_2"),
        CommunicationTestAgent("agent_3"),
    ]


# =============================================================================
# Broadcast Messaging Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestBroadcastMessaging:
    """Integration tests for broadcast messaging."""

    @pytest.mark.asyncio
    async def test_broadcast_delivers_to_all_members(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_comm_agents: List[CommunicationTestAgent],
    ):
        """Broadcast delivers message to all team members."""
        for agent in three_comm_agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Status update required",
            message_type=MessageType.QUERY,
        )

        await coordinator.broadcast(message)

        for agent in three_comm_agents:
            assert len(agent.received_messages) == 1
            assert agent.received_messages[0].content == "Status update required"

    @pytest.mark.asyncio
    async def test_broadcast_returns_all_responses(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_comm_agents: List[CommunicationTestAgent],
    ):
        """Broadcast returns responses from all members."""
        for agent in three_comm_agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Report status",
            message_type=MessageType.QUERY,
        )

        responses = await coordinator.broadcast(message)

        assert len(responses) == 3
        assert all(r is not None for r in responses)

    @pytest.mark.asyncio
    async def test_broadcast_responses_contain_correct_senders(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_comm_agents: List[CommunicationTestAgent],
    ):
        """Broadcast responses correctly identify senders."""
        for agent in three_comm_agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Identify yourselves",
            message_type=MessageType.QUERY,
        )

        responses = await coordinator.broadcast(message)

        sender_ids = {r.sender_id for r in responses if r}
        expected_ids = {"agent_1", "agent_2", "agent_3"}
        assert sender_ids == expected_ids

    @pytest.mark.asyncio
    async def test_broadcast_preserves_message_type(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_comm_agents: List[CommunicationTestAgent],
    ):
        """Broadcast preserves the original message type."""
        for agent in three_comm_agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Task assignment",
            message_type=MessageType.TASK,
        )

        await coordinator.broadcast(message)

        for agent in three_comm_agents:
            assert agent.received_messages[0].message_type == MessageType.TASK

    @pytest.mark.asyncio
    async def test_broadcast_preserves_metadata(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_comm_agents: List[CommunicationTestAgent],
    ):
        """Broadcast preserves message metadata."""
        for agent in three_comm_agents:
            coordinator.add_member(agent)

        msg_data = {"priority": "high", "deadline": "2h"}
        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Urgent task",
            message_type=MessageType.TASK,
            data=msg_data,
        )

        await coordinator.broadcast(message)

        for agent in three_comm_agents:
            received_meta = agent.received_messages[0].metadata
            assert received_meta["priority"] == "high"
            assert received_meta["deadline"] == "2h"


# =============================================================================
# Direct Messaging Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestDirectMessaging:
    """Integration tests for direct messaging between agents."""

    @pytest.mark.asyncio
    async def test_direct_message_to_specific_agent(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_comm_agents: List[CommunicationTestAgent],
    ):
        """Direct message is delivered only to specified agent."""
        for agent in three_comm_agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="agent_2",
            content="Private instruction",
            message_type=MessageType.TASK,
        )

        await coordinator.send_message(message)

        # Only agent_2 should receive the message
        assert len(three_comm_agents[0].received_messages) == 0
        assert len(three_comm_agents[1].received_messages) == 1
        assert len(three_comm_agents[2].received_messages) == 0

    @pytest.mark.asyncio
    async def test_direct_message_returns_response(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Direct message returns response from recipient."""
        agent = CommunicationTestAgent(
            "responder",
            response_content="Received and understood",
        )
        coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="responder",
            content="Confirm receipt",
            message_type=MessageType.QUERY,
        )

        response = await coordinator.send_message(message)

        assert response is not None
        assert response.content == "Received and understood"

    @pytest.mark.asyncio
    async def test_direct_message_to_nonexistent_agent(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_comm_agents: List[CommunicationTestAgent],
    ):
        """Direct message to nonexistent agent returns None."""
        for agent in three_comm_agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="nonexistent_agent",
            content="Message to nobody",
            message_type=MessageType.QUERY,
        )

        response = await coordinator.send_message(message)

        assert response is None

    @pytest.mark.asyncio
    async def test_direct_message_response_type(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Direct message response has correct type."""
        agent = CommunicationTestAgent("responder")
        coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="responder",
            content="Query",
            message_type=MessageType.QUERY,
        )

        response = await coordinator.send_message(message)

        assert response.message_type == MessageType.RESULT


# =============================================================================
# TeamMessageBus Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestTeamMessageBus:
    """Integration tests for TeamMessageBus."""

    @pytest.mark.asyncio
    async def test_message_bus_register_agent(
        self,
        message_bus: TeamMessageBus,
    ):
        """Agents can be registered on message bus."""
        message_bus.register_agent("agent_1")
        message_bus.register_agent("agent_2")

        # Agents should have queues
        assert message_bus.get_pending_count("agent_1") == 0
        assert message_bus.get_pending_count("agent_2") == 0

    @pytest.mark.asyncio
    async def test_message_bus_send_directed_message(
        self,
        message_bus: TeamMessageBus,
    ):
        """Message bus delivers directed messages."""
        message_bus.register_agent("sender")
        message_bus.register_agent("receiver")

        message = TeamAgentMessage(
            message_type=TeamMessageType.DISCOVERY,
            sender_id="sender",
            recipient_id="receiver",
            content="Found important code",
        )

        await message_bus.send(message)

        # Receiver should have the message
        received = await message_bus.receive("receiver", timeout=1.0)
        assert received is not None
        assert received.content == "Found important code"

    @pytest.mark.asyncio
    async def test_message_bus_broadcast(
        self,
        message_bus: TeamMessageBus,
    ):
        """Message bus broadcasts to all agents except sender."""
        message_bus.register_agent("sender")
        message_bus.register_agent("receiver_1")
        message_bus.register_agent("receiver_2")

        message = TeamAgentMessage(
            sender_id="sender",
            recipient_id=None,  # broadcast
            content="Status update",
            message_type=TeamMessageType.STATUS,
        )

        await message_bus.send(message)

        # Receivers should have the message
        msg1 = await message_bus.receive("receiver_1", timeout=1.0)
        msg2 = await message_bus.receive("receiver_2", timeout=1.0)
        sender_msg = await message_bus.receive("sender", timeout=0)

        assert msg1 is not None
        assert msg2 is not None
        assert sender_msg is None  # Sender doesn't receive their own broadcast

    @pytest.mark.asyncio
    async def test_message_bus_logs_messages(
        self,
        message_bus: TeamMessageBus,
    ):
        """Message bus maintains message log."""
        message_bus.register_agent("agent_1")
        message_bus.register_agent("agent_2")

        for i in range(5):
            await message_bus.send(
                TeamAgentMessage(
                    message_type=TeamMessageType.DISCOVERY,
                    sender_id="agent_1",
                    recipient_id="agent_2",
                    content=f"Message {i}",
                )
            )

        log = message_bus.get_message_log()
        assert len(log) == 5

    @pytest.mark.asyncio
    async def test_message_bus_filter_by_type(
        self,
        message_bus: TeamMessageBus,
    ):
        """Message log can be filtered by type."""
        message_bus.register_agent("agent")

        # Send different types
        await message_bus.send(
            TeamAgentMessage(
                sender_id="agent",
                recipient_id=None,  # broadcast
                content="Discovery",
                message_type=TeamMessageType.DISCOVERY,
            )
        )
        await message_bus.send(
            TeamAgentMessage(
                sender_id="agent",
                recipient_id=None,  # broadcast
                content="Status",
                message_type=TeamMessageType.STATUS,
            )
        )
        await message_bus.send(
            TeamAgentMessage(
                sender_id="agent",
                recipient_id=None,  # broadcast
                content="Another discovery",
                message_type=TeamMessageType.DISCOVERY,
            )
        )

        discoveries = message_bus.get_message_log(message_type=TeamMessageType.DISCOVERY)
        assert len(discoveries) == 2

    @pytest.mark.asyncio
    async def test_message_bus_timeout(
        self,
        message_bus: TeamMessageBus,
    ):
        """Receive times out when no message available."""
        message_bus.register_agent("agent")

        received = await message_bus.receive("agent", timeout=0.1)
        assert received is None

    @pytest.mark.asyncio
    async def test_message_bus_subscribe(
        self,
        message_bus: TeamMessageBus,
    ):
        """Subscribers are notified of specific message types."""
        message_bus.register_agent("agent")

        received_alerts = []

        def on_alert(msg):
            received_alerts.append(msg)

        message_bus.subscribe(TeamMessageType.ALERT, on_alert)

        await message_bus.send(
            TeamAgentMessage(
                sender_id="agent",
                recipient_id=None,  # broadcast
                content="Important alert",
                message_type=TeamMessageType.ALERT,
            )
        )

        assert len(received_alerts) == 1
        assert received_alerts[0].content == "Important alert"

    @pytest.mark.asyncio
    async def test_message_bus_context_summary(
        self,
        message_bus: TeamMessageBus,
    ):
        """Message bus generates context summary."""
        message_bus.register_agent("researcher")
        message_bus.register_agent("executor")

        await message_bus.send(
            TeamAgentMessage(
                sender_id="researcher",
                recipient_id=None,  # broadcast
                content="Found authentication module",
                message_type=TeamMessageType.DISCOVERY,
            )
        )

        summary = message_bus.get_context_summary()
        assert "Team Communication" in summary
        assert "researcher" in summary


# =============================================================================
# TeamSharedMemory Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestTeamSharedMemory:
    """Integration tests for TeamSharedMemory."""

    def test_shared_memory_set_and_get(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Shared memory stores and retrieves values."""
        shared_memory.set("api_endpoints", ["/login", "/logout"], "researcher")

        endpoints = shared_memory.get("api_endpoints")
        assert endpoints == ["/login", "/logout"]

    def test_shared_memory_append(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Shared memory supports appending to lists."""
        shared_memory.append("files_analyzed", "auth.py", "researcher")
        shared_memory.append("files_analyzed", "login.py", "researcher")
        shared_memory.append("files_analyzed", "session.py", "executor")

        files = shared_memory.get("files_analyzed")
        assert files == ["auth.py", "login.py", "session.py"]

    def test_shared_memory_tracks_contributors(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Shared memory tracks which agents contributed."""
        shared_memory.set("key1", "value1", "agent_1")
        shared_memory.set("key1", "value2", "agent_2")

        contributors = shared_memory.get_contributors("key1")
        assert "agent_1" in contributors
        assert "agent_2" in contributors

    def test_shared_memory_update_dict(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Shared memory supports updating dictionaries."""
        shared_memory.update("config", {"host": "localhost"}, "agent_1")
        shared_memory.update("config", {"port": 8080}, "agent_2")

        config = shared_memory.get("config")
        assert config == {"host": "localhost", "port": 8080}

    def test_shared_memory_has_key(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Shared memory checks key existence."""
        shared_memory.set("exists", True, "agent")

        assert shared_memory.has("exists") is True
        assert shared_memory.has("nonexistent") is False

    def test_shared_memory_get_all(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Shared memory returns all data."""
        shared_memory.set("key1", "value1", "agent")
        shared_memory.set("key2", "value2", "agent")

        all_data = shared_memory.get_all()
        assert all_data == {"key1": "value1", "key2": "value2"}

    def test_shared_memory_summary(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Shared memory generates summary."""
        shared_memory.set("findings", "Important discovery", "researcher")

        summary = shared_memory.get_summary()
        assert "Shared Team Knowledge" in summary
        assert "findings" in summary
        assert "researcher" in summary

    def test_shared_memory_clear(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Shared memory can be cleared."""
        shared_memory.set("key", "value", "agent")
        shared_memory.clear()

        assert shared_memory.get("key") is None
        assert len(shared_memory.keys()) == 0


# =============================================================================
# Message Routing Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestMessageRouting:
    """Integration tests for message routing logic."""

    @pytest.mark.asyncio
    async def test_message_routing_by_recipient_id(
        self,
        message_bus: TeamMessageBus,
    ):
        """Messages are routed by recipient ID."""
        message_bus.register_agent("agent_a")
        message_bus.register_agent("agent_b")
        message_bus.register_agent("agent_c")

        # Send to agent_b specifically
        await message_bus.send(
            TeamAgentMessage(
                message_type=TeamMessageType.REQUEST,
                sender_id="agent_a",
                recipient_id="agent_b",
                content="Only for B",
            )
        )

        msg_a = await message_bus.receive("agent_a", timeout=0)
        msg_b = await message_bus.receive("agent_b", timeout=0)
        msg_c = await message_bus.receive("agent_c", timeout=0)

        assert msg_a is None
        assert msg_b is not None
        assert msg_c is None

    @pytest.mark.asyncio
    async def test_message_routing_preserves_order(
        self,
        message_bus: TeamMessageBus,
    ):
        """Messages are delivered in order."""
        message_bus.register_agent("sender")
        message_bus.register_agent("receiver")

        for i in range(5):
            await message_bus.send(
                TeamAgentMessage(
                    message_type=TeamMessageType.STATUS,
                    sender_id="sender",
                    recipient_id="receiver",
                    content=f"Message {i}",
                )
            )

        for i in range(5):
            msg = await message_bus.receive("receiver", timeout=1.0)
            assert msg.content == f"Message {i}"

    @pytest.mark.asyncio
    async def test_message_routing_with_replies(
        self,
        message_bus: TeamMessageBus,
    ):
        """Messages can reference replies."""
        message_bus.register_agent("requester")
        message_bus.register_agent("responder")

        original = TeamAgentMessage(
            message_type=TeamMessageType.REQUEST,
            sender_id="requester",
            recipient_id="responder",
            content="What is the status?",
        )
        await message_bus.send(original)

        # Simulate response
        response = TeamAgentMessage(
            message_type=TeamMessageType.RESPONSE,
            sender_id="responder",
            recipient_id="requester",
            content="All good",
            reply_to=original.id,
        )
        await message_bus.send(response)

        # Receive the response
        received = await message_bus.receive("requester", timeout=1.0)
        assert received is not None
        assert received.reply_to == original.id


# =============================================================================
# Response Aggregation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestResponseAggregation:
    """Integration tests for response aggregation."""

    @pytest.mark.asyncio
    async def test_aggregate_broadcast_responses(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Broadcast responses can be aggregated."""
        agents = [
            CommunicationTestAgent("agent_1", response_content="OK"),
            CommunicationTestAgent("agent_2", response_content="OK"),
            CommunicationTestAgent("agent_3", response_content="FAIL"),
        ]

        for agent in agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Status check",
            message_type=MessageType.QUERY,
        )

        responses = await coordinator.broadcast(message)

        # Aggregate responses
        ok_count = sum(1 for r in responses if r and "OK" in r.content)
        fail_count = sum(1 for r in responses if r and "FAIL" in r.content)

        assert ok_count == 2
        assert fail_count == 1

    @pytest.mark.asyncio
    async def test_aggregate_with_varying_delays(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Aggregation handles varying response times."""
        agents = [
            CommunicationTestAgent("fast", response_delay=0.01),
            CommunicationTestAgent("medium", response_delay=0.05),
            CommunicationTestAgent("slow", response_delay=0.1),
        ]

        for agent in agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Response time test",
            message_type=MessageType.QUERY,
        )

        responses = await coordinator.broadcast(message)

        # All responses collected despite different delays
        assert len(responses) == 3
        assert all(r is not None for r in responses)

    @pytest.mark.asyncio
    async def test_aggregate_metadata_from_responses(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Response metadata can be aggregated."""
        agents = [
            CommunicationTestAgent("agent_1"),
            CommunicationTestAgent("agent_2"),
        ]

        for agent in agents:
            coordinator.add_member(agent)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Query",
            message_type=MessageType.QUERY,
        )

        responses = await coordinator.broadcast(message)

        # All responses should have metadata about original message
        for response in responses:
            assert "original_message_type" in response.metadata
            assert response.metadata["original_message_type"] == "query"


# =============================================================================
# Integration with Team Execution
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestCommunicationWithTeamExecution:
    """Tests combining communication with team execution."""

    @pytest.mark.asyncio
    async def test_communication_during_sequential_execution(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Agents can receive messages during sequential execution."""
        agent1 = CommunicationTestAgent("agent_1")
        agent2 = CommunicationTestAgent("agent_2")

        coordinator.add_member(agent1).add_member(agent2)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        # Execute task
        await coordinator.execute_task("Test task", {})

        # Now communicate
        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="Post-execution message",
            message_type=MessageType.FEEDBACK,
        )
        responses = await coordinator.broadcast(message)

        assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_message_bus_integration_with_coordinator(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Message bus works alongside team coordinator."""
        bus = TeamMessageBus("test_team")

        agent1 = CommunicationTestAgent("agent_1")
        agent2 = CommunicationTestAgent("agent_2")

        coordinator.add_member(agent1).add_member(agent2)
        bus.register_agent("agent_1")
        bus.register_agent("agent_2")

        # Execute via coordinator
        await coordinator.execute_task("Team task", {})

        # Communicate via bus
        await bus.send(
            TeamAgentMessage(
                message_type=TeamMessageType.RESULT,
                sender_id="agent_1",
                recipient_id="agent_2",
                content="Execution complete",
            )
        )

        msg = await bus.receive("agent_2", timeout=1.0)
        assert msg is not None
        assert msg.content == "Execution complete"


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestCommunicationErrorHandling:
    """Tests for communication error handling."""

    @pytest.mark.asyncio
    async def test_unregistered_sender_raises_error(
        self,
        message_bus: TeamMessageBus,
    ):
        """Sending from unregistered agent raises error."""
        message_bus.register_agent("receiver")

        with pytest.raises(ValueError, match="not registered"):
            await message_bus.send(
                TeamAgentMessage(
                    message_type=TeamMessageType.STATUS,
                    sender_id="unregistered",
                    recipient_id="receiver",
                    content="Test",
                )
            )

    @pytest.mark.asyncio
    async def test_unregistered_receiver_logged(
        self,
        message_bus: TeamMessageBus,
    ):
        """Sending to unregistered agent logs warning but doesn't raise."""
        message_bus.register_agent("sender")

        # Should not raise, just log warning
        await message_bus.send(
            TeamAgentMessage(
                message_type=TeamMessageType.STATUS,
                sender_id="sender",
                recipient_id="unregistered",
                content="Test",
            )
        )

        # Message should still be in log
        log = message_bus.get_message_log()
        assert len(log) == 1

    def test_append_to_non_list_raises_error(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Appending to non-list value raises TypeError."""
        shared_memory.set("not_a_list", "string_value", "agent")

        with pytest.raises(TypeError, match="Cannot append"):
            shared_memory.append("not_a_list", "new_value", "agent")

    def test_update_non_dict_raises_error(
        self,
        shared_memory: TeamSharedMemory,
    ):
        """Updating non-dict value raises TypeError."""
        shared_memory.set("not_a_dict", ["list", "value"], "agent")

        with pytest.raises(TypeError, match="Cannot update"):
            shared_memory.update("not_a_dict", {"key": "value"}, "agent")
