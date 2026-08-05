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

"""Concurrent Observability Tests for Multi-Agent Systems.

This module tests observability event emission during concurrent agent execution,
particularly relevant for VSCode extension and WebSocket API integrations.

Use Cases:
1. VSCode Extension receiving events via WebSocket
2. Architect API broadcasting progress updates
3. Multiple concurrent teams emitting events
4. Event ordering and correlation

Test Scenarios:
- Concurrent agent execution emits properly ordered events
- Events include correlation IDs for tracking
- WebSocket subscribers receive all events
- Event aggregation works under load
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.teams import (
    AgentMessage,
    ITeamCoordinator,
    MessageType,
    TeamFormation,
    UnifiedTeamCoordinator,
    create_coordinator,
)

# =============================================================================
# Lightweight Event Bus for Testing
# =============================================================================


@dataclass
class TestEventBus:
    """Lightweight event bus for testing observability patterns.

    This is a simplified event bus that demonstrates the pattern
    without the full complexity of the production EventBus.
    """

    subscribers: Dict[str, List[Callable]] = field(default_factory=lambda: defaultdict(list))
    all_events: List[Dict[str, Any]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def subscribe(
        self,
        event_pattern: str,
        handler: Callable[[str, Dict[str, Any]], None],
    ) -> Callable[[], None]:
        """Subscribe to events matching a pattern."""
        self.subscribers[event_pattern].append(handler)

        def unsubscribe():
            self.subscribers[event_pattern].remove(handler)

        return unsubscribe

    async def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to all matching subscribers."""
        async with self.lock:
            event = {"type": event_type, "data": data}
            self.all_events.append(event)

            # Notify matching subscribers
            for pattern, handlers in self.subscribers.items():
                if self._matches(event_type, pattern):
                    for handler in handlers:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event_type, data)
                        else:
                            handler(event_type, data)

    def _matches(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches pattern (supports * wildcard)."""
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix)
        return event_type == pattern


# =============================================================================
# Mock Event Subscriber (simulates VSCode WebSocket client)
# =============================================================================


@dataclass
class MockEventSubscriber:
    """Mock event subscriber simulating a WebSocket client.

    This simulates what a VSCode extension would receive when subscribing
    to agent execution events via the Architect WebSocket API.
    """

    name: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    event_types_received: Set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def on_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle incoming event (simulates WebSocket message handler)."""
        async with self.lock:
            self.events.append(
                {
                    "type": event_type,
                    "data": data,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )
            self.event_types_received.add(event_type)

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Filter events by type."""
        return [e for e in self.events if e["type"] == event_type]

    def get_events_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Filter events for a specific agent."""
        return [e for e in self.events if e.get("data", {}).get("agent_id") == agent_id]


# =============================================================================
# Mock Agent for Observability Testing
# =============================================================================


@dataclass
class ObservableAgent:
    """Agent that emits observability events during execution."""

    id: str
    event_bus: TestEventBus
    output: str = "Task completed"
    delay: float = 0.1
    events_emitted: List[str] = field(default_factory=list)

    @property
    def role(self) -> MagicMock:
        return MagicMock(name=self.id)

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute task while emitting observability events."""
        correlation_id = context.get("correlation_id", "unknown")

        # Emit start event
        await self._emit(
            "agent.execution.started",
            {
                "agent_id": self.id,
                "correlation_id": correlation_id,
                "task": task,
            },
        )

        # Simulate work
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Emit completion event
        await self._emit(
            "agent.execution.completed",
            {
                "agent_id": self.id,
                "correlation_id": correlation_id,
                "output": self.output,
            },
        )

        return self.output

    async def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to the event bus."""
        self.events_emitted.append(event_type)
        await self.event_bus.emit(event_type, data)

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive message (protocol compliance)."""
        return None


# =============================================================================
# Tests: Basic Event Emission
# =============================================================================


@pytest.mark.unit
class TestBasicEventEmission:
    """Test basic event emission from agents."""

    @pytest.fixture
    def event_bus(self) -> TestEventBus:
        """Create fresh event bus."""
        return TestEventBus()

    @pytest.fixture
    def subscriber(self) -> MockEventSubscriber:
        """Create mock subscriber."""
        return MockEventSubscriber("vscode_client")

    @pytest.mark.asyncio
    async def test_agent_emits_start_and_complete_events(self, event_bus, subscriber):
        """Agent should emit start and complete events during execution."""
        # Arrange: Subscribe to events
        await event_bus.subscribe("agent.execution.*", subscriber.on_event)
        agent = ObservableAgent(id="test_agent", event_bus=event_bus, delay=0.01)

        # Act: Execute agent
        await agent.execute_task("Test task", {"correlation_id": "test-123"})

        # Assert: Both events received
        assert "agent.execution.started" in subscriber.event_types_received
        assert "agent.execution.completed" in subscriber.event_types_received
        assert len(subscriber.events) == 2

    @pytest.mark.asyncio
    async def test_events_include_correlation_id(self, event_bus, subscriber):
        """Events should include correlation ID for tracking."""
        # Arrange
        await event_bus.subscribe("agent.execution.*", subscriber.on_event)
        agent = ObservableAgent(id="corr_test", event_bus=event_bus, delay=0.01)

        # Act
        await agent.execute_task("Task", {"correlation_id": "track-456"})

        # Assert: All events have correlation ID
        for event in subscriber.events:
            assert event["data"]["correlation_id"] == "track-456"


# =============================================================================
# Tests: Concurrent Agent Events
# =============================================================================


@pytest.mark.unit
class TestConcurrentAgentEvents:
    """Test event emission during concurrent agent execution."""

    @pytest.fixture
    def event_bus(self) -> TestEventBus:
        return TestEventBus()

    @pytest.fixture
    def subscriber(self) -> MockEventSubscriber:
        return MockEventSubscriber("concurrent_test_client")

    @pytest.mark.asyncio
    async def test_concurrent_agents_emit_ordered_events(self, event_bus, subscriber):
        """Events from concurrent agents should maintain order within each agent."""
        # Arrange: Create multiple agents
        await event_bus.subscribe("agent.execution.*", subscriber.on_event)

        agents = [
            ObservableAgent(id=f"agent_{i}", event_bus=event_bus, delay=0.05) for i in range(3)
        ]

        # Act: Execute all agents concurrently
        await asyncio.gather(
            *[
                agent.execute_task(f"Task {i}", {"correlation_id": f"concurrent-{i}"})
                for i, agent in enumerate(agents)
            ]
        )

        # Assert: Each agent has start before complete
        for i in range(3):
            agent_events = [
                e for e in subscriber.events if e["data"].get("agent_id") == f"agent_{i}"
            ]
            assert len(agent_events) == 2
            assert agent_events[0]["type"] == "agent.execution.started"
            assert agent_events[1]["type"] == "agent.execution.completed"

    @pytest.mark.asyncio
    async def test_high_concurrency_event_delivery(self, event_bus, subscriber):
        """Event bus should handle high concurrency without losing events."""
        # Arrange: Create many agents
        await event_bus.subscribe("agent.execution.*", subscriber.on_event)
        num_agents = 10

        agents = [
            ObservableAgent(
                id=f"agent_{i}",
                event_bus=event_bus,
                delay=0.01,
            )
            for i in range(num_agents)
        ]

        # Act: Execute all concurrently
        await asyncio.gather(
            *[
                agent.execute_task(f"Task {i}", {"correlation_id": f"high-{i}"})
                for i, agent in enumerate(agents)
            ]
        )

        # Assert: All events received (2 per agent)
        expected_events = num_agents * 2
        assert len(subscriber.events) == expected_events

    @pytest.mark.asyncio
    async def test_multiple_subscribers_receive_all_events(self, event_bus):
        """Multiple subscribers should all receive the same events."""
        # Arrange: Multiple subscribers (simulating multiple VSCode clients)
        subscribers = [MockEventSubscriber(f"client_{i}") for i in range(3)]

        for sub in subscribers:
            await event_bus.subscribe("agent.execution.*", sub.on_event)

        agent = ObservableAgent(id="multi_sub", event_bus=event_bus, delay=0.01)

        # Act
        await agent.execute_task("Task", {"correlation_id": "multi-sub-test"})

        # Assert: All subscribers got all events
        for sub in subscribers:
            assert len(sub.events) == 2


# =============================================================================
# Tests: Team Execution Events
# =============================================================================


@pytest.mark.integration
class TestTeamExecutionEvents:
    """Test observability events during team execution."""

    @pytest.fixture
    def event_bus(self) -> TestEventBus:
        return TestEventBus()

    @pytest.mark.asyncio
    async def test_team_emits_formation_events(self, event_bus):
        """Team coordinator should emit formation-level events."""
        # Arrange
        subscriber = MockEventSubscriber("team_events")
        await event_bus.subscribe("agent.*", subscriber.on_event)

        # Create observable agents
        agents = [
            ObservableAgent(id=f"member_{i}", event_bus=event_bus, delay=0.01) for i in range(2)
        ]

        # Use lightweight coordinator for testing
        coordinator = create_coordinator(lightweight=True)

        for agent in agents:
            coordinator.add_member(agent)
        coordinator.set_formation(TeamFormation.PARALLEL)

        # Act
        result = await coordinator.execute_task(
            "Team task",
            {"correlation_id": "team-exec-123"},
        )

        # Assert: Team executed successfully
        assert result["success"] is True
        # Note: The lightweight coordinator doesn't emit events,
        # but the observable agents do
        assert len(subscriber.events) >= 4  # At least 2 agents * 2 events


# =============================================================================
# Tests: WebSocket Integration Simulation
# =============================================================================


@pytest.mark.integration
class TestWebSocketIntegration:
    """Test patterns for WebSocket integration (VSCode extension use case)."""

    @dataclass
    class MockWebSocket:
        """Mock WebSocket connection simulating VSCode extension."""

        connection_id: str
        messages: List[str] = field(default_factory=list)
        is_connected: bool = True

        async def send(self, message: str) -> None:
            """Send message to WebSocket client."""
            if self.is_connected:
                self.messages.append(message)

        def get_json_messages(self) -> List[Dict[str, Any]]:
            """Parse messages as JSON."""
            return [json.loads(m) for m in self.messages]

    @pytest.fixture
    def event_bus(self) -> TestEventBus:
        return TestEventBus()

    @pytest.mark.asyncio
    async def test_websocket_receives_agent_events(self, event_bus):
        """WebSocket client should receive agent execution events."""
        # Arrange: Create WebSocket and bridge to event bus
        ws = self.MockWebSocket("vscode-1")

        async def bridge_to_websocket(event_type: str, data: Dict[str, Any]):
            """Bridge event bus events to WebSocket."""
            message = json.dumps(
                {
                    "type": event_type,
                    "data": data,
                }
            )
            await ws.send(message)

        await event_bus.subscribe("agent.*", bridge_to_websocket)

        agent = ObservableAgent(id="ws_agent", event_bus=event_bus, delay=0.01)

        # Act: Execute agent
        await agent.execute_task("Task", {"correlation_id": "ws-test"})

        # Assert: WebSocket received messages
        messages = ws.get_json_messages()
        assert len(messages) == 2
        assert messages[0]["type"] == "agent.execution.started"
        assert messages[1]["type"] == "agent.execution.completed"

    @pytest.mark.asyncio
    async def test_multiple_websockets_concurrent_agents(self, event_bus):
        """Multiple WebSocket clients with concurrent agents."""
        # Arrange: Multiple WebSocket connections
        websockets = [self.MockWebSocket(f"vscode-{i}") for i in range(3)]

        for i, ws in enumerate(websockets):

            async def create_bridge(ws_ref):
                async def bridge(event_type: str, data: Dict[str, Any]):
                    await ws_ref.send(json.dumps({"type": event_type, "data": data}))

                return bridge

            bridge = await create_bridge(ws)
            await event_bus.subscribe("agent.*", bridge)

        agents = [
            ObservableAgent(id=f"concurrent_{i}", event_bus=event_bus, delay=0.02) for i in range(2)
        ]

        # Act: Execute agents concurrently
        await asyncio.gather(
            *[
                agent.execute_task(f"Task {i}", {"correlation_id": f"multi-ws-{i}"})
                for i, agent in enumerate(agents)
            ]
        )

        # Assert: Each WebSocket got all events
        for ws in websockets:
            messages = ws.get_json_messages()
            assert len(messages) == 4  # 2 agents * 2 events each

    @pytest.mark.asyncio
    async def test_websocket_disconnect_handling(self, event_bus):
        """Disconnected WebSocket should not receive events."""
        # Arrange
        ws = self.MockWebSocket("disconnected")

        async def bridge(event_type: str, data: Dict[str, Any]):
            await ws.send(json.dumps({"type": event_type, "data": data}))

        await event_bus.subscribe("agent.*", bridge)

        agent = ObservableAgent(id="disconnect_test", event_bus=event_bus, delay=0.01)

        # Act: Disconnect after first event
        await agent.execute_task("Task1", {"correlation_id": "disc-1"})
        ws.is_connected = False
        await agent.execute_task("Task2", {"correlation_id": "disc-2"})

        # Assert: Only first task's events received
        messages = ws.get_json_messages()
        assert len(messages) == 2  # Only from first task


# =============================================================================
# Tests: Event Aggregation
# =============================================================================


@pytest.mark.unit
class TestEventAggregation:
    """Test event aggregation patterns for UI updates."""

    @pytest.fixture
    def event_bus(self) -> TestEventBus:
        return TestEventBus()

    @pytest.mark.asyncio
    async def test_aggregate_progress_events(self, event_bus):
        """Events can be aggregated for progress tracking."""
        # Arrange: Progress aggregator
        progress = {"started": 0, "completed": 0}

        async def aggregate_progress(event_type: str, data: Dict[str, Any]):
            if "started" in event_type:
                progress["started"] += 1
            elif "completed" in event_type:
                progress["completed"] += 1

        await event_bus.subscribe("agent.execution.*", aggregate_progress)

        agents = [
            ObservableAgent(id=f"prog_{i}", event_bus=event_bus, delay=0.01) for i in range(5)
        ]

        # Act
        await asyncio.gather(
            *[
                agent.execute_task(f"Task {i}", {"correlation_id": f"prog-{i}"})
                for i, agent in enumerate(agents)
            ]
        )

        # Assert
        assert progress["started"] == 5
        assert progress["completed"] == 5

    @pytest.mark.asyncio
    async def test_group_events_by_correlation_id(self, event_bus):
        """Events can be grouped by correlation ID."""
        # Arrange: Event grouper
        grouped_events: Dict[str, List[Dict]] = defaultdict(list)

        async def group_events(event_type: str, data: Dict[str, Any]):
            corr_id = data.get("correlation_id", "unknown")
            grouped_events[corr_id].append(
                {
                    "type": event_type,
                    "data": data,
                }
            )

        await event_bus.subscribe("agent.*", group_events)

        # Create agents for different correlations
        agent1 = ObservableAgent(id="group_1", event_bus=event_bus, delay=0.01)
        agent2 = ObservableAgent(id="group_2", event_bus=event_bus, delay=0.01)

        # Act: Execute with different correlation IDs
        await asyncio.gather(
            agent1.execute_task("Task", {"correlation_id": "session-A"}),
            agent2.execute_task("Task", {"correlation_id": "session-B"}),
        )

        # Assert: Events grouped correctly
        assert len(grouped_events["session-A"]) == 2
        assert len(grouped_events["session-B"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
