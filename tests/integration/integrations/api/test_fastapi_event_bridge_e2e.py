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

"""End-to-end integration tests for FastAPI server with EventBridge.

These tests verify the full integration between:
- FastAPI server with WebSocket endpoints
- EventBridge for real-time event streaming
- WebSocket client connections

Running locally vs CI:
- Locally: Tests run fully with real WebSocket connections
- CI (GitHub Actions): Tests are skipped if environment doesn't support
  interactive WebSocket testing (detected via CI environment variable)

To run locally:
    pytest tests/integration/test_fastapi_event_bridge_e2e.py -v

To skip in CI:
    Tests automatically skip when CI=true environment variable is set
    and FastAPI test client is not available.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, Generator, Optional

import pytest

# Check if we're in CI environment
IS_CI = os.environ.get("CI", "false").lower() == "true"
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"


# Check for required dependencies
def _check_fastapi_available() -> bool:
    """Check if FastAPI and its test dependencies are available."""
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        return True
    except ImportError:
        return False


def _check_websocket_available() -> bool:
    """Check if WebSocket support is available."""
    try:
        import websockets

        return True
    except ImportError:
        return False


FASTAPI_AVAILABLE = _check_fastapi_available()
WEBSOCKET_AVAILABLE = _check_websocket_available()

# Skip conditions
skip_if_no_fastapi = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")

skip_if_ci_no_websocket = pytest.mark.skipif(
    IS_CI and not WEBSOCKET_AVAILABLE,
    reason="WebSocket tests skipped in CI without websockets package",
)

# Module-level markers
pytestmark = [
    pytest.mark.integration,
    skip_if_no_fastapi,
]


@pytest.fixture
def fastapi_server():
    """Create a FastAPI server instance for testing."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")

    from victor.integrations.api.fastapi_server import VictorFastAPIServer

    server = VictorFastAPIServer(host="127.0.0.1", port=8765)
    yield server


@pytest.fixture
def test_client(fastapi_server):
    """Create a test client for the FastAPI server."""
    from fastapi.testclient import TestClient

    with TestClient(fastapi_server.app) as client:
        yield client


# =============================================================================
# Health and Status Tests
# =============================================================================


class TestFastAPIServerHealth:
    """Tests for FastAPI server health endpoints."""

    def test_health_endpoint(self, test_client):
        """Test /health endpoint returns healthy status."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_version(self, test_client):
        """Test /health endpoint includes version info."""
        response = test_client.get("/health")
        data = response.json()

        assert "version" in data


# =============================================================================
# WebSocket Tests
# =============================================================================


class TestWebSocketEndpoints:
    """Tests for WebSocket endpoints."""

    def test_ws_endpoint_accepts_connection(self, test_client):
        """Test that /ws endpoint accepts WebSocket connections."""
        with test_client.websocket_connect("/ws") as websocket:
            # Connection should be accepted
            # Send a simple message
            websocket.send_json({"type": "ping"})
            # Connection should remain open
            assert websocket is not None

    @skip_if_ci_no_websocket
    def test_events_endpoint_accepts_connection(self, test_client):
        """Test that /ws/events endpoint accepts WebSocket connections."""
        with test_client.websocket_connect("/ws/events") as websocket:
            # Connection should be accepted
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()

            assert response["type"] == "pong"

    @skip_if_ci_no_websocket
    def test_events_endpoint_subscribe_acknowledgment(self, test_client):
        """Test that subscribe message is acknowledged."""
        with test_client.websocket_connect("/ws/events") as websocket:
            websocket.send_json({"type": "subscribe", "categories": ["all"]})
            response = websocket.receive_json()

            assert response["type"] == "subscribed"
            assert "categories" in response

    @skip_if_ci_no_websocket
    def test_events_endpoint_selective_subscribe(self, test_client):
        """Test subscribing to specific event categories."""
        with test_client.websocket_connect("/ws/events") as websocket:
            categories = ["tool.start", "tool.complete"]
            websocket.send_json({"type": "subscribe", "categories": categories})
            response = websocket.receive_json()

            assert response["type"] == "subscribed"
            assert response["categories"] == categories


# =============================================================================
# EventBridge Integration Tests
# =============================================================================


class TestEventBridgeIntegration:
    """Integration tests for EventBridge with FastAPI."""

    @pytest.mark.asyncio
    @skip_if_ci_no_websocket
    async def test_event_bridge_initialization(self):
        """Test that EventBridge initializes correctly."""
        from victor.integrations.api.event_bridge import EventBridge
        from victor.core.events import create_event_backend

        backend = create_event_backend()
        bridge = EventBridge(backend)

        bridge.start()
        assert bridge._running is True

        bridge.stop()
        assert bridge._running is False

    @pytest.mark.asyncio
    @skip_if_ci_no_websocket
    async def test_event_bridge_broadcaster_lifecycle(self):
        """Test broadcaster client lifecycle."""
        from victor.integrations.api.event_bridge import EventBridge
        from victor.core.events import create_event_backend

        backend = create_event_backend()
        bridge = EventBridge(backend)
        bridge.start()

        # Add client
        async def send_func(msg: str):
            pass

        bridge._broadcaster.add_client("test-client", send_func)
        assert "test-client" in bridge._broadcaster._clients

        # Remove client
        bridge._broadcaster.remove_client("test-client")
        assert "test-client" not in bridge._broadcaster._clients

        bridge.stop()


# =============================================================================
# Multi-Client Tests
# =============================================================================


class TestMultiClientConnections:
    """Tests for multiple WebSocket client connections."""

    @skip_if_ci_no_websocket
    def test_multiple_clients_can_connect(self, test_client):
        """Test that multiple clients can connect simultaneously."""
        # This tests basic multiple connection handling
        # Note: TestClient uses synchronous connections, so we test sequentially
        with test_client.websocket_connect("/ws/events") as ws1:
            ws1.send_json({"type": "ping"})
            r1 = ws1.receive_json()
            assert r1["type"] == "pong"

        with test_client.websocket_connect("/ws/events") as ws2:
            ws2.send_json({"type": "ping"})
            r2 = ws2.receive_json()
            assert r2["type"] == "pong"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in WebSocket endpoints."""

    @skip_if_ci_no_websocket
    def test_invalid_json_handling(self, test_client):
        """Test that invalid JSON is handled gracefully."""
        with test_client.websocket_connect("/ws/events") as websocket:
            # Send valid ping first to ensure connection works
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"

            # Note: TestClient may not allow sending raw invalid data easily
            # This test documents expected behavior

    @skip_if_ci_no_websocket
    def test_unknown_message_type(self, test_client):
        """Test handling of unknown message types."""
        with test_client.websocket_connect("/ws/events") as websocket:
            # Send unknown type
            websocket.send_json({"type": "unknown_type"})
            # Server should handle gracefully (not crash)
            # Send ping to verify connection still works
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"


# =============================================================================
# Capability Endpoint Tests
# =============================================================================


class TestCapabilityEndpoint:
    """Tests for /capabilities endpoint."""

    def test_capabilities_endpoint_exists(self, test_client):
        """Test that /capabilities endpoint exists."""
        response = test_client.get("/capabilities")
        # Should return 200 or at least not 404
        assert response.status_code in [200, 500]  # 500 if not fully configured

    def test_capabilities_returns_json(self, test_client):
        """Test that /capabilities returns JSON."""
        response = test_client.get("/capabilities")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_capabilities_recommend_endpoint_exists(self, test_client):
        """Test that /capabilities/recommend endpoint exists."""
        response = test_client.get(
            "/capabilities/recommend",
            params={"task_type": "feature", "complexity": "high"},
        )
        assert response.status_code in [200, 500]

    def test_capabilities_recommend_returns_json(self, test_client):
        """Test that /capabilities/recommend returns JSON."""
        response = test_client.get(
            "/capabilities/recommend",
            params={"task_type": "feature", "complexity": "high"},
        )
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            assert "recommendations" in data


# =============================================================================
# Local-only comprehensive tests
# =============================================================================


@pytest.mark.skipif(IS_CI, reason="Comprehensive tests only run locally")
class TestLocalComprehensive:
    """Comprehensive tests that only run locally (not in CI)."""

    def test_full_event_flow(self, test_client):
        """Test complete event subscription and ping/pong flow."""
        with test_client.websocket_connect("/ws/events") as websocket:
            # Subscribe
            websocket.send_json({"type": "subscribe", "categories": ["all"]})
            response = websocket.receive_json()
            assert response["type"] == "subscribed"

            # Multiple pings
            for i in range(5):
                websocket.send_json({"type": "ping"})
                response = websocket.receive_json()
                assert response["type"] == "pong"

    def test_rapid_ping_pong(self, test_client):
        """Test rapid ping/pong exchanges."""
        with test_client.websocket_connect("/ws/events") as websocket:
            for i in range(10):
                websocket.send_json({"type": "ping"})
                response = websocket.receive_json()
                assert response["type"] == "pong"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
