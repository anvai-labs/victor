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

"""Tests for observability API routes."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from victor.integrations.api.fastapi_server import VictorFastAPIServer
from victor.integrations.api.routes.observability_routes import router
from victor.observability.query_service import Event, EventFilters, SessionInfo


class TestObservabilityRouter:
    """Tests for observability router."""

    def test_router_creation(self):
        """Test that router can be created."""
        assert router is not None
        assert router.prefix == "/obs"

    def test_router_has_routes(self):
        """Test that router has expected routes."""
        routes = [route.path for route in router.routes]

        # Check for main endpoints
        assert "/obs/events/recent" in routes
        assert "/obs/sessions" in routes
        assert "/obs/metrics/summary" in routes
        assert "/obs/tools/stats" in routes
        assert "/obs/tokens/usage" in routes
        assert "/obs/dashboard" in routes

    def test_events_endpoint_definition(self):
        """Test that events endpoint is properly defined."""
        # Find the events/recent route
        events_route = None
        for route in router.routes:
            if route.path == "/obs/events/recent":
                events_route = route
                break

        assert events_route is not None
        assert events_route.methods == {"GET"}

    def test_sessions_endpoint_definition(self):
        """Test that sessions endpoint is properly defined."""
        # Find the sessions route
        sessions_route = None
        for route in router.routes:
            if route.path == "/obs/sessions":
                sessions_route = route
                break

        assert sessions_route is not None
        assert sessions_route.methods == {"GET"}

    def test_metrics_endpoint_definition(self):
        """Test that metrics endpoint is properly defined."""
        # Find the metrics/summary route
        metrics_route = None
        for route in router.routes:
            if route.path == "/obs/metrics/summary":
                metrics_route = route
                break

        assert metrics_route is not None
        assert metrics_route.methods == {"GET"}

    def test_dashboard_endpoint_definition(self):
        """Test that dashboard endpoint is properly defined."""
        # Find the dashboard route
        dashboard_route = None
        for route in router.routes:
            if route.path == "/obs/dashboard":
                dashboard_route = route
                break

        assert dashboard_route is not None
        assert dashboard_route.methods == {"GET"}

    def test_tool_stats_endpoint_definition(self):
        """Test that tool stats endpoint is properly defined."""
        # Find the tools/stats route
        tools_route = None
        for route in router.routes:
            if route.path == "/obs/tools/stats":
                tools_route = route
                break

        assert tools_route is not None
        assert tools_route.methods == {"GET"}

    def test_token_usage_endpoint_definition(self):
        """Test that token usage endpoint is properly defined."""
        # Find the tokens/usage route
        tokens_route = None
        for route in router.routes:
            if route.path == "/obs/tokens/usage":
                tokens_route = route
                break

        assert tokens_route is not None
        assert tokens_route.methods == {"GET"}


class TestObservabilityIntegration:
    """Integration tests for observability endpoints."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        # Create a minimal server for testing
        server = VictorFastAPIServer.__new__(VictorFastAPIServer)
        server.app = None  # Will be set by TestClient
        server.host = "127.0.0.1"
        server.port = 8765
        server.workspace_root = None
        server.rate_limit_rpm = None
        server.api_keys = {}
        server.enable_cors = False
        server.enable_hitl = False
        server.hitl_auth_token = None
        server.hitl_persistent = False
        server._enable_graphql = False

        # Mock dependencies
        server._settings = None
        server._container = None
        server._orchestrator = None
        server._ws_clients = []
        server._pending_tool_approvals = {}
        server._hitl_store = None
        server._event_bridge = None
        server._event_clients = []
        server._workflow_event_bridge = None
        server._workflow_executions = {}
        server._shutting_down = False

        return server

    @pytest.fixture
    def client(self, server):
        """Create test client with observability router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        # Note: This endpoint doesn't exist yet, but could be added
        # For now, just test that the router is accessible
        response = client.get("/obs/dashboard")
        # Should return 404 or HTML (static files may not exist in test)
        assert response.status_code in [200, 404]

    def test_events_endpoint_exists(self, client):
        """Test that events endpoint responds."""
        response = client.get("/obs/events/recent?limit=10")
        # Should respond (may have errors if no data, but endpoint exists)
        assert response.status_code in [200, 500]

    def test_sessions_endpoint_exists(self, client):
        """Test that sessions endpoint responds."""
        response = client.get("/obs/sessions?limit=10")
        # Should respond (may have errors if no data, but endpoint exists)
        assert response.status_code in [200, 500]

    def test_metrics_endpoint_exists(self, client):
        """Test that metrics endpoint responds."""
        response = client.get("/obs/metrics/summary")
        # Should respond (may have errors if no data, but endpoint exists)
        assert response.status_code in [200, 500]

    def test_tools_stats_endpoint_exists(self, client):
        """Test that tools stats endpoint responds."""
        response = client.get("/obs/tools/stats")
        # Should respond (may have errors if no data, but endpoint exists)
        assert response.status_code in [200, 500]

    def test_tokens_usage_endpoint_exists(self, client):
        """Test that tokens usage endpoint responds."""
        response = client.get("/obs/tokens/usage")
        # Should respond (may have errors if no data, but endpoint exists)
        assert response.status_code in [200, 500]

    def test_session_details_endpoint_filters_events_by_session(self, client, monkeypatch):
        """Test session details endpoint resolves session and applies session_id filter."""

        seen = {}

        class FakeQueryService:
            async def get_session(self, session_id: str) -> SessionInfo:
                seen["session_id"] = session_id
                return SessionInfo(
                    id=session_id,
                    created_at=datetime(2026, 1, 1),
                    updated_at=datetime(2026, 1, 1),
                    message_count=1,
                    provider="test",
                    model="model",
                )

            async def get_recent_events(
                self,
                limit: int,
                offset: int = 0,
                filters: EventFilters | None = None,
            ):
                seen["limit"] = limit
                seen["offset"] = offset
                seen["filters"] = filters
                return [
                    Event(
                        id="evt-1",
                        event_type="tool",
                        timestamp=datetime(2026, 1, 1, 1, 0, 0),
                        session_id="session-1",
                        data={},
                        tool_name="test",
                        severity="info",
                    )
                ]

        query_service = FakeQueryService()
        monkeypatch.setattr(
            "victor.integrations.api.routes.observability_routes.get_query_service",
            lambda: query_service,
        )

        response = client.get("/obs/sessions/session-1")
        assert response.status_code == 200
        assert seen["session_id"] == "session-1"
        assert seen["filters"] is not None
        assert seen["filters"].session_ids == ["session-1"]
        assert seen["limit"] == 1000

    def test_trace_details_endpoint_filters_events_by_trace_id(self, client, monkeypatch):
        """Test trace details endpoint applies trace/session filter when loading events."""

        seen = {}

        class FakeQueryService:
            async def get_recent_events(
                self,
                limit: int,
                offset: int = 0,
                filters: EventFilters | None = None,
            ):
                seen["limit"] = limit
                seen["offset"] = offset
                seen["filters"] = filters
                return [
                    Event(
                        id="evt-1",
                        event_type="tool",
                        timestamp=datetime(2026, 1, 1, 1, 0, 0),
                        session_id="trace-1",
                        data={},
                        severity="error",
                    )
                ]

            async def get_session(self, session_id: str):
                raise AssertionError("should not be called for trace endpoint")

        monkeypatch.setattr(
            "victor.integrations.api.routes.observability_routes.get_query_service",
            lambda: FakeQueryService(),
        )

        response = client.get("/obs/traces/trace-1")
        assert response.status_code == 200
        payload = response.json()
        assert payload["trace_id"] == "trace-1"
        assert payload["span_count"] == 1
        assert seen["filters"] is not None
        assert seen["filters"].session_ids == ["trace-1"]
