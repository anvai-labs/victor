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

"""Integration tests for Observability API routes."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from victor.integrations.api.routes.observability_routes import router


@pytest.fixture
def app():
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_session_info():
    """Create mock session info."""
    from victor.observability.query_service import SessionInfo

    return SessionInfo(
        id="test-session-123",
        created_at=datetime.now() - timedelta(hours=1),
        updated_at=datetime.now(),
        message_count=10,
        provider="anthropic",
        model="claude-3-5-sonnet",
        title="Test Session",
        tags=["test"],
    )


@pytest.fixture
def mock_events():
    """Create mock events."""
    from victor.observability.query_service import Event

    events = []
    base_time = datetime.now() - timedelta(minutes=30)

    for i in range(10):
        event = Event(
            id=f"event-{i}",
            event_type="tool_call" if i % 2 == 0 else "message",
            timestamp=base_time + timedelta(minutes=i * 3),
            session_id="test-session-123",
            data={"index": i, "value": f"test_{i}"},
            tool_name=f"tool_{i}" if i % 2 == 0 else None,
            severity="info" if i != 5 else "error",
        )
        events.append(event)

    return events


class TestSessionDetailsEndpoint:
    """Tests for GET /obs/sessions/{session_id} endpoint."""

    def test_get_session_details_success(self, client, mock_session_info, mock_events):
        """Test successful session details retrieval."""
        with patch(
            "victor.integrations.api.routes.observability_routes.get_query_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_session = AsyncMock(return_value=mock_session_info)
            mock_service.get_recent_events = AsyncMock(return_value=mock_events)
            mock_get_service.return_value = mock_service

            response = client.get("/obs/sessions/test-session-123")

            assert response.status_code == 200
            data = response.json()

            assert "session" in data
            assert data["session"]["id"] == "test-session-123"
            assert "metrics" in data
            assert data["metrics"]["tool_calls"] == 5
            assert data["metrics"]["errors"] == 1
            assert "events" in data
            assert len(data["events"]) == 10

    def test_get_session_details_not_found(self, client):
        """Test session details with non-existent session."""
        with patch(
            "victor.integrations.api.routes.observability_routes.get_query_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_session = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.get("/obs/sessions/non-existent-session")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


class TestTracesListEndpoint:
    """Tests for GET /obs/traces endpoint."""

    def test_list_traces_success(self, client, mock_events):
        """Test successful traces listing."""
        with patch(
            "victor.integrations.api.routes.observability_routes.get_query_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_recent_events = AsyncMock(return_value=mock_events)
            mock_get_service.return_value = mock_service

            response = client.get("/obs/traces")

            assert response.status_code == 200
            data = response.json()

            assert "traces" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert isinstance(data["traces"], list)

    def test_list_traces_pagination(self, client, mock_events):
        """Test traces listing with pagination."""
        with patch(
            "victor.integrations.api.routes.observability_routes.get_query_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_recent_events = AsyncMock(return_value=mock_events)
            mock_get_service.return_value = mock_service

            response = client.get("/obs/traces?limit=5&offset=2")

            assert response.status_code == 200
            data = response.json()

            assert data["limit"] == 5
            assert data["offset"] == 2


class TestTraceDetailsEndpoint:
    """Tests for GET /obs/traces/{trace_id} endpoint."""

    def test_get_trace_details_success(self, client, mock_events):
        """Test successful trace details retrieval."""
        with patch(
            "victor.integrations.api.routes.observability_routes.get_query_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_recent_events = AsyncMock(return_value=mock_events)
            mock_get_service.return_value = mock_service

            response = client.get("/obs/traces/test-session-123")

            assert response.status_code == 200
            data = response.json()

            assert "trace_id" in data
            assert data["trace_id"] == "test-session-123"
            assert "spans" in data
            assert len(data["spans"]) == 10
            assert "span_count" in data
            assert data["span_count"] == 10
            assert "tool_calls" in data
            assert data["tool_calls"] == 5
            assert "errors" in data
            assert data["errors"] == 1

    def test_get_trace_details_not_found(self, client):
        """Test trace details with non-existent trace."""
        with patch(
            "victor.integrations.api.routes.observability_routes.get_query_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_recent_events = AsyncMock(return_value=[])
            mock_get_service.return_value = mock_service

            response = client.get("/obs/traces/non-existent-trace")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_get_trace_details_span_structure(self, client, mock_events):
        """Test that span structure is correct."""
        with patch(
            "victor.integrations.api.routes.observability_routes.get_query_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_recent_events = AsyncMock(return_value=mock_events)
            mock_get_service.return_value = mock_service

            response = client.get("/obs/traces/test-session-123")

            assert response.status_code == 200
            data = response.json()

            # Check first span structure
            first_span = data["spans"][0]
            assert "span_id" in first_span
            assert "operation" in first_span
            assert "start_time" in first_span
            assert "status" in first_span
            assert "tags" in first_span
            assert "data" in first_span
