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

"""Tests for Microsoft Teams tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.tools.teams_tool import (
    teams,
    is_teams_configured,
    _get_teams_access_token,
    _get_headers,
    HTTPX_AVAILABLE,
)


class TestTeamsAccessToken:
    """Tests for access token functions."""

    def test_get_access_token_from_context(self):
        """Should get access token from context."""
        context = {"teams_access_token": "test_token_123"}
        result = _get_teams_access_token(context)
        assert result == "test_token_123"

    def test_get_access_token_none_context(self):
        """Should return None if context is None."""
        result = _get_teams_access_token(None)
        assert result is None

    def test_get_access_token_missing_key(self):
        """Should return None if key is missing."""
        context = {"other_key": "value"}
        result = _get_teams_access_token(context)
        assert result is None


class TestIsTeamsConfigured:
    """Tests for is_teams_configured function."""

    def test_configured_with_token(self):
        """Should return True when token is present."""
        context = {"teams_access_token": "test_token"}
        assert is_teams_configured(context) is True

    def test_not_configured_no_context(self):
        """Should return False when no context."""
        assert is_teams_configured(None) is False

    def test_not_configured_no_token(self):
        """Should return False when no token in context."""
        context = {"other_key": "value"}
        assert is_teams_configured(context) is False


class TestGetHeaders:
    """Tests for _get_headers function."""

    def test_headers_with_token(self):
        """Should return headers for an already-resolved bearer token."""
        # _get_headers takes a resolved access-token string (token resolution
        # lives in _resolve_access_token, covered in test_teams_tool_auth.py).
        headers = _get_headers("test_token")
        assert headers["Authorization"] == "Bearer test_token"
        assert headers["Content-Type"] == "application/json"

    def test_headers_without_token(self):
        """An empty token yields an empty bearer (no credential)."""
        headers = _get_headers("")
        assert headers["Authorization"] == "Bearer "


class TestTeamsTool:
    """Tests for teams tool function."""

    @pytest.mark.asyncio
    async def test_no_httpx_error(self):
        """Should return error if httpx not installed."""
        with patch("victor.tools.teams_tool.HTTPX_AVAILABLE", False):
            result = await teams(operation="list_teams")
            assert result["success"] is False
            assert "httpx not installed" in result["error"]

    @pytest.mark.asyncio
    async def test_no_token_error(self):
        """Should return error if no access token."""
        result = await teams(operation="list_teams", context=None)
        assert result["success"] is False
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_unsupported_operation(self):
        """Should return error for unsupported operation."""
        context = {"teams_access_token": "test_token"}
        with patch("victor.tools.teams_tool.httpx"):
            mock_client = AsyncMock()
            with patch("victor.tools.teams_tool.httpx.AsyncClient") as mock_async:
                mock_async.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_async.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await teams(operation="invalid_op", context=context)
                assert result["success"] is False
                assert "Unsupported operation" in result["error"]

    @pytest.mark.asyncio
    async def test_send_message_missing_params(self):
        """Should return error if send_message missing params."""
        context = {"teams_access_token": "test_token"}
        with patch("victor.tools.teams_tool.httpx"):
            mock_client = AsyncMock()
            with patch("victor.tools.teams_tool.httpx.AsyncClient") as mock_async:
                mock_async.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_async.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await teams(operation="send_message", context=context)
                assert result["success"] is False
                assert "Missing required parameters" in result["error"]

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Should send message successfully."""
        context = {"teams_access_token": "test_token"}
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "msg-123",
            "createdDateTime": "2025-01-01T00:00:00Z",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("victor.tools.teams_tool.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await teams(
                operation="send_message",
                team_id="team-123",
                channel_id="channel-456",
                text="Hello Teams!",
                context=context,
            )

            assert result["success"] is True
            assert result["message_id"] == "msg-123"

    @pytest.mark.asyncio
    async def test_list_teams_success(self):
        """Should list teams successfully."""
        context = {"teams_access_token": "test_token"}
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "value": [
                {"id": "team-1", "displayName": "Team 1", "description": "Desc 1"},
                {"id": "team-2", "displayName": "Team 2", "description": "Desc 2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("victor.tools.teams_tool.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await teams(operation="list_teams", context=context)

            assert result["success"] is True
            assert result["count"] == 2
            assert result["results"][0]["name"] == "Team 1"

    @pytest.mark.asyncio
    async def test_list_channels_missing_team_id(self):
        """Should return error if list_channels missing team_id."""
        context = {"teams_access_token": "test_token"}
        with patch("victor.tools.teams_tool.httpx"):
            mock_client = AsyncMock()
            with patch("victor.tools.teams_tool.httpx.AsyncClient") as mock_async:
                mock_async.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_async.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await teams(operation="list_channels", context=context)
                assert result["success"] is False
                assert "Missing required parameter: team_id" in result["error"]

    @pytest.mark.asyncio
    async def test_list_channels_success(self):
        """Should list channels successfully."""
        context = {"teams_access_token": "test_token"}
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "value": [
                {"id": "ch-1", "displayName": "General", "membershipType": "standard"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("victor.tools.teams_tool.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await teams(
                operation="list_channels",
                team_id="team-123",
                context=context,
            )

            assert result["success"] is True
            assert result["count"] == 1
            assert result["results"][0]["name"] == "General"

    @pytest.mark.asyncio
    async def test_search_messages_missing_query(self):
        """Should return error if search_messages missing query."""
        context = {"teams_access_token": "test_token"}
        with patch("victor.tools.teams_tool.httpx"):
            mock_client = AsyncMock()
            with patch("victor.tools.teams_tool.httpx.AsyncClient") as mock_async:
                mock_async.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_async.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await teams(operation="search_messages", context=context)
                assert result["success"] is False
                assert "Missing required parameter: query" in result["error"]

    @pytest.mark.asyncio
    async def test_search_messages_success(self):
        """Should search messages successfully."""
        context = {"teams_access_token": "test_token"}
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "value": [
                {
                    "hitsContainers": [
                        {
                            "hits": [
                                {
                                    "summary": "Found matching message",
                                    "resource": {
                                        "id": "msg-1",
                                        "createdDateTime": "2025-01-01T00:00:00Z",
                                        "from": {"user": {"displayName": "John Doe"}},
                                    },
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("victor.tools.teams_tool.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await teams(
                operation="search_messages",
                query="test search",
                context=context,
            )

            assert result["success"] is True
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_create_channel_missing_params(self):
        """Should return error if create_channel missing params."""
        context = {"teams_access_token": "test_token"}
        with patch("victor.tools.teams_tool.httpx"):
            mock_client = AsyncMock()
            with patch("victor.tools.teams_tool.httpx.AsyncClient") as mock_async:
                mock_async.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_async.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await teams(operation="create_channel", context=context)
                assert result["success"] is False
                assert "Missing required parameters" in result["error"]

    @pytest.mark.asyncio
    async def test_create_channel_success(self):
        """Should create channel successfully."""
        context = {"teams_access_token": "test_token"}
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "ch-new",
            "displayName": "New Channel",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("victor.tools.teams_tool.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await teams(
                operation="create_channel",
                team_id="team-123",
                channel_name="New Channel",
                channel_description="A new channel",
                context=context,
            )

            assert result["success"] is True
            assert result["channel_id"] == "ch-new"

    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """Should handle HTTP errors gracefully."""
        import httpx

        context = {"teams_access_token": "test_token"}

        with patch("victor.tools.teams_tool.httpx.AsyncClient") as mock_async:
            # Create a mock HTTPStatusError
            mock_response = MagicMock()
            mock_response.status_code = 401

            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )
            mock_async.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_async.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await teams(operation="list_teams", context=context)

            assert result["success"] is False
            assert "HTTP error" in result["error"]

    @pytest.mark.asyncio
    async def test_generic_exception_handling(self):
        """Should handle generic exceptions gracefully."""
        context = {"teams_access_token": "test_token"}

        with patch("victor.tools.teams_tool.httpx.AsyncClient") as mock_async:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Network error")
            mock_async.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_async.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await teams(operation="list_teams", context=context)

            assert result["success"] is False
            assert "Network error" in result["error"]
