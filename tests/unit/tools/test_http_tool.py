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

"""Tests for http_tool module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from victor.tools.http_tool import http


class TestHttpRequest:
    """Tests for http_request function."""

    @pytest.mark.asyncio
    async def test_http_request_success(self):
        """Test successful HTTP request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason_phrase = "OK"
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://api.example.com/data"
        mock_response.json.return_value = {"result": "success"}

        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await http(method="GET", url="https://api.example.com/data")

            assert result["success"] is True
            assert result["status_code"] == 200
            assert result["body"] == {"result": "success"}
            assert "duration_ms" in result

    @pytest.mark.asyncio
    async def test_http_request_missing_url(self):
        """Test http_request with missing URL."""
        result = await http(method="GET", url="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_http_request_with_headers(self):
        """Test HTTP request with custom headers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason_phrase = "OK"
        mock_response.headers = {}
        mock_response.url = "https://api.example.com"
        mock_response.json.side_effect = ValueError("Not JSON")  # ValueError, not Exception
        mock_response.text = "Plain text response"

        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            headers = {"X-Custom-Header": "value"}
            result = await http(method="GET", url="https://api.example.com", headers=headers)

            assert result["success"] is True
            assert result["body"] == "Plain text response"

    @pytest.mark.asyncio
    async def test_http_request_with_bearer_auth(self):
        """Test HTTP request with Bearer authentication."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason_phrase = "OK"
        mock_response.headers = {}
        mock_response.url = "https://api.example.com"
        mock_response.json.return_value = {"authenticated": True}

        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await http(method="GET", url="https://api.example.com", auth="Bearer token123")

            assert result["success"] is True
            mock_instance.request.assert_called_once()
            call_kwargs = mock_instance.request.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_http_request_timeout(self):
        """Test HTTP request timeout handling."""
        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            # __aexit__ should return False to propagate exceptions
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.request = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.return_value = mock_instance

            result = await http(method="GET", url="https://api.example.com")

            assert result["success"] is False
            assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_http_request_generic_exception(self):
        """Test HTTP request generic exception handling."""
        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.request = AsyncMock(side_effect=RuntimeError("Network error"))
            mock_client.return_value = mock_instance

            result = await http(method="GET", url="https://api.example.com")

            assert result["success"] is False
            assert "Request failed" in result["error"]
            assert "Network error" in result["error"]

    @pytest.mark.asyncio
    async def test_http_request_post_with_json(self):
        """Test POST request with JSON body."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.reason_phrase = "Created"
        mock_response.headers = {}
        mock_response.url = "https://api.example.com/items"
        mock_response.json.return_value = {"id": 123}

        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            json_data = {"name": "test item"}
            result = await http(method="POST", url="https://api.example.com/items", json=json_data)

            assert result["success"] is True
            assert result["status_code"] == 201


class TestHttpTest:
    """Tests for http_test function."""

    @pytest.mark.asyncio
    async def test_http_test_success(self):
        """Test successful API test with validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://api.example.com"

        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await http(
                method="GET",
                url="https://api.example.com",
                mode="test",
                expected_status=200,
            )

            assert result["success"] is True
            assert result["all_passed"] is True
            assert len(result["validations"]) == 1
            assert result["validations"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_http_test_missing_url(self):
        """Test http_test with missing URL."""
        result = await http(method="GET", url="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_http_test_validation_failure(self):
        """Test API test with validation failure."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "https://api.example.com"

        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await http(
                method="GET",
                url="https://api.example.com",
                mode="test",
                expected_status=200,
            )

            assert result["success"] is False
            assert result["all_passed"] is False
            assert result["validations"][0]["passed"] is False
            assert result["validations"][0]["expected"] == 200
            assert result["validations"][0]["actual"] == 404

    @pytest.mark.asyncio
    async def test_http_test_with_auth(self):
        """Test API test with authentication."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://api.example.com"

        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await http(method="GET", url="https://api.example.com", auth="Bearer token123")

            assert result["success"] is True
            mock_instance.request.assert_called_once()
            call_kwargs = mock_instance.request.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_http_test_timeout(self):
        """Test API test timeout handling."""
        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.request = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.return_value = mock_instance

            result = await http(method="GET", url="https://api.example.com")

            assert result["success"] is False
            assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_http_test_generic_exception(self):
        """Test API test generic exception handling."""
        with patch("victor.tools.http_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.request = AsyncMock(side_effect=RuntimeError("Connection error"))
            mock_client.return_value = mock_instance

            result = await http(method="GET", url="https://api.example.com")

            assert result["success"] is False
            assert "Request failed" in result["error"]
