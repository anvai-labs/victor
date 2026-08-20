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

"""Tests for LMStudio provider.

Tests the dedicated LMStudioProvider which uses:
- httpx.AsyncClient (not AsyncOpenAI SDK)
- Tiered URL selection with /v1/models health check
- 300s timeout (matching Ollama)
- OpenAI-compatible API format
"""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.lmstudio_provider import LMStudioProvider


@pytest.fixture
def lmstudio_provider():
    """Create LMStudioProvider instance for testing."""
    return LMStudioProvider(
        base_url="http://127.0.0.1:1234",
        _skip_discovery=True,
        max_retries=0,
        use_circuit_breaker=False,
    )


@pytest.mark.asyncio
async def test_provider_name(lmstudio_provider):
    """Test provider name property."""
    assert lmstudio_provider.name == "lmstudio"


@pytest.mark.asyncio
async def test_supports_tools(lmstudio_provider):
    """Test tools support."""
    assert lmstudio_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(lmstudio_provider):
    """Test streaming support."""
    assert lmstudio_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_default_timeout():
    """Test that default timeout is 300s (matching Ollama)."""
    assert LMStudioProvider.DEFAULT_TIMEOUT == 300


@pytest.mark.asyncio
async def test_default_port():
    """Test that default port is 1234."""
    assert LMStudioProvider.DEFAULT_PORT == 1234


@pytest.mark.asyncio
async def test_list_models(lmstudio_provider):
    """Test model listing."""
    mock_response = {
        "object": "list",
        "data": [
            {"id": "qwen3-coder-30b", "object": "model"},
            {"id": "llama-3.1-8b", "object": "model"},
        ],
    }

    with patch.object(
        lmstudio_provider.client,
        "get",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_response_obj = AsyncMock()
        mock_response_obj.json = lambda: mock_response
        mock_response_obj.raise_for_status = lambda: None
        mock_get.return_value = mock_response_obj

        models = await lmstudio_provider.list_models()

        assert len(models) == 2
        assert models[0]["id"] == "qwen3-coder-30b"


def test_tiered_url_selection_single():
    """Test URL selection with single URL."""
    with patch("httpx.Client") as mock_client:
        # Mock successful health check
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "model"}]}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        provider = LMStudioProvider(base_url="http://192.168.1.20:1234")
        assert "192.168.1.20" in provider.base_url


def test_tiered_url_selection_list():
    """Test URL selection with list of URLs (tiered fallback)."""
    with patch("httpx.Client") as mock_client:
        call_count = 0

        def mock_get(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First URL fails
                raise httpx.ConnectError("Connection refused")
            # Second URL succeeds
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"id": "model"}]}
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.side_effect = mock_get
        mock_client.return_value = mock_client_instance

        provider = LMStudioProvider(base_url=["http://192.168.1.20:1234", "http://127.0.0.1:1234"])
        # Should fall back to second URL
        assert "127.0.0.1" in provider.base_url


def test_env_var_url_override():
    """Test LMSTUDIO_ENDPOINTS environment variable override."""
    import os

    with patch.dict(os.environ, {"LMSTUDIO_ENDPOINTS": "http://custom:1234"}):
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"id": "model"}]}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            provider = LMStudioProvider(base_url="http://localhost:1234")
            # Should use env var URL, not the provided base_url
            assert "custom" in provider.base_url


@pytest.mark.asyncio
async def test_close(lmstudio_provider):
    """Test client cleanup."""
    with patch.object(
        lmstudio_provider.client,
        "aclose",
        new_callable=AsyncMock,
    ) as mock_close:
        await lmstudio_provider.close()
        mock_close.assert_called_once()


# Registry integration tests
def test_registry_returns_lmstudio_provider():
    """Test that ProviderRegistry returns LMStudioProvider for 'lmstudio'."""
    from victor.providers.registry import ProviderRegistry

    provider_class = ProviderRegistry.get("lmstudio")
    assert provider_class == LMStudioProvider


def test_registry_create_lmstudio():
    """Test creating LMStudio provider via registry."""
    from victor.providers.registry import ProviderRegistry

    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        provider = ProviderRegistry.create(
            "lmstudio",
            base_url="http://127.0.0.1:1234",
        )
        assert isinstance(provider, LMStudioProvider)
        assert provider.name == "lmstudio"
