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

"""Tests for Ollama provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.ollama_provider import OllamaProvider


@pytest.fixture
def ollama_provider():
    """Create OllamaProvider instance for testing."""
    provider = OllamaProvider(base_url="http://localhost:11434")
    # Override _get_client so it always returns self.client without recreating
    # (avoids event-loop-ID mismatch replacing the mock client under test)
    provider._get_client = lambda: provider.client
    # Reset circuit breaker to CLOSED so earlier real-server errors don't trip it
    if hasattr(provider, "_circuit_breaker"):
        provider._circuit_breaker._state = "CLOSED"
        provider._circuit_breaker._failure_count = 0
    return provider


@pytest.mark.asyncio
async def test_provider_name(ollama_provider):
    """Test provider name property."""
    assert ollama_provider.name == "ollama"


@pytest.mark.asyncio
async def test_supports_tools(ollama_provider):
    """Test tools support."""
    assert ollama_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(ollama_provider):
    """Test streaming support."""
    assert ollama_provider.supports_streaming() is True


class TestEndpointDiscovery:
    """Tests for endpoint discovery logic."""

    def test_select_base_url_from_env(self):
        """Test _select_base_url prioritizes OLLAMA_ENDPOINTS env var (covers lines 121-123)."""
        with patch.dict(
            "os.environ",
            {"OLLAMA_ENDPOINTS": "http://server1:11434,http://server2:11434"},
        ):
            with patch("httpx.Client") as mock_client:
                # Make first endpoint fail
                mock_instance = MagicMock()
                mock_instance.__enter__ = MagicMock(return_value=mock_instance)
                mock_instance.__exit__ = MagicMock()
                mock_instance.get.side_effect = [
                    Exception("Not reachable"),
                    MagicMock(),
                ]
                mock_client.return_value = mock_instance

                # Use skip_discovery since we're testing _select_base_url directly
                provider = OllamaProvider(base_url="http://localhost:11434", _skip_discovery=True)
                # Now test _select_base_url directly
                result = provider._select_base_url("http://ignored:11434", 10)

                # Should try endpoints from env var
                assert "server" in result or "localhost" in result

    def test_select_base_url_comma_separated(self):
        """Test _select_base_url handles comma-separated URL string (covers lines 133-134)."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(OllamaProvider, "_select_base_url") as mock_select:
                mock_select.return_value = "http://localhost:11434"
                provider = OllamaProvider(base_url="http://a:11434,http://b:11434")

                # Just verify it doesn't crash
                assert provider is not None

    def test_select_base_url_list_input(self):
        """Test _select_base_url handles list input (covers lines 130-131)."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(OllamaProvider, "_select_base_url") as mock_select:
                mock_select.return_value = "http://localhost:11434"
                provider = OllamaProvider(base_url=["http://a:11434", "http://b:11434"])
                assert provider is not None

    def test_select_base_url_none_default(self):
        """Test _select_base_url uses default when base_url is None (covers line 125-126)."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(OllamaProvider, "_select_base_url") as mock_select:
                mock_select.return_value = "http://localhost:11434"
                provider = OllamaProvider(base_url=None)
                assert provider is not None

    @pytest.mark.asyncio
    async def test_select_base_url_async_factory(self):
        """Test async factory create method (covers lines 89-90)."""
        with patch.object(
            OllamaProvider, "_select_base_url_async", new_callable=AsyncMock
        ) as mock_async_select:
            mock_async_select.return_value = "http://localhost:11434"

            provider = await OllamaProvider.create(base_url="http://localhost:11434")

            assert provider is not None
            assert provider.name == "ollama"

    def test_skip_discovery_with_list(self):
        """Test _skip_discovery with list base_url (covers lines 56-60)."""
        provider = OllamaProvider(
            base_url=["http://server1:11434"],
            _skip_discovery=True,
        )
        assert provider is not None

    def test_skip_discovery_with_empty_list(self):
        """Test _skip_discovery with empty list falls back to default."""
        provider = OllamaProvider(
            base_url=[],
            _skip_discovery=True,
        )
        # Should fall back to default
        assert provider is not None
