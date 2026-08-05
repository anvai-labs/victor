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

"""Comprehensive tests for Anthropic provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.base import (
    ToolDefinition,
)
from victor.workflows.services.credentials import SSOTokens


@pytest.fixture
def anthropic_provider():
    """Create AnthropicProvider instance for testing."""
    return AnthropicProvider(
        api_key="test-api-key",
        base_url="https://api.anthropic.com",
        timeout=30,
        max_retries=0,
        use_circuit_breaker=False,
    )


@pytest.mark.asyncio
async def test_initialization():
    """Test provider initialization."""
    provider = AnthropicProvider(
        api_key="test-key",
        base_url="https://custom.url",
        timeout=45,
        max_retries=5,
    )

    assert provider.api_key == "test-key"
    assert provider.base_url == "https://custom.url"
    assert provider.timeout == 45
    assert provider.max_retries == 5
    # Transport is owned by the Sandhi typed variant; the policy shell must NOT
    # own a provider generation client (TD-0002 deletion gate).
    assert not hasattr(provider, "client")


@pytest.mark.asyncio
async def test_oauth_mode_uses_claude_code_token_source():
    """Anthropic OAuth mode should consume Claude Code tokens via OAuthTokenManager."""
    tokens = SSOTokens(
        access_token="claude_oauth_token",
        refresh_token="ref",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    with patch("victor.providers.anthropic_provider.OAuthTokenManager") as MockMgr:
        mock_instance = MagicMock()
        mock_instance.get_valid_token = AsyncMock(return_value="claude_oauth_token")
        mock_instance._load_cached = MagicMock(return_value=tokens)
        MockMgr.return_value = mock_instance

        provider = AnthropicProvider(auth_mode="oauth", oauth_source="claude-code")

    MockMgr.assert_called_once_with("anthropic", token_source="claude-code")
    # OAuth acquisition stays in Victor; the cached token lands on _api_key,
    # which the Sandhi typed handle consumes with bearer auth (no SDK client).
    assert provider._api_key == "claude_oauth_token"
    assert provider._sandhi_auth_scheme == "bearer"


@pytest.mark.asyncio
async def test_provider_name(anthropic_provider):
    """Test provider name property."""
    assert anthropic_provider.name == "anthropic"


@pytest.mark.asyncio
async def test_supports_tools(anthropic_provider):
    """Test tools support."""
    assert anthropic_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(anthropic_provider):
    """Test streaming support."""
    assert anthropic_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_convert_tools(anthropic_provider):
    """Test tool conversion to Anthropic format."""
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        ),
    ]

    converted = anthropic_provider._convert_tools(tools)

    assert len(converted) == 2
    assert converted[0]["name"] == "get_weather"
    assert converted[0]["description"] == "Get weather information"
    assert converted[0]["input_schema"]["type"] == "object"
    assert "location" in converted[0]["input_schema"]["properties"]
    assert converted[1]["name"] == "search"


@pytest.mark.asyncio
async def test_close_is_safe_without_generation_client(anthropic_provider):
    """Transport is owned by Sandhi; the policy shell owns no client to close."""
    # The policy shell must not own a provider generation client (TD-0002 gate),
    # so close() is a safe no-op rather than a client.teardown.
    assert not hasattr(anthropic_provider, "client")
    await anthropic_provider.close()  # must not raise


@pytest.mark.asyncio
async def test_list_models_uses_sandhi_catalog(monkeypatch):
    """When the Sandhi catalog is available, list_models() uses it (Victor shapes the facts)."""
    provider = AnthropicProvider(api_key="test-key")
    monkeypatch.setattr(
        provider,
        "_models_from_sandhi",
        lambda: [
            {
                "id": "claude-fable-5",
                "name": "Claude Fable 5",
                "context_window": 1_000_000,
                "max_output_tokens": 131_072,
            }
        ],
    )
    # The SDK must NOT be consulted when the catalog is present.
    with patch(
        "anthropic.AsyncAnthropic", side_effect=AssertionError("SDK unused when catalog present")
    ):
        models = await provider.list_models()

    assert models == [
        {
            "id": "claude-fable-5",
            "name": "Claude Fable 5",
            "context_window": 1_000_000,
            "max_output_tokens": 131_072,
        }
    ]


@pytest.mark.asyncio
async def test_list_models_uses_live_sdk_when_catalog_absent(monkeypatch):
    """When the Sandhi catalog is unavailable, list_models() falls back to the SDK /v1/models.

    The SDK client is discovery-only (transient); transport stays Sandhi-owned.
    """
    provider = AnthropicProvider(api_key="test-key")
    monkeypatch.setattr(provider, "_models_from_sandhi", lambda: None)
    fake_model = SimpleNamespace(
        id="claude-future-9",
        display_name="Claude Future 9",
        type="model",
        max_input_tokens=2_000_000,
        max_tokens=200_000,
    )
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.models.list = AsyncMock(return_value=SimpleNamespace(data=[fake_model]))

    with patch("anthropic.AsyncAnthropic", return_value=fake_client) as mock_ctor:
        models = await provider.list_models()

    mock_ctor.assert_called_once()
    assert models == [
        {
            "id": "claude-future-9",
            "name": "Claude Future 9",
            "type": "model",
            "context_window": 2_000_000,
            "max_output_tokens": 200_000,
        }
    ]


@pytest.mark.asyncio
async def test_list_models_falls_back_to_current_static_list(monkeypatch):
    """When catalog + SDK both fail, list_models() returns the curated static fallback."""
    provider = AnthropicProvider(api_key="test-key")
    monkeypatch.setattr(provider, "_models_from_sandhi", lambda: None)
    with patch("anthropic.AsyncAnthropic", side_effect=RuntimeError("offline")):
        models = await provider.list_models()

    ids = [m["id"] for m in models]
    # Current Claude lineup (web-sourced 2026-07): catalog + live discovery down -> static.
    assert "claude-fable-5" in ids
    assert "claude-opus-4-8" in ids
    assert "claude-sonnet-5" in ids
    assert "claude-haiku-4-5-20251001" in ids
    # Retired/ancient models must not be advertised in the fallback.
    assert "claude-sonnet-4-20250514" not in ids
    assert "claude-3-5-sonnet-20241022" not in ids


@pytest.mark.asyncio
async def test_models_from_sandhi_reads_real_catalog_when_available():
    """Integration: _models_from_sandhi reads the real Sandhi catalog when the binding exposes it.

    Skips cleanly when the installed Sandhi predates the catalog surface (TD-0004 Phase A),
    so it is not CI-flaky.
    """
    try:
        import sandhi_gateway as sg
    except Exception:  # pragma: no cover - sandhi absent
        pytest.skip("sandhi-gateway not installed")
    if not hasattr(sg, "provider_models_json"):
        pytest.skip("installed sandhi predates the catalog surface (TD-0004 Phase A)")

    provider = AnthropicProvider(api_key="test-key")
    models = provider._models_from_sandhi()
    assert models is not None
    ids = [m["id"] for m in models]
    # The curated Anthropic lineup admitted in sandhi-providers catalog.rs.
    assert "claude-fable-5" in ids
    assert "claude-opus-4-8" in ids
