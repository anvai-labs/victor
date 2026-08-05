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

"""Comprehensive tests for OpenAI provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


from victor.providers.openai_provider import OpenAIProvider


@pytest.fixture
def openai_provider():
    """Create OpenAIProvider instance for testing."""
    return OpenAIProvider(
        api_key="test-api-key",
        organization="test-org",
        base_url="https://api.openai.com/v1",
        timeout=30,
        max_retries=0,
        use_circuit_breaker=False,
    )


@pytest.mark.asyncio
async def test_initialization():
    """Test provider initialization."""
    provider = OpenAIProvider(
        api_key="test-key",
        organization="test-org",
        base_url="https://custom.url",
        timeout=45,
        max_retries=5,
    )

    assert provider.api_key == "test-key"
    assert provider.base_url == "https://custom.url"
    assert provider.timeout == 45
    assert provider.max_retries == 5
    assert provider.client is not None


@pytest.mark.asyncio
async def test_initialization_without_organization():
    """Test provider initialization without organization."""
    provider = OpenAIProvider(api_key="test-key")

    assert provider.api_key == "test-key"
    assert provider.client is not None


@pytest.mark.asyncio
async def test_provider_name(openai_provider):
    """Test provider name property."""
    assert openai_provider.name == "openai"


@pytest.mark.asyncio
async def test_supports_tools(openai_provider):
    """Test tools support."""
    assert openai_provider.supports_tools() is True


@pytest.mark.asyncio
async def test_supports_streaming(openai_provider):
    """Test streaming support."""
    assert openai_provider.supports_streaming() is True


@pytest.mark.asyncio
async def test_close(openai_provider):
    """Test closing the provider client."""
    with patch.object(
        openai_provider.client,
        "close",
        new_callable=AsyncMock,
    ) as mock_close:
        await openai_provider.close()
        mock_close.assert_called_once()


def _mock_chat_response():
    msg = MagicMock()
    msg.content = "ok"
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    resp.model_dump = lambda: {}
    return resp


@pytest.mark.asyncio
async def test_supports_reasoning_effort(openai_provider):
    assert openai_provider.supports_reasoning_effort("o3-mini") is True
    assert openai_provider.supports_reasoning_effort("gpt-5.1") is True
    assert openai_provider.supports_reasoning_effort("gpt-4o") is False
    assert openai_provider.supports_reasoning_effort(None) is False


@pytest.mark.asyncio
async def test_list_models_uses_sandhi_catalog(monkeypatch, openai_provider):
    """When the Sandhi catalog is available, list_models() uses it (Victor shapes the facts)."""
    monkeypatch.setattr(
        openai_provider,
        "_models_from_sandhi",
        lambda: [
            {
                "id": "gpt-5",
                "name": "GPT-5",
                "context_window": 400_000,
                "max_output_tokens": 128_000,
            }
        ],
    )
    # The SDK must NOT be consulted when the catalog is present.
    openai_provider.client.models.list = AsyncMock(
        side_effect=AssertionError("SDK unused when catalog present")
    )

    models = await openai_provider.list_models()

    assert models == [
        {
            "id": "gpt-5",
            "name": "GPT-5",
            "context_window": 400_000,
            "max_output_tokens": 128_000,
        }
    ]


@pytest.mark.asyncio
async def test_list_models_falls_back_to_sdk_when_catalog_absent(monkeypatch, openai_provider):
    """When the Sandhi catalog is unavailable, list_models() falls back to the live API list."""
    monkeypatch.setattr(openai_provider, "_models_from_sandhi", lambda: None)
    fake_models = [
        MagicMock(id="gpt-4o", owned_by="openai", created=1),
        MagicMock(id="whisper-1", owned_by="openai", created=2),  # filtered: not chat-capable
    ]
    openai_provider.client.models.list = AsyncMock(return_value=MagicMock(data=fake_models))

    models = await openai_provider.list_models()

    assert [m["id"] for m in models] == ["gpt-4o"]


@pytest.mark.asyncio
async def test_models_from_sandhi_reads_real_catalog_when_available(openai_provider):
    """Integration: the shared catalog reader returns the curated OpenAI lineup.

    Skips cleanly when the installed Sandhi predates the catalog surface (TD-0004
    Phase A), so it is not CI-flaky.
    """
    try:
        import sandhi_gateway as sg
    except Exception:  # pragma: no cover - sandhi absent
        pytest.skip("sandhi-gateway not installed")
    if not hasattr(sg, "provider_models_json"):
        pytest.skip("installed sandhi predates the catalog surface (TD-0004 Phase A)")

    models = openai_provider._models_from_sandhi()
    assert models is not None
    ids = [m["id"] for m in models]
    assert "gpt-5" in ids
