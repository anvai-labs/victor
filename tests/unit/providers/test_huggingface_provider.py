# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for HuggingFace Inference API provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from victor.providers.huggingface_provider import (
    HuggingFaceProvider,
    HUGGINGFACE_MODELS,
)


class TestHuggingFaceProviderInitialization:
    """Tests for HuggingFaceProvider initialization."""

    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = HuggingFaceProvider(api_key="hf_test_key_123")
        assert provider._api_key == "hf_test_key_123"
        assert provider.name == "huggingface"

    def test_initialization_from_hf_token_env(self):
        """Test API key loading from HF_TOKEN environment variable."""
        with patch.dict("os.environ", {"HF_TOKEN": "hf_env_token", "HUGGINGFACE_API_KEY": ""}):
            provider = HuggingFaceProvider()
            assert provider._api_key == "hf_env_token"

    def test_initialization_from_huggingface_api_key_env(self):
        """Test API key loading from HUGGINGFACE_API_KEY environment variable."""
        with patch.dict("os.environ", {"HF_TOKEN": "", "HUGGINGFACE_API_KEY": "hf_api_key"}):
            provider = HuggingFaceProvider()
            assert provider._api_key == "hf_api_key"

    @pytest.mark.skip(
        reason="Keyring mocking not working in test environment - requires system keyring configuration"
    )
    def test_initialization_from_keyring(self):
        """Test API key loading from keyring when env var not set."""
        with patch.dict("os.environ", {"HF_TOKEN": "", "HUGGINGFACE_API_KEY": ""}, clear=False):
            # Mock the keyring source in unified resolver
            with patch(
                "victor.providers.resolution._get_key_from_keyring",
                return_value="keyring-hf-key",
            ):
                provider = HuggingFaceProvider(non_interactive=False)
                assert provider._api_key == "keyring-hf-key"

    def test_initialization_error_without_key(self):
        """Test APIKeyNotFoundError is raised when no API key provided."""
        from victor.providers.resolution import APIKeyNotFoundError

        with patch.dict("os.environ", {"HF_TOKEN": "", "HUGGINGFACE_API_KEY": ""}, clear=False):
            # Mock all sources to return None
            with patch(
                "victor.providers.resolution._get_key_from_keyring",
                return_value=None,
            ):
                with pytest.raises(APIKeyNotFoundError) as exc_info:
                    HuggingFaceProvider(non_interactive=True)

                assert "HUGGINGFACE API key not found" in str(exc_info.value)

    def test_hf_token_priority_over_huggingface_api_key(self):
        """Test HF_TOKEN takes priority over HUGGINGFACE_API_KEY."""
        with patch.dict(
            "os.environ",
            {"HF_TOKEN": "primary_token", "HUGGINGFACE_API_KEY": "secondary_key"},
        ):
            provider = HuggingFaceProvider()
            assert provider._api_key == "primary_token"


class TestHuggingFaceProviderCapabilities:
    """Tests for HuggingFaceProvider capability reporting."""

    def test_name_property(self):
        """Test provider name."""
        provider = HuggingFaceProvider(api_key="test-key")
        assert provider.name == "huggingface"

    def test_supports_tools(self):
        """Test tool support reporting (model dependent)."""
        provider = HuggingFaceProvider(api_key="test-key")
        assert provider.supports_tools() is True  # Reports True, but model dependent

    def test_supports_streaming(self):
        """Test streaming support reporting."""
        provider = HuggingFaceProvider(api_key="test-key")
        assert provider.supports_streaming() is True


class TestHuggingFaceProviderModels:
    """Tests for model definitions."""

    def test_model_definitions_exist(self):
        """Test that model definitions are present."""
        assert len(HUGGINGFACE_MODELS) > 0

    def test_llama_models_defined(self):
        """Test Llama models are defined."""
        assert "meta-llama/Llama-3.3-70B-Instruct" in HUGGINGFACE_MODELS
        assert "meta-llama/Llama-3.1-70B-Instruct" in HUGGINGFACE_MODELS
        assert "meta-llama/Llama-3.1-8B-Instruct" in HUGGINGFACE_MODELS

    def test_qwen_models_defined(self):
        """Test Qwen models are defined."""
        assert "Qwen/Qwen2.5-72B-Instruct" in HUGGINGFACE_MODELS
        assert "Qwen/Qwen2.5-Coder-32B-Instruct" in HUGGINGFACE_MODELS

    def test_model_api_type(self):
        """Test models have correct API type."""
        for model_id, model_info in HUGGINGFACE_MODELS.items():
            assert model_info.get("api_type") == "chat", f"{model_id} should have chat API type"

    def test_tool_support_varies_by_model(self):
        """Test that tool support varies by model."""
        # Llama models support tools
        assert HUGGINGFACE_MODELS["meta-llama/Llama-3.3-70B-Instruct"]["supports_tools"] is True
        # Mistral 7B doesn't
        assert HUGGINGFACE_MODELS["mistralai/Mistral-7B-Instruct-v0.3"]["supports_tools"] is False


class TestHuggingFaceProviderRequestPayload:
    """Tests for request payload building."""

    def test_basic_payload_structure(self):
        """Test basic request payload structure."""
        provider = HuggingFaceProvider(api_key="test-key")
        from victor.providers.base import Message

        messages = [Message(role="user", content="Hello")]
        payload = provider._build_request_payload(
            messages=messages,
            model="meta-llama/Llama-3.3-70B-Instruct",
            temperature=0.7,
            max_tokens=4096,
            tools=None,
            stream=False,
        )

        assert payload["model"] == "meta-llama/Llama-3.3-70B-Instruct"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 4096
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1

    def test_payload_with_tools(self):
        """Test request payload includes tools when provided."""
        provider = HuggingFaceProvider(api_key="test-key")
        from victor.providers.base import Message, ToolDefinition

        messages = [Message(role="user", content="Get current time")]
        tools = [
            ToolDefinition(
                name="get_time",
                description="Get the current time",
                parameters={"type": "object", "properties": {}},
            )
        ]

        payload = provider._build_request_payload(
            messages=messages,
            model="meta-llama/Llama-3.3-70B-Instruct",
            temperature=0.7,
            max_tokens=4096,
            tools=tools,
            stream=False,
        )

        assert "tools" in payload
        assert payload["tools"][0]["function"]["name"] == "get_time"
        assert payload["tool_choice"] == "auto"


class TestHuggingFaceProviderResponseParsing:
    """Tests for response parsing."""

    def test_parse_basic_response(self):
        """Test parsing a basic chat response."""
        provider = HuggingFaceProvider(api_key="test-key")

        response = {
            "id": "chatcmpl-hf-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 6,
                "total_tokens": 14,
            },
        }

        result = provider._parse_response(response, "meta-llama/Llama-3.3-70B-Instruct")

        assert result.content == "Hello! How can I help?"
        assert result.role == "assistant"
        assert result.stop_reason == "stop"

    def test_parse_response_with_tool_calls(self):
        """Test parsing response with tool calls."""
        provider = HuggingFaceProvider(api_key="test-key")

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_hf_1",
                                "type": "function",
                                "function": {
                                    "name": "search_code",
                                    "arguments": '{"query": "def main"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        result = provider._parse_response(response, "test-model")

        assert result.tool_calls is not None
        assert result.tool_calls[0]["name"] == "search_code"
        assert result.tool_calls[0]["arguments"] == {"query": "def main"}

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        provider = HuggingFaceProvider(api_key="test-key")

        response = {"choices": []}
        result = provider._parse_response(response, "test-model")

        assert result.content == ""


class TestHuggingFaceProviderStreaming:
    """Tests for streaming functionality."""

    def test_parse_stream_chunk_content(self):
        """Test parsing a stream chunk with content."""
        provider = HuggingFaceProvider(api_key="test-key")

        chunk_data = {
            "choices": [
                {
                    "delta": {"content": "Here's"},
                    "finish_reason": None,
                }
            ]
        }

        result = provider._parse_stream_chunk(chunk_data, [])

        assert result.content == "Here's"
        assert result.is_final is False

    def test_parse_stream_chunk_final(self):
        """Test parsing final stream chunk."""
        provider = HuggingFaceProvider(api_key="test-key")

        chunk_data = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                }
            ]
        }

        result = provider._parse_stream_chunk(chunk_data, [])

        assert result.is_final is True
        assert result.stop_reason == "stop"


class TestHuggingFaceProviderChat:
    """Tests for chat completion."""

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Test successful chat completion."""
        provider = HuggingFaceProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'm a Hugging Face model!",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_response

            from victor.providers.base import Message

            result = await provider.chat(
                messages=[Message(role="user", content="Who are you?")],
                model="meta-llama/Llama-3.3-70B-Instruct",
            )

            assert result.content == "I'm a Hugging Face model!"

    @pytest.mark.asyncio
    async def test_chat_model_loading_error(self):
        """Test handling of 503 model loading status."""
        provider = HuggingFaceProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Model is loading"

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = httpx.HTTPStatusError(
                "503", request=MagicMock(), response=mock_response
            )

            from victor.providers.base import Message, ProviderError

            with pytest.raises(ProviderError) as exc_info:
                await provider.chat(
                    messages=[Message(role="user", content="Hello")],
                    model="meta-llama/Llama-3.3-70B-Instruct",
                )

            assert "loading" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_chat_timeout_error(self):
        """Test chat timeout handling."""
        provider = HuggingFaceProvider(api_key="test-key")

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = httpx.TimeoutException("Timeout")

            from victor.providers.base import Message, ProviderTimeoutError

            with pytest.raises(ProviderTimeoutError):
                await provider.chat(
                    messages=[Message(role="user", content="Hello")],
                    model="test-model",
                )

    @pytest.mark.asyncio
    async def test_chat_url_includes_model(self):
        """Test that chat URL includes model name."""
        provider = HuggingFaceProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            provider, "_execute_with_circuit_breaker", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = mock_response

            from victor.providers.base import Message

            await provider.chat(
                messages=[Message(role="user", content="Hello")],
                model="meta-llama/Llama-3.3-70B-Instruct",
            )

            # Check the URL passed to _execute_with_circuit_breaker
            call_args = mock_exec.call_args
            url = call_args[0][1]  # Second positional arg is URL
            assert "meta-llama/Llama-3.3-70B-Instruct" in url
            assert "/v1/chat/completions" in url


class TestHuggingFaceProviderCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test provider cleanup."""
        provider = HuggingFaceProvider(api_key="test-key")

        with patch.object(provider.client, "aclose", new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_called_once()
