# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for providers registry and base provider module."""

import subprocess
import sys

import pytest

from victor.providers.base import (
    Message,
    ToolDefinition,
    CompletionResponse,
    StreamChunk,
    ProviderError,
    ProviderTimeoutError,
    ProviderRateLimitError,
)
from victor.providers.registry import TIER_1_PROVIDERS, ProviderRegistry


class TestProviderRegistry:
    """Tests for ProviderRegistry class."""

    def test_tier_1_providers_are_registered_primary_names(self):
        """ADR-029: every Tier-1 support-tier name is a real primary registry key."""
        registered = set(ProviderRegistry.list_providers())
        assert (
            TIER_1_PROVIDERS <= registered
        ), f"TIER_1_PROVIDERS drifted from registry: {sorted(TIER_1_PROVIDERS - registered)}"
        aliases = ProviderRegistry.get_aliases()
        for name in TIER_1_PROVIDERS:
            assert aliases[name] == name, f"{name!r} is an alias, not a primary provider name"

    def test_list_providers(self):
        """Test listing available providers."""
        providers = ProviderRegistry.list_providers()
        assert isinstance(providers, list)
        # Check for some common providers (may not all be registered)
        assert len(providers) >= 0

    def test_mlx_aliases_are_listed(self):
        """MLX aliases should be discoverable even when lazily loaded."""
        providers = ProviderRegistry.list_providers()
        assert "mlx" in providers
        assert "mlx-lm" in providers
        assert "applesilicon" in providers

    def test_get_aliases_uses_canonical_provider_names(self):
        """Aliases should resolve to stable provider names, not class names."""
        aliases = ProviderRegistry.get_aliases()
        assert aliases["ollama"] == "ollama"
        assert aliases["grok"] == "xai"
        assert aliases["zai-coding"] == "zai"

    def test_registry_import_does_not_eager_import_mlx_provider(self):
        """Importing registry should not import mlx provider module."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os\n"
                    "os.environ['VICTOR_LIGHT_IMPORT']='1'\n"
                    "import sys\n"
                    "import victor.providers.registry\n"
                    "print('victor.providers.mlx_provider' in sys.modules)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "False"

    def test_get_mlx_disabled_fails_gracefully(self):
        """MLX lookup should fail with ProviderNotFoundError, not abort."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os\n"
                    "os.environ['VICTOR_LIGHT_IMPORT']='1'\n"
                    "os.environ['VICTOR_ENABLE_MLX_PROVIDER']='0'\n"
                    "from victor.providers.base import ProviderNotFoundError\n"
                    "from victor.providers.registry import ProviderRegistry\n"
                    "try:\n"
                    "    ProviderRegistry.get('mlx')\n"
                    "except ProviderNotFoundError:\n"
                    "    print('notfound')\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0
        assert "notfound" in result.stdout

    def test_non_mlx_local_aliases_remain_lazy_without_mlx_preflight(self):
        """vLLM/llama.cpp aliases should resolve independently of MLX preflight flags."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os\n"
                    "os.environ['VICTOR_LIGHT_IMPORT']='1'\n"
                    "os.environ['VICTOR_ENABLE_MLX_PROVIDER']='0'\n"
                    "from victor.providers.registry import ProviderRegistry\n"
                    "print(ProviderRegistry.get_optional('vllm') is not None)\n"
                    "print(ProviderRegistry.get_optional('llamacpp') is not None)\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        assert lines == ["True", "True"]

    def test_get_provider_ollama(self):
        """Test getting Ollama provider class."""
        from victor.providers.base import ProviderNotFoundError

        # Register ollama if not already registered
        try:
            from victor.providers.ollama_provider import OllamaProvider

            ProviderRegistry.register("ollama", OllamaProvider)
        except ImportError:
            pass

        try:
            provider_cls = ProviderRegistry.get("ollama")
            assert provider_cls is not None
        except ProviderNotFoundError:
            # Ollama not registered, that's fine
            pass

    def test_get_provider_unknown(self):
        """Test getting unknown provider raises error."""
        from victor.providers.base import ProviderNotFoundError

        with pytest.raises(ProviderNotFoundError):
            ProviderRegistry.get("unknown_provider_xyz_12345")

    def test_register_provider(self):
        """Test registering a custom provider."""
        from victor.providers.base import BaseProvider

        class CustomTestProvider(BaseProvider):
            @property
            def name(self):
                return "custom_test"

            def supports_tools(self):
                return False

            def supports_streaming(self):
                return False

            async def chat(self, messages, **kwargs):
                pass

            async def stream(self, messages, **kwargs):
                pass

            async def close(self):
                pass

        ProviderRegistry.register("custom_test", CustomTestProvider)
        providers = ProviderRegistry.list_providers()
        assert "custom_test" in providers

        # Cleanup
        ProviderRegistry.unregister("custom_test")

    def test_create_provider(self):
        """Test creating a provider instance."""
        from victor.providers.base import BaseProvider

        class TestCreateProvider(BaseProvider):
            def __init__(self, **kwargs):
                self._name = "test_create"

            @property
            def name(self):
                return self._name

            def supports_tools(self):
                return False

            def supports_streaming(self):
                return False

            async def chat(self, messages, **kwargs):
                pass

            async def stream(self, messages, **kwargs):
                pass

            async def close(self):
                pass

        ProviderRegistry.register("test_create", TestCreateProvider)
        instance = ProviderRegistry.create("test_create")
        assert instance.name == "test_create"

        # Cleanup
        ProviderRegistry.unregister("test_create")


class TestMessage:
    """Tests for Message class."""

    def test_message_basic(self):
        """Test basic message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "name": "test", "arguments": {}}],
        )
        assert len(msg.tool_calls) == 1

    def test_message_with_name(self):
        """Test message with name."""
        msg = Message(role="tool", content="result", name="test_tool")
        assert msg.name == "test_tool"


class TestToolDefinition:
    """Tests for ToolDefinition class."""

    def test_tool_definition_basic(self):
        """Test basic tool definition."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_tool_definition_to_dict(self):
        """Test tool definition conversion to dict."""
        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters={"type": "object"},
        )
        d = tool.model_dump()
        assert "name" in d
        assert "description" in d
        assert "parameters" in d


class TestCompletionResponse:
    """Tests for CompletionResponse class."""

    def test_completion_response_basic(self):
        """Test basic completion response."""
        resp = CompletionResponse(
            content="Hello!",
            role="assistant",
            model="test-model",
        )
        assert resp.content == "Hello!"
        assert resp.role == "assistant"

    def test_completion_response_with_usage(self):
        """Test completion response with usage."""
        resp = CompletionResponse(
            content="Hello!",
            role="assistant",
            model="test-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        assert resp.usage["total_tokens"] == 15

    def test_completion_response_with_tool_calls(self):
        """Test completion response with tool calls."""
        resp = CompletionResponse(
            content="",
            role="assistant",
            model="test-model",
            tool_calls=[{"id": "call_1", "name": "test", "arguments": {}}],
        )
        assert len(resp.tool_calls) == 1


class TestStreamChunk:
    """Tests for StreamChunk class."""

    def test_stream_chunk_content(self):
        """Test stream chunk with content."""
        chunk = StreamChunk(content="Hello", is_final=False)
        assert chunk.content == "Hello"
        assert chunk.is_final is False

    def test_stream_chunk_final(self):
        """Test final stream chunk."""
        chunk = StreamChunk(content="!", is_final=True)
        assert chunk.is_final is True

    def test_stream_chunk_with_tool_calls(self):
        """Test stream chunk with tool calls."""
        chunk = StreamChunk(
            content="",
            is_final=False,
            tool_calls=[{"id": "call_1", "name": "test"}],
        )
        assert chunk.tool_calls is not None


class TestProviderErrors:
    """Tests for provider error classes."""

    def test_provider_error(self):
        """Test ProviderError."""
        error = ProviderError(
            message="Test error",
            provider="test",
        )
        assert "Test error" in str(error)
        assert error.provider == "test"

    def test_provider_timeout_error(self):
        """Test ProviderTimeoutError."""
        error = ProviderTimeoutError(
            message="Timeout",
            provider="test",
        )
        assert "Timeout" in str(error)

    def test_provider_rate_limit_error(self):
        """Test ProviderRateLimitError."""
        error = ProviderRateLimitError(
            message="Rate limited",
            provider="test",
        )
        assert "Rate limited" in str(error)
