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

"""Unit tests for MLX LM provider."""

import asyncio
import importlib.util
import subprocess
import sys

import pytest

from victor.providers.base import Message, ProviderConnectionError, ProviderError
from victor.providers.mlx_provider import MLXProvider, _model_supports_tools


def _mlx_runtime_available() -> bool:
    """Return True when mlx_lm imports successfully in this runtime."""
    if importlib.util.find_spec("mlx_lm") is None:
        return False
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mlx_lm\nimport mlx.core as mx\nprint(mx.default_device())",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return False
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _mlx_runtime_available(), reason="mlx-lm runtime unavailable")


class TestMLXProvider:
    """Test suite for MLX LM provider."""

    def test_provider_initialization(self):
        """Test provider can be initialized."""
        provider = MLXProvider(model="mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert provider.model_path == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert provider.base_url == "in-process"

    def test_model_supports_tools_detection(self):
        """Test tool support detection based on model name."""
        # Models that should support tools
        assert _model_supports_tools("mlx-community/Qwen2.5-Coder-7B-Instruct")
        assert _model_supports_tools("mlx-community/Llama-3.2-3B-Instruct")
        assert _model_supports_tools("mlx-community/Mistral-7B-Instruct-v0.3")

        # Models that might not support tools
        assert not _model_supports_tools("mlx-community/Some-Base-Model")
        assert not _model_supports_tools("mlx-community/gpt2")

    def test_supports_streaming(self):
        """Test provider claims streaming support."""
        provider = MLXProvider()
        assert provider.supports_streaming() is True

    def test_supports_tools_for_capable_models(self):
        """Test tool support detection for capable models."""
        provider = MLXProvider(model="mlx-community/Qwen2.5-Coder-7B-Instruct")
        assert provider.supports_tools() is True

    def test_list_models(self):
        """Test model listing returns curated list."""
        provider = MLXProvider()
        models = asyncio.run(provider.list_models())

        assert isinstance(models, list)
        assert len(models) > 0
        assert any("Qwen2.5" in m for m in models)
        assert any("Llama-3.2" in m for m in models)

    def test_provider_repr(self):
        """Test provider string representation."""
        provider = MLXProvider(model="mlx-community/Test-Model")
        repr_str = repr(provider)
        assert "MLXProvider" in repr_str
        assert "mlx-community/Test-Model" in repr_str


class TestMLXProviderIntegration:
    """Integration tests for MLX provider (requires MLX installed and model download).

    These tests are marked as slow and require actual model downloads.
    They should be skipped in CI environments by default.
    """

    @pytest.fixture
    def provider(self):
        """Create MLX provider instance."""
        return MLXProvider(model="mlx-community/Qwen2.5-7B-Instruct-4bit")

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skip(reason="MLX integration tests require model download and are slow")
    async def test_check_connection(self, provider):
        """Test connection check."""
        # This will try to load the model
        is_connected = await provider.check_connection()
        assert isinstance(is_connected, bool)

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skip(reason="MLX integration tests require model download and are slow")
    async def test_simple_completion(self, provider):
        """Test simple text completion."""
        messages = [Message(role="user", content="Say 'Hello, MLX!'")]
        response = await provider._make_request(
            messages=messages,
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            temperature=0.7,
            max_tokens=50,
        )

        assert response.content is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skip(reason="MLX integration tests require model download and are slow")
    async def test_multi_turn_conversation(self, provider):
        """Test multi-turn conversation."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="My name is Alice."),
            Message(role="assistant", content="Hello Alice! How can I help you today?"),
            Message(role="user", content="What's my name?"),
        ]

        response = await provider._make_request(
            messages=messages,
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            temperature=0.7,
            max_tokens=100,
        )

        assert response.content is not None
        # Response should mention "Alice"
        assert "alice" in response.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skip(reason="MLX integration tests require model download and are slow")
    async def test_streaming(self, provider):
        """Test streaming response."""
        messages = [Message(role="user", content="Count from 1 to 5")]

        chunks = []
        async for chunk in provider.stream(
            messages=messages,
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            temperature=0.7,
            max_tokens=50,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_content = "".join(c.content for c in chunks)
        assert len(full_content) > 0


class TestMLXProviderErrorHandling:
    """Test error handling in MLX provider."""

    def test_raises_import_error_when_mlx_not_available(self, monkeypatch):
        """Test that ImportError is raised when MLX is not available."""
        # Temporarily hide MLX
        import victor.providers.mlx_provider as mlx_module

        monkeypatch.setattr(mlx_module, "_MLX_IMPORT_ATTEMPTED", True)
        monkeypatch.setattr(mlx_module, "_MLX_AVAILABLE", False)
        monkeypatch.setattr(mlx_module, "_MLX_IMPORT_ERROR", RuntimeError("forced test failure"))

        with pytest.raises(ImportError, match="mlx-lm is not available"):
            MLXProvider(model="test-model")

    @pytest.mark.asyncio
    async def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        provider = MLXProvider(model="invalid/model/that/does/not/exist")

        with pytest.raises((ProviderConnectionError, ProviderError)):
            await provider._make_request(
                messages=[Message(role="user", content="Test")],
                model="invalid/model",
                temperature=0.7,
                max_tokens=10,
            )
