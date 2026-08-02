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

"""Integration tests for 01_first_agent notebook using Ollama.

These tests run against local Ollama models only.

Model Selection (Qwen 3.5 - latest 2026 generation):
- Primary: qwen3.5:4b (4B params, 3.4GB, 256K context, tool calling + coding)
- Fallbacks: qwen3.5:2b, qwen3.5:0.8b, qwen3-coder, qwen2.5:3b

Pull the recommended model:
  ollama pull qwen3.5:4b

Qwen 3.5 features:
- 256K context window
- Native tool calling support
- Multimodal (text + image)
- 201 languages
"""

import json
import os
import socket
from urllib.parse import urlparse
from urllib.request import urlopen

import pytest

from victor.framework import Agent
from victor.framework.events import EventType

PREFERRED_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _ollama_base_url() -> str:
    """Return normalized Ollama base URL from environment."""
    raw = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def _ollama_host_port() -> tuple[str, int]:
    """Resolve Ollama host and port from OLLAMA_HOST."""
    parsed = urlparse(_ollama_base_url())
    return parsed.hostname or "localhost", parsed.port or 11434


def _check_ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    host, port = _ollama_host_port()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _list_ollama_models() -> list[str]:
    """Return available local Ollama model names."""
    try:
        with urlopen(f"{_ollama_base_url()}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    names: list[str] = []
    for model in payload.get("models", []):
        name = model.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _select_model(models: list[str]) -> str:
    """Select preferred model with sensible fallback.

    Priority order (Qwen 3.5 first - latest 2026 generation):
    1. qwen3.5:4b - 4B parameters, 3.4GB, 256K context, tool calling
    2. qwen3.5:2b - 2B parameters, 2.7GB
    3. qwen3.5:0.8b - 0.8B parameters, 1GB
    4. qwen3-coder - Coding optimized
    5. qwen2.5:3b - Legacy fallback
    6. First available model as fallback
    """
    # Qwen 3.5 models first (latest generation, 256K context)
    fast_models = [
        "qwen3.5:4b",  # 3.4GB - Best balance: speed + tool calling
        "qwen3.5:2b",  # 2.7GB - Very fast
        "qwen3.5:0.8b",  # 1GB - Ultra fast
        "qwen3.5:9b",  # 6.6GB - Capable
        "qwen3-coder",  # Qwen3 coding optimized
        "qwen3-coder-next",
        # Legacy Qwen 2.5 fallbacks
        "qwen2.5:3b",
        "qwen2.5-coder:3b",
        "qwen2.5-coder:7b",
        "qwen2.5:7b",
        # Other fast models
        "phi-4:3.8b-mini",
        "phi-4-mini",
        "llama3.2:3b",
        "llama3.1:8b",
        "gemma3:4b",
    ]

    for model in fast_models:
        # Match with or without tag suffix (e.g., "qwen3.5:4b" matches "qwen3.5:4b-q4_K_M")
        if any(
            model == m or m.startswith(model + "-") or m.startswith(model + ":") for m in models
        ):
            return model

    # Fallback to first available
    return models[0]


@pytest.fixture(scope="module")
def ollama_model() -> str:
    """Pick a local Ollama model for integration tests."""
    models = _list_ollama_models()
    if not models:
        pytest.skip("No Ollama models available")
    return _select_model(models)


@pytest.fixture
async def ollama_agent(ollama_model: str):
    """Create and clean up an Ollama-backed agent."""
    agent = await Agent.create(provider="ollama", model=ollama_model)
    try:
        yield agent
    finally:
        await agent.close()


@pytest.mark.integration
@pytest.mark.skipif(not _check_ollama_available(), reason="Ollama server not available")
class TestFirstAgentNotebook:
    """Integration tests for the first agent notebook.

    Based on: docs/tutorials/notebooks/01_first_agent.ipynb
    """

    @pytest.mark.asyncio
    async def test_basic_agent_with_default_settings(self, ollama_agent):
        """Test creating agent with default settings and running a query."""
        result = await ollama_agent.run("2+2=? Answer: one number.")

        assert result.success is True
        assert result.content
        assert "4" in result.content.lower()
        print(f"Agent response: {result.content}")

    @pytest.mark.asyncio
    async def test_agent_with_ollama_provider(self, ollama_model):
        """Test agent with explicit Ollama provider."""
        async with await Agent.create(provider="ollama", model=ollama_model) as agent:
            result = await agent.run("Python is? One word.")
            assert result.success is True
            assert result.content

    @pytest.mark.asyncio
    async def test_streaming_responses(self, ollama_agent):
        """Test streaming responses for real-time feedback."""
        events_received = []
        content_parts = []

        async for event in ollama_agent.stream("Count: 1 2 3"):
            events_received.append(event.type)

            if event.type == EventType.CONTENT and event.content:
                content_parts.append(event.content)
            elif event.type == EventType.TOOL_CALL:
                print(f"[Tool: {event.tool_name}]")
            elif event.type == EventType.THINKING:
                print("[Thinking...]")

        assert len(events_received) > 0
        assert EventType.CONTENT in events_received
        assert "".join(content_parts).strip()

    @pytest.mark.asyncio
    async def test_multi_turn_conversations(self, ollama_agent):
        """Test maintaining context across multiple messages."""
        session = ollama_agent.chat("Color=blue.")
        response2 = await session.send("What color? One word.")
        assert response2.success is True
        assert response2.content

        content_lower = response2.content.lower()
        assert "blue" in content_lower

    @pytest.mark.asyncio
    async def test_agent_with_explicit_provider_config(self, ollama_model):
        """Test agent creation with explicit provider configuration."""
        async with await Agent.create(provider="ollama", model=ollama_model) as agent:
            result = await agent.run("Say: test")
            assert result.success is True
            assert "test" in result.content.lower()

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, ollama_agent):
        """Test agent handles errors gracefully."""
        result = await ollama_agent.run("")
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error

        # Shorter query for speed
        long_query = "History of universe: 10 words. " * 3
        result = await ollama_agent.run(long_query)
        assert isinstance(result.success, bool)


if __name__ == "__main__":
    # Run tests directly
    import sys

    pytest.main([__file__, "-v", "-s"])
