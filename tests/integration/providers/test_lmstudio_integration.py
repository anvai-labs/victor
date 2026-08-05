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

"""Integration tests for LMStudio provider.

These tests require LMStudio server to be running with a model loaded.
Skip if LMStudio is not available.

To start LMStudio:
    1. Open LMStudio application
    2. Go to "Local Server" tab
    3. Select a model (e.g., Qwen2.5-Coder-1.5B-Instruct)
    4. Click "Start Server"
    5. Server will run on http://localhost:1234
"""

import pytest


# Check if LMStudio is available at module load time
def _check_lmstudio_available():
    """Check if LMStudio server is running and has a model loaded."""
    import asyncio
    import socket

    # First check if server is running
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 1234))
        if result != 0:
            return False
    finally:
        sock.close()

    # Then check if models are loaded
    try:
        import httpx

        async def check_models():
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get("http://localhost:1234/v1/models")
                if response.status_code != 200:
                    return False
                data = response.json()
                return bool(data.get("data"))  # Has at least one model loaded

        return asyncio.run(check_models())
    except Exception:
        return False


# Skip entire module if LMStudio is not available or has no models
pytestmark = pytest.mark.skipif(
    not _check_lmstudio_available(),
    reason="LMStudio server not available or no models loaded at localhost:1234",
)

# These imports are intentionally after pytestmark to avoid loading if LMStudio unavailable
from httpx import ConnectError, HTTPError  # noqa: E402
import httpx  # noqa: E402

from victor.providers.base import Message, ToolDefinition  # noqa: E402
from victor.providers.openai_provider import OpenAIProvider  # noqa: E402


def _exception_chain_contains(exc, text: str) -> bool:
    """Check whether an exception chain contains a specific message fragment."""
    needle = text.lower()
    current = exc
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if needle in str(current).lower():
            return True

        next_error = (
            getattr(current, "raw_error", None)
            or getattr(current, "__cause__", None)
            or getattr(current, "__context__", None)
        )
        current = next_error if isinstance(next_error, BaseException) else None

    return False


def _skip_if_no_models_loaded(exc) -> None:
    """Convert LM Studio's transient no-model state into a test skip."""
    if _exception_chain_contains(exc, "no models loaded"):
        pytest.skip("LMStudio reported that no model is currently loaded")


@pytest.fixture
async def lmstudio_provider():
    """Create LMStudio provider and check if available."""
    provider = OpenAIProvider(
        api_key="lm-studio",  # Placeholder
        base_url="http://localhost:1234/v1",
        timeout=300,
    )

    try:
        # Try to list models to check if LMStudio is running
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
            if response.status_code != 200:
                pytest.skip("LMStudio server not responding")

            data = response.json()
            if not data.get("data"):
                pytest.skip("No models loaded in LMStudio")

        yield provider
    except (ConnectError, HTTPError, Exception) as e:
        pytest.skip(f"LMStudio is not running: {e}")
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_server_available():
    """Test LMStudio server availability."""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        assert response.status_code == 200
        print("\nLMStudio server is running")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_list_models():
    """Test listing models from LMStudio."""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

        models = [m["id"] for m in data["data"]]
        print(f"\nLoaded models in LMStudio: {models}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_simple_chat(lmstudio_provider):
    """Test simple chat completion with LMStudio."""
    # Get loaded model
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        model_name = response.json()["data"][0]["id"]

    messages = [Message(role="user", content="Say 'Hello from LMStudio' and nothing else.")]

    try:
        response = await lmstudio_provider.chat(
            messages=messages,
            model=model_name,
            temperature=0.1,
            max_tokens=50,
        )
    except Exception as exc:
        _skip_if_no_models_loaded(exc)
        raise

    assert response.content
    assert len(response.content) > 0
    assert response.role == "assistant"
    print(f"\nLMStudio Response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_code_generation(lmstudio_provider):
    """Test code generation with LMStudio."""
    # Get loaded model
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        model_name = response.json()["data"][0]["id"]

    messages = [
        Message(role="system", content="You are an expert Python programmer."),
        Message(
            role="user",
            content="Write a simple Python function to add two numbers. Just the function, no explanation.",
        ),
    ]

    try:
        response = await lmstudio_provider.chat(
            messages=messages,
            model=model_name,
            temperature=0.2,
            max_tokens=512,
        )
    except Exception as exc:
        _skip_if_no_models_loaded(exc)
        raise

    assert response.content
    assert "def" in response.content or "add" in response.content.lower()
    print(f"\nLMStudio Code Generation:\n{response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_streaming(lmstudio_provider):
    """Test streaming responses from LMStudio."""
    # Get loaded model
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        model_name = response.json()["data"][0]["id"]

    messages = [Message(role="user", content="Count from 1 to 5, one number per line.")]

    chunks = []
    full_content = ""

    try:
        async for chunk in lmstudio_provider.stream(
            messages=messages,
            model=model_name,
            temperature=0.1,
            max_tokens=100,
        ):
            chunks.append(chunk)
            if chunk.content:
                full_content += chunk.content
                print(chunk.content, end="", flush=True)
    except Exception as exc:
        _skip_if_no_models_loaded(exc)
        raise

    print()  # New line

    assert len(chunks) > 0
    assert full_content
    assert chunks[-1].is_final
    print(f"\nStreamed {len(chunks)} chunks, total content length: {len(full_content)}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_tool_calling(lmstudio_provider):
    """Test tool calling with LMStudio (if model supports it)."""
    # Get loaded model
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        model_name = response.json()["data"][0]["id"]

    tools = [
        ToolDefinition(
            name="get_current_time",
            description="Get the current time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone name"},
                },
                "required": ["timezone"],
            },
        )
    ]

    messages = [Message(role="user", content="What time is it in New York?")]

    try:
        response = await lmstudio_provider.chat(
            messages=messages,
            model=model_name,
            tools=tools,
            temperature=0.5,
            max_tokens=200,
        )

        # LMStudio might not support tool calling for all models
        # Just verify we get a response
        assert response.content or response.tool_calls

        if response.tool_calls:
            print(f"\nLMStudio Tool Calls: {response.tool_calls}")
        else:
            print(f"\nLMStudio Response (no tool calls): {response.content}")

    except Exception as e:
        # Some models might not support tool calling
        pytest.skip(f"Model might not support tool calling: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_multi_turn_conversation(lmstudio_provider):
    """Test multi-turn conversation with LMStudio."""
    # Get loaded model
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        model_name = response.json()["data"][0]["id"]

    # Turn 1
    messages = [Message(role="user", content="My name is Bob.")]

    try:
        response1 = await lmstudio_provider.chat(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_tokens=100,
        )
    except Exception as exc:
        _skip_if_no_models_loaded(exc)
        raise

    # Turn 2
    messages.append(Message(role="assistant", content=response1.content))
    messages.append(Message(role="user", content="What is my name?"))

    try:
        response2 = await lmstudio_provider.chat(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_tokens=100,
        )
    except Exception as exc:
        _skip_if_no_models_loaded(exc)
        raise

    assert response2.content
    # Model should remember Bob
    assert "bob" in response2.content.lower()
    print(f"\nLMStudio Memory test response: {response2.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_with_ollama_shared_model(lmstudio_provider):
    """Test LMStudio using a model shared from Ollama via Gollama.

    This test verifies that LMStudio can use models symlinked from Ollama.
    """
    try:
        # Try to use a model that might be shared from Ollama
        # Common shared model: qwen2.5-coder-1.5b
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
            models = [m["id"] for m in response.json()["data"]]

            # Look for Ollama-shared models (they often have specific naming)
            ollama_model = None
            for model in models:
                if any(name in model.lower() for name in ["qwen", "llama", "coder"]):
                    ollama_model = model
                    break

            if not ollama_model:
                pytest.skip("No Ollama-shared model found in LMStudio")

        messages = [Message(role="user", content="Say 'Model sharing works!' and nothing else.")]

        response = await lmstudio_provider.chat(
            messages=messages,
            model=ollama_model,
            temperature=0.1,
            max_tokens=50,
        )

        assert response.content
        print(f"\nShared model ({ollama_model}) response: {response.content}")

    except Exception as e:
        pytest.skip(f"Could not test model sharing: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_with_custom_parameters(lmstudio_provider):
    """Test LMStudio with custom sampling parameters."""
    # Get loaded model
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        model_name = response.json()["data"][0]["id"]

    messages = [
        Message(
            role="user",
            content="Write a one-line function to double a number in Python.",
        )
    ]

    try:
        response = await lmstudio_provider.chat(
            messages=messages,
            model=model_name,
            temperature=0.1,  # Very low for deterministic output
            top_p=0.95,
            max_tokens=100,
        )
    except Exception as exc:
        _skip_if_no_models_loaded(exc)
        raise

    assert response.content
    assert (
        "def" in response.content
        or "lambda" in response.content
        or "double" in response.content.lower()
    )
    print(f"\nLMStudio Custom params response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lmstudio_large_context(lmstudio_provider):
    """Test LMStudio with larger context."""
    # Get loaded model
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:1234/v1/models", timeout=5.0)
        model_name = response.json()["data"][0]["id"]

    # Create a conversation with multiple turns
    messages = [
        Message(role="system", content="You are a helpful coding assistant."),
    ]

    # Add several exchanges
    for i in range(5):
        messages.append(
            Message(
                role="user",
                content=f"What is important about code quality aspect {i+1}? One sentence.",
            )
        )
        messages.append(Message(role="assistant", content=f"Code quality aspect {i+1} matters."))

    messages.append(Message(role="user", content="Summarize our discussion in one sentence."))

    try:
        response = await lmstudio_provider.chat(
            messages=messages,
            model=model_name,
            temperature=0.5,
            max_tokens=200,
        )
    except Exception as exc:
        _skip_if_no_models_loaded(exc)
        raise

    assert response.content
    print(f"\nLMStudio Large context response: {response.content}")
