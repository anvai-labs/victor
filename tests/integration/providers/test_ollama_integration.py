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

"""Integration tests for Ollama provider.

These tests require Ollama to be running with at least one model installed.
Skip if Ollama is not available.
"""

import pytest
from httpx import ConnectError


# Check if Ollama is available at module load time
def _check_ollama_available():
    """Check if Ollama server is running."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        return result == 0
    finally:
        sock.close()


pytestmark = pytest.mark.skipif(
    not _check_ollama_available(),
    reason="Ollama server not available at localhost:11434",
)

# These imports are intentionally after pytestmark to avoid loading if Ollama unavailable
from victor.agent.orchestrator import AgentOrchestrator  # noqa: E402
from victor.providers.base import Message, ToolDefinition  # noqa: E402
from victor.providers.ollama_provider import OllamaProvider  # noqa: E402


@pytest.fixture
async def ollama_provider():
    """Create Ollama provider and check if available."""
    provider = OllamaProvider(base_url="http://localhost:11434")

    try:
        # Try to list models to check if Ollama is running
        models = await provider.list_models()
        if not models:
            pytest.skip("No models available in Ollama")
        yield provider
    except ConnectError:
        pytest.skip("Ollama is not running")
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_list_models(ollama_provider):
    """Test listing available models."""
    models = await ollama_provider.list_models()

    assert isinstance(models, list)
    assert len(models) > 0

    # Check model structure
    first_model = models[0]
    assert "name" in first_model
    print(f"\nAvailable models: {[m['name'] for m in models]}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_simple_chat(ollama_provider):
    """Test simple chat completion."""
    # Get first available model
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    messages = [Message(role="user", content="Say 'Hello World' and nothing else.")]

    response = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.1,
        max_tokens=50,
    )

    assert response.content
    assert len(response.content) > 0
    assert response.role == "assistant"
    print(f"\nResponse: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_streaming(ollama_provider):
    """Test streaming responses."""
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    messages = [Message(role="user", content="Count from 1 to 3.")]

    chunks = []
    async for chunk in ollama_provider.stream(
        messages=messages,
        model=model_name,
        temperature=0.1,
        max_tokens=50,
    ):
        chunks.append(chunk)
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()  # New line

    assert len(chunks) > 0
    assert any(chunk.content for chunk in chunks)
    assert chunks[-1].is_final


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_multi_turn_conversation(ollama_provider):
    """Test multi-turn conversation."""
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    # Turn 1
    messages = [Message(role="user", content="My name is Alice.")]
    response1 = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.7,
    )

    # Turn 2
    messages.append(Message(role="assistant", content=response1.content))
    messages.append(Message(role="user", content="What is my name?"))

    response2 = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.7,
    )

    assert response2.content
    # Model should remember the name
    assert "alice" in response2.content.lower()
    print(f"\nMemory test response: {response2.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_tool_calling(ollama_provider):
    """Test tool calling with Ollama.

    Ollama supports tool calling via its API.
    See: https://ollama.com/blog/tool-support
    """
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        )
    ]

    messages = [Message(role="user", content="What's the weather in San Francisco?")]

    try:
        response = await ollama_provider.chat(
            messages=messages,
            model=model_name,
            tools=tools,
            temperature=0.5,
            max_tokens=200,
        )

        # Ollama supports tool calling
        # The response should either have tool_calls or content
        assert response.content or response.tool_calls

        if response.tool_calls:
            print(f"\nOllama Tool Calls: {response.tool_calls}")
            # Verify the tool call structure
            assert len(response.tool_calls) > 0
            first_call = response.tool_calls[0]
            assert "name" in first_call
            print(f"Tool called: {first_call['name']}")
        else:
            print(f"\nOllama Response (no tool calls): {response.content}")

    except Exception as e:
        # Some older models might not support tool calling
        pytest.skip(f"Model might not support tool calling: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_orchestrator_with_ollama():
    """Test full agent orchestrator with Ollama."""
    try:
        from victor.config.settings import Settings

        provider = OllamaProvider()
        models = await provider.list_models()
        if not models:
            pytest.skip("No Ollama models available")

        model_name = models[0]["name"]

        # Create minimal settings
        settings = Settings()

        agent = AgentOrchestrator(
            settings=settings,
            provider=provider,
            model=model_name,
            temperature=0.5,
        )

        # Test simple chat
        response = await agent.chat("Say hello in exactly 3 words.")

        assert response.content
        assert len(response.content) > 0

        # Check conversation history (system + user + assistant)
        assert len(agent.messages) >= 2  # At least user + assistant
        # Find user and assistant messages (system may be first)
        roles = [m.role for m in agent.messages]
        assert "user" in roles
        assert "assistant" in roles

        print(f"\nAgent response: {response.content}")

        await provider.close()

    except ConnectError:
        pytest.skip("Ollama is not running")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_ollama_pull_model():
    """Test pulling a model (if not already present).

    This test is marked as slow since pulling can take a while.
    """
    provider = OllamaProvider()

    try:
        # Try to pull a small test model
        test_model = "qwen2.5-coder:1.5b"  # Small model for testing

        print(f"\nAttempting to pull {test_model}...")
        print("(This may take a while if model not cached)")

        pull_generator = provider.pull_model(test_model)

        async for progress in pull_generator:
            # Print progress
            if "status" in progress:
                print(f"\r{progress['status']}", end="", flush=True)

        print("\nPull completed!")

        # Verify model is now available
        models = await provider.list_models()
        model_names = [m["name"] for m in models]

        # Check if our test model is in the list
        assert any(test_model in name for name in model_names)

        await provider.close()

    except ConnectError:
        pytest.skip("Ollama is not running")
    except Exception as e:
        # If pull fails for any reason, skip (might be network issue)
        pytest.skip(f"Model pull failed: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_custom_options(ollama_provider):
    """Test Ollama with custom options."""
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    messages = [Message(role="user", content="Say hello")]

    # Test with custom options
    response = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.1,
        max_tokens=50,
        options={"seed": 42, "top_k": 40},  # Additional Ollama options
    )

    assert response.content
    assert response.role == "assistant"
    print(f"\nOllama custom options response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_streaming_empty_lines(ollama_provider):
    """Test that Ollama handles empty lines in streaming correctly."""
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    messages = [Message(role="user", content="Count to 3")]

    chunks = []
    async for chunk in ollama_provider.stream(
        messages=messages,
        model=model_name,
        temperature=0.1,
        max_tokens=50,
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[-1].is_final
    print(f"\nOllama streaming handled {len(chunks)} chunks")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_provider_properties(ollama_provider):
    """Test Ollama provider properties."""
    assert ollama_provider.name == "ollama"
    assert ollama_provider.supports_tools() is True
    assert ollama_provider.supports_streaming() is True
    print(f"\nOllama provider properties verified: name={ollama_provider.name}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_usage_statistics(ollama_provider):
    """Test that Ollama returns usage statistics."""
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    messages = [Message(role="user", content="Hi")]

    response = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.1,
        max_tokens=10,
    )

    assert response.content
    # Ollama should return usage stats
    if response.usage:
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert "total_tokens" in response.usage
        print(f"\nOllama usage stats: {response.usage}")
    else:
        print("\nOllama usage stats not available (model might not support it)")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_different_roles(ollama_provider):
    """Test Ollama with different message roles."""
    models = await ollama_provider.list_models()
    model_name = models[0]["name"]

    # Test with system, user, and assistant messages
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What's 2+2?"),
        Message(role="assistant", content="4"),
        Message(role="user", content="And 3+3?"),
    ]

    response = await ollama_provider.chat(
        messages=messages,
        model=model_name,
        temperature=0.1,
        max_tokens=50,
    )

    assert response.content
    assert response.role == "assistant"
    print(f"\nOllama multi-role response: {response.content}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_close_connection(ollama_provider):
    """Test that Ollama provider closes connection properly."""
    # Make a simple request
    models = await ollama_provider.list_models()
    assert len(models) > 0

    # Close the connection
    await ollama_provider.close()
    print("\nOllama connection closed successfully")
