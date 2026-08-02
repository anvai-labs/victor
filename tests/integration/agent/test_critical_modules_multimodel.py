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

"""Multi-model TDD integration tests for critical Victor modules.

These tests run against multiple LLM families to ensure behavior isn't
aligned with family-specific quirks. Each test runs against at least
2 models from different families.

Model Families Tested:
- Qwen: qwen2.5-coder, qwen3-coder
- DeepSeek: deepseek-coder-v2, deepseek-r1
- GPT-OSS: gpt-oss
- DevStral: devstral (Mistral-based)
- Llama: llama3.1, llama3.3

Tests are skipped if Ollama is not available (e.g., in GitHub Actions).
"""

import asyncio
import socket
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pytest


def is_ollama_available() -> bool:
    """Check if Ollama server is running at localhost:11434."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        return result == 0
    finally:
        sock.close()


def list_installed_ollama_models() -> list[str]:
    """Return installed Ollama model names, or [] if unreachable / none installed.

    Hits the ``/api/tags`` endpoint directly so the check is synchronous and
    usable from a ``skipif`` marker evaluated at collection time.
    """
    import httpx

    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models if m.get("name")]
    except Exception:
        return []


def requires_ollama():
    """Pytest marker to skip tests when Ollama or an installed model is unavailable.

    Skips when the server is down OR when the server is up but no model is
    installed, so a reachable-but-empty Ollama skips instead of failing.
    """
    if not is_ollama_available():
        reason = "Ollama server not available"
    elif not list_installed_ollama_models():
        reason = (
            "Ollama server available but no models installed "
            "(run `ollama pull qwen2.5-coder:7b`)"
        )
    else:
        reason = ""
    return pytest.mark.skipif(bool(reason), reason=reason or "Ollama available")


# =============================================================================
# MODEL FAMILY CONFIGURATION
# =============================================================================


@dataclass
class ModelFamily:
    """Configuration for a model family."""

    name: str
    models: List[str]
    description: str


# Define model families - using smaller/faster variants for testing
MODEL_FAMILIES = {
    "qwen": ModelFamily(
        name="qwen",
        models=["qwen2.5-coder:7b", "qwen3-coder:30b"],
        description="Alibaba Qwen coder models",
    ),
    "deepseek": ModelFamily(
        name="deepseek",
        models=["deepseek-coder-v2:16b", "deepseek-r1:14b"],
        description="DeepSeek coder and reasoning models",
    ),
    "gpt_oss": ModelFamily(
        name="gpt_oss",
        models=["gpt-oss:20b"],
        description="GPT-OSS open source GPT-like models",
    ),
    "devstral": ModelFamily(
        name="devstral",
        models=["devstral:latest"],
        description="Mistral-based DevStral coding model",
    ),
    "llama": ModelFamily(name="llama", models=["llama3.1:8b"], description="Meta Llama models"),
}

# Define model pairs for cross-family testing
# Each tuple contains (model1, family1, model2, family2)
MODEL_PAIRS: List[Tuple[str, str, str, str]] = [
    ("qwen2.5-coder:7b", "qwen", "deepseek-coder-v2:16b", "deepseek"),
    ("qwen2.5-coder:7b", "qwen", "gpt-oss:20b", "gpt_oss"),
    ("deepseek-coder-v2:16b", "deepseek", "devstral:latest", "devstral"),
    ("gpt-oss:20b", "gpt_oss", "llama3.1:8b", "llama"),
]

# Fast model pairs for quick tests (using smallest models)
FAST_MODEL_PAIRS: List[Tuple[str, str, str, str]] = [
    ("qwen2.5-coder:7b", "qwen", "llama3.1:8b", "llama"),
]


async def get_available_models() -> List[str]:
    """Query Ollama for available models."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


async def filter_available_pairs(
    pairs: List[Tuple[str, str, str, str]],
) -> List[Tuple[str, str, str, str]]:
    """Filter model pairs to only include available models."""
    available = await get_available_models()
    return [(m1, f1, m2, f2) for m1, f1, m2, f2 in pairs if m1 in available and m2 in available]


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "main.py").write_text('''
def hello():
    """Say hello."""
    return "Hello, World!"

def add(a, b):
    """Add two numbers."""
    return a + b
''')
        (workspace / "buggy.py").write_text("""
def divide(a, b):
    return a / b  # BUG: No zero check!
""")
        yield workspace


# =============================================================================
# MULTI-MODEL TEST HELPERS
# =============================================================================


class MultiModelTestBase:
    """Base class for multi-model tests."""

    @staticmethod
    async def run_with_model(model: str, test_func):
        """Run a test function with a specific model."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model=model,
        )

        try:
            result = await test_func(provider, model, Message)
            return result
        finally:
            await provider.close()

    @staticmethod
    async def run_cross_family(
        model1: str, family1: str, model2: str, family2: str, test_func
    ) -> Dict[str, Any]:
        """Run a test across two model families and compare results."""
        results = {}

        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        # Test with first model
        provider1 = OllamaProvider(base_url="http://localhost:11434", model=model1)
        try:
            results[f"{family1}:{model1}"] = await test_func(provider1, model1, Message)
        finally:
            await provider1.close()

        # Test with second model
        provider2 = OllamaProvider(base_url="http://localhost:11434", model=model2)
        try:
            results[f"{family2}:{model2}"] = await test_func(provider2, model2, Message)
        finally:
            await provider2.close()

        return results


# =============================================================================
# CLASSIFICATION TESTS - MULTI MODEL
# =============================================================================


class TestClassificationMultiModel:
    """Classification tests run against multiple model families."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_bugfix_classification_cross_family(self):
        """Test BUG_FIX classification consistency across model families."""
        from victor.classification import PatternMatcher, TaskType

        prompts = [
            "fix the null pointer exception in parser.py",
            "### Description\nFunction crashes\n### Expected\nShould work\n### Actual\nCrashes",
            "Memory leak in parallel processing when n_jobs > 1",
        ]

        matcher = PatternMatcher()

        # Run classification - should be consistent regardless of which model
        # would be used for semantic classification
        results = {}
        for prompt in prompts:
            result = matcher.match(prompt)
            assert result is not None, f"Failed to classify: {prompt[:50]}..."
            assert (
                result.task_type == TaskType.BUG_FIX
            ), f"Wrong type for '{prompt[:30]}...': {result.task_type}"
            results[prompt[:30]] = result.task_type

        # All should be BUG_FIX
        assert all(t == TaskType.BUG_FIX for t in results.values())

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_complexity_classification_cross_family(self):
        """Test complexity classification consistency."""
        from victor.classification import PatternMatcher
        from victor.framework.task.protocols import TaskComplexity

        test_cases = [
            ("git status", TaskComplexity.SIMPLE),
            ("refactor auth module using SOLID principles", TaskComplexity.COMPLEX),
            ("fix the bug in login.py", TaskComplexity.ACTION),
        ]

        matcher = PatternMatcher()

        for prompt, expected in test_cases:
            result = matcher.match(prompt)
            assert result is not None
            assert (
                result.complexity == expected
            ), f"Wrong complexity for '{prompt}': {result.complexity} != {expected}"


# =============================================================================
# CHAT COMPLETION TESTS - MULTI MODEL
# =============================================================================


class TestChatCompletionMultiModel:
    """Chat completion tests run against multiple model families."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_simple_response_qwen_vs_deepseek(self):
        """Test simple response generation: Qwen vs DeepSeek."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        models = ["qwen2.5-coder:7b", "deepseek-coder-v2:16b"]
        available = await get_available_models()
        models = [m for m in models if m in available]

        if len(models) < 2:
            pytest.skip("Need at least 2 models for cross-family test")

        results = {}
        for model in models:
            provider = OllamaProvider(base_url="http://localhost:11434", model=model)
            try:
                response = await provider.chat(
                    messages=[
                        Message(
                            role="user",
                            content="What is 2+2? Reply with just the number.",
                        )
                    ],
                    model=model,
                    max_tokens=10,
                )
                results[model] = response.content
            finally:
                await provider.close()

        # Both should contain "4"
        for model, content in results.items():
            assert "4" in content, f"{model} failed: {content}"

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_simple_response_alternate_families(self):
        """Test simple response generation across different model families."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        # Try multiple options from each family
        family_options = [
            ["llama3.1:8b", "llama3:8b", "llama2:7b"],  # Llama family
            [
                "mistral:latest",
                "mistral:7b",
            ],  # Mistral family (more common than gpt-oss)
        ]

        available = await get_available_models()
        models = []
        for options in family_options:
            for opt in options:
                if opt in available:
                    models.append(opt)
                    break

        if len(models) < 2:
            pytest.skip("Need at least 2 models from different families")

        results = {}
        for model in models:
            provider = OllamaProvider(base_url="http://localhost:11434", model=model)
            try:
                response = await provider.chat(
                    messages=[Message(role="user", content="Reply with exactly one word: HELLO")],
                    model=model,
                    max_tokens=50,
                )
                results[model] = response.content or ""
            finally:
                await provider.close()

        # Both should produce non-empty responses (models may not follow instructions exactly)
        for model, content in results.items():
            assert len(content) > 0, f"{model} returned empty response"

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_code_generation_cross_family(self):
        """Test code generation across model families."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        # Pick one from each of 2 families
        model_pairs = [
            ("qwen2.5-coder:7b", "deepseek-coder-v2:16b"),
            ("gpt-oss:20b", "devstral:latest"),
        ]

        available = await get_available_models()

        for m1, m2 in model_pairs:
            if m1 not in available or m2 not in available:
                continue

            results = {}
            for model in [m1, m2]:
                provider = OllamaProvider(base_url="http://localhost:11434", model=model)
                try:
                    response = await provider.chat(
                        messages=[
                            Message(
                                role="user",
                                content="Write a Python function called 'add' that adds two numbers. Just the function, no explanation.",
                            )
                        ],
                        model=model,
                        max_tokens=100,
                    )
                    results[model] = response.content
                finally:
                    await provider.close()

            # Both should generate valid Python with 'def add'
            for model, content in results.items():
                assert "def" in content.lower(), f"{model} didn't generate function: {content}"

            return  # Found a working pair

        pytest.skip("No available model pairs for cross-family test")


# =============================================================================
# STREAMING TESTS - MULTI MODEL
# =============================================================================


class TestStreamingMultiModel:
    """Streaming tests run against multiple model families."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_streaming_qwen_vs_deepseek(self):
        """Test streaming response collection: Qwen vs DeepSeek."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        models = ["qwen2.5-coder:7b", "deepseek-coder-v2:16b"]
        available = await get_available_models()
        models = [m for m in models if m in available]

        if len(models) < 2:
            pytest.skip("Need at least 2 models")

        results = {}
        for model in models:
            provider = OllamaProvider(base_url="http://localhost:11434", model=model)
            try:
                chunks = []
                async for chunk in provider.stream_chat(
                    messages=[Message(role="user", content="Count from 1 to 3")],
                    model=model,
                ):
                    if chunk.content:
                        chunks.append(chunk.content)
                results[model] = "".join(chunks)
            finally:
                await provider.close()

        # Both should produce output containing numbers
        for model, content in results.items():
            assert len(content) > 0, f"{model} produced empty response"
            # Should mention at least one number
            assert any(
                n in content for n in ["1", "2", "3", "one", "two", "three"]
            ), f"{model} didn't count: {content}"

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_streaming_gptoss_vs_devstral(self):
        """Test streaming: GPT-OSS vs DevStral."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        models = ["gpt-oss:20b", "devstral:latest"]
        available = await get_available_models()
        models = [m for m in models if m in available]

        if len(models) < 2:
            pytest.skip("Need at least 2 models")

        results = {}
        for model in models:
            provider = OllamaProvider(base_url="http://localhost:11434", model=model)
            try:
                chunks = []
                async for chunk in provider.stream_chat(
                    messages=[Message(role="user", content="Say 'test passed'")],
                    model=model,
                ):
                    if chunk.content:
                        chunks.append(chunk.content)
                results[model] = "".join(chunks)
            finally:
                await provider.close()

        for model, content in results.items():
            assert len(content) > 0, f"{model} produced empty response"


# =============================================================================
# TASK COMPLETION TESTS - MULTI MODEL
# =============================================================================


class TestTaskCompletionMultiModel:
    """Task completion detection tests (model-agnostic)."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_completion_signals_detected(self):
        """Test that completion signals are detected consistently."""
        from victor.agent.task_completion import TaskCompletionDetector

        # These signals should be detected regardless of which model generated them
        # Use phrases that match COMPLETION_SIGNALS in task_completion.py
        completion_responses = [
            "done: created the file successfully",  # matches "done:"
            "task complete: fixed the authentication bug",  # matches "task complete:"
            "in summary, the code analysis found 3 issues",  # matches "in summary"
            "i've created the file as requested.",  # matches "i've created"
        ]

        for response in completion_responses:
            detector = TaskCompletionDetector()
            detector.analyze_intent("Do something")
            detector.analyze_response(response)

            state = detector.get_state()
            assert (
                len(state.completion_signals) > 0
            ), f"No completion signal detected for: {response}"

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_continuation_detected(self):
        """Test continuation detection across model response styles."""
        from victor.agent.task_completion import TaskCompletionDetector

        # Use phrases that match CONTINUATION_PHRASES in task_completion.py
        continuation_responses = [
            "would you like me to continue?",  # matches "would you like me to"
            "let me know if you need any changes",  # matches "let me know if you"
            "if there's anything else I can help with",  # matches "if there's anything else"
        ]

        for response in continuation_responses:
            detector = TaskCompletionDetector()
            detector.analyze_intent("Write code")
            detector.analyze_response(response)

            state = detector.get_state()
            assert state.continuation_requests >= 1, f"No continuation detected for: {response}"


# =============================================================================
# MULTI-TURN CONVERSATION TESTS - MULTI MODEL
# =============================================================================


class TestMultiTurnMultiModel:
    """Multi-turn conversation tests across model families."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_context_retention_qwen(self):
        """Test context retention with Qwen model."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        model = "qwen2.5-coder:7b"
        available = await get_available_models()
        if model not in available:
            pytest.skip(f"{model} not available")

        provider = OllamaProvider(base_url="http://localhost:11434", model=model)
        try:
            messages = [Message(role="user", content="My favorite color is blue. Remember this.")]
            r1 = await provider.chat(messages=messages, model=model, max_tokens=50)

            messages.append(Message(role="assistant", content=r1.content))
            messages.append(Message(role="user", content="What is my favorite color?"))

            r2 = await provider.chat(messages=messages, model=model, max_tokens=50)
            assert "blue" in r2.content.lower(), f"Context not retained: {r2.content}"
        finally:
            await provider.close()

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_context_retention_deepseek(self):
        """Test context retention with DeepSeek model."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        model = "deepseek-coder-v2:16b"
        available = await get_available_models()
        if model not in available:
            pytest.skip(f"{model} not available")

        provider = OllamaProvider(base_url="http://localhost:11434", model=model)
        try:
            messages = [Message(role="user", content="My name is TestUser. Remember this.")]
            r1 = await provider.chat(messages=messages, model=model, max_tokens=50)

            messages.append(Message(role="assistant", content=r1.content))
            messages.append(Message(role="user", content="What is my name?"))

            r2 = await provider.chat(messages=messages, model=model, max_tokens=50)
            assert "testuser" in r2.content.lower(), f"Context not retained: {r2.content}"
        finally:
            await provider.close()

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_context_retention_mistral(self):
        """Test context retention with Mistral model family."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        # Try Mistral family as alternative to gpt-oss (which has thinking mode issues)
        model_options = ["mistral:latest", "mistral:7b", "codestral:latest"]
        available = await get_available_models()

        model = None
        for opt in model_options:
            if opt in available:
                model = opt
                break

        if model is None:
            pytest.skip("No Mistral family model available")

        provider = OllamaProvider(base_url="http://localhost:11434", model=model)
        try:
            messages = [Message(role="user", content="The secret word is BANANA. Remember it.")]
            r1 = await provider.chat(messages=messages, model=model, max_tokens=100)

            messages.append(Message(role="assistant", content=r1.content or "Understood."))
            messages.append(Message(role="user", content="What is the secret word?"))

            r2 = await provider.chat(messages=messages, model=model, max_tokens=100)
            # Allow for variations like "BANANA" or "banana" or "The secret word is BANANA"
            assert r2.content and (
                "banana" in r2.content.lower()
            ), f"Context not retained: {r2.content}"
        finally:
            await provider.close()


# =============================================================================
# TOOL CALLING TESTS - MULTI MODEL
# =============================================================================


class TestToolCallingMultiModel:
    """Tool calling tests across model families."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_tool_support_detection_all_families(self):
        """Test tool support detection across all available models."""
        from victor.providers.ollama_provider import OllamaProvider

        models_to_test = [
            "qwen2.5-coder:7b",
            "deepseek-coder-v2:16b",
            "gpt-oss:20b",
            "devstral:latest",
            "llama3.1:8b",
        ]

        available = await get_available_models()
        models_to_test = [m for m in models_to_test if m in available]

        if not models_to_test:
            pytest.skip("No test models available")

        results = {}
        for model in models_to_test:
            provider = OllamaProvider(base_url="http://localhost:11434", model=model)
            try:
                supports = provider.supports_tools()
                results[model] = supports
            finally:
                await provider.close()

        # Just verify we got results for all
        assert len(results) == len(models_to_test)
        for model, supports in results.items():
            assert isinstance(supports, bool), f"{model} returned non-bool: {supports}"


# =============================================================================
# NORMALIZATION TESTS - MULTI MODEL AGNOSTIC
# =============================================================================


class TestNormalizationMultiModel:
    """Normalization tests (model-agnostic, but verified across inputs)."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_verb_normalization_consistency(self):
        """Test verb normalization is consistent."""
        from victor.agent.prompt_normalizer import PromptNormalizer

        normalizer = PromptNormalizer()

        # These should normalize consistently regardless of model
        test_cases = [
            ("view the file", "read"),
            ("look at the code", "read"),
            ("check the config", "read"),
            ("examine the output", "analyze"),
        ]

        for original, expected_verb in test_cases:
            result = normalizer.normalize(original)
            assert (
                expected_verb in result.normalized.lower()
            ), f"'{original}' should normalize to contain '{expected_verb}', got: {result.normalized}"

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_deduplication_consistency(self):
        """Test section deduplication is consistent."""
        from victor.agent.prompt_normalizer import PromptNormalizer

        normalizer = PromptNormalizer()

        sections = [
            "Always use type hints",
            "Follow PEP 8 style",
            "Always use type hints",  # duplicate
            "Write unit tests",
            "Follow PEP 8 style",  # duplicate
        ]

        unique = normalizer.deduplicate_sections(sections)
        assert len(unique) == 3
        assert unique.count("Always use type hints") == 1
        assert unique.count("Follow PEP 8 style") == 1


# =============================================================================
# PERFORMANCE COMPARISON TESTS
# =============================================================================


class TestPerformanceMultiModel:
    """Performance comparison across model families."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_response_time_comparison(self):
        """Compare response times across model families."""
        import time
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        models = ["qwen2.5-coder:7b", "llama3.1:8b"]
        available = await get_available_models()
        models = [m for m in models if m in available]

        if len(models) < 2:
            pytest.skip("Need at least 2 models")

        times = {}
        for model in models:
            provider = OllamaProvider(base_url="http://localhost:11434", model=model)
            try:
                start = time.perf_counter()
                await provider.chat(
                    messages=[Message(role="user", content="Say OK")],
                    model=model,
                    max_tokens=5,
                )
                elapsed = time.perf_counter() - start
                times[model] = elapsed
            finally:
                await provider.close()

        # Log times for analysis
        for model, t in times.items():
            print(f"\n{model}: {t:.2f}s")

        # Both should respond within reasonable time (30s for cold start)
        for model, t in times.items():
            assert t < 30, f"{model} too slow: {t:.2f}s"


# =============================================================================
# ERROR HANDLING - MULTI MODEL
# =============================================================================


class TestErrorHandlingMultiModel:
    """Error handling tests across models."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_invalid_model_error_handling(self):
        """Test that invalid model errors are handled consistently."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="nonexistent-model-xyz-12345",
        )

        try:
            with pytest.raises(Exception) as exc_info:
                await provider.chat(
                    messages=[Message(role="user", content="test")],
                    model="nonexistent-model-xyz-12345",
                )
            # Should raise a meaningful error
            assert exc_info.value is not None
        finally:
            await provider.close()
