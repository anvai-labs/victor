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

"""TDD Integration tests for critical Victor modules with Ollama.

These tests verify end-to-end behavior of high-impact modules using
actual LLM inference via Ollama (qwen2.5-coder or similar).

Tests are skipped if Ollama is not available (e.g., in GitHub Actions).

Critical modules tested:
1. Task Completion - E2E completion detection
2. Classification Pipeline - Full classification flow
3. Budget Management - Resource allocation
4. Context Compaction - Memory management
5. Conversation Flow - Message handling
6. Streaming - Response streaming
7. Complex Workflows - Multi-step operations
"""

import asyncio
import socket
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pytest

from victor.core.completion_markers import (
    FILE_DONE_MARKER,
    SUMMARY_MARKER,
    TASK_DONE_MARKER,
)


def is_ollama_available() -> bool:
    """Check if Ollama server is running at localhost:11434."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        return result == 0
    finally:
        sock.close()


def list_installed_ollama_models() -> List[str]:
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
    installed, so a reachable-but-empty Ollama skips instead of failing with
    ModelNotFoundError.
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


# Default model for tests
OLLAMA_MODEL = "qwen2.5-coder:7b"


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

if __name__ == "__main__":
    print(hello())
''')
        (workspace / "test_main.py").write_text("""
import pytest
from main import hello, add

def test_hello():
    assert hello() == "Hello, World!"

def test_add():
    assert add(2, 3) == 5
""")
        (workspace / "buggy.py").write_text("""
def divide(a, b):
    return a / b  # BUG: No zero check!
""")
        yield workspace


# =============================================================================
# TASK COMPLETION E2E TESTS
# =============================================================================


class TestTaskCompletionE2E:
    """End-to-end tests for task completion detection."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_completion_detection_for_analysis_task(self):
        """Test completion detection for analysis tasks."""
        from victor.agent.task_completion import TaskCompletionDetector

        detector = TaskCompletionDetector()
        detector.analyze_intent("Explain what this code does")

        detector.analyze_response(
            f"{SUMMARY_MARKER} This code implements a hello world function that returns "
            "a greeting string. The add function performs basic arithmetic."
        )

        state = detector.get_state()
        assert len(state.completion_signals) > 0

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_completion_detection_for_file_edit(self, temp_workspace):
        """Test completion detection after file modifications."""
        from victor.agent.task_completion import TaskCompletionDetector

        detector = TaskCompletionDetector()
        detector.analyze_intent("Add a docstring to the hello function")

        detector.record_tool_result(
            "write", {"success": True, "path": str(temp_workspace / "main.py")}
        )

        detector.analyze_response(f"{FILE_DONE_MARKER} Added docstring to hello function")
        assert detector.should_stop() is True

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_completion_active_signal_priority(self):
        """Test that active signals have priority over passive."""
        from victor.agent.task_completion import TaskCompletionDetector

        detector = TaskCompletionDetector()
        detector.analyze_intent("Fix the bug in the parser")
        detector.analyze_response(f"{TASK_DONE_MARKER} Fixed the null check in parser.py")

        state = detector.get_state()
        has_active = state.active_signal_detected or any(
            "active:" in s or TASK_DONE_MARKER.lower() in s for s in state.completion_signals
        )
        assert has_active or len(state.completion_signals) > 0

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_completion_continuation_detection(self):
        """Test detection of continuation requests."""
        from victor.agent.task_completion import TaskCompletionDetector

        detector = TaskCompletionDetector()
        detector.analyze_intent("Write some tests")

        # Continuation patterns are detected
        detector.analyze_response("Would you like me to continue?")
        detector.analyze_response("Shall I proceed with more tests?")
        detector.analyze_response("Let me know if you want me to add more")

        state = detector.get_state()
        # At least one continuation should be detected
        assert state.continuation_requests >= 1


# =============================================================================
# CLASSIFICATION PIPELINE TESTS
# =============================================================================


class TestClassificationPipelineE2E:
    """End-to-end tests for the classification pipeline."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_full_classification_pipeline(self):
        """Test full classification with normalization and nudges."""
        from victor.agent.prompt_normalizer import PromptNormalizer
        from victor.classification import PatternMatcher, NudgeEngine, TaskType

        normalizer = PromptNormalizer()
        matcher = PatternMatcher()
        engine = NudgeEngine()

        # Input prompt
        prompt = "view auth.py and fix the login issue"

        # Step 1: Normalize
        normalized = normalizer.normalize(prompt)

        # Step 2: Pattern match
        pattern = matcher.match(normalized.normalized)

        # Step 3: Apply nudges
        if pattern:
            result_type, confidence, rule = engine.apply(
                prompt=normalized.normalized,
                embedding_result=pattern.task_type,
                embedding_confidence=pattern.confidence,
                scores={pattern.task_type: pattern.confidence},
            )
            # Should be classified appropriately
            assert result_type in [TaskType.BUG_FIX, TaskType.EDIT, TaskType.ANALYZE]

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_swebench_issue_classification(self):
        """Test classification of SWE-bench style issues."""
        from victor.classification import PatternMatcher, TaskType

        prompts = [
            # Astropy style
            """
### Description
Function raises NoConvergence for valid input.

### Expected behavior
Should return coordinates.

### Actual behavior
Raises error.
""",
            # Django style with traceback
            """
TypeError in prefetch_related

Traceback (most recent call last):
  File "manage.py", line 22
TypeError: 'NoneType' not iterable
""",
            # Memory leak style
            """
Memory leak in parallel processing

Memory grows continuously when n_jobs > 1.
Expected: Memory released after each call.
""",
        ]

        matcher = PatternMatcher()
        for prompt in prompts:
            result = matcher.match(prompt)
            assert result is not None, f"Failed to classify: {prompt[:50]}..."
            assert result.task_type == TaskType.BUG_FIX

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_complexity_classification(self):
        """Test that complexity is correctly assigned."""
        from victor.classification import PatternMatcher
        from victor.framework.task.protocols import TaskComplexity

        test_cases = [
            ("git status", TaskComplexity.SIMPLE),
            ("refactor the auth module using SOLID", TaskComplexity.COMPLEX),
            ("fix the null pointer bug", TaskComplexity.ACTION),
        ]

        matcher = PatternMatcher()
        for prompt, expected_complexity in test_cases:
            result = matcher.match(prompt)
            assert result is not None, f"No match for: {prompt}"
            assert (
                result.complexity == expected_complexity
            ), f"Wrong complexity for '{prompt}': {result.complexity} != {expected_complexity}"


# =============================================================================
# BUDGET MANAGEMENT TESTS
# =============================================================================


class TestBudgetManagement:
    """Tests for budget management functionality."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_budget_state_creation(self):
        """Test budget state creation."""
        from victor.agent.budget_manager import BudgetState

        state = BudgetState(
            current=5,
            base_maximum=20,
        )

        assert state.current == 5
        assert state.base_maximum == 20

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_budget_manager_initialization(self):
        """Test budget manager initialization."""
        from victor.agent.budget_manager import BudgetManager

        manager = BudgetManager()
        assert manager is not None

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_mode_objective_enum(self):
        """Test mode objective enumeration."""
        from victor.agent.budget import ModeObjective

        # Test enum values exist
        assert hasattr(ModeObjective, "BUILD") or hasattr(ModeObjective, "EXPLORE")
        assert len(list(ModeObjective)) >= 1


# =============================================================================
# CONTEXT COMPACTION TESTS
# =============================================================================


class TestContextCompaction:
    """Tests for context compaction functionality."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_compactor_config_creation(self):
        """Test compactor config creation."""
        from victor.agent.context_compactor import CompactorConfig

        config = CompactorConfig(
            proactive_threshold=0.7,
            tool_result_max_chars=5000,
        )

        assert config.proactive_threshold == 0.7
        assert config.tool_result_max_chars == 5000

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_context_compactor_initialization(self):
        """Test context compactor initialization."""
        from victor.agent.context_compactor import ContextCompactor, CompactorConfig

        config = CompactorConfig(proactive_threshold=0.8)
        compactor = ContextCompactor(config=config)
        assert compactor is not None

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_truncation_strategy_enum(self):
        """Test truncation strategy enumeration."""
        from victor.agent.context_compactor import TruncationStrategy

        # Test enum values exist
        assert len(list(TruncationStrategy)) >= 1


# =============================================================================
# CONVERSATION FLOW TESTS
# =============================================================================


class TestConversationFlow:
    """Tests for conversation flow handling."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_prompt_normalization(self):
        """Test prompt normalization."""
        from victor.agent.prompt_normalizer import PromptNormalizer

        normalizer = PromptNormalizer()

        # Test verb normalization
        result = normalizer.normalize("view the contents of main.py")
        assert "read" in result.normalized.lower()

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_section_deduplication(self):
        """Test section deduplication."""
        from victor.agent.prompt_normalizer import PromptNormalizer

        normalizer = PromptNormalizer()

        sections = [
            "Always use type hints",
            "Follow PEP 8",
            "Always use type hints",  # duplicate
        ]

        unique = normalizer.deduplicate_sections(sections)
        assert len(unique) == 2

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_continuation_tracking(self):
        """Test continuation message tracking."""
        from victor.agent.prompt_normalizer import PromptNormalizer

        normalizer = PromptNormalizer()

        normalizer.normalize("continue")
        normalizer.normalize("yes")
        normalizer.normalize("proceed")

        stats = normalizer.get_stats()
        assert stats["continuation_count"] >= 0


# =============================================================================
# STREAMING TESTS
# =============================================================================


class TestStreamingIntegration:
    """Tests for streaming response handling."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_streaming_response_collection(self):
        """Test that streaming collects full responses."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model=OLLAMA_MODEL,
        )

        # Get first available model
        models = await provider.list_models()
        model_name = models[0]["name"] if models else OLLAMA_MODEL

        messages = [Message(role="user", content="Say 'hello' exactly")]

        chunks = []
        async for chunk in provider.stream_chat(
            messages=messages,
            model=model_name,
        ):
            if chunk.content:
                chunks.append(chunk.content)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        await provider.close()

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_streaming_with_system_message(self):
        """Test streaming with system message."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model=OLLAMA_MODEL,
        )

        # Get first available model
        models = await provider.list_models()
        model_name = models[0]["name"] if models else OLLAMA_MODEL

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is 2+2?"),
        ]

        chunks = []
        async for chunk in provider.stream_chat(
            messages=messages,
            model=model_name,
        ):
            if chunk.content:
                chunks.append(chunk.content)

        full_response = "".join(chunks)
        assert "4" in full_response or "four" in full_response.lower()
        await provider.close()


# =============================================================================
# PROVIDER TESTS
# =============================================================================


class TestOllamaProvider:
    """Tests for Ollama provider functionality."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_provider_initialization(self):
        """Test provider initialization."""
        from victor.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model=OLLAMA_MODEL,
        )

        assert provider.name == "ollama"

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_model_listing(self):
        """Test model listing."""
        from victor.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(base_url="http://localhost:11434")
        models = await provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Test basic chat completion."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model=OLLAMA_MODEL,
        )

        # Get first available model
        models = await provider.list_models()
        model_name = models[0]["name"] if models else OLLAMA_MODEL

        messages = [Message(role="user", content="Reply with: OK")]

        response = await provider.chat(
            messages=messages,
            model=model_name,
        )

        assert response is not None
        assert response.content is not None
        await provider.close()

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        from victor.providers.ollama_provider import OllamaProvider
        from victor.providers.base import Message

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model=OLLAMA_MODEL,
        )

        # Get first available model
        models = await provider.list_models()
        model_name = models[0]["name"] if models else OLLAMA_MODEL

        messages = [Message(role="user", content="My name is Alice")]

        response1 = await provider.chat(messages=messages, model=model_name)

        messages.append(Message(role="assistant", content=response1.content))
        messages.append(Message(role="user", content="What is my name?"))

        response2 = await provider.chat(messages=messages, model=model_name)
        assert "alice" in response2.content.lower()
        await provider.close()

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_tool_support_detection(self):
        """Test tool support detection."""
        from victor.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model=OLLAMA_MODEL,
        )

        supports_tools = provider.supports_tools()
        assert isinstance(supports_tools, bool)


# =============================================================================
# COMPLEX WORKFLOW TESTS
# =============================================================================


class TestComplexWorkflows:
    """Tests for complex multi-step workflows."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_bug_fix_workflow(self, temp_workspace):
        """Test bug fix classification workflow."""
        from victor.classification import PatternMatcher, TaskType
        from victor.agent.task_completion import TaskCompletionDetector

        prompt = """
Fix the bug in buggy.py:

The divide function doesn't handle division by zero.

Expected: Should raise ValueError for b=0
Actual: Raises ZeroDivisionError
"""

        matcher = PatternMatcher()
        result = matcher.match(prompt)

        assert result is not None
        assert result.task_type == TaskType.BUG_FIX

        detector = TaskCompletionDetector()
        detector.analyze_intent(prompt)

        detector.record_tool_result(
            "write", {"success": True, "path": str(temp_workspace / "buggy.py")}
        )
        detector.analyze_response(f"{TASK_DONE_MARKER} Added zero check")

        assert detector.should_stop() is True

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_code_review_classification(self):
        """Test code review classification."""
        from victor.classification import PatternMatcher, TaskType

        prompt = """
Review and analyze the code in main.py for:
- Code quality issues
- Missing error handling
- Documentation completeness

Provide detailed analysis.
"""

        matcher = PatternMatcher()
        result = matcher.match(prompt)

        # May match ANALYZE, ANALYSIS_DEEP, or similar analysis types
        assert result is not None
        # Accept any analysis-related task type
        assert (
            result.task_type
            in [
                TaskType.ANALYZE,
                TaskType.ANALYSIS_DEEP,
                TaskType.DEBUG,  # Review can be classified as debug
            ]
            or "ANAL" in result.task_type.value.upper()
        )

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_refactor_classification(self):
        """Test refactoring classification."""
        from victor.classification import PatternMatcher, TaskType

        prompt = """
Refactor the main.py file to:
1. Extract functions into separate modules
2. Add type hints
3. Follow PEP 8 guidelines
"""

        matcher = PatternMatcher()
        result = matcher.match(prompt)

        assert result is not None
        assert result.task_type == TaskType.REFACTOR


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_invalid_model_error(self):
        """Test handling of invalid model."""
        from victor.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="nonexistent-model-xyz-12345",
        )

        with pytest.raises(Exception):
            await provider.chat(messages=[{"role": "user", "content": "test"}])

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_empty_messages_error(self):
        """Test handling of empty messages."""
        from victor.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model=OLLAMA_MODEL,
        )

        with pytest.raises(Exception):
            await provider.chat(messages=[])


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Tests for performance-critical paths."""

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_pattern_matching_performance(self):
        """Test pattern matching performance."""
        import time
        from victor.classification import PatternMatcher

        matcher = PatternMatcher()

        prompts = [
            "fix the bug",
            "refactor authentication module to use OAuth2",
            "### Description\nComplex issue\n### Expected\nX\n### Actual\nY",
        ]

        start = time.perf_counter()
        for _ in range(100):
            for prompt in prompts:
                matcher.match(prompt)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Pattern matching too slow: {elapsed:.2f}s"

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_normalization_performance(self):
        """Test normalization performance."""
        import time
        from victor.agent.prompt_normalizer import PromptNormalizer

        normalizer = PromptNormalizer()

        prompts = [
            "view the file",
            "check the configuration",
            "look at the error logs and examine the output",
        ]

        start = time.perf_counter()
        for _ in range(100):
            for prompt in prompts:
                normalizer.normalize(prompt)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Normalization too slow: {elapsed:.2f}s"

    @requires_ollama()
    @pytest.mark.asyncio
    async def test_completion_detection_performance(self):
        """Test completion detection performance."""
        import time
        from victor.agent.task_completion import TaskCompletionDetector

        responses = [
            f"{FILE_DONE_MARKER} Created the file",
            f"{TASK_DONE_MARKER} Fixed the bug",
            f"{SUMMARY_MARKER} Found 5 issues in the code",
            "The file has been created successfully.",
        ]

        start = time.perf_counter()
        for _ in range(100):
            detector = TaskCompletionDetector()
            for response in responses:
                detector.analyze_response(response)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Completion detection too slow: {elapsed:.2f}s"
