"""Tests for LLM-powered compaction summarizer."""

from unittest.mock import MagicMock

import pytest

from victor.agent.llm_compaction_summarizer import LLMCompactionSummarizer
from victor.providers.base import Message


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.chat.return_value = (
        "User asked about JWT auth. Read auth/middleware.py. Plan: add validate_token()."
    )
    return provider


@pytest.fixture
def mock_ledger():
    ledger = MagicMock()
    entry1 = MagicMock()
    entry1.category = "file_read"
    entry1.key = "auth.py"
    entry1.summary = "Read auth module"
    entry2 = MagicMock()
    entry2.category = "decision"
    entry2.key = "decision_1"
    entry2.summary = "Use factory pattern"
    ledger.entries = [entry1, entry2]
    return ledger


@pytest.fixture
def fallback():
    fb = MagicMock()
    fb.summarize.return_value = "[Fallback summary]"
    return fb


@pytest.fixture
def summarizer(mock_provider, fallback):
    return LLMCompactionSummarizer(
        provider=mock_provider,
        fallback=fallback,
        timeout_seconds=5.0,
    )


@pytest.fixture
def sample_messages():
    return [
        Message(role="user", content="Implement JWT authentication"),
        Message(role="assistant", content="I'll read the auth middleware first."),
        Message(role="tool", content="Contents of auth/middleware.py..."),
    ]


class TestLLMCompactionSummarizer:
    def test_summarize_calls_provider_chat(self, summarizer, mock_provider, sample_messages):
        result = summarizer.summarize(sample_messages)
        mock_provider.chat.assert_called_once()
        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(messages) == 1
        assert "Summarize" in messages[0].content
        assert "JWT" in messages[0].content
        assert "[Compacted context:" in result

    def test_summarize_includes_ledger_entries(
        self, summarizer, mock_provider, sample_messages, mock_ledger
    ):
        summarizer.summarize(sample_messages, ledger=mock_ledger)
        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        prompt = messages[0].content
        assert "auth.py" in prompt
        assert "factory pattern" in prompt

    def test_fallback_on_provider_error(self, summarizer, mock_provider, fallback, sample_messages):
        mock_provider.chat.side_effect = RuntimeError("API error")
        result = summarizer.summarize(sample_messages)
        assert result == "[Fallback summary]"
        fallback.summarize.assert_called_once()

    def test_fallback_on_timeout(self, fallback, sample_messages):
        import concurrent.futures

        slow_provider = MagicMock()
        slow_provider.chat.side_effect = concurrent.futures.TimeoutError("timed out")

        summarizer = LLMCompactionSummarizer(
            provider=slow_provider,
            fallback=fallback,
            timeout_seconds=0.01,
        )
        result = summarizer.summarize(sample_messages)
        assert result == "[Fallback summary]"

    def test_truncates_long_input(self, summarizer, mock_provider):
        long_messages = [
            Message(role="user", content="x" * 10000),
            Message(role="assistant", content="y" * 10000),
        ]
        summarizer.summarize(long_messages)
        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        prompt = messages[0].content
        # The total conversation content should be truncated to ~8000 chars
        assert len(prompt) < 12000

    def test_implements_protocol(self, summarizer):
        # Verify summarizer has the summarize method matching the protocol
        assert hasattr(summarizer, "summarize")
        assert callable(summarizer.summarize)
        # Structural check: same signature as CompactionSummaryStrategy
        import inspect

        sig = inspect.signature(summarizer.summarize)
        params = list(sig.parameters.keys())
        assert "removed_messages" in params
        assert "ledger" in params

    def test_empty_messages_returns_empty(self, summarizer, mock_provider):
        result = summarizer.summarize([])
        assert result == ""
        mock_provider.chat.assert_not_called()
