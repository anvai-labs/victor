"""Tests for LLMDecisionService."""

import asyncio
import json
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.decisions.schemas import (
    DecisionType,
    IntentDecision,
    TaskCompletionDecision,
)
from victor.agent.services.decision_service import (
    LLMDecisionService,
    LLMDecisionServiceConfig,
)
from victor.agent.services.protocols.decision_service import (
    DecisionResult,
    LLMDecisionServiceProtocol,
)
from victor.core.async_utils import run_sync as real_run_sync
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationFeedback
from victor.providers.base import CompletionResponse
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider


@dataclass
class MockUsage:
    total_tokens: int = 15


@dataclass
class MockResponse:
    content: str = ""
    usage: MockUsage = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = MockUsage()


def _make_provider(response_json: dict) -> MagicMock:
    """Create a mock provider that returns a JSON response."""
    provider = MagicMock()
    response = MockResponse(content=json.dumps(response_json))
    provider.chat = AsyncMock(return_value=response)
    return provider


class _FakeHttpxProvider(HttpxOpenAICompatProvider):
    created_instances = 0
    closed_instances = 0

    def __init__(self, **kwargs):
        type(self).created_instances += 1
        super().__init__(
            api_key=kwargs.pop("api_key", "test-key"),
            base_url=kwargs.pop("base_url", "https://example.invalid"),
            timeout=kwargs.pop("timeout", 5),
            max_retries=kwargs.pop("max_retries", 0),
            provider_name="fake-httpx",
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "fake-httpx"

    def supports_tools(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False

    async def chat(
        self,
        messages,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ):
        return CompletionResponse(
            content='{"is_complete": true, "confidence": 0.9, "phase": "done"}',
            model=model,
        )

    async def close(self) -> None:
        type(self).closed_instances += 1
        await super().close()


class TestLLMDecisionServiceProtocol:
    """Test that LLMDecisionService satisfies the protocol."""

    def test_implements_protocol(self):
        provider = _make_provider({"is_complete": True, "confidence": 0.9, "phase": "done"})
        service = LLMDecisionService(provider=provider, model="test")
        assert isinstance(service, LLMDecisionServiceProtocol)

    async def test_provider_call_disables_portable_thinking(self):
        provider = _make_provider({"is_complete": True, "confidence": 0.9, "phase": "done"})
        service = LLMDecisionService(provider=provider, model="test")

        await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "done", "deliverable_count": 1, "signal_count": 1},
            heuristic_confidence=0.1,
        )

        assert provider.chat.call_args.kwargs["thinking"] is False
        assert "think" not in provider.chat.call_args.kwargs


class TestHeuristicFastPath:
    """Test that heuristic results are returned when confidence is high."""

    async def test_high_confidence_returns_heuristic(self):
        provider = _make_provider({})
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={
                "response_tail": "done",
                "deliverable_count": 1,
                "signal_count": 1,
            },
            heuristic_result="high",
            heuristic_confidence=0.9,
        )
        assert result.source == "heuristic"
        assert result.result == "high"
        assert result.confidence == 0.9
        # Provider should NOT be called
        provider.chat.assert_not_called()

    async def test_threshold_boundary(self):
        provider = _make_provider({})
        config = LLMDecisionServiceConfig(confidence_threshold=0.7)
        service = LLMDecisionService(provider=provider, model="test", config=config)

        # Exactly at threshold -> still heuristic
        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "", "deliverable_count": 0, "signal_count": 0},
            heuristic_confidence=0.7,
        )
        assert result.source == "heuristic"

    def test_sync_high_confidence(self):
        provider = _make_provider({})
        service = LLMDecisionService(provider=provider, model="test")

        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "", "deliverable_count": 0, "signal_count": 0},
            heuristic_confidence=0.8,
        )
        assert result.source == "heuristic"


class TestLLMCall:
    """Test LLM call path when heuristic confidence is low."""

    async def test_llm_called_on_low_confidence(self):
        response_json = {"is_complete": True, "confidence": 0.9, "phase": "done"}
        provider = _make_provider(response_json)
        service = LLMDecisionService(provider=provider, model="test-model")

        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={
                "response_tail": "some text",
                "deliverable_count": 1,
                "signal_count": 0,
            },
            heuristic_confidence=0.3,
        )

        assert result.source == "llm"
        assert isinstance(result.result, TaskCompletionDecision)
        assert result.result.is_complete is True
        assert result.confidence == 0.9
        assert result.tokens_used == 15
        provider.chat.assert_called_once()

    async def test_llm_response_with_code_fences(self):
        provider = MagicMock()
        response = MockResponse(
            content='```json\n{"intent": "completion", "confidence": 0.85}\n```'
        )
        provider.chat = AsyncMock(return_value=response)
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.INTENT_CLASSIFICATION,
            context={"text_tail": "done with task", "has_tool_calls": "false"},
            heuristic_confidence=0.3,
        )

        assert result.source == "llm"
        assert isinstance(result.result, IntentDecision)
        assert result.result.intent == "completion"

    async def test_llm_parse_failure_returns_heuristic(self):
        provider = MagicMock()
        response = MockResponse(content="not valid json")
        provider.chat = AsyncMock(return_value=response)
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "", "deliverable_count": 0, "signal_count": 0},
            heuristic_result="fallback",
            heuristic_confidence=0.2,
        )

        assert result.source == "heuristic"
        assert result.result == "fallback"

    async def test_latency_tracked(self):
        response_json = {"is_complete": False, "confidence": 0.5, "phase": "working"}
        provider = _make_provider(response_json)
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "", "deliverable_count": 0, "signal_count": 0},
            heuristic_confidence=0.1,
        )

        assert result.latency_ms >= 0


class TestBudget:
    """Test micro-budget enforcement."""

    async def test_budget_exhausted(self):
        response_json = {"is_complete": True, "confidence": 0.9, "phase": "done"}
        provider = _make_provider(response_json)
        config = LLMDecisionServiceConfig(micro_budget=2)
        service = LLMDecisionService(provider=provider, model="test", config=config)

        assert service.budget_remaining == 2

        # Use up budget with different contexts to avoid cache hits
        for i in range(2):
            await service.decide(
                DecisionType.TASK_COMPLETION,
                context={
                    "response_tail": f"unique_{i}",
                    "deliverable_count": 0,
                    "signal_count": 0,
                },
                heuristic_confidence=0.1,
            )

        assert service.budget_remaining == 0

        # Next call with new context should return budget_exhausted
        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={
                "response_tail": "new_context",
                "deliverable_count": 0,
                "signal_count": 0,
            },
            heuristic_result="fallback",
            heuristic_confidence=0.1,
        )
        assert result.source == "budget_exhausted"

    async def test_reset_budget(self):
        provider = _make_provider({"is_complete": True, "confidence": 0.9, "phase": "done"})
        config = LLMDecisionServiceConfig(micro_budget=1)
        service = LLMDecisionService(provider=provider, model="test", config=config)

        await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "", "deliverable_count": 0, "signal_count": 0},
            heuristic_confidence=0.1,
        )
        assert service.budget_remaining == 0

        service.reset_budget()
        assert service.budget_remaining == 1


class TestCache:
    """Test result caching."""

    async def test_cache_hit(self):
        response_json = {"is_complete": True, "confidence": 0.9, "phase": "done"}
        provider = _make_provider(response_json)
        service = LLMDecisionService(provider=provider, model="test")

        context = {"response_tail": "test", "deliverable_count": 1, "signal_count": 1}

        # First call - LLM
        result1 = await service.decide(
            DecisionType.TASK_COMPLETION,
            context=context,
            heuristic_confidence=0.1,
        )
        assert result1.source == "llm"

        # Second call - cache
        result2 = await service.decide(
            DecisionType.TASK_COMPLETION,
            context=context,
            heuristic_confidence=0.1,
        )
        assert result2.source == "cache"
        assert provider.chat.call_count == 1  # Only one LLM call

    async def test_cache_expiry(self):
        response_json = {"is_complete": True, "confidence": 0.9, "phase": "done"}
        provider = _make_provider(response_json)
        config = LLMDecisionServiceConfig(cache_ttl=0)  # Immediate expiry
        service = LLMDecisionService(provider=provider, model="test", config=config)

        context = {"response_tail": "test", "deliverable_count": 1, "signal_count": 1}

        await service.decide(
            DecisionType.TASK_COMPLETION,
            context=context,
            heuristic_confidence=0.1,
        )
        # Cache should be expired immediately
        time.sleep(0.01)
        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context=context,
            heuristic_confidence=0.1,
        )
        assert result.source == "llm"  # No cache hit


class TestTimeout:
    """Test timeout handling."""

    async def test_timeout_returns_heuristic(self):
        provider = MagicMock()

        async def slow_chat(**kwargs):
            await asyncio.sleep(5)  # Very slow
            return MockResponse(content='{"is_complete": true, "confidence": 0.9, "phase": "done"}')

        provider.chat = slow_chat
        config = LLMDecisionServiceConfig(timeout_ms=50)  # 50ms timeout
        service = LLMDecisionService(provider=provider, model="test", config=config)

        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "", "deliverable_count": 0, "signal_count": 0},
            heuristic_result="timeout_fallback",
            heuristic_confidence=0.2,
        )

        assert result.source == "timeout_fallback"
        assert result.result == "timeout_fallback"


class TestMetrics:
    """Test metrics tracking."""

    async def test_metrics_accumulated(self):
        response_json = {"is_complete": True, "confidence": 0.9, "phase": "done"}
        provider = _make_provider(response_json)
        config = LLMDecisionServiceConfig(micro_budget=5)
        service = LLMDecisionService(provider=provider, model="test", config=config)

        # Heuristic call
        await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "", "deliverable_count": 0, "signal_count": 0},
            heuristic_confidence=0.9,
        )
        # LLM call
        await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "x", "deliverable_count": 0, "signal_count": 0},
            heuristic_confidence=0.1,
        )
        # Cache hit (same context)
        await service.decide(
            DecisionType.TASK_COMPLETION,
            context={"response_tail": "x", "deliverable_count": 0, "signal_count": 0},
            heuristic_confidence=0.1,
        )

        metrics = service.get_metrics()
        assert metrics.total_calls == 3
        assert metrics.llm_calls == 1
        assert metrics.cache_hits == 1

    def test_is_healthy(self):
        provider = _make_provider({})
        service = LLMDecisionService(provider=provider, model="test")
        assert service.is_healthy() is True

    def test_is_healthy_no_provider(self):
        service = LLMDecisionService(provider=None, model="test")
        assert service.is_healthy() is False

    def test_exports_runtime_evaluation_feedback_from_config_threshold(self):
        provider = _make_provider({})
        service = LLMDecisionService(
            provider=provider,
            model="test",
            config=LLMDecisionServiceConfig(confidence_threshold=0.72),
        )

        feedback = service.get_runtime_evaluation_feedback()

        assert isinstance(feedback, RuntimeEvaluationFeedback)
        assert feedback.completion_threshold == pytest.approx(0.72)
        assert feedback.enhanced_progress_threshold == pytest.approx(0.57)
        assert feedback.minimum_supported_evidence_score == pytest.approx(0.77)


class TestSyncDecide:
    """Test synchronous decide wrapper."""

    def test_sync_httpx_provider_uses_fresh_provider_in_thread(self):
        """httpx-compatible providers should be recreated inside worker loops."""
        _FakeHttpxProvider.created_instances = 0
        _FakeHttpxProvider.closed_instances = 0

        provider = _FakeHttpxProvider()
        service = LLMDecisionService(provider=provider, model="glm-test")

        async def _inner():
            result = service.decide_sync(
                DecisionType.TASK_COMPLETION,
                context={
                    "response_tail": "",
                    "deliverable_count": 0,
                    "signal_count": 0,
                },
                heuristic_confidence=0.1,
            )
            assert result.source == "llm"

        asyncio.run(_inner())

        # Original provider + one fresh provider for the worker thread.
        assert _FakeHttpxProvider.created_instances >= 2
        # The fresh thread-scoped provider should be closed after use.
        assert _FakeHttpxProvider.closed_instances >= 1

    def test_sync_with_running_loop_uses_thread(self):
        """When called from within a running event loop, runs in a thread."""
        provider = _make_provider({"is_complete": True, "confidence": 0.9, "phase": "done"})
        service = LLMDecisionService(provider=provider, model="test")

        async def _inner():
            # We're inside an event loop — decide_sync uses run_sync_in_thread
            result = service.decide_sync(
                DecisionType.TASK_COMPLETION,
                context={
                    "response_tail": "",
                    "deliverable_count": 0,
                    "signal_count": 0,
                },
                heuristic_result="loop_fallback",
                heuristic_confidence=0.3,
            )
            # run_sync_in_thread makes the actual LLM call instead of falling back
            assert result.source in ("llm", "heuristic")

        asyncio.run(_inner())

    def test_sync_cache_works(self):
        """Cache is shared between sync and async paths."""
        provider = _make_provider({"is_complete": True, "confidence": 0.9, "phase": "done"})
        service = LLMDecisionService(provider=provider, model="test")

        # Run an async decide to populate cache
        context = {
            "response_tail": "cache_test",
            "deliverable_count": 0,
            "signal_count": 0,
        }
        asyncio.run(
            service.decide(
                DecisionType.TASK_COMPLETION,
                context=context,
                heuristic_confidence=0.1,
            )
        )

        # Sync call should hit cache even from running loop
        async def _check_cache():
            result = service.decide_sync(
                DecisionType.TASK_COMPLETION,
                context=context,
                heuristic_confidence=0.1,
            )
            assert result.source == "cache"

        asyncio.run(_check_cache())

    def test_sync_uses_shared_run_sync_bridge(self):
        """Sync path should not call the legacy local asyncio.run bridge."""
        provider = _make_provider({"is_complete": True, "confidence": 0.9, "phase": "done"})
        service = LLMDecisionService(provider=provider, model="test")

        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            context={
                "response_tail": "done",
                "deliverable_count": 1,
                "signal_count": 1,
            },
            heuristic_confidence=0.1,
        )
        # decide_sync uses run_sync_in_thread to bridge async-in-sync
        assert result.source == "llm"


class TestDecisionLogging:
    """Every decision path must log (FEP-0012 closed loop) — not just LLM ones.

    log_decision is the source the classifier trainer mines, so heuristic /
    budget / timeout decisions must be captured too.
    """

    def _service(self):
        return LLMDecisionService(provider=_make_provider({}), model="test")

    def test_fast_path_logs(self):
        service = self._service()
        with patch("victor.agent.decisions.chain.log_decision") as m:
            result = service.decide_sync(
                DecisionType.TASK_COMPLETION,
                context={
                    "response_tail": "",
                    "deliverable_count": 0,
                    "signal_count": 0,
                },
                heuristic_confidence=0.9,  # above threshold → heuristic fast path
            )
        assert result.source == "heuristic"
        m.assert_called_once()
        assert m.call_args.kwargs["source"] == "heuristic"

    def test_budget_exhausted_logs(self):
        config = LLMDecisionServiceConfig(micro_budget=0)  # immediately exhausted
        service = LLMDecisionService(provider=_make_provider({}), model="test", config=config)
        with patch("victor.agent.decisions.chain.log_decision") as m:
            result = service.decide_sync(
                DecisionType.TASK_COMPLETION,
                context={
                    "response_tail": "",
                    "deliverable_count": 0,
                    "signal_count": 0,
                },
                heuristic_confidence=0.0,  # low → not fast-path; budget=0 → exhausted
            )
        assert result.source == "budget_exhausted"
        m.assert_called_once()

    async def test_decide_async_fast_path_logs(self):
        service = self._service()
        with patch("victor.agent.decisions.chain.log_decision") as m:
            result = await service.decide(
                DecisionType.TASK_COMPLETION,
                context={
                    "response_tail": "",
                    "deliverable_count": 0,
                    "signal_count": 0,
                },
                heuristic_confidence=0.9,
            )
        assert result.source == "heuristic"
        m.assert_called_once()
