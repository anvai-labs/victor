"""End-to-end integration tests for LLM Decision Service.

These tests verify the full decision service flow with a real (or mocked)
provider. They exercise the complete pipeline: prompt building, provider call,
JSON parsing, caching, and budget management.

Note: Tests using real providers (Ollama) are marked with @pytest.mark.slow
and require a running Ollama instance.
"""

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock


from victor.agent.decisions.prompts import DECISION_PROMPTS
from victor.agent.decisions.schemas import (
    ContinuationDecision,
    DecisionType,
    ErrorClassDecision,
    IntentDecision,
    LoopDetection,
    QuestionTypeDecision,
    TaskCompletionDecision,
    TaskTypeDecision,
)
from victor.agent.services.decision_service import (
    LLMDecisionService,
    LLMDecisionServiceConfig,
)
from victor.agent.task_completion import TaskCompletionDetector
from victor.providers.base import Message


@dataclass
class MockUsage:
    total_tokens: int = 10


@dataclass
class MockResponse:
    content: str = ""
    usage: MockUsage = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = MockUsage()


def _provider_returning(response_json: dict) -> MagicMock:
    """Create a mock provider that returns a specific JSON response."""
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=MockResponse(content=json.dumps(response_json)))
    return provider


class TestAllDecisionTypes:
    """Verify each decision type can be called end-to-end."""

    async def test_provider_receives_standard_message_objects(self):
        provider = _provider_returning({"task_type": "analysis", "confidence": 0.9})
        service = LLMDecisionService(provider=provider, model="test")

        await service.decide(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message_excerpt": "Review the graph module"},
            heuristic_confidence=0.3,
        )

        provider.chat.assert_awaited_once()
        call_kwargs = provider.chat.await_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert all(isinstance(message, Message) for message in messages)
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    async def test_task_completion_e2e(self):
        provider = _provider_returning({"is_complete": True, "confidence": 0.95, "phase": "done"})
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={
                "response_tail": "I've completed all requested changes.",
                "deliverable_count": 3,
                "signal_count": 2,
            },
            heuristic_confidence=0.3,
        )

        assert result.source == "llm"
        assert isinstance(result.result, TaskCompletionDecision)
        assert result.result.is_complete is True
        assert result.result.phase == "done"

    async def test_intent_classification_e2e(self):
        provider = _provider_returning({"intent": "continuation", "confidence": 0.85})
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.INTENT_CLASSIFICATION,
            context={
                "text_tail": "Let me check the implementation...",
                "has_tool_calls": "true",
            },
            heuristic_confidence=0.2,
        )

        assert result.source == "llm"
        assert isinstance(result.result, IntentDecision)
        assert result.result.intent == "continuation"

    async def test_task_type_classification_e2e(self):
        provider = _provider_returning({"task_type": "analysis", "confidence": 0.9})
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message_excerpt": "Review the authentication module for security issues"},
            heuristic_confidence=0.4,
        )

        assert isinstance(result.result, TaskTypeDecision)
        assert result.result.task_type == "analysis"

    async def test_question_classification_e2e(self):
        provider = _provider_returning({"question_type": "rhetorical", "confidence": 0.8})
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.QUESTION_CLASSIFICATION,
            context={"question_text": "Should I continue with the implementation?"},
            heuristic_confidence=0.3,
        )

        assert isinstance(result.result, QuestionTypeDecision)
        assert result.result.question_type == "rhetorical"

    async def test_loop_detection_e2e(self):
        provider = _provider_returning({"is_loop": True, "loop_type": "stalling"})
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.LOOP_DETECTION,
            context={
                "content_excerpt": "I need to check the file... Let me examine...",
                "recent_blocks": "5",
            },
            heuristic_confidence=0.4,
        )

        assert isinstance(result.result, LoopDetection)
        assert result.result.is_loop is True
        assert result.result.loop_type == "stalling"

    async def test_error_classification_e2e(self):
        provider = _provider_returning({"error_type": "transient", "confidence": 0.85})
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.ERROR_CLASSIFICATION,
            context={"error_message": "Connection reset by peer"},
            heuristic_confidence=0.3,
        )

        assert isinstance(result.result, ErrorClassDecision)
        assert result.result.error_type == "transient"

    async def test_continuation_action_e2e(self):
        provider = _provider_returning(
            {"action": "prompt_tool_call", "reason": "Task needs more exploration"}
        )
        service = LLMDecisionService(provider=provider, model="test")

        result = await service.decide(
            DecisionType.CONTINUATION_ACTION,
            context={
                "response_excerpt": "I've analyzed the first file but need to check more.",
                "continuation_prompts": "2",
                "task_type": "analysis",
            },
            heuristic_confidence=0.4,
        )

        assert isinstance(result.result, ContinuationDecision)
        assert result.result.action == "prompt_tool_call"


class TestPromptTemplateIntegrity:
    """Verify all prompt templates have matching context keys."""

    def test_all_decision_types_have_prompts(self):
        for dt in DecisionType:
            assert dt in DECISION_PROMPTS, f"Missing prompt for {dt}"

    def test_task_completion_template_keys(self):
        prompt = DECISION_PROMPTS[DecisionType.TASK_COMPLETION]
        # Should format without error with required keys
        result = prompt.user_template.format(
            response_tail="test",
            deliverable_count=0,
            signal_count=0,
        )
        assert "test" in result

    def test_intent_template_keys(self):
        prompt = DECISION_PROMPTS[DecisionType.INTENT_CLASSIFICATION]
        result = prompt.user_template.format(
            text_tail="test",
            has_tool_calls="false",
        )
        assert "test" in result

    def test_continuation_template_keys(self):
        prompt = DECISION_PROMPTS[DecisionType.CONTINUATION_ACTION]
        result = prompt.user_template.format(
            response_excerpt="test",
            continuation_prompts="3",
            task_type="analysis",
        )
        assert "test" in result


class TestFullPipelineWithTaskCompletion:
    """Test the full pipeline from TaskCompletionDetector through LLMDecisionService."""

    def test_detector_with_service_full_flow(self):
        """Simulate a complete flow: ambiguous response -> LLM augments -> completion detected."""
        provider = _provider_returning({"is_complete": True, "confidence": 0.9, "phase": "done"})
        service = LLMDecisionService(provider=provider, model="test")
        detector = TaskCompletionDetector(decision_service=service)

        # Simulate intent analysis
        detector.analyze_intent("Fix the authentication bug in login.py")

        # Simulate tool recording
        detector.record_tool_result("edit_file", {"success": True, "path": "login.py"})

        # Simulate ambiguous response (no clear active signal)
        detector.analyze_response(
            "I've made the necessary changes to the authentication flow "
            "in login.py to handle the edge case properly."
        )

        # LLM should have been consulted and added completion signal
        assert "llm:task_complete" in detector._state.completion_signals

    async def test_service_budget_across_multiple_detectors(self):
        """Budget is shared when the same service instance is used."""
        provider = _provider_returning({"is_complete": True, "confidence": 0.9, "phase": "done"})
        config = LLMDecisionServiceConfig(micro_budget=2)
        service = LLMDecisionService(provider=provider, model="test", config=config)

        # Use direct service calls with unique contexts to consume budget
        await service.decide(
            DecisionType.TASK_COMPLETION,
            context={
                "response_tail": "unique_a",
                "deliverable_count": 0,
                "signal_count": 0,
            },
            heuristic_confidence=0.1,
        )
        await service.decide(
            DecisionType.TASK_COMPLETION,
            context={
                "response_tail": "unique_b",
                "deliverable_count": 0,
                "signal_count": 0,
            },
            heuristic_confidence=0.1,
        )
        assert service.budget_remaining == 0

        # Next call with new context should hit budget exhaustion
        result = await service.decide(
            DecisionType.TASK_COMPLETION,
            context={
                "response_tail": "unique_c",
                "deliverable_count": 0,
                "signal_count": 0,
            },
            heuristic_confidence=0.1,
        )
        assert result.source == "budget_exhausted"
        assert service.get_metrics().budget_exhaustions >= 1

    def test_service_reset_between_turns(self):
        """Verify budget resets work correctly between conversation turns."""
        provider = _provider_returning(
            {"is_complete": False, "confidence": 0.6, "phase": "working"}
        )
        config = LLMDecisionServiceConfig(micro_budget=2)
        service = LLMDecisionService(provider=provider, model="test", config=config)

        # Exhaust budget
        asyncio.run(
            service.decide(
                DecisionType.TASK_COMPLETION,
                context={
                    "response_tail": "a",
                    "deliverable_count": 0,
                    "signal_count": 0,
                },
                heuristic_confidence=0.1,
            )
        )
        asyncio.run(
            service.decide(
                DecisionType.TASK_COMPLETION,
                context={
                    "response_tail": "b",
                    "deliverable_count": 0,
                    "signal_count": 0,
                },
                heuristic_confidence=0.1,
            )
        )
        assert service.budget_remaining == 0

        # Simulate new turn
        service.reset_budget()
        assert service.budget_remaining == 2
