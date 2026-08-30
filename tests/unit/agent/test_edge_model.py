# Copyright 2025 Vijaykumar Singh
# Licensed under the Apache License, Version 2.0

"""Tests for edge model provider and integrations.

Tests cover:
- EdgeModelConfig defaults and validation
- EdgeModelProvider creation with/without Ollama
- Tool selection via edge model
- Prompt section selection via edge model
- Integration with LLMDecisionService
- Fallback behavior when edge model unavailable
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.edge_model import (
    EdgeModelConfig,
    PROMPT_FOCUS_PROMPT,
    create_edge_decision_service,
    select_prompt_sections_with_edge_model,
    select_tools_with_edge_model,
)


class TestEdgeModelConfig:
    """Test EdgeModelConfig defaults and validation."""

    def test_defaults(self):
        config = EdgeModelConfig()
        assert config.enabled is True
        assert config.provider == "ollama"
        assert config.model == "qwen3.5:2b"
        assert config.timeout_ms == 4000
        assert config.max_tokens == 50
        assert config.micro_budget == 20
        assert config.max_tools == 6

    def test_disabled(self):
        config = EdgeModelConfig(enabled=False)
        assert config.enabled is False

    def test_custom_model(self):
        config = EdgeModelConfig(model="tinyllama")
        assert config.model == "tinyllama"


class TestCreateEdgeDecisionService:
    """Test edge decision service creation."""

    def test_disabled_returns_none(self):
        config = EdgeModelConfig(enabled=False)
        assert create_edge_decision_service(config) is None

    def test_provider_disabled_returns_none(self):
        config = EdgeModelConfig(provider="disabled")
        assert create_edge_decision_service(config) is None

    @patch("victor.agent.edge_model._check_model_available", return_value=True)
    @patch("victor.providers.registry.ProviderRegistry.create")
    def test_creates_service_when_available(self, mock_create, mock_check):
        mock_provider = MagicMock()
        mock_create.return_value = mock_provider

        service = create_edge_decision_service(EdgeModelConfig())
        assert service is not None
        assert service._config.micro_budget == 20

    @patch("victor.agent.edge_model._check_model_available", return_value=False)
    def test_returns_none_when_model_missing(self, mock_check):
        service = create_edge_decision_service(EdgeModelConfig())
        assert service is None


class TestToolSelection:
    """Test edge model tool selection via decide_sync."""

    def _make_mock_service(self, tools: list, confidence: float = 0.9):
        """Create a mock service that returns tool selection via decide_sync."""
        from victor.agent.decisions.schemas import DecisionType, ToolSelectionDecision
        from victor.agent.services.protocols.decision_service import DecisionResult

        result = ToolSelectionDecision(tools=tools, confidence=confidence)
        service = MagicMock()
        service.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.TOOL_SELECTION,
            result=result,
            source="llm",
            confidence=confidence,
            latency_ms=50.0,
            tokens_used=30,
        )
        return service

    def test_selects_relevant_tools(self):
        service = self._make_mock_service(["read", "edit", "grep", "ls", "shell"])
        available = [
            "read",
            "write",
            "edit",
            "grep",
            "ls",
            "shell",
            "git",
            "overview",
            "find",
            "symbol",
            "refs",
            "jira",
            "scan",
        ]
        result = select_tools_with_edge_model(service, "Fix the auth bug", available, "execution")
        assert result is not None
        assert "read" in result
        assert "edit" in result
        assert len(result) <= 6

    def test_filters_invalid_tool_names(self):
        service = self._make_mock_service(["read", "nonexistent_tool", "edit"])
        available = ["read", "write", "edit", "grep"]
        result = select_tools_with_edge_model(service, "test", available)
        assert result is not None
        assert "nonexistent_tool" not in result
        assert "read" in result

    def test_returns_none_on_error(self):
        service = MagicMock()
        service.decide_sync.side_effect = Exception("connection refused")
        result = select_tools_with_edge_model(service, "test", ["read", "edit"])
        assert result is None

    def test_returns_none_for_heuristic_fallback(self):
        """When decide_sync returns heuristic (inside async), return None."""
        from victor.agent.decisions.schemas import DecisionType
        from victor.agent.services.protocols.decision_service import DecisionResult

        service = MagicMock()
        service.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.TOOL_SELECTION,
            result=None,
            source="heuristic",
            confidence=0.0,
        )
        result = select_tools_with_edge_model(service, "test", ["read", "edit"])
        assert result is None


class TestPromptSectionSelection:
    """Test edge model prompt section selection via decide_sync."""

    def _make_mock_service(self, sections: list, confidence: float = 0.88):
        from victor.agent.decisions.schemas import DecisionType, PromptFocusDecision
        from victor.agent.services.protocols.decision_service import DecisionResult

        result = PromptFocusDecision(sections=sections, confidence=confidence)
        service = MagicMock()
        service.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.PROMPT_FOCUS,
            result=result,
            source="llm",
            confidence=confidence,
            latency_ms=50.0,
            tokens_used=20,
        )
        return service

    def test_selects_sections_for_fix_task(self):
        service = self._make_mock_service(["completion", "tool_guidance"])
        available = [
            "concise_mode",
            "task_guidance",
            "tool_constraint",
            "completion",
            "tool_guidance",
        ]
        result = select_prompt_sections_with_edge_model(
            service, "Fix the auth bug", "action", available
        )
        assert result is not None
        assert "completion" in result

    def test_selects_sections_for_analysis(self):
        service = self._make_mock_service(["concise_mode", "completion"])
        available = [
            "concise_mode",
            "task_guidance",
            "tool_constraint",
            "completion",
            "tool_guidance",
        ]
        result = select_prompt_sections_with_edge_model(
            service, "Analyze the codebase", "analysis", available
        )
        assert result is not None
        assert len(result) <= len(available)

    def test_returns_none_on_error(self):
        service = MagicMock()
        service.decide_sync.side_effect = Exception("timeout")
        result = select_prompt_sections_with_edge_model(service, "test", "action", ["completion"])
        assert result is None

    def test_prompt_focus_prompt_uses_canonical_catalog(self):
        assert '"file_pagination"' in PROMPT_FOCUS_PROMPT
        assert '"parallel_read"' in PROMPT_FOCUS_PROMPT
        assert '"grounding"' in PROMPT_FOCUS_PROMPT


class TestBootstrapIntegration:
    """Test edge model bootstrap wiring."""

    def test_feature_flag_exists(self):
        from victor.core.feature_flags import FeatureFlag

        assert hasattr(FeatureFlag, "USE_EDGE_MODEL")
        assert FeatureFlag.USE_EDGE_MODEL.value == "use_edge_model"

    def test_decision_type_has_task_type(self):
        from victor.agent.decisions.schemas import DecisionType

        assert hasattr(DecisionType, "TASK_TYPE_CLASSIFICATION")

    def test_task_type_decision_has_deliverables(self):
        from victor.agent.decisions.schemas import TaskTypeDecision

        d = TaskTypeDecision(
            task_type="action",
            confidence=0.9,
            deliverables=["file_modified"],
        )
        assert d.deliverables == ["file_modified"]

    def test_prompt_template_exists(self):
        from victor.agent.decisions.prompts import DECISION_PROMPTS
        from victor.agent.decisions.schemas import DecisionType

        assert DecisionType.TASK_TYPE_CLASSIFICATION in DECISION_PROMPTS
        prompt = DECISION_PROMPTS[DecisionType.TASK_TYPE_CLASSIFICATION]
        assert "deliverables" in prompt.user_template


class TestStageDetectionEdge:
    """Test edge model integration in conversation stage detection."""

    def test_edge_stage_detection_returns_none_gracefully(self):
        """Edge stage detection returns None when service container unavailable."""
        from victor.agent.conversation.state_machine import ConversationStateMachine

        machine = ConversationStateMachine()
        # Should return (None, 0.0) gracefully (no service container in test)
        stage, confidence = machine._detect_stage_with_edge_model("Fix the auth bug")
        assert stage is None
        assert confidence == 0.0

    def test_edge_stage_detection_prefers_runtime_intelligence(self):
        """Stage detection should use the canonical runtime-intelligence seam first."""
        from victor.agent.conversation.state_machine import ConversationStateMachine
        from victor.core.shared_types import ConversationStage

        runtime_intelligence = MagicMock()
        runtime_intelligence.decide_sync.return_value = MagicMock(
            source="llm",
            result=MagicMock(stage="execution"),
            confidence=0.92,
        )
        machine = ConversationStateMachine(runtime_intelligence=runtime_intelligence)

        with patch(
            "victor.core.service_resolution.get_container",
            side_effect=AssertionError("container path should not be used"),
        ):
            stage, confidence = machine._detect_stage_with_edge_model("Fix the auth bug")

        assert stage == ConversationStage.EXECUTION
        assert confidence == 0.92
        runtime_intelligence.decide_sync.assert_called_once()

    def test_keyword_detection_still_works(self):
        from victor.agent.conversation.state_machine import ConversationStateMachine

        machine = ConversationStateMachine()
        # Simulate files already observed (post-exploration)
        machine.state.message_count = 2
        machine.state.observed_files = {"auth.py"}
        # "fix" (0.5) + "change" (0.5) + "modify" would be needed for ≥2
        # Use strong keywords to ensure EXECUTION
        result = machine._detect_stage_from_content("modify and implement the auth module")
        from victor.core.shared_types import ConversationStage

        assert result == ConversationStage.EXECUTION


class TestComplexityEdge:
    """Test edge model integration in task complexity classification."""

    def test_classify_with_edge_model_returns_none_without_service(self):
        from victor.framework.task.complexity import TaskComplexityService

        service = TaskComplexityService(use_semantic=False)
        result = service._classify_with_edge_model("Fix the auth bug")
        # Should return None gracefully when no service container
        assert result is None

    def test_classify_still_works_without_edge(self):
        from victor.framework.task.complexity import TaskComplexityService

        service = TaskComplexityService(use_semantic=False)
        result = service.classify("fix the auth bug")
        # Should still classify via regex patterns
        assert result is not None
        assert result.confidence > 0


class TestEdgeFallbackBehavior:
    """Test that everything works when edge model is unavailable."""

    def test_tool_selection_returns_none_without_service(self):
        result = select_tools_with_edge_model(None, "test", ["read", "edit"])  # type: ignore
        assert result is None

    def test_prompt_selection_returns_none_without_service(self):
        result = select_prompt_sections_with_edge_model(
            None, "test", "action", ["completion"]  # type: ignore
        )
        assert result is None

    def test_completion_detector_works_without_edge(self):
        """Completion detector should work with regex fallback."""
        from victor.agent.task_completion import (
            DeliverableType,
            TaskCompletionDetector,
        )

        detector = TaskCompletionDetector(decision_service=None)
        result = detector.analyze_intent("fix the auth bug")
        assert DeliverableType.FILE_MODIFIED in result
