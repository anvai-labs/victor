"""Tests for victor.framework.perception_integration module.

Tests the PerceptionIntegration layer that combines existing Victor
components (ActionIntent, TaskAnalyzer) with new requirement extraction.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from victor.agent.action_authorizer import ActionIntent
from victor.framework.perception_integration import (
    Perception,
    PerceptionIntegration,
    Requirement,
    RequirementType,
    SimilarExperience,
    perceive,
)
from victor.framework.task.protocols import TaskComplexity

# ============================================================================
# Perception dataclass tests
# ============================================================================


class TestPerception:
    """Tests for Perception dataclass."""

    def test_to_dict_basic(self):
        from victor.agent.task_analyzer import TaskAnalysis

        p = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(spec=TaskAnalysis, task_type="code_generation"),
            confidence=0.75,
        )
        d = p.to_dict()
        assert d["intent"] == "write_allowed"
        assert d["complexity"] == "medium"
        assert d["confidence"] == 0.75
        assert d["similar_experiences_count"] == 0
        assert d["requirements"] == []
        assert d["needs_clarification"] is False
        assert d["clarification_prompt"] is None

    def test_to_dict_with_requirements(self):
        p = Perception(
            intent=ActionIntent.READ_ONLY,
            complexity=TaskComplexity.SIMPLE,
            task_analysis=MagicMock(task_type="analysis"),
            requirements=[
                Requirement(
                    type=RequirementType.FUNCTIONAL,
                    description="Must handle errors",
                    priority=4,
                )
            ],
            confidence=0.6,
        )
        d = p.to_dict()
        assert len(d["requirements"]) == 1
        assert d["requirements"][0]["type"] == "functional"
        assert d["requirements"][0]["description"] == "Must handle errors"

    def test_task_type_property(self):
        p = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.COMPLEX,
            task_analysis=MagicMock(task_type="debugging"),
        )
        assert p.task_type == "write_allowed"

    def test_primary_intent_from_task_analysis(self):
        p = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.COMPLEX,
            task_analysis=MagicMock(task_type="debugging"),
        )
        assert p.primary_intent == "debugging"

    def test_primary_intent_fallback_to_intent(self):
        p = Perception(
            intent=ActionIntent.DISPLAY_ONLY,
            complexity=TaskComplexity.SIMPLE,
            task_analysis=MagicMock(task_type=None),
        )
        assert p.primary_intent == "display_only"

    def test_primary_intent_no_analysis(self):
        p = Perception(
            intent=ActionIntent.AMBIGUOUS,
            complexity=TaskComplexity.SIMPLE,
            task_analysis=None,
        )
        assert p.primary_intent == ""


# ============================================================================
# Requirement extraction tests
# ============================================================================


class TestRequirementExtraction:
    """Tests for requirement extraction from queries."""

    def setup_method(self):
        self.integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )

    def test_extract_must_requirement(self):
        reqs = self.integration._extract_requirements(
            "The function must handle edge cases properly.",
            None,
        )
        assert len(reqs) >= 1
        assert any(r.type == RequirementType.FUNCTIONAL for r in reqs)
        assert any("must" in r.description.lower() for r in reqs)

    def test_extract_should_requirement(self):
        reqs = self.integration._extract_requirements(
            "The API should return JSON.",
            None,
        )
        assert len(reqs) >= 1

    def test_extract_quality_keywords(self):
        reqs = self.integration._extract_requirements(
            "Make it fast and secure.",
            None,
        )
        quality_reqs = [r for r in reqs if r.type == RequirementType.QUALITY]
        assert len(quality_reqs) >= 2
        descs = [r.description for r in quality_reqs]
        assert "Optimize for performance" in descs
        assert "Follow security best practices" in descs

    def test_extract_test_quality(self):
        reqs = self.integration._extract_requirements(
            "Add a test for the login function.",
            None,
        )
        quality_reqs = [r for r in reqs if r.type == RequirementType.QUALITY]
        assert any(
            "tests" in r.description.lower() or "test" in r.description.lower()
            for r in quality_reqs
        )

    def test_no_requirements_in_simple_query(self):
        reqs = self.integration._extract_requirements(
            "What is the weather?",
            None,
        )
        assert len(reqs) == 0

    def test_disabled_requirement_extraction(self):
        integration = PerceptionIntegration(
            enable_requirement_extraction=False,
            enable_similarity_search=False,
        )
        # _extract_requirements is only called when enabled, but we test
        # the perceive method respects the flag
        # We'll test this in the perceive tests


# ============================================================================
# Confidence calculation tests
# ============================================================================


class TestConfidenceCalculation:
    """Tests for calibrated confidence calculation (geometric mean)."""

    def setup_method(self):
        self.integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )

    def test_ambiguous_intent_low_confidence(self):
        conf = self.integration._calculate_confidence(
            intent=ActionIntent.AMBIGUOUS,
            complexity=TaskComplexity.SIMPLE,
            requirements=[],
            similar_experiences=[],
        )
        # Geometric mean of [0.3 (ambiguous), 0.8 (simple complexity)]
        assert 0.3 < conf < 0.7

    def test_clear_intent_higher_confidence(self):
        conf = self.integration._calculate_confidence(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.SIMPLE,
            requirements=[],
            similar_experiences=[],
            intent_confidence=0.9,
        )
        # Should be higher than ambiguous
        assert conf > 0.7

    def test_complex_task_slightly_lower(self):
        conf = self.integration._calculate_confidence(
            intent=ActionIntent.AMBIGUOUS,
            complexity=TaskComplexity.COMPLEX,
            requirements=[],
            similar_experiences=[],
        )
        # Complex tasks have lower complexity_signal (0.6)
        assert 0.3 < conf < 0.7

    def test_requirements_add_signal(self):
        conf_without = self.integration._calculate_confidence(
            intent=ActionIntent.AMBIGUOUS,
            complexity=TaskComplexity.SIMPLE,
            requirements=[],
            similar_experiences=[],
        )
        conf_with = self.integration._calculate_confidence(
            intent=ActionIntent.AMBIGUOUS,
            complexity=TaskComplexity.SIMPLE,
            requirements=[Requirement(type=RequirementType.FUNCTIONAL, description="x")],
            similar_experiences=[],
        )
        # Adding requirements should change the confidence (3 signals vs 2)
        assert conf_with != conf_without

    def test_similar_experiences_affect_confidence(self):
        exps = [
            SimilarExperience(task_id="1", description="test", similarity_score=0.8, outcome=True)
        ]
        conf = self.integration._calculate_confidence(
            intent=ActionIntent.AMBIGUOUS,
            complexity=TaskComplexity.SIMPLE,
            requirements=[],
            similar_experiences=exps,
        )
        # Should produce a reasonable value
        assert 0.3 < conf < 0.8

    def test_max_confidence_capped_at_1(self):
        exps = [
            SimilarExperience(task_id="1", description="test", similarity_score=1.0, outcome=True)
        ]
        conf = self.integration._calculate_confidence(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.COMPLEX,
            requirements=[Requirement(type=RequirementType.FUNCTIONAL, description="x")],
            similar_experiences=exps,
            intent_confidence=0.95,
        )
        assert conf <= 1.0


# ============================================================================
# PerceptionIntegration.perceive() tests
# ============================================================================


class TestPerceive:
    """Tests for PerceptionIntegration.perceive()."""

    async def test_perceive_basic(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive("Fix the authentication bug")
        assert isinstance(result, Perception)
        assert isinstance(result.intent, ActionIntent)
        assert isinstance(result.complexity, TaskComplexity)
        assert 0.0 <= result.confidence <= 1.0

    async def test_perceive_with_requirements(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive("The code must handle errors and should be fast")
        assert len(result.requirements) >= 1

    async def test_perceive_without_requirements(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_requirement_extraction=False,
            enable_similarity_search=False,
        )
        result = await integration.perceive("The code must handle errors")
        assert len(result.requirements) == 0

    async def test_perceive_metadata(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive(
            "Hello",
            context={"project": "test"},
        )
        assert result.metadata["query_length"] == 5
        assert result.metadata["has_context"] is True

    async def test_perceive_no_context(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive("Hello")
        assert result.metadata["has_context"] is False

    async def test_perceive_with_memory_coordinator(self):
        mock_memory = AsyncMock()
        mock_memory.search_all = AsyncMock(return_value=[])

        integration = PerceptionIntegration(
            memory_coordinator=mock_memory,
            enable_similarity_search=True,
        )
        result = await integration.perceive("Fix the bug")
        mock_memory.search_all.assert_called_once()
        assert result.similar_experiences == []

    async def test_perceive_memory_error_handled(self):
        mock_memory = AsyncMock()
        mock_memory.search_all = AsyncMock(side_effect=Exception("DB error"))

        integration = PerceptionIntegration(
            memory_coordinator=mock_memory,
            enable_similarity_search=True,
        )
        result = await integration.perceive("Fix the bug")
        assert result.similar_experiences == []

    async def test_perceive_flags_underspecified_action_for_clarification(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive("Fix it and add tests.")

        assert result.needs_clarification is True
        assert result.clarification_reason == "target artifact or scope is underspecified"
        assert result.clarification_prompt is not None
        assert "Which file, component, or bug" in result.clarification_prompt

    async def test_perceive_uses_configured_policy_prompt_for_underspecified_action(
        self,
    ):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
            config={"underspecified_target_prompt": "Name the exact file to change."},
        )
        result = await integration.perceive("Fix it and add tests.")

        assert result.needs_clarification is True
        assert result.clarification_prompt == "Name the exact file to change."

    async def test_perceive_explicit_target_avoids_clarification(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive("Fix src/auth/login.py and add tests.")

        assert result.needs_clarification is False
        assert result.clarification_prompt is None

    async def test_perceive_does_not_flag_relative_clause_report_request(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive(
            "Identify all dead code or deprecated call sites that should be modified first "
            "to enable clean code principle based implementations."
        )

        assert result.needs_clarification is False
        assert result.clarification_prompt is None

    async def test_perceive_uses_history_to_resolve_ambiguous_follow_up_target(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        history = [
            {
                "role": "assistant",
                "content": "Start with victor/agent/services/tool_compat.py and migrate its "
                "deprecated call sites first.",
            }
        ]

        result = await integration.perceive(
            "Fix that first and add tests.",
            conversation_history=history,
        )

        assert result.needs_clarification is False
        assert result.clarification_prompt is None

    async def test_perceive_does_not_flag_checklist_creation_task(self):
        """'Create a comprehensive checklist' is a self-contained generation task.

        This task was previously misclassified because 'create' is an action marker
        and the task has no file path.  The checklist IS the output artifact — the
        task is well-specified and does not require clarification.

        Regression: victor chat -p zai-coding step 5 'Create comprehensive Rust best
        practices checklist' was aborted with EvaluationDecision.FAIL because
        PerceptionIntegration fired the underspecified_target check.
        """
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive(
            "Create comprehensive Rust best practices checklist: Arc vs Rc, unnecessary "
            "cloning, immutable bindings, concurrency safety, Send/Sync correctness, "
            "zero-copy patterns, Cow usage, lifetime efficiency, trait object vs generics, "
            "error handling, async best practices, ownership transfer vs reference, "
            "allocation hotspots, Interior mutability (RefCell/Mutex/RwLock), lock "
            "granularity, static dispatch vs dynamic dispatch"
        )

        assert result.needs_clarification is False, (
            f"Checklist creation task should not require clarification; "
            f"got reason={result.clarification_reason}"
        )

    async def test_perceive_does_not_flag_report_generation_task(self):
        """'Generate a report' is self-contained — no file target needed."""
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive(
            "Generate a performance report summarizing memory usage and latency across all "
            "services."
        )

        assert result.needs_clarification is False, (
            f"Report generation should not require clarification; "
            f"got reason={result.clarification_reason}"
        )

    async def test_perceive_does_not_flag_write_guide_task(self):
        """'Write a guide on X' is self-contained — no file target needed."""
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive(
            "Write a best practices guide for async Rust covering tokio, error handling, "
            "and cancellation safety."
        )

        assert result.needs_clarification is False, (
            f"Guide writing task should not require clarification; "
            f"got reason={result.clarification_reason}"
        )

    async def test_perceive_still_flags_vague_create_in_file_task(self):
        """'Create a function in it' without file context remains ambiguous."""
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        result = await integration.perceive("Create it in the file and add tests.")

        # This is a file-modifying vague task — clarification should still fire
        assert result.needs_clarification is True


# ============================================================================
# Convenience function tests
# ============================================================================


class TestPerceiveConvenience:
    """Tests for the module-level perceive() function."""

    async def test_perceive_convenience(self):
        result = await perceive("Fix the bug in main.py")
        assert isinstance(result, Perception)
        assert isinstance(result.intent, ActionIntent)

    async def test_perceive_with_history(self):
        history = [{"role": "user", "content": "previous message"}]
        result = await perceive(
            "Continue fixing the bug",
            conversation_history=history,
        )
        assert isinstance(result, Perception)


# ============================================================================
# Enhanced Perception property tests
# ============================================================================


class TestPerceptionEnhancedProperties:
    """Tests for new Perception properties from TaskAnalysis."""

    def test_tool_budget_available(self):
        p = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(tool_budget=5),
        )
        assert p.tool_budget == 5

    def test_tool_budget_none(self):
        p = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.SIMPLE,
            task_analysis=MagicMock(spec=[]),
        )
        assert p.tool_budget is None

    def test_should_spawn_team(self):
        p = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.COMPLEX,
            task_analysis=MagicMock(should_spawn_team=True),
        )
        assert p.should_spawn_team is True

    def test_coordination_suggestion(self):
        p = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.COMPLEX,
            task_analysis=MagicMock(coordination_suggestion="parallel"),
        )
        assert p.coordination_suggestion == "parallel"

    def test_to_dict_includes_new_fields(self):
        p = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(
                task_type="code_generation",
                tool_budget=5,
                should_spawn_team=False,
                coordination_suggestion=None,
            ),
        )
        d = p.to_dict()
        assert "tool_budget" in d
        assert "should_spawn_team" in d
        assert "coordination_suggestion" in d
        assert "needs_clarification" in d
        assert "clarification_prompt" in d


# ============================================================================
# Calibrated confidence tests
# ============================================================================


class TestCalibratedConfidence:
    """Tests for calibrated confidence calculation."""

    def setup_method(self):
        self.integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )

    def test_geometric_mean_produces_calibrated_result(self):
        """Geometric mean should be lower than linear sum for same inputs."""
        conf = self.integration._calculate_confidence(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.COMPLEX,
            requirements=[],
            similar_experiences=[],
            intent_confidence=0.8,
        )
        # Should be a reasonable value between 0 and 1
        assert 0.3 < conf < 1.0

    def test_ambiguous_intent_lowers_confidence(self):
        conf_clear = self.integration._calculate_confidence(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.SIMPLE,
            requirements=[],
            similar_experiences=[],
            intent_confidence=0.9,
        )
        conf_ambiguous = self.integration._calculate_confidence(
            intent=ActionIntent.AMBIGUOUS,
            complexity=TaskComplexity.SIMPLE,
            requirements=[],
            similar_experiences=[],
            intent_confidence=0.0,
        )
        assert conf_clear > conf_ambiguous

    def test_more_signals_refine_confidence(self):
        conf_minimal = self.integration._calculate_confidence(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            requirements=[],
            similar_experiences=[],
        )
        conf_rich = self.integration._calculate_confidence(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            requirements=[Requirement(type=RequirementType.FUNCTIONAL, description="x")],
            similar_experiences=[
                SimilarExperience(task_id="1", description="t", similarity_score=0.9, outcome=True)
            ],
        )
        # Both should be reasonable (geometric mean makes this non-trivial)
        assert 0.3 < conf_minimal < 1.0
        assert 0.3 < conf_rich < 1.0

    async def test_multi_turn_context_passed_to_analyzer(self):
        integration = PerceptionIntegration(
            memory_coordinator=None,
            enable_similarity_search=False,
        )
        history = [
            {"role": "user", "content": "Fix the login bug"},
            {"role": "assistant", "content": "I found the issue in auth.py"},
        ]
        result = await integration.perceive(
            "Now add tests for that fix",
            conversation_history=history,
        )
        assert isinstance(result, Perception)
