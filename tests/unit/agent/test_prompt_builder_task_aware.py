"""Tests for task-aware system prompts — Layer 2 of agentic execution quality."""

from unittest.mock import MagicMock, patch


from victor.agent.prompt_builder import SystemPromptBuilder
from victor.agent.query_classifier import QueryClassification, QueryType
from victor.framework.task.protocols import TaskComplexity


def _make_classification(query_type: QueryType) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        complexity=TaskComplexity.MEDIUM,
        should_plan=False,
        should_use_subagents=False,
        continuation_budget_hint=4,
        confidence=0.8,
    )


def _make_builder(query_type=None, available_tools=None):
    classification = _make_classification(query_type) if query_type else None
    return SystemPromptBuilder(
        provider_name="anthropic",
        model="claude-sonnet-4-20250514",
        available_tools=available_tools,
        query_classification=classification,
    )


class TestTaskGuidance:
    def test_exploration_prompt_includes_systematic_guidance(self):
        builder = _make_builder(QueryType.EXPLORATION)
        prompt = builder.build()
        assert "systematically" in prompt.lower() or "map structure" in prompt.lower()

    def test_implementation_prompt_includes_plan_guidance(self):
        builder = _make_builder(QueryType.IMPLEMENTATION)
        prompt = builder.build()
        assert "plan before" in prompt.lower() or "break into" in prompt.lower()

    def test_debugging_prompt_includes_error_focus(self):
        builder = _make_builder(QueryType.DEBUGGING)
        prompt = builder.build()
        assert "error messages" in prompt.lower() or "stack traces" in prompt.lower()

    def test_quick_question_prompt_is_concise(self):
        builder = _make_builder(QueryType.QUICK_QUESTION)
        prompt = builder.build()
        assert "directly" in prompt.lower() or "concisely" in prompt.lower()

    def test_no_classification_omits_task_guidance(self):
        builder = _make_builder(None)
        prompt = builder.build()
        # Without classification, no task-specific guidance injected
        assert "TASK GUIDANCE:" not in prompt

    def test_tool_constraint_lists_available_tools(self):
        builder = _make_builder(available_tools=["read_file", "write_file", "shell"])
        prompt = builder.build()
        assert "read" in prompt
        assert "write" in prompt
        assert "shell" in prompt
        assert "read_file" not in prompt
        assert "write_file" not in prompt

    def test_mode_guidance_is_injected(self):
        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            mode_prompt_addition="PLAN mode: Only edit files in .victor/sandbox/.",
        )
        prompt = builder.build()
        assert "PLAN mode" in prompt
        assert ".victor/sandbox/" in prompt

    def test_prompt_content_varies_by_type(self):
        quick = _make_builder(QueryType.QUICK_QUESTION).build()
        explore = _make_builder(QueryType.EXPLORATION).build()
        # Each type includes its own task guidance
        assert "Answer directly" in quick or "concisely" in quick.lower()
        assert "systematically" in explore.lower() or "map structure" in explore.lower()
        # They should differ (different task guidance injected)
        assert quick != explore

    def test_edge_focus_uses_shared_selector_catalog(self):
        builder = _make_builder()
        builder.concise_mode = False
        builder._task_type = "analysis"
        builder._user_message = "Review the graph codebase"

        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_container.get.return_value = mock_service

        with (
            patch(
                "victor.core.service_resolution.get_container",
                return_value=mock_container,
            ),
            patch(
                "victor.agent.edge_model.select_prompt_sections_with_edge_model",
                return_value=["tool_guidance"],
            ) as select_sections,
        ):
            sections = builder._get_active_sections()

        assert {
            "mode_guidance",
            "task_guidance",
            "tool_constraint",
            "completion",
        } <= sections
        assert "tool_guidance" in sections
        assert "few_shot_examples" not in sections
        assert select_sections.call_args.kwargs["available_sections"] == [
            "grounding",
            "completion",
            "tool_guidance",
            "file_pagination",
            "concise_mode",
            "parallel_read",
        ]

    def test_edge_focus_only_enables_concise_mode_when_active(self):
        builder = _make_builder()
        builder.concise_mode = False

        assert builder._map_edge_focus_to_builder_sections({"concise_mode"}) == set()

        builder.concise_mode = True
        assert builder._map_edge_focus_to_builder_sections({"concise_mode"}) == {"concise_mode"}


class TestGEPAPromptIntegration:
    """Tests for GEPA-evolved prompt section integration in the builder."""

    def test_default_prompt_includes_static_grounding(self):
        """System prompt always includes static GROUNDING_RULES."""
        from victor.agent.prompt_builder import GROUNDING_RULES

        builder = _make_builder()
        prompt = builder.build()
        assert GROUNDING_RULES in prompt

    def test_static_grounding_keeps_structured_data_guardrails(self):
        """Baseline grounding should retain promoted raw-output handling guidance."""
        from victor.agent.prompt_builder import GROUNDING_RULES

        assert "raw data structures" in GROUNDING_RULES
        assert "verify field existence and types" in GROUNDING_RULES
        assert "targeted code_search()" in GROUNDING_RULES

    def test_default_prompt_includes_static_completion(self):
        """System prompt always includes static COMPLETION_GUIDANCE."""
        from victor.agent.prompt_builder import COMPLETION_GUIDANCE

        builder = _make_builder()
        prompt = builder.build()
        assert COMPLETION_GUIDANCE in prompt

    def test_default_prompt_includes_static_tool_effectiveness_guidance(self):
        """System prompt always includes static ASI tool-effectiveness guidance."""
        from unittest.mock import patch

        from victor.agent.prompt_builder import ASI_TOOL_EFFECTIVENESS_GUIDANCE

        builder = _make_builder()
        with patch.object(builder, "_get_active_sections", return_value={"tool_guidance"}):
            prompt = builder.build()
        assert ASI_TOOL_EFFECTIVENESS_GUIDANCE in prompt

    def test_static_completion_keeps_tool_execution_discipline(self):
        """Baseline completion guidance should retain promoted tool discipline notes."""
        from victor.agent.prompt_builder import COMPLETION_GUIDANCE

        assert "TOOL EXECUTION DISCIPLINE" in COMPLETION_GUIDANCE
        assert "Never repeat an identical failing call" in COMPLETION_GUIDANCE
        assert "offset/limit/search" in COMPLETION_GUIDANCE

    def test_concise_mode_can_use_scoped_evolved_guidance(self):
        """Scoped concise guidance should resolve through the prompt builder."""
        from victor.agent.evolved_content_resolver import ResolvedContent

        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            concise_mode=True,
        )

        with patch(
            "victor.agent.evolved_content_resolver.EvolvedContentResolver.resolve_section",
            return_value=ResolvedContent(
                section_name="CONCISE_MODE_GUIDANCE",
                text="EVOLVED CONCISE MODE",
                source="evolved",
                metadata={},
            ),
        ):
            prompt = builder.build()

        assert "EVOLVED CONCISE MODE" in prompt

    def test_deepseek_prompt_can_use_scoped_large_file_guidance(self):
        """Provider-specific large-file guidance should be evolvable in place."""
        from victor.agent.evolved_content_resolver import ResolvedContent

        builder = SystemPromptBuilder(
            provider_name="deepseek",
            model="deepseek-chat",
        )

        with patch(
            "victor.agent.evolved_content_resolver.EvolvedContentResolver.resolve_section",
            return_value=ResolvedContent(
                section_name="LARGE_FILE_PAGINATION_GUIDANCE",
                text="EVOLVED LARGE FILE GUIDANCE",
                source="evolved",
                metadata={},
            ),
        ):
            prompt = builder.build()

        assert "EVOLVED LARGE FILE GUIDANCE" in prompt

    def test_cloud_prompt_can_use_scoped_parallel_read_guidance(self):
        """Parallel-read guidance should resolve through the scoped prompt path."""
        from victor.agent.evolved_content_resolver import ResolvedContent

        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )

        with patch(
            "victor.agent.evolved_content_resolver.EvolvedContentResolver.resolve_section",
            return_value=ResolvedContent(
                section_name="PARALLEL_READ_GUIDANCE",
                text="EVOLVED PARALLEL READ GUIDANCE",
                source="evolved",
                metadata={},
            ),
        ):
            prompt = builder.build()

        assert "EVOLVED PARALLEL READ GUIDANCE" in prompt

    def test_deepseek_prompt_can_use_scoped_extended_grounding_guidance(self):
        """Extended grounding guidance should be evolvable in provider prompts."""
        from victor.agent.evolved_content_resolver import ResolvedContent

        builder = SystemPromptBuilder(
            provider_name="deepseek",
            model="deepseek-chat",
        )

        with patch(
            "victor.agent.evolved_content_resolver.EvolvedContentResolver.resolve_section",
            side_effect=[
                ResolvedContent(
                    section_name="LARGE_FILE_PAGINATION_GUIDANCE",
                    text="Static large file guidance",
                    source="static",
                    metadata={},
                ),
                ResolvedContent(
                    section_name="GROUNDING_RULES_EXTENDED",
                    text="EVOLVED EXTENDED GROUNDING",
                    source="evolved",
                    metadata={},
                ),
            ],
        ):
            prompt = builder.build()

        assert "EVOLVED EXTENDED GROUNDING" in prompt

    def test_prompt_can_use_scoped_completion_guidance(self):
        """Completion guidance should resolve through the document builder path."""
        from victor.agent.evolved_content_resolver import ResolvedContent

        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )

        with patch(
            "victor.agent.evolved_content_resolver.EvolvedContentResolver.resolve_section",
            side_effect=lambda section_name, *args, **kwargs: ResolvedContent(
                section_name=section_name,
                text=(
                    "EVOLVED COMPLETION GUIDANCE"
                    if section_name == "COMPLETION_GUIDANCE"
                    else kwargs.get("fallback_text") or ""
                ),
                source="evolved" if section_name == "COMPLETION_GUIDANCE" else "static",
                metadata={},
            ),
        ):
            prompt = builder.build()

        assert "EVOLVED COMPLETION GUIDANCE" in prompt

    def test_prompt_can_use_scoped_tool_effectiveness_guidance(self):
        """ASI tool guidance should resolve through the document builder path."""
        from victor.agent.evolved_content_resolver import ResolvedContent

        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )

        with (
            patch.object(builder, "_get_active_sections", return_value={"tool_guidance"}),
            patch(
                "victor.agent.evolved_content_resolver.EvolvedContentResolver.resolve_section",
                side_effect=lambda section_name, *args, **kwargs: ResolvedContent(
                    section_name=section_name,
                    text=(
                        "EVOLVED TOOL GUIDANCE"
                        if section_name == "ASI_TOOL_EFFECTIVENESS_GUIDANCE"
                        else kwargs.get("fallback_text") or ""
                    ),
                    source=(
                        "evolved" if section_name == "ASI_TOOL_EFFECTIVENESS_GUIDANCE" else "static"
                    ),
                    metadata={},
                ),
            ),
        ):
            prompt = builder.build()

        assert "EVOLVED TOOL GUIDANCE" in prompt

    def test_optimized_grounding_via_optimization_injector(self):
        """GEPA-evolved GROUNDING_RULES is served via OptimizationInjector."""
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )
        from victor.framework.rl.base import RLRecommendation
        from victor.agent.optimization_injector import OptimizationInjector

        evolved_text = "EVOLVED GROUNDING: Always verify tool output."

        mock_rec = RLRecommendation(
            value=evolved_text,
            confidence=0.9,
            reason="GEPA gen-2",
            sample_size=10,
            is_baseline=False,
        )
        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = mock_rec
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch(
                "victor.config.settings.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            sections = injector.get_evolved_sections("deepseek", "deepseek-chat", "edit")

        assert any(evolved_text in s for s in sections)

    def test_optimized_completion_replaces_static(self):
        """GEPA-evolved completion guidance should flow into the system prompt."""
        from victor.agent.prompt_builder import COMPLETION_GUIDANCE
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )
        from victor.framework.rl.base import RLRecommendation
        from victor.agent.optimization_injector import OptimizationInjector

        evolved_text = "EVOLVED COMPLETION: Signal with **FINISHED**: marker."

        mock_rec = RLRecommendation(
            value=evolved_text,
            confidence=0.9,
            reason="GEPA gen-3",
            sample_size=10,
            is_baseline=False,
        )
        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = mock_rec
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch(
                "victor.config.settings.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            sections = injector.get_evolved_sections("deepseek", "deepseek-chat", "edit")

            # System prompt retains static COMPLETION_GUIDANCE
            builder = _make_builder()
            prompt = builder.build()

        # Evolved version goes to user prefix via injector
        assert any(evolved_text in s for s in sections)
        # Canonical builder now resolves the evolved completion section in place
        assert evolved_text in prompt
        assert COMPLETION_GUIDANCE not in prompt

    def test_prompt_optimization_disabled_uses_static_prompts(self):
        """When prompt_optimization.enabled=False, no GEPA candidates used."""
        from victor.agent.prompt_builder import GROUNDING_RULES
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=False)

        with patch(
            "victor.config.settings.get_settings",
            return_value=mock_settings,
        ):
            builder = _make_builder()
            prompt = builder.build()

        # Static GROUNDING_RULES should remain (no optimization applied)
        assert GROUNDING_RULES in prompt


class TestPromptOptimizationSettings:
    """Tests for the strategy-agnostic prompt optimization config."""

    def test_defaults(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings()
        assert s.enabled  # Enabled by default for GEPA/MIPROv2/CoT
        assert s.default_strategies == ["gepa"]
        assert s.section_strategies == {}

    def test_get_strategies_returns_default(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=True)
        assert s.get_strategies_for_section("UNLISTED_SECTION") == ["gepa"]

    def test_get_strategies_uses_builtin_section_defaults(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=True)
        assert s.get_strategies_for_section("FEW_SHOT_EXAMPLES") == ["miprov2"]
        assert s.get_strategies_for_section("ASI_TOOL_EFFECTIVENESS_GUIDANCE") == [
            "gepa",
            "cot_distillation",
        ]

    def test_get_strategies_uses_runtime_registered_section_defaults(self, monkeypatch):
        from victor.agent import prompt_section_registry as registry_module
        from victor.agent.prompt_section_registry import (
            SectionCategory,
            SectionDefinition,
            UnifiedSectionRegistry,
            _initialize_default_sections,
        )
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        fresh_registry = UnifiedSectionRegistry()
        _initialize_default_sections(fresh_registry)
        fresh_registry.register(
            SectionDefinition(
                name="CUSTOM_REVIEW_GUIDANCE",
                aliases={"custom_review"},
                category=SectionCategory.TASK_HINTS,
                default_text="Review API drift first.",
                evolvable=True,
                required=False,
                priority=42,
                default_strategies=("gepa", "prefpo"),
            )
        )
        monkeypatch.setattr(registry_module, "_registry", fresh_registry)

        s = PromptOptimizationSettings(enabled=True)
        assert s.get_strategies_for_section("CUSTOM_REVIEW_GUIDANCE") == [
            "gepa",
            "prefpo",
        ]

    def test_get_strategies_uses_override(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(
            enabled=True,
            section_strategies={"COMPLETION_GUIDANCE": []},
        )
        assert s.get_strategies_for_section("COMPLETION_GUIDANCE") == []
        assert s.get_strategies_for_section("GROUNDING_RULES") == ["gepa", "prefpo"]

    def test_get_strategies_empty_when_disabled(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=False)
        assert s.get_strategies_for_section("GROUNDING_RULES") == []

    def test_is_strategy_active(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=True)
        assert s.is_strategy_active("gepa")
        assert s.is_strategy_active("prefpo")
        assert s.is_strategy_active("miprov2")
        assert s.is_strategy_active("cot_distillation")
        assert not s.is_strategy_active("nonexistent")

    def test_nested_gepa_config(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=True)
        assert s.gepa.max_prompt_chars == 1500
        assert s.gepa.default_tier == "balanced"
        assert s.get_strategies_for_section("CONCISE_MODE_GUIDANCE") == ["prefpo"]
        assert s.get_strategies_for_section("LARGE_FILE_PAGINATION_GUIDANCE") == ["gepa"]


class TestOptimizationInjectorFewShots:
    def test_bound_prompt_candidate_bypasses_recommendation_sampling(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from victor.agent.optimization_injector import OptimizationInjector
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        candidate = SimpleNamespace(
            text="BOUND GROUNDING",
            text_hash="cand-bound",
            section_name="GROUNDING_RULES",
            provider="anthropic",
            strategy_name="gepa",
            strategy_chain="gepa,cot_distillation",
        )
        mock_learner = MagicMock()
        mock_learner.get_candidate.return_value = candidate
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=False)

        with (
            patch("victor.config.settings.get_settings", return_value=mock_settings),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            injector.bind_prompt_candidate(
                section_name="GROUNDING_RULES",
                prompt_candidate_hash="cand-bound",
                provider="anthropic",
            )
            payloads = injector.get_evolved_section_payloads(
                provider="anthropic",
                model="claude-sonnet",
                task_type="edit",
            )

        grounding = next(
            payload for payload in payloads if payload["section_name"] == "GROUNDING_RULES"
        )
        assert grounding["text"] == "BOUND GROUNDING"
        assert grounding["provider"] == "anthropic"
        assert grounding["prompt_candidate_hash"] == "cand-bound"
        assert grounding["strategy_name"] == "gepa"
        assert grounding["strategy_chain"] == "gepa,cot_distillation"
        assert grounding["source"] == "bound_candidate"
        mock_learner.get_recommendation.assert_not_called()

    def test_evolved_section_payloads_include_prompt_identity(self):
        from unittest.mock import MagicMock, patch

        from victor.agent.optimization_injector import OptimizationInjector
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )
        from victor.framework.rl.base import RLRecommendation

        mock_rec = RLRecommendation(
            value="EVOLVED GROUNDING",
            confidence=0.9,
            reason="GEPA gen-4",
            sample_size=12,
            is_baseline=False,
            metadata={
                "provider": "anthropic",
                "prompt_candidate_hash": "cand-123",
                "section_name": "GROUNDING_RULES",
                "strategy_name": "gepa",
            },
        )
        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = mock_rec
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch("victor.config.settings.get_settings", return_value=mock_settings),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            payloads = injector.get_evolved_section_payloads(
                provider="anthropic",
                model="claude-sonnet",
                task_type="edit",
            )

        grounding = next(
            payload for payload in payloads if payload["section_name"] == "GROUNDING_RULES"
        )
        assert grounding["text"] == "EVOLVED GROUNDING"
        assert grounding["provider"] == "anthropic"
        assert grounding["prompt_candidate_hash"] == "cand-123"
        assert grounding["prompt_section_name"] == "GROUNDING_RULES"

    def test_turn_prefix_payloads_exclude_scoped_sections(self):
        """Scoped sections should resolve on demand, not in every turn prefix."""
        from unittest.mock import MagicMock, patch

        from victor.agent.optimization_injector import OptimizationInjector
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = None
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch("victor.config.settings.get_settings", return_value=mock_settings),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            payloads = injector.get_evolved_section_payloads(
                provider="anthropic",
                model="claude-sonnet",
                task_type="edit",
            )

        names = {payload["section_name"] for payload in payloads}
        assert "CONCISE_MODE_GUIDANCE" not in names
        assert "LARGE_FILE_PAGINATION_GUIDANCE" not in names
        assert "INIT_SYNTHESIS_RULES" not in names

    def test_turn_prefix_payloads_use_registry_required_sections(self, monkeypatch):
        """Required evolvable registry sections should drive the turn-prefix bundle."""
        from unittest.mock import MagicMock, patch

        from victor.agent import prompt_section_registry as registry_module
        from victor.agent.optimization_injector import OptimizationInjector
        from victor.agent.prompt_section_registry import (
            SectionCategory,
            SectionDefinition,
            UnifiedSectionRegistry,
            _initialize_default_sections,
        )
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )
        from victor.framework.rl.base import RLRecommendation

        fresh_registry = UnifiedSectionRegistry()
        _initialize_default_sections(fresh_registry)
        fresh_registry.register(
            SectionDefinition(
                name="CUSTOM_REVIEW_GUIDANCE",
                aliases={"custom_review"},
                category=SectionCategory.TASK_HINTS,
                default_text="Review API drift first.",
                evolvable=True,
                required=True,
                priority=55,
            )
        )
        monkeypatch.setattr(registry_module, "_registry", fresh_registry)

        def _recommendation(_provider, _model, _task_type, *, section_name):
            if section_name != "CUSTOM_REVIEW_GUIDANCE":
                return None
            return RLRecommendation(
                value="EVOLVED CUSTOM REVIEW",
                confidence=0.9,
                reason="GEPA gen-1",
                sample_size=4,
                is_baseline=False,
                metadata={
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-custom",
                    "section_name": section_name,
                    "strategy_name": "gepa",
                },
            )

        mock_learner = MagicMock()
        mock_learner.get_recommendation.side_effect = _recommendation
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch("victor.config.settings.get_settings", return_value=mock_settings),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            payloads = injector.get_evolved_section_payloads(
                provider="anthropic",
                model="claude-sonnet",
                task_type="edit",
            )

        names = [payload["section_name"] for payload in payloads]
        assert "CUSTOM_REVIEW_GUIDANCE" in names
        custom_payload = next(
            payload for payload in payloads if payload["section_name"] == "CUSTOM_REVIEW_GUIDANCE"
        )
        assert custom_payload["text"] == "EVOLVED CUSTOM REVIEW"
        assert custom_payload["prompt_candidate_hash"] == "cand-custom"

    def test_turn_prefix_asi_static_fallback_uses_registry_text(self, monkeypatch):
        """ASI fallback payloads should use the registry baseline, not prompt-builder imports."""
        from unittest.mock import MagicMock, patch

        from victor.agent import prompt_section_registry as registry_module
        from victor.agent.optimization_injector import OptimizationInjector
        from victor.agent.prompt_section_registry import (
            SectionCategory,
            SectionDefinition,
            UnifiedSectionRegistry,
            _initialize_default_sections,
        )
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        fresh_registry = UnifiedSectionRegistry()
        _initialize_default_sections(fresh_registry)
        fresh_registry.register(
            SectionDefinition(
                name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
                aliases={
                    "tool_effectiveness_guidance",
                    "tool_hints",
                    "asi_tool_guidance",
                },
                category=SectionCategory.TOOL_GUIDANCE,
                default_text="Registry-owned ASI fallback guidance.",
                evolvable=True,
                required=True,
                priority=50,
                default_strategies=("gepa", "cot_distillation"),
            )
        )
        monkeypatch.setattr(registry_module, "_registry", fresh_registry)

        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = None
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch("victor.config.settings.get_settings", return_value=mock_settings),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            payloads = injector.get_evolved_section_payloads(
                provider="anthropic",
                model="claude-sonnet",
                task_type="edit",
            )

        asi_payload = next(
            payload
            for payload in payloads
            if payload["section_name"] == "ASI_TOOL_EFFECTIVENESS_GUIDANCE"
        )
        assert asi_payload["text"] == "Registry-owned ASI fallback guidance."
        assert asi_payload["source"] == "static_fallback"

    def test_query_aware_few_shots_are_cached_per_query(self):
        from unittest.mock import MagicMock, patch

        from victor.agent.optimization_injector import OptimizationInjector
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        mock_learner = MagicMock()
        mock_learner.get_query_aware_few_shots.side_effect = lambda query: f"few-shot for {query}"
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch(
                "victor.config.settings.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            first = injector.get_few_shots("fix auth bug")
            second = injector.get_few_shots("fix billing bug")
            third = injector.get_few_shots("fix auth bug")

        assert first == "few-shot for fix auth bug"
        assert second == "few-shot for fix billing bug"
        assert third == "few-shot for fix auth bug"
        assert mock_learner.get_query_aware_few_shots.call_count == 2

    def test_query_aware_few_shots_are_scoped_by_provider_and_model(self):
        from unittest.mock import MagicMock, patch

        from victor.agent.optimization_injector import OptimizationInjector
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        mock_learner = MagicMock()
        mock_learner.get_query_aware_few_shots.return_value = "few-shot for same query"
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch("victor.config.settings.get_settings", return_value=mock_settings),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            first = injector.get_few_shot_payload(
                "same query",
                provider="anthropic",
                model="claude-sonnet",
                task_type="analysis",
            )
            second = injector.get_few_shot_payload(
                "same query",
                provider="zai",
                model="glm-5.1",
                task_type="analysis",
            )

        assert first is not None
        assert second is not None
        assert first["provider"] == "anthropic"
        assert second["provider"] == "zai"
        assert mock_learner.get_query_aware_few_shots.call_count == 2

    def test_few_shot_payload_uses_canonical_identity_shape(self):
        from unittest.mock import MagicMock, patch

        from victor.agent.optimization_injector import OptimizationInjector
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        mock_learner = MagicMock()
        mock_learner.get_query_aware_few_shots.return_value = "few-shot for fix auth bug"
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch("victor.config.settings.get_settings", return_value=mock_settings),
            patch(
                "victor.agent.services.rl_runtime.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            payload = injector.get_few_shot_payload(
                "fix auth bug",
                provider="anthropic",
                model="claude-sonnet",
                task_type="edit",
            )

        assert payload is not None
        assert payload["text"] == "few-shot for fix auth bug"
        assert payload["provider"] == "anthropic"
        assert payload["prompt_candidate_hash"] is None
        assert payload["section_name"] == "FEW_SHOT_EXAMPLES"
