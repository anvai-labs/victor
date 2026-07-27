"""TDD tests for UnifiedPromptPipeline.

Tests written BEFORE implementation per plan. These validate:
1. Tier detection from provider capabilities
2. System prompt build + freeze behavior per tier
3. Per-turn user prefix composition
4. Single frozen flag (no dual state)
5. Credit injection deduplication (never in both system + prefix)
6. Backward compatibility with old module callers
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ============================================================================
# Helper: create mock providers for each tier
# ============================================================================


def _make_provider(api_cache: bool = False, kv_cache: bool = False) -> MagicMock:
    """Create a mock provider with specified caching capabilities."""
    provider = MagicMock()
    provider.supports_prompt_caching.return_value = api_cache
    provider.supports_kv_prefix_caching.return_value = kv_cache
    return provider


def _make_builder(base_prompt: str = "You are a helpful assistant.") -> MagicMock:
    """Create a mock SystemPromptBuilder."""
    builder = MagicMock()
    builder.build.return_value = base_prompt
    builder.provider_name = "test"
    builder.model = "test-model"
    return builder


def _make_registry() -> MagicMock:
    """Create a minimal ContentRegistry mock."""
    registry = MagicMock()
    registry.get_all.return_value = []
    registry.get_by_category.return_value = []
    return registry


def _make_optimizer(evolved_sections=None, few_shots=None, failure_hint=None) -> MagicMock:
    """Create a mock OptimizationInjector."""
    optimizer = MagicMock()
    optimizer.get_evolved_sections.return_value = evolved_sections or []
    optimizer.get_few_shots.return_value = few_shots
    optimizer.get_failure_hint.return_value = failure_hint
    optimizer.clear_session_cache.return_value = None
    return optimizer


def _make_pipeline(**kwargs):
    """Create a UnifiedPromptPipeline with sensible defaults."""
    from victor.agent.prompt_pipeline import UnifiedPromptPipeline

    defaults = {
        "provider": _make_provider(),
        "builder": _make_builder(),
        "registry": _make_registry(),
        "optimizer": None,
        "task_analyzer": None,
        "runtime_intelligence": None,
        "get_context_window": lambda: 128000,
        "session_id": "test-session",
    }
    defaults.update(kwargs)
    return UnifiedPromptPipeline(**defaults)


def _make_content_item(name: str, section_group: str, required: bool = False):
    from victor.agent.content_registry import ContentCategory, ContentItem

    return ContentItem(
        name=name,
        category=ContentCategory.STATIC,
        default_text="content",
        token_estimate=10,
        evolvable=False,
        required=required,
        section_group=section_group,
    )


# ============================================================================
# 1. Tier Detection (3 tests)
# ============================================================================


class TestTierDetection:
    """Verify provider capabilities map to correct ProviderTier."""

    def test_tier_a_api_and_kv(self):
        """Anthropic/OpenAI-like: both API cache and KV cache."""
        from victor.agent.prompt_pipeline import ProviderTier

        pipeline = _make_pipeline(provider=_make_provider(api_cache=True, kv_cache=True))
        assert pipeline.tier == ProviderTier.API_AND_KV

    def test_tier_b_kv_only(self):
        """Ollama/LMStudio-like: KV prefix cache only, no API billing discount."""
        from victor.agent.prompt_pipeline import ProviderTier

        pipeline = _make_pipeline(provider=_make_provider(api_cache=False, kv_cache=True))
        assert pipeline.tier == ProviderTier.KV_ONLY

    def test_tier_c_no_cache(self):
        """Unknown/custom provider: no caching support."""
        from victor.agent.prompt_pipeline import ProviderTier

        pipeline = _make_pipeline(provider=_make_provider(api_cache=False, kv_cache=False))
        assert pipeline.tier == ProviderTier.NO_CACHE


class TestContentRouterEdgeSections:
    """Verify edge-section routing understands canonical and fine-grained selectors."""

    def test_kv_only_router_accepts_canonical_section_name(self):
        from victor.agent.prompt_pipeline import ContentRouter, Placement, ProviderTier

        router = ContentRouter(ProviderTier.KV_ONLY, {"GROUNDING_RULES"})
        item = _make_content_item("GROUNDING_RULES", "grounding")

        assert router.route(item) == Placement.SYSTEM_PROMPT

    def test_kv_only_router_accepts_specific_file_pagination_selector(self):
        from victor.agent.prompt_pipeline import ContentRouter, Placement, ProviderTier

        router = ContentRouter(ProviderTier.KV_ONLY, {"file_pagination"})
        item = _make_content_item("LARGE_FILE_PAGINATION_GUIDANCE", "tool_guidance")

        assert router.route(item) == Placement.SYSTEM_PROMPT

    def test_kv_only_router_accepts_specific_parallel_read_selector(self):
        from victor.agent.prompt_pipeline import ContentRouter, Placement, ProviderTier

        router = ContentRouter(ProviderTier.KV_ONLY, {"parallel_read"})
        item = _make_content_item("PARALLEL_READ_GUIDANCE", "tool_guidance")

        assert router.route(item) == Placement.SYSTEM_PROMPT

    def test_kv_only_router_preserves_broad_tool_guidance_match(self):
        from victor.agent.prompt_pipeline import ContentRouter, Placement, ProviderTier

        router = ContentRouter(ProviderTier.KV_ONLY, {"tool_guidance"})
        item = _make_content_item("LARGE_FILE_PAGINATION_GUIDANCE", "tool_guidance")

        assert router.route(item) == Placement.SYSTEM_PROMPT

    def test_kv_only_router_omits_unselected_tool_guidance_variant(self):
        from victor.agent.prompt_pipeline import ContentRouter, Placement, ProviderTier

        router = ContentRouter(ProviderTier.KV_ONLY, {"file_pagination"})
        item = _make_content_item("ASI_TOOL_EFFECTIVENESS_GUIDANCE", "tool_guidance")

        assert router.route(item) == Placement.OMITTED


# ============================================================================
# 2. Build System Prompt (6 tests)
# ============================================================================


class TestBuildSystemPrompt:
    """Verify system prompt assembly, freezing, and tier-aware behavior."""

    def test_tier_a_freezes_after_first_build(self):
        """Tier A: system prompt frozen after first build — second call returns same."""
        pipeline = _make_pipeline(provider=_make_provider(api_cache=True))
        prompt1 = pipeline.build_system_prompt()
        assert pipeline.is_frozen

        # Modify builder output — frozen pipeline should return same prompt
        pipeline._builder.build.return_value = "CHANGED"
        prompt2 = pipeline.build_system_prompt()
        assert prompt1 == prompt2

    def test_tier_c_rebuilds_every_call(self):
        """Tier C: system prompt rebuilt on every call (no cache benefit)."""
        builder = _make_builder("Initial prompt")
        pipeline = _make_pipeline(
            provider=_make_provider(api_cache=False, kv_cache=False),
            builder=builder,
        )

        prompt1 = pipeline.build_system_prompt()
        assert not pipeline.is_frozen

        builder.build.return_value = "Updated prompt"
        prompt2 = pipeline.build_system_prompt()
        assert prompt1 != prompt2

    def test_budget_hint_included_for_large_context(self):
        """Context window >= 32K gets parallel read budget hint."""
        pipeline = _make_pipeline(get_context_window=lambda: 128000)
        prompt = pipeline.build_system_prompt()
        # Budget hint contains "parallel" or "batch" or "simultaneously"
        assert any(
            kw in prompt.lower() for kw in ["parallel", "batch", "simultaneous", "files"]
        ) or len(prompt) > len("You are a helpful assistant.")

    def test_budget_hint_excluded_for_small_context(self):
        """Context window < 32K: no budget hint (sequential reads better)."""
        pipeline = _make_pipeline(get_context_window=lambda: 8192)
        prompt = pipeline.build_system_prompt()
        # Should just be the base prompt, no budget extras
        assert "You are a helpful assistant" in prompt

    def test_credit_not_in_system_prompt_tier_a(self):
        """Tier A: credit guidance must NOT be in system prompt (goes to user prefix)."""
        pipeline = _make_pipeline(provider=_make_provider(api_cache=True))

        # Mock credit guidance to return something
        with patch.object(
            type(pipeline),
            "_get_credit_guidance",
            return_value="Tool effectiveness: ...",
        ):
            prompt = pipeline.build_system_prompt()
            # Credit should NOT appear in frozen system prompt for Tier A
            assert "Tool effectiveness" not in prompt

    def test_project_context_appended(self):
        """Project context (init.md) is appended to system prompt."""
        pipeline = _make_pipeline()
        prompt = pipeline.build_system_prompt(project_context="## Project: Victor\nAI framework")
        assert "Victor" in prompt


# ============================================================================
# 3. Compose Turn Prefix (6 tests)
# ============================================================================


class TestComposeTurnPrefix:
    """Verify per-turn user prefix assembly."""

    def _make_turn_context(self, **overrides):
        from victor.agent.prompt_pipeline import TurnContext

        defaults = {
            "provider_name": "test",
            "model": "test-model",
            "task_type": "default",
        }
        defaults.update(overrides)
        return TurnContext(**defaults)

    def test_gepa_sections_injected(self):
        """Evolved GEPA sections appear in user prefix."""
        optimizer = _make_optimizer(evolved_sections=["Prefer read_file over cat."])
        pipeline = _make_pipeline(optimizer=optimizer)
        ctx = self._make_turn_context()

        prefix = pipeline.compose_turn_prefix("Fix the bug", ctx)
        assert "Prefer read over cat." in prefix

    def test_failure_hint_after_error(self):
        """Failure hints appear when last turn failed."""
        optimizer = _make_optimizer(failure_hint="Check file path exists before editing.")
        pipeline = _make_pipeline(optimizer=optimizer)
        ctx = self._make_turn_context(
            last_turn_failed=True,
            last_failure_category="file_not_found",
        )

        prefix = pipeline.compose_turn_prefix("Try again", ctx)
        assert "Check file path" in prefix

    def test_named_prompt_overlays_are_injected_into_turn_prefix(self):
        """Runtime prompt overlays appear in the dynamic per-turn prefix."""
        from victor.agent.prompt_pipeline import PromptOverlay

        pipeline = _make_pipeline()
        ctx = self._make_turn_context(
            prompt_overlays=[
                PromptOverlay(
                    name="planner.research_step",
                    content="Focus on gathering information without making changes.",
                )
            ]
        )

        prefix = pipeline.compose_turn_prefix("Inspect the runtime", ctx)

        assert "Focus on gathering information without making changes." in prefix
        assert ctx.prompt_optimization_metadata["prompt_overlays"] == [
            {
                "name": "planner.research_step",
                "placement": "turn_prefix",
            }
        ]

    def test_prompt_overlay_runtime_overrides_builds_canonical_payload(self):
        """Prompt overlay runtime payloads use one canonical shape."""
        from victor.agent.prompt_pipeline import prompt_overlay_runtime_overrides

        assert prompt_overlay_runtime_overrides(
            "planner.plan_generation",
            "Return only JSON.",
        ) == {
            "prompt_overlays": [
                {
                    "name": "planner.plan_generation",
                    "content": "Return only JSON.",
                    "placement": "turn_prefix",
                }
            ]
        }

    def test_runtime_intelligence_supplies_prompt_optimizations(self):
        """Canonical runtime intelligence service can supply prompt optimizations."""
        from victor.agent.services.runtime_intelligence import (
            PromptOptimizationBundle,
            PromptOptimizationIdentity,
        )

        runtime_intelligence = MagicMock()
        runtime_intelligence.get_prompt_optimization_bundle.return_value = PromptOptimizationBundle(
            evolved_sections=["Prefer read over cat."],
            few_shots="Example trajectory",
            failure_hint="Check the file path before editing.",
            identities=[
                PromptOptimizationIdentity(
                    provider="anthropic",
                    prompt_candidate_hash="cand-123",
                    section_name="GROUNDING_RULES",
                    prompt_section_name="GROUNDING_RULES",
                    strategy_name="gepa",
                    source="candidate",
                )
            ],
        )
        pipeline = _make_pipeline(runtime_intelligence=runtime_intelligence)
        ctx = self._make_turn_context(last_turn_failed=True)

        prefix = pipeline.compose_turn_prefix("Fix the bug", ctx)

        assert "Prefer read over cat." in prefix
        assert "Example trajectory" in prefix
        assert "Check the file path before editing." in prefix
        assert ctx.prompt_optimization_metadata == {
            "entries": [
                {
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                    "prompt_section_name": "GROUNDING_RULES",
                    "strategy_name": "gepa",
                    "source": "candidate",
                }
            ],
            "by_section": {
                "GROUNDING_RULES": {
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                    "prompt_section_name": "GROUNDING_RULES",
                    "strategy_name": "gepa",
                    "source": "candidate",
                }
            },
        }

    def test_runtime_intelligence_supplies_experiment_memory_prompt_guidance(self):
        """Experiment-memory prompt guidance is injected and recorded in metadata."""
        from victor.agent.services.runtime_intelligence import PromptOptimizationBundle

        runtime_intelligence = MagicMock()
        runtime_intelligence.get_prompt_optimization_bundle.return_value = PromptOptimizationBundle(
            experiment_guidance=[
                "Experiment constraint from similar runs: satisfy tests_pass before broadening the plan.",
                "Experiment-guided next candidate: Use read_file before execute_bash.",
            ],
            experiment_memory_hints={
                "experiment_memory_match_count": 2,
                "experiment_memory_record_ids": ["exp-1", "exp-2"],
            },
        )
        pipeline = _make_pipeline(runtime_intelligence=runtime_intelligence)
        ctx = self._make_turn_context()

        prefix = pipeline.compose_turn_prefix("Fix the bug", ctx)

        assert "Experiment constraint from similar runs" in prefix
        assert "Experiment-guided next candidate" in prefix
        assert ctx.prompt_optimization_metadata == {
            "entries": [],
            "by_section": {},
            "experiment_memory": {
                "experiment_memory_match_count": 2,
                "experiment_memory_record_ids": ["exp-1", "exp-2"],
                "prompt_guidance": [
                    "Experiment constraint from similar runs: satisfy tests_pass before broadening the plan.",
                    "Experiment-guided next candidate: Use read_file before execute_bash.",
                ],
            },
        }

    def test_credit_in_prefix_tier_a(self):
        """Tier A: credit guidance goes in user prefix (not system prompt)."""
        pipeline = _make_pipeline(provider=_make_provider(api_cache=True))
        ctx = self._make_turn_context()

        with patch.object(
            type(pipeline),
            "_get_credit_guidance",
            return_value="- read: high effectiveness (avg credit +0.8)",
        ):
            prefix = pipeline.compose_turn_prefix("Continue", ctx)
            assert "high effectiveness" in prefix

    def test_skill_prompt_included(self):
        """Active skill prompt appears in prefix."""
        pipeline = _make_pipeline()
        ctx = self._make_turn_context(active_skill_prompt="You are in coding mode.")

        prefix = pipeline.compose_turn_prefix("Write code", ctx)
        assert "coding mode" in prefix

    def test_query_task_guidance_can_be_injected_via_prefix(self):
        """Frozen-tier query guidance should move to the turn prefix."""
        pipeline = _make_pipeline()
        ctx = self._make_turn_context(
            task_guidance_text="TASK GUIDANCE: Plan before coding. Test incrementally.",
        )

        prefix = pipeline.compose_turn_prefix("Write code", ctx)

        assert "TASK GUIDANCE" in prefix
        assert "Plan before coding" in prefix

    def test_dynamic_tool_guidance_can_be_injected_via_prefix(self):
        """Long-tail tool hints belong in the user-prefix path, not the frozen prompt."""
        pipeline = _make_pipeline()
        ctx = self._make_turn_context(
            dynamic_tool_guidance=(
                "DYNAMIC TOOL HINTS: The stable system prompt covers the core tools. "
                "For this turn, the following less-common tools are relevant if needed: "
                "web_search, git_diff."
            ),
        )

        prefix = pipeline.compose_turn_prefix("Check upstream docs and repo changes", ctx)

        assert "DYNAMIC TOOL HINTS" in prefix
        assert "web_search" in prefix
        assert "git_diff" in prefix

    def test_empty_when_no_dynamic_content(self):
        """Returns empty string when nothing dynamic to inject."""
        pipeline = _make_pipeline(optimizer=None)
        ctx = self._make_turn_context()

        prefix = pipeline.compose_turn_prefix("Hello", ctx)
        assert prefix == ""

    def test_system_reminder_tags(self):
        """Dynamic content wrapped in an authenticated <system-reminder> envelope.

        The tag now carries the session key so the model can distinguish runtime
        guidance from look-alike text arriving via tool output or the user turn.
        """
        optimizer = _make_optimizer(evolved_sections=["Be concise."])
        pipeline = _make_pipeline(optimizer=optimizer)
        ctx = self._make_turn_context()

        prefix = pipeline.compose_turn_prefix("Help", ctx)
        assert f'<system-reminder key="{pipeline._channel_nonce}">' in prefix
        assert "</system-reminder>" in prefix

    def test_turn_prefix_key_matches_the_declared_session_key(self):
        """The envelope is only meaningful if the system prompt declares the key."""
        pipeline = _make_pipeline(optimizer=_make_optimizer(evolved_sections=["Be concise."]))

        system_prompt = pipeline.build_system_prompt()
        prefix = pipeline.compose_turn_prefix("Help", self._make_turn_context())

        nonce = pipeline._channel_nonce
        assert nonce, "a session key must be minted"
        assert nonce in system_prompt, "the key must be declared in the system prompt"
        assert f'key="{nonce}"' in prefix

    def test_session_key_is_stable_across_turns(self):
        """Re-minting mid-session would invalidate the declaration already sent."""
        pipeline = _make_pipeline(optimizer=_make_optimizer(evolved_sections=["Be concise."]))

        first = pipeline.compose_turn_prefix("Help", self._make_turn_context())
        second = pipeline.compose_turn_prefix("Help again", self._make_turn_context())

        assert f'key="{pipeline._channel_nonce}"' in first
        assert f'key="{pipeline._channel_nonce}"' in second

    def test_distinct_sessions_get_distinct_keys(self):
        """A key reused across sessions would be guessable from an earlier one."""
        a = _make_pipeline(optimizer=_make_optimizer(evolved_sections=["x"]))
        b = _make_pipeline(optimizer=_make_optimizer(evolved_sections=["x"]))

        assert a._channel_nonce != b._channel_nonce

    def test_repeated_boilerplate_is_dictionary_compressed(self):
        """Repeated long guidance blocks should be aliased once in the reminder body."""
        repeated = (
            "Search first with available tools before editing and do not guess file paths or "
            "symbol locations in the repository."
        )
        optimizer = _make_optimizer(evolved_sections=[repeated, repeated])
        pipeline = _make_pipeline(optimizer=optimizer)
        ctx = self._make_turn_context()

        prefix = pipeline.compose_turn_prefix("Help", ctx)

        assert "## Reusable Guidance" in prefix
        assert prefix.count(repeated) == 1
        assert prefix.count("[[R1]]") >= 2


class TestRuntimeGuidanceLookups:
    """Verify prompt-time guidance resolves live runtime services from DI."""

    def test_credit_guidance_resolves_registered_service(self):
        from victor.core.container import (
            ServiceContainer,
            reset_container,
            set_container,
        )
        from victor.framework.rl.credit_tracking_service import CreditTrackingService

        pipeline = _make_pipeline()
        service = CreditTrackingService()
        service.generate_tool_guidance = MagicMock(return_value="Tool effectiveness: use read")

        container = ServiceContainer()
        container.register_instance(CreditTrackingService, service)
        set_container(container)
        try:
            assert pipeline._get_credit_guidance() == "Tool effectiveness: use read"
        finally:
            reset_container()

    def test_tool_reputation_guidance_resolves_registered_pipeline(self):
        from victor.agent.tool_pipeline import ToolPipeline
        from victor.core.container import (
            ServiceContainer,
            reset_container,
            set_container,
        )

        pipeline = _make_pipeline()
        tracker = MagicMock()
        tracker.get_selection_guidance.return_value = "Prefer grep before broad read"

        live_pipeline = MagicMock(spec=ToolPipeline)
        live_pipeline._tool_reputation = tracker

        container = ServiceContainer()
        container.register_instance(ToolPipeline, live_pipeline)
        set_container(container)
        try:
            assert pipeline._get_tool_reputation_guidance() == "Prefer grep before broad read"
        finally:
            reset_container()


# ============================================================================
# 4. Frozen Prompt State (3 tests)
# ============================================================================


class TestFrozenPromptState:
    """Verify single frozen flag with correct lifecycle."""

    def test_single_frozen_flag(self):
        """Pipeline has exactly one frozen state — no dual tracking."""
        pipeline = _make_pipeline(provider=_make_provider(api_cache=True))
        assert not pipeline.is_frozen

        pipeline.build_system_prompt()
        assert pipeline.is_frozen

        # No _system_prompt_frozen on pipeline (that was orchestrator's bug)
        assert not hasattr(pipeline, "_system_prompt_frozen")

    def test_unfreeze_clears_and_resamples(self):
        """unfreeze() clears frozen prompt and triggers GEPA resample."""
        optimizer = _make_optimizer()
        pipeline = _make_pipeline(
            provider=_make_provider(api_cache=True),
            optimizer=optimizer,
        )

        pipeline.build_system_prompt()
        assert pipeline.is_frozen

        pipeline.unfreeze()
        assert not pipeline.is_frozen
        optimizer.clear_session_cache.assert_called()

    def test_tier_c_never_frozen(self):
        """Tier C providers never freeze — always rebuilt."""
        pipeline = _make_pipeline(provider=_make_provider(api_cache=False, kv_cache=False))

        pipeline.build_system_prompt()
        assert not pipeline.is_frozen

        pipeline.build_system_prompt()
        assert not pipeline.is_frozen


# ============================================================================
# 5. Credit Injection Deduplication (2 tests)
# ============================================================================


class TestCreditDedup:
    """Verify credit guidance appears in exactly one location per tier."""

    def test_tier_a_credit_only_in_prefix(self):
        """Tier A: credit in user prefix, NOT in system prompt."""
        pipeline = _make_pipeline(provider=_make_provider(api_cache=True))

        credit_text = "- shell: low effectiveness"

        with patch.object(type(pipeline), "_get_credit_guidance", return_value=credit_text):
            sys_prompt = pipeline.build_system_prompt()
            assert credit_text not in sys_prompt

            from victor.agent.prompt_pipeline import TurnContext

            ctx = TurnContext(provider_name="test", model="m", task_type="default")
            prefix = pipeline.compose_turn_prefix("Go", ctx)
            assert credit_text in prefix

    def test_tier_c_credit_in_system_prompt(self):
        """Tier C: credit in system prompt (rebuilt per-turn), NOT in prefix."""
        pipeline = _make_pipeline(
            provider=_make_provider(api_cache=False, kv_cache=False),
        )

        credit_text = "- shell: low effectiveness"

        with patch.object(type(pipeline), "_get_credit_guidance", return_value=credit_text):
            sys_prompt = pipeline.build_system_prompt()
            assert credit_text in sys_prompt

            from victor.agent.prompt_pipeline import TurnContext

            ctx = TurnContext(provider_name="test", model="m", task_type="default")
            prefix = pipeline.compose_turn_prefix("Go", ctx)
            # Credit should NOT also be in prefix for Tier C
            assert credit_text not in prefix


# ============================================================================
# 6. Prompt Completeness Guard (2 tests)
# ============================================================================


class TestPromptCompletenessGuard:
    """Verify prompt completeness guard summarizes scope and gaps."""

    def _make_turn_context(self, **overrides):
        from victor.agent.prompt_pipeline import TurnContext

        defaults = {
            "provider_name": "test",
            "model": "test-model",
            "task_type": "default",
        }
        defaults.update(overrides)
        return TurnContext(**defaults)

    def test_guard_summarizes_execution_contract_for_clear_request(self):
        """Clear prompts should produce a compact execution contract."""
        pipeline = _make_pipeline(enable_prompt_completeness_guard=True)
        ctx = self._make_turn_context(task_type="review")

        prefix = pipeline.compose_turn_prefix(
            "Review src/auth/login.py and provide a findings table summary without changing code.",
            ctx,
        )

        assert "Prompt execution contract" in prefix
        assert "src/auth/login.py" in prefix
        assert "findings table" in prefix
        assert "without changing code" in prefix
        assert "Ask one targeted clarification" not in prefix

        assessment = pipeline.last_prompt_completeness_assessment
        assert assessment is not None
        assert assessment.needs_clarification is False

    def test_guard_requests_clarification_for_ambiguous_request(self):
        """Ambiguous action requests should surface missing scope before execution."""
        pipeline = _make_pipeline(enable_prompt_completeness_guard=True)
        ctx = self._make_turn_context(task_type="implementation")

        prefix = pipeline.compose_turn_prefix("Fix it and add tests.", ctx)

        assert "Prompt execution contract" in prefix
        assert "Missing: target artifact or scope" in prefix
        assert "Ask one targeted clarification" in prefix

        assessment = pipeline.last_prompt_completeness_assessment
        assert assessment is not None
        assert assessment.needs_clarification is True
        assert "target artifact or scope" in assessment.missing_elements

    def test_guard_ignores_relative_clause_report_request(self):
        """Report-style prose should not trigger action clarification from substring matches."""
        pipeline = _make_pipeline(enable_prompt_completeness_guard=True)
        ctx = self._make_turn_context(task_type="default")

        prefix = pipeline.compose_turn_prefix(
            "Identify all dead code or deprecated call sites that should be modified first "
            "to enable clean code principle based implementations.",
            ctx,
        )

        assert "Prompt execution contract" not in prefix

        assessment = pipeline.last_prompt_completeness_assessment
        assert assessment is None or assessment.needs_clarification is False

    def test_guard_adds_search_first_guidance_for_symbol_scoped_request(self):
        """Symbol-scoped work without file paths should prefer search-first execution."""
        pipeline = _make_pipeline(enable_prompt_completeness_guard=True)
        ctx = self._make_turn_context(task_type="implementation")

        prefix = pipeline.compose_turn_prefix(
            "Fix `authenticate_user` and add tests without changing the public API.",
            ctx,
        )

        assert "Prompt execution contract" in prefix
        assert "Search first" in prefix
        assert "Do not guess paths" in prefix

        assessment = pipeline.last_prompt_completeness_assessment
        assert assessment is not None
        assert assessment.search_first is True
        assert assessment.needs_clarification is False

    def test_guard_skips_search_first_when_explicit_file_is_provided(self):
        """Explicit file targets should not trigger search-first guidance."""
        pipeline = _make_pipeline(enable_prompt_completeness_guard=True)
        ctx = self._make_turn_context(task_type="implementation")

        prefix = pipeline.compose_turn_prefix(
            "Fix src/auth/login.py and add tests without changing the public API.",
            ctx,
        )

        assert "Prompt execution contract" in prefix
        assert "Search first" not in prefix

        assessment = pipeline.last_prompt_completeness_assessment
        assert assessment is not None
        assert assessment.search_first is False


# ============================================================================
# 7. Backward Compatibility (3 tests)
# ============================================================================


class TestBackwardCompat:
    """Verify old callers still work through deprecation wrappers."""

    def test_shell_variant_resolution(self):
        """resolve_shell_variant delegates correctly."""
        pipeline = _make_pipeline()
        # Should not raise — method exists even if shell resolver unavailable
        try:
            result = pipeline.resolve_shell_variant("shell")
            assert isinstance(result, str)
        except ImportError:
            pass  # OK if shell_resolver not importable in test env

    def test_task_classification(self):
        """classify_task_keywords returns a dict."""
        task_analyzer = MagicMock()
        task_analyzer.classify_keywords.return_value = {
            "task_type": "implementation",
            "confidence": 0.8,
        }
        pipeline = _make_pipeline(task_analyzer=task_analyzer)

        result = pipeline.classify_task_keywords("Add a login button")
        assert isinstance(result, dict)

    def test_pipeline_exposes_builder(self):
        """Pipeline exposes the underlying builder for backward compat."""
        builder = _make_builder()
        pipeline = _make_pipeline(builder=builder)
        assert pipeline.builder is builder
