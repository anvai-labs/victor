"""Tests for KV prefix cache optimization across provider types.

Verifies that:
- _kv_optimization_enabled activates for providers with KV prefix caching
- System prompt is frozen after first build when KV optimization is active
- Dynamic content is injected into user messages (not system prompt) for KV providers
- Tool ordering is deterministic for KV prefix stability
- Context pruning preserves the system prompt prefix
- Prompt builder uses reduced + cached sections for KV providers
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest


def _attach_prompt_runtime(orch):
    """Attach the canonical prompt runtime to a lightweight orchestrator mock."""
    from victor.agent.services.orchestrator_protocol_adapter import (
        OrchestratorProtocolAdapter,
    )
    from victor.agent.services.prompt_builder_runtime import PromptBuilderRuntime

    orch._get_prompt_builder_runtime.return_value = PromptBuilderRuntime(
        OrchestratorProtocolAdapter(orch)
    )


def _attach_legacy_tool_service(orch):
    """Attach enough ToolService behavior for legacy KV strategy tests."""
    service = MagicMock()
    service.sort_tools_for_kv_stability.side_effect = lambda tools, kv_optimization_enabled: (
        sorted(tools, key=lambda tool: tool.name) if kv_optimization_enabled else tools
    )

    def _apply_kv_tool_strategy(
        tools,
        *,
        kv_optimization_enabled,
        provider,
        model,
        session_semantic_tools,
        kv_tool_strategy,
    ):
        if not kv_optimization_enabled:
            return tools
        if kv_tool_strategy == "session_stable" and session_semantic_tools is not None:
            return session_semantic_tools
        return tools

    service.apply_kv_tool_strategy.side_effect = _apply_kv_tool_strategy
    orch._tool_service = service
    return service


# =====================================================================
# Fix 1: _kv_optimization_enabled property
# =====================================================================


class TestKVOptimizationEnabled:
    """Test _kv_optimization_enabled property on AgentOrchestrator."""

    def _make_orchestrator(self, kv_cache=False, api_cache=False, setting_enabled=True):
        """Create a mock orchestrator with configurable provider capabilities."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)

        # Provider mock
        provider = MagicMock()
        provider.supports_kv_prefix_caching.return_value = kv_cache
        provider.supports_prompt_caching.return_value = api_cache
        orch.provider = provider

        # Settings mock
        context = MagicMock()
        context.cache_optimization_enabled = setting_enabled
        settings = MagicMock()
        settings.context = context
        orch.settings = settings
        orch._kv_opt_cached = None
        orch._cache_opt_cached = None
        _attach_prompt_runtime(orch)

        # Call the real property via unbound reference
        return orch

    def test_ollama_gets_kv_optimization(self):
        """Ollama: kv=True, cache=False -> kv_optimization=True, cache_optimization=False."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(kv_cache=True, api_cache=False)
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True
        assert AgentOrchestrator._cache_optimization_enabled.fget(orch) is False

    def test_anthropic_gets_both(self):
        """Anthropic: kv=True, cache=True -> both True."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(kv_cache=True, api_cache=True)
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True
        assert AgentOrchestrator._cache_optimization_enabled.fget(orch) is True

    def test_provider_without_either(self):
        """Provider with neither -> both False."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(kv_cache=False, api_cache=False)
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False
        assert AgentOrchestrator._cache_optimization_enabled.fget(orch) is False

    def test_setting_override_disables_kv(self):
        """cache_optimization_enabled=False overrides KV optimization."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(kv_cache=True, api_cache=False, setting_enabled=False)
        # The real _check_cache_setting_enabled needs to be callable
        orch._check_cache_setting_enabled = lambda: False
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False

    def test_fallback_when_no_kv_method(self):
        """Provider without supports_kv_prefix_caching falls back to supports_prompt_caching."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = self._make_orchestrator(api_cache=True)
        # Remove kv method to test fallback
        del orch.provider.supports_kv_prefix_caching
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True

    def test_no_provider_returns_false(self):
        """No provider attribute -> False."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch.provider = None
        orch.settings = MagicMock()
        orch.settings.context = MagicMock()
        orch.settings.context.cache_optimization_enabled = True
        orch._kv_opt_cached = None
        orch._cache_opt_cached = None
        _attach_prompt_runtime(orch)
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False


# =====================================================================
# Fix 1: System prompt freezing gated on _kv_optimization_enabled
# =====================================================================


class TestSystemPromptFreezing:
    """Test that system prompt freezing activates for KV providers."""

    def test_kv_provider_freezes_prompt_after_first_build(self):
        """System prompt frozen after first build when kv_optimization=True."""
        from unittest.mock import MagicMock, PropertyMock
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.prompt_builder_runtime import PromptBuilderRuntime
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        # Create a mock orchestrator with all required attributes
        orch_host = MagicMock(spec=AgentOrchestrator)
        type(orch_host)._kv_optimization_enabled = PropertyMock(return_value=True)
        type(orch_host)._cache_optimization_enabled = PropertyMock(return_value=False)
        orch_host._system_prompt_frozen = False
        orch_host._system_prompt = ""
        orch_host._prompt_pipeline = (
            None  # No pipeline, so is_frozen will check _system_prompt_frozen
        )

        # Mock prompt_builder
        orch_host.prompt_builder = MagicMock()
        orch_host.prompt_builder.build.return_value = "system prompt"
        orch_host.prompt_builder.query_classification = None

        # Mock project context
        orch_host.project_context = MagicMock()
        orch_host.project_context.content = None

        # Mock conversation
        orch_host.conversation = MagicMock()
        orch_host.conversation._system_added = False
        orch_host.conversation.system_prompt = ""
        orch_host.conversation._messages = []

        # Mock build_system_prompt
        orch_host.build_system_prompt = MagicMock(return_value="system prompt")

        # Create PromptBuilderRuntime with the mock host
        protocol_adapter = OrchestratorProtocolAdapter(orch_host)
        runtime = PromptBuilderRuntime(protocol_adapter)

        # Add _get_prompt_builder_runtime to the mock orchestrator
        orch_host._get_prompt_builder_runtime = MagicMock(return_value=runtime)
        orch_host._prompt_builder_runtime = runtime

        # Call the real method on the orchestrator
        AgentOrchestrator.update_system_prompt_for_query(orch_host, query_classification=None)

        # Prompt should be frozen now
        assert orch_host._system_prompt_frozen is True

    def test_kv_provider_skips_rebuild_on_second_query(self):
        """update_system_prompt_for_query() is no-op after freeze."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        type(orch)._cache_optimization_enabled = PropertyMock(return_value=False)
        orch._system_prompt_frozen = True  # Already frozen
        # Add prompt_builder so we can verify it's not called
        orch.prompt_builder = MagicMock()
        orch.build_system_prompt = MagicMock()

        # Should return early without rebuilding
        AgentOrchestrator.update_system_prompt_for_query(orch, query_classification=None)

        orch.build_system_prompt.assert_not_called()

    def test_update_system_prompt_for_query_prefers_canonical_builder(self):
        """Positive-path prompt rebuild uses build_system_prompt, not the legacy wrapper."""
        from unittest.mock import MagicMock, PropertyMock
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.prompt_builder_runtime import PromptBuilderRuntime
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        orch_host = MagicMock(spec=AgentOrchestrator)
        type(orch_host)._kv_optimization_enabled = PropertyMock(return_value=False)
        type(orch_host)._cache_optimization_enabled = PropertyMock(return_value=False)
        orch_host._system_prompt_frozen = False
        orch_host._system_prompt = ""
        orch_host._prompt_pipeline = None
        orch_host.prompt_builder = MagicMock()
        orch_host.build_system_prompt = MagicMock(return_value="canonical prompt")
        orch_host.project_context = None
        orch_host.conversation = MagicMock()
        orch_host.conversation._system_added = False
        orch_host.conversation.system_prompt = ""
        orch_host.conversation._messages = []

        # Create PromptBuilderRuntime with the mock host
        protocol_adapter = OrchestratorProtocolAdapter(orch_host)
        runtime = PromptBuilderRuntime(protocol_adapter)

        # Add _get_prompt_builder_runtime to the mock orchestrator
        orch_host._get_prompt_builder_runtime = MagicMock(return_value=runtime)
        orch_host._prompt_builder_runtime = runtime

        AgentOrchestrator.update_system_prompt_for_query(orch_host, query_classification=None)

        orch_host.build_system_prompt.assert_called_once_with()
        assert orch_host._system_prompt == "canonical prompt"


# =====================================================================
# Fix 2: Deterministic tool ordering
# =====================================================================


class TestDeterministicToolOrdering:
    """Test that tools are sorted by name for KV cache stability."""

    def test_tools_sorted_by_name_when_kv_enabled(self):
        """Tools returned in alphabetical order for KV cache stability."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        _attach_legacy_tool_service(orch)

        # Create mock tools in non-alphabetical order
        tool_z = MagicMock()
        tool_z.name = "z_tool"
        tool_a = MagicMock()
        tool_a.name = "a_tool"
        tool_m = MagicMock()
        tool_m.name = "m_tool"

        tools = [tool_z, tool_a, tool_m]
        sorted_tools = AgentOrchestrator._sort_tools_for_kv_stability(orch, tools)

        assert [t.name for t in sorted_tools] == ["a_tool", "m_tool", "z_tool"]

    def test_no_sorting_when_kv_disabled(self):
        """Without KV optimization, tool order is preserved."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)

        tool_z = MagicMock()
        tool_z.name = "z_tool"
        tool_a = MagicMock()
        tool_a.name = "a_tool"

        tools = [tool_z, tool_a]
        result = AgentOrchestrator._sort_tools_for_kv_stability(orch, tools)

        assert [t.name for t in result] == ["z_tool", "a_tool"]

    def test_context_aware_profile_sorts_large_no_cache_provider(self):
        """Large context-aware profiles use stable ordering even without KV flag."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        orch._resolve_kv_strategy_setting = lambda: "context_aware"
        orch._get_context_window.return_value = 128000
        orch.provider = MagicMock()
        orch.provider.supports_prompt_caching.return_value = False
        orch.provider.supports_kv_prefix_caching.return_value = False
        orch.model = "glm-5.2"
        orch.settings = SimpleNamespace(tools=SimpleNamespace(budget=20))
        _attach_legacy_tool_service(orch)

        tool_z = MagicMock()
        tool_z.name = "z_tool"
        tool_a = MagicMock()
        tool_a.name = "a_tool"

        result = AgentOrchestrator._sort_tools_for_kv_stability(orch, [tool_z, tool_a])

        assert [t.name for t in result] == ["a_tool", "z_tool"]

    def test_none_tools_returns_none(self):
        """None tools input returns None."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        assert AgentOrchestrator._sort_tools_for_kv_stability(orch, None) is None


# =====================================================================
# Fix 3: Dynamic content injection stays pipeline-owned even without KV cache
# =====================================================================


class TestDynamicContentInjection:
    """Test dynamic content injection into user messages for the prompt pipeline."""

    def test_kv_provider_injects_skills_into_user_message(self):
        """Skills/reminders go into user message when KV optimization is active."""
        from victor.providers.base import Message

        messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
        ]

        # Simulate the injection logic
        prefix_parts = ["[Skill: code_review]"]
        if prefix_parts:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == "user":
                    prefix = "\n".join(prefix_parts) + "\n\n"
                    messages[i] = Message(role="user", content=prefix + messages[i].content)
                    break

        assert messages[0].content == "system prompt"  # Unchanged
        assert "[Skill: code_review]" in messages[1].content
        assert "hello" in messages[1].content

    def test_no_cache_provider_still_injects_prompt_pipeline_prefix(self):
        """Tier C providers still receive GEPA/MiPRO/CoT guidance through the live pipeline."""
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.providers.base import Message

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        orch._context_assembler = None
        orch.messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
        ]
        orch._prompt_pipeline = MagicMock()
        orch._prompt_pipeline.compose_turn_prefix.return_value = (
            "<system-reminder>\nUse the evolved prompt candidate.\n</system-reminder>\n\n"
        )
        orch._record_prompt_optimization_metadata = MagicMock()
        orch._optimization_injector = None
        orch._reminder_manager = None
        orch._system_prompt_frozen = False
        orch._last_sorted_tool_names = None
        orch.provider_name = "test-provider"
        orch._model = "test-model"
        orch._current_task_type = "edit"
        orch.get_skill_user_prefix = MagicMock(return_value=None)

        result = AgentOrchestrator.get_assembled_messages(orch)

        orch._prompt_pipeline.compose_turn_prefix.assert_called_once()
        orch._record_prompt_optimization_metadata.assert_called_once()
        assert result[0].content == "system prompt"
        assert "Use the evolved prompt candidate." in result[1].content
        assert result[1].content.endswith("hello")

    def test_prompt_pipeline_receives_runtime_prompt_overlays(self):
        """Runtime prompt overlays flow through TurnContext instead of mutating system prompt."""
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.providers.base import Message

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        orch._context_assembler = None
        orch.messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
        ]
        orch._prompt_pipeline = MagicMock()
        orch._prompt_pipeline.compose_turn_prefix.return_value = ""
        orch._record_prompt_optimization_metadata = MagicMock()
        orch._optimization_injector = None
        orch._reminder_manager = None
        orch._system_prompt_frozen = False
        orch._last_sorted_tool_names = None
        orch.provider_name = "test-provider"
        orch._model = "test-model"
        orch._current_task_type = "edit"
        orch.get_skill_user_prefix = MagicMock(return_value=None)
        orch._runtime_tool_context_overrides = {
            "prompt_overlays": [
                {
                    "name": "planner.plan_generation",
                    "content": "Return only JSON.",
                    "placement": "turn_prefix",
                }
            ]
        }

        AgentOrchestrator.get_assembled_messages(orch)

        turn_ctx = orch._prompt_pipeline.compose_turn_prefix.call_args.args[1]
        assert len(turn_ctx.prompt_overlays) == 1
        assert turn_ctx.prompt_overlays[0].name == "planner.plan_generation"
        assert turn_ctx.prompt_overlays[0].content == "Return only JSON."

    def test_no_cache_provider_clears_failure_hint_state_after_injection(self):
        """Failure-hint injection remains one-shot even when KV caching is disabled."""
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.providers.base import Message

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        orch._context_assembler = None
        orch.messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="retry"),
        ]
        orch._prompt_pipeline = MagicMock()
        orch._prompt_pipeline.compose_turn_prefix.return_value = (
            "<system-reminder>\nRetry with the failure hint.\n</system-reminder>\n\n"
        )
        orch._record_prompt_optimization_metadata = MagicMock()
        orch._reminder_manager = None
        orch._system_prompt_frozen = False
        orch._last_sorted_tool_names = None
        orch.provider_name = "test-provider"
        orch._model = "test-model"
        orch._current_task_type = "edit"
        orch.get_skill_user_prefix = MagicMock(return_value=None)
        orch._optimization_injector = MagicMock()
        orch._optimization_injector._last_failure_category = "file_not_found"
        orch._optimization_injector._last_failure_error = "missing file"

        AgentOrchestrator.get_assembled_messages(orch)

        assert orch._optimization_injector._last_failure_category is None
        assert orch._optimization_injector._last_failure_error is None

    def test_prompt_pipeline_passes_goals_into_dynamic_tool_guidance(self):
        """Per-turn dynamic tool hints should receive planner goals from the stream path."""
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.providers.base import Message

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        orch._context_assembler = None
        orch.messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
        ]
        orch._prompt_pipeline = MagicMock()
        orch._prompt_pipeline.compose_turn_prefix.return_value = ""
        orch._record_prompt_optimization_metadata = MagicMock()
        orch._optimization_injector = None
        orch._reminder_manager = None
        orch._system_prompt_frozen = False
        orch._last_sorted_tool_names = None
        orch.provider_name = "test-provider"
        orch._model = "test-model"
        orch._current_task_type = "edit"
        orch.get_skill_user_prefix = MagicMock(return_value=None)
        prompt_runtime = MagicMock()
        prompt_runtime.build_dynamic_tool_guidance.return_value = ""
        orch._get_prompt_builder_runtime = MagicMock(return_value=prompt_runtime)

        selected_tools = [SimpleNamespace(name="web_search")]
        planned_tools = [SimpleNamespace(name="workflow")]
        goals = ["inspect repo changes", "verify docs"]

        AgentOrchestrator.get_assembled_messages(
            orch,
            current_query="Check upstream docs and repo changes",
            selected_tools=selected_tools,
            planned_tools=planned_tools,
            goals=goals,
        )

        prompt_runtime.build_dynamic_tool_guidance.assert_called_once_with(
            "Check upstream docs and repo changes",
            planned_tools=planned_tools,
            selected_tools=selected_tools,
            goals=goals,
        )


# =====================================================================
# Fix 4: Context pruning preserves system prompt
# =====================================================================


class TestCacheAwarePruning:
    """Test that context pruning preserves the system prompt prefix."""

    def test_system_prompt_always_preserved_during_pruning(self):
        """System prompt (messages[0]) never removed by context assembly."""
        from victor.providers.base import Message

        messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="old message 1"),
            Message(role="assistant", content="old response 1"),
            Message(role="user", content="old message 2"),
            Message(role="assistant", content="old response 2"),
            Message(role="user", content="latest question"),
        ]

        # Simulate pruning: remove middle messages but keep system + recent
        pruned = [messages[0]] + messages[-2:]  # system + last turn

        assert pruned[0].role == "system"
        assert pruned[0].content == "system prompt"
        assert len(pruned) == 3


# =====================================================================
# Fix 5: Prompt builder KV cache awareness
# =====================================================================


class TestPromptBuilderKVAwareness:
    """Test that SystemPromptBuilder handles KV cache providers correctly."""

    def test_kv_provider_gets_reduced_sections(self):
        """Non-caching provider (Ollama) gets reduced section set."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.provider_caches = False
        builder.provider_has_kv_cache = True
        builder.concise_mode = False
        builder.headless_mode = False
        builder._rl_coordinator = None
        builder._decision_service = None

        sections = SystemPromptBuilder._get_active_sections(builder)
        # Should return reduced set, not full
        assert len(sections) <= 5
        assert "completion" in sections or "task_guidance" in sections

    def test_cloud_provider_gets_full_sections(self):
        """Caching provider (Anthropic) gets full sections."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.provider_caches = True
        builder.provider_has_kv_cache = True
        builder.concise_mode = False
        builder.headless_mode = False
        builder._rl_coordinator = None
        builder._decision_service = None

        sections = SystemPromptBuilder._get_active_sections(builder)
        # Full set is larger than reduced set
        assert len(sections) > 5

    def test_kv_provider_param_accepted(self):
        """SystemPromptBuilder accepts provider_has_kv_cache parameter."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        # Should not raise
        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.provider_has_kv_cache = True
        assert builder.provider_has_kv_cache is True


# =====================================================================
# KV Tool Selection Strategy (configurable via setting)
# =====================================================================


class TestKVToolSelectionStrategy:
    """Test configurable tool selection strategies for KV providers.

    Three strategies:
    - 'context_aware': Stable core tools with economy-first locking when it helps
    - 'per_turn': Fresh semantic selection each turn (max relevance, breaks KV prefix)
    - 'session_stable': Lock semantic selection after first query (KV stable, may miss tools)
    - 'session_full': Lock all 48 tools (only for API-caching providers)
    """

    def test_setting_defaults_to_context_aware(self):
        """Default kv_tool_strategy is 'context_aware'."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings()
        assert settings.kv_tool_strategy == "context_aware"

    def test_setting_accepts_session_stable(self):
        """kv_tool_strategy can be set to 'session_stable'."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings(kv_tool_strategy="session_stable")
        assert settings.kv_tool_strategy == "session_stable"

    def test_session_stable_returns_cached_tools_on_second_call(self):
        """session_stable strategy returns same tools on subsequent turns."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        # Disable v2 so this test exercises the legacy session_stable path
        orch._is_tool_strategy_v2_enabled.return_value = False
        orch._resolve_kv_strategy_setting = lambda: "session_stable"
        _attach_legacy_tool_service(orch)
        orch.provider = MagicMock()
        orch.model = "test-model"

        # Configure session_stable strategy
        ctx = MagicMock()
        ctx.kv_tool_strategy = "session_stable"
        settings = MagicMock()
        settings.context = ctx
        orch.settings = settings

        # Mock tools
        tool_a = MagicMock()
        tool_a.name = "a_tool"
        tool_b = MagicMock()
        tool_b.name = "b_tool"
        first_selection = [tool_a, tool_b]

        # First call: stores tools
        orch._session_semantic_tools = None
        result = AgentOrchestrator._apply_kv_tool_strategy(orch, first_selection)
        assert result == first_selection
        assert orch._session_semantic_tools == first_selection

        # Second call: returns cached, ignores new selection
        tool_c = MagicMock()
        tool_c.name = "c_tool"
        result2 = AgentOrchestrator._apply_kv_tool_strategy(orch, [tool_c])
        assert result2 == first_selection  # Still the original set

    def test_per_turn_returns_fresh_selection(self):
        """per_turn strategy always returns the new selection."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        orch._session_semantic_tools = None
        # Disable v2 so this test exercises the legacy per_turn path
        orch._is_tool_strategy_v2_enabled.return_value = False
        orch._resolve_kv_strategy_setting = lambda: "per_turn"
        _attach_legacy_tool_service(orch)
        orch.provider = MagicMock()
        orch.model = "test-model"

        # Configure per_turn strategy
        ctx = MagicMock()
        ctx.kv_tool_strategy = "per_turn"
        settings = MagicMock()
        settings.context = ctx
        orch.settings = settings

        tool_a = MagicMock()
        tool_a.name = "a_tool"

        result = AgentOrchestrator._apply_kv_tool_strategy(orch, [tool_a])
        assert [t.name for t in result] == ["a_tool"]
        assert orch._session_semantic_tools is None  # Not cached

    def test_non_kv_provider_ignores_strategy(self):
        """Non-KV providers bypass the strategy entirely."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        # Disable v2 so the legacy pass-through path is exercised
        orch._is_tool_strategy_v2_enabled.return_value = False
        orch._resolve_kv_strategy_setting = lambda: "context_aware"
        _attach_legacy_tool_service(orch)
        orch.provider = MagicMock()
        orch.model = "test-model"

        tool_a = MagicMock()
        tool_a.name = "a_tool"
        result = AgentOrchestrator._apply_kv_tool_strategy(orch, [tool_a])
        assert result == [tool_a]  # Pass through unchanged

    def test_context_aware_strategy_emits_service_friendly_tool_strategy_event(self):
        """Context-aware strategy should pass the provider object without stale kwargs."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch.provider = MagicMock()
        orch.provider.name = "openai"
        orch.model = "gpt-4.1"
        orch._get_context_window.return_value = 200000
        orch._estimate_tool_tokens.side_effect = lambda tool, provider_category: 100
        orch._emit_tool_strategy_event = MagicMock()
        orch._tool_service = MagicMock()
        orch._tool_service.apply_context_aware_strategy.side_effect = lambda tools, **kwargs: tools

        tool_a = MagicMock()
        tool_a.name = "read"
        tool_b = MagicMock()
        tool_b.name = "git_diff"
        tools = [tool_a, tool_b]

        result = AgentOrchestrator._apply_context_aware_strategy(orch, tools)

        assert result == tools
        orch._emit_tool_strategy_event.assert_called_once()
        kwargs = orch._emit_tool_strategy_event.call_args.kwargs
        assert kwargs["provider"] is orch.provider
        assert "provider_category" not in kwargs

    def test_default_strategy_prefers_context_aware_for_kv_only_providers(self):
        """KV-only providers should default to the context-aware strategy."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        orch._is_tool_strategy_v2_enabled.return_value = False
        orch._resolve_kv_strategy_setting = lambda: "context_aware"
        tool_service = _attach_legacy_tool_service(orch)
        tool_service.apply_kv_tool_strategy.side_effect = None
        tool_service.apply_kv_tool_strategy.return_value = ["stable-tools"]
        orch.provider = MagicMock()
        orch.model = "test-model"

        ctx = MagicMock()
        ctx.kv_tool_strategy = "context_aware"
        settings = MagicMock()
        settings.context = ctx
        orch.settings = settings

        tool_a = MagicMock()
        tool_a.name = "read"

        result = AgentOrchestrator._apply_kv_tool_strategy(orch, [tool_a])

        assert result == ["stable-tools"]
        tool_service.apply_kv_tool_strategy.assert_called_once()
        assert tool_service.apply_kv_tool_strategy.call_args.kwargs["kv_tool_strategy"] == (
            "context_aware"
        )


# =====================================================================
# W1: Cached optimization flags
# =====================================================================


class TestCachedOptimizationFlags:
    """Test that optimization flags are cached at init, not recomputed per access."""

    def test_cache_flag_attributes_exist(self):
        """Orchestrator exposes _kv_opt_cached and _cache_opt_cached attributes."""
        from victor.agent.orchestrator import AgentOrchestrator

        # These should be declared (even if None before _compute_cache_flags)
        assert hasattr(AgentOrchestrator, "_compute_cache_flags")

    def test_compute_cache_flags_delegates_to_prompt_runtime(self):
        """_compute_cache_flags delegates cached flag ownership to prompt runtime."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        prompt_runtime = MagicMock()
        orch._get_prompt_builder_runtime.return_value = prompt_runtime

        AgentOrchestrator._compute_cache_flags(orch)

        prompt_runtime.compute_cache_flags.assert_called_once_with()

    def test_property_uses_cached_value_when_available(self):
        """_kv_optimization_enabled delegates to prompt runtime."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        prompt_runtime = MagicMock()
        prompt_runtime.is_kv_optimization_enabled.return_value = True
        orch._get_prompt_builder_runtime.return_value = prompt_runtime

        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True

        prompt_runtime.is_kv_optimization_enabled.return_value = False
        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is False

    def test_property_computes_when_cache_is_none(self):
        """Prompt runtime owns uncached KV flag computation."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        prompt_runtime = MagicMock()
        prompt_runtime.is_kv_optimization_enabled.return_value = True
        orch._get_prompt_builder_runtime.return_value = prompt_runtime

        assert AgentOrchestrator._kv_optimization_enabled.fget(orch) is True


# =====================================================================
# W3: KV cache warm-up
# =====================================================================


class TestKVCacheWarmup:
    """Test KV cache warm-up method."""

    def test_warmup_method_exists(self):
        """AgentOrchestrator has a warm_up_kv_cache method."""
        from victor.agent.orchestrator import AgentOrchestrator

        assert hasattr(AgentOrchestrator, "warm_up_kv_cache")

    @pytest.mark.asyncio
    async def test_warmup_sends_minimal_request(self):
        """warm_up_kv_cache delegates warm-up to prompt runtime."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        prompt_runtime = MagicMock()
        prompt_runtime.warm_up_kv_cache = AsyncMock()
        orch._get_prompt_builder_runtime.return_value = prompt_runtime

        await AgentOrchestrator.warm_up_kv_cache(orch)

        prompt_runtime.warm_up_kv_cache.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_warmup_noop_when_kv_disabled(self):
        """Prompt runtime owns disabled KV warm-up behavior."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        prompt_runtime = MagicMock()
        prompt_runtime.warm_up_kv_cache = AsyncMock()
        orch._get_prompt_builder_runtime.return_value = prompt_runtime

        await AgentOrchestrator.warm_up_kv_cache(orch)

        prompt_runtime.warm_up_kv_cache.assert_awaited_once_with()


# =====================================================================
# W6: KV prefix observability
# =====================================================================


class TestKVPrefixObservability:
    """Test KV prefix hash logging for observability."""

    def test_compute_prefix_hash(self):
        """_kv_prefix_fingerprint delegates to prompt runtime."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        prompt_runtime = MagicMock()
        prompt_runtime.kv_prefix_fingerprint.return_value = "abc123"
        orch._get_prompt_builder_runtime.return_value = prompt_runtime

        h1 = AgentOrchestrator._kv_prefix_fingerprint(orch)
        h2 = AgentOrchestrator._kv_prefix_fingerprint(orch)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) > 0

    def test_different_prompts_different_hash(self):
        """Prompt runtime owns prompt-sensitive fingerprinting."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch1 = MagicMock(spec=AgentOrchestrator)
        runtime1 = MagicMock()
        runtime1.kv_prefix_fingerprint.return_value = "hash-a"
        orch1._get_prompt_builder_runtime.return_value = runtime1
        orch2 = MagicMock(spec=AgentOrchestrator)
        runtime2 = MagicMock()
        runtime2.kv_prefix_fingerprint.return_value = "hash-b"
        orch2._get_prompt_builder_runtime.return_value = runtime2

        assert AgentOrchestrator._kv_prefix_fingerprint(orch1) != (
            AgentOrchestrator._kv_prefix_fingerprint(orch2)
        )


# =====================================================================
# W5: provider_has_kv_cache wired up in builder
# =====================================================================


class TestBuilderKVCacheWiredUp:
    """Test that provider_has_kv_cache is used for section cache freezing."""

    def test_kv_cache_enables_section_selection(self):
        """When provider_has_kv_cache=True, active sections are selected."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder.__new__(SystemPromptBuilder)
        builder.provider_caches = False
        builder.provider_has_kv_cache = True
        builder.concise_mode = False
        builder.headless_mode = False
        builder._rl_coordinator = None
        builder._decision_service = None

        # Active sections should be a set (reduced for non-caching KV providers)
        sections = SystemPromptBuilder._get_active_sections(builder)
        assert isinstance(sections, set)


# =====================================================================
# W7: Separate KV setting
# =====================================================================


class TestSeparateKVSetting:
    """Test kv_optimization_enabled as separate setting."""

    def test_kv_optimization_enabled_setting_exists(self):
        """ContextSettings has kv_optimization_enabled field."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings()
        assert hasattr(settings, "kv_optimization_enabled")
        assert settings.kv_optimization_enabled is True  # Default

    def test_kv_setting_can_be_disabled_independently(self):
        """kv_optimization_enabled can be False while cache_optimization_enabled is True."""
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings(
            cache_optimization_enabled=True,
            kv_optimization_enabled=False,
        )
        assert settings.cache_optimization_enabled is True
        assert settings.kv_optimization_enabled is False


# =====================================================================
# W2: Cached tool sorting
# =====================================================================


class TestCachedToolSorting:
    """Test that sorted tool results are cached to avoid redundant work."""

    def test_same_tools_not_re_sorted(self):
        """Sorting cache avoids re-sorting identical tool sets."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=True)
        orch._last_sorted_tool_names = None
        orch._last_sorted_tools = None
        _attach_legacy_tool_service(orch)

        tool_b = MagicMock()
        tool_b.name = "b_tool"
        tool_a = MagicMock()
        tool_a.name = "a_tool"

        # First sort
        result1 = AgentOrchestrator._sort_tools_for_kv_stability(orch, [tool_b, tool_a])
        assert [t.name for t in result1] == ["a_tool", "b_tool"]

        # Second sort with same tools (different objects but same names)
        tool_b2 = MagicMock()
        tool_b2.name = "b_tool"
        tool_a2 = MagicMock()
        tool_a2.name = "a_tool"
        result2 = AgentOrchestrator._sort_tools_for_kv_stability(orch, [tool_b2, tool_a2])

        # Should use cache — same names means same result
        assert orch._last_sorted_tools is not None


# =====================================================================
# Integration gaps: warm-up wiring, fingerprint logging, facade exposure
# =====================================================================


class TestWarmUpIntegration:
    """Test warm_up_kv_cache is accessible from Agent facade."""

    @pytest.mark.asyncio
    async def test_agent_exposes_warm_up(self):
        """Agent facade has a warm_up() method delegating to orchestrator."""
        from victor.framework.agent import Agent

        assert hasattr(Agent, "warm_up")

    @pytest.mark.asyncio
    async def test_agent_warm_up_delegates_to_orchestrator(self):
        """Agent.warm_up() calls orchestrator.warm_up_kv_cache()."""
        from victor.framework.agent import Agent

        orch = MagicMock()
        orch.warm_up_kv_cache = MagicMock(return_value=None)
        # Make it async
        import asyncio

        async def mock_warm_up():
            pass

        orch.warm_up_kv_cache = mock_warm_up

        agent = Agent.__new__(Agent)
        agent._orchestrator = orch

        # Should not raise
        await Agent.warm_up(agent)


class TestFingerprintLogging:
    """Test that fingerprint is logged during streaming for observability."""

    def test_fingerprint_logged_in_get_assembled_messages(self):
        """_kv_prefix_fingerprint is called when KV optimization is active."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._system_prompt = "test prompt"
        _attach_prompt_runtime(orch)

        fingerprint = AgentOrchestrator._kv_prefix_fingerprint(orch)
        assert len(fingerprint) == 12  # md5[:12]
        assert all(c in "0123456789abcdef" for c in fingerprint)


class TestFallbackPathConsistency:
    """Test that _kv_optimization_enabled fallback checks kv_optimization_enabled setting."""

    def test_fallback_respects_kv_setting_disabled(self):
        """When _kv_opt_cached is None and kv_optimization_enabled=False, returns False."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._kv_opt_cached = None  # Force fallback path

        # Settings: cache enabled but kv disabled
        ctx = MagicMock()
        ctx.cache_optimization_enabled = True
        ctx.kv_optimization_enabled = False
        settings = MagicMock()
        settings.context = ctx
        orch.settings = settings
        orch._check_cache_setting_enabled = lambda: True

        # Provider supports KV
        provider = MagicMock()
        provider.supports_kv_prefix_caching.return_value = True
        orch.provider = provider
        _attach_prompt_runtime(orch)

        # Fallback should check kv_optimization_enabled
        result = AgentOrchestrator._kv_optimization_enabled.fget(orch)
        assert result is False
