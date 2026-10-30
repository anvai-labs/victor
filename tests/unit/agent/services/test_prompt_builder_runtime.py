from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import os

import pytest

from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.services.prompt_builder_runtime import PromptBuilderRuntime
from victor.agent.services.metrics_service import AgentMetricsService
from victor.agent.session_cost_tracker import SessionCostTracker
from victor.providers.base import CompletionResponse


def test_sync_prompt_builder_runtime_state_updates_builder_and_invalidates_cache():
    builder = SimpleNamespace(
        available_tools=["read"],
        mode_prompt_addition="old mode",
        invalidate_cache=MagicMock(),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        get_enabled_tools=MagicMock(return_value={"write", "read"}),
        get_mode_system_prompt=MagicMock(return_value="new mode"),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_prompt_builder_runtime_state()

    assert builder.available_tools == ["read", "write"]
    assert builder.stable_prompt_tools == ["read", "write"]
    assert builder.dynamic_prompt_tools == []
    assert builder.mode_prompt_addition == "new mode"
    builder.invalidate_cache.assert_called_once_with()


def test_prompt_builder_runtime_computes_cache_flags_from_provider_capabilities():
    provider = SimpleNamespace(
        supports_prompt_caching=MagicMock(return_value=False),
        supports_kv_prefix_caching=MagicMock(return_value=True),
    )
    host = SimpleNamespace(
        provider=provider,
        settings=SimpleNamespace(
            context=SimpleNamespace(
                cache_optimization_enabled=True,
                kv_optimization_enabled=True,
            )
        ),
        _kv_opt_cached=None,
        _cache_opt_cached=None,
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.compute_cache_flags()

    assert host._kv_opt_cached is True
    assert host._cache_opt_cached is False


def test_prompt_builder_runtime_disables_cache_flags_when_settings_disable_cache():
    provider = SimpleNamespace(
        supports_prompt_caching=MagicMock(return_value=True),
        supports_kv_prefix_caching=MagicMock(return_value=True),
    )
    host = SimpleNamespace(
        provider=provider,
        settings=SimpleNamespace(
            context=SimpleNamespace(
                cache_optimization_enabled=False,
                kv_optimization_enabled=True,
            )
        ),
        _kv_opt_cached=None,
        _cache_opt_cached=None,
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.compute_cache_flags()

    assert host._kv_opt_cached is False
    assert host._cache_opt_cached is False
    provider.supports_prompt_caching.assert_not_called()
    provider.supports_kv_prefix_caching.assert_not_called()


def test_prompt_builder_runtime_reads_cached_kv_flag_when_available():
    host = SimpleNamespace(_kv_opt_cached=True)
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.is_kv_optimization_enabled() is True


def test_prompt_builder_runtime_computes_kv_flag_before_cache_is_initialized():
    provider = SimpleNamespace(supports_kv_prefix_caching=MagicMock(return_value=True))
    host = SimpleNamespace(
        _kv_opt_cached=None,
        provider=provider,
        settings=SimpleNamespace(
            context=SimpleNamespace(
                cache_optimization_enabled=True,
                kv_optimization_enabled=True,
            )
        ),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.is_kv_optimization_enabled() is True


def test_prompt_builder_runtime_computes_prefix_fingerprint_from_system_prompt():
    host = SimpleNamespace(_system_prompt="You are a helpful assistant.")
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    first = runtime.kv_prefix_fingerprint()
    second = runtime.kv_prefix_fingerprint()

    assert first == second
    assert isinstance(first, str)
    assert len(first) == 12


def test_prompt_builder_runtime_prefix_fingerprint_changes_with_prompt():
    first = PromptBuilderRuntime(
        OrchestratorProtocolAdapter(SimpleNamespace(_system_prompt="Prompt A"))
    ).kv_prefix_fingerprint()
    second = PromptBuilderRuntime(
        OrchestratorProtocolAdapter(SimpleNamespace(_system_prompt="Prompt B"))
    ).kv_prefix_fingerprint()

    assert first != second


@pytest.mark.asyncio
async def test_prompt_builder_runtime_warms_up_kv_cache_with_minimal_request():
    provider = SimpleNamespace(
        name="unknown",
        chat=AsyncMock(return_value=CompletionResponse(content="", role="assistant")),
    )
    host = SimpleNamespace(
        _kv_opt_cached=True,
        _metrics_coordinator=AgentMetricsService(MagicMock(), SessionCostTracker(), {}),
        provider=provider,
        model="test-model",
        _system_prompt="You are an assistant.",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    await runtime.warm_up_kv_cache()

    provider.chat.assert_awaited_once()
    _, kwargs = provider.chat.await_args
    assert kwargs["model"] == "test-model"
    assert kwargs["max_tokens"] == 1
    assert kwargs["messages"][0].role == "system"
    assert kwargs["messages"][0].content == "You are an assistant."


def test_sync_prompt_builder_runtime_state_clears_stale_values_on_errors():
    builder = SimpleNamespace(
        available_tools=["read"],
        mode_prompt_addition="old mode",
        invalidate_cache=MagicMock(),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        get_enabled_tools=MagicMock(side_effect=RuntimeError("tools unavailable")),
        get_mode_system_prompt=MagicMock(side_effect=RuntimeError("mode unavailable")),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_prompt_builder_runtime_state()

    assert builder.available_tools == []
    assert builder.stable_prompt_tools == []
    assert builder.dynamic_prompt_tools == []
    assert builder.mode_prompt_addition == ""
    builder.invalidate_cache.assert_called_once_with()


@pytest.mark.parametrize("project_addition", [None, "project addition"])
def test_build_system_prompt_fallback_prefers_runtime_emit_hook(project_addition):
    prompt_used_hook = MagicMock()
    prompt_orchestrator = MagicMock()
    prompt_orchestrator.build_system_prompt.return_value = "runtime prompt"
    host = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            available_tools=[],
            mode_prompt_addition="",
            invalidate_cache=MagicMock(),
            task_type="edit",
        ),
        get_enabled_tools=MagicMock(return_value=set()),
        get_mode_system_prompt=MagicMock(return_value=""),
        _emit_prompt_used_event=prompt_used_hook,
        _system_prompt_coordinator=None,
        _prompt_orchestrator=prompt_orchestrator,
        provider_name="anthropic",
        model="claude",
        _get_model_context_window=lambda: 65536,
        project_context=SimpleNamespace(
            content=project_addition,
            get_system_prompt_addition=MagicMock(return_value=project_addition),
        ),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.build_system_prompt()

    expected = "runtime prompt" + ("\n\n" + project_addition if project_addition else "")
    assert result == expected
    assert host.project_context.get_system_prompt_addition.call_count == bool(project_addition)
    prompt_orchestrator.build_system_prompt.assert_called_once_with(
        builder_type="legacy",
        provider="anthropic",
        model="claude",
        task_type="edit",
        builder=host.prompt_builder,
        get_context_window=host._get_model_context_window,
        on_prompt_built=prompt_used_hook,
    )


def test_compose_system_prompt_appends_project_context_when_present():
    host = SimpleNamespace(
        project_context=SimpleNamespace(
            content="repo context",
            get_system_prompt_addition=MagicMock(return_value="project addition"),
        )
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.compose_system_prompt("base prompt")

    assert result == "base prompt\n\nproject addition"


def test_update_system_prompt_for_query_updates_runtime_and_conversation():
    builder = SimpleNamespace(
        query_classification=None,
        invalidate_cache=MagicMock(),
    )
    conversation = SimpleNamespace(
        system_prompt="old prompt",
        _system_added=True,
        _messages=[SimpleNamespace(role="system", content="old prompt")],
    )
    project_context = SimpleNamespace(
        content="repo context",
        get_system_prompt_addition=MagicMock(return_value="project addition"),
    )
    classification = object()
    host = SimpleNamespace(
        prompt_builder=builder,
        build_system_prompt=MagicMock(return_value="base prompt\n\nproject addition"),
        project_context=project_context,
        conversation=conversation,
        _prompt_pipeline=None,
        _system_prompt_frozen=False,
        _kv_optimization_enabled=True,
        _system_prompt="",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.update_system_prompt_for_query(query_classification=classification)

    assert builder.query_classification is classification
    builder.invalidate_cache.assert_called_once_with()
    host.build_system_prompt.assert_called_once_with()
    assert host._system_prompt == "base prompt\n\nproject addition"
    project_context.get_system_prompt_addition.assert_not_called()
    assert host._system_prompt_frozen is True
    assert conversation.system_prompt == host._system_prompt
    assert conversation._messages[0].role == "system"
    assert conversation._messages[0].content == host._system_prompt


def test_update_system_prompt_for_query_skips_when_prompt_is_frozen():
    builder = SimpleNamespace(
        invalidate_cache=MagicMock(),
        get_task_guidance_text=MagicMock(return_value="TASK GUIDANCE: Plan before coding."),
        query_classification=None,
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        build_system_prompt=MagicMock(),
        _prompt_pipeline=SimpleNamespace(is_frozen=True),
        _kv_optimization_enabled=True,
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.update_system_prompt_for_query(query_classification="coding")

    builder.invalidate_cache.assert_not_called()
    host.build_system_prompt.assert_not_called()
    assert host._dynamic_task_guidance == "TASK GUIDANCE: Plan before coding."


def test_refresh_system_prompt_preserves_existing_classification():
    builder = SimpleNamespace(
        query_classification="existing",
        invalidate_cache=MagicMock(),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        _prompt_pipeline=SimpleNamespace(unfreeze=MagicMock()),
        _system_prompt_frozen=True,
        _session_tools=["read"],
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))
    runtime.update_system_prompt_for_query = MagicMock()

    runtime.refresh_system_prompt()

    host._prompt_pipeline.unfreeze.assert_called_once_with()
    builder.invalidate_cache.assert_called_once_with()
    assert host._system_prompt_frozen is False
    assert host._session_tools is None
    runtime.update_system_prompt_for_query.assert_called_once_with(query_classification="existing")


def test_sync_prompt_builder_runtime_state_splits_stable_and_dynamic_tools():
    builder = SimpleNamespace(
        available_tools=[],
        stable_prompt_tools=[],
        dynamic_prompt_tools=[],
        mode_prompt_addition="",
        invalidate_cache=MagicMock(),
        provider_name="anthropic",
        model="claude",
        provider_caches=True,
        provider_has_kv_cache=True,
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        get_enabled_tools=MagicMock(return_value={"read", "shell", "web_search"}),
        get_mode_system_prompt=MagicMock(return_value=""),
        _get_model_context_window=lambda: 32768,
        provider=SimpleNamespace(
            supports_prompt_caching=lambda: True,
            supports_kv_prefix_caching=lambda: True,
        ),
        provider_name="anthropic",
        model="claude",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_prompt_builder_runtime_state()

    # P5 capability-first: this provider supports prompt caching, so the FULL tool set
    # is kept in the stable prefix (cached ~free) and the dynamic/per-turn split is empty.
    assert builder.stable_prompt_tools == ["read", "shell", "web_search"]
    assert builder.dynamic_prompt_tools == []


def test_sync_prompt_builder_runtime_state_tiers_split_for_non_caching_provider():
    builder = SimpleNamespace(
        available_tools=[],
        stable_prompt_tools=[],
        dynamic_prompt_tools=[],
        mode_prompt_addition="",
        invalidate_cache=MagicMock(),
        provider_name="ollama",
        model="qwen",
        provider_caches=False,
        provider_has_kv_cache=True,
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        get_enabled_tools=MagicMock(return_value={"read", "shell", "web_search"}),
        get_mode_system_prompt=MagicMock(return_value=""),
        _get_model_context_window=lambda: 32768,
        provider=SimpleNamespace(
            supports_prompt_caching=lambda: False,
            supports_kv_prefix_caching=lambda: True,
        ),
        provider_name="ollama",
        model="qwen",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_prompt_builder_runtime_state()

    # Non-caching provider keeps the window-tiered CORE-stable + ADAPTIVE-per-turn split.
    assert builder.stable_prompt_tools == ["read", "shell"]
    assert builder.dynamic_prompt_tools == ["web_search"]


def test_ensure_system_prompt_current_refreshes_when_project_context_changes(tmp_path):
    from victor.context.project_context import ProjectContext

    root = tmp_path
    context_dir = root / ".victor"
    context_dir.mkdir(exist_ok=True)
    context_file = context_dir / "init.md"
    context_file.write_text("Initial context", encoding="utf-8")

    project_context = ProjectContext(root_path=str(root))
    assert project_context.load(force_reload=True) is True

    builder = SimpleNamespace(
        available_tools=["read", "shell"],
        stable_prompt_tools=["read", "shell"],
        dynamic_prompt_tools=[],
        mode_prompt_addition="",
        invalidate_cache=MagicMock(),
        provider_name="anthropic",
        model="claude",
        provider_caches=True,
        provider_has_kv_cache=True,
        query_classification=None,
    )
    pipeline = SimpleNamespace(is_frozen=True)
    pipeline.unfreeze = MagicMock(side_effect=lambda: setattr(pipeline, "is_frozen", False))
    host = SimpleNamespace(
        prompt_builder=builder,
        get_enabled_tools=MagicMock(return_value={"read", "shell"}),
        get_mode_system_prompt=MagicMock(return_value=""),
        _get_model_context_window=lambda: 128000,
        provider=SimpleNamespace(
            supports_prompt_caching=lambda: True,
            supports_kv_prefix_caching=lambda: True,
        ),
        provider_name="anthropic",
        model="claude",
        project_context=project_context,
        build_system_prompt=MagicMock(
            side_effect=lambda: "rebuilt prompt\n\n" + project_context.get_system_prompt_addition()
        ),
        conversation=None,
        _prompt_pipeline=pipeline,
        _system_prompt_frozen=True,
        _kv_optimization_enabled=True,
        _session_tools=["read"],
        _session_semantic_tools=["shell"],
        _system_prompt="rebuilt prompt\n\nInitial context",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.ensure_system_prompt_current()
    host.build_system_prompt.assert_not_called()

    context_file.write_text("Updated context", encoding="utf-8")
    stat = context_file.stat()
    os.utime(context_file, (stat.st_atime, stat.st_mtime + 5))

    runtime.ensure_system_prompt_current()

    host.build_system_prompt.assert_called_once_with()
    host._prompt_pipeline.unfreeze.assert_called_once_with()
    assert host._session_tools is None
    assert host._session_semantic_tools is None
    assert "Updated context" in host._system_prompt


def test_build_dynamic_tool_guidance_uses_relevant_dynamic_tools():
    builder = SimpleNamespace(
        dynamic_prompt_tools=["web_search", "git_diff"],
        get_dynamic_tool_guidance_text=MagicMock(return_value="DYNAMIC TOOL HINTS: web_search"),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        tools=SimpleNamespace(
            list_tools=lambda: [
                SimpleNamespace(
                    name="web_search",
                    description="Search the web for documentation",
                    metadata=SimpleNamespace(keywords=["docs", "documentation", "web"]),
                ),
                SimpleNamespace(
                    name="git_diff",
                    description="Inspect recent repository changes",
                    metadata=SimpleNamespace(keywords=["diff", "patch"]),
                ),
            ]
        ),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    guidance = runtime.build_dynamic_tool_guidance("Search docs for the latest API behavior")

    assert guidance == "DYNAMIC TOOL HINTS: web_search"
    builder.get_dynamic_tool_guidance_text.assert_called_once_with(
        ["web_search"],
        selection_source="message_keywords",
        tool_rationale={"web_search": "Search the web for documentation"},
    )


def test_build_dynamic_tool_guidance_prefers_tool_selector_keyword_slice():
    builder = SimpleNamespace(
        dynamic_prompt_tools=["web_search", "git_diff", "workflow"],
        get_dynamic_tool_guidance_text=MagicMock(return_value="DYNAMIC TOOL HINTS: git_diff"),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        tool_selector=SimpleNamespace(
            select_keywords=MagicMock(
                return_value=[
                    SimpleNamespace(name="read"),
                    SimpleNamespace(name="git_diff"),
                    SimpleNamespace(name="shell"),
                ]
            )
        ),
        tools=SimpleNamespace(list_tools=lambda: []),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    guidance = runtime.build_dynamic_tool_guidance(
        "Check the patch and summarize the repo delta",
        selected_tools=[SimpleNamespace(name="read"), SimpleNamespace(name="workflow")],
    )

    assert guidance == "DYNAMIC TOOL HINTS: git_diff"
    host.tool_selector.select_keywords.assert_called_once_with(
        "Check the patch and summarize the repo delta",
        planned_tools=None,
        _record=False,
    )
    builder.get_dynamic_tool_guidance_text.assert_called_once_with(
        ["git_diff"],
        selection_source="keyword_selector",
    )


def test_build_dynamic_tool_guidance_prefers_planned_tools_over_keyword_slice():
    builder = SimpleNamespace(
        dynamic_prompt_tools=["web_search", "git_diff", "workflow"],
        get_dynamic_tool_guidance_text=MagicMock(return_value="DYNAMIC TOOL HINTS: workflow"),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        tool_selector=SimpleNamespace(
            select_keywords=MagicMock(return_value=[SimpleNamespace(name="git_diff")])
        ),
        tools=SimpleNamespace(list_tools=lambda: []),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    guidance = runtime.build_dynamic_tool_guidance(
        "Continue execution",
        planned_tools=[
            SimpleNamespace(name="workflow", description="Coordinate multi-step execution"),
            SimpleNamespace(name="read", description="Read file contents"),
        ],
        selected_tools=[SimpleNamespace(name="web_search")],
    )

    assert guidance == "DYNAMIC TOOL HINTS: workflow"
    host.tool_selector.select_keywords.assert_not_called()
    builder.get_dynamic_tool_guidance_text.assert_called_once_with(
        ["workflow"],
        selection_source="planned_tools",
        tool_rationale={"workflow": "Coordinate multi-step execution"},
    )


def test_build_dynamic_tool_guidance_falls_back_to_selected_dynamic_tools_when_selector_misses():
    builder = SimpleNamespace(
        dynamic_prompt_tools=["web_search", "git_diff", "workflow"],
        get_dynamic_tool_guidance_text=MagicMock(return_value="DYNAMIC TOOL HINTS: workflow"),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        tool_selector=SimpleNamespace(
            select_keywords=MagicMock(return_value=[SimpleNamespace(name="read")])
        ),
        tools=SimpleNamespace(list_tools=lambda: []),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    guidance = runtime.build_dynamic_tool_guidance(
        "Continue execution",
        selected_tools=[SimpleNamespace(name="workflow"), SimpleNamespace(name="read")],
    )

    assert guidance == "DYNAMIC TOOL HINTS: workflow"
    builder.get_dynamic_tool_guidance_text.assert_called_once_with(
        ["workflow"],
        selection_source="selected_tools",
    )


def test_build_dynamic_tool_guidance_passes_goals_and_intent_context():
    builder = SimpleNamespace(
        dynamic_prompt_tools=["web_search", "git_diff", "workflow"],
        get_dynamic_tool_guidance_text=MagicMock(return_value="DYNAMIC TOOL HINTS: workflow"),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        tool_selector=SimpleNamespace(
            select_keywords=MagicMock(return_value=[SimpleNamespace(name="git_diff")])
        ),
        tools=SimpleNamespace(list_tools=lambda: []),
        _current_intent="read_only",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    guidance = runtime.build_dynamic_tool_guidance(
        "Continue execution",
        goals=["inspect repo changes", "verify docs"],
        planned_tools=[
            SimpleNamespace(name="workflow", description="Coordinate multi-step execution"),
            SimpleNamespace(name="read", description="Read file contents"),
        ],
        selected_tools=[SimpleNamespace(name="web_search")],
    )

    assert guidance == "DYNAMIC TOOL HINTS: workflow"
    builder.get_dynamic_tool_guidance_text.assert_called_once_with(
        ["workflow"],
        goals=["inspect repo changes", "verify docs"],
        current_intent="read_only",
        selection_source="planned_tools",
        tool_rationale={"workflow": "Coordinate multi-step execution"},
    )


def test_build_dynamic_tool_guidance_uses_registry_metadata_for_tool_rationale():
    builder = SimpleNamespace(
        dynamic_prompt_tools=["web_search", "git_diff"],
        get_dynamic_tool_guidance_text=MagicMock(return_value="DYNAMIC TOOL HINTS: web_search"),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        tools=SimpleNamespace(
            list_tools=lambda: [
                SimpleNamespace(
                    name="web_search",
                    description="Search the web for documentation",
                    metadata=SimpleNamespace(
                        use_cases=["Check upstream docs for API changes"],
                        keywords=["docs", "documentation", "web"],
                    ),
                ),
                SimpleNamespace(
                    name="git_diff",
                    description="Inspect recent repository changes",
                    metadata=SimpleNamespace(keywords=["diff", "patch"]),
                ),
            ]
        ),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    guidance = runtime.build_dynamic_tool_guidance("Search docs for the latest API behavior")

    assert guidance == "DYNAMIC TOOL HINTS: web_search"
    builder.get_dynamic_tool_guidance_text.assert_called_once_with(
        ["web_search"],
        selection_source="message_keywords",
        tool_rationale={"web_search": "Check upstream docs for API changes"},
    )


def test_get_set_and_append_system_prompt_delegate_to_builder():
    builder = SimpleNamespace(
        build=MagicMock(return_value="base prompt"),
        set_custom_prompt=MagicMock(),
    )
    host = SimpleNamespace(prompt_builder=builder)
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.get_system_prompt() == "base prompt"

    runtime.set_system_prompt("custom prompt")
    runtime.append_to_system_prompt("extra guidance")

    builder.set_custom_prompt.assert_any_call("custom prompt")
    builder.set_custom_prompt.assert_any_call("base prompt\n\nextra guidance")


def test_sync_conversation_system_prompt_updates_live_system_message():
    conversation = SimpleNamespace(
        system_prompt="old prompt",
        _system_added=True,
        _messages=[SimpleNamespace(role="system", content="old prompt")],
    )
    host = SimpleNamespace(
        conversation=conversation,
        _system_prompt="new prompt",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_conversation_system_prompt()

    assert conversation.system_prompt == "new prompt"
    assert conversation._messages[0].role == "system"
    assert conversation._messages[0].content == "new prompt"


@pytest.mark.parametrize("api_cache,kv_cache", [(False, False), (True, False), (False, True)])
def test_custom_prompt_capability_refreshes_real_pipeline_and_live_conversation(
    api_cache, kv_cache
):
    from victor.agent.capability_registry import OrchestratorCapabilityMixin
    from victor.agent.prompt_builder import SystemPromptBuilder
    from victor.agent.prompt_pipeline import UnifiedPromptPipeline
    from victor.providers.base import Message

    class Host(OrchestratorCapabilityMixin):
        def set_system_prompt(self, prompt):
            self.prompt_runtime.set_system_prompt(prompt)

    host = Host()
    host.prompt_builder = SystemPromptBuilder(provider_name="openai", model="fixture")
    provider = SimpleNamespace(
        supports_prompt_caching=lambda: api_cache,
        supports_kv_prefix_caching=lambda: kv_cache,
    )
    host._prompt_pipeline = UnifiedPromptPipeline(
        provider,
        host.prompt_builder,
        enable_prompt_completeness_guard=False,
    )
    host.build_system_prompt = host._prompt_pipeline.build_system_prompt
    host._system_prompt = host.build_system_prompt()
    original = host._system_prompt
    host._kv_optimization_enabled = kv_cache
    host.conversation = SimpleNamespace(
        system_prompt=original,
        _system_added=True,
        _messages=[Message(role="system", content=original)],
    )
    host.prompt_runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    # The vertical applies its capability after the initial system prefix is built.
    host._set_custom_prompt_via_builder("Research policy for this session")
    assert "Research policy for this session" in host._system_prompt
    assert host.build_system_prompt() == host._system_prompt
    assert host.conversation.system_prompt == host._system_prompt
    assert host.conversation._messages[0].content == host._system_prompt
    host.prompt_runtime.append_to_system_prompt("Additional domain rule")
    assert host._system_prompt.count("Research policy for this session") == 1
    assert host._system_prompt.count("Additional domain rule") == 1
    assert host._system_prompt.count(original.split("\n")[0]) == 1
    host._set_custom_prompt_via_builder("New session policy")
    assert "Research policy for this session" not in host._system_prompt
    host._set_custom_prompt_via_builder(None)
    assert host._system_prompt == original
    assert host.conversation._messages[0].content == original
    other_builder = SystemPromptBuilder(provider_name="openai", model="fixture")
    # Existing builder consumers can also change policy; signature invalidation
    # must refresh a previously frozen conversation on its next turn.
    host.prompt_runtime.ensure_system_prompt_current()
    host.prompt_builder.set_custom_prompt("Changed policy through builder")
    host.prompt_runtime.ensure_system_prompt_current()
    assert "Changed policy through builder" in host.conversation._messages[0].content
    assert "Changed policy through builder" not in other_builder.build()
    before = host._system_prompt
    with pytest.raises(TypeError):
        host._set_custom_prompt_via_builder({"content": "untyped policy"})
    assert host._system_prompt == before
    assert host.conversation._messages[0].content == before


def test_custom_prompt_capability_refuses_unsupported_builder():
    from victor.agent.capability_registry import OrchestratorCapabilityMixin

    host = OrchestratorCapabilityMixin()
    host.prompt_builder = SimpleNamespace()
    with pytest.raises(ValueError, match="custom prompt"):
        host._set_custom_prompt_via_builder("Required domain policy")
