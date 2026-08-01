"""Interop tests for SDK runtime adapter seams."""

import pytest

from victor_contracts.capability_runtime import (
    CodebaseIndexFactoryProtocol,
    create_lazy_capability_proxy,
)
from victor_contracts.chain_runtime import get_chain_registry
from victor_contracts.init_runtime import InitSynthesizer
from victor_contracts.lsp_runtime import CompletionItemKind
from victor_contracts.processing_runtime import FileEditor
from victor_contracts.provider_runtime import ProviderRegistry
from victor_contracts.rl_runtime import (
    RLManager,
    analyze_prompt_rollout_experiment,
    analyze_prompt_rollout_experiment_async,
    apply_prompt_rollout_recommendation,
    apply_prompt_rollout_recommendation_async,
    process_prompt_candidate_evaluation_suite,
    process_prompt_candidate_evaluation_suite_async,
    create_prompt_rollout_experiment,
    create_prompt_rollout_experiment_async,
    get_rl_coordinator_async,
)
from victor_contracts.search_runtime import QueryExpander


def test_sdk_runtime_adapters_resolve_host_types() -> None:
    assert FileEditor.__name__ == "FileEditor"
    assert CompletionItemKind.__name__ == "CompletionItemKind"
    assert QueryExpander.__name__ == "QueryExpander"
    assert RLManager.__name__ == "RLManager"
    assert callable(create_prompt_rollout_experiment)
    assert callable(create_prompt_rollout_experiment_async)
    assert callable(analyze_prompt_rollout_experiment)
    assert callable(analyze_prompt_rollout_experiment_async)
    assert callable(apply_prompt_rollout_recommendation)
    assert callable(apply_prompt_rollout_recommendation_async)
    assert callable(process_prompt_candidate_evaluation_suite)
    assert callable(process_prompt_candidate_evaluation_suite_async)
    assert callable(get_rl_coordinator_async)
    assert CodebaseIndexFactoryProtocol.__name__ == "CodebaseIndexFactoryProtocol"
    assert callable(create_lazy_capability_proxy)
    assert callable(get_chain_registry)
    assert InitSynthesizer.__name__ == "InitSynthesizer"
    assert ProviderRegistry.__name__ == "ProviderRegistry"


# --- Deprecated bridge modules (victor-contracts CONTRACT_STABILITY.md) ------
#
# The six consumer-less bridges warn on attribute access starting with SDK
# 0.9.0 but must keep resolving host symbols until removal (>= 0.10.0).


def test_deprecated_agent_spec_runtime_warns_and_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"agent_spec_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.agent_spec_runtime import AgentSpec

    assert AgentSpec.__name__ == "AgentSpec"


def test_deprecated_graph_runtime_warns_and_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"graph_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.graph_runtime import END, StateGraph

    assert StateGraph.__name__ == "StateGraph"
    assert END == "__end__"


def test_deprecated_handler_runtime_warns_and_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"handler_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.handler_runtime import BaseHandler

    assert BaseHandler.__name__ == "BaseHandler"


def test_deprecated_subagent_runtime_warns_and_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"subagent_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.subagent_runtime import set_role_tool_provider

    assert callable(set_role_tool_provider)


def test_deprecated_tool_runtime_warns_and_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"tool_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.tool_runtime import RuntimeToolSet

    assert RuntimeToolSet.__name__ == "ToolSet"


def test_deprecated_workflow_executor_runtime_warns_and_resolves() -> None:
    with pytest.warns(
        DeprecationWarning, match=r"workflow_executor_runtime is deprecated.*0\.10\.0"
    ):
        from victor_contracts.workflow_executor_runtime import WorkflowExecutor

    # WorkflowExecutor is now an alias for CompiledWorkflowExecutor
    assert WorkflowExecutor.__name__ in ("WorkflowExecutor", "CompiledWorkflowExecutor")
