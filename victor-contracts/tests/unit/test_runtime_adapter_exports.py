"""Tests for processing, LSP, search, and RL runtime adapters."""

import pytest

pytest.importorskip("victor", reason="host runtime adapters require the victor-ai package")

from victor_contracts.capability_runtime import (
    EditorProtocol,
    TreeSitterParserProtocol,
    create_lazy_capability_proxy,
    detect_enhanced_index_factory,
)
from victor_contracts.capabilities import create_runtime_capability_loader
from victor_contracts.chain_runtime import ChainRegistry, get_chain_registry
from victor_contracts.init_runtime import InitSynthesizer
from victor_contracts.lsp_runtime import LSPServiceProtocol
from victor_contracts.processing_runtime import (
    CompletionItem,
    CompletionItemKind,
    EditTransaction,
    Position,
    get_default_text_chunker,
)
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
    get_rl_coordinator,
    get_rl_coordinator_async,
)
from victor_contracts.provider_runtime import Message, ProviderRegistry
from victor_contracts.search_runtime import QueryExpander, QueryExpansionConfig
from victor_contracts.workflow_runtime import (
    ComputeNode as WorkflowRuntimeComputeNode,
    ExecutorNodeStatus as WorkflowRuntimeExecutorNodeStatus,
    NodeResult as WorkflowRuntimeNodeResult,
    WorkflowContext as WorkflowRuntimeWorkflowContext,
    WorkflowDefinition,
    WorkflowExecutor as WorkflowRuntimeWorkflowExecutor,
    WorkflowResult as WorkflowRuntimeWorkflowResult,
    register_compute_handler as workflow_runtime_register_compute_handler,
)


def test_processing_runtime_exports_host_helpers() -> None:
    assert EditTransaction.__name__ == "EditTransaction"
    assert CompletionItem.__name__ == "CompletionItem"
    assert Position.__name__ == "Position"
    assert CompletionItemKind.__name__ == "CompletionItemKind"
    assert callable(get_default_text_chunker)


def test_lsp_runtime_exports_host_helpers() -> None:
    assert Position.__name__ == "Position"
    assert LSPServiceProtocol.__name__ == "LSPServiceProtocol"


def test_search_runtime_exports_host_helpers() -> None:
    assert QueryExpander.__name__ == "QueryExpander"
    assert QueryExpansionConfig.__name__ == "QueryExpansionConfig"


def test_rl_runtime_exports_host_helpers() -> None:
    assert RLManager.__name__ == "RLManager"
    assert callable(get_rl_coordinator)
    assert callable(get_rl_coordinator_async)
    assert callable(create_prompt_rollout_experiment)
    assert callable(create_prompt_rollout_experiment_async)
    assert callable(analyze_prompt_rollout_experiment)
    assert callable(analyze_prompt_rollout_experiment_async)
    assert callable(apply_prompt_rollout_recommendation)
    assert callable(apply_prompt_rollout_recommendation_async)
    assert callable(process_prompt_candidate_evaluation_suite)
    assert callable(process_prompt_candidate_evaluation_suite_async)


def test_capability_runtime_exports_host_helpers() -> None:
    assert TreeSitterParserProtocol.__name__ == "TreeSitterParserProtocol"
    assert EditorProtocol.__name__ == "EditorProtocol"
    assert callable(create_lazy_capability_proxy)
    assert callable(detect_enhanced_index_factory)
    assert callable(create_runtime_capability_loader)


def test_capability_runtime_lazy_proxy_falls_back_without_host_runtime(
    monkeypatch,
) -> None:
    import importlib

    from victor_contracts import capability_runtime

    class Provider:
        value = "ready"

        def __call__(self, suffix: str) -> str:
            return f"called:{suffix}"

    calls = 0

    def factory() -> Provider:
        nonlocal calls
        calls += 1
        return Provider()

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "victor.core.plugins.context":
            raise ImportError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(capability_runtime.importlib, "import_module", fake_import_module)

    proxy = capability_runtime.create_lazy_capability_proxy(factory)

    assert calls == 0
    assert proxy.value == "ready"
    assert calls == 1
    assert proxy("ok") == "called:ok"
    assert calls == 1


def test_chain_provider_and_init_runtime_exports_host_helpers() -> None:
    assert ChainRegistry.__name__ == "ChainRegistry"
    assert callable(get_chain_registry)
    assert InitSynthesizer.__name__ == "InitSynthesizer"
    assert Message.__name__ == "Message"
    assert ProviderRegistry.__name__ == "ProviderRegistry"


def test_workflow_runtime_exports_definition_and_executor_helpers() -> None:
    assert WorkflowDefinition.__name__ == "WorkflowDefinition"
    assert WorkflowRuntimeComputeNode.__name__ == "ComputeNode"
    assert WorkflowRuntimeWorkflowExecutor.__name__ in (
        "WorkflowExecutor",
        "CompiledWorkflowExecutor",
    )
    assert WorkflowRuntimeWorkflowContext.__name__ == "WorkflowContext"
    assert WorkflowRuntimeWorkflowResult.__name__ == "WorkflowResult"
    assert WorkflowRuntimeNodeResult.__name__ == "NodeResult"
    assert WorkflowRuntimeExecutorNodeStatus.__name__ == "ExecutorNodeStatus"
    assert callable(workflow_runtime_register_compute_handler)


# --- Deprecated bridge modules (CONTRACT_STABILITY.md) -----------------------
#
# These six bridges have zero in-tree consumers, warn on attribute access
# starting with 0.9.0, and are removed no earlier than 0.10.0. Each test
# asserts the DeprecationWarning fires while symbol resolution keeps working.


def test_agent_spec_runtime_deprecated_but_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"agent_spec_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.agent_spec_runtime import (
            AgentCapabilities,
            AgentConstraints,
            AgentSpec,
            ModelPreference,
            OutputFormat,
        )

    assert AgentSpec.__name__ == "AgentSpec"
    assert AgentCapabilities.__name__ == "AgentCapabilities"
    assert AgentConstraints.__name__ == "AgentConstraints"
    assert ModelPreference.__name__ == "ModelPreference"
    assert OutputFormat.__name__ == "OutputFormat"


def test_graph_runtime_deprecated_but_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"graph_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.graph_runtime import END, StateGraph

    assert StateGraph.__name__ == "StateGraph"
    assert END == "__end__"


def test_handler_runtime_deprecated_but_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"handler_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.handler_runtime import BaseHandler, handler_decorator

    assert BaseHandler.__name__ == "BaseHandler"
    assert callable(handler_decorator)


def test_handler_runtime_offline_shim_still_warns(monkeypatch) -> None:
    """The BaseHandler compatibility shim path warns and still functions."""
    import importlib

    from victor_contracts import handler_runtime

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name.startswith("victor."):
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(handler_runtime.importlib, "import_module", fake_import_module)

    with pytest.warns(DeprecationWarning, match=r"handler_runtime is deprecated.*0\.10\.0"):
        shim = handler_runtime.BaseHandler

    assert shim.__name__ == "BaseHandler"
    assert shim.__doc__ == "Compatibility base class for class-based workflow handlers."


def test_subagent_runtime_deprecated_but_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"subagent_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.subagent_runtime import RoleToolProvider, set_role_tool_provider

    assert RoleToolProvider.__name__ == "RoleToolProvider"
    assert callable(set_role_tool_provider)


def test_tool_runtime_deprecated_but_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"tool_runtime is deprecated.*0\.10\.0"):
        from victor_contracts.tool_runtime import RuntimeToolSet

    assert RuntimeToolSet.__name__ == "ToolSet"


def test_workflow_executor_runtime_deprecated_but_resolves() -> None:
    with pytest.warns(
        DeprecationWarning, match=r"workflow_executor_runtime is deprecated.*0\.10\.0"
    ):
        from victor_contracts.workflow_executor_runtime import (
            ComputeNode,
            ExecutorNodeStatus,
            NodeResult,
            WorkflowContext,
            WorkflowExecutor,
            WorkflowResult,
            register_compute_handler,
        )

    assert WorkflowExecutor.__name__ in ("WorkflowExecutor", "CompiledWorkflowExecutor")
    assert WorkflowContext.__name__ == "WorkflowContext"
    assert WorkflowResult.__name__ == "WorkflowResult"
    assert NodeResult.__name__ == "NodeResult"
    assert ExecutorNodeStatus.__name__ == "ExecutorNodeStatus"
    assert ComputeNode.__name__ == "ComputeNode"
    assert callable(register_compute_handler)
