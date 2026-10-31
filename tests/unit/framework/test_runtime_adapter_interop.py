"""Interop tests for SDK runtime adapter seams."""

import pytest

from victor_contracts.capability_runtime import (
    CodebaseIndexFactoryProtocol,
    create_lazy_capability_proxy,
)
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
    assert InitSynthesizer.__name__ == "InitSynthesizer"
    assert ProviderRegistry.__name__ == "ProviderRegistry"


# --- Removed bridge modules (victor-contracts 0.12.0) -----------------------
#
# The six consumer-less bridges deprecated in SDK 0.9.0 were removed in 0.12.0
# (CONTRACT_STABILITY.md). Importing them must fail loudly, never resolve.

_REMOVED_BRIDGES = (
    "agent_spec_runtime",
    "graph_runtime",
    "handler_runtime",
    "subagent_runtime",
    "tool_runtime",
    "workflow_executor_runtime",
)


@pytest.mark.parametrize("module", _REMOVED_BRIDGES)
def test_removed_bridge_modules_are_not_importable(module: str) -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"victor_contracts.{module}")


@pytest.mark.parametrize(
    "module", ["victor.framework.chains", "victor.framework.multi_agent.persona_provider"]
)
def test_versioned_registries_are_removed(module) -> None:
    """FEP-0042 phase 2b: the deprecated versioned registries are gone."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_multi_agent_package_no_longer_exports_persona_provider() -> None:
    import victor.framework.multi_agent as multi_agent

    assert not hasattr(multi_agent, "FrameworkPersonaProvider")
    assert "FrameworkPersonaProvider" not in multi_agent.__all__


def test_chain_runtime_shim_is_detached_from_the_host_versioned_registry() -> None:
    """Since victor-contracts 0.14.0 chain_runtime is an inert SDK shim: it no longer
    resolves to victor.framework.chains, so registrations never reach the host."""
    from victor_contracts import chain_runtime

    with pytest.warns(DeprecationWarning, match="inert compatibility shim"):
        registry = chain_runtime.get_chain_registry()
    assert type(registry).__module__ == "victor_contracts.chain_runtime"
    registry.register_chain(name="interop_probe", version="1.0.0", chain=object())
    assert registry.get_chain("interop_probe") is None


def test_runtime_persona_provider_sdk_hook_removed() -> None:
    from victor_contracts import multi_agent

    assert not hasattr(multi_agent, "get_runtime_persona_provider")
