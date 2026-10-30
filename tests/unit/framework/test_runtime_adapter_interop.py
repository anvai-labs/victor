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


def test_versioned_registries_are_deprecated_but_package_imports_stay_quiet(
    monkeypatch,
) -> None:
    import importlib
    import sys
    import warnings

    # Fresh imports so the module-level warnings fire; monkeypatch restores
    # the original module objects (and their singletons) afterwards.
    for module in (
        "victor.framework.chains",
        "victor.framework.chains.registry",
        "victor.framework.multi_agent.persona_provider",
    ):
        monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.warns(DeprecationWarning, match=r"deprecated.*victor-contracts 0\.14\.0"):
        importlib.import_module("victor.framework.chains")
    with pytest.warns(DeprecationWarning, match=r"deprecated.*victor-contracts 0\.14\.0"):
        importlib.import_module("victor.framework.multi_agent.persona_provider")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.import_module("victor.framework.multi_agent")


def test_lazy_persona_provider_export_warns_at_caller() -> None:
    import victor.framework.multi_agent as multi_agent

    with pytest.warns(DeprecationWarning, match="FrameworkPersonaProvider") as record:
        provider_class = multi_agent.FrameworkPersonaProvider
    assert provider_class.__name__ == "FrameworkPersonaProvider"
    assert len(record) == 1
    assert record[0].filename == __file__
    assert "FrameworkPersonaProvider" in dir(multi_agent)
