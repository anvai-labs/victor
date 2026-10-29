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
    from victor.native.protocols import (
        ChunkInfo as HostChunkInfo,
        TextChunkerProtocol as HostChunker,
    )
    from victor_contracts.processing_runtime import ChunkInfo, TextChunkerProtocol

    assert EditTransaction.__name__ == "EditTransaction"
    assert CompletionItem.__name__ == "CompletionItem"
    assert Position.__name__ == "Position"
    assert CompletionItemKind.__name__ == "CompletionItemKind"
    assert callable(get_default_text_chunker)
    assert ChunkInfo is HostChunkInfo
    assert TextChunkerProtocol is HostChunker


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
        "StateGraphWorkflowExecutor",
    )
    assert WorkflowRuntimeWorkflowContext.__name__ == "WorkflowContext"
    assert WorkflowRuntimeWorkflowResult.__name__ == "WorkflowResult"
    assert WorkflowRuntimeNodeResult.__name__ == "NodeResult"
    assert WorkflowRuntimeExecutorNodeStatus.__name__ == "ExecutorNodeStatus"
    assert callable(workflow_runtime_register_compute_handler)


def test_embeddings_runtime_exports_host_helpers() -> None:
    """The embeddings seam resolves to the ONE implementation in
    victor.storage.vector_stores (dedup round 8 follow-up: the victor-coding
    model fork was deleted; verticals reach the models through this shim)."""
    from victor_contracts.embeddings_runtime import (
        BaseEmbeddingModel,
        BaseEmbeddingProvider,
        CohereEmbeddingModel,
        EmbeddingConfig,
        EmbeddingModelConfig,
        EmbeddingRegistry,
        EmbeddingSearchResult,
        OllamaEmbeddingModel,
        OpenAIEmbeddingModel,
        SentenceTransformerModel,
        create_embedding_model,
    )

    from victor.storage.vector_stores import models as core_models

    assert SentenceTransformerModel is core_models.SentenceTransformerModel
    assert OllamaEmbeddingModel is core_models.OllamaEmbeddingModel
    assert OpenAIEmbeddingModel is core_models.OpenAIEmbeddingModel
    assert CohereEmbeddingModel is core_models.CohereEmbeddingModel
    assert BaseEmbeddingModel is core_models.BaseEmbeddingModel
    assert create_embedding_model is core_models.create_embedding_model
    assert EmbeddingModelConfig is core_models.EmbeddingModelConfig
    # Unknown names fail loudly, not silently.
    import pytest

    with pytest.raises(AttributeError):
        _ = victor_contracts.embeddings_runtime.NotARealSymbol  # type: ignore[attr-defined]


import victor_contracts.embeddings_runtime  # noqa: E402  (module-level for the AttributeError probe)
