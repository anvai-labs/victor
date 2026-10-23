"""SDK host adapter for the embedding-model runtime helpers.

Vertical-facing embedding primitives (the four backend models, the base
classes, the config, and the registry) live in ONE implementation —
``victor.storage.vector_stores`` — and reach verticals through this lazy
seam, exactly like ``processing_runtime``. Verticals never import
``victor.*`` directly (CI-enforced boundary); victor-contracts itself stays
runtime-dependency-free because resolution happens only at first attribute
access, inside a Victor host.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.storage.vector_stores import (
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

__all__ = [
    "BaseEmbeddingProvider",
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingModelConfig",
    "EmbeddingSearchResult",
    "SentenceTransformerModel",
    "OllamaEmbeddingModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "create_embedding_model",
    "EmbeddingRegistry",
]

_LAZY_IMPORTS = {
    "BaseEmbeddingProvider": "victor.storage.vector_stores",
    "BaseEmbeddingModel": "victor.storage.vector_stores",
    "EmbeddingConfig": "victor.storage.vector_stores",
    "EmbeddingModelConfig": "victor.storage.vector_stores",
    "EmbeddingSearchResult": "victor.storage.vector_stores",
    "SentenceTransformerModel": "victor.storage.vector_stores",
    "OllamaEmbeddingModel": "victor.storage.vector_stores",
    "OpenAIEmbeddingModel": "victor.storage.vector_stores",
    "CohereEmbeddingModel": "victor.storage.vector_stores",
    "create_embedding_model": "victor.storage.vector_stores",
    "EmbeddingRegistry": "victor.storage.vector_stores",
}


def __getattr__(name: str) -> Any:
    """Resolve embedding helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module 'victor_contracts.embeddings_runtime' has no attribute {name!r}"
        )

    module = importlib.import_module(module_name)
    return getattr(module, name)
