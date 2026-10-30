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

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

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


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
