# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared encoder for multi-task RL learning.

This module provides embedding representations for tasks, providers,
and models that can be shared across different RL learners, enabling
transfer learning between verticals.

Architecture:
    ┌─────────────────────────────────────────┐
    │         SHARED ENCODER                   │
    │  (Task embedding + Provider embedding)   │
    └─────────────────┬───────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Coding  │  │ DevOps  │  │ Data    │
    │ Head    │  │ Head    │  │ Science │
    └─────────┘  └─────────┘  └─────────┘

Benefits:
- Transfer learning across verticals
- Better generalization for similar tasks
- Reduced data requirements per vertical

Sprint 5: Advanced RL Patterns
"""

import hashlib
from victor.core.json_utils import json_loads
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from victor.core.constants import DEFAULT_VERTICAL

logger = logging.getLogger(__name__)


@dataclass
class ContextEmbedding:
    """Embedding representation of a learning context.

    Provides a fixed-dimensional vector representation of the
    task context that can be used across different learners.

    Attributes:
        task_type: Original task type string
        provider: Provider name
        model: Model name
        vertical: Vertical (coding, devops, data_science)
        vector: Embedding vector
        metadata: Additional context metadata
    """

    task_type: str
    provider: str
    model: str
    vertical: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.vector)

    def similarity(self, other: "ContextEmbedding") -> float:
        """Compute cosine similarity with another embedding.

        Args:
            other: Another context embedding

        Returns:
            Cosine similarity [-1, 1]
        """
        if len(self.vector) != len(other.vector):
            return 0.0

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = math.sqrt(sum(a * a for a in self.vector))
        norm_b = math.sqrt(sum(b * b for b in other.vector))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class SharedEncoder:
    """Shared encoder for multi-task learning.

    Creates consistent embeddings for contexts (task_type, provider, model)
    that can be used across different RL learners. Uses learned embeddings
    that improve over time as more data is collected.

    Embedding Components:
    1. Task type embedding (semantic similarity between task types)
    2. Provider embedding (captures provider-specific characteristics)
    3. Model embedding (captures model capabilities)
    4. Vertical embedding (coding/devops/data_science)

    Usage:
        encoder = SharedEncoder()
        embedding = encoder.encode("analysis", "anthropic", "claude-3-opus", "coding")
        similar = encoder.find_similar(embedding, top_k=5)
    """

    # Embedding dimension for each component
    TASK_DIM = 8
    PROVIDER_DIM = 4
    MODEL_DIM = 4
    VERTICAL_DIM = 3
    TOTAL_DIM = TASK_DIM + PROVIDER_DIM + MODEL_DIM + VERTICAL_DIM  # 19

    # Known task type categories for semantic grouping
    TASK_CATEGORIES = {
        # Analysis tasks
        "analysis": ["analysis", "explain", "understand", "review", "audit"],
        # Search tasks
        "search": ["search", "find", "locate", "query", "lookup"],
        # Creation tasks
        "create": ["create", "generate", "write", "build", "implement"],
        # Modification tasks
        "edit": ["edit", "modify", "update", "change", "refactor"],
        # Action tasks
        "action": ["action", "execute", "run", "perform", "do"],
        # Debug tasks
        "debug": ["debug", "fix", "troubleshoot", "diagnose", "repair"],
    }

    # Known providers with characteristic vectors
    PROVIDER_CHARACTERISTICS = {
        "anthropic": [1.0, 0.9, 0.85, 0.9],  # [quality, safety, reasoning, tool_use]
        "openai": [0.9, 0.85, 0.9, 0.95],
        "google": [0.85, 0.8, 0.85, 0.85],
        "deepseek": [0.8, 0.7, 0.85, 0.8],
        "groq": [0.75, 0.75, 0.8, 0.8],
        "ollama": [0.7, 0.6, 0.7, 0.7],
        "lmstudio": [0.7, 0.6, 0.7, 0.7],
        "mistral": [0.8, 0.75, 0.8, 0.8],
    }

    # Vertical embeddings
    VERTICAL_EMBEDDINGS = {
        "coding": [1.0, 0.3, 0.2],
        "devops": [0.3, 1.0, 0.2],
        "data_science": [0.2, 0.2, 1.0],
        "research": [0.5, 0.2, 0.5],
        "general": [0.5, 0.5, 0.5],
    }

    def __init__(self, db_connection: Optional[Any] = None):
        """Initialize shared encoder.

        Args:
            db_connection: Optional SQLite connection for persistence
        """
        self.db = db_connection

        # Learned embeddings (updated over time)
        self._task_embeddings: Dict[str, List[float]] = {}
        self._model_embeddings: Dict[str, List[float]] = {}

        # Embedding cache for fast lookup
        self._cache: Dict[str, ContextEmbedding] = {}

        if db_connection:
            self._ensure_tables()
            self._load_embeddings()

    def _ensure_tables(self) -> None:
        """Create tables for embedding storage."""
        if not self.db:
            return

        cursor = self.db.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shared_embeddings (
                context_key TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                vertical TEXT NOT NULL,
                vector TEXT NOT NULL,
                usage_count INTEGER DEFAULT 1,
                last_updated TEXT NOT NULL
            )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_task_embeddings (
                task_type TEXT PRIMARY KEY,
                vector TEXT NOT NULL,
                sample_count INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL
            )
            """)

        self.db.commit()

    def _load_embeddings(self) -> None:
        """Load learned embeddings from database."""
        if not self.db:
            return

        cursor = self.db.cursor()

        try:
            cursor.execute("SELECT task_type, vector FROM learned_task_embeddings")
            for row in cursor.fetchall():
                self._task_embeddings[row[0]] = json_loads(row[1])

            logger.debug(f"SharedEncoder: Loaded {len(self._task_embeddings)} task embeddings")
        except Exception as e:
            logger.debug(f"SharedEncoder: Could not load embeddings: {e}")

    def encode(
        self,
        task_type: str,
        provider: str,
        model: str,
        vertical: str = DEFAULT_VERTICAL,
    ) -> ContextEmbedding:
        """Encode a context into an embedding.

        Args:
            task_type: Task type (analysis, action, create, etc.)
            provider: Provider name
            model: Model name
            vertical: Vertical (coding, devops, data_science)

        Returns:
            ContextEmbedding with fixed-dimensional vector
        """
        cache_key = f"{task_type}:{provider}:{model}:{vertical}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build embedding vector
        vector = []

        # Task type embedding
        vector.extend(self._encode_task_type(task_type))

        # Provider embedding
        vector.extend(self._encode_provider(provider))

        # Model embedding
        vector.extend(self._encode_model(model, provider))

        # Vertical embedding
        vector.extend(self._encode_vertical(vertical))

        embedding = ContextEmbedding(
            task_type=task_type,
            provider=provider,
            model=model,
            vertical=vertical,
            vector=vector,
        )

        self._cache[cache_key] = embedding
        return embedding

    def _encode_task_type(self, task_type: str) -> List[float]:
        """Encode task type into embedding.

        Uses semantic similarity to known task categories.

        Args:
            task_type: Task type string

        Returns:
            Embedding vector of dimension TASK_DIM
        """
        # Check for learned embedding
        if task_type in self._task_embeddings:
            return self._task_embeddings[task_type]

        # Compute similarity to each category
        task_lower = task_type.lower()
        category_scores = []

        for category, keywords in self.TASK_CATEGORIES.items():
            score = 0.0
            for keyword in keywords:
                if keyword in task_lower or task_lower in keyword:
                    score = 1.0
                    break
                # Partial match
                if any(c in task_lower for c in keyword[:3]) or task_lower[:3] in keyword:
                    score = max(score, 0.5)

            category_scores.append(score)

        # Pad to TASK_DIM
        while len(category_scores) < self.TASK_DIM:
            # Add hash-based features for unknown task types
            hash_val = int(hashlib.md5(task_type.encode()).hexdigest()[:8], 16)
            feature = (hash_val % 1000) / 1000.0
            category_scores.append(feature)
            hash_val //= 1000

        return category_scores[: self.TASK_DIM]

    def _encode_provider(self, provider: str) -> List[float]:
        """Encode provider into embedding.

        Args:
            provider: Provider name

        Returns:
            Embedding vector of dimension PROVIDER_DIM
        """
        provider_lower = provider.lower()

        if provider_lower in self.PROVIDER_CHARACTERISTICS:
            return self.PROVIDER_CHARACTERISTICS[provider_lower]

        # Unknown provider - use hash-based embedding
        hash_val = int(hashlib.md5(provider.encode()).hexdigest()[:8], 16)
        return [0.5 + 0.3 * ((hash_val >> (i * 8)) % 256) / 255.0 for i in range(self.PROVIDER_DIM)]

    def _encode_model(self, model: str, provider: str) -> List[float]:
        """Encode model into embedding.

        Args:
            model: Model name
            provider: Provider name (for context)

        Returns:
            Embedding vector of dimension MODEL_DIM
        """
        model_key = f"{provider}:{model}"

        if model_key in self._model_embeddings:
            return self._model_embeddings[model_key]

        # Extract features from model name
        model_lower = model.lower()
        features = []

        # Size indicator (larger models = higher capability)
        size_score = 0.5
        if any(x in model_lower for x in ["opus", "large", "70b", "72b"]):
            size_score = 1.0
        elif any(x in model_lower for x in ["sonnet", "medium", "32b", "34b"]):
            size_score = 0.8
        elif any(x in model_lower for x in ["haiku", "small", "7b", "8b"]):
            size_score = 0.5
        elif any(x in model_lower for x in ["mini", "tiny", "1b", "3b"]):
            size_score = 0.3
        features.append(size_score)

        # Instruction tuning indicator
        instruct_score = 0.5
        if any(x in model_lower for x in ["instruct", "chat", "it"]):
            instruct_score = 1.0
        features.append(instruct_score)

        # Code specialization
        code_score = 0.5
        if any(x in model_lower for x in ["code", "coder", "codex", "starcoder"]):
            code_score = 1.0
        features.append(code_score)

        # Version/generation (newer = better)
        version_score = 0.5
        if any(x in model_lower for x in ["4", "3.5", "v3", "v4"]):
            version_score = 0.9
        elif any(x in model_lower for x in ["3", "v2", "2.5"]):
            version_score = 0.7
        features.append(version_score)

        return features[: self.MODEL_DIM]

    def _encode_vertical(self, vertical: str) -> List[float]:
        """Encode vertical into embedding.

        Args:
            vertical: Vertical name

        Returns:
            Embedding vector of dimension VERTICAL_DIM
        """
        vertical_lower = vertical.lower()

        if vertical_lower in self.VERTICAL_EMBEDDINGS:
            return self.VERTICAL_EMBEDDINGS[vertical_lower]

        return self.VERTICAL_EMBEDDINGS["general"]

    def find_similar(
        self,
        embedding: ContextEmbedding,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[Tuple[ContextEmbedding, float]]:
        """Find similar contexts from cache.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (embedding, similarity) tuples
        """
        similarities = []

        for cached in self._cache.values():
            if cached.task_type == embedding.task_type and cached.provider == embedding.provider:
                continue  # Skip exact match

            sim = embedding.similarity(cached)
            if sim >= min_similarity:
                similarities.append((cached, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def update_task_embedding(
        self,
        task_type: str,
        gradient: List[float],
        learning_rate: float = 0.01,
    ) -> None:
        """Update learned task embedding based on gradient.

        Args:
            task_type: Task type to update
            gradient: Gradient direction
            learning_rate: Update rate
        """
        if task_type not in self._task_embeddings:
            self._task_embeddings[task_type] = self._encode_task_type(task_type)

        current = self._task_embeddings[task_type]
        updated = [c + learning_rate * g for c, g in zip(current, gradient[: self.TASK_DIM])]

        # Normalize
        norm = math.sqrt(sum(x * x for x in updated))
        if norm > 0:
            updated = [x / norm for x in updated]

        self._task_embeddings[task_type] = updated

        # Invalidate cache
        keys_to_remove = [k for k in self._cache if k.startswith(f"{task_type}:")]
        for key in keys_to_remove:
            del self._cache[key]

    def get_transfer_weight(
        self,
        source: ContextEmbedding,
        target: ContextEmbedding,
    ) -> float:
        """Compute transfer learning weight between contexts.

        Higher weight means more knowledge should transfer.

        Args:
            source: Source context
            target: Target context

        Returns:
            Transfer weight [0, 1]
        """
        # Base similarity
        similarity = source.similarity(target)

        # Boost for same vertical
        if source.vertical == target.vertical:
            similarity = min(1.0, similarity * 1.2)

        # Boost for same provider
        if source.provider == target.provider:
            similarity = min(1.0, similarity * 1.1)

        return max(0.0, similarity)

    def export_metrics(self) -> Dict[str, Any]:
        """Export encoder metrics.

        Returns:
            Dictionary with encoder stats
        """
        return {
            "cache_size": len(self._cache),
            "learned_task_embeddings": len(self._task_embeddings),
            "learned_model_embeddings": len(self._model_embeddings),
            "embedding_dimension": self.TOTAL_DIM,
        }


# Global singleton
_shared_encoder: Optional[SharedEncoder] = None


def get_shared_encoder(db_connection: Optional[Any] = None) -> SharedEncoder:
    """Get global shared encoder (lazy init).

    Args:
        db_connection: Optional database connection

    Returns:
        SharedEncoder singleton
    """
    global _shared_encoder
    if _shared_encoder is None:
        _shared_encoder = SharedEncoder(db_connection)
    return _shared_encoder
