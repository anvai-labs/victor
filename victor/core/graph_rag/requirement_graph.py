# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Requirement graph builder - maps requirements to code symbols.

This module implements the GraphCodeAgent pattern for requirement-to-code
traceability. It links natural language requirements (from issues, PRs,
user stories) to code symbols via semantic similarity and graph traversal.

PH5-001 to PH5-004: Requirement graph schema, extraction, mapping, and similarity.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.storage.graph.protocol import (
        GraphStoreProtocol,
        RequirementNode,
        GraphNode,
        GraphEdge,
    )

logger = logging.getLogger(__name__)


def _get_embedding_service() -> Any:
    """Shared EmbeddingService singleton, or None when unavailable."""
    try:
        from victor.storage.embeddings.service import get_embedding_service

        return get_embedding_service()
    except Exception:  # noqa: BLE001 - embeddings are strictly optional here
        return None


# =============================================================================
# Requirement Type Enum (PH5-001)
# =============================================================================


class RequirementType(str, Enum):
    """Type of requirement.

    Based on standard requirement categorization:
    - FEATURE: New functionality
    - BUG: Defect or issue
    - TASK: Work item or task
    - USER_STORY: Agile user story
    - EPIC: Large feature spanning multiple stories
    - TECHNICAL: Technical requirement or refactoring
    - PERFORMANCE: Performance-related requirement
    - SECURITY: Security-related requirement
    """

    FEATURE = "feature"
    BUG = "bug"
    TASK = "task"
    USER_STORY = "user_story"
    EPIC = "epic"
    TECHNICAL = "technical"
    PERFORMANCE = "performance"
    SECURITY = "security"


# =============================================================================
# Requirement Priority Enum (PH5-001)
# =============================================================================


class RequirementPriority(str, Enum):
    """Priority level for requirements.

    Uses MoSCoW method plus additional levels:
    - CRITICAL: Must have, blocks release
    - HIGH: Should have, important
    - MEDIUM: Could have, nice to have
    - LOW: Won't have this time
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Requirement Status Enum (PH5-001)
# =============================================================================


class RequirementStatus(str, Enum):
    """Status of a requirement in the lifecycle.

    Standard requirement lifecycle states.
    """

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


# =============================================================================
# Requirement Similarity Result (PH5-004)
# =============================================================================


@dataclass
class RequirementSimilarity:
    """Result from requirement similarity calculation.

    Attributes:
        requirement_id: First requirement ID
        similar_requirement_id: Second requirement ID
        similarity_score: Similarity score (0-1)
        similarity_type: Type of similarity (textual, semantic, structural)
    """

    requirement_id: str
    similar_requirement_id: str
    similarity_score: float
    similarity_type: str = "textual"


# =============================================================================
# Requirement Similarity Calculator (PH5-004)
# =============================================================================


class RequirementSimilarityCalculator:
    """Calculate similarity between requirements (PH5-004).

    This class implements multiple similarity strategies:
    1. Textual similarity (Jaccard, cosine on token overlap)
    2. Semantic similarity (using embeddings)
    3. Structural similarity (based on mapped symbols)

    Attributes:
        graph_store: Graph store for accessing requirements
        use_embeddings: Whether to use semantic similarity
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        use_embeddings: bool = True,
    ) -> None:
        """Initialize the similarity calculator.

        Args:
            graph_store: Graph store for accessing requirements
            use_embeddings: Whether to use semantic similarity. Safe to leave on:
                when the embedding service is unavailable, semantic scoring
                falls back to textual similarity.
        """
        self.graph_store = graph_store
        self.use_embeddings = use_embeddings

    async def find_similar_requirements(
        self,
        requirement_id: str,
        threshold: float = 0.5,
        max_results: int = 10,
        similarity_type: str = "textual",
    ) -> List[RequirementSimilarity]:
        """Find requirements similar to the given requirement.

        Args:
            requirement_id: Requirement ID to find similarities for
            threshold: Minimum similarity score (0-1)
            max_results: Maximum results to return
            similarity_type: Type of similarity ("textual", "semantic", "all")

        Returns:
            List of RequirementSimilarity results
        """
        # Get the source requirement
        source_req = await self.graph_store.get_node_by_id(requirement_id)
        if not source_req or source_req.type != "requirement":
            logger.warning(f"Requirement {requirement_id} not found")
            return []

        # Get all requirement nodes
        all_requirements = await self._get_all_requirements()

        # Calculate similarities
        similarities: List[RequirementSimilarity] = []

        for req in all_requirements:
            if req.node_id == requirement_id:
                continue

            if similarity_type in {"textual", "all"}:
                score = self._textual_similarity(source_req, req)
                if score >= threshold:
                    similarities.append(
                        RequirementSimilarity(
                            requirement_id=requirement_id,
                            similar_requirement_id=req.node_id,
                            similarity_score=score,
                            similarity_type="textual",
                        )
                    )

            if similarity_type in {"semantic", "all"} and self.use_embeddings:
                score = await self._semantic_similarity(source_req, req)
                if score >= threshold:
                    similarities.append(
                        RequirementSimilarity(
                            requirement_id=requirement_id,
                            similar_requirement_id=req.node_id,
                            similarity_score=score,
                            similarity_type="semantic",
                        )
                    )

        # Sort by score descending
        similarities.sort(key=lambda s: s.similarity_score, reverse=True)

        return similarities[:max_results]

    async def calculate_requirement_similarity_matrix(
        self,
        requirement_ids: List[str] | None = None,
        threshold: float = 0.5,
    ) -> Dict[Tuple[str, str], float]:
        """Calculate similarity matrix for requirements.

        Args:
            requirement_ids: List of requirement IDs (all if None)
            threshold: Minimum similarity to include in matrix

        Returns:
            Dict mapping (req_id_1, req_id_2) to similarity score
        """
        # Get requirements
        if requirement_ids is None:
            req_nodes = await self._get_all_requirements()
            requirement_ids = [r.node_id for r in req_nodes]

        # Calculate pairwise similarities
        matrix: Dict[Tuple[str, str], float] = {}

        for i, req_id_1 in enumerate(requirement_ids):
            for req_id_2 in requirement_ids[i + 1 :]:
                req_1 = await self.graph_store.get_node_by_id(req_id_1)
                req_2 = await self.graph_store.get_node_by_id(req_id_2)

                if req_1 and req_2:
                    score = self._textual_similarity(req_1, req_2)
                    if score >= threshold:
                        matrix[(req_id_1, req_id_2)] = score
                        matrix[(req_id_2, req_id_1)] = score

        return matrix

    async def create_similarity_edges(
        self,
        threshold: float = 0.6,
    ) -> int:
        """Create SEMANTIC_SIMILAR edges between similar requirements.

        Args:
            threshold: Minimum similarity for edge creation

        Returns:
            Number of edges created
        """
        from victor.storage.graph.protocol import GraphEdge
        from victor.storage.graph.edge_types import EdgeType

        # Get all requirements
        req_nodes = await self._get_all_requirements()
        req_ids = [r.node_id for r in req_nodes]

        # Calculate similarities and create edges
        edges: List[GraphEdge] = []
        edge_count = 0

        for i, req_id_1 in enumerate(req_ids):
            for req_id_2 in req_ids[i + 1 :]:
                req_1 = await self.graph_store.get_node_by_id(req_id_1)
                req_2 = await self.graph_store.get_node_by_id(req_id_2)

                if req_1 and req_2:
                    score = self._textual_similarity(req_1, req_2)
                    if score >= threshold:
                        edge = GraphEdge(
                            src=req_id_1,
                            dst=req_id_2,
                            type=EdgeType.SEMANTIC_SIMILAR,
                            weight=score,
                        )
                        edges.append(edge)
                        edge_count += 1

        # Upsert edges
        if edges:
            await self.graph_store.upsert_edges(edges)

        return edge_count

    async def _get_all_requirements(self) -> List[GraphNode]:
        """Get all requirement nodes from the graph.

        Returns:
            List of requirement nodes
        """
        try:
            # Try to get nodes by type
            nodes = await self.graph_store.find_nodes(type="requirement")
            return nodes
        except Exception as e:
            logger.warning(f"Error getting requirements: {e}")
            return []

    def _textual_similarity(self, req_1: GraphNode, req_2: GraphNode) -> float:
        """Calculate textual similarity between requirements.

        Args:
            req_1: First requirement
            req_2: Second requirement

        Returns:
            Similarity score (0-1)
        """
        # Get text content
        text_1 = self._get_requirement_text(req_1)
        text_2 = self._get_requirement_text(req_2)

        # Tokenize
        tokens_1 = self._tokenize(text_1)
        tokens_2 = self._tokenize(text_2)

        if not tokens_1 or not tokens_2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens_1 & tokens_2)
        union = len(tokens_1 | tokens_2)

        if union == 0:
            return 0.0

        return intersection / union

    async def _semantic_similarity(
        self,
        req_1: GraphNode,
        req_2: GraphNode,
    ) -> float:
        """Calculate semantic similarity via embedding cosine.

        Embeddings come from the shared EmbeddingService (cached, so repeated
        pairwise comparisons re-embed nothing). Falls back to textual
        similarity when the service is unavailable. Negative cosine is clamped
        to 0 to honor the 0-1 score contract.

        Args:
            req_1: First requirement
            req_2: Second requirement

        Returns:
            Similarity score (0-1)
        """
        service = _get_embedding_service()
        if service is None:
            return self._textual_similarity(req_1, req_2)
        try:
            import numpy as np

            v1 = await service.embed_text(self._get_requirement_text(req_1), use_cache=True)
            v2 = await service.embed_text(self._get_requirement_text(req_2), use_cache=True)
            if not (np.any(v1) and np.any(v2)):
                # Degraded service (model unavailable emits zero vectors).
                return self._textual_similarity(req_1, req_2)
            return max(0.0, min(1.0, service.cosine_similarity(v1, v2)))
        except Exception as e:
            logger.debug(f"Semantic similarity fell back to textual: {e}")
            return self._textual_similarity(req_1, req_2)

    def _get_requirement_text(self, req: GraphNode) -> str:
        """Extract text content from a requirement node.

        Args:
            req: Requirement node

        Returns:
            Combined text content
        """
        parts = [req.name]

        # Get description from metadata
        description = req.metadata.get("description", "") if req.metadata else ""
        if description:
            parts.append(description)

        return " ".join(parts)

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            Set of lowercase tokens
        """
        # Extract words
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
        }

        return {w for w in words if len(w) > 2 and w not in stop_words}


@dataclass
class RequirementMapping:
    """Result from requirement-to-code mapping.

    Attributes:
        requirement_id: Requirement node ID
        title: Requirement title
        description: Requirement description
        mapped_symbols: List of symbol IDs that satisfy this requirement
        confidence_scores: Confidence scores for each mapping
        mapping_method: Method used for mapping (semantic, graph, hybrid)
    """

    requirement_id: str
    title: str
    description: str | None = None
    mapped_symbols: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    mapping_method: str = "semantic"


@dataclass
class RequirementSource:
    """Source for requirements (issues, PRs, etc.).

    Attributes:
        source_type: Type of source (github_issue, jira, slack, etc.)
        source_path: Path or URL to source
        priority: Priority weighting for this source
    """

    source_type: str
    source_path: str | Path
    priority: float = 1.0


class RequirementGraphBuilder:
    """Map requirements to code symbols (GraphCodeAgent pattern).

    This class implements the requirement-to-code mapping methodology:
    1. Parse requirements from various sources
    2. Generate embeddings for requirements
    3. Find semantically similar code symbols
    4. Traverse graph to find dependencies
    5. Create SATISFIES/TESTS edges

    Attributes:
        graph_store: Graph store for persistence
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
    ) -> None:
        """Initialize the requirement graph builder.

        Args:
            graph_store: Graph store for persistence
        """
        self.graph_store = graph_store

    async def map_requirement(
        self,
        requirement: str,
        requirement_type: str = "feature",
        source: str | None = None,
        max_symbols: int = 10,
    ) -> RequirementMapping:
        """Map a requirement to code symbols.

        Args:
            requirement: Requirement text (title or description)
            requirement_type: Type of requirement (feature, bug, task, user_story)
            source: Optional source identifier
            max_symbols: Maximum number of symbols to map

        Returns:
            RequirementMapping with mapped symbols
        """
        # Generate requirement ID
        req_id = self._generate_requirement_id(requirement, source)

        # Extract title and description
        title, description = self._parse_requirement_text(requirement)

        # Create requirement node
        await self._create_requirement_node(
            req_id,
            requirement_type,
            source,
            title,
            description,
        )

        # Find semantically similar symbols
        mapped_symbols = await self._find_similar_symbols(
            req_id,
            requirement,
            max_symbols,
        )

        # Create SATISFIES edges
        await self._create_requirement_edges(
            req_id,
            mapped_symbols,
        )

        return RequirementMapping(
            requirement_id=req_id,
            title=title,
            description=description,
            mapped_symbols=[s.symbol_id for s in mapped_symbols],
            confidence_scores={s.symbol_id: s.confidence for s in mapped_symbols},
            mapping_method="semantic",
        )

    async def map_requirements_from_file(
        self,
        file_path: Path,
        requirement_type: str = "feature",
    ) -> List[RequirementMapping]:
        """Map multiple requirements from a file.

        Args:
            file_path: Path to requirements file (one per line or markdown)
            requirement_type: Type of requirements

        Returns:
            List of RequirementMapping results
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read requirements file {file_path}: {e}")
            return []

        # Parse requirements (one per line, or markdown headers)
        requirements = self._parse_requirements_file(content)

        # Map each requirement
        mappings: List[RequirementMapping] = []
        for req_text in requirements:
            mapping = await self.map_requirement(
                req_text,
                requirement_type,
                source=str(file_path),
            )
            mappings.append(mapping)

        return mappings

    def _generate_requirement_id(
        self,
        requirement: str,
        source: str | None = None,
    ) -> str:
        """Generate unique ID for a requirement.

        Args:
            requirement: Requirement text
            source: Optional source identifier

        Returns:
            Unique requirement ID
        """
        content = f"{requirement}:{source or ''}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _parse_requirement_text(
        self,
        requirement: str,
    ) -> tuple[str, str | None]:
        """Parse requirement text into title and description.

        Args:
            requirement: Raw requirement text

        Returns:
            Tuple of (title, description)
        """
        # If newline present, first line is title
        if "\n" in requirement:
            title, desc = requirement.split("\n", 1)
            return title.strip(), desc.strip()

        # Use first ~50 chars as title
        if len(requirement) > 50:
            return requirement[:50].strip() + "...", requirement.strip()

        return requirement.strip(), None

    def _parse_requirements_file(self, content: str) -> List[str]:
        """Parse requirements from file content.

        Args:
            content: File content

        Returns:
            List of requirement texts
        """
        requirements: List[str] = []

        # Try markdown format (# headers)
        lines = content.splitlines()
        current_req: List[str] = []

        for line in lines:
            if line.startswith("# "):
                # New requirement
                if current_req:
                    requirements.append("\n".join(current_req).strip())
                current_req = [line[2:].strip()]
            elif line.strip() and current_req:
                current_req.append(line.strip())

        if current_req:
            requirements.append("\n".join(current_req).strip())

        # If no markdown format found, split by blank lines
        if not requirements:
            requirements = [r.strip() for r in content.split("\n\n") if r.strip()]

        return requirements

    async def _create_requirement_node(
        self,
        req_id: str,
        req_type: str,
        source: str | None,
        title: str,
        description: str | None,
    ) -> None:
        """Create a requirement node in the graph.

        Args:
            req_id: Requirement ID
            req_type: Requirement type
            source: Source identifier
            title: Requirement title
            description: Requirement description
        """
        from victor.storage.graph.protocol import GraphNode

        node = GraphNode(
            node_id=req_id,
            type="requirement",
            name=title,
            file="",
            metadata={
                "requirement_type": req_type,
                "source": source or "",
                "description": description or "",
            },
        )

        await self.graph_store.upsert_nodes([node])

    async def _find_similar_symbols(
        self,
        req_id: str,
        requirement: str,
        max_symbols: int,
    ) -> List[Any]:
        """Find semantically similar symbols for a requirement.

        FTS candidates are re-scored with real embedding cosine between the
        requirement text and each symbol's embedded text (name + signature +
        docstring — the same text the indexing pipeline embeds), replacing the
        old hardcoded 0.8 confidence. Without an embedding service the FTS
        candidates keep a flat default confidence.

        Args:
            req_id: Requirement ID
            requirement: Requirement text
            max_symbols: Maximum symbols to return

        Returns:
            List of similar symbols with confidence scores, best first
        """
        try:
            # Over-fetch so semantic re-ranking has candidates to choose from.
            nodes = await self.graph_store.search_symbols(
                requirement,
                limit=max(max_symbols * 2, max_symbols),
            )
        except Exception as e:
            logger.warning(f"Symbol search failed: {e}")
            return []
        if not nodes:
            return []

        confidences = await self._score_symbols(requirement, nodes)
        scored = sorted(zip(nodes, confidences), key=lambda pair: pair[1], reverse=True)
        return [
            type(
                "Symbol",
                (),
                {
                    "symbol_id": n.node_id,
                    "confidence": confidence,
                    "node": n,
                },
            )()
            for n, confidence in scored[:max_symbols]
        ]

    async def _score_symbols(self, requirement: str, nodes: List[Any]) -> List[float]:
        """Embedding-cosine confidence per candidate symbol (0-1, clamped).

        Falls back to a flat 0.8 (the historical default) when the embedding
        service is unavailable, so mapping still works air-gapped-without-model.
        """
        service = _get_embedding_service()
        if service is None:
            return [0.8] * len(nodes)
        try:
            import numpy as np

            from victor.processing.graph_embeddings import GraphAwareEmbedder

            req_vector = await service.embed_text(requirement, use_cache=True)
            if not np.any(req_vector):
                # Degraded service (model unavailable emits zero vectors) —
                # cosine would be 0 everywhere and silently drop all mappings.
                return [0.8] * len(nodes)
            symbol_vectors = await service.embed_batch(
                [GraphAwareEmbedder.node_text(n) for n in nodes]
            )
            return [
                max(0.0, min(1.0, service.cosine_similarity(req_vector, vec)))
                for vec in symbol_vectors
            ]
        except Exception as e:
            logger.debug(f"Semantic symbol scoring fell back to default confidence: {e}")
            return [0.8] * len(nodes)

    async def _create_requirement_edges(
        self,
        req_id: str,
        mapped_symbols: List[Any],
    ) -> None:
        """Create edges between requirement and mapped symbols.

        Args:
            req_id: Requirement ID
            mapped_symbols: Mapped symbols with confidence
        """
        from victor.storage.graph.protocol import GraphEdge
        from victor.storage.graph.edge_types import EdgeType

        edges: List[GraphEdge] = []

        for symbol in mapped_symbols:
            if symbol.confidence > 0.5:  # Only create edges for confident matches
                edge = GraphEdge(
                    src=req_id,
                    dst=symbol.symbol_id,
                    type=EdgeType.SATISFIES,
                    weight=symbol.confidence,
                )
                edges.append(edge)

        if edges:
            await self.graph_store.upsert_edges(edges)


__all__ = [
    # Schema (PH5-001)
    "RequirementType",
    "RequirementPriority",
    "RequirementStatus",
    # Mapping (PH5-003)
    "RequirementMapping",
    "RequirementSource",
    "RequirementGraphBuilder",
    # Similarity (PH5-004)
    "RequirementSimilarity",
    "RequirementSimilarityCalculator",
]
