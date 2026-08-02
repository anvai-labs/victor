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

"""Hybrid search combining semantic and keyword search using RRF algorithm.

This module implements Reciprocal Rank Fusion (RRF) to combine results from:
1. Semantic search (embedding-based similarity)
2. Keyword search (term-frequency exact matching)

RRF Algorithm:
    For each result r and ranking source s:
        RRF_score(r) = Σ_s weight_s / (k + rank_s(r))

    where:
        - k = 60 (standard RRF parameter)
        - rank_s(r) = position of result r in ranking s (starting from 0)
        - weight_s = weight for ranking source s

This approach provides:
- **Better recall** than pure semantic search (catches exact term matches)
- **Better precision** than pure keyword search (understands concepts)
- **Robustness** to individual search failures

This module is domain-agnostic and can be used by any vertical:
- Coding: Code search with symbol understanding
- Research: Document search with concept matching
- RAG: Knowledge retrieval with semantic similarity
- DevOps: Log search with pattern matching
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from victor.core.search_types import SearchHit

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search with combined score.

    Attributes:
        file_path: Path to the file/document
        content: The content that matched
        combined_score: RRF-combined score
        semantic_score: Score from semantic search (0-1)
        keyword_score: Score from keyword search
        semantic_rank: Rank in semantic results (-1 if not found)
        keyword_rank: Rank in keyword results (-1 if not found)
        line_number: Line number in file (if applicable)
        metadata: Additional result metadata
    """

    file_path: str
    content: str
    combined_score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    semantic_rank: int = -1  # -1 if not in semantic results
    keyword_rank: int = -1  # -1 if not in keyword results
    line_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_search_hit(self) -> SearchHit:
        """Convert to a generic SearchHit for cross-layer consumers."""
        return SearchHit(
            file_path=self.file_path,
            content=self.content,
            score=self.combined_score,
            line_number=self.line_number or None,
            metadata=self.metadata,
        )


class HybridSearchEngine:
    """Hybrid search engine combining semantic and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from different
    search methods, providing better coverage than either method alone.

    Example:
        >>> engine = HybridSearchEngine(semantic_weight=0.6, keyword_weight=0.4)
        >>> results = engine.combine_results(semantic_results, keyword_results)
        >>> for result in results:
        ...     print(f"{result.file_path}: {result.combined_score:.4f}")
    """

    def __init__(
        self,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        """Initialize hybrid search engine.

        Args:
            semantic_weight: Weight for semantic search (0.0-1.0)
            keyword_weight: Weight for keyword search (0.0-1.0)
            rrf_k: RRF parameter (typically 60)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

    def combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        max_results: int = 10,
    ) -> List[HybridSearchResult]:
        """Combine semantic and keyword search results using RRF.

        Args:
            semantic_results: Results from semantic search (list of dicts)
            keyword_results: Results from keyword search (list of dicts)
            max_results: Maximum number of results to return

        Returns:
            Combined and ranked results
        """
        semantic_results = self._filter_ranked_results(semantic_results)
        keyword_results = self._filter_ranked_results(keyword_results)

        # Build mapping of file_path -> (semantic_rank, semantic_score)
        semantic_map: Dict[str, Tuple[int, float]] = {}
        for rank, result in enumerate(semantic_results):
            file_path = result.get("file_path", "")
            score = result.get("score", 0.0)
            semantic_map[file_path] = (rank, score)

        # Build mapping of file_path -> (keyword_rank, keyword_score)
        keyword_map: Dict[str, Tuple[int, float]] = {}
        for rank, result in enumerate(keyword_results):
            file_path = result.get("file_path", "")
            score = result.get("score", 0.0)
            keyword_map[file_path] = (rank, score)

        # Get all unique file paths
        all_paths = set(semantic_map.keys()) | set(keyword_map.keys())

        # Compute RRF scores
        hybrid_results = []

        for file_path in all_paths:
            # Get semantic ranking/score
            semantic_rank, semantic_score = semantic_map.get(file_path, (-1, 0.0))
            semantic_rrf = (
                self.semantic_weight / (self.rrf_k + semantic_rank) if semantic_rank >= 0 else 0.0
            )

            # Get keyword ranking/score
            keyword_rank, keyword_score = keyword_map.get(file_path, (-1, 0.0))
            keyword_rrf = (
                self.keyword_weight / (self.rrf_k + keyword_rank) if keyword_rank >= 0 else 0.0
            )

            # Combined RRF score
            combined_score = semantic_rrf + keyword_rrf

            # Get result details from whichever source has it
            if semantic_rank >= 0:
                source_result = semantic_results[semantic_rank]
            else:
                source_result = keyword_results[keyword_rank]

            hybrid_result = HybridSearchResult(
                file_path=file_path,
                content=source_result.get("content", ""),
                combined_score=combined_score,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                semantic_rank=semantic_rank,
                keyword_rank=keyword_rank,
                line_number=source_result.get("line_number", 0),
                metadata=source_result.get("metadata", {}),
            )

            hybrid_results.append(hybrid_result)

        # Sort by combined score (descending)
        hybrid_results.sort(key=lambda r: r.combined_score, reverse=True)

        # Limit results
        hybrid_results = hybrid_results[:max_results]

        logger.debug(
            f"Hybrid search combined {len(semantic_results)} semantic + "
            f"{len(keyword_results)} keyword results → {len(hybrid_results)} final results"
        )

        return hybrid_results

    def _filter_ranked_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop empty or non-positive results before applying rank-only fusion."""
        filtered: List[Dict[str, Any]] = []
        for result in results:
            file_path = str(result.get("file_path", "") or "").strip()
            score = result.get("score", 0.0)
            if not file_path:
                continue
            if not isinstance(score, (int, float)) or score <= 0:
                continue
            filtered.append(result)
        return filtered

    def keyword_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Simple keyword search using TF-based scoring.

        Args:
            query: Search query
            documents: List of documents to search (each with 'file_path' and 'content')
            max_results: Maximum results to return

        Returns:
            Ranked search results
        """
        query_terms = query.lower().split()

        # Single-pass scoring: join terms into one alternation regex
        # so we scan each document once instead of len(terms) times.
        import re

        if query_terms:
            term_pattern = re.compile("|".join(re.escape(t) for t in query_terms))
        else:
            return []

        scores = []
        for doc in documents:
            content = doc.get("content", "").lower()
            file_path = doc.get("file_path", "")

            # Single regex pass counts all term matches
            score = float(len(term_pattern.findall(content)))

            if score > 0:
                scores.append(
                    {
                        "file_path": file_path,
                        "content": doc.get("content", ""),
                        "score": score,
                        "line_number": doc.get("line_number", 0),
                        "metadata": doc.get("metadata", {}),
                    }
                )

        # Sort by score (descending)
        scores.sort(key=lambda x: x["score"], reverse=True)

        return scores[:max_results]

    def explain_ranking(self, result: HybridSearchResult) -> str:
        """Generate explanation for why a result was ranked.

        Args:
            result: Hybrid search result

        Returns:
            Human-readable explanation
        """
        lines = [
            f"File: {result.file_path}",
            f"Combined Score: {result.combined_score:.4f}",
        ]

        if result.semantic_rank >= 0:
            lines.append(
                f"  Semantic: rank #{result.semantic_rank + 1}, "
                f"score {result.semantic_score:.4f}"
            )
        else:
            lines.append("  Semantic: not in results")

        if result.keyword_rank >= 0:
            lines.append(
                f"  Keyword: rank #{result.keyword_rank + 1}, " f"score {result.keyword_score:.1f}"
            )
        else:
            lines.append("  Keyword: not in results")

        # Explain why it ranked
        if result.semantic_rank >= 0 and result.keyword_rank >= 0:
            lines.append("  → Found by BOTH semantic and keyword search (high confidence)")
        elif result.semantic_rank >= 0:
            lines.append("  → Found only by semantic search (concept match)")
        else:
            lines.append("  → Found only by keyword search (exact term match)")

        return "\n".join(lines)


def create_hybrid_search_engine(
    semantic_weight: float = 0.6, keyword_weight: float = 0.4
) -> HybridSearchEngine:
    """Create a hybrid search engine with specified weights.

    Args:
        semantic_weight: Weight for semantic search (0.0-1.0)
        keyword_weight: Weight for keyword search (0.0-1.0)

    Returns:
        Configured HybridSearchEngine instance

    Example:
        >>> engine = create_hybrid_search_engine(semantic_weight=0.6, keyword_weight=0.4)
        >>> hybrid_results = engine.combine_results(semantic_results, keyword_results)
    """
    # Normalize weights to sum to 1.0
    total = semantic_weight + keyword_weight
    if total > 0:
        semantic_weight /= total
        keyword_weight /= total

    return HybridSearchEngine(
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        rrf_k=60,
    )


__all__ = [
    "HybridSearchEngine",
    "HybridSearchResult",
    "create_hybrid_search_engine",
]
