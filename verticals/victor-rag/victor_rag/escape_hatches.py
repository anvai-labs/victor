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

"""Escape hatches for RAG YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_retrieval
      type: condition
      condition: "retrieval_quality"  # References escape hatch
      branches:
        "sufficient": synthesize
        "needs_more": expand_search
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

RETRIEVAL_GAP_TYPES = {
    "missing_support",
    "weak_authority",
    "contradictory_evidence",
    "query_ambiguity",
    "low_utility",
    "answer_revision",
}


# =============================================================================
# Condition Functions
# =============================================================================


def retrieval_quality(ctx: Dict[str, Any]) -> str:
    """Assess retrieval quality based on relevance scores.

    Args:
        ctx: Workflow context with keys:
            - retrieved_chunks (list): Retrieved document chunks
            - min_relevance (float): Minimum relevance threshold
            - min_chunks (int): Minimum required chunks

    Returns:
        "sufficient", "marginal", or "insufficient"
    """
    chunks = ctx.get("retrieved_chunks", [])
    min_relevance = ctx.get("min_relevance", 0.7)
    min_chunks = ctx.get("min_chunks", 3)

    if not chunks:
        return "insufficient"

    # Calculate relevance scores
    relevant_chunks = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            score = chunk.get("relevance_score", chunk.get("score", 0))
            if score >= min_relevance:
                relevant_chunks.append(chunk)

    if len(relevant_chunks) >= min_chunks:
        return "sufficient"

    if len(relevant_chunks) > 0:
        return "marginal"

    return "insufficient"


def document_quality(ctx: Dict[str, Any]) -> str:
    """Assess document quality for ingestion.

    Args:
        ctx: Workflow context with keys:
            - parsed_content (dict): Parsed document content
            - min_length (int): Minimum content length
            - language (str): Expected language

    Returns:
        "high", "acceptable", "low", or "skip"
    """
    content = ctx.get("parsed_content", {})
    min_length = ctx.get("min_length", 100)

    text = content.get("text", "")
    metadata = content.get("metadata", {})

    text_length = len(text) if isinstance(text, str) else 0

    if text_length < min_length:
        return "skip"

    # Check for quality indicators
    extraction_confidence = metadata.get("confidence", 0.5)

    if text_length > 500 and extraction_confidence >= 0.8:
        return "high"

    if extraction_confidence >= 0.5:
        return "acceptable"

    return "low"


def chunking_strategy(ctx: Dict[str, Any]) -> str:
    """Determine optimal chunking strategy based on document type.

    Args:
        ctx: Workflow context with keys:
            - document_type (str): Type of document (pdf, code, markdown, etc.)
            - content_length (int): Total content length
            - has_structure (bool): Whether document has clear structure

    Returns:
        "semantic", "fixed", "sentence", or "paragraph"
    """
    doc_type = ctx.get("document_type", "text")
    content_length = ctx.get("content_length", 0)
    has_structure = ctx.get("has_structure", False)

    if doc_type in ("code", "python", "javascript", "typescript"):
        return "semantic"

    if doc_type == "markdown" and has_structure:
        return "paragraph"

    if content_length > 10000:
        return "semantic"

    if content_length < 1000:
        return "sentence"

    return "fixed"


def should_reindex(ctx: Dict[str, Any]) -> str:
    """Determine if document should be reindexed.

    Args:
        ctx: Workflow context with keys:
            - document_hash (str): Current document hash
            - stored_hash (str): Previously stored hash
            - force_reindex (bool): Force reindex flag
            - last_indexed (str): Last indexing timestamp

    Returns:
        "reindex" or "skip"
    """
    current_hash = ctx.get("document_hash", "")
    stored_hash = ctx.get("stored_hash", "")
    force = ctx.get("force_reindex", False)

    if force:
        return "reindex"

    if not stored_hash:
        return "reindex"

    if current_hash != stored_hash:
        return "reindex"

    return "skip"


def query_complexity(ctx: Dict[str, Any]) -> str:
    """Assess query complexity for routing.

    Args:
        ctx: Workflow context with keys:
            - query (str): User query
            - intent (str): Detected intent
            - entities (list): Extracted entities

    Returns:
        "simple", "moderate", "complex", or "multi_step"
    """
    query = ctx.get("query", "")
    intent = ctx.get("intent", "")
    entities = ctx.get("entities", [])

    query_length = len(query.split()) if isinstance(query, str) else 0
    entity_count = len(entities) if isinstance(entities, list) else 0

    if intent in ("comparison", "analysis", "synthesis"):
        return "multi_step"

    if query_length > 20 or entity_count > 3:
        return "complex"

    if query_length > 10 or entity_count > 1:
        return "moderate"

    return "simple"


def answer_confidence(ctx: Dict[str, Any]) -> str:
    """Assess confidence in generated answer.

    Args:
        ctx: Workflow context with keys:
            - answer (str): Generated answer
            - source_count (int): Number of sources used
            - relevance_scores (list): Chunk relevance scores

    Returns:
        "high", "medium", "low", or "uncertain"
    """
    answer = ctx.get("answer", "")
    source_count = ctx.get("source_count", 0)
    relevance_scores = ctx.get("relevance_scores", [])

    if not answer:
        return "uncertain"

    if source_count == 0:
        return "uncertain"

    avg_relevance = 0
    if relevance_scores:
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

    if source_count >= 3 and avg_relevance >= 0.8:
        return "high"

    if source_count >= 2 and avg_relevance >= 0.6:
        return "medium"

    if source_count >= 1 and avg_relevance >= 0.4:
        return "low"

    return "uncertain"


def retrieval_repair_decision(ctx: Dict[str, Any]) -> str:
    """Choose whether to repair retrieval, revise, or ask for clarification.

    Args:
        ctx: Workflow context with keys:
            - coverage_assessment (dict): Coverage result from context check
            - verification (dict): Verification result for generated answer
            - retrieval_utility (dict): Utility-aware retrieval scores
            - repair_attempt_count (int): Number of automatic repair attempts
            - max_repair_attempts (int): Maximum automatic repair attempts

    Returns:
        "repair", "revise", or "clarify"
    """
    coverage = ctx.get("coverage_assessment", {}) or {}
    repair_attempt_count = ctx.get("repair_attempt_count", 0)
    max_repair_attempts = ctx.get("max_repair_attempts", 2)

    has_answer = bool(coverage.get("has_answer", False))
    gap_type = classify_retrieval_gap(ctx)

    if repair_attempt_count >= max_repair_attempts:
        if gap_type == "answer_revision" and has_answer:
            return "revise"
        return "clarify"

    if gap_type == "query_ambiguity":
        return "clarify"

    if gap_type in {
        "missing_support",
        "weak_authority",
        "contradictory_evidence",
        "low_utility",
    }:
        return "repair"

    if gap_type == "answer_revision":
        return "revise"

    return "clarify"


def classify_retrieval_gap(ctx: Dict[str, Any]) -> str:
    """Classify the dominant retrieval gap for repair-policy routing.

    Returns one of:
    - "missing_support"
    - "weak_authority"
    - "contradictory_evidence"
    - "query_ambiguity"
    - "low_utility"
    - "answer_revision"
    """
    diagnosis = ctx.get("retrieval_gap_diagnosis", {}) or {}
    coverage = ctx.get("coverage_assessment", {}) or {}
    verification = ctx.get("verification", {}) or {}
    utility = ctx.get("retrieval_utility", {}) or {}

    explicit_gap_type = _extract_explicit_gap_type(diagnosis)
    if explicit_gap_type:
        return explicit_gap_type

    has_answer = bool(coverage.get("has_answer", False))
    confidence = str(coverage.get("confidence", "")).lower()
    utility_score = float(utility.get("utility_score", 0.0) or 0.0)
    authority_hits = int(utility.get("authority_hits", 0) or 0)
    verification_passed = verification.get("passed", True)
    issues = verification.get("issues", []) or []
    issue_text = " ".join(str(issue).lower() for issue in issues)

    if any(
        phrase in issue_text
        for phrase in ("ambiguous", "unclear", "clarify", "underspecified", "specify")
    ):
        return "query_ambiguity"

    if any(
        phrase in issue_text for phrase in ("conflict", "contradict", "inconsistent", "disagree")
    ):
        return "contradictory_evidence"

    if any(
        phrase in issue_text
        for phrase in (
            "weak source",
            "unreliable source",
            "authority",
            "authoritative",
            "credibility",
        )
    ):
        return "weak_authority"

    if any(
        phrase in issue_text
        for phrase in (
            "missing support",
            "missing evidence",
            "unsupported",
            "citation",
            "grounding",
        )
    ):
        return "missing_support"

    if not has_answer or confidence in {"low", "none", "uncertain"}:
        return "missing_support"

    if utility_score < 0.55 or authority_hits == 0:
        return "low_utility"

    if verification_passed is False and has_answer:
        return "answer_revision"

    return "low_utility"


def _extract_explicit_gap_type(diagnosis: Any) -> str:
    """Normalize gap-type data coming from the diagnosis node."""
    if isinstance(diagnosis, dict):
        for key in ("gap_type", "type", "diagnosis"):
            value = diagnosis.get(key)
            normalized = _normalize_gap_type(value)
            if normalized:
                return normalized
        return ""

    return _normalize_gap_type(diagnosis)


def _normalize_gap_type(value: Any) -> str:
    """Normalize gap-type aliases to a stable vocabulary."""
    if not value:
        return ""

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "missing_evidence": "missing_support",
        "low_support": "missing_support",
        "weak_sources": "weak_authority",
        "source_quality": "weak_authority",
        "conflicting_evidence": "contradictory_evidence",
        "contradiction": "contradictory_evidence",
        "ambiguous_query": "query_ambiguity",
        "ambiguity": "query_ambiguity",
        "clarification": "query_ambiguity",
        "revise": "answer_revision",
        "revision": "answer_revision",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in RETRIEVAL_GAP_TYPES else ""


def embedding_batch_size(ctx: Dict[str, Any]) -> str:
    """Determine optimal embedding batch size.

    Args:
        ctx: Workflow context with keys:
            - chunk_count (int): Number of chunks to embed
            - available_memory (int): Available memory in MB
            - model_name (str): Embedding model name

    Returns:
        "small", "medium", "large", or "xlarge"
    """
    chunk_count = ctx.get("chunk_count", 0)
    available_memory = ctx.get("available_memory", 4096)

    if chunk_count <= 10:
        return "small"

    if available_memory < 2048:
        return "small"

    if chunk_count <= 100:
        return "medium"

    if chunk_count <= 1000:
        return "large"

    return "xlarge"


# =============================================================================
# Transform Functions
# =============================================================================


def merge_retrieved_chunks(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge and deduplicate retrieved chunks from multiple sources.

    Args:
        ctx: Workflow context with parallel retrieval results

    Returns:
        Merged and ranked chunks
    """
    semantic_results = ctx.get("semantic_results", [])
    keyword_results = ctx.get("keyword_results", [])
    graph_results = ctx.get("graph_results", [])

    all_chunks = []
    seen_ids = set()

    for results, source_type in [
        (semantic_results, "semantic"),
        (keyword_results, "keyword"),
        (graph_results, "graph"),
    ]:
        if isinstance(results, list):
            for chunk in results:
                if isinstance(chunk, dict):
                    chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
                    if chunk_id and chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        chunk["retrieval_source"] = source_type
                        all_chunks.append(chunk)

    # Sort by relevance score
    all_chunks.sort(key=lambda x: x.get("relevance_score", x.get("score", 0)), reverse=True)

    return {
        "chunks": all_chunks,
        "total_count": len(all_chunks),
        "by_source": {
            "semantic": len([c for c in all_chunks if c.get("retrieval_source") == "semantic"]),
            "keyword": len([c for c in all_chunks if c.get("retrieval_source") == "keyword"]),
            "graph": len([c for c in all_chunks if c.get("retrieval_source") == "graph"]),
        },
    }


def format_context_window(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Format retrieved chunks into context window for LLM.

    Args:
        ctx: Workflow context with ranked_chunks

    Returns:
        Formatted context with citations
    """
    chunks = ctx.get("ranked_chunks", [])
    max_tokens = ctx.get("max_context_tokens", 4000)

    context_parts = []
    citations = []
    current_tokens = 0

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = chunk.get("text", chunk.get("content", ""))
            source = chunk.get("source", chunk.get("document_id", f"source_{i}"))

            # Rough token estimation (4 chars per token)
            chunk_tokens = len(text) // 4

            if current_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(f"[{i+1}] {text}")
            citations.append(
                {
                    "index": i + 1,
                    "source": source,
                    "relevance": chunk.get("relevance_score", 0),
                }
            )
            current_tokens += chunk_tokens

    return {
        "context": "\n\n".join(context_parts),
        "citations": citations,
        "token_count": current_tokens,
        "chunks_used": len(context_parts),
    }


def score_retrieval_utility(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Score retrieval utility and apply bounded reranking adjustments."""
    ranked_results = list(ctx.get("ranked_results", []) or [])
    rescored_results = []
    unique_sources = set()
    seen_content = set()
    authority_hits = 0
    repeated_source_hits = 0
    duplicate_content_hits = 0
    total_relevance = 0.0

    for position, chunk in enumerate(ranked_results):
        relevance = _chunk_relevance(chunk)
        total_relevance += relevance

        source_key = _chunk_source_key(chunk)
        content_key = _chunk_content_key(chunk)
        has_source = bool(source_key)
        source_novelty_bonus = 0.0
        content_redundancy_penalty = 0.0

        if has_source:
            authority_hits += 1
            if source_key in unique_sources:
                repeated_source_hits += 1
                source_novelty_bonus = -0.03
            else:
                unique_sources.add(source_key)
                source_novelty_bonus = 0.04

        if content_key:
            if content_key in seen_content:
                duplicate_content_hits += 1
                content_redundancy_penalty = 0.05
            else:
                seen_content.add(content_key)

        authority_bonus = 0.08 if has_source else 0.0
        authority_bonus += 0.04 if _looks_authoritative_source(chunk) else 0.0
        utility_rank_score = round(
            relevance + authority_bonus + source_novelty_bonus - content_redundancy_penalty,
            6,
        )
        rescored_results.append(
            (
                utility_rank_score,
                relevance,
                -position,
                _annotated_chunk(
                    chunk,
                    utility_rank_score=utility_rank_score,
                    source_key=source_key,
                    has_source=has_source,
                ),
            )
        )

    rescored_results.sort(reverse=True)
    reranked_results = [chunk for _, _, _, chunk in rescored_results]
    candidate_count = len(ranked_results)
    source_diversity = len(unique_sources)
    average_relevance = total_relevance / candidate_count if candidate_count else 0.0
    average_rank_score = (
        sum(score for score, _, _, _ in rescored_results) / candidate_count
        if candidate_count
        else 0.0
    )

    retrieval_utility = {
        "candidate_count": candidate_count,
        "authority_hits": authority_hits,
        "source_diversity": source_diversity,
        "repeated_source_hits": repeated_source_hits,
        "duplicate_content_hits": duplicate_content_hits,
        "average_relevance": round(average_relevance, 4),
        "average_rank_score": round(average_rank_score, 4),
        "utility_score": _utility_score(
            candidate_count=candidate_count,
            authority_hits=authority_hits,
            source_diversity=source_diversity,
            average_relevance=average_relevance,
            repeated_source_hits=repeated_source_hits,
            duplicate_content_hits=duplicate_content_hits,
        ),
    }
    return {"ranked_results": reranked_results, "retrieval_utility": retrieval_utility}


def aggregate_ingestion_stats(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate statistics from parallel ingestion operations.

    Args:
        ctx: Workflow context with parallel ingestion results

    Returns:
        Aggregated statistics
    """
    pdf_stats = ctx.get("pdf_stats", {})
    docx_stats = ctx.get("docx_stats", {})
    markdown_stats = ctx.get("markdown_stats", {})
    code_stats = ctx.get("code_stats", {})

    total_docs = 0
    total_chunks = 0
    total_errors = 0

    for stats in [pdf_stats, docx_stats, markdown_stats, code_stats]:
        if isinstance(stats, dict):
            total_docs += stats.get("documents", 0)
            total_chunks += stats.get("chunks", 0)
            total_errors += stats.get("errors", 0)

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_errors": total_errors,
        "success_rate": ((total_docs - total_errors) / total_docs if total_docs > 0 else 0),
        "by_type": {
            "pdf": pdf_stats.get("documents", 0),
            "docx": docx_stats.get("documents", 0),
            "markdown": markdown_stats.get("documents", 0),
            "code": code_stats.get("documents", 0),
        },
    }


def _chunk_relevance(chunk: Any) -> float:
    """Extract a stable relevance score from a retrieval chunk."""
    value = _chunk_value(chunk, "relevance_score")
    if value in (None, ""):
        value = _chunk_value(chunk, "score")

    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _chunk_source_key(chunk: Any) -> str:
    """Normalize a chunk source to a site-level key."""
    source_url = str(_chunk_value(chunk, "source_url") or "").strip()
    if source_url:
        parsed = urlparse(source_url)
        if parsed.netloc:
            return parsed.netloc.lower()

    source_title = str(_chunk_value(chunk, "source_title") or "").strip().lower()
    return source_title


def _chunk_content_key(chunk: Any) -> str:
    """Normalize content for lightweight redundancy detection."""
    text = str(_chunk_value(chunk, "text") or _chunk_value(chunk, "content") or "").strip().lower()
    if not text:
        return ""
    return " ".join(text.split())[:160]


def _chunk_value(chunk: Any, key: str) -> Any:
    """Read a value from dict-like or object-like chunks."""
    if isinstance(chunk, dict):
        return chunk.get(key)
    return getattr(chunk, key, None)


def _annotated_chunk(chunk: Any, **annotations: Any) -> Any:
    """Attach utility metadata while preserving the original chunk shape."""
    if isinstance(chunk, dict):
        enriched_chunk = dict(chunk)
        enriched_chunk.update(annotations)
        return enriched_chunk
    return chunk


def _looks_authoritative_source(chunk: Any) -> bool:
    """Apply a small boost for sources that look official or canonical."""
    source_url = str(_chunk_value(chunk, "source_url") or "").strip().lower()
    source_title = str(_chunk_value(chunk, "source_title") or "").strip().lower()
    authority_markers = (
        ".gov",
        ".edu",
        "arxiv.org",
        "doi.org",
        "docs.",
        "developer.",
        "documentation",
        "official",
        "reference",
    )
    return any(marker in source_url or marker in source_title for marker in authority_markers)


def _utility_score(
    *,
    candidate_count: int,
    authority_hits: int,
    source_diversity: int,
    average_relevance: float,
    repeated_source_hits: int,
    duplicate_content_hits: int,
) -> float:
    """Aggregate retrieval utility with bounded bonuses and penalties."""
    score = 0.2
    score += min(candidate_count, 5) * 0.08
    score += min(authority_hits, 3) * 0.09
    score += min(source_diversity, 3) * 0.05
    score += max(0.0, average_relevance - 0.5) * 0.35
    score -= min(repeated_source_hits, 3) * 0.04
    score -= min(duplicate_content_hits, 3) * 0.04
    return round(max(0.0, min(1.0, score)), 4)


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    "retrieval_quality": retrieval_quality,
    "document_quality": document_quality,
    "chunking_strategy": chunking_strategy,
    "should_reindex": should_reindex,
    "query_complexity": query_complexity,
    "answer_confidence": answer_confidence,
    "classify_retrieval_gap": classify_retrieval_gap,
    "retrieval_repair_decision": retrieval_repair_decision,
    "embedding_batch_size": embedding_batch_size,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    "merge_retrieved_chunks": merge_retrieved_chunks,
    "format_context_window": format_context_window,
    "score_retrieval_utility": score_retrieval_utility,
    "aggregate_ingestion_stats": aggregate_ingestion_stats,
}

__all__ = [
    # Conditions
    "retrieval_quality",
    "document_quality",
    "chunking_strategy",
    "should_reindex",
    "query_complexity",
    "answer_confidence",
    "classify_retrieval_gap",
    "retrieval_repair_decision",
    "embedding_batch_size",
    # Transforms
    "merge_retrieved_chunks",
    "format_context_window",
    "score_retrieval_utility",
    "aggregate_ingestion_stats",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
