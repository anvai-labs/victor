# Graph Search Reliability Spec

**Status:** Draft  
**Created:** 2026-11-15  
**Author:** Victor RL Team  
**Priority:** P1 (High)

---

## Problem Statement

The `graph_search` tool has a **36.4% success rate** (66 total calls: 24 successes, 42 failures) based on RL outcome data from `~/.victor/victor.db`. This is critically low for a core search tool:

| Tool | Success Rate | Total Calls |
|------|-------------|-------------|
| `code_search` | 97.1% | 1,098 |
| `read` | 94.4% | 6,251 |
| `graph` | 87.6% | 793 |
| `graph_search` | **36.4%** | 66 |

### Evidence

```sql
-- Graph search outcomes by success
SELECT task_type, success, COUNT(*) as count,
       ROUND(AVG(quality_score), 3) as avg_quality,
       ROUND(AVG(reward), 3) as avg_reward
FROM rl_tool_outcome
WHERE tool_name = 'graph_search'
GROUP BY task_type, success;

-- Result:
-- general | 0 | 42 | 0.300 | 0.110  (failures)
-- general | 1 | 24 | 0.911 | 0.932  (successes)
```

### Impact

1. **Code navigation broken:** Agents can't reliably search the code graph for symbols, files, modules
2. **Workflow disruption:** Graph-based exploration fails, forcing fallback to slower text search
3. **RL learning:** Low success rate poisons tool_selector Q-values, reducing graph_search selection
4. **User frustration:** Graph search is a key differentiator; failures undermine confidence

---

## Root Cause Analysis

### Potential Root Causes

Based on the graph_search implementation (`victor/tools/graph_tool.py:3739`), potential failure modes:

1. **Graph not indexed:** Search fails if code graph hasn't been indexed yet
2. **Embedding model mismatch:** Query embeddings don't match index embeddings
3. **Query too vague:** Natural language queries may not match symbol names
4. **Threshold too high:** Similarity threshold filters out valid results
5. **Index corruption:** Graph index may be corrupted or incomplete
6. **Timeout:** Large codebases may cause search timeouts
7. **Path resolution:** Working directory path may not match indexed paths

### What We Don't Know

The `rl_tool_outcome.metadata` for graph_search failures is not available in the current data. We need to:
1. Add structured metadata to graph_search outcomes
2. Instrument failure points in the search pipeline
3. Collect data on which failure mode dominates

### Hypotheses (Ranked by Likelihood)

1. **Graph not indexed (40% likely):** Most common cause — user hasn't run `victor index`
2. **Embedding mismatch (25% likely):** Different embedding model used for query vs index
3. **Threshold too high (20% likely):** Default threshold filters out valid results
4. **Query format (10% likely):** Query doesn't match expected format (symbol name vs description)
5. **Other (5% likely):** Timeout, corruption, path issues

---

## Proposed Solution

### Phase 1: Instrument Failure Points (P0 — 2 days)

**Goal:** Understand why graph_search fails by adding structured error metadata.

#### 1.1 Define Graph Search Outcome Metadata

```python
@dataclass
class GraphSearchOutcomeMetadata:
    """Structured metadata for graph_search outcomes."""
    # Query info
    query: str                          # The search query
    path: str                           # Codebase root path
    top_k: int                          # Requested result count
    
    # Index state
    index_exists: bool                  # Whether graph index exists
    index_size: int                     # Number of nodes in index
    embedding_model: str                # Embedding model used for index
    query_embedding_model: str          # Embedding model used for query
    models_match: bool                  # Whether index and query models match
    
    # Search execution
    threshold_used: float               # Similarity threshold applied
    results_found: int                  # Number of results before threshold
    results_returned: int               # Number of results after threshold
    search_duration_ms: float           # Search execution time
    
    # Error classification
    error_category: str                 # "none", "no_index", "model_mismatch", 
                                        # "no_results", "timeout", "corruption"
    error_message: str                  # Human-readable error
    
    # Result quality
    avg_similarity: float               # Average similarity score of results
    min_similarity: float               # Minimum similarity score
    max_similarity: float               # Maximum similarity score
```

#### 1.2 Instrument graph_search Function

**File:** `victor/verticals/victor-coding/victor_coding/tools/graph_tool.py`

```python
async def graph_search(query, path, top_k, _exec_ctx):
    start_time = time.monotonic()
    
    try:
        # Check if index exists
        index = await _get_graph_index(path)
        if index is None:
            metadata = GraphSearchOutcomeMetadata(
                query=query, path=path, top_k=top_k,
                index_exists=False, index_size=0,
                embedding_model="", query_embedding_model="",
                models_match=False, threshold_used=0.0,
                results_found=0, results_returned=0,
                search_duration_ms=(time.monotonic() - start_time) * 1000,
                error_category="no_index",
                error_message=f"Graph index not found at {path}. Run 'victor index' first.",
                avg_similarity=0.0, min_similarity=0.0, max_similarity=0.0,
            )
            _record_tool_outcome(
                tool_name="graph_search",
                success=False,
                metadata=asdict(metadata),
            )
            return {"error": "Graph not indexed", "results": []}
        
        # Check embedding model match
        query_model = await _get_query_embedding_model()
        index_model = index.embedding_model
        models_match = query_model == index_model
        
        if not models_match:
            metadata = GraphSearchOutcomeMetadata(
                query=query, path=path, top_k=top_k,
                index_exists=True, index_size=index.size,
                embedding_model=index_model,
                query_embedding_model=query_model,
                models_match=False, threshold_used=0.0,
                results_found=0, results_returned=0,
                search_duration_ms=(time.monotonic() - start_time) * 1000,
                error_category="model_mismatch",
                error_message=f"Embedding model mismatch: index={index_model}, query={query_model}",
                avg_similarity=0.0, min_similarity=0.0, max_similarity=0.0,
            )
            _record_tool_outcome(
                tool_name="graph_search",
                success=False,
                metadata=asdict(metadata),
            )
            return {"error": "Embedding model mismatch", "results": []}
        
        # Perform search
        results = await index.search(query, top_k=top_k)
        
        # Apply threshold
        threshold = _get_search_threshold()
        filtered = [r for r in results if r.similarity >= threshold]
        
        if not filtered:
            metadata = GraphSearchOutcomeMetadata(
                query=query, path=path, top_k=top_k,
                index_exists=True, index_size=index.size,
                embedding_model=index_model,
                query_embedding_model=query_model,
                models_match=True, threshold_used=threshold,
                results_found=len(results), results_returned=0,
                search_duration_ms=(time.monotonic() - start_time) * 1000,
                error_category="no_results",
                error_message=f"No results above threshold {threshold}",
                avg_similarity=0.0, min_similarity=0.0, max_similarity=0.0,
            )
            _record_tool_outcome(
                tool_name="graph_search",
                success=False,
                metadata=asdict(metadata),
            )
            return {"results": [], "message": "No matches found"}
        
        # Success
        similarities = [r.similarity for r in filtered]
        metadata = GraphSearchOutcomeMetadata(
            query=query, path=path, top_k=top_k,
            index_exists=True, index_size=index.size,
            embedding_model=index_model,
            query_embedding_model=query_model,
            models_match=True, threshold_used=threshold,
            results_found=len(results), results_returned=len(filtered),
            search_duration_ms=(time.monotonic() - start_time) * 1000,
            error_category="none",
            error_message="",
            avg_similarity=sum(similarities) / len(similarities),
            min_similarity=min(similarities),
            max_similarity=max(similarities),
        )
        _record_tool_outcome(
            tool_name="graph_search",
            success=True,
            metadata=asdict(metadata),
        )
        
        return {"results": [r.to_dict() for r in filtered]}
        
    except Exception as e:
        metadata = GraphSearchOutcomeMetadata(
            query=query, path=path, top_k=top_k,
            index_exists=False, index_size=0,
            embedding_model="", query_embedding_model="",
            models_match=False, threshold_used=0.0,
            results_found=0, results_returned=0,
            search_duration_ms=(time.monotonic() - start_time) * 1000,
            error_category="unknown",
            error_message=str(e),
            avg_similarity=0.0, min_similarity=0.0, max_similarity=0.0,
        )
        _record_tool_outcome(
            tool_name="graph_search",
            success=False,
            metadata=asdict(metadata),
        )
        raise
```

### Phase 2: Improve Error Messages (P1 — 2 days)

**Goal:** Provide actionable error messages that guide users to fix the issue.

#### 2.1 No Index Error

```python
if not index_exists:
    return {
        "error": "Graph not indexed",
        "message": (
            f"The code graph at '{path}' has not been indexed yet.\n"
            f"Run 'victor index graph index' to create the index.\n"
            f"For large codebases, this may take several minutes."
        ),
        "results": [],
        "suggestion": "victor index graph index",
    }
```

#### 2.2 Model Mismatch Error

```python
if not models_match:
    return {
        "error": "Embedding model mismatch",
        "message": (
            f"The graph index was created with embedding model '{index_model}',\n"
            f"but the current query uses model '{query_model}'.\n"
            f"Either re-index with the current model or switch to the indexed model."
        ),
        "results": [],
        "suggestion": f"victor index graph index --force --embedding-model {index_model}",
    }
```

#### 2.3 No Results Error

```python
if not filtered:
    return {
        "error": "No matches found",
        "message": (
            f"No results found for query '{query}' above similarity threshold {threshold:.2f}.\n"
            f"Try:\n"
            f"  - Using a more specific query (e.g., symbol name instead of description)\n"
            f"  - Lowering the threshold: victor config set graph.search_threshold 0.5\n"
            f"  - Using 'graph' tool with mode='search' for broader search"
        ),
        "results": [],
        "suggestion": "Lower threshold or use more specific query",
    }
```

### Phase 3: Adaptive Threshold (P2 — 3 days)

**Goal:** Dynamically adjust search threshold based on results.

#### 3.1 Threshold Auto-Tuning

```python
def _get_adaptive_threshold(query: str, index: GraphIndex) -> float:
    """Dynamically adjust threshold based on query characteristics."""
    from victor.framework.rl.learners.semantic_threshold import SemanticThresholdLearner
    
    learner = get_semantic_threshold_learner()
    
    # Get recommended threshold from RL learner
    recommendation = learner.get_recommendation(
        context_key=f"graph_search:{query[:50]}",
    )
    
    if recommendation and recommendation.confidence > 0.6:
        return recommendation.value
    
    # Fall back to default
    return 0.7

def _fallback_if_no_results(query: str, results: list, threshold: float) -> list:
    """If no results above threshold, return top-K with lower threshold."""
    if results:
        return results
    
    # Retry with lower threshold
    lower_threshold = threshold * 0.7
    filtered = [r for r in results if r.similarity >= lower_threshold]
    
    if filtered:
        logger.info(
            f"graph_search: No results at threshold {threshold:.2f}, "
            f"found {len(filtered)} at {lower_threshold:.2f}"
        )
        return filtered
    
    return []
```

#### 3.2 Learn from Search Outcomes

Update `semantic_threshold` learner to track graph_search outcomes:

```python
# In graph_search, after search completes:
learner = get_semantic_threshold_learner()
learner.record_outcome(
    context_key=f"graph_search:{query[:50]}",
    threshold_used=threshold,
    results_found=len(results),
    results_returned=len(filtered),
    success=len(filtered) > 0,
)
```

### Phase 4: Query Normalization (P2 — 2 days)

**Goal:** Improve query matching by normalizing common query patterns.

#### 4.1 Symbol Name Extraction

```python
def _normalize_query(query: str) -> str:
    """Extract symbol name from natural language query."""
    import re
    
    # Common patterns: "find the foo function", "search for Bar class"
    patterns = [
        r"(?:find|search|look for|show me)\s+(?:the\s+)?(\w+)\s+(?:function|class|method|variable)",
        r"(?:where is|find)\s+(\w+)",
        r"(\w+)\s+(?:definition|implementation|usage)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # If no pattern matches, return original query
    return query
```

#### 4.2 Multi-Query Search

```python
async def _multi_query_search(query: str, index: GraphIndex, top_k: int) -> list:
    """Try multiple query variations to improve recall."""
    queries = [query]
    
    # Add normalized version
    normalized = _normalize_query(query)
    if normalized != query:
        queries.append(normalized)
    
    # Add lowercase version
    queries.append(query.lower())
    
    # Search with all variations
    all_results = []
    for q in queries:
        results = await index.search(q, top_k=top_k)
        all_results.extend(results)
    
    # Deduplicate by node ID
    seen = set()
    unique = []
    for r in all_results:
        if r.node_id not in seen:
            seen.add(r.node_id)
            unique.append(r)
    
    # Sort by similarity
    unique.sort(key=lambda r: r.similarity, reverse=True)
    
    return unique[:top_k]
```

### Phase 5: Index Health Checks (P3 — 1 day)

**Goal:** Detect and report index health issues.

#### 5.1 Index Validation

```python
def _validate_index_health(index: GraphIndex) -> dict:
    """Validate graph index health."""
    issues = []
    
    # Check index size
    if index.size == 0:
        issues.append("Index is empty — no nodes indexed")
    
    # Check embedding model
    if not index.embedding_model:
        issues.append("Index missing embedding model metadata")
    
    # Check for corruption (sample random nodes)
    try:
        sample = index.get_random_nodes(10)
        if len(sample) < 10 and index.size >= 10:
            issues.append("Index may be corrupted — random sampling failed")
    except Exception as e:
        issues.append(f"Index corruption detected: {e}")
    
    return {
        "healthy": len(issues) == 0,
        "issues": issues,
        "size": index.size,
        "embedding_model": index.embedding_model,
    }
```

#### 5.2 Add to `victor doctor`

```python
def check_graph_index_health():
    """Check graph index health."""
    from victor.storage.code_graph import get_graph_index
    
    try:
        index = get_graph_index(".")
        if index is None:
            return {
                "status": "missing",
                "message": "Graph index not found. Run 'victor index graph index'.",
            }
        
        health = _validate_index_health(index)
        
        if not health["healthy"]:
            return {
                "status": "unhealthy",
                "message": f"Graph index has issues: {'; '.join(health['issues'])}",
                "issues": health["issues"],
            }
        
        return {
            "status": "healthy",
            "message": f"Graph index healthy ({health['size']} nodes, model={health['embedding_model']})",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to check graph index: {e}",
        }
```

---

## Implementation Plan

### Sprint 1: Instrumentation (Week 1)

- [ ] Define `GraphSearchOutcomeMetadata` dataclass
- [ ] Instrument `graph_search` function with structured metadata
- [ ] Add error classification logic
- [ ] Add unit tests for metadata recording
- [ ] Verify metadata round-trips correctly

### Sprint 2: Error Messages (Week 2)

- [ ] Improve "no index" error message with actionable guidance
- [ ] Improve "model mismatch" error message with fix instructions
- [ ] Improve "no results" error message with suggestions
- [ ] Add unit tests for error messages
- [ ] Test with real queries

### Sprint 3: Adaptive Threshold (Week 3)

- [ ] Implement adaptive threshold logic
- [ ] Integrate with `semantic_threshold` learner
- [ ] Add fallback logic for no-results cases
- [ ] Add unit tests for threshold tuning
- [ ] A/B test with real usage

### Sprint 4: Query Normalization (Week 4)

- [ ] Implement query normalization patterns
- [ ] Implement multi-query search
- [ ] Add unit tests for normalization
- [ ] Test with common query patterns

### Sprint 5: Health Checks (Week 5)

- [ ] Implement index validation logic
- [ ] Add graph index health check to `victor doctor`
- [ ] Add unit tests for health checks
- [ ] Document monitoring approach

---

## Verification

### Metadata Verification

```bash
# Query recent graph_search outcomes
sqlite3 ~/.victor/victor.db "
SELECT 
    json_extract(metadata, '$.error_category') as category,
    json_extract(metadata, '$.index_exists') as index_exists,
    json_extract(metadata, '$.models_match') as models_match,
    COUNT(*) as count
FROM rl_tool_outcome
WHERE tool_name = 'graph_search'
  AND timestamp > datetime('now', '-1 day')
GROUP BY category, index_exists, models_match;
"
```

### Success Rate Verification

```bash
# Check graph_search success rate
sqlite3 ~/.victor/victor.db "
SELECT 
    ROUND(100.0 * SUM(success) / COUNT(*), 1) as success_pct,
    COUNT(*) as total
FROM rl_tool_outcome
WHERE tool_name = 'graph_search'
  AND timestamp > datetime('now', '-7 days');
"
```

### Error Category Breakdown

```bash
# Check error breakdown
sqlite3 ~/.victor/victor.db "
SELECT 
    json_extract(metadata, '$.error_category') as category,
    COUNT(*) as count
FROM rl_tool_outcome
WHERE tool_name = 'graph_search'
  AND success = 0
  AND timestamp > datetime('now', '-7 days')
GROUP BY category
ORDER BY count DESC;
"
```

---

## Acceptance Criteria

- [ ] Graph search outcomes include complete, structured metadata
- [ ] Error categories are recorded for all failures
- [ ] Error messages provide actionable guidance
- [ ] Adaptive threshold improves recall without sacrificing precision
- [ ] Query normalization improves match rate
- [ ] `victor doctor` reports graph index health
- [ ] Graph search success rate improves from 36% to >80%
- [ ] Unit tests cover all new functionality
- [ ] No regression in existing graph_search functionality

---

## Open Questions

1. **Default threshold:** What should the default similarity threshold be?
   - Current: 0.7 (from RL data)
   - Option: Lower to 0.5 for better recall
   - Recommendation: Use adaptive threshold, default to 0.6

2. **Re-indexing strategy:** How often should we re-index?
   - Option: On every file change (expensive)
   - Option: Daily (may miss changes)
   - Recommendation: Incremental indexing on file save

3. **Query embedding caching:** Should we cache query embeddings?
   - Option: Cache for 1 hour (good for repeated queries)
   - Risk: Stale cache if model changes
   - Recommendation: Cache with model version key

4. **Fallback strategy:** What if graph_search still fails?
   - Option: Automatically fall back to `code_search` (text-based)
   - Risk: Slower, less precise
   - Recommendation: Suggest fallback in error message, don't auto-fallback

---

## Follow-Up Work

1. **Query pattern analysis:** Analyze which query patterns succeed/fail
2. **Embedding model comparison:** A/B test different embedding models
3. **Index size optimization:** Reduce index size for large codebases
4. **Incremental indexing:** Implement incremental index updates

---

## References

- RL Data Analysis: `docs/analysis/rl-data-summary-2026-11-15.md`
- Graph Tool: `victor/verticals/victor-coding/victor_coding/tools/graph_tool.py`
- Semantic Threshold Learner: `victor/framework/rl/learners/semantic_threshold.py`
- Graph Architecture: `docs/architecture/graph-architecture-alignment.md`
