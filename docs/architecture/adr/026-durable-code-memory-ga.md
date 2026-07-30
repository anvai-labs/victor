# ADR-026: Durable Code Memory — Correlated Graph+Vector on One `oid`

## Metadata

- **Status**: Proposed
- **Date**: 2026-07-29
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 014 (shared `victor-codegraph` chunker), 015 (Victor core adopts codegraph,
  phased), 002 (state management)
- **Work tracked by**: [TD-11, TD-12, TD-13](../../tech-stack.md#technical-debt-register) (existing —
  no new work item)
- **External**: ProximaDB ADR-044 (stable line-independent symbol `oid`), ProximaDB ADR-029
- **Benchmark**: [competitive-benchmark-2026-07.md](../competitive-benchmark-2026-07.md) §2

## Context

A terminal coding agent's usefulness on a real repo is bounded by the quality of its **code memory**.
The incumbents make this their edge: Aider's repo-map and Cursor's index give the model a durable,
navigable view of the codebase. Victor has the pieces — the shared `victor-codegraph` CPG chunker
(ADR-014, shipped), Victor core's phased adoption of it (ADR-015, Phase 1 live in
`graph_rag/indexing.py`), and a detailed backend plan (`proximadb-codegraph-backend.md`, TD-11/12/13)
— but the memory substrate is **not yet GA**:

- **Dual-store skew.** Graph lives in SQLite; embeddings in LanceDB; `graph_node.embedding_ref` is
  unpopulated, so a code change can rewrite text and re-embedding in two non-atomic writes (TD-12).
- **No correlated backend.** `proximadb_provider.py` is not yet a real `GraphStoreProtocol`
  implementation, so the SQLite+Lance pair can't be replaced by one correlated collection (TD-11).
- **Whole-CPG in RAM doesn't scale.** Indexing symbols + cross-fn edges *and* intra-procedural CPG
  into the live graph is memory-heavy; the Tier-A/Tier-B split (TD-13) is designed but not landed.

The decisions (ADR-014/015) and the work (TD-11/12/13) exist; what's missing is the ADR that names
the **GA target** those converge toward.

## Decision

Adopt **"one `oid`, one atomic upsert"** as the durable code-memory target, and record it as the
destination for ADR-015's later phases and TD-11/12/13:

1. **Correlation key = symbol `oid`.** Graph node and its embedding share
   `graph/{repo}/node/{symbol_oid}` (ProximaDB ADR-044's stable, line-independent identity). A code
   change rewrites text + re-embedding in **one atomic upsert**; `embedding_ref` is retired (TD-12).
2. **Correlated backend.** A real `ProximaGraphStore` implementing `GraphStoreProtocol` replaces the
   SQLite+LanceDB pair with one collection holding graph + vector + relational on that `oid` (TD-11),
   behind the existing `GraphStoreProtocol` seam so the SQLite path remains the fallback.
3. **Tier-A / Tier-B split for scale.** Symbols (~80K) + cross-fn edges (~96K) stay in the
   traversable in-RAM graph (~120 MB f32 / ~35 MB SQ8); intra-procedural CPG (~96% of edges, 100%
   intra-file) offloads to columnar fragments fetched on dataflow drill-down (TD-13).

GA criterion: the correlated backend match-or-beats the SQLite+Lance pair on retrieval quality and
stays within the RAM envelope, with the SQLite path retained as fallback (Rust-fallback discipline
applied to storage).

## Rationale

- **First principles.** Code memory must be *atomic* (no dual-write skew), *stable across edits* (an
  identity that survives line moves — ADR-044's `oid`), and *bounded in RAM* (tiering). The current
  SQLite+Lance pair violates all three; the target fixes each with an existing design.
- **Reuse + co-design across repos.** This is co-designed with ProximaDB (ADR-044/029) and consumes
  `victor-codegraph` — it does not reinvent chunking or identity; it lands the backend those decided.
- **Competitive.** A correlated, edit-stable, tiered memory is a stronger substrate than a flat
  repo-map; it is the foundation the terminal agent (ADR-020) and RAG vertical build on.

## Consequences

- **Positive**: atomic code updates; edit-stable retrieval; RAM-bounded live graph; a single backend
  replaces a skew-prone pair; the durable-memory competitive gap (benchmark §2) closes.
- **Negative**: a real ProximaDB dependency for the GA path (SQLite fallback retained); migration of
  existing project graphs to the `oid` scheme; tiering adds a drill-down fetch path.
- **Neutral**: `victor-codegraph` and the chunker are unchanged (ADR-014); the `GraphStoreProtocol`
  seam is preserved.

## Implementation

Internal storage — **no companion FEP** (no public framework-API change; `GraphStoreProtocol` is the
seam). Sequenced by the existing register items:

1. **TD-12** — correlate embedding↔node on `oid`; retire `embedding_ref`; make the upsert atomic.
2. **TD-11** — implement `ProximaGraphStore` behind `GraphStoreProtocol`; keep SQLite as fallback.
3. **TD-13** — land the Tier-A/Tier-B split; verify the RAM envelope.
4. On GA-criterion pass, complete ADR-015's later phases and flip this ADR to Accepted.

## Alternatives Considered

- **Keep SQLite + LanceDB dual store.** Rejected: the unatomic dual-write skew (TD-12) is a
  correctness bug, not just an efficiency one.
- **Whole-CPG in RAM (no tiering).** Rejected: the memory envelope on real repos is prohibitive
  (TD-13's measurements).
- **Flat repo-map (Aider-style) instead of a CPG.** Rejected: loses the dataflow/graph traversal
  Victor's tools already exploit; the CPG is the differentiator, tiering makes it affordable.

## References

- [ADR-014](014-shared-codegraph-chunker-package.md), [ADR-015](015-victor-core-adopts-codegraph.md)
- [proximadb-codegraph-backend.md](../proximadb-codegraph-backend.md),
  [TD-11/12/13](../../tech-stack.md#technical-debt-register)
- `victor/core/graph_rag/indexing.py`, `victor/storage/` (`proximadb_provider.py`), ProximaDB
  ADR-044 (external — symbol `oid`)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-29 | 1.0 | Initial ADR — durable code-memory GA target (one oid, correlated, tiered) | Vijaykumar Singh |
