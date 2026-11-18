---
fep: "0049"
title: "Intent-Tier Code Embeddings with Merkle Incremental Sync"
type: Standards Track
status: Draft
created: 2026-11-18
modified: 2026-11-18
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0049
---

# FEP-0049: Intent-Tier Code Embeddings with Merkle Incremental Sync

## Summary

Victor's code graph answers **structure** questions precisely — call paths, override
chains, blast radius — through the typed CCG (18 relation types, SCIP-precise
references). It answers **intent** questions ("where do we rate-limit with a token
bucket?") poorly: the only embeddings today are symbol-summary documents
(name + signature + docstring) from `victor-codegraph.embeddings`, with no persisted
vector index. The 2026-11-18 competitive analysis against Cursor's codebase indexing
(the Merkle-tree-of-hashes + chunk-embedding + vector-search architecture) concluded
the two are complements: Cursor holds intent retrieval and effortless incremental sync;
Victor holds verifiable structure. This FEP closes the intent gap **without** giving up
the structural model: a chunk-source embedding tier (real code bodies, not summaries)
anchored to graph symbols by stable oid, kept fresh by a Merkle-projection-driven
incremental sync, and surfaced through the existing hybrid query paths.

## Two-Tier Embedding Model

| Tier | Text contract | Input | Consumer | Status |
|------|---------------|-------|----------|--------|
| Symbol-summary | `symbol-summary-v1` | name + signature + docstring | node-level semantic leg | shipped |
| Chunk-source (new) | `chunk-source-v1` | `CodeChunk` source text (bodies, module gaps) | intent tier | this FEP |

Both tiers use the same pinned model (`BAAI/bge-small-en-v1.5`, immutable revision,
384-dim). The `text_contract` field participates in the `EmbeddingProfile.identity`
hash, so the two tiers can never be mixed in one store — a stored-profile mismatch
forces a rebuild rather than silent cross-tier ranking.

## Merkle Discipline

The flat `path -> sha256(content)` manifest stays the single source of truth. The new
`victor_codegraph.merkle` module derives a per-directory rolling blake2b tree **from**
that manifest (deepest-first construction; deterministic; root-equality is the O(1)
no-op short-circuit). `diff_merkle` yields the file-level work set plus the minimal
changed-directory frontier (one deep edit reports only its own directory, not the
ancestor chain). The Merkle root is intentionally independent of
`repository_index.generation_id` — different construction, different purpose.

Sync state (`intent_file_state`: file, source_hash, profile_id) lives in the graph
SQLite DB under WAL, rebuilt on first run from `graph_file_mtime.content_hash`.
Crash safety is by **ordering, not locking**: LanceDB chunk upserts are idempotent by
deterministic `chunk_id` (`{symbol.id}#body#{idx}`), and the state row commits *after*
the vectors. A crash between the two re-embeds identical vectors on the next run.

## Storage Decision

Chunk vectors live in a **dedicated LanceDB table (`code_chunks`)** created through the
existing `EmbeddingRegistry`; NOT in `graph_node.embedding_ref` — that column is
one-value-per-node (chunks are many-per-symbol) and the Proxima backend retires it in
favor of oid correlation. Vector rows carry `symbol_oid` (ADR-044 stable oid) in
metadata, so query-side correlation to graph nodes works uniformly across sqlite,
duckdb, lancedb and proxima backends. Guard test: nothing in the intent path ever
writes `embedding_ref`. Below ~5K chunks a numpy brute-force
`StaticEmbeddingCollection` fallback serves the same contract.

## Query Surfaces

`graph_semantic_search` gains `mode="intent"`; `hybrid_seeds` gains a flag-gated
intent leg with **per-leg min-max normalization** (body-text cosine scores are not on
the same scale as symbol-summary scores); results resolve chunk → symbol oid → 1-hop
CCG expansion, deduplicated by oid, annotated with the winning leg. A degraded encoder
never serves the intent leg from the zero-vector fallback — the leg drops with a note,
structural+semantic legs unaffected.

## Rollout

`codebase_intent_indexing_enabled: bool = False` (off-by-default flag, mirroring
`codebase_structural_indexing_enabled`); graduation requires the ADR-036 evidence
battery: embedding wall-time delta, index wall-time delta on the root-shortcut path,
and retrieval quality vs the existing semantic mode. `victor-codegraph` 0.15.0 is
purely additive (merkle module, chunk embeddings); core feature-detects the new APIs
and degrades to flat-diff sync when an older package is installed.

## Alternatives Considered

- **Embed into `graph_node.embedding_ref`**: rejected — forks the backends and
  misfits the node/chunk cardinality.
- **Cloud embedding API (Cursor-style)**: rejected — air-gapped installs are a
  first-class Victor requirement; local BGE at 384-dim is already the unified model.
- **Fine-tuned code model (CodeBERT etc.)**: deferred — same-model two-tier keeps
  profile identity and deployment simple; a model swap is a profile change with
  rebuild semantics already specified.

## References

- `victor-codegraph/victor_codegraph/manifest.py` — flat manifest contract
- `victor-codegraph/victor_codegraph/embeddings.py` — profile identity + staleness
- `victor/storage/graph/sqlite_store.py` — `graph_file_mtime` content hashes, WAL
- `victor/core/graph_rag/seeding.py` — hybrid seeding precedent
- ADR-044 (stable oids), ADR-036 (flag graduation policy)
