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

Victor's code graph holds verifiable **structure** (typed CCG, SCIP-precise
references) but no **intent** retrieval over source bodies: the only embeddings are
symbol-summary documents with no persisted vector index. This FEP adds a
chunk-source embedding tier (real code bodies) anchored to graph symbols by stable
oid, kept fresh by Merkle-tree incremental sync (Cursor-class cost discipline,
fully local), surfaced through the existing hybrid query paths — so both structure
and intent are answerable algorithmically. Flag-gated off by default; graduation per
ADR-036.

## Motivation

Victor's code graph answers **structure** questions precisely — call paths, override
chains, blast radius — through the typed CCG (18 relation types, SCIP-precise
references). It answers **intent** questions ("where do we rate-limit with a token
bucket?") poorly: the only embeddings today are symbol-summary documents
(name + signature + docstring) from `victor-codegraph.embeddings`, with no persisted
vector index. The 2026-11-18 competitive analysis against Cursor's codebase indexing
(the Merkle-tree-of-hashes + chunk-embedding + vector-search architecture) concluded
the two are complements: Cursor holds intent retrieval and effortless incremental
sync; Victor holds verifiable structure. This FEP closes the intent gap without
giving up the structural model: a chunk-source embedding tier (real code bodies,
not summaries) anchored to graph symbols by stable oid, kept fresh by a
Merkle-projection-driven incremental sync, and surfaced through the existing hybrid
query paths.

## Proposed Change

1. **Two-tier embedding model**: the shipped symbol-summary tier
   (`text_contract="symbol-summary-v1"`, name + signature + docstring) is joined by
   a chunk-source tier (`chunk-source-v1`, `CodeChunk` source text). Both use the
   same pinned model (`BAAI/bge-small-en-v1.5`, immutable revision, 384-dim);
   `text_contract` participates in `EmbeddingProfile.identity`, so the two tiers can
   never be mixed in one store — a stored-profile mismatch forces a rebuild rather
   than silent cross-tier ranking.

2. **Merkle projection** (`victor_codegraph.merkle`, pure stdlib): per-directory
   rolling blake2b tree derived FROM the flat `path -> sha256` manifest (the flat
   manifest stays the single source of truth). Leaves extend to the file's
   CONSTITUENT CHUNKS (deterministic `{symbol.id}#body#{idx}` ids) when chunk
   digests are supplied, so chunk-relevant graph identity is represented at leaf
   granularity. `diff_merkle` yields the file-level work set, the chunk-precise
   work set, and the minimal changed-directory frontier; root equality is the O(1)
   no-op short-circuit. The Merkle root is intentionally independent of
   `repository_index.generation_id`.

3. **Storage split**: chunk vectors live in a dedicated LanceDB table
   (`code_chunks`) created through the existing `EmbeddingRegistry`; the per-file
   sync journal (`intent_file_state`) lives in the graph SQLite DB under WAL. The
   correlation key is `symbol_oid` (ADR-044 stable oid) carried in vector metadata —
   uniform across sqlite/duckdb/lancedb and proxima-by-oid backends. The legacy
   per-node embedding column on the node table is never written (guard-tested).

4. **Crash safety by ordering, not locking**: chunk upserts are idempotent by
   deterministic `chunk_id`; the sync marker commits AFTER the vectors. A crash
   between the two re-embeds identical vectors on the next run.

5. **Incremental sync** (`victor/core/graph_rag/intent_sync.py`): hooked into
   `GraphIndexingPipeline` AFTER publication and edge resolution, never inside the
   sqlite replacement. Root-equality shortcut; flat diff is the authoritative work
   set; per-file chunk -> embed -> delete-then-upsert -> marker commit; profile
   change forces a full rebuild; the encoder constructs lazily so a no-op run never
   loads the model.

6. **Query surfaces**: `graph_semantic_search` gains `mode="intent"`;
   `hybrid_seeds` gains a flag-gated intent leg with per-leg min-max normalization
   (body-text cosine scores are not on the same scale as symbol-summary scores);
   results resolve chunk -> symbol -> 1-hop CCG expansion, deduplicated by oid.

## Benefits

- Intent questions become answerable algorithmically over the indexed corpus, with
  results anchored to graph nodes (not floating chunks).
- Re-embedding cost after a save-index cycle is proportional to the changed chunks,
  and zero when the tree is unchanged (root shortcut).
- No new runtime dependencies; no cloud embedding calls — air-gapped installs are
  first-class.
- The existing structural graph is untouched: the intent tier is additive, flagged
  off by default, and degradable per-leg.

## Drawbacks and Alternatives

- Body-text embedding is more expensive than symbol summaries (more tokens per
  input) — mitigated by the root shortcut, the 512-input batch cap, and the
  resumable state table.
- Chunk ids shift when a symbol body grows (`#body#idx`); handled by per-file
  delete-then-upsert, but the vector table churns on refactor-heavy commits.
- Alternative considered: embedding into the per-node embedding column — rejected
  (one-value-per-node misfits many-chunks-per-symbol; forks the Proxima-by-oid
  backend).
- Alternative considered: a cloud embedding API (Cursor-style) — rejected;
  air-gapped installs are a first-class requirement.
- Alternative considered: a fine-tuned code model — deferred; same-model two-tier
  keeps profile identity and deployment simple, and a model swap is a profile
  change with rebuild semantics already specified.

## Compatibility

- `victor-codegraph` 0.15.0 is purely additive (new `merkle` module, new profile and
  `embed_chunks`; existing contracts untouched); core feature-detects the new APIs
  and degrades to flat-diff sync when an older package is installed.
- `graph_node.embedding_ref` and the Proxima-by-oid correlation model are unchanged.
- Hybrid query behavior is byte-identical when the intent flag is off.
- Old persisted snapshots and policy metadata are unaffected (no schema change to
  existing tables; one new table).

## Migration Path

1. Land FEP + package additions (0.15.0) — no behavior change.
2. Land the storage layer (`IntentIndexStore`, `intent_file_state`, config flag,
   default off) — no behavior change.
3. Land the sync hook + CLI (`victor graph index --intent`) — opt-in only.
4. Land the query surfaces behind the same flag — opt-in only.
5. ADR-036 graduation battery: embedding wall-time delta, index wall-time delta on
   the root-shortcut path, retrieval quality vs the semantic mode; then flip the
   default if evidence supports it.

## Unresolved Questions

- Hybrid merge weights between intent, semantic and structural legs (initial:
  equal-after-normalization; revisited with field data).
- Whether chunk embeddings should also cover module-gap chunks or only symbol
  bodies (current: whatever `chunk_path` emits, including gap and window chunks).
- Index-sharing across machines/CI (the store is project-local; multi-machine sync
  is out of scope for this FEP).

## Implementation Plan

| Phase | PR | Content |
|-------|-----|---------|
| 0 | (this FEP) | Spec |
| 1 | #1915 | `victor-codegraph` 0.15.0: `merkle.py` (chunk-granularity leaves), `CHUNK_SOURCE_PROFILE`, `embed_chunks` |
| 2 | #1916 | `IntentIndexStore` (LanceDB `code_chunks`), `intent_file_state` journal, config flag |
| 3 | #1917 | `sync_intent_tier` + pipeline hook + `victor graph index --intent` |
| 4 | queued | Query surfaces: intent seed leg, `graph similar --mode intent`, tool mode |

Each phase lands behind the default-off flag through step 3; step 4 stays flagged
until the ADR-036 evidence battery (embed wall-time delta, root-shortcut index
delta, retrieval quality vs semantic mode) supports graduation.

## References

- `victor-codegraph/victor_codegraph/manifest.py` — flat manifest contract
- `victor-codegraph/victor_codegraph/merkle.py` — the projection (this FEP)
- `victor-codegraph/victor_codegraph/embeddings.py` — profile identity + staleness
- `victor/storage/graph/sqlite_store.py` — `graph_file_mtime` content hashes, WAL
- `victor/storage/graph/intent_index.py` — intent-tier store (phase 2)
- `victor/core/graph_rag/seeding.py` — hybrid seeding precedent
- ADR-044 (stable oids), ADR-036 (flag graduation policy)
- 2026-11-18 Cursor indexing comparison (Engineer's Codex; cursor.com security blog)
