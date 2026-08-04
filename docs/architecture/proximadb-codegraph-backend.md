# ProximaDB as the Code Context Graph (CCG) Backend

Status: Embedded backend implemented behind a per-repo flag; SQLite remains default
(tracked as TD-11, TD-12, TD-13 in `../tech-stack.md`). Service mode is WIP.
Date: 2026-06-22

## Implementation status (2026-06-22)

The embedded ProximaDB backend is implemented and parity-verified at the adapter
level; SQLite stays the default and nothing flips automatically.

- `victor-codegraph` exclusively owns stable symbol identity and emits the full
  correlated `graph/{repo}/node/{symbol_oid}` record `oid`; storage persists it
  unchanged. `victor/storage/proxima_runtime.py` provides optional-dependency
  detection, repository/graph naming, the `ProximaEmbeddingMode` (`memory`/`cold`)
  encoding of the Rust `EmbeddingMode`, and embedded bootstrap.
- `victor/storage/vector_stores/proximadb_provider.py` — now a real
  `EmbeddingProvider` over `proximadb_sdk`'s embedded API with an in-process
  embedding model (in-RAM fp32 = `EmbeddingMode::Memory`). Documents are keyed by
  their `oid`, so the always-empty `embedding_ref` bridge is unnecessary.
- `victor/storage/graph/proxima_store.py` — `ProximaGraphStore` implements
  `GraphStoreProtocol` over `proximadb_sdk.graph.ProximaDBGraph`
  (`upsert_nodes/edges`, `get_neighbors`, `search_symbols`, `find_nodes`,
  `multi_hop_traverse_parallel`, …). The graph node id **is** the vector id (one
  `oid`); `embedding_ref` is dropped.
- Selection: `create_graph_store("proxima", …)`, or per-repo via a
  `<project>/.victor/graph_backend` marker honored by `create_graph_store("auto", …)`
  (default `sqlite`). `impact_analysis` and the hybrid graph query tool resolve
  the backend through `"auto"`.
- Parity: `tests/unit/storage/graph/test_proxima_store_parity.py` drives the real
  `ProximaDBGraph` against an in-memory fake client and asserts impact_analysis
  (forward/backward) and hybrid seed→expand match SQLite — runs without the
  server binary. `tests/integration/storage/graph/test_proxima_embedded_parity.py`
  repeats it against a real embedded instance, skipping when the binary is absent.
- **WIP / gated:** the multi-tenant **service** path (`server_url=`,
  `EmbeddingMode::Cold`/SQ8) is marked WIP — gated on ProximaDB TD-127 (secondary
  indexes) + TD-130/131 (graph bulk-load + REST v2 hybrid). As of 2026-06-22 the
  **engine-side gate is now satisfied on ProximaDB `develop`**: TD-127/128 merged
  (PR #215, `40c08076`), TD-130 graph bulk-load merged (PR #220, `967f15db`), and
  REST v2 hybrid is live (`/api/v2/hybrid/search` + `/strategies` in the SDK). The
  remaining blocker is operational, not API: there is **no built `proximadb-server`
  binary** in this environment, so the embedded engine cannot start and live parity
  cannot be measured yet. The code keeps the service-mode WIP guard and SQLite
  default until a live parity bench passes (`cd proximaDB && cargo build --release`
  to unblock). Arrow Flight bulk-load and ORION native centrality (steps below) are
  also still pending on that live verification.

## Why

Victor's durable code memory — the thing that lets the agent answer "who calls X / blast radius /
what is semantically near this" without re-reading files — lives today in **two embedded stores**:

- **SQLite** (`.victor/project.db`, ~2.4 GB): `graph_node` / `graph_edge` / `graph_module_metric` /
  `graph_node_fts` — a statement-level **Code Property Graph** (CFG/CDG/DDG + CALLS/IMPORTS/INHERITS).
- **LanceDB** (`.victor/embeddings/embeddings.lance`): 384-d `BAAI/bge-small-en-v1.5` vectors at
  **symbol granularity** (measured 77,902 vectors; function 65,622 + class 12,280), with the symbol
  snippet co-stored.

These are hand-joined: `graph_node.embedding_ref` is meant to bridge them but is **unpopulated**, and a
watch daemon keeps both in sync on file change. The abstraction to swap them already exists — a
`GraphStoreProtocol` (sqlite/memory/duckdb-stub) and an `EmbeddingProvider` protocol with a
`proximadb_provider.py` referencing ProximaDB's SST (vector) + ORION (graph) engines.

The opportunity: collapse the two stores into **one correlated ProximaDB collection** where a code
**symbol is one entity** — a relational row, an ORION graph node, and an HNSW vector — addressed by a
single `oid`. This removes the dual-write skew, makes the embedding update atomic with the code change,
and gives the agent native graph algorithms (impact analysis, centrality, hybrid seed→expand) instead of
hand-rolled Python.

## Measured shape (one real repo, 3,659 files)

| | value |
|---|---|
| graph nodes | 1.26M (**94% `statement`**) |
| symbol nodes (module/class/function/method) | **79,744** |
| edges | 2.97M |
| Tier-A cross-fn edges (CALLS/IMPORTS/INHERITS/CONTAINS/…) | **96,538** (CALLS 61% cross-file) |
| Tier-B intra-fn edges (DDG/CFG/CDG) | **2,875,449** (DDG measured 100% intra-file) |
| embeddings (Lance) | 77,902 rows / **68,612 distinct** @ 384-d ≈ 100 MB f32 / 25 MB SQ8 + 5.5 MB snippet |

**Current correlation reality (measured):** the two stores are disjoint — SQLite `graph_node` has
`signature`/`docstring`/`embedding_ref` **0% populated** (pure topology + file/line); code snippet +
vector are **LanceDB-only**, keyed `symbol:{file}:{name}`; the graph hex `node_id` and the Lance id
namespace **do not intersect** (correlation is implicit by `(file, symbol_name)`, the `embedding_ref`
bridge is empty). Only **~5.4% of nodes** (≈86% of symbols) carry a vector. The ProximaDB one-`oid`
record makes embedding optional-per-node (NF² props), removing the always-empty bridge column.

## Design — three tiers, one `oid` per symbol

A code symbol becomes **one ProximaDB record** with the `victor-codegraph`-emitted
`oid = graph/{repo}/node/{symbol_oid}`, carrying its
relational columns (props), its embedding (`EmbeddingCell`), its branch (`branch_id`), and a ref to its
intra-procedural detail. The same `oid` is what the vector index (HNSW) and graph engine (ORION CSR) both
key on, so vector hit → graph node is identity, not a join.

- **Tier A — semantic graph (HOT, in-RAM):** ~80K symbol nodes + ~96K cross-fn edges + per-node 384-d
  vector → ORION graph + co-indexed vector. Drives `impact_analysis` (forward/backward k-hop), call paths,
  and hybrid semantic-seed → expand. ~120 MB f32 / ~35 MB SQ8 — fits memory.
- **Tier B — intra-procedural CPG (COLD, columnar):** statements + DDG/CFG/CDG → columnar fragments per
  function, fetched on dataflow drill-down. Never globally loaded; the real size driver stays off the hot
  graph.
- **Tier C — relational facts:** `code_file` / `code_import` / `code_module_metric` / `code_file_mtime`
  served from the same records; point-reads/upserts on the re-index hot path.

### What changes in Victor

- `victor/storage/vector_stores/proximadb_provider.py` → make real (currently emerging); use ProximaDB's
  `EmbeddingMode::Memory` for the embedded/local case so semantic BFS scores neighbors inline.
- `victor/storage/graph/` → add a `ProximaGraphStore` implementing `GraphStoreProtocol`
  (`upsert_nodes/edges`, `get_neighbors`, `search_symbols`, `multi_hop_traverse_parallel`) over the
  ProximaDB graph/hybrid API. The watch daemon's incremental path becomes idempotent `insert_proxima_records`
  upserts; initial load uses Arrow Flight bulk.
- `victor/core/graph_rag/retrieval.py` (`MultiHopRetriever`) and `victor/framework/search/hybrid.py`
  (RRF) → can delegate to ProximaDB's native `GraphHybridQuery` (VectorFirst fusion) instead of hand-rolled
  seed→expand + fuse.
- `graph_module_metric` (pagerank/betweenness/coupling/instability/hotspot) → can be computed by ORION's
  native centrality/community algorithms rather than in Python.

### Embedded vs service

- **Embedded (local single-repo):** one `EmbeddedProximaDB` (PyO3) per repo, `EmbeddingMode::Memory`.
  Drop-in for the current SQLite + LanceDB pair.
- **Service (multi-tenant, via anvaiops):** collection `{tenant}_{repo}_codegraph`, `graph_id`=repo,
  `branch_id`=git branch, `EmbeddingMode::Cold` (SQ8) to bound RAM. See the ProximaDB design spec
  `proximaDB/docs/12-design/CODE_GRAPH_CORRELATED_SUBSTRATE_2026_06_22.adoc` and the anvaiops ADR.

## ProximaDB-side enabling work (not Victor's)

The ProximaDB engine asks are filed there as **TD-127..TD-134** (OLTP secondary indexes by name/file,
IN-list pushdown, `ON CONFLICT DO UPDATE`, graph edge bulk-load, graph REST v2 + co-planned hybrid,
Tier-B PAX fragment contract, optional transactional multi-modal write, code-embedding KEU meter).

## Migration / verification (when picked up)

1. ✅ Stand up `ProximaGraphStore` behind the existing `GraphStoreProtocol`; keep SQLite as the default.
2. ✅ Parity test on a fixture repo: `impact_analysis(forward/backward)` and hybrid seed→expand match
   the SQLite store on known symbols (adapter-level always-on + embedded gated).
3. ⏳ Bench Arrow Flight bulk-load + k-hop + hybrid latency; compare footprint vs the 2.4 GB SQLite + Lance
   pair (projected ~120 MB f32 / ~35 MB SQ8 for Tier-A). Blocked on a built `proximadb-server` binary.
4. ⏳ Flip the default provider per-repo once parity holds (per-repo `.victor/graph_backend` flag exists; SQLite stays default).
