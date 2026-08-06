# ProximaDB as the Code Context Graph (CCG) Backend

Status: Embedded backend and local Tier-A/Tier-B boundary implemented behind a
per-repo flag; SQLite remains default (tracked as TD-11, TD-12, TD-13 in
`../tech-stack.md`). Proxima columnar/service mode is WIP.
Date: 2026-08-05

## Implementation status (2026-08-05)

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
  `multi_hop_traverse_parallel`, …). Each Tier-A symbol is durably written as one
  ProximaRecord envelope containing its graph properties, vector, and staleness
  markers under one `oid`; `embedding_ref` is dropped. The indexing pipeline no
  longer performs a vector write followed by a graph-metadata write.
- `victor/storage/graph/cpg_fragments.py` — the Tier-B boundary is live. The
  Proxima adapter routes `statement` nodes and every CFG/CDG/DDG edge away from
  ORION into a durable SQLite fragment index keyed by file, scope, statement
  type, and edge endpoint. Default scans and unfiltered traversals remain Tier A;
  explicit CCG filters and statement/scope lookups drill into Tier B. File/repo
  deletion, restart persistence, bounded edge iteration, and per-tier stats are
  contract-tested. This local indexed representation establishes the replaceable
  boundary; Proxima PAX/columnar fragments remain the service-mode follow-up.
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
  service-mode WIP guard and the SQLite default stay in place until a live
  **bench** (not just parity) passes. Arrow Flight bulk-load and ORION native
  centrality (steps below) remain pending on that measurement.
- **Live parity is verified (2026-08-05).** An earlier revision of this document
  claimed live parity could not be measured because no `proximadb-server` binary
  existed here. That was stale: a release binary was available, and both live
  suites now pass against a real embedded instance — `impact_analysis`
  (forward/backward) and hybrid seed→expand match SQLite, and a vector resolves to
  its graph node by `oid`. Getting there required two fixes, because a live run
  exercised seams the fake-client unit tests cannot reach:
  - **Embedded transport.** ProximaDB #264 made embedded mode portless (UDS
    sockets, no TCP port), so `rest_url` degrades to the host-header sentinel
    `http://localhost`. `ProximaRepoConnection` built a plain TCP client from it,
    so every ORION call failed with `ECONNREFUSED` while ProximaRecord writes
    still succeeded through the SDK's own UDS-aware client. That split transport
    meant the authoritative record committed and its projection never landed —
    the exact skew the atomic-record boundary exists to prevent. Fixed by
    plumbing the socket through (needs `ProximaDBClient(uds_path=…)`, proximaDB
    PR #1447).
  - **Stale parity fixture.** The embedded parity test injected a pre-built
    graph/client, bypassing the connection that owns the record collection, so it
    failed before any assertion. It now drives the production bootstrap.
- **Local source dependency gate:** Victor's development virtualenv resolves the
  pure-Python `proximadb` 0.2.2 SDK directly from `../proximaDB/clients/python`.
  Do not pin a newer PyPI version until ProximaDB publishes it. The native
  `proximadb_embedded` wheel could not be rebuilt because ProximaDB #1021 moved
  the PyO3 bindings into `crates/binding/proximadb-embedded` without updating the
  Python build config: maturin still targeted the root manifest (whose `python`
  feature is now an empty stub and whose lib is `rlib`-only), requested the
  removed `pylib` feature, and no crate declared a `cdylib`. Fixed in proximaDB
  PR #1448. Note this wheel is a **separate artifact** from the pure-Python SDK
  Victor actually uses at runtime — Victor's embedded mode spawns a
  `proximadb-server` subprocess and never imports the native module, so the wheel
  gates the PyPI release, not Victor's local verification.

## Atomic record boundary and failure model

“Atomic” has one precise meaning here: the authoritative write for one symbol is
one `/api/v2/collections/{collection}/records/batch` ProximaRecord containing
`id`, `vector`, and the complete `props` map. ProximaDB commits that envelope on
its canonical record/WAL path. ORION does not currently accept this rich public
record shape as a graph mutation, so its node is a rebuildable traversal
projection applied only after the record commit succeeds.

- A new or changed symbol first writes a pending record with full graph props,
  `has_embedding=false`, and a 384-dimensional zero placeholder. The placeholder
  is necessary because the current v2 public record contract requires a vector;
  semantic search always adds `has_embedding=true`, so pending records cannot
  enter retrieval.
- Embedding completion replaces that same record once with the real vector,
  `has_embedding=true`, and `content_version`. There is no subsequent metadata
  mutation.
- A record failure prevents the ORION projection from being created or changed.
  A projection failure leaves a correct committed record and is retriable; local
  reads prefer the record authority over the stale projection.
- Metadata changes preserve the committed vector and replace the complete record.
  If Victor cannot prove it has the vector needed for replacement, it fails
  closed instead of overwriting it with a placeholder.
- File/repository deletion removes the unified record before deleting its ORION
  projection. The legacy `clear_embeddings` argument cannot split the modalities
  and is retained only as a compatibility input.

This boundary eliminates graph-only and vector-only *authoritative* states. It
does not claim a distributed transaction between the record WAL and ORION; that
would require ProximaDB to expose a graph projection sourced transactionally from
the rich ProximaRecord itself.

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

The opportunity: collapse the two stores into **one authoritative ProximaDB record collection** where a
code **symbol is one durable entity** — complete graph properties plus an HNSW vector — addressed by a
single `oid`, with ORION as its traversal projection. This removes authoritative dual-write skew, makes
the embedding and staleness update one record replacement, and gives the agent native graph algorithms
(impact analysis, centrality, hybrid seed→expand) instead of hand-rolled Python.

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

A code symbol becomes **one authoritative ProximaDB record** with the `victor-codegraph`-emitted
`oid = graph/{repo}/node/{symbol_oid}`, carrying its
properties, its embedding, and a ref to its intra-procedural detail. The same `oid`
keys its vector index entry and derived ORION node, so vector hit → graph node is
identity, not a join.

- **Tier A — semantic graph (HOT, in-RAM):** ~80K symbol nodes + ~96K cross-fn edges + per-node 384-d
  vector → ORION graph + co-indexed vector. Drives `impact_analysis` (forward/backward k-hop), call paths,
  and hybrid semantic-seed → expand. ~120 MB f32 / ~35 MB SQ8 — fits memory.
- **Tier B — intra-procedural CPG (COLD fragments):** statements + DDG/CFG/CDG
  are routed to the durable fragment store and fetched on dataflow drill-down.
  The local implementation is SQLite indexed by file/scope; the service target
  is PAX/columnar fragments per function. Tier B is never included in default
  global scans, so the real size driver stays off the hot graph.
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
   the SQLite store on known symbols — adapter-level always-on, and **verified
   live against a real embedded instance on 2026-08-05** (previously unmeasured).
3. ✅ Enforce the local Tier-A/Tier-B storage boundary with durable, on-demand
   CPG fragments and restart/routing/deletion contract tests.
4. ✅ Replace separate Tier-A vector and graph-metadata mutations with one
   authoritative ProximaRecord replacement; keep ORION explicitly rebuildable.
5. ⏳ Replace the local Tier-B representation with Proxima PAX/columnar
   fragments and verify per-function drill-down parity in service mode.
6. 🟡 Bench footprint + k-hop latency — **measured 2026-08-06** via
   `scripts/benchmark_graph_backends.py`; see "Measured backend comparison"
   below. Arrow Flight bulk-load is **not** benched and cannot be: Victor has no
   Arrow Flight code path (ingest is REST `insert_records` + `batch_create_nodes`),
   so that clause is a feature to build, not a measurement to take.
7. ⏳ Flip the default provider per-repo once parity holds (per-repo `.victor/graph_backend` flag exists; SQLite stays default).

## Measured backend comparison (2026-08-06)

Same corpus, same `GraphIndexingPipeline`, both backends holding **verified
identical data** (the bench suppresses the ratio outright if node counts differ,
because a footprint comparison across differing data is worse than no number).
Tier-A only (`enable_ccg=False`).

| corpus | embeddings | nodes | SQLite | ProximaDB | ratio | SQLite B/node | Proxima B/node |
|---|---|---|---|---|---|---|---|
| 10 files | off | 251 | 1.4 MB | 309 KB | 4.58× | 5,777 | 1,261 |
| 87 files | off | 1,328 | 3.0 MB | 1.1 MB | 2.80× | 2,361 | 843 |
| 10 files | on | 251 | 8.1 MB | 987 KB | 8.38× | 33,729 | 4,026 |
| 87 files | on | 1,328 | **42.8 MB** | **4.3 MB** | **9.88×** | 33,827 | 3,424 |

Four findings, none of which were visible before the bench existed:

- **The footprint win is real and comes from vectors.** Graph-only, the ratio
  *shrinks* with scale (4.58× → 2.80×). With embeddings it *grows* (8.38× →
  9.88×), and SQLite's cost stays pinned near 33 KB/node while Proxima's falls.
  This is the design's premise — the SQLite+LanceDB pair stores vectors
  inefficiently — confirmed by measurement rather than projection. It also means
  a graph-only benchmark understates the case, and extrapolating any single
  corpus size overstates or understates depending on which mode it ran in.
- **Ingest reverses under embeddings.** Proxima is ~3× *faster* to index without
  them (3.5 s vs 10.4 s) and ~3.5× *slower* with them (40.3 s vs 11.5 s). The
  embedding path replaces one full ProximaRecord per symbol, so vector
  completion pays a whole-record write; SQLite updates a column. Worth
  attention before recommending Proxima for large first-time indexes.
- **SQLite holds a consistent ~10× traversal advantage** (k-hop p50 0.06 ms vs
  0.65 ms across every configuration).
- **Step 7 is therefore a trade-off, not a graduation.** ~10× less disk against
  ~10× slower traversal and ~3.5× slower embedded ingest. Which side wins
  depends on whether a given repo is disk-bound or query-bound; that argues for
  keeping the per-repo flag and a documented recommendation rather than flipping
  a global default.

Not yet measured: hybrid seed→expand latency under load, SQ8 cold mode, and
behaviour at the 3,659-file / 2.4 GB scale the original figure came from.
