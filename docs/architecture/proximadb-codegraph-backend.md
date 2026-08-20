# ProximaDB as the Code Context Graph (CCG) Backend

Status: Embedded backend and local Tier-A/Tier-B boundary implemented behind a
per-repo flag; SQLite remains default (tracked as TD-11, TD-12, TD-13 in
`../tech-stack.md`). Proxima columnar/service mode is WIP.
Date: 2026-08-12

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
  A projection failure leaves a correct committed record and is retriable.
  Logical edge enumeration reads the record authority exclusively: it never
  unions ORION into the result and fails closed if the record scan is unreadable.
  Otherwise a lagging projection could resurrect a canonically deleted edge.
  Cached enumerations are revalidated against ProximaDB's server-owned
  `content_revision_token`. The token combines the monotonic content revision
  with a server-incarnation epoch, so a restart cannot validate stale data via
  revision-number reuse. Process-local mutation generations remain a fast path,
  not the cross-process freshness authority.
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

- **Embedded (local single-repo):** one `proximadb-server` subprocess owns the
  repo's local writable roots; clients share that owner over UDS. ProximaDB
  rejects a second server for any overlapping local authority. SDK startup
  accepts `/health` only when its process identity matches the child it spawned,
  so a pre-existing server cannot be mistaken for a successful launch.
  `EmbeddingMode::Memory` remains the Tier-A vector mode. The separate native
  PyO3 package is not the runtime used by this adapter.
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

> **Superseded — the SQLite footprints below are inflated by a measurement bug.**
> The harness summed the project directory before SQLite's write-ahead log was
> checkpointed. Victor holds a process-global connection, so `project.db-wal`
> survived `store.close()` and was counted, then deleted moments later at
> interpreter exit. The error is large and **one-sided**: ProximaDB measured
> identically before and after, so every ratio here flatters ProximaDB.
> Corrected repo-scale figure: SQLite 196.8 MB as measured vs **125 MB at rest**.
> The "SQLite costs ~33 KB/node with embeddings" result is the same artifact —
> at rest it is ~3.8 KB/node. Fixed in `_checkpoint_sqlite_wals`; see the
> 2026-08-19 section for numbers taken with the fix in place.

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

> **Read the ratios above with the caveat below.** Every row runs
> `enable_ccg=False`, which is *not* what `victor init` does. In the default
> CCG-on configuration the footprint advantage **disappears** — see
> "End-to-end reality check". The vector-efficiency result is real but applies
> to the symbol tier, which statement-level CPG dwarfs in practice.

Four findings from the Tier-A slice:

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
- **SQLite's ~10× traversal advantage does not matter at these magnitudes.**
  k-hop p50 is 0.06 ms vs 0.65 ms — a large ratio on a negligible absolute
  number. Graph reads happen in tool calls inside an agent turn whose LLM
  round-trip is measured in *seconds*; even 100 graph queries per turn costs
  ~65 ms on Proxima, well under 1% of the turn. Retrieval latency is not a
  differentiator between these backends and should not be weighted as one.
- **Footprint is the decisive axis, and worktrees are why.** Victor development
  routinely runs many linked worktrees at once, each carrying its own
  `.victor/project.db` and embeddings. Per-worktree indexing is where a ~10×
  reduction stops being a nice-to-have: the SQLite+Lance pair does not scale
  across concurrent worktrees; Proxima does.
- **Ingest is the one real regression.** ~3.5× slower with embeddings, paid on
  first index and session start — the moments a developer actually waits.
  Extrapolating the 87-file result linearly to Victor's ~1,452 source files
  suggests roughly 12 min (Proxima) vs 3.4 min (SQLite). That is an
  order-of-magnitude estimate, not a measurement.

## End-to-end reality check (`victor init`, 2026-08-06)

The table above measures a slice, not the product. Running the **real CLI** —
`victor init --no-deep --no-interactive --force`, CCG on (the default), identical
70-file corpora, backend chosen by the `.victor/graph_backend` marker:

| | SQLite | ProximaDB |
|---|---|---|
| init wall time | 22.59 s | **10.71 s** |
| nodes | 27,265 | 25,937 |
| edges | 75,626 | 74,204 |
| footprint | **52 MB** | 54 MB |

**Footprint is a wash, not ~10×.** The Tier-A bench disables CCG; init enables it,
and statement-level CFG/CDG/DDG then dominates the graph. The Tier-A/Tier-B
boundary routes all of that to a local SQLite fragment store, so the Proxima
configuration is mostly SQLite by volume:

```
50 MB   .victor/proximadb/cpg_fragments.sqlite3   <- Tier-B SQLite
1.1 MB  .victor/proximadb/data                    <- actual ProximaDB storage
```

ProximaDB holds 1.1 MB of the 54 MB. The vector-efficiency result stays true but
governs only the symbol tier. **Until Tier-B moves to Proxima PAX/columnar
fragments (checklist step 5), a footprint argument for this backend is measuring
the wrong thing.**

**What the default configuration does show is ingest: ProximaDB is 2.1× faster
end to end** (10.71 s vs 22.59 s). That, not disk, is the adoption argument
today — which makes the write-amplification gap (ProximaDB issue #1479, no
partial/vector-only record update) the thing worth pushing upstream, since it is
what stops that lead from being larger.

**Blocking gap for step 7:** the backends **do not hold identical graphs** through
init — Proxima is short 1,328 nodes and 1,422 edges. Flipping any default before
that has a root cause would ship a silently lossier index.

Two bugs had to be fixed before this comparison could run at all, both silent:
`victor init` hardcoded the SQLite backend while the read paths honored the
marker (split-brain), and `ProximaGraphStore.stats()` omitted top-level
`nodes`/`edges`, so init raised `KeyError('nodes')`, printed only
`! CCG indexing skipped: 'nodes'`, and reported success having built no index.

Not yet measured: hybrid seed→expand latency under load, SQ8 cold mode, and
behaviour at the 3,659-file / 2.4 GB scale the original figure came from.

## Repo-scale measurement and the ingest blocker (2026-08-19)

The 2026-08-06 comparison above measured corpora of 10–87 files. This run
measured the **whole repository**, and the conclusion changes at scale: the
blocker is not traversal latency, it is **ingest and recovery**.

Both backends indexed the same corpus through the same `GraphIndexingPipeline`
and ended at byte-identical graph size — **93,457 nodes / 177,031 edges** —
so nothing below is a data-difference artifact. Embeddings off (Tier-A only).

| | SQLite | ProximaDB | ratio |
|---|---|---|---|
| index time, before the fix below | 447 s | 3,269 s | 7.3× |
| **index time, current** | **395 s** | **1,748 s** | **4.4×** |
| footprint | 196.8 MB | 473.4 MB | 2.4× larger |

**The ingest gap was superlinear, and the cause was on the client.** The same
comparison on a 33,205-node corpus gave 379 s vs 331 s — only 1.14×. A ratio
that grows with the graph is the signature of per-item work scaling with graph
size, and here it was `upsert_edges` preloading `graph.get_all_edges()` to
dedup: the SDK implements that as per-node outgoing-edge scans, ~93k HTTP
requests, re-paid on cache invalidation. Removing it (below) cut ProximaDB
ingest **47%** with no server change, and the ratio fell 7.3× → 4.4×.

What remains is a **constant-factor** ~4.4× on ingest — a far less alarming
shape than the quadratic it replaced, and one nothing here has yet attributed.
Candidates are the HTTP/UDS boundary and per-batch JSON, the canonical record
write accompanying each edge batch, and WAL volume. `bulk_insert_edges` already
emits `validate/wal/mempool/index/csr` timings, so the per-phase trace is cheap
to obtain; it should be taken before anyone optimizes.

**A correction worth recording:** this section first attributed the ingest gap
to ORION rebuilding its CSR per edge insert. That was wrong. `batch_create_edges`
already routes through `bulk_insert_edges`, which batches index creation and
defers to a threshold-triggered compaction, so the per-edge rebuild was never on
the ingest path. It *is* real on the **replay** path, which is why recovery
behaves as described below; anvai-labs/proximaDB#1673 was rescoped accordingly
and anvai-labs/proximaDB#1678 fixes it.

**Traversal latency is not quoted here on purpose.** k-hop p50 measured
SQLite 0.33 ms vs Proxima 1.76 ms in one run and SQLite 4.77 ms vs Proxima
1.50 ms in the next — SQLite moved 14× between runs while Proxima barely moved.
The harness samples 75 traversals with seeds re-chosen per run and no cache
control or repeats, so it cannot support a claim in either direction yet.

**Recovery inherits the same defect, and this is the adoption blocker.**
Reopening the 478 MB data directory replays ~77k batched operations through the
same per-edge rebuild. It never reached `serving`: 900 s via the SDK, and still
recovering at 813 s when run manually on an idle machine. A graph that cannot be
reopened in 15 minutes is unusable regardless of query speed.

**Recommendation: keep SQLite as the default backend for repo-scale graphs
until #1673/#1678 land.** Traversal latency is not the reason and never was —
a millisecond inside an agent turn whose LLM round-trip is measured in seconds
is noise, as the 2026-08-06 analysis already argued, and the current numbers are
too unstable to quote anyway. Recovery is the blocker: a graph that cannot be
reopened is unusable at any query speed. Ingest and recovery are the
gating axes.

### What this run fixed on the Victor side

Two defects found while measuring, both fixed here:

- **`upsert_edges` no longer preloads every edge id.** It called
  `graph.get_all_edges()`, which the SDK implements as per-node outgoing-edge
  scans — **one HTTP request per node**, ~93k round trips, re-paid on any cache
  invalidation. It existed to work around ORION's create-only batch aborting at
  the first duplicate and silently discarding the rest; proximaDB#1647 replaced
  that with per-edge admission ("remainder of the batch applies"), so the scan
  is obsolete. A free in-process set still prevents re-sending within one run.
- **The 30 s startup ceiling is gone.** `start_embedded_db` hardcoded it, and
  since proximaDB#1668 an explicit caller timeout deliberately overrides the
  SDK's progress-aware wait — so Victor's constant became the binding limit.
  Startup cost scales with the data directory (replay and rebuild happen before
  the listener binds), so a fixed budget is the wrong shape: it passes on toy
  repos and fails on real ones. The default now delegates to the phase-watching
  wait, which gives up on a *stall* rather than a clock.

Neither fix can show its full benefit while the server is quadratic; re-measure
ingest once #1673 is fixed.

### Method notes (so these numbers can be re-run)

- Reproduce with `scripts/benchmark_graph_backends.py --corpus . --backends
  sqlite,proxima`, and retrieval quality with
  `scripts/eval_graph_retrieval_value.py`.
- The SDK under measurement must be a **clean** checkout — the benchmark refuses
  to measure otherwise, because a number that cannot be attributed to a commit
  cannot be reproduced.
- Timing runs need a quiet machine. Recall does not: it is unaffected by load,
  so quality numbers from a busy machine are still sound while latency is not.
- Two plausible causes were checked and **rejected** before landing on the CSR
  rebuild: WAL replay re-appending to the WAL (the `is_replaying()` gate works;
  the surrounding debug lines print even when the append is suppressed), and
  replay dropping edges on ordering (exactly 1 failed frame out of ~77k).

## Total-cost comparison with embeddings (2026-08-19)

Every figure in the section above is **graph-tier only** — those runs had
embeddings off, so the SQLite arm never wrote LanceDB and the ProximaDB arm
never wrote vectors. That excludes exactly the tier the backend was adopted
for, so it cannot support a storage conclusion in either direction.

With embeddings on, both arms measured after the footprint fix
(`_checkpoint_sqlite_wals`), `victor/storage` corpus, 1,350 nodes / 2,258 edges:

| | SQLite + LanceDB | ProximaDB | ratio |
|---|---|---|---|
| index time | 15.78 s | **6.33 s** | ProximaDB **2.5× faster** |
| footprint at rest | **4.9 MB** | 7.0 MB | ProximaDB **1.4× larger** |
| bytes/node | 3,829 | 5,404 | |

Two results here reverse earlier conclusions, and both reversals come from
fixing measurement rather than from any code change:

- **ProximaDB wins ingest once vectors are counted.** The 2026-08-06 analysis
  concluded the opposite ("ingest reverses under embeddings … ~3.5× slower"),
  and the graph-only runs above show ProximaDB 4.4× *slower*. Both omitted the
  LanceDB write the SQLite arm must pay. Including it, ProximaDB is 2.5×
  faster, because one system holds graph and vectors while the other writes
  twice.
- **ProximaDB loses storage.** The 9.88× advantage previously reported was the
  WAL artifact described above. At rest, ProximaDB is *larger* — 5,404 B/node
  against 3,829 — at this corpus size.

Vector **read** cost is still unmeasured on both sides; nothing here compares
LanceDB similarity search against ProximaDB's vector engine, so no claim is
made about it.

Scale caveat: this corpus is 1,350 nodes. The graph-only runs show ratios
moving with scale, so these numbers should be re-taken at repo scale before
they decide anything. They are reported now because they overturn the sign of
two published conclusions, not because they settle the question.
