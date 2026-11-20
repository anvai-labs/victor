# Native Acceleration Strategy

**Status:** Adopted engineering guidance; implementation investigations remain incremental

**Last verified:** 2026-11-19 (FOU-C3 parity inventory; CCG identity, graph semantics and wheel matrix)

Victor should keep Python as the orchestration, integration, and I/O language and use the existing
Rust/PyO3 workspace as its default compiled acceleration path. Adding Cython, Numba, or a direct
CPython C/C++ extension would create another compiler, wheel, debugging, and parity surface. Such a
tool is an exception that requires measurements showing the existing choices cannot meet the need.

## Current codebase reality

- `rust/crates/python-bindings` builds `victor_native` with PyO3 0.30 and maturin. The workspace
  already covers tokenization, context fitting, similarity, embeddings, graph algorithms, parsing,
  filtering, indexing, and security scans.
- Python adapters under `victor/processing/native/` retain reference or fallback behavior. The
  required `native-parity` CI job builds a release wheel and checks native/fallback agreement —
  originally for the tokenizer and context fitter, broadened by FOU-C3 to embeddings, secrets,
  graph algorithms, trace scanning, dedup/hashing, classifier keywords, and JSON repair (see the
  inventory below; the remaining gaps are listed there as future increments).
- Rust release builds use optimization level 3, fat LTO, one codegen unit, and panic unwinding so a
  Rust panic does not abort the Python process.
- Release automation reuses the affected-change native qualification matrix: Linux x86-64 and
  ARM64, macOS x86-64 and ARM64, and Windows x64, each on Python 3.12 and 3.13. Windows ARM64 is
  absent. Builds, installed-wheel checks, source-platform qualification and releases are distinct. All packages
  require Python 3.12+; a supported interpreter does not imply a wheel for every platform.
- The FFI is already batch-oriented in several places, but the current source scan finds
  `Python::detach` concentrated in similarity operations. Several embedding APIs still accept
  nested `Vec<Vec<f32>>`, which copies Python-owned data at the boundary. These are higher-value
  investigations than adding another native language.

## Required decision sequence

The [2026-11-02 control-dependence correction](graph-control-dependence.md)
supersedes the earlier complete-CCG reachability interpretation below. Statement
schema v2 uses physical byte spans; the native batch returns CFG/DDG inputs and
the facade derives bounded exit-path postdominance in typed Python. Historical
shape/parity timings do not establish semantic correctness or current output
parity. Source CFGs remain explicitly heuristic and do not supply verified
semantic-ranking confidence.

1. **Define the user-level target.** Record the deployment shape and a latency, throughput, CPU, or
   memory target. Include realistic input distributions and concurrency.
2. **Measure the whole path.** Profile Python and native variants, including import/startup cost,
   serialization, allocation, FFI conversion, locking, and fallback selection.
3. **Fix structure first.** Remove repeated work, use a better algorithm, batch calls, bound caches,
   reduce object churn, and avoid blocking the event loop. Re-measure.
4. **Qualify a native candidate.** The operation must be deterministic or have an explicit state
   contract, CPU-bound, batchable, and responsible for at least 10% of end-to-end time or a missed
   service target. A prototype must improve its production-size operation by at least 2x and the
   user-level path by at least 10%, after conversion costs.
5. **Choose the narrowest implementation.** Use the table below. Preserve one authoritative
   contract and avoid parallel execution engines.
6. **Prove delivery.** Add parity and failure tests, benchmarks at break-even and production sizes,
   wheel/install checks for supported platforms, and native/fallback dispatch telemetry.

These thresholds are admission gates, not promises that a rewrite will merge. Correctness,
maintainability, portability, and operational failure behavior can still outweigh a speedup.

## Implementation choice

| Choice | Use when | Constraints |
|---|---|---|
| Typed async Python | Control flow, provider/tool I/O, orchestration, rapidly changing policy | Default. Offload blocking work and keep dependencies lazy. |
| Existing vectorized library | A supported deployment shape already carries NumPy or another optimized kernel and the operation maps naturally to it | Do not make a heavy optional dependency mandatory for core execution. Measure data conversion. |
| Rust + PyO3 | Stable CPU-bound parsing, scanning, graph, token, or numeric batch work; shared logic with the Rust edge runtime | Preferred compiled path. One crossing per batch, owned/contiguous data where practical, `Python::detach` for multi-millisecond Rust-only work, no Python callbacks while detached. |
| Rust subprocess/sidecar | Crash or memory isolation is required, or the same engine must run independently of Python | Budget IPC, startup, cancellation, version negotiation, and process cleanup. |
| Cython | A hot typed loop is tightly coupled to an existing C API or Python object model and a Rust implementation would add materially more risk | Exception only. Requires comparative prototype, cross-platform wheels, GIL/free-threading plan, generated-code debugging plan, and explicit owner. |
| Direct C/C++ extension or CFFI | An unavoidable upstream C ABI has no safe maintained binding | Keep the boundary narrow. Prefer wrapping it behind the existing Rust/native facade; document memory ownership and sanitizer coverage. |
| GPU kernel/runtime | Large, batched tensor work dominates the selected deployment shape | Optional extra only; include transfer/setup cost and a CPU fallback. Never move agent control flow to the GPU. |

Numba or runtime compilation is not a core-runtime default: cold compilation, cache behavior,
platform support, and deployment dependencies conflict with predictable CLI and library startup.

## FFI and platform rules

- Pass batches or persistent native handles, not one Python object per call. Establish and benchmark
  the break-even size; keep smaller inputs on Python when the crossing costs more than the work.
- Prefer contiguous buffers or the Python buffer protocol for numeric arrays. Avoid repeated nested
  list-to-vector copies and return compact results.
- Release interpreter attachment for Rust-only computation expected to take milliseconds, and never
  wait on a Rust lock or async operation while attached when that can deadlock Python progress.
- Convert panics and typed native errors at the boundary. A fallback may handle a pure computation
  that failed before effects; it must not silently retry network, filesystem, tool, or billing
  effects.
- Publish conservative CPU baselines and use runtime dispatch for optional SIMD. Test x86-64 and
  ARM64 explicitly; never publish artifacts compiled for the build host's CPU.
- Consider PyO3 `abi3` only after an install/parity benchmark proves its API restrictions fit the
  extension. Stable ABI can reduce the wheel matrix, but it does not remove OS/architecture builds
  and must be evaluated separately from free-threaded `abi3t` support.

## Investigation backlog

1. Build a benchmark and parity inventory for every exported native operation: reference coverage,
   Python/native wall time, break-even input size, peak memory, conversion share, interpreter
   attachment, and call volume in representative agent sessions.
2. Audit CPU-heavy bindings outside `similarity.rs` for safe `Python::detach` regions. Start with
   tokenizer batches, embedding transforms, graph algorithms, and trace scanning.
   **Audited and landed 2026-11-18 — see the `Python::detach` audit subsection below.**
3. Prototype contiguous-buffer inputs and persistent native indexes for embedding/similarity paths;
   compare them with current nested-vector conversion and the existing NumPy fallback.
   **Parity half landed 2026-11-19 (FOU-C3)** — see the parity/benchmark inventory subsection
   below; the contiguous-buffer/persistent-index prototype remains open.
4. Make the wheel support table explicit and tested. Track the declared Python minimum alongside
   the wheel matrix, then evaluate an `abi3` proof build before expanding per-version artifacts.
5. Add advisory performance reports first. Ratchet only stable production-size benchmarks with
   noise controls; never make microbenchmarks the sole merge gate.
6. Revisit Cython only if this work identifies a qualifying hotspot for which Rust/PyO3 and an
   already-installed vector library both fail the decision gates.
7. **Code-index parse parallelism (measured 2026-10-24, self-analysis round 2 W2).** The graph
   indexer's parse phase is the dominant cost (1,719s busy of a 957s wall run on the victor
   corpus at 16 cores; the thread pool achieves only ~1.8x effective). Three measurements now
   constrain the fix space:
   - **py-tree-sitter's `parse()` holds the GIL** — a controlled probe (589KiB python source,
     8 threads x 5 parses) measured 1,386ms wall vs 181ms perfect-scaling and 1,449ms
     GIL-bound predictions. Threads cannot parallelize the parse even at the C-binding level.
   - **Repository prep is NOT a hidden serial bound**: `prepare_repository_snapshot`
     (codegraph `parse_repo` over 4,921 files, scope binding on) measures **45.4s** serial —
     the executor phase (CCG building + projection) is where the time is.
   - **Process-IPC parallelism is a measured dead end**: a full fork-based process-parse
     implementation (fork-inherited pipeline, warmed imports, parent-side phase timing) ran the
     same corpus at 1,411s wall vs 957s thread mode — normalizing for a fresh 499s pyright
     harvest (vs 56s cached), the pipeline itself was a wash (~912s vs ~901s). The GIL win was
     consumed by pickling full `ParseResult`s — the ~1.5M CCG statement `GraphNode`s make the
     per-file IPC payload enormous. The experiment was removed per the completion gate.
   Consequence: the only remaining path to parse-phase parallelism is the **in-process native
   extraction engine** (Rust/PyO3 in victor-codegraph: tree-sitter crates are first-class Rust
   over the identical C core, rayon intra-batch, `py.allow_threads`, extraction stays
   in-process so no result pickling). The serial write path (595s busy, 62% occupancy) is the
   second bound — WAL/batch tuning pairs with the engine for the full win. Any future
   "just use processes" proposal must answer the measured IPC objection first.

   **Engine phase 1 LANDED (vertical slice, 2026-10-24)**: native CCG statement extraction
   in the victor-ai Rust workspace (`ccg_extract.rs`, tree-sitter-python/-javascript/
   -typescript grammars, `Python::detach` + rayon, one FFI crossing per batch).
   Record-for-record byte parity with the Python walk (differential tests over repo
   fixtures, python + typescript; scope cascade and sha256-16 ids identical) and a
   measured **28.2x** on the statement-extraction walk (800 files / 188,200 statements:
   3.7s Python → 0.1s native, counts identical). The Python walk STAYS as oracle and
   fallback; `GraphNode` construction remains in Python. Next engine phases: the CFG/CDG/
   DDG edge builders over the native statement records, then codegraph symbol extraction.

   **Engine phase 2 LANDED (complete CCG, 2026-10-24)**: `build_ccg_batch` — statements +
   CFG (sequential + structural) + CDG (reachability domination) + DDG (def-use chains) in
   one detached batch call. **81.1x measured** on the complete per-file CCG (600 files /
   146,191 nodes / 4,334,417 edges: 81.2s Python → 1.0s native, counts identical).
   Parity quirks the port had to reproduce exactly (all pinned by differential tests):
   the class defines `_should_connect_cfg` TWICE — the second definition (file-based,
   `abs(gap) <= 30`, same-line pairs connected) shadows the first (10-line) — the port
   follows the executed one; structural edges walk in LIFO stack order; CDG emits from a
   hash-randomized set (native returns sorted; consumers upsert by key — order-insensitive
   by contract); and `_build_ddg` DOUBLE-RECURSES (`_walk_ast` drives a visitor that also
   recurses children — every subtree visited once per ancestor path, 63k visits on a
   750-line file), massively redundant and precisely the Python-side cost the native path
   removes. Remaining engine phase: codegraph symbol extraction (`victor-codegraph`).

   **Boundary measurement + resolution knob (2026-10-24, #1500-series):** the
   plugin-vs-native delta measured precisely on precise_providers.py — the
   victor-coding plugin graph is a DIFFERENT SHAPE, not a superset: 1161 vs
   754 statement nodes (plugin emits `block` bodies, +400 nodes incl.
   148 blocks/96 conditions/113 returns), per-node metadata enrichment
   (definitions/uses/language provenance), and fundamentally different
   CDG/DDG algorithms (plugin CDG 255 vs native 30,119 edges from
   reachability domination; plugin DDG 2,025 vs native 568 from the
   double-recursion semantics). Full-parity porting was scoped out at the
   B1 gate per the recorded fallback. Landed instead:
   `search.graph.ccg_builder = auto|native|plugin|builtin` (default auto =
   plugin-first, the richer graph — never a silent flip), threaded from
   GraphIndexConfig into every per-thread CodeContextGraphBuilder;
   `native` buys the 81x engine at the measured shape trade. The full
   merge (native absorbs plugin semantics) remains the documented
   follow-up with this delta table as its evidence base.

   **Plugin-builder phase profile + the boundary decision sharpened (2026-10-25).**
   A phase-level wall-clock profile of the enhanced builder
   (`scripts/benchmark_ccg_builder_phases.py`, boundary corpus,
   1161 nodes/3321 edges) — six instrumented call sites per file, NOT
   per-call wrappers: walk **40.5 ms (79%)**, edges 4.5 ms, hints 2.5 ms,
   dedupe 2.4 ms, parse 2.2 ms. The identifier/call extractors the
   instrumented profile had put at "58%" cost **~4 ms** in wall time —
   cProfile's per-call overhead on 2.8M calls fabricated the share. A
   memoized single-pass extraction was built, proven byte-identical
   (6 files, 11k nodes incl. metadata), measured **perf-neutral**
   (54-59 ms both ways), and removed per the complete-only-if-improved
   gate; per-node interpreter steps (~50k tree nodes) are the walk's
   floor, so no Python-side reordering of the builder buys more than a
   few ms. **Decision: Option B — route the plugin builder through the
   native statement walk** (extend `extract_statements_batch` with the
   plugin's emission classes: block bodies, conditions, returns, plus
   per-statement definitions/uses/calls/preview; differential parity on
   shape AND metadata per language; then the plugin consumes native
   records and keeps its CDG/DDG/dedupe in Python, ~10 ms/file combined).
   Projected file cost ~15 ms vs ~51 ms today with the graph shape
   preserved exactly. The parity oracle is the golden corpus at
   `verticals/victor-coding/tests/codebase/goldens/` (six languages,
   statement shape + enrichment, path-normalized) — the native port's
   differential test reads the same samples/goldens as pure data, so
   semantics stay owned by victor-coding and no root→vertical import
   appears. Symbol hints stay Python-side in the first phase (~2.7 ms);
   they ride into the native batch as data.

   **Option B phase 1 LANDED (2026-10-25):** ``extract_plugin_statements_batch``
   in the root crate — a table-driven walk (classification/identifier/scope
   tables + hints passed as data; the engine stays language-agnostic) with
   per-statement enrichment (definitions/uses/calls, preview, visibility,
   symbol hints, keyword filtering, the ``pieces[-3:]`` qualified-name
   rule). Golden parity: python AND typescript reproduce all 63/35
   statements field-for-field on the first differential run; languages
   without a native grammar decline (None), never guess. Measured on the
   boundary corpus: native walk+parse **15.7 ms** vs Python walk 36.8 +
   parse 2.3 — a 2.5x slice gain, not the builtin's 81x: the port
   FAITHFULLY reproduces the plugin's per-statement subtree re-walks
   (~410k node visits at native speed). The next engine increment is the
   bottom-up single-pass memo IN RUST (the Python memo was perf-neutral
   because interpreter overhead dominated; in Rust one post-order pass
   replaces the re-walks, projected ~2-4 ms walk, ~10 ms/file end-to-end)
   — re-validated against the same goldens. Not yet wired into the
   consult path; phase 2 flips ``PluginBackedCCGBuilder`` onto it behind
   the builder-resolution knob.

   **Single-pass memo LANDED (same date):** one post-order pass memoizes
   per-node (identifiers, calls) keyed by ``Node::id``, replacing the
   faithful per-statement subtree re-walks; misses (nodes under skip
   subtrees) fall back to the recursive walk so the memo is exact, never
   approximate. Golden parity unchanged. Measured boundary corpus:
   walk+parse **15.7 -> ~10 ms** (39.1 Python -> 10 native = 3.9x on the
   slice; the remaining ~9 ms is parse + memo build + emission — sha256
   ids, previews, string building for 1161 records). End-to-end lands at
   phase 2 (hints 4.8 Python + walk 10 native + edges/dedupe ~7 Python).

   **Phase 2 LANDED — the consult-path flip (2026-10-25):** the runtime
   consult site injects the native extractor into the enhanced builder as
   a duck-typed attribute (``_native_extractor``; no cross-package import
   in either direction — the builder passes its OWN tables via
   ``native_tables()``; an absent, declining, or failing extractor simply
   keeps the Python walk, which stays the typed reference
   implementation). The builder's edge algorithms and metadata assembly
   never moved. Two parity-level fixes surfaced by the end-to-end
   differential: Rust returned 25-capped enrichment lists while the
   Python StatementRecord carries full sets (DDG edges on >25-identifier
   statements would have diverged — the cap belongs to the metadata
   layer; Rust now returns full sorted sets), and preview capping sliced
   CODEPOINTS in Python but BYTES in Rust (160 chars vs 158 on multibyte
   docstrings). After both: **the boundary corpus is fully identical
   end-to-end** — 1161 nodes, 3321 edges, every metadata field — with
   python and typescript samples likewise. Measured: **69.7 -> 46.1
   ms/file through ``build_ccg_for_file`` with the default auto
   resolution (1.5x)**; the remaining delta vs the ~22 ms projection is
   Python-side GraphNode assembly per statement plus the double parse
   (py-tree-sitter for hints, Rust for the walk). The ``plugin`` knob
   value remains the forced-Python-walk escape hatch.

   **Codegraph symbol extraction, phases 0-1 (2026-10-25).** The recorded
   "remaining engine phase" is now measured precisely
   (`scripts/benchmark_codegraph_phases.py`, wall-clock only). Repo mix:
   4,950 source files, **96.5% python** (stdlib-ast path), 99 ts, 37
   rust. Per-file: python 44.3 ms (59.4 with refs), ts 52.5/85.8, rust
   13.8/30.9; the parse() wrapper and repository binding are innocent
   (~1 ms and small). Phase 1 (pure-Python, output-identical —
   differential-pinned on 6 real files x refs on/off): one-pass
   post-order cyclomatic precompute replacing per-symbol subtree
   re-walks; ``_params`` unpar sed once (was twice per function);
   ``_calls_owned_by`` once per function (was twice under
   emit_references); resolver class-namespace set precompute (was
   O(R*C*S)); per-grammar Parser cache; repository binding via the
   existing ``by_path`` index (was a linear scan per relation x
   binding). Result: **python 37.8 ms (-15%), +refs 46.0 (-22%)**;
   rust/ts ~-10%. Extrapolated index impact ~4.5% of the ~1,412 s parse
   busy — the halving gate is NOT met, and the measured remainder is
   DISTRIBUTED visitor cost (~19 ms/file: per-symbol source slicing,
   lexical-binding/reference walks, unparse, dataclass assembly) with no
   single algorithmic defect left. Phase 3 (native tree-sitter walk for
   the symbol path) is therefore justified BY MEASUREMENT, with the
   python stdlib-ast parity trap noted in the plan. Measurement trap
   recorded: victor-codegraph's ``__editable__`` finder intercepts
   BEFORE sys.path, so worktree benchmarks silently measure develop
   twice — the /tmp/tier3-shim sitecustomize now evicts all
   ``__editable___victor`` finders.

   **Reproducibility defect found and fixed while measuring (same date):**
   the enhanced builder's CFG and DDG ordered records by the path-derived
   `node_id`, so byte-identical source indexed from two directories
   produced different graphs (3333 vs 3331 vs 3321 edges on the boundary
   corpus — same-line statements chained in hash order). Both sorts now
   tie-break by walk order (tree-sitter source order); graphs are
   path-independent (pinned by a regression test; three directories,
   identical edge signatures). Graphs built before the fix may differ by
   a few same-line CFG/DDG edges — reindex reconciles (derived,
   rebuildable state by design).

   **End-to-end re-measure with phases 1-2 active (2026-10-24) — two findings that
   reshape the remaining work:**
   - **The victor-coding enhanced CCG builder shadows the native path for the dominant
     language.** `build_ccg_for_file` consults the CapabilityRegistry's enhanced builder
     FIRST; on installs carrying the victor-coding vertical (this repo's editable install,
     and any `victor-ai[graph]` + verticals deployment) Python files go to the enhanced
     builder's own Python implementation — the native engine only serves the languages the
     enhanced builder does not claim. E2E evidence: parse busy fell only 1,719s → 1,412s
     (the non-Python languages), not the collapse the 81x slice predicts. Before any
     further engine investment, decide the boundary: either port/absorb the enhanced
     builder's semantics natively, or route it through the native statement walk.
   - **The write path dominates and its obvious suspects are measured-INNOCENT.**
     Persist busy 579.7s (wall 1,385.8s incl. a fresh 481.9s pyright harvest; normalized
     ≈ the pre-engine 957s baseline). Micro-benchmarks of the suspected hot spots, both
     REJECTED: the FTS5 sync triggers cost ~1.2s/150k node upserts ≈ 12s/run (statement
     nodes are the vast majority and are FTS garbage — skipping them in the triggers is a
     real but small cleanup, ~2%); the edge-upsert per-row COALESCE file subqueries cost
     ~0.9s/400k edges ≈ 9s/run. Together ~20s of ~580s — the remaining ~550s lives in the
     Python-side row preparation and orchestration inside `_graph_store_write_batch`
     (GraphEdge/GraphNode → row tuples with per-edge `json.dumps`,
     `_drop_dangling_edge_rows`/`_clear_dangling_parent_links` per-batch queries,
     `_order_rows_parents_first`, mtime writes). Next step, before any rewrite: cProfile a
     single `_IndexingStreamPipeline._commit_batch` of ~20 realistic files and attribute
     the batch wall time to those components; only then choose the fix (batched JSON
     serialization, hoisted lookups, or precomputed ordering).

   **Write-path attribution COMPLETE (2026-11-18, `scripts/benchmark_persist_batch.py`):
   no rewrite is justified — the residual time is SQLite doing the write volume.**
   The prescribed profiling step ran: a deterministic benchmark builds realistic
   flush batches (shapes measured from this repo's live project.db: ~311 nodes/file at
   94% statements, ~767 edges/file in the observed type mix, absolute file hints on
   100% of edge metadata) and times one `_IndexingStreamPipeline._flush`-shaped
   `write_batch` with perf_counter brackets per statement group (cProfile only as
   supporting evidence, per the tiny-call-count lesson).

   Attribution, cold 200-file flush (3,431 symbol + 57,752 statement nodes,
   2,541 symbol + 150,982 CCG edges), total wall 4,325 ms on the WSL bench host:

   | component                                   | wall ms | share |
   |---------------------------------------------|--------:|------:|
   | SQLite `executemany` — node/edge ON CONFLICT upserts | 3,004 | 69% |
   | Python row prep on the loop (per-row `json.dumps`, canonical-path, tuple build) | ~700 | 16% |
   | dangling-guard SELECTs + parent ordering     |     213 |   5% |
   | `BEGIN`/`COMMIT` (WAL auto-checkpoint backfill) | 254  |   6% |
   | per-file mtime upserts (20 executor hops + hash) | ~43 |   1% |

   Findings that close the question:

   - **The executemany statements ARE the cost** (~11.5 µs/edge row, ~19.7 µs/node
     row, fresh inserts; ~1.5x on the ON CONFLICT UPDATE re-index path). The SQL is
     already `executemany`-batched, every plan is a PK/secondary-index search
     (`EXPLAIN QUERY PLAN` printed by `--explain` — no missing index), and the
     COALESCE subqueries short-circuit because 100% of real edge rows carry a file.
     There is no Python-side orchestration defect left to fix; the earlier "~550s in
     row preparation" hypothesis was an artifact of pre-memoization measurements
     (the canonical-path cache already landed).
   - **`json.dumps` is the only real Python item** (0.38 s cProfile cumtime per
     214,706 dumps ≈ 13 s per full index) — an order of magnitude below the rewrite
     bar, and already irreducible without changing row shapes.
   - **Commit cost GROWS with database size** (245 ms → 3.3 s per flush as the bench
     DB accumulates) because one flush writes far more WAL than the 2000-page (8 MB)
     auto-checkpoint threshold, so every commit backfills the whole WAL. This is
     inherent total work, not a defect: with `wal_autocheckpoint=0` the commit falls
     to 58–117 ms but the WAL grows 109 → 691 MB across four flushes. `synchronous`
     is already NORMAL (the project DB manager's `_configure_connection`), so there
     is no fsync knob left either. On the real 3 GB index this backfill is plausibly
     the single largest persist component — the only lever would be checkpoint
     scheduling/threshold tuning, a durability-adjacent policy decision for the
     project-DB owners, not a store-local fix.
   - **Guards, ordering, and mtime are all ≤5%** — the previously suspected
     `_select_known_node_ids`/`_drop_dangling_edge_rows` chunked SELECTs (~7 ms per
     20-file flush) and `_order_rows_parents_first` (~1.3 ms) are innocent.

   Per the gate in this document: attribution first, and the attribution says STOP —
   remaining persist cost is proportional to actual bytes written into SQLite plus
   WAL backfill. No `benchmark_persist_batch.py`-motivated production change was
   made in this step.

### `Python::detach` audit — tokenizer, embeddings, graph_algo, trace_scanner (2026-11-18, landed)

Backlog item 2 executed against the four remaining CPU-bound modules in
`rust/crates/python-bindings`. House rules applied: release the interpreter only for
pure-Rust spans plausibly multi-millisecond on realistic inputs; keep all Python-object
access (argument extraction, `PyDict`/`&str`/`PyRef` borrows, result conversion) outside
the detached region; `PyErr` is `!Ungil`, so fallible batch paths use the crate's
wrapper/`*_impl` idiom with errors crossing the detach as `String`s (same as
`similarity.rs`/`ccg_extract.rs`); no side effect can be replayed after an FFI failure
(the one I/O-bearing module, trace_scanner, opens files inside the detach and returns
partial results — identical to its previous semantics).

| Module | Function | Profile | Duration class | Action |
|---|---|---|---|---|
| tokenizer | `BpeTokenizer::new` | Pure Rust (owned ranks table → `AHashMap`) | Multi-ms with real ranks (cl100k ≈ 100k entries) | **Detached** (owned inputs; `parking_lot` cache write is a short Rust lock) |
| tokenizer | `BpeTokenizer.count_tokens` | Pure Rust (BPE encode) | µs–sub-ms per single chat/chunk text | No detach (trivial-span rule) |
| tokenizer | `BpeTokenizer.count_tokens_batch` | Pure Rust + rayon | Plausibly multi-ms (batch hot path) | **Detached** (owned `Vec<String>`, `Arc` ranks) |
| tokenizer | `BpeTokenizer.encode` | Pure Rust | µs–sub-ms single text | No detach |
| tokenizer | `count_tokens_fast` | Pure Rust (linear heuristic) | µs | No detach |
| tokenizer | `count_tokens_fast_batch` | Pure Rust + rayon | Plausibly multi-ms (batch) | **Detached** |
| embeddings | `quantize_embedding` | Pure Rust O(d) | µs | No detach |
| embeddings | `dequantize_embedding` | Pure Rust O(d) | µs | No detach |
| embeddings | `quantized_cosine_similarity` | Pure Rust O(d) | µs | No detach |
| embeddings | `batch_quantize_embeddings` | Pure Rust + rayon (>100) | Plausibly multi-ms | **Detached** |
| embeddings | `batch_quantized_cosine_similarity` | Pure Rust O(n·d), rayon >100 | Plausibly multi-ms | **Detached** (query int8 data cloned out of the `PyRef` borrow) |
| embeddings | `matmul_vector` | Pure Rust O(rows·d) | Sub-ms realistic dims | No detach |
| embeddings | `batch_matmul_vector` | Pure Rust O(n·rows·d), rayon >10 | Plausibly multi-ms | **Detached**; `FlatMatrix::from_nested` retyped to `Result<_, String>` so no `PyErr` crosses |
| embeddings | `random_projection` | Pure Rust O(n·target·d) + LCG projection build | Multi-ms on realistic RAG shapes (500×1536→256 ≈ 200M FLOP) | **Detached** (wrapper/`random_projection_impl` split) |
| embeddings | `pairwise_distances` | Pure Rust O(n²·d), serial | Multi-ms for n≳100 | **Detached** (wrapper/`_impl` split) |
| embeddings | `knn_graph` | Pure Rust O(n²·d) + rayon | Plausibly multi-ms | **Detached** (wrapper/`_impl` split) |
| graph_algo | `pagerank` | GIL-bound dict extraction + pure-Rust power iteration O(iter·E) | Compute plausibly multi-ms | **Split**: extraction attached, iteration **detached** (`pagerank_compute`) |
| graph_algo | `weighted_pagerank` | Extraction + O(iter·E) fixed sweeps | Compute plausibly multi-ms (100 sweeps) | **Split**, compute **detached** |
| graph_algo | `betweenness_centrality` | Extraction + Brandes O(V·E) | Firmly multi-ms (most expensive here) | **Split**, Brandes sweep **detached** |
| graph_algo | `connected_components` | Extraction-dominated; single O(V+E·α) union-find pass | Sub-ms realistic module graphs | No detach (trivial compute; restructure not warranted) |
| graph_algo | `detect_cycles` | Extraction-dominated; single O(V+E) DFS | Sub-ms realistic module graphs | No detach |
| trace_scanner | `scan_usage_file` | Rust-only I/O + per-line serde_json parse | Firmly multi-ms on real usage logs (multi-MB) | **Detached** (owned path `String`; file I/O inside detach is safe — no Python API, no replayable side effect) |
| trace_scanner | `scan_usage_files` | Multi-file I/O + parse | Multi-ms | **Detached** (one FFI crossing per batch; scoring/sort tail unified into `finalize_session_stats`) |

Parity evidence: full `cargo test` workspace green (victor_native 198/198, zero warnings;
interpreter-dependent tokenizer tests now call `Python::initialize()` first, matching the
`thinking.rs` pattern); `pytest tests/unit/native/ -q` green **345 passed, 11 skipped —
identical counts with the pre-change installed extension (0.30.0) and the rebuilt
extension (0.31.1)**, including `test_tokenizer_parity.py` and
`test_context_fitter_parity.py`. `tests/unit/native/test_graph_algo.py` exercises the
split graph functions against the rebuilt `.so`. Known non-ownership cost that remains:
nested-`Vec` conversion at FFI extraction stays attached — that is backlog item 3
(contiguous-buffer inputs), out of scope for this audit.

### Parity/benchmark inventory (FOU-C3, 2026-11-19, landed)

Backlog item 3's parity half: every exported operation of `victor_native`
(`rust/crates/python-bindings/src/lib.rs`: 104 functions + 26 classes) mapped to its Python
reference/fallback and its differential-parity coverage. The `native-parity` CI job runs
`pytest tests/unit/native -m native_parity` after building the wheel, so every suite marked
`native_parity` below is a merge gate on a freshly built extension; the same suites run (and
self-skip when the wheel is absent) in ordinary unit runs.

Coverage categories: **differential** = the same inputs asserted through both dispatch paths;
**dual-backend** = protocol/backend-parametrized tests that run the same assertions against the
Python and Rust implementations; **wrapper** = property tests against whichever path dispatch
selects (no cross-path comparison).

| Rust module | Exported operations | Python reference / fallback | Parity coverage |
|---|---|---|---|
| protocol (FEP-0010) | `PyRole`, `PyToolCall`, `PyUsage`, `PyMessage`, `PyStreamChunk`, `echo_message`, `echo_stream_chunk` | `victor_protocol` shared crate shapes | Differential at the FFI: `test_protocol_ffi.py`, `test_native_protocols.py` |
| tokenizer | `BpeTokenizer`, `count_tokens_fast`, `count_tokens_fast_batch` | tiktoken encoder (+ word estimate fallback) | Differential: `test_tokenizer_parity.py` (±1 documented divergence pinned) |
| context_fitter | `fit_context`, `truncate_message`, `batch_score_messages`, `MessageSlot`, `FitResult` | `processing/native/context_fitter.py` | Differential: `test_context_fitter_parity.py` (`batch_score_messages`: wrapper tests only) |
| ccg_extract | `extract_statements_batch`, `build_ccg_batch`, `StatementRecord`, `EdgeRecord`, `CcgFileResult` | `core/indexing/ccg_builder.py` walk (oracle) | Differential over repo/golden fixtures: `test_ccg_extract.py` + control-dependence suites (native-parity job, second command) |
| plugin_ccg | `extract_plugin_statements_batch`, `PluginStatementRecord` | victor-coding builder tables (goldens) | Differential (golden corpus, python + typescript): per the Option-B record above |
| secrets | `scan_secrets`, `has_secrets`, `get_secret_types`, `mask_secrets`, `list_secret_patterns`, `scan_secrets_summary`, `SecretMatch` | `security/safety/secrets.py` `SecretScanner` | **NEW (FOU-C3)** `test_secrets_parity.py`: full agreement corpus + benign corpus through both paths, `has_secrets`/mask-leak properties; plus the FOU-A2b scanner suite. The FOU-C3 registry-divergence ledger was FIXED in #1975: both registries now detect the union (Rust gained bearer/PGP/bare-40-char-AWS; Python gained the vendor families and registered the refresh token + real-shape Slack pattern), pinned as exact per-engine label sets in `TestRegistryParity`; dispatch-path assertions skip with reason on wheels predating the port (behavioral probe). Mask STYLE differs by design (Rust partial-masks, Python `[REDACTED]`) — the contract is leak-freedom |
| graph_algo | `pagerank`, `weighted_pagerank`, `betweenness_centrality`, `connected_components`, `detect_cycles` | `victor/native/python/graph_algo.py` | **NEW (FOU-C3)** `test_graph_algo_parity.py`: deterministic graphs through both sides; PageRank within 1e-6 (never bit-exact — ahash iteration order varies per-process), Brandes within 1e-9, structural results exact |
| trace_scanner | `scan_usage_file`, `scan_usage_files`, `SessionStats` | `_scan_usage_file_python` | **NEW (FOU-C3)** `test_trace_scanner_parity.py`: plain/gzip/malformed/filtering corpora both paths; the same-session-across-files divergence was FIXED in #1975 — the Python fallback now shares one session accumulator across the file list (native cross-file aggregation semantics) and the duplicate-session case is asserted as strict equality |
| embeddings | `quantize_embedding`, `batch_quantize_embeddings`, `dequantize_embedding`, `quantized_cosine_similarity`, `batch_quantized_cosine_similarity`, `matmul_vector`, `batch_matmul_vector`, `random_projection`, `pairwise_distances`, `knn_graph`, `QuantizedEmbedding` | none in-repo (native-only surface) — spec reference implemented in NumPy f32 in the test | **NEW (FOU-C3)** `test_embeddings_parity.py`: quantization bit-parity (incl. half-away-from-zero rounding), integer-space cosine, LCG projection reproduced bit-exactly (matmul tolerance ~1e-4), pairwise/KNN vs NumPy reference |
| dedup | `normalize_block`, `rolling_hash_blocks`, `find_duplicate_blocks` | `processing/native/deduplication.py` | **NEW (FOU-C3)** `test_text_pipeline_parity.py`: normalization bytes, block splits, duplicate flags/indices identical; digest VALUES pinned as divergent by design (xxHash3 vs MD5) |
| hashing | `compute_signature`, `compute_batch_signatures`, `signature_similarity` | `processing/native/deduplication.py` | **NEW (FOU-C3)** same file: determinism, key-order independence, length/format, similarity extremes; digests divergent by design (as above) |
| classifier | `classify_task`, `has_action_keywords`, `has_analysis_keywords`, `has_generation_keywords`, `has_negation`, `find_all_keywords`, `TaskClassifier`, `ClassificationResult`, `TaskType` | keyword fallbacks in `processing/native/classifier.py`; `classify_task` falls back to `victor.agent.unified_classifier` | **NEW (FOU-C3)** keyword predicates + `find_all_keywords` differential. `classify_task` deliberately has NO parity contract: the fallback is a different weighted-pattern algorithm by design — a future decision item, not a test target |
| json_repair | `repair_json`, `extract_json_objects`, `coerce_string_type` | `processing/native/json_repair.py` | **NEW (FOU-C3)** `test_text_pipeline_parity.py`: repair/extract corpora identical; both formerly-pinned divergences FIXED in #1960/#1962 — `repair_json`'s Python fallback now ports the Rust quote/escape-aware state machine (byte-parity proven), and `coerce_string_type` runs one shared coercion pipeline so dispatch never changes semantics |
| similarity | `cosine_similarity`, `batch_cosine_similarity`, `similarity_matrix`, `top_k_similar`, `batch_normalize_vectors`, `batch_cosine_similarity_normalized`, `top_k_similar_normalized`, `EmbeddingIndex` | `processing/native/similarity.py` (NumPy) + `victor/native/python/similarity.py` | Dual-backend protocol tests: `test_native.py`, `test_python_fallbacks.py`. `EmbeddingIndex` (native-only class): uncovered |
| ast_indexer | `is_stdlib_module`, `batch_is_stdlib_modules`, `filter_stdlib_imports`, `extract_identifiers`, `extract_identifiers_with_positions` | `victor/native/python/ast_indexer.py` | Dual-backend: `test_ast_indexer.py`, `test_python_fallbacks.py` |
| arg_normalizer | `coerce_string_type`, `batch_coerce_string_types`, `normalize_json_string`, `batch_normalize_json_strings`, `repair_quotes`, `is_valid_json`, `get_json_type` | `victor/native/python/arg_normalizer.py` | Dual-backend: `test_python_fallbacks.py`; scalar `coerce_string_type` parity asserted in `test_text_pipeline_parity.py` (divergence fixed #1962) |
| chunking | `chunk_by_sentences`, `chunk_by_chars`, `chunk_by_paragraphs`, `detect_doc_type`, `count_tokens_approx`, `count_lines`, `find_line_boundaries`, `line_at_offset`, `chunk_with_overlap`, `ChunkInfoRust` | `processing/native/chunking.py` + `victor/native/python/chunker.py` | Dual-backend protocol tests + wrapper property tests + `test_chunking_security.py`; no per-op differential corpus |
| streaming_filter | `strip_thinking_tokens`, `contains_thinking_tokens`, `find_thinking_tokens`, `extract_thinking_content`, `StreamingFilter`, `StreamingChunkResult` | `processing/native/streaming.py` | Wrapper tests: `test_processing_native_streaming.py`; no differential corpus |
| thinking | `ThinkingDetector`, `PatternAnalysis`, `detect_circular_phrases`, `count_circular_patterns`, `find_circular_patterns` | `victor/agent/thinking_detector.py` | Wrapper tests only; no differential corpus |
| pattern_match | `contains_any_pattern`, `find_all_patterns`, `count_pattern_matches`, `get_matched_pattern_indices`, `batch_contains_any`, `weighted_pattern_score`, `PatternMatcher`, `PatternMatch` | consumer-side Python implementations (classification, enrichment) | Wrapper/consumer tests only; no differential corpus |
| extractor | `extract_file_path`, `extract_code_blocks`, `extract_shell_commands`, `extract_tool_call`, `batch_extract_file_paths` | `processing/native/tool_extraction.py` | Wrapper + consumer tests; no differential corpus |
| sanitizer | `sanitize_response`, `is_garbage_content`, `detect_leakage_patterns`, `strip_markup`, `validate_tool_name` | `victor/agent/response_sanitizer.py` | Wrapper tests: `test_response_sanitizer.py`; no differential corpus |
| yaml_loader | `parse_yaml`, `parse_yaml_with_env`, `parse_yaml_file`, `parse_yaml_file_with_env`, `validate_yaml`, `extract_workflow_names` | `processing/native/accelerator.py` (PyYAML) | Wrapper tests only; no differential corpus |

Count: **~110 exported operations; 41 newly under differential parity (FOU-C3)**, 23 already
differential (protocol/ccg/tokenizer/context-fitter/secret-scanner), ~20 dual-backend, the rest
wrapper-level. **Remaining uncovered differentials for a future increment:** `EmbeddingIndex`,
`batch_score_messages` (dedicated corpus), streaming_filter, thinking, pattern_match, extractor,
sanitizer, yaml_loader, and the remaining chunking/arg_normalizer batch ops. **Findings recorded
by the FOU-C3 suites**: Python `repair_json` string-content corruption (FIXED #1960);
`coerce_string_type` install-shape split (FIXED #1962); `scan_usage_files` same-session
aggregation vs concatenation (FIXED #1975 — Python fallback adopts the native cross-file
aggregation); secret-registry divergences on both sides (FIXED #1975 — both registries detect
the union: Rust gained bearer/PGP/bare-40-char-AWS; Python gained anthropic/openai/twilio/
sendgrid/slack-webhook/heroku/discord, the registered refresh token, and the real-shape Slack
pattern). Remaining BY DESIGN: xxHash3-vs-MD5 digest divergence (the structure-level contract —
normalized bytes, duplicate markers and indices are identical; only digest values differ).

### Harvest environment constraint (2026-10-25, landed)

The python tier-3 harvest ran **2,980 s** (half the full-index wall clock):
scip-python's embedded Pyright resolves imports against the RUNNING
interpreter's environment — 17 GB of site-packages on a shared conda env,
~130 GB of reads with WAL writeback thrashing the page cache.

victor now derives the environment from the CODEBASE and hands
scip-python `--environment` (schema verified against the installed
bundle: `[{name, version, files}]` — providing it skips Pyright's
whole-environment enumeration): an AST import scan (the tree-parse basis
of the earlier passes; ~7 s, 138 third-party roots) mapped to installed
distributions plus dependency closure (264 of 762), case-canonicalized,
round-trip validated; any failure declines to the unconstrained run.
**Reflection is part of the contract**: the builder logs direct-vs-closure
provenance and unmapped roots; every fresh constrained dump is compared
against the previous baseline dump (external packages via
`Index.external_symbols` + occurrence counts), and a real coverage
regression triggers ONE unconstrained re-dump — never a silent
degradation.

Measured: harvest **2,980 s → ~8.5 min (~5.8×)**; dump size within
0.002%; occurrences 2,885,017 → 2,885,269 (+0.009%); zero packages lost
vs baseline. Operational notes: the production node heap (16 GB) is
still required — the constraint shrinks the search universe, not
Pyright's working set; and the repo carries a stale `build/` tree that
scip-python indexes as project files (6,789 total) — deletion candidate.
Other languages surveyed, NOT applicable: scip-typescript is
tsconfig-bounded, scip-clang compilation-database-bounded, rust-analyzer
cargo-workspace-bounded, java maven/gradle — the interpreter-mediated
environment walk is python-specific.

## Primary references

- [PyO3 performance guidance](https://pyo3.rs/main/performance.html)
- [PyO3 parallelism and interpreter detachment](https://pyo3.rs/main/parallelism)
- [maturin distribution and cross-compilation](https://www.maturin.rs/distribution)
- [maturin platform support](https://www.maturin.rs/platform_support)
- [CPython Limited API and Stable ABI](https://docs.python.org/3/c-api/stable.html)
- [Cython and the GIL](https://docs.cython.org/en/latest/src/userguide/nogil.html)
- [Cython compilation model](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html)
