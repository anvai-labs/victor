# Native Acceleration Strategy

**Status:** Adopted engineering guidance; implementation investigations remain incremental

**Last verified:** 2026-09-20

Victor should keep Python as the orchestration, integration, and I/O language and use the existing
Rust/PyO3 workspace as its default compiled acceleration path. Adding Cython, Numba, or a direct
CPython C/C++ extension would create another compiler, wheel, debugging, and parity surface. Such a
tool is an exception that requires measurements showing the existing choices cannot meet the need.

## Current codebase reality

- `rust/crates/python-bindings` builds `victor_native` with PyO3 0.29 and maturin. The workspace
  already covers tokenization, context fitting, similarity, embeddings, graph algorithms, parsing,
  filtering, indexing, and security scans.
- Python adapters under `victor/processing/native/` retain reference or fallback behavior. The
  required `native-parity` CI job builds a release wheel and checks native/fallback agreement for
  the tokenizer and context fitter; equivalent differential coverage is not yet universal.
- Rust release builds use optimization level 3, fat LTO, one codegen unit, and panic unwinding so a
  Rust panic does not abort the Python process.
- Release automation builds Linux x86-64 and ARM64 wheels for Python 3.12–3.13, plus the configured
  macOS and Windows targets. The matrix is not uniform: both macOS architectures and Windows x64
  currently build only for the active Python 3.12 and Windows ARM64 is absent. All packages
  require Python 3.12+; a supported interpreter does not imply a wheel for every platform.
- The FFI is already batch-oriented in several places, but the current source scan finds
  `Python::detach` concentrated in similarity operations. Several embedding APIs still accept
  nested `Vec<Vec<f32>>`, which copies Python-owned data at the boundary. These are higher-value
  investigations than adding another native language.

## Required decision sequence

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
3. Prototype contiguous-buffer inputs and persistent native indexes for embedding/similarity paths;
   compare them with current nested-vector conversion and the existing NumPy fallback.
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

## Primary references

- [PyO3 performance guidance](https://pyo3.rs/main/performance.html)
- [PyO3 parallelism and interpreter detachment](https://pyo3.rs/main/parallelism)
- [maturin distribution and cross-compilation](https://www.maturin.rs/distribution)
- [maturin platform support](https://www.maturin.rs/platform_support)
- [CPython Limited API and Stable ABI](https://docs.python.org/3/c-api/stable.html)
- [Cython and the GIL](https://docs.cython.org/en/latest/src/userguide/nogil.html)
- [Cython compilation model](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html)
