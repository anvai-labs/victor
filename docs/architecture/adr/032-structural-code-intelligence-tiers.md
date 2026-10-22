# ADR-032: Structural Code Intelligence — Scope-Graph Binding and Precise-Index Harvesting

## Metadata

- **Status**: Accepted (Phase 1 / tier 2.5 implemented; tiers 3+ planned)
- **Date**: 2026-10-19 (rev 1.1: 2026-10-20; rev 1.2: 2026-10-20)
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 014 (shared codegraph chunker package), 015 (Victor Core adopts
  victor-codegraph), 026 (durable code memory GA — one `oid`, correlated graph+vector)
- **Scope**: Graph-RAG indexing pipeline and victor-codegraph integration only. No
  framework public-API changes (no FEP required); no runtime dependency on language
  servers at query time; victor-codegraph keeps its independent release train.

## Context

Victor's structural memory is the Graph-RAG stack: `victor-codegraph` tree-sitter
chunking (tier 0/1), the SQLite/ProximaDB graph stores with CONTAINS/IMPORTS/CALLS
edges and per-language import resolvers (tier 2 heuristic), CCG statement graphs
(CFG/CDG/DDG), FTS5 + vector hybrid retrieval with graph-expansion reranking, and
content-hash staleness for incremental re-indexing. This is the right substrate —
but two precision gaps remain between it and what IDEs and code-intelligence
platforms provide:

1. **Name binding is heuristic.** Tree-sitter gives structure (CST) but zero binding
   semantics. Which `impl` block a method belongs to, which `use` a path resolves
   through, and which local a reference names are resolved today by position and
   textual import matching — correct most of the time, unprovable any of the time.
   The parent-FK incident fixed in #1397 was exactly this class: emitter-order
   assumptions standing in for a resolution model.
2. **No compiler-accurate cross-references.** Callers/implementers/trait-resolution
   edges — the relationships an LLM cannot cheaply infer from a context window —
   exist only in approximated form. IDEs get them from real compiler front-ends
   (rust-analyzer is built from rustc's own libraries over the salsa incremental
   query framework; clangd runs Clang's frontend; gopls drives the Go type checker),
   and code platforms get them from offline index dumps: GitHub's stack graphs
   (tree-sitter + graph-construction DSL solving name binding as graph reachability,
   per-file incremental, no build step) and SCIP (Sourcegraph's successor to LSIF;
   `scip-python` drives pyright, `scip-go` drives gopls, etc.).

The cost argument is now measured, not assumed: coding agents burn their budget on
*finding* code rather than changing it — a 30B OpenHands agent averages 23 rounds and
631K tokens per resolved SWE-Bench Verified issue, much of it on grep/glob/view
during repository exploration ([CodeGrep, arXiv:2608.05886]). Every query a graph
edge answers is tokens never spent; structure narrows, the model concludes.

Evidence base (arxiv corpus on dataserver3, `/data/arxive`):

| Paper | Relevance |
|---|---|
| LLMs and Language Server Protocol: a match made in context (arXiv:2609.03086) | LSP as the precision substrate for LLM tooling; LLM output verified by static analysis rather than trusted. |
| IncSFS: incremental full-sparse flow-sensitive pointer analysis (arXiv:2608.24391) | Incremental recomputation on evolving code is what makes deep static analysis affordable — the same property content-hash staleness gives our indexer. |
| Dissecting Software Graphs (arXiv:2608.12859) | Static call graph as the shared backbone onto which dynamic/derived views project. |
| Graph-grounded software knowledge for reliable LLM programs (arXiv:2609.01095) | Hybrid retrieval over a heterogeneous software knowledge graph grounds generation; a request alone never specifies APIs/dependencies/conventions. |
| MAP4CS (arXiv:2610.11727) | Code-retrieever adaptation to evolving repositories; corpus noise motivates stable structural chunking over churny text windows. |
| When Does the Dependency Graph Help Code-to-Architecture Mapping? (arXiv:2610.19769) | Typed dependency graphs as the substrate for architectural-level reasoning (semi-supervised propagation over the graph). |
| STRATA (arXiv:2610.19386), StateHarness (arXiv:2610.17616) | Structured, typed graphs as durable agent memory beat undifferentiated embedding pools; agents need an evolving project-state view maintained outside model parameters. |
| WEAVE (arXiv:2610.22039) | Dependency-structured tasks as the evaluation substrate for agentic code work. |
| RepoReuse / Do Coding Agents Reuse Existing Code? (arXiv:2609.35357) | Reuse auditing requires AST-based dependency graphs; duplicated implementations accumulate unsupervised. |
| NexusRAG (arXiv:2609.37661) | Graph-RAG propagation must go beyond query similarity to reach bridging evidence — validates graph-expansion reranking over pure vector recall. |

## Decision

Close the two gaps with two additional precision tiers behind the existing graph
surface, preserving the repo's fallback discipline (every tier degrades to the one
below; nothing requires a build system or a running toolchain).

### Tier 2.5 — scope-graph name binding in victor-codegraph

Implement a stack-graphs-style binding pass over the CSTs victor-codegraph already
produces:

- Per-language scope rules emit def/ref vertices and scope edges; reference
  resolution is a reachability query over that graph (the stack-graphs model),
  executed per file with only the file's imports as cross-file inputs — preserving
  per-file incremental re-indexing.
- Output lands in the existing graph as resolved REFERENCES edges plus corrected
  parentage for lexical nesting, replacing positional heuristics where resolution
  succeeds; heuristics remain the fallback where a language's rules are not yet
  written or resolution is ambiguous.
- ~~Rust-side (existing workspace, PyO3, `_NATIVE_AVAILABLE` fallback)~~ — **rev 1.1
  correction:** victor-codegraph is pure Python (the monorepo's Rust workspace is
  victor-core's, not the codegraph package's). Tier 2.5 shipped as pure Python;
  native acceleration remains a future measured decision under the repo's
  native-acceleration strategy.
- **Build-vs-adopt — SETTLED (rev 1.1):** `github/stack-graphs` is archived
  (RUSTSEC-2026-0303, Sep 2026 — "no published successor"; companion
  RUSTSEC-2026-0302 unsoundness advisory on its C API exports). Adopting an
  unmaintained, soundness-flagged crate is disqualified for a production
  framework. Tier 2.5 is the hand-rolled subset this ADR's fallback branch
  anticipated.

**Rev 1.1 — tier 2.5 as implemented (#1404, victor-codegraph 0.11.0; #1406 core
integration):**

- A language-agnostic two-pass collector (`scope_binding.py`) plus a stdlib-`ast`
  adapter (`python_references.py`, the reference implementation): occurrence
  extraction with innermost-owner attribution, span-keyed exclusions for call
  callees and member positions, field-directed shadow collection, per-language
  noise roots, deterministic first-use dedup per (owner, name), unresolved cap
  64/owner. `CodeRelationType.REFERENCES` (previously dead) is the carrier;
  resolution rides the existing two-stage resolver (`scope_resolver` →
  `repository_resolver` via exact import bindings), ambiguity retained at 0.5.
- **Language membership is honest capability, not grammar availability** — every
  grammar is probed against a three-fact contract (one resolved in-file edge, one
  retained-unresolved edge, shadowed locals producing no edge). Verified and
  enabled at 1.1: python, javascript, typescript, tsx, rust, go, java, c, bash.
  **Rev 1.2 (victor-codegraph 0.12.0, #1408): the eight pending grammars
  joined — cpp, csharp, php, swift, scala, kotlin, ruby, lua — for seventeen
  reference-emitting languages; sql remains the sole, asserted, vacuous
  exclusion.** kotlin/ruby extraction unlocked en route (ruby `method` nodes
  and kotlin/swift `simple_identifier` names now mint function symbols —
  a default-path change, hence the minor version). An emptiness ratchet
  test pins every exclusion: adding a language is a deliberate verified act.
- Opt-in end to end: `emit_references=False` default on `parse`/`parse_repo`
  (package output byte-identical until asked); victor-core gates on
  `GraphIndexConfig.enable_scope_binding` (default OFF, settings-cascaded,
  capability-probed against codegraph ≥ 0.11.0 with an explicit downgrade log).
- Known v1 limits: module-level uses skipped (no owner in the id-map rewrite);
  declaration-subtree over-shadowing errs toward suppressed edges; ON→OFF flag
  flips leave stale REFERENCES edges until a force rebuild.

### Tier 3 — precise-index harvesting (SCIP/LSP), optional and lazy

Compiler-accurate cross-references when — and only when — the toolchain exists:

- Pluggable per-language precise providers harvest from offline index dumps:
  `scip-python` (pyright), `scip-go` (gopls), rust-analyzer-derived SCIP for Rust,
  tsserver for TypeScript. Harvest is invoked by explicit configuration or
  capability detection, never by default.
- Harvested edges/nodes carry provenance and **override** heuristic edges;
  heuristic edges never delete precise ones. No language server runs in the query
  hot path — index once, serve graph lookups (the SCIP model).
- Failure of a provider is logged and skipped; the tier 2/2.5 graph remains fully
  usable (fallback preservation).

### Provenance model

Every binding-bearing edge carries a precision level so retrieval can prefer
precise edges, the agent layer knows what is provable vs inferred, and
reconciliation is deterministic (compiler > scope-resolved > heuristic). This
extends the existing edge metadata convention; no schema break.

**Rev 1.1 vocabulary decision:** one concept, one vocabulary — the values are
victor-codegraph's resolver strings verbatim (`scope_resolver`,
`repository_resolver`, later SCIP harvesting's `compiler`), carried in
`GraphEdge.metadata` as `{"provenance": …, "confidence": …, "use": …}` via
`ProjectedRelationEdge` — not a dasherized variant. `REFERENCED_BY` is computed
downstream, never emitted (the `CALLED_BY` precedent). Heuristic passes never
write REFERENCES (the edge type is greenfield), which makes the
last-writer-wins `(src, dst, type)` semantics of every store backend safe.

### Retrieval and memory integration

- Hybrid retrieval (FTS5 + vector + graph expansion) is unchanged in shape; tiers
  2.5/3 upgrade the expansion edges the reranker walks. Embeddings remain derived,
  rebuildable state keyed to the stable `oid` identity (ADR-026).
- The durable structural memory thesis is STRATA/StateHarness-shaped: the graph is
  the evolving project state outside model parameters; the LLM stays stateless and
  terminal.

## Beyond tier 3 — outlook (rev 1.1, not yet committed work)

The ladder above tier 3, in the order the value compounds. Each tier is a
separate decision with its own evidence bar; none is authorized by this ADR.

- **Tier 3.5 — type-directed queries.** Keep type/receiver payloads in harvested
  relations (the `use`/qualifier/arity schema already rides tier-2.5 edges) so
  SCIP enrichment answers trait/dispatch and caller-of-type-signature questions
  without re-harvesting. Nearly free if tier 3 keeps its payloads.
- **Tier 4a — precise-CCG fusion.** The CCG's CFG/CDG/DDG edges sharpen from
  heuristic to compiler-informed once tier 3 supplies real call/receiver types,
  plus a small traversal query language (SQLite recursive CTEs already express
  reachability). Victor's compounding payoff: the CCG already exists.
- **Tier 4b — interprocedural summaries.** Datalog-style fixpoint over the graph
  for reachability/taint/impact sets — upgrades the shipped impact tool from
  name-index adjacency to computed blast radius. Must be incremental
  (edit → seconds) or it dies in practice; IncSFS (arXiv:2608.24391) is the
  incremental discipline to follow.
- **Tier 5 — dynamic grounding.** Project test/coverage telemetry onto the static
  graph as edge weights (the Dissecting Software Graphs pattern,
  arXiv:2608.12859): retrieval prefers executed paths, dead code becomes
  visible, and the graph is verified against reality. Highest evidence-per-hour
  for an *agentic* system — the agents already execute code — and consistent
  with the execution-program conclusion that only in-container verification
  yields ground truth.
- **Stays out:** CodeQL/Kythe-class semantic databases, symbolic execution,
  always-on LSP (the Alternatives below).

## Alternatives Considered

- **Always-on LSP integration at query time.** Rejected: couples retrieval latency
  and reliability to per-user toolchain state; SCIP-style offline dumps give the
  same precision with none of the runtime coupling.
- **Full semantic databases (CodeQL/Kythe-class).** Rejected: build-coupled,
  heavy, per-language toolchain commitment that cannot be a default for a
  framework that must index arbitrary projects with no build info.
- **LLM-extracted structure.** Rejected as a substrate (cost, nondeterminism,
  unverifiable) — this ADR's premise. LLM remains the consumer, not the indexer.
- **Adopting `tree-sitter-stack-graphs`.** Settled against by evidence at rev 1.1:
  archived upstream (RUSTSEC-2026-0303, no successor) with a companion
  unsoundness advisory (RUSTSEC-2026-0302).

## Flag graduation dossier (rev 1.2, 2026-10-20)

Evidence gathered for the `enable_scope_binding` default-ON decision. The
decision remained the maintainer's until rev 1.3.

**GRADUATED 2026-10-20 (rev 1.3): default-ON approved by the maintainer.**
`enable_scope_binding` now defaults to True across all six construction
sites; explicit False opts out; the capability probe keeps older
victor-codegraph installs safe (graceful degradation to off); an ON-to-OFF
flip requires a force rebuild to clear stale REFERENCES edges.

**Parse-level (victor monorepo, 4,847 files — 4,684 py / 97 ts / 34 rs / 26 sh,
55s `parse_repo(emit_references=True)`):**

- 104,841 REFERENCES emitted; 28,216 resolved (12,277 `scope_resolver` +
  15,939 `repository_resolver`), 73.1% retained for cross-file/external
  targets — an expected shape for import-heavy Python.
- Resolution rate **26.9%**, versus the heuristic CALLS baseline's **21.3%**
  on the same corpus: the binding tier resolves better than the existing
  call heuristic it would augment.
- Volume control confirmed: **1.14 refs/symbol** average (dedup + cap
  working as designed).
- Per-language resolution: python 27.4%, typescript 17.6%, rust 14.5%,
  bash 22.6%.

**Pipeline-level (103-file real-code fixture — victor graph_rag Python +
vscode-victor TypeScript — full `GraphIndexingPipeline` into SQLite,
flag OFF vs ON):**

- Identical node sets (1,453 both runs), **+675 REFERENCES edges**
  (562 `scope_resolver` + 113 `repository_resolver`) on a 2,496-edge
  baseline (+27% edge volume), zero errors, and 100% of persisted
  REFERENCES carrying provenance — the monotonicity contract holds in a
  real pipeline run, not just tests.
- Retrieval integration: multi-hop traversal from a REFERENCES-connected
  seed expands through CALLS + REFERENCES + CONTAINS mixed neighborhoods —
  the precision-first ordering has real edges to order.
- Wall-clock: dominated by per-process first-pipeline initialization
  (~12-13s regardless of flag); the binding run as second-in-process
  completes in 2.9s doing strictly more work than a baseline
  second-in-process run — **binding overhead is below measurement noise**
  at this scale.

**Second corpus (proximaDB, polyglot — completed in a 50-minute window):**
~35.9k REFERENCES at ~34.3% resolved, with every one of the fourteen
emitting grammars exercised on real code — go strongest (4,032/8,802,
45.8%), python 31.2%, typescript 28.2%, and the entire promoted long tail
resolving at 7-33% (swift 26%, kotlin 32%, scala 24%, csharp 46%,
lua 8%). No `rust` rows appeared in that first sweep — resolved on re-run as a
measurement artifact (the sweep process's source tree was removed mid-run,
degrading lazy grammar loads to silent fallback), not a codegraph defect. A
dedicated rust-only re-sweep from a stable tree (2,788 files: 2,772 success
/ 16 partial, 507s) produced **220,056 REFERENCES at 12.6% resolved
(24,993 scope_resolver + 2,755 repository_resolver) against the CALLS
baseline's 5.7%** — the largest relative resolution gain of any language
measured (2.2x). The graduation dossier now has no open questions.

## Tier 3 implementation + real-contact dossier (rev 1.4, 2026-10-21)

Tier 3 shipped in 0.14.0 (#1412) validated against synthetic shapes only.
First real contact with the actual toolchain (2026-10-21; #1432 + #1436,
released 0.14.1) re-verified every integration assumption against a live
scip-python 0.6.6 and fixed four independent breaks — the shipped harvester
could not have produced a single edge on real data:

1. **Reader**: the toolchain is npm-only (`npm install -g
   @sourcegraph/scip-python`); no pip package provides the schema (PyPI's
   `scip` is unrelated cytometry tooling). The protobuf bindings are now
   vendored (`victor/core/graph_rag/_scip/`, scip-code/scip proto,
   Apache-2.0). An absent protobuf runtime degrades to `no-reader`
   (ImportError is reader absence, not failure).
2. **Symbol grammar**: 0.6.x emits package-qualified descriptors —
   `scip-python python <pkg> <ver> \`dotted.module\`/Class#method().` with
   backticked dotted module paths, `#` class members, `()` callables —
   not the `python <file>\`<path>` backtick form the first parser
   assumed. Measured on a real dump: the old parser joined 0 of 900
   occurrences.
3. **Range encodings**: the typed ranges are now flat oneof members
   (`single_line_range{line,start_character}` /
   `multi_line_range{start_line,start_character}`); the old nested-Position
   reader would have crashed on them. All three encodings read correctly,
   with proto3 presence via `HasField` (unset message fields return default
   instances, not None).
4. **Join semantics**: the join is metadata-based (unique (file, name)
   match with scope agreement via parent_id container walks — GraphNode
   persists no `scope_chain` field — and the scip-kind filter, which is
   dormant with 0.6.x because it emits `UnspecifiedKind` on every symbol
   information). Non-symbol descriptors (module inits, parameters, locals,
   attribute fields, module-level assignments) and out-of-corpus references
   (builtins/stdlib/site-packages: defined nowhere in the index) are
   uncounted — never join-failure noise. The document bucket is the file
   binding (symbols carry no file); the path-drift fallback is
   segment-aligned so `tests/m.py` can never bind `m.py` nodes.

**Toolchain operational facts**: scip-python fatals on a directory without
package metadata (run against a project root with `pyproject.toml`); a dump
is a full pyright pass (the configurable `precise_index_timeout_seconds`
defaults to 1200s; timeouts are labeled distinctly from toolchain
absence); kind agreement rides `document.symbols` when a producer
populates it.

**Parse-level (real dumps, fixed grammar):**

- Slice (victor `graph_rag`, 3 files / 30 symbol nodes; 96.6 KB dump,
  ~13 s): **26 compiler REFERENCES edges, 90% join rate (26/29)** — 0%
  before the grammar fix.
- **Full monorepo** (6,707 python files — the dump is single-language by
  design; the repo's TypeScript/Rust await their own providers, which is
  the rev 1.5 contract's point): 336 MB SCIP dump in ~8m20s under a 16 GB
  Node heap (the default ~4 GB heap aborts — V8 OOM core dump — around
  2,200 files); protobuf parse 1.0 s; 2,873,127 occurrences (590,312
  definition-role, 121,420 joinable-symbol definitions); join 61.3 s
  against the 97,078 persisted symbol nodes:
  **143,558 compiler REFERENCES edges at a 76.9% join rate**
  (143,558/186,729 evidence pairs; 43,171 unjoined — definition-side
  failures are ~4.1k, dominated by same-name ambiguity that survives
  scope-chain disambiguation, plus 136 documents outside the codegraph
  symbol set; the remainder is ownerless references, counted honestly).

**Pipeline-level (full `GraphIndexingPipeline` over the monorepo, flag
OFF baseline, harvest decomposition):** flag-OFF run 260.8 s producing
97,078 symbol nodes / 187,469 edges — 28,226 tier-2.5 REFERENCES
(12,284 `scope_resolver` + 15,942 `repository_resolver`) against a
58,352 CALLS heuristic baseline. Flag-ON = flag-OFF + dump (~500 s,
dominated by the pyright pass) + parse/join (~62 s): **+143,558
REFERENCES pairs of `compiler` provenance — 5.1× the tier-2.5 REFERENCES
volume, +77% total edge volume** (upsert semantics: pairs shared with
tier-2.5 rows are overwritten with compiler provenance, so the count is
pre-reconciliation volume, not net-new rows).

**Owner attribution (v1, documented limit):** positional — the greatest
definition offset at or before the reference. Correct for the common
top-level/method body case; approximate for references inside nested
functions defined later in an enclosing scope. Tier 3.5's payload work
(SCIP `enclosing_range`) is the precise path.

**Graduation: NOT taken.** Unlike tier 2.5 (pure in-process, zero
toolchain dependency, overhead below noise), tier 3 shells out to an
external npm toolchain and pays a minutes-scale full-repo pyright pass.
The recommendation is to **stay opt-in** (`enable_precise_index`, default
OFF): projects with the toolchain and the patience get compiler-proven
REFERENCES; everyone else keeps the default tier-2.5 experience with one
explicit skip line. The maintainer may revisit if incremental SCIP dumps
or a faster python indexer change the cost shape.

## Tier 3 provider contract (rev 1.5, 2026-10-21 — design, pre-implementation)

Tier 3's toolchain reality is irreducibly polyglot: every precise indexer is
built on its language's reference toolchain and drags that runtime along
(scip-python → Node/Pyright, scip-go → Go/gopls binary, scip-java → JVM,
`cargo scip` → rust-analyzer, scip-csharp → .NET/Roslyn; a future in-process
Jedi or `ty`-based provider would carry none). The harvester therefore
standardizes a **provider contract** so runtimes are a provider detail, not
an architecture. Three lessons from first real contact shape it:

1. **Symbol grammars are PRODUCER properties, not language properties**
   (scip-python 0.6.x's package-qualified descriptors ≠ the documented
   backtick form). Grammar parsing belongs to the provider that reads the
   dump, never to the join.
2. **Foreign runtimes fail in their own ways** (V8 heap OOM at ~2,200 files
   under Node's default cap; JVM heap; missing-package-metadata fatals).
   Resource and environment policy is provider-declared, victor-bounded.
3. **Payloads must ride from day one** — tier 3.5 is only "nearly free" if
   the normalized schema carries `syntax_kind` / enclosing ranges / use
   qualifiers from the first provider, or every later tier re-dumps.

### The contract

```python
class PreciseIndexProvider(Protocol):
    name: str                       # "scip-python"
    languages: frozenset[str]       # {"python"} — the registry key
    execution: ExecutionClass       # SUBPROCESS | IN_PROCESS

    def detect(self, env) -> Availability:
        # Present/absent + version + declared env policy. Never installs,
        # never raises, never touches the network.

    def dump(self, root, artifact, budget) -> DumpOutcome:
        # Produce the artifact (SCIP bytes on disk today) offline and
        # bounded. The provider owns transport (subprocess argv/env/cwd,
        # or in-process); victor owns the clock and the crash fence.

    def read(self, artifact) -> ParsedIndex | None:
        # Artifact -> NORMALIZED occurrences: (file, offset, roles, name,
        # scope, kind, payloads...). Producer grammar parsed HERE.
```

**Invariants (what does not vary by provider):**

- **The artifact-on-disk is the seam.** Whatever the runtime, the join sees
  only the normalized in-memory index. Dumps are cacheable and auditable
  (SCIP `tool_info` records emitter+version); no language server, IPC
  channel, or FFI handle is ever alive in the query path. An LSP-session
  harvester is just another SUBPROCESS transport and stays offline.
- **Registry keyed by language, one flag.** `enable_precise_index` stays a
  single opt-in; a polyglot repo runs one dump per language with providers
  present, and detection reports per-language coverage honestly.
- **Uniform degradation taxonomy** — `toolchain-absent` |
  `dump-failed:{crash|timeout|metadata|exit}` | `reader-absent` | `ok` —
  every skip names its reason; unjoined/uncounted accounting rules
  (out-of-corpus, non-symbol descriptors) are join-level and shared.
- **Timeouts are caller-enforced.** In-process providers accept best-effort
  cancellation (or get wrapped in a worker process); subprocess providers
  get killed. No provider can hang the harvest.
- **Provenance stays `compiler`/1.0** — the tier's vocabulary, not the
  provider's. Reconciliation semantics are unchanged.
- **Normalization, not schema union.** A future LSIF adapter, Jedi
  in-process provider, or `ty` emitter implements `read` against the same
  normalized schema; the join and the store never learn which producer ran.

**Status: design recorded BEFORE multi-provider implementation** (one
provider exists today, scip-python, as module-shaped code in
`precise_index.py`). The refactor into the Protocol + registry is a
follow-up implementation PR against this contract; new providers land as
new implementations, not join rewrites.

## Validation

- **Differential parity tests**: heuristic vs scope-resolved edges over the repo
  corpus; every disagreement is a case study, not noise.
- **Golden-file binding tests** per language: resolution results pinned on
  representative constructs (impl/trait, closures, shadowing, re-exports) —
  landed in rev 1.1 as the three-fact contract (resolved / retained / shadowed)
  per verified language, plus the pending-language emptiness ratchet.
- **Provenance reconciliation tests**: precise-override and fallback-preservation
  behaviors pinned positively and negatively — rev 1.1 pins the monotonicity
  contract (every REFERENCES edge carries provenance; heuristic passes never
  write the type).
- **Retrieval quality**: existing Graph-RAG retrieval parity harness plus
  end-to-end agent benchmarks; success is measured recall/precision of retrieved
  context and token spend per resolved task, not index size.
- **Performance budget**: per-file incremental re-bind must stay within a small
  multiple of parse cost (IncSFS-style incremental discipline); no full-repo
  recompute on single-file edit.

## Consequences

- Per-language scope rules are the main new maintenance surface; the verified set
  (rev 1.1: nine languages) grows one deliberate verification at a time.
- Tier 3 availability is environment-dependent by design; documentation must state
  what is gained when a toolchain is present, and the default experience must never
  imply compiler precision without one.
- def/ref edges grow storage; references are deduped per (owner, name), capped per
  owner when unresolved, and excluded from per-node metadata mirroring.

## Revision History

| Rev | Date | Change |
|-----|------|--------|
| 1.0 | 2026-10-19 | Initial proposal: tier 2.5 scope-graph binding + tier 3 SCIP/LSP harvesting, provenance model, evidence base from the dataserver3 arxiv corpus. |
| 1.1 | 2026-10-20 | Phase 1 (tier 2.5) implemented: adopt-vs-build settled (RUSTSEC-2026-0303), pure-Python correction, verified nine-language honest-capability set, provenance vocabulary pinned to resolver strings, v1 limits; beyond-tier-3 outlook added (3.5 / 4a / 4b / 5). Status Proposed → Accepted. |
| 1.2 | 2026-10-20 | Language matrix completed (0.12.0: seventeen languages incl. kotlin/ruby extractor unlocks); flag-graduation dossier added — parse-level 26.9% resolution vs the 21.3% CALLS baseline, 1.14 refs/symbol; pipeline-level +27% edge volume with 100% provenance and overhead below measurement noise; proximaDB large-repo sweep noted as the open gap. Graduation decision remains the maintainer's. |
| 1.3 | 2026-10-20 | Flag graduated: `enable_scope_binding` default-ON (maintainer approval; six construction sites, opt-out preserved, force-rebuild caveat documented); CHANGELOG entry added. |
| 1.7 | 2026-10-22 | Hardening pass: harvest accounting decomposed and made honest (out-of-corpus-by-codegraph documents counted separately, never as failures; definitions/target/ownerless unjoined split — corpus edge counts byte-identical through the change, and the TS bucket-pollution hypothesis refuted by instrument: 0/89 misses, the real TS noise is ownerless import-level refs); dump artifacts cached per provider under .victor/precise (unchanged corpus: 642s → 54s, ~12x, identical edges; per-language invalidation; process-unique atomic writes after a raced fixed-tmp name persisted a truncated artifact that parses as a valid EMPTY index); provider-measured dump timeouts (1200/600/120s; 0 = provider default); rust-analyzer fixed-name artifact bug fixed (every real rust dump had failed); tier 3.5 retrieval-first: provenance-VALUE edge ranking, sort-before-cap on the lazy path, precision-ordered parallel scorer, query-cache per-edge metadata fidelity. |
| 1.6 | 2026-10-21 | Contract implemented: `precise_providers.py` (detect/dump/symbol_parts per producer), scip-typescript + rust-analyzer providers added alongside scip-python, per-language harvest accounting, kind map corrected to the vendored enum (the old 3/4/5 constants were from a different proto revision — live bug for kind-populating producers). Polyglot corpus (this monorepo): python 143,558 edges @ 76.9% (identical to the rev 1.4 run — the refactor is measurement-stable), TypeScript 1,701 @ 67.4%, Rust 952 @ 89.2%; 146,211 compiler REFERENCES total. |
| 1.5 | 2026-10-21 | Provider contract recorded pre-implementation: artifact-on-disk seam, producer-owned symbol grammars, SUBPROCESS/IN_PROCESS execution classes with caller-enforced budgets, uniform degradation taxonomy, tier-3.5 payloads in the normalized schema from day one; polyglot runtimes (Node/Go/JVM/cargo/.NET/in-process) become provider details. |
| 1.4 | 2026-10-21 | Tier 3 real-contact dossier: four integration breaks found and fixed against a live scip-python 0.6.6 (#1432/#1436, released 0.14.1) — vendored bindings, real package-qualified symbol grammar, flat range encodings, metadata join with honest accounting; slice-level 90% join rate; full-monorepo numbers recorded; graduation declined with rationale (external toolchain + minutes-scale dump). |
