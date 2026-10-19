# ADR-032: Structural Code Intelligence — Scope-Graph Binding and Precise-Index Harvesting

## Metadata

- **Status**: Proposed
- **Date**: 2026-10-19
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
- Rust-side (existing workspace, PyO3, `_NATIVE_AVAILABLE` fallback), Python
  reference implementation first per the native-acceleration policy: rules for
  Python and Rust first (highest in-repo value), then TypeScript.
- **Build-vs-adopt**: evaluate GitHub's `tree-sitter-stack-graphs` crate before
  hand-rolling rules. Adopt if its per-language rule sets meet our incremental and
  embedding-free constraints; hand-roll only the subset otherwise. Decision point
  recorded at implementation time.

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

Every binding-bearing edge carries a precision level — `heuristic` |
`scope-resolved` | `compiler` — so retrieval can prefer precise edges, the agent
layer knows what is provable vs inferred, and reconciliation is deterministic
(compiler > scope-resolved > heuristic). This extends the existing edge metadata
convention; no schema break.

### Retrieval and memory integration

- Hybrid retrieval (FTS5 + vector + graph expansion) is unchanged in shape; tiers
  2.5/3 upgrade the expansion edges the reranker walks. Embeddings remain derived,
  rebuildable state keyed to the stable `oid` identity (ADR-026).
- The durable structural memory thesis is STRATA/StateHarness-shaped: the graph is
  the evolving project state outside model parameters; the LLM stays stateless and
  terminal.

## Alternatives Considered

- **Always-on LSP integration at query time.** Rejected: couples retrieval latency
  and reliability to per-user toolchain state; SCIP-style offline dumps give the
  same precision with none of the runtime coupling.
- **Full semantic databases (CodeQL/Kythe-class).** Rejected: build-coupled,
  heavy, per-language toolchain commitment that cannot be a default for a
  framework that must index arbitrary projects with no build info.
- **LLM-extracted structure.** Rejected as a substrate (cost, nondeterminism,
  unverifiable) — this ADR's premise. LLM remains the consumer, not the indexer.
- **Scope-graph binding only in Python (no Rust/native path).** Rejected as the
  end state: per-file binding over repo-scale corpora is a measured-hot-path
  candidate for the existing Rust workspace; Python stays as the typed reference
  implementation per the native-acceleration strategy.

## Validation

- **Differential parity tests**: heuristic vs scope-resolved edges over the repo
  corpus (Python + Rust first); every disagreement is a case study, not noise.
- **Golden-file binding tests** per language: resolution results pinned on
  representative constructs (impl/trait, closures, shadowing, re-exports).
- **Provenance reconciliation tests**: precise-override and fallback-preservation
  behaviors pinned positively and negatively.
- **Retrieval quality**: existing Graph-RAG retrieval parity harness plus
  end-to-end agent benchmarks; success is measured recall/precision of retrieved
  context and token spend per resolved task, not index size.
- **Performance budget**: per-file incremental re-bind must stay within a small
  multiple of parse cost (IncSFS-style incremental discipline); no full-repo
  recompute on single-file edit.

## Consequences

- Per-language scope rules are the main new maintenance surface; the rule set
  starts at two languages and grows on demand.
- Tier 3 availability is environment-dependent by design; documentation must state
  what is gained when a toolchain is present, and the default experience must never
  imply compiler precision without one.
- def/ref edges grow storage; references are stored compactly (or harvested
  on-demand per query target) rather than fully materialized when the
  node/edge budget matters.

## Revision History

| Rev | Date | Change |
|-----|------|--------|
| 1.0 | 2026-10-19 | Initial proposal: tier 2.5 scope-graph binding + tier 3 SCIP/LSP harvesting, provenance model, evidence base from the dataserver3 arxiv corpus. |
