# ADR-014: Extract the code→CPG chunker into a shared `victor-codegraph` package

## Metadata

- **Status**: Accepted (2026-07-02 — `victor-codegraph` 0.1.x released, `victor-coding` delegates; was Proposed)
- **Date**: 2026-06-26
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR 007 (Vertical Distribution Model and Contracts Boundary), ADR 015 (core adoption)
- **Cross-repo**: ProximaDB `docs/12-design/adr/ADR-029-shared-codegraph-chunker-package.adoc` (authoritative cross-repo decision); ProximaDB `ADR-044` (stable symbol oid — extends this package's identity contract); AnvaiOps `docs/adr/0018-consume-shared-codegraph-chunker.md` (SaaS consumer)

## Context

Victor's coding agent reasons from a **Code Property Graph (CPG)** + code embeddings — it answers
"who calls X / blast radius / what is semantically near this" from durable memory instead of re-reading
files each turn. The component that turns source into that graph — a tree-sitter chunker that extracts
**symbols** (function/class/method/struct…) and **relations** (CALLS/INHERITS/IMPLEMENTS/IMPORTS…) — lives
today in the `victor-coding` vertical:

- `victor_coding/codebase/chunker.py` — `CodeChunker` (Python `ast`, body-aware) + `TierAwareChunker`
  (multi-language fallback chain: `python-ast → tree-sitter → config-aware → sliding-window`).
- `victor_coding/codebase/tree_sitter_extractor.py` — `TreeSitterExtractor` (symbol + edge extraction).
- `victor_coding/codebase/tree_sitter_analysis.py`, `embeddings/chunker.py` — analysis provider + AST-aware
  embedding chunker.

An audit across the owner's sibling repos found the **same primitive re-implemented** in ProximaDB's
Python SDK (`clients/python/src/proximadb_sdk/chunking_strategies/code.py`): a richer symbol/relation
taxonomy (18 symbol types, 18 relation types, complexity metrics) but **no size-capping** and a **stubbed
JS/TS parser**. A third copy is about to be written in AnvaiOps (its planned code-graph-sync worker).

Three diverging copies of one capability is the "silent duplication of an existing primitive" failure
mode. Victor owns the most complete implementation, but it is **trapped inside the `victor-coding`
vertical** — a consumer that only wants the parser would have to depend on the whole vertical (and its
`victor-sdk`/runtime surface), violating the thin-dependency spirit of ADR 007.

For reference posture: LlamaIndex `CodeSplitter` is a *size-bounded, AST-aligned* splitter (`max_chars`)
with **no** symbol/relation extraction — it is the size-discipline floor our chunker should adopt, not a
substitute for the CPG extraction we already do.

## Decision

**Extract the chunker into a new standalone package, `victor-codegraph`**, released on PyPI under the same
`vjsingh1984/*` topology as `victor-contracts` and the verticals, and make Victor the **owner of record**.

- **Dependencies, hair-thin and neutral**: `victor-contracts` (zero-runtime-dep protocol layer) +
  `tree-sitter` + grammar packs. Apache-2.0. **No** framework coupling, **no** SaaS/commercial concept —
  so ProximaDB and AnvaiOps can both depend on it without pulling Victor's runtime and without tripping
  ProximaDB's OSS-boundary guard.
- **Donor**: `victor-coding`'s chunker + `TreeSitterExtractor` are promoted into `victor-codegraph` (they
  are the best implementation). We merge in ProximaDB's richer taxonomy and **fix the two gaps once**:
  add size-capping/body-split discipline (à la `CodeSplitter`) and a real JS/TS parser.
- **Surface**:
  - `parse(content, language, path) -> ParsedCode` (symbols + relations + complexity + the fallback chain).
  - Neutral `CodeSymbol` / `CodeRelation` / `SourceLocation` data model with **deterministic IDs**
    (idempotent incremental re-index).
  - `to_proxima_records(parsed)` adapter emitting the ProximaDB substrate-keystone `ProximaRecord` shape
    (one record per symbol = relational row + graph node + vector); emits the *shape*, does not embed or
    call the DB.
- **`victor-coding` consumes it back**: re-exports the moved symbols for backward compatibility, then
  depends on `victor-codegraph`. No behavior change for existing Victor users.
- **Victor's embedded CPG** (the `proximadb_provider.py` swap seam / `GraphStoreProtocol`) feeds
  `to_proxima_records()` into `EmbeddedProximaDB`, retiring the SQLite + LanceDB dual-store and the
  unpopulated `embedding_ref` bridge.

This keeps the chunker in Victor's domain (where the expertise is) while making it consumable by ProximaDB
(optional SDK extra) and AnvaiOps (SaaS worker) without duplication.

## Consequences

**Positive**
- One chunker, three consumers (Victor, ProximaDB SDK, AnvaiOps); the JS/TS stub and missing size-cap are
  fixed once, for everyone.
- Respects ADR 007: consumers depend on a thin, contract-aligned package, not the whole vertical.
- Victor retains ownership of its core competency; improvements flow downstream automatically.

**Negative / cost**
- A new repo + release pipeline (`victor-codegraph-v*` tag) to maintain.
- A deprecation window where `victor-coding` re-exports from the new package; `code.py` in ProximaDB is
  deprecated-but-present until one minor release passes.
- A shape contract (`to_proxima_records()` ↔ `ProximaRecord`) to keep in lockstep across repos.

**Neutral**
- Victor's **document** chunker (`victor-rag/victor_rag/chunker.py`) overlaps ProximaDB SDK + AnvaiOps
  connector-sdk document chunkers — **out of scope here**; a separate consolidation decision if warranted.

## Migration (phased)

1. Stand up `victor-codegraph`; promote `victor-coding`'s chunker; merge taxonomy; add size-capping; fix
   JS/TS. Ship round-trip + recall-parity tests **and an eval suite** (ranked/extracted surface) per the
   ecosystem's tests-vs-evals discipline. (ProximaDB TD-CG1)
2. `victor-coding` re-exports + depends on the package; verify no behavior change.
3. ProximaDB SDK adds a `[codegraph]` extra and deprecates `code.py`. (TD-CG2)
4. AnvaiOps F3 worker imports the package; adds SaaS layer only, gated by AnvaiOps ADR-0017 Gate #1. (TD-CG3)
