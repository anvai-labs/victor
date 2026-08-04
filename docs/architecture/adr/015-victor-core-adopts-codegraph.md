# ADR-015: Victor Core Adopts victor-codegraph as the Foundational Code Parser

## Metadata

- **Status**: Accepted (2026-08-04 — Phases 0–1 shipped; Phase 2 partial; Phase 3 pending; was Proposed)
- **Date**: 2026-06-26
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR 014 (shared victor-codegraph package), ADR 007 (vertical/contracts boundary)
- **Cross-repo**: ProximaDB `ADR-029` (shared chunker), ProximaDB `ADR-044` (stable symbol oid — shipped in victor-codegraph 0.1.2), AnvaiOps `ADR-0018` (consumer)

## Context

`victor-codegraph` (ADR 014) is now the single, neutral code→CPG parser, and the
`victor-coding` vertical already delegates its symbol extraction to it (parse-layer
delegation). But **victor CORE still carries its own hand-rolled code-parsing mechanisms**,
which both duplicate `victor_codegraph` and — more importantly — produce **divergent symbol
ids / edges / chunk boundaries** from the shared parser.

This divergence breaks the correlated-CPG invariant. The whole value (ProximaDB's
`CODE_GRAPH_CORRELATED_SUBSTRATE` keystone) is that a symbol is one entity with a **shared
`oid`** across a row, a graph node, and a vector — and across the *lifecycle*: author (Victor)
→ parse → chunk → embed → store (ProximaDB) → serve (AnvaiOps). That holds **only if the same
parser emits the ids/edges everywhere.** If victor core parses differently from the vertical,
ProximaDB, and AnvaiOps, then "who calls X / blast radius / nearest" answers **differ by tool**.
Single parser = lifecycle consistency, not cleanup.

### Inventory of victor-core mechanisms (audited)

| # | Location | Mechanism | Priority |
|---|----------|-----------|----------|
| 1 | `core/graph_rag/indexing.py:1489` | `provider.extract_symbols(content, language)` (TreeSitterAnalysisProtocol) | **HIGH** |
| 2 | `core/graph_rag/indexing.py:2097` | `_extract_symbols_fallback` — regex def/class | **HIGH** |
| 3 | `framework/search/codebase_embedding_bridge.py:843` | `provider.extract_symbols()` + raw parse tree | **HIGH** |
| 4 | `core/chunking/strategies/code.py` | regex function/class boundary chunker (8+ langs) | MED |
| 5 | `storage/vector_stores/code_chunking.py` | symbol-span/structural chunkers (need symbol *input*) | MED |
| 6 | `native/python/symbol_extractor.py` | pure `ast` function/class/import extractor (Python) | LOW |
| 7 | `storage/memory/extractors/tree_sitter_extractor.py` | tree-sitter → Entity memory | LOW |
| 8 | `contrib/codebase/analyzer.py` | `BasicCodebaseAnalyzer` — regex, **currently unused** | LOW |

Keep as-is (not hand-rolled parsing): the `contrib/parsing/` Null* capability stubs +
`vertical_protocols.py` protocols (runtime injection seams), and `core/utils/ast_helpers.py`
(already re-exports from victor-contracts). The chunking *strategies* (#4, #5) are good — only
their **symbol source** should change.

## Decision

Adopt `victor_codegraph` as victor core's foundational parser via **soft-import, default-on
delegation** (the same mixed-read-safe pattern as the vertical and the ProximaDB SDK): when
`victor_codegraph` is importable, core sources symbols/relations from `victor_codegraph.parse()`;
otherwise it falls back to the existing path. `victor-codegraph` is installed in CI
(foundational) so the real path is exercised. No protocol/registry is removed — `victor_codegraph`
becomes the *implementation behind* the existing `extract_symbols`/analysis seams.

### Phased migration (each phase a separate, CI-verified PR)

- **Phase 0 (done):** `victor-codegraph` installed in CI before verticals; vertical delegates
  (ADR 014). 
- **Phase 1 — Core indexing keystone (HIGH, #1/#2/#3):** route the graph-RAG indexer's symbol
  extraction (and its regex fallback) and the embedding bridge's parse context through
  `victor_codegraph.parse()`, mapping `CodeSymbol`/`CodeRelation` onto the existing
  symbol-dict / edge shapes. This is the seam that makes victor's CPG `oid`s match
  ProximaDB/AnvaiOps. **Gate on the full graph-RAG test suite** (cannot be verified offline;
  must be green in CI).
- **Phase 2 — Chunking symbol source (MED, #4/#5):** keep the chunking *strategies*; swap their
  symbol input to `victor_codegraph.parse().symbols`; let `core/chunking/strategies/code.py`
  delegate to `victor_codegraph.chunk()` (AST-aligned + size-capped vs regex).
- **Phase 3 — Utilities (LOW, #6/#7/#8):** `native/python/symbol_extractor.py`,
  the entity-memory extractor, and the (unused) `BasicCodebaseAnalyzer`.

### Invariants

- **Determinism:** the `oid`/symbol-id `victor_codegraph` emits must be identical to what
  ProximaDB stores and AnvaiOps serves — verified by a cross-surface fixture (same source →
  same ids) as part of Phase 1.
- **Soft + reversible:** every seam keeps its legacy fallback until the phase is baked; no
  flag-day.
- **Eval the ranked surface:** graph-RAG retrieval is a ranked/generated surface — its eval
  suite (recall/trajectory) must not regress across the swap (tests-vs-evals discipline).

## Consequences

**Positive:** one parser → consistent symbol identity across the whole lifecycle (the
correlated-CPG promise actually holds); core sheds duplicated regex/ast parsing; future
language support lands once (in `victor_codegraph`) for every consumer.

**Negative / risk:** the HIGH-priority core indexing pipeline is complex and well-tested; the
swap must be verified in CI (the suite is too heavy to run reliably offline). Output shape must
map exactly (symbol-dict / edge types) to avoid silent recall regressions — hence the eval gate.

**Neutral:** tree-sitter stays a core dep (fallback + grammars); the capability-registry
stubs stay (runtime injection).

## Status of work

- Phase 0: shipped (CI install + vertical delegation, ADR 014 PRs).
- Phase 1: shipped. Core graph indexing preserves `victor-codegraph` v2 symbol IDs and
  uses the same v2 identity for synthetic file/module nodes. One filtered repository snapshot
  now supplies import-aware cross-file calls, inheritance/implementation, and module-import
  edges; semantic sources bypass the older name-only fan-out resolver while mixed/absent package
  installs retain the existing soft fallback. The coding analysis provider delegates symbols,
  relations, and imports with a cross-surface conformance fixture.
- Phase 2: partial/live. Coding-vertical chunkers delegate where the package is installed; the
  remaining core chunk strategies still need convergence on the v2 chunk planner.
- Phase 3: pending.

The v2 target and its invariants are recorded in
[`codegraph-v2-design.md`](../codegraph-v2-design.md).
