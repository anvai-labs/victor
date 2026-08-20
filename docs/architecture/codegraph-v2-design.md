# Victor CodeGraph v2 Design

Status: implementation target

## Vision

Victor CodeGraph is a deterministic repository-to-semantic-index compiler. It converts a
versioned repository snapshot into stable symbol identities, explicit relationship evidence,
retrieval-ready chunks, diagnostics, and an incremental delta. Identity survives non-semantic
edits; uncertainty and unsupported capabilities are never hidden.

The default product is a **Code Semantic Graph**: files, modules, named symbols, imports,
containment, inheritance, calls, and embedding chunks. Intra-procedural CFG/CDG/DDG generation is
an optional higher-fidelity layer and is not implied by the default package.

## First-principles invariants

1. **One identity.** Parser symbols, chunks, relations, and storage records use the same stable
   structural key. The old line-coupled identifier is an explicit migration alias only.
2. **Repository-relative paths.** Identity never depends on checkout location, operating system,
   or an absolute path.
3. **No false graph endpoints.** A relation target is either a resolved symbol key or a structured
   unresolved reference. A bare textual name is never presented as an existing symbol node.
4. **Evidence is preserved.** Recursive calls and call-site spans are retained. Resolution records
   its confidence and provenance.
5. **Bounded chunks.** Every chunk respects the configured hard budget. Forced character splitting
   is explicit; source spans identify the actual chunk, not its whole parent symbol.
6. **No silent degradation.** Parse results report success, partial extraction, fallback, or error,
   plus diagnostics and a language capability tier.
7. **Incremental equals full.** Applying an index delta to the prior snapshot produces the same
   semantic index as a clean rebuild, including deletions.
8. **Adapters validate.** Embedding cardinality and dimensions, identity collisions, and dangling
   endpoints fail before storage mutation.

## Pipeline

```text
RepositorySnapshot
        |
        v
Language frontends ------> diagnostics + capability tier
        |
        v
Canonical semantic IR
        |
        v
Repository resolver
   |              |
   v              v
Chunk planner   Index delta
   |              |
   +------+-------+
          v
Versioned storage adapters
```

## Compatibility

The v2 model retains `legacy_id` and dual-emits it in adapter properties. Existing entry points
remain callable. New repository-aware entry points provide path normalization, cross-file
resolution, and deletion-aware deltas. Behavior-changing identity fields are versioned with
`identity_version="v2"`.

## Verification gates

- no stable-key collision across language fixtures and representative repositories;
- identical keys after line shifts and across checkout roots;
- exact relation targets for scoped and cross-file fixtures;
- every adapter edge points to an emitted node or an explicit external-reference node;
- every chunk is within the configured hard budget and reports exact spans;
- incremental-delta application equals a full rebuild;
- consumer conformance fixtures produce identical IDs and relations.
