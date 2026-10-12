---
fep: "0040"
title: "CodeGraph embedding conformance across local and hosted stores"
type: Standards Track
status: Review
created: 2026-10-12
modified: 2026-10-12
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
---

# FEP-0040: CodeGraph embedding conformance

## Summary

A consumer of the shared compiler must also preserve its embedding-input and
provenance contract. SQLite/LanceDB and ProximaDB are storage projections of the
same canonical symbols, source versions and semantic vectors. AnvaiOps adds
policy, scheduling, billing and UX; it must not independently derive identities
or silently replace semantic embeddings with hashes.

Status is Review. The implementation is a reviewable candidate; publication of
new public contracts requires recorded design acceptance. Existing parser and
storage defect corrections proceed independently. User authorization to align
the consumers is recorded in the originating session, not formal FEP acceptance.

## Evidence and proposed change

- Victor graph nodes currently embed name, signature and documentation. Code-body
  chunks are a different input contract and must remain separately versioned.
- AnvaiOps previously used placeholder vectors despite sharing the parser package.
- Victor's graph embedder can substitute hash vectors, and its model service can
  return zeros on model failure. These must never be published as semantic evidence.
- Add a dependency-free embedding profile/input/result contract to victor-codegraph:
  pinned model revision, dimensions, text contract, token limit, normalization,
  query prefix, input hash, source hash, canonical symbol ID and profile identity.
- One shared symbol-summary preparation and strict vector validation. Require exact
  batch cardinality, finite non-boolean values, correct dimensions and nonzero norm.
  Reject duplicate symbol IDs before invoking inference. No hash/zero fallback.
- Optional sentence-transformers execution reuses that existing third-party model
  implementation, pins revision, disables remote code, bounds batches and records
  preprocessing. No mandatory Torch/model dependency in the compiler base package.
- Victor uses the shared preparation/validation at its graph boundary; AnvaiOps
  consumes the public package contract with local BGE or an explicitly qualified
  injected encoder. Existing graph-only mode stays explicit. No automatic export
  of private code to a remote provider.

## Compatibility and migration

The existing compiler/parser APIs and graph-only storage remain available. Real
embeddings require an explicit configured profile. Existing placeholder generations
remain descriptive only. A model or preprocessing change creates a new generation;
reusing cached vectors requires the full profile/input hash and canonical identity
to match. Do not treat 384-dimensional shape alone as compatibility.

The runtime graph boundary must report an embedding failure instead of counting
fallback vectors as successfully embedded. Existing successful text preparation
is byte-identical. Body edits change source provenance even when summary text
remains unchanged. The candidate refreshes vectors and their source evidence on
body-only changes; vector reuse with metadata-only refresh is a later optimization.
Missing compiler source hashes fail explicitly. Existing unknown/different graph
profiles require a fresh index; neither incremental nor force parsing may erase
those markers to bypass the check. Storage-generation publication remains the
consumer's concern.

## Acceptance criteria

1. Shared synthetic fixtures preserve text, identity and vector evidence in both
   adapters; missing/malformed inference cannot publish a semantic generation.
2. Real pinned BGE vectors are generated locally and compared across consumer
   projections. Separately measure relevance on labeled queries; a deterministic
   vector fixture proves plumbing only.
3. Validate wheel optional dependencies, typed exports and Python 3.12/3.13.
4. Run existing graph embedding/indexing and consumer publication suites; extend
   their owners rather than duplicate tests by storage implementation.
5. Independent adversarial review, all applicable CI gates, recorded design
   acceptance and a new immutable package release precede consumer pin adoption.

## Alternatives and limits

Copying Victor's internal runtime into AnvaiOps recreates drift and breaks package
boundaries. Loading the full agent runtime merely to embed symbols adds unrelated
services. Shared preparation with interchangeable trusted inference/storage
adapters preserves optional deployment choices. Native graph acceleration, Q&A,
repository policy, durable jobs and relevance acceptance remain separate gates.

## Review process and prototype evidence — 2026-10-12

The implementation is a local prototype, not a released contract. Independent
adversarial review identified and corrected single-node validation bypass, stale
model caches, structural vectors mislabeled as semantic vectors, profile checks
limited to changed files, and missing source provenance. Existing test owners
were extended rather than creating another embedding suite per adapter.

| Check | Evidence / limit |
|---|---|
| Shared package | 198 tests; 92.28% coverage against the existing 90% gate |
| Victor affected suites | 621 passed; collection: 34,345 tests, no errors in the isolated worktree environment |
| AnvaiOps API/connector-SDK suites | 1,703 passed, 3 skipped; candidate supplied by local PYTHONPATH, released pin remains 0.9.0 |
| Real BGE storage parity | Actual Victor generation → LanceDBProvider and AnvaiOps publication → ProximaDB 0.4.0 SST: 3 synthetic symbols / 8 published graph records; profile, canonical ID, input/source hash and vectors match; 6.799 s single local run |
| Packaging | Candidate wheel/sdist build; no publication or 0.9.0 replacement. Clean Python 3.12 wheel import/optional-dependency smoke passed; Linux ARM64 Python 3.13 wheel installed and 18 contract tests passed. Optional native/model execution on 3.13 and the full release matrix remain gates. |

**Collection/query conformance remains open.** The guard inspects graph-node
markers, not every existing/orphan vector row. LanceDB can independently configure
its query model. A persisted collection-level model contract, query encoder
compatibility check and safe migration are prerequisites to accepting semantic
search. No similarity search, relevance, hosted authorization, concurrent writer,
restart or full Q&A acceptance is implied by the document-vector parity run.
The local BGE run uses immutable revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; its profile is
`468b9ab3e92f01de25d4244adf05ee5812bcfbcd90647c8e4466142a61d2b34b`.

## Ordered implementation and release plan

1. Record design acceptance under the repository FEP process. Complete collection/
   query-model ownership and migration before marking Q1 semantic search accepted.
2. Qualify the optional package on supported Python versions and publish a new
   immutable release. Never republish the candidate over 0.9.0.
3. Pin both consumers to that release, rerun actual-vector storage parity, then
   measure lexical versus semantic relevance on a labeled representative corpus.
4. Add authorized generation-bound retrieval, citations and graph navigation;
   commercial policy/jobs/UI remain in AnvaiOps. Native acceleration requires
   measured need and follows the existing architecture mandate.

## References

- [CodeGraph tracker](../docs/architecture/codegraph-v2-design.md)
- AnvaiOps TD-F1 and CODEGRAPH_POLYGLOT_HANDBACK
- ProximaDB ADR-044 (identity) and ADR-061 (canonical authority versus projections)
