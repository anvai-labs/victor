---
fep: "0040"
title: "CodeGraph embedding conformance across local and hosted stores"
type: Standards Track
status: Accepted
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

Status is Accepted. The repository owner explicitly approved FEP-0040 and an
expedited review in the originating session ("i approve"). This waives the normal
14-day review period for this proposal. Acceptance authorizes implementation and
qualification; it does not claim release, semantic relevance or hosted acceptance.

## Motivation

Sharing parser output alone left an untested semantic boundary: the commercial
consumer stored placeholders while the local runtime could silently substitute
hash or zero vectors. Equal vector dimensions do not establish equal meaning.
A reusable contract must identify exactly what text, model and preprocessing
produced each vector and connect that evidence to the compiler's symbol identity.
The customer should obtain compatible document vectors regardless of storage.

## Proposed Change

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

## Benefits

One neutral contract reduces consumer drift and makes inference failures explicit.
Optional dependencies preserve a lightweight compiler installation. The same test
corpus can qualify different storage adapters without copying extraction logic.
Model and source provenance distinguish stale data from a legitimate empty result.
The commercial layer retains policy, deployment, metering and user experience.

## Compatibility

The existing compiler/parser APIs and graph-only storage remain available. Real
embeddings require an explicit configured profile. Existing placeholder generations
remain descriptive only. A model or preprocessing change creates a new generation;
reusing cached vectors requires the full profile/input hash and canonical identity
to match. Do not treat 384-dimensional shape alone as compatibility.

## Migration Path

The runtime graph boundary must report an embedding failure instead of counting
fallback vectors as successfully embedded. Existing successful text preparation
is byte-identical. Body edits change source provenance even when summary text
remains unchanged. The candidate refreshes vectors and their source evidence on
body-only changes; vector reuse with metadata-only refresh is a later optimization.
Missing compiler source hashes fail explicitly. Existing unknown/different graph
profiles require a fresh index; neither incremental nor force parsing may erase
those markers to bypass the check. Storage-generation publication remains the
consumer's concern.

## Acceptance Criteria

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

## Drawbacks and Alternatives

Copying Victor's internal runtime into AnvaiOps recreates drift and breaks package
boundaries. Loading the full agent runtime merely to embed symbols adds unrelated
services. Shared preparation with interchangeable trusted inference/storage
adapters preserves optional deployment choices. Native graph acceleration, Q&A,
repository policy, durable jobs and relevance acceptance remain separate gates.

## Review Process

Prototype evidence dated 2026-10-12.

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

**Local collection/query enforcement is qualified locally; merge/release gates remain.**
LanceDB commits the complete profile with the initial Arrow schema, retains the
binding when empty, checks all row markers including orphans, and requires the
actual query encoder's profile to match. Semantic vectors are validated before
and after float32 conversion. Semantic upsert is one storage commit. Graph-index
preflight rejects an incompatible collection before force deletion. Use a fresh
collection/directory for legacy or changed profiles; no in-place migration or
cross-store atomic publication is claimed. Existing unprofiled document indexes
retain their defaults. Multi-process hosted publication, relevance, restart and
full Q&A acceptance remain separate gates.
The local BGE run uses immutable revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; its profile is
`468b9ab3e92f01de25d4244adf05ee5812bcfbcd90647c8e4466142a61d2b34b`.

### 0.10.0 qualification update

- Owner accepted this FEP and expedited its review. Package metadata, runtime
  version and Victor graph-extra pin agree on the new 0.10.0 candidate.
- 958 affected tests passed, 5 skipped in an explicitly offline model environment;
  the first broader run was interrupted during upstream model-loading retries.
  Shared package: 198 passed, 92.28% coverage. Independent focused review: 77 passed;
  its query float32 overflow/underflow finding was fixed before final review.
- AnvaiOps API/SDK: 1,703 passed, 3 skipped against the built 0.10.0 wheel, with no
  source-path override. Published consumer pins remain unchanged until publication.
- Wheel/sdist identities validated. Clean Python 3.12 base install and Linux ARM64
  Python 3.13 wheel/18-contract-test run passed. Optional model execution was
  measured on Python 3.12; no full native-model platform matrix is implied.
- Real BGE persistence parity plus three synthetic top-1 queries and a table-reopen
  query passed in 9.572 s. These are smoke cases, not representative relevance.
- A 10,000-row/384-dimension local storage probe (20 repetitions, deterministic
  encoder) measured median profile check 1.71 ms and search including check 8.02 ms;
  initial write 3.797 s. Model inference, multi-writer behavior and service SLOs
  are outside that measurement. No language rewrite or additional native toolchain.
- Existing test owners were extended; the CI selector now maps the embedding-model
  module to its existing embeddings suite instead of requiring duplicate tests.

### CI bootstrap correction

The first 0.10.0 candidate passed 39 checks but failed the vertical suite on a
bare mocked table schema in a duplicate core-provider test. That environment
also resolved published CodeGraph 0.9.0 instead of the in-repo contract. Both
development bootstraps now install the shared source package before consumers.
Two duplicate vertical provider tests were removed; the core owner retains
initialization coverage and adds a real legacy round trip with CodeGraph absent
during write/query. Affected suites: 45 core/bootstrap plus 15 vertical tests
passed; independent review repeated all 60. Production code is unchanged by
this correction. New-head CI remains required before release.

## Unresolved Questions

LanceDB binds its Arrow schema at creation and queries through the actual loaded
encoder profile. Other vector adapters require equivalent qualification. What
representative labeled corpus establishes relevance across languages? Those
are open acceptance gates, not guarantees inferred from the prototype. Hosted
policy, concurrent publication and restart semantics require separate evidence.

## Implementation Plan

1. Design accepted with an explicitly approved expedited review. Complete collection/
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

## Copyright

This proposal is licensed under the Apache License, Version 2.0.

### Publication receipt

PR [#1303](https://github.com/anvai-labs/victor/pull/1303) squash-merged into develop at `91bae704573ecd80dc5aa2919fb8ba4cb7cdccec` after exact-commit independent review and 41 applicable green checks. The independent CodeGraph 0.10.0 tag points to that source; Python 3.12/3.13 release tests, build, metadata checks and trusted PyPI publication passed in [run 38208368472](https://github.com/anvai-labs/victor/actions/runs/38208368472). This closes package publication, not hosted retrieval, representative relevance or Victor AI release acceptance. Consumer adoption remains tracked in the CodeGraph roadmap and AnvaiOps TD-F1.
