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


## Polyglot consumer qualification and service roadmap — 2026-10-11

Status: AnvaiOps handback acknowledged and ingested; shared corrections in progress, Sandesha handback pending. This extends the existing
v2 contract and ADR-014/015, not a second parser or independent identity system.
The Rust qualification landed in Victor [#1297](https://github.com/anvai-labs/victor/pull/1297)
(`dacac4d1`). AnvaiOps ADR-0018 already chooses the shared package; ADR-0017 owns
its commercial CodeGraph service. Reconcile their implemented surfaces before
creating new ones. Source version 0.9.0 is not proof of a published wheel: qualify
an exact source revision now and verify publication before changing release pins.

### Ownership and reusable boundaries

| Layer | Owner and boundary |
|---|---|
| Language detection, parsing, repository resolution, stable identities, diagnostics, incremental deltas, graph projections | OSS `victor-codegraph`; fix once with minimized cross-language fixtures. Python orchestration and the existing tree-sitter native parser remain the baseline. |
| Agent tools, bounded context selection, framework/service integration | OSS Victor, consuming the same package. A new public framework API requires the repository's FEP process. |
| TypeScript consumption | Versioned serialized contracts and generated types/client over the shared service, or a process adapter when measured requirements justify it. No second TypeScript parser implementation merely to serve a TypeScript UI. |
| Storage projection | Existing neutral adapters and ProximaDB substrate. Preserve one symbol identity and explicit unresolved references; report actual persisted/exported counts. |
| Commercial hosted product | AnvaiOps: tenancy, repository authorization, connector/webhook orchestration, billing, deployment and commercial UI composition. Reuse its existing CodeGraph API/React Flow surface. |
| Consumer qualification | Sandesha Client supplies TS/TSX/JS and application-boundary evidence. Repository code analysis is separate from mailbox processing; no live mail, secrets or private-source publication. |

The OSS core must remain independently usable and self-hostable: no AnvaiOps
account, commercial service, entitlement check or billing dependency is required
for parsing, graph analysis or the neutral service contract. AnvaiOps consumes
released core capabilities through versioned APIs/adapters rather than a parser
fork. Generic authentication/authorization extension points, input isolation,
resource limits and secure defaults remain available to OSS deployments;
commercial tenant policy, entitlement decisions and hosted operations stay in
AnvaiOps. Do not confuse removing commercial policy with removing security.

Ownership follows the same core-versus-commercial direction as the other AnvaiOps
products. Commercial requirements may inform neutral contracts, but implementation
dependencies point from AnvaiOps to the OSS core, never the reverse. Validate both
standalone OSS operation and commercial adapter parity before service acceptance.

### Ordered gates and status

| Gate | Acceptance evidence | Status |
|---|---|---|
| CG-P0: Rust baseline | Sandhi audit at `a2b0f0c`, Victor #1297; parser dependencies present, explicit conservative resolution, 107 Rust files without parse diagnostics. Static results are lower bounds, not complete runtime calls or a performance diagnosis. | Complete for that snapshot only |
| CG-P1: Consumer inventory and handback | AnvaiOps polyglot and Sandesha TS/TSX/JS snapshots; capability/diagnostic matrix; labeled relationship truth set; measured cold/warm indexing and bounded query performance; minimized reproducible gaps. | AnvaiOps evidence received; Sandesha pending |
| CG-P2: Shared correctness fixes | TDD in existing parser/resolver/identity/delta suites. Preserve unresolved ambiguity, deletion equivalence, byte/line spans and stable IDs across checkout roots; deduplicate tests by invariant. Both consumers rerun pinned fixes. | CG-S01–CG-S05 implemented and locally qualified; pinned consumer reruns remain |
| CG-P3: Release/package qualification | Build/install exact wheels in supported Python/platform shapes; dependencies complete, missing-grammar failures explicit before index mutation, migration and consumer compatibility verified. Publish source 0.9.0 or its approved successor before released consumers require it. | Local wheel/sdist qualified; publication tracked by `victor-codegraph-v0.9.0`; consumer qualification pending |
| CG-P4: Service contract and lifecycle | Review existing Victor and AnvaiOps routes, then propose one versioned neutral contract with bounded snapshot jobs, explicit partial/error/cancelled states, generation-bound queries and pagination. FEP before new public framework API implementation. | Planned; no new service claim |
| CG-P5: UI and agent acceptance | Same snapshot IDs/edges/diagnostics in CLI, API and UI; source navigation, ambiguity display, filters, bounded expansion, accessibility and shared dark/light/system theme. Synthetic authorized/denied/cancel/restart journeys through headed AgentBrowser. | Planned after CG-P2–P4 |

Security and correctness precede throughput tuning. Predeclare representative
workloads and numeric targets before profiling. Report wall time, CPU, peak RSS,
index size, cold/warm differences and p50/p95 bounded-query latency. Optimize
algorithms, copies and incremental work first. Native acceleration needs the
existing measured Python/Rust decision process; language/file size alone is not
performance evidence. Service concurrency, persistence and caches must be bounded.

### Common audit and handback protocol

1. Read each repository's instructions and authoritative plan. Record local HEAD,
   fetched integration SHA and dirty status. Use a clean linked worktree; qualify
   intentional dirty changes separately with a sanitized manifest/digest. Never
   overwrite another session's work or mix evidence from different snapshots.
2. Pin Victor/CodeGraph/parser/grammar/tool versions and inventory production,
   tests, generated, vendored and unsupported files separately. Initially disable
   embeddings and LLM calls. Do not execute code, package scripts, imports or build
   hooks from the repository under analysis merely to parse it.
3. Run repository-aware parsing plus Victor graph index/stats/export/impact on
   that snapshot. Distinguish parser semantic graphs from optional CCG and vector
   index outputs. Record skipped/fallback/error files; an empty result is not a
   successful complete analysis. No inference of a complete call graph from a
   zero-error parser run.
4. Label a reviewable truth set: exact expected symbols, fully scoped endpoints,
   call/import/reference type, byte/line spans, resolution outcome and provenance.
   Include negative and ambiguous examples; report reviewed true/false/missing
   edges with denominators per language/relation. Separate same-name guesses,
   framework conventions and HTTP/schema links from statically proven CALLS.
5. Analyze cohesion/coupling with production/test/generated separation: observed
   dependency cycles, fan-in/out, reachability and ownership boundaries. High fan-in
   may be correct API reuse; no rewrite or deletion based only on file length or
   graph degree. Validate suspected defects against source and relevant tests.
6. Extend existing tests for confirmed gaps, providing minimized permissively
   shareable fixtures and expected results to Victor. Keep private source, tokens,
   customer identifiers, mail and full graph exports local; use artifact digests.
   Consumer-specific product fixes stay in that consumer's PR and plan.
7. Return a Markdown handback plus machine-readable JSON matching the fields below.
   Include exact commands, exit codes, limitations, failed evidence and final PR
   refs. Mark proposed requirements separately from implemented/verified behavior.

```json
{
  "schema_version": 1,
  "consumer": "anvaiops-or-sandesha-client",
  "source": {"head": "", "integration_head": "", "dirty": false, "manifest_sha256": ""},
  "toolchain": {"victor_sha": "", "codegraph_version": "", "python": "", "tree_sitter": "", "language_pack": ""},
  "commands": [{"argv": [], "exit_code": null, "evidence_sha256": ""}],
  "coverage": [{"language": "", "scope": "production", "files": 0, "success": 0, "partial": 0, "fallback": 0, "error": 0, "unsupported": 0}],
  "truth_sets": [{"language": "", "relation": "", "reviewed": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0, "unresolved": 0}],
  "findings": [{"id": "", "severity": "", "owner": "victor-codegraph-or-consumer", "expected": "", "actual": "", "fixture": "", "status": "proposed"}],
  "performance": {"workload": "", "hardware": "", "targets": {}, "samples": [], "limitations": []},
  "service_requirements": [],
  "ui_requirements": [],
  "artifacts": [{"path": "", "sha256": "", "visibility": "private"}],
  "pull_requests": [],
  "remaining_gates": []
}
```

This is an evidence interchange template, not an implemented public API. Fill actual
counts; use explicit unknown/not-measured values and reasons instead of invented
zeros. The templates' zero counters are placeholders, not results.

### Service and UI questions to answer with the handbacks

- Which existing queries solve a verified agent/user task: definition, references,
  direct callers versus transitive impact, change scope, dependency path or context
  selection? Return source spans, evidence and uncertainty with every result.
- What existing IDs and API DTOs lose information? Preserve opaque IDs end-to-end;
  display labels and lossy identifier sanitization must not determine edge targets.
- How will jobs be deduplicated by authorized repository/snapshot/config, resumed
  after restart and atomically published as a complete generation? A database
  transaction protects stored state; it does not make clone/parse/embedding calls
  part of that transaction. Queries must not mix generations.
- How will authorization apply before retrieval/traversal/export, including counts
  and caches? Use trusted repository mappings, not arbitrary server filesystem paths;
  bound file sizes, traversal depth, returned nodes, time, concurrency and parser
  resource use. Treat source as untrusted data, including archive/symlink boundaries.
- Which UI components already exist in AnvaiOps and Victor? Prefer shared schemas
  and optional presentation components; no commercial tenancy/billing dependency
  may flow into the OSS compiler. Neither an attractive graph nor a browser smoke
  proves relation accuracy or full service readiness.

Handbacks are requested at AnvaiOps
`docs/operations/CODEGRAPH_POLYGLOT_HANDBACK.md` and Sandesha Client
`docs/CODEGRAPH_TYPESCRIPT_HANDBACK.md`, with adjacent `.json` evidence. Their
handoffs are `CODEGRAPH_POLYGLOT_HANDOFF.md` and `CODEGRAPH_TYPESCRIPT_HANDOFF.md`
in those same directories. Fold accepted results into CG-P1/P2 here; retire redundant
interim notes only after evidence and dispositions are preserved in the owning plan.


### AnvaiOps handback accepted — 2026-10-11

Received `docs/operations/CODEGRAPH_POLYGLOT_HANDBACK.md` and adjacent JSON
from the sibling AnvaiOps repository (consumer snapshot `1aeef9fa`, shared source
`b09e8d88`). Evidence ingestion acknowledges the findings, not service acceptance.
The consumer corrections remain owned by AnvaiOps draft PR #488. Only minimized
synthetic fixtures may enter this OSS repository; private source and graphs stay local.

| Finding | Owner | Disposition / qualification gate |
|---|---|---|
| CG-S01: Python false targets and missed aliases | victor-codegraph | Implemented and qualified: exact import bindings and conservative receiver/scope resolution; positive and negative fixtures |
| CG-S02: JS object-method identity collision | victor-codegraph | Implemented and qualified: structural object scopes; unique IDs stable under line shifts |
| CG-S03: TS imports/barrels/type-only evidence | victor-codegraph | Implemented and qualified: explicit module/import binding; type-only imports never prove runtime CALLS |
| CG-S04: persisted graph/query parity | Victor | Implemented and qualified: one canonical projection, materialized endpoints, truthful counts and opaque-ID lookup |
| CG-S05: Python call byte spans | victor-codegraph | Implemented and qualified: exact UTF-8 spans, including multiline calls |

The six recorded synthetic failures now pass. Existing owning suites were extended
with positive and adversarial cases; no parallel regression suite was introduced.
Unknown receivers, shadowed bindings, unsupported module aliases and ambiguous
exports remain unresolved rather than producing confident guessed edges.

| Qualification | Result / scope |
|---|---|
| Shared package | 180 tests; 91.99% coverage (90% gate); six original handback regressions pass |
| Labeled synthetic truth set | 13 cases; no false positives/negatives in the labeled set; root/line-shift identity, determinism, rename/deletion delta parity pass |
| Victor runtime | 330 passed, 1 skipped in Graph-RAG; 39 adjacent CLI/storage tests passed; type check covers 2,001 runtime modules |
| Private frozen snapshot | 738 parsed files; 7,647 symbols and unique IDs; 735 success / 3 partial files |
| SQLite projection | Existing walker filters select 735 files / 7,638 symbols; exact eligible ID parity, zero dropped resolved repository relations, 9/9 opaque-ID lookups |
| Persistence accounting | 7,638 stored nodes / 13,776 edges; attempted edge upserts remain separately labeled (15,150), never claimed as stored edge count |
| Local diagnostic sample | Cold graph index 8.71 s, parse 3.48 s, peak RSS 428 MB on Apple ARM64/Python 3.12; not a controlled benchmark or service SLO |
| Packaging | Wheel + sdist metadata passed; clean zero-dependency Python install, explicit missing-grammar diagnostics, optional-grammar wheel tests; Linux Python 3.12/3.13 remain CI gates |

Qualification disables CCG/embeddings and isolates vector cleanup for fresh fixtures
that own no vectors. Existing SQLite deletion can instantiate an embedding provider
even for graph-only work; track that separate lifecycle gap before claiming an
entirely offline production CLI. Runtime suites use Hugging Face offline mode.
Private source, graphs and detailed evidence remain local; only aggregate results
and synthetic fixtures are published.

**Consumer contract:** rebuild derived graphs for the 0.9.0 structural object IDs and
relation corrections; retain opaque IDs and full byte spans, parser status and
unresolved evidence. `nodes_created`/`edges_created` remain attempted-upsert counters;
`persisted_nodes`/`persisted_edges` report actual store totals and may be null for a
store without a stats operation. File eligibility remains a consumer configuration,
so compare the same eligible snapshot. See the package README's migration table.

**Release receipt:** the exact source and package pin are owned by
[`victor-codegraph-v0.9.0`](https://github.com/anvai-labs/victor/releases/tag/victor-codegraph-v0.9.0)
and its release workflow; this pre-publication qualification does not establish that
the tag has published. After publication the consumer pin is
`victor-codegraph[treesitter]==0.9.0`. AnvaiOps #488 owns its consumer rerun and atomic
snapshot/read-boundary corrections; do not duplicate them here. Sandesha's handback,
CG-P4 lifecycle and CG-P5 headed UI acceptance remain open.


### Release alignment follow-up — 2026-10-11

Owner: AnvaiOps consumer session, branch `fix/codegraph-release-alignment`,
original baseline `fb310ec02` from parent PR #1299. #1299 is now merged as
`62552e965`; the release-only commit was rebased onto that integration commit
to remove duplicate pre-squash history. This branch owns publication/version checks.

Independent package versions are preserved. Victor's graph extra now requires
exactly `victor-codegraph[treesitter]==0.9.0`; the version checker and sync helper
cover the CodeGraph package, runtime and consumer pin. The release workflows
validate source/tag and wheel/sdist versions, isolate TestPyPI rehearsals, and
require the exact published CodeGraph wheel before any Victor publication.
See [release ordering](../development/PR_WORKFLOW.md#codegraph-release-ordering-and-version-checks).

Local evidence: the original 12 release-contract regressions failed before the
change; the extended suite and native version owner pass (49 tests) after implementation.
Independent review found and corrected missing checkout permissions in publisher
jobs and a test fixture directory collision. Both package wheel/sdist pairs were
built and passed the artifact validator. Full-suite collection passed (34,257
tests); this session performed no publication. The parent subsequently published
CodeGraph 0.9.0 from `62552e9659c320c6ad22e875d099d56f0b5c9727`. Its
[release receipt](https://github.com/anvai-labs/victor/releases/tag/victor-codegraph-v0.9.0)
and PyPI wheel are now verified available; the earlier registry-blocker observation
is superseded. No Victor AI release was published by that CodeGraph release.

Execution ownership is now handed entirely to the parent session at the user's
request. This consumer session winds down after reconciliation; do not run a
second release or consumer migration lane here. Parent next: review/merge
[release safeguards #1300](https://github.com/anvai-labs/victor/pull/1300) for future
publication, then adopt the already-published
the exact pin and complete commercial adapter/UI acceptance through
[AnvaiOps #488](https://github.com/anvai-labs/anvaiops/pull/488). The Python 3.12
API/worker migration is committed there at `3511a40`. The canonical detailed
consumer receipt is AnvaiOps `docs/operations/CODEGRAPH_POLYGLOT_HANDBACK.md`
and its adjacent JSON; those preserve exact source qualification and local
image/test evidence. No tag, package publication, deployment or gate promotion
was performed by this session.
