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

Status: co-design requested; consumer handbacks pending. This extends the existing
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
| CG-P1: Consumer inventory and handback | AnvaiOps polyglot and Sandesha TS/TSX/JS snapshots; capability/diagnostic matrix; labeled relationship truth set; measured cold/warm indexing and bounded query performance; minimized reproducible gaps. | Requested; handbacks pending |
| CG-P2: Shared correctness fixes | TDD in existing parser/resolver/identity/delta suites. Preserve unresolved ambiguity, deletion equivalence, byte/line spans and stable IDs across checkout roots; deduplicate tests by invariant. Both consumers rerun pinned fixes. | Pending CG-P1 |
| CG-P3: Release/package qualification | Build/install exact wheels in supported Python/platform shapes; dependencies complete, missing-grammar failures explicit before index mutation, migration and consumer compatibility verified. Publish source 0.9.0 or its approved successor before released consumers require it. | Publication and consumer qualification pending |
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
