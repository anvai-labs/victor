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
| CG-P2: Shared correctness fixes | TDD in existing parser/resolver/identity/delta suites. Preserve unresolved ambiguity, deletion equivalence, byte/line spans and stable IDs across checkout roots; deduplicate tests by invariant. Both consumers rerun pinned fixes. | CG-S01–CG-S05 qualified; AnvaiOps published-0.9.0 canonical rebuild passes; Sandesha rerun pending |
| CG-P3: Release/package qualification | Build/install exact wheels in supported Python/platform shapes; dependencies complete, missing-grammar failures explicit before index mutation, migration and consumer compatibility verified. Publish source 0.9.0 or its approved successor before released consumers require it. | Published 0.9.0 from `62552e965`; AnvaiOps API/worker installation and canonical store parity qualified; Sandesha pending |
| CG-P4: Service contract and lifecycle | Review existing Victor and AnvaiOps routes, then propose one versioned neutral contract with bounded snapshot jobs, explicit partial/error/cancelled states, generation-bound queries and pagination. FEP before new public framework API implementation. | Planned; no new service claim |
| CG-P5: UI and agent acceptance | Same snapshot IDs/edges/diagnostics in CLI, API and UI; source navigation, ambiguity display, filters, bounded expansion, accessibility and shared dark/light/system theme. Synthetic authorized/denied/cancel/restart journeys through headed AgentBrowser. | AnvaiOps bounded descriptive preview qualified locally; hosted SSO, lifecycle and scale acceptance remain open |
| CG-P6: Grounded repository Q&A | Real embedding/lexical retrieval selects authorized candidate symbols; canonical graph expansion and source citations substantiate answers. One immutable generation, typed evidence IDs and click-through navigation. FEP before new shared public APIs. | Planned after descriptive evidence/retrieval gates; hashed placeholder vectors are not semantic search |

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
The foundation corrections landed in AnvaiOps #488; #489 owns the canonical
consumer and descriptive explorer follow-up. Only minimized
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

**Release receipt:**
[`victor-codegraph-v0.9.0`](https://github.com/anvai-labs/victor/releases/tag/victor-codegraph-v0.9.0)
published from `62552e9659c320c6ad22e875d099d56f0b5c9727`; consumer pin is
`victor-codegraph[treesitter]==0.9.0`. Victor #1300 adds mandatory future-release
safeguards. Never republish 0.9.0 to change its artifacts. CG-S04 is a Victor runtime
correction on develop; this receipt does not claim a new Victor AI distribution.
AnvaiOps #488 is merged and #489 carries the bounded consumer follow-up. Sandesha,
CG-P4 lifecycle and hosted CG-P5 acceptance remain open.


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

### Consumer qualification and next customer journey — 2026-10-12

AnvaiOps adopted the published 0.9.0 distribution, removed its duplicate parser/
resolver and ID derivation, and preserved canonical relation endpoints, diagnostics,
status and source locations. Complete snapshots require reindex; no silent legacy
name-join path remains. Its commercial tracker is `docs/td/TD-F1-code-graph-service.md`;
the original handback and JSON remain the evidence owners.

| Evidence | Result / limit |
|---|---|
| Frozen private snapshot | 738 files, 7,647 canonical symbols and 35,428 semantic relation keys; exact compiler → consumer → store parity. Aggregate evidence only; no private source or graph export enters OSS. |
| Released storage runtime | ProximaDB 0.4.0, Linux ARM64 image `sha256:c24757088a29b996160f1535a4e92bf905f203bc76d41bb15b9c8aa281e79dad`; explicit SST run took 39.828 s to publish and 21.031 s to scan. Not percentile, durability or service-SLO acceptance. |
| Failed older runtime | Cached ProximaDB 0.2.2 rejected its page-two named-collection cursor. Failed evidence retained; 0.4.0 passes without consumer cursor rewriting. |
| Consumer checks | 1,732 API/package tests passed, 3 skipped; 153 console tests; two source→engine→HTTP replacement/empty-snapshot cases; API and worker image compilation/publication smokes. |
| Headed descriptive preview | AgentBrowser verified calls/imports/containment, source inspector and keyboard selection, zoom/fit/full screen, layered layouts, file/symbol drill-down and breadcrumbs back. Local synthetic identity; not hosted SSO or full authorization-lifecycle acceptance. |

Post-flush qualification found a separate released-engine defect: automatic
selection chose HELIX, whose record enumeration returned zero after flush
(ProximaDB TD-HELIX-2, still open in refreshed upstream). Failed evidence is
preserved. The consumer now explicitly creates SST collections and validates
existing engine/dimension metadata before writing. A confirmed SST flush retained
six synthetic graph records plus the manifest. All 43,075 full-snapshot records
also retained exact parity after a confirmed flush (25.277 s scan). Restart and
production digest rollout remain separate gates.

Victor's index/graph commands remain the shared indexing and graph-navigation
foundation. Commercial ingestion uses the same compiler identity/evidence contracts,
while retaining its own policy, hosted jobs, UI and metering. Different store/file
eligibility projections must report their scope rather than claiming equal counts.

The next useful customer flow is **question → candidate retrieval → canonical graph
traversal → grounded answer → clickable symbol/relationship citation**:

1. Reuse Victor's existing BGE baseline (`BAAI/bge-small-en-v1.5`, 384 dimensions)
   and embedding service through a qualified public seam. Local SQLite graph +
   LanceDB remains the default; Q1 will store qualified corresponding vectors in
   ProximaDB. `victor index` is semantic indexing; `victor graph index --embeddings`
   opts into graph-node embeddings. The standalone compiler itself does not run
   an embedding model. Version node text versus body chunks, preprocessing and
   query normalization: reuse stored vectors only when that complete contract
   and canonical symbol/source mapping match. No copied parser or new model
   implementation. Qualify retrieval against a lexical baseline. Record model/version,
   dimension, source/chunk hash, authorization and snapshot generation. Explicitly
   distinguish missing embeddings from a valid empty result; do not treat hashed
   placeholder vectors as semantic retrieval.
2. Retrieve only authorized candidates, then expand a bounded canonical graph in
   the same generation. Embedding similarity is evidence of relevance, never proof
   that one symbol calls another. Preserve ambiguity, partial parsing and no-answer
   states; prevent retrieved source instructions from granting tool permissions.
3. Return typed answer/evidence references tied to repository/commit, symbol IDs,
   source spans and relation endpoints. UI citations focus the graph and retain a
   path back to the answer. Measure grounded-answer precision on labeled questions.
4. Add reviewed structural metrics and only then prescriptive recommendations.
   Cite evidence, uncertainty, alternatives and expected impact. Preview changes
   and verify tests/outcomes before approved mutations; graph degree alone is not
   a design defect.

Remaining shared package work: add `py.typed` in a future release and qualify
non-Python call spans. Remaining commercial gates include repository policy before
retrieval, ordered publication, durable job/recovery/GC behavior, hosted OIDC,
large-canvas performance and theme/accessibility qualification. No new parser,
service implementation or recommendation engine is introduced by this roadmap.


### Shared indexing parity gate

The AnvaiOps handback exposed a boundary that was too narrow: sharing a parser
package alone did not ensure the consumer preserved graph projection or real
embedding semantics. AnvaiOps #489 removes the duplicate parser/identity path;
embedding reuse is the next shared contract, not a commercial reimplementation.

Before Q1 adoption, use one synthetic corpus and matched eligibility/configuration
to compare local SQLite/LanceDB and hosted ProximaDB adapters: canonical IDs,
embedding-input hashes, model revision, dimensions, preprocessing/normalization,
source spans, generation binding and deletion behavior. Test failures and missing
models explicitly; test actual BGE relevance separately from deterministic fixture
vectors. Release a neutral public seam after the FEP gate, keeping heavyweight
model dependencies optional. Consumers pin the release and run this conformance
suite before upgrading. AnvaiOps owns tenancy, policy, jobs, billing and UX;
neither those concerns nor storage choice justify different semantic identities.


### Embedding alignment release — 2026-10-12

Foundation consumer PR AnvaiOps #489 merged at
`54b43ccf3a3d8fa2a57602f2943045d830e66b37`; Victor receipt #1302 merged at
`37113d8f1db30872b790ca013dd0f215b3fa63b0`. Both had clean independent review
and all applicable CI green. Conditional skips are not full integration acceptance.

[FEP-0040](../../feps/fep-0040-codegraph-embedding-conformance.md) owns the next
shared contract and its precise evidence. The released package adds one optional
BGE encoder, canonical symbol-summary input, immutable model/profile identity,
source/input hashes and strict vector validation. Both storage adapters consumed
the same actual BGE evidence in a synthetic run (3 symbols; 8 published graph
records; 6.799 s); no private source was exported. Missing/zero/malformed embeddings
cannot become successful hashes. Semantic persistence rejects structural mixing,
unknown source hashes and incompatible tracked graph profiles.

| Ordered remaining gate | State |
|---|---|
| Shared document-vector contract and local conformance | ✅ #1303 merged `91bae704573ecd80dc5aa2919fb8ba4cb7cdccec`: 958 affected tests passed, 5 skipped; 198 standalone package tests (92.28% coverage); 1,703 AnvaiOps API/SDK tests, 3 skipped against the built 0.10.0 wheel |
| FEP acceptance | Accepted by the repository owner with an explicitly approved expedited review; see FEP-0040 |
| Immutable package release | ✅ [CodeGraph 0.10.0](https://github.com/anvai-labs/victor/releases/tag/victor-codegraph-v0.10.0), source `91bae704573ecd80dc5aa2919fb8ba4cb7cdccec`; PyPI publication and independent release tests/build passed. 0.9.0 remains immutable. AnvaiOps adoption #490 merged; hosted Q1 acceptance remains open |
| Collection/query model ownership and migration | Local LanceDB implementation: schema-bound profile, whole-table orphan/mixed marker checks, actual query model validation, float32 validation and atomic semantic upsert. Fresh collection required for migration. Local qualification and independent review passed; hosted adapters remain separate. |
| Semantic relevance and authorized generation-bound retrieval | Open; ingestion parity does not qualify ranking or Q&A |
| Consumer published pin + browser/hosted lifecycle | Published 0.10.0 adopted in AnvaiOps #490. Repository browser qualification in progress below; hosted lifecycle remains open |

Extend these gates here and in AnvaiOps TD-F1; do not create another parser,
embedding implementation or disconnected completion tracker.

Release evidence: exact-candidate independent review was clean; 41 applicable PR checks passed, including Vertical Py3.12. The first failed CI run is preserved; its correction installs the in-repo CodeGraph package before vertical consumers and replaces duplicate mocked provider checks with a real legacy-storage test. The final collection has 34,367 tests. Release run [38208368472](https://github.com/anvai-labs/victor/actions/runs/38208368472) passed. No new Victor AI release or hosted Q1 acceptance is implied.


### Repository audit and visible duplicate diagnostics — 2026-10-12

This extends CG-P1/P2/P5 using the actual Victor polyglot repository at baseline
`76c72e2eb32204c587485204f996543891905fed`. It is a candidate for CodeGraph
0.10.1, not a published-release or hosted-service acceptance receipt.

| Finding | Resolution / evidence |
|---|---|
| Misleading successful-looking partial index | Baseline had 19 failed files and two partial parses despite a green heading. CLI now distinguishes incomplete indexing, completed passes with coverage gaps, and clean passes. A no-op pass does not certify previously indexed files as clean. |
| Seven Python files with shadowed definitions | Removed duplicate runtime methods while preserving the last effective implementation. Consolidated test classes preserve unique assertions; task classification belongs in the existing analyzer suite. 604 affected Python tests passed. Repeated names in different scopes are not automatically defects. |
| TypeScript keyword confused with an object literal | Shared parser skips the unnamed `object` keyword node. Two valid inline-import generic fixtures use equivalent named aliases; the grammar limitation remains explicit and regression-tested. All 21 original problem files parse successfully with declared grammar dependencies. |
| Hidden legitimate sources | Root artifact globs no longer omit nested source `models/`, `deps/` and coverage modules. Rooted glob matching respects path segments. Full nested gitignore/negation semantics remain outside this fix. |
| Duplicates should not hide healthy graph evidence | Shared repository projection preserves a canonical file marker with `identity_collision` diagnostics for each declaration's line/column and omitted-symbol/relation counts. Ambiguous files are excluded from symbol and call resolution. Raw parse evidence remains available. |
| Unsafe downstream publication | Duplicate symbol/alias/output identities fail before embedding. Victor copies shared diagnostic metadata onto its canonical module node; AnvaiOps rejects unquarantined single-file collisions rather than inventing another resolver. |
| User-visible actionability | AnvaiOps candidate displays diagnostic badges on nodes and the symbol outline, with structured code/severity/message/source coordinates in the inspector. Missing evidence is unknown; explicit empty diagnostics means none reported. |

A one-file replay of the actual baseline `orchestrator_protocol_adapter.py`
retains one module, zero edges and six diagnostics for three conflicting pairs.
It completes with coverage gaps; it does not claim a clean audit. General
copy/paste similarity detection is not part of structural identity validation.

Qualification uses tree-sitter 0.26.0 and language-pack 1.21.3, not older
packages incidentally installed in a shared environment. The built 0.10.1 wheel
passes 209 standalone tests. Compiler upgrades require a **full rebuild**:
current incremental generation identity hashes source content, not compiler
version; unchanged source must not reuse a graph from an older compiler.

Local headed AgentBrowser qualification covers 36 representative combinations
(six languages, three relationship types, both directions), comparing rendered
edges to the SQLite snapshot. Empty resolved relations remain explicitly empty.
This is not exhaustive node/path coverage. The actual repository exceeds the
hosted API's current 20,000-symbol/60,000-edge scan caps; that API needs bounded
indexed queries before whole-repository hosted acceptance. No cap increase,
authentication bypass, cross-language FFI resolution or semantic Q&A is implied.

Final full rebuild with the installed candidate wheel processed **4,818 files**,
with **zero errors and zero partial parses**, producing **96,153 nodes / 157,160
edges** in 101.1 seconds (one local run, not a throughput benchmark). Baseline
evidence is preserved separately. Full test collection found 34,518 tests; Black,
Ruff and mypy (2,001 runtime files) passed.

Remaining gates: final browser evidence, reviewed PR/CI,
immutable package publication and consumer repin, then hosted query scaling,
OIDC/lifecycle and Q1 relevance. Keep private graph/source artifacts local.
