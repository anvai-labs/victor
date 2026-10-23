# Changelog

All notable changes to Victor are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.16.0] - 2026-10-23

### Added

- ADR-032 tier 3 covers **eight languages** (rev 1.9): new `scip-clang`
  (C/C++), `scip-java` (Java), `scip-ruby` (Ruby) and `scip-go` (Go)
  providers join scip-python / scip-typescript / rust-analyzer. Every
  producer grammar is pinned against a real captured dump; toolchain
  install matrix in the graph-RAG guide. Scala deferred with evidence (no
  verifiable emitter in scip-java 0.13.2 — sbt unsupported, Gradle
  instrumentation is JavaCompile-only).
- Tier 3 is drivable and observable from the CLI (ADR rev 1.8): the graph
  CLI/init/watcher resolve settings through `Settings.from_sources()` (the
  env-only snapshot silently ignored every yaml-configured graph option);
  `victor index graph index` gains `--precise/--no-precise`,
  `--precise-timeout` and `--list-partial`; `precise_index_timeout_seconds`
  is wired end to end; per-provider harvest outcomes and join accounting
  surface in the index summary, `victor index graph stats` and the watch
  manifest.
- FEP-0032: interrupted and crashed workflow runs resume at the right node.
- Canonical code-intelligence section in `docs/architecture.md` plus a
  tier-3 deck diagram (`d13`) for the VICTOR_AGENT_GUIDE presentation.

### Changed

- Layering round 2 (#1453–#1466): layering ratchet live (sanctioned pairs
  200 → 98; concrete upward graph edges 612 → 275), RL coordinator reached
  through a core port, conversation-store port, prompt assembly under
  `agent/prompting`, approval chain and policy engine in the runtime.
- ToolPipeline single-call path split into stages (#1472); teams
  coordinator formation run phased with pure delegate-contract builders
  (#1467/#1468 — Stage C item 30 complete); SSO login moved out of the
  workflow package (#1474); FileEditor unified behind one core
  implementation (#1447); embedding models behind a contracts runtime seam
  (#1449).
- CI: the ~1,320 integration tests no workflow ran now run, with the six
  rotted ones fixed (#1462); unit shards ~95% faster after dropping a
  per-test full `gc.collect()` (#1460); circuit-breaker isolation tests
  aligned with per-endpoint keying (#1471).

### Fixed

- victor-rag wheels now ship the live capabilities package (#1446);
  victor-coding's dead completion package removed (−2,628 lines, #1445);
  embedding revision resolved under transformers 5 (#1473); dead protocol
  tail deleted after victor's self-indexed self-analysis (#1451).
- Dependency maintenance: python (10 updates) and rust (3 updates) groups
  (#1434/#1443).

## [0.15.0] - 2026-10-22

### Added

- ADR-032 tier 3 is polyglot with dump caching (full detail in the 0.14.1
  unreleased block below, now released here): scip-python /
  scip-typescript / rust-analyzer providers with producer-owned grammars
  (python 143,558 compiler REFERENCES edges @ 76.9%, TypeScript 1,701 @
  67.4%, Rust 952 @ 89.2% on this monorepo); artifacts cached under
  `.victor/precise/` (unchanged corpus 642s → 54s; the python artifact is
  ~336 MB on monorepo corpora — no eviction, `rm -rf` busts); honest
  per-language accounting; provider-measured dump timeouts.

### Changed

- ADR-032 tier 3.5 (retrieval-first): graph expansion now ranks edges by
  provenance VALUE (compiler > scope/repository resolver > parser >
  unknown/absent) instead of mere provenance presence; the lazy neighbor
  cap sorts before cutting (store order used to drop high-precision edges
  that arrived late in a batch); the parallel traversal scorer consumes
  precision-ordered edges for tie behavior. The retrieval query cache now
  round-trips per-edge metadata — cache hits previously stripped
  provenance, silently degrading expansion order versus misses (legacy
  entries reconstruct with empty metadata, ranked conservatively).

### Added

- ADR-032 tier-3 dump artifacts now cache on disk (`.victor/precise/`):
  one artifact per provider keyed by tool identity + that language's file
  content hashes — an unchanged corpus reuses the (potentially
  minutes-scale) dump across harvests, and editing one language's files
  re-dumps only that language. Atomic, process-unique writes (a raced
  fixed tmp name once persisted a truncated artifact that parsed as a
  valid EMPTY index — now structurally impossible); a zero-byte artifact
  is never cached; a schema bump or `rm -rf .victor/precise` busts
  everything. Note: the python artifact is ~336 MB on a monorepo-scale
  corpus (no eviction yet — the tier is opt-in). Measured on this
  monorepo: an unchanged-corpus harvest drops 642s → 54s (~12x) with
  identical edge counts across runs.

### Changed

- Dump timeouts are provider-measured defaults (python/pyright 1200s,
  typescript 600s, rust-analyzer 120s) instead of one flat 1200s;
  `precise_index_timeout_seconds = 0` (new default) selects them, a
  positive value overrides uniformly.

- ADR-032 tier-3 harvest accounting is decomposed and honest: documents
  whose file has no codegraph node bucket are out-of-corpus-by-codegraph
  (counted separately, never as join failures, and their definitions no
  longer pollute the reference-evidence pool), and the unjoined metric
  splits into definitions-unjoined / target-unjoined references /
  ownerless (import/module-level) references. Per-language harvest
  outcomes and `GraphIndexStats` carry the decomposition. Corpus gate:
  python/TypeScript/Rust compiler-edge counts byte-identical (143,558 /
  1,701 / 952) — accounting only, no join-predicate change.

### Added

- ADR-032 tier 3 is now polyglot (contract rev 1.5 implemented): precise
  SCIP harvesting runs one provider per language present — scip-python
  (python), scip-typescript (TypeScript/JavaScript), rust-analyzer's
  native `scip` CLI (Rust) — each owning its toolchain detection, dump
  transport, and PRODUCER symbol grammar. Per-language outcomes ride the
  harvest (`per_language`: ok/edges, toolchain-absent, dump-failed,
  no-reader); a language without its toolchain skips honestly and never
  blocks the others. First polyglot corpus numbers (this monorepo):
  TypeScript 1,701 compiler REFERENCES edges at 67.4% join rate; Rust 952
  at 89.2%; python per the rev 1.4 dossier (143,558 at 76.9%).

### Fixed

- The SCIP kind map used values from a different proto revision than the
  vendored schema (3/4/5 vs the real Class=7/Method=26/Function=17/...):
  kind-populating producers (scip-typescript, rust-analyzer) would have
  filtered on garbage. Now pinned by test against the vendored enum.

- ADR-032 tier-3 dumps no longer OOM-crash on large corpora: scip-python
  is a Node process whose pyright pass holds the program in memory, and
  Node's default ~4 GB heap aborts around ~2,200 files (observed on the
  victor monorepo). The harvest subprocess now injects
  `NODE_OPTIONS=--max-old-space-size=16384` when the environment does not
  set `NODE_OPTIONS` itself.

- ADR-032 tier-3 symbol parsing now matches the REAL npm scip-python
  grammar (0.6.x package-qualified descriptors, backticked dotted module
  paths, `#` class members — verified against a live 0.6.6 dump; the
  previously assumed `python <file>\`<path>` backtick form is not what
  the tool emits, so real dumps joined nothing). References to symbols
  defined nowhere in the index (stdlib, site-packages) and
  non-symbol descriptors (parameters, locals, attribute fields, module
  inits) are no longer counted as join failures. First real-contact
  numbers on a victor graph_rag slice: 26 compiler REFERENCES edges, 90%
  join rate (was 0%).

## [0.14.1] - 2026-10-21

### Added

- CI duplication ratchet: a jscpd exact-clone scan over the code formats of
  `victor/`, `verticals/`, and `vscode-victor/src/` (shared scan config in
  `.jscpd.json`) now gates every develop PR — the duplicated-line count may
  not rise past `scripts/ci/duplication_baseline.json` (15,911 lines /
  541 clones at introduction). Deliberate dedup lowers the baseline in the
  same PR via `scripts/ci/duplication_ratchet.py --update-baseline`.

### Fixed

- ADR-032 tier-3 precise-index harvesting produced **zero edges on real
  data** and skipped with a misleading reason. The SCIP protobuf reader now
  uses vendored generated bindings (the toolchain is npm-only; the previously
  probed pip packages do not provide the schema), the occurrence join is
  metadata-based against persisted nodes (unique (file, name) match with
  scip-kind agreement and parent-derived scope chains — the old hash join
  mismatched the persisted key shape on every symbol), occurrence offsets
  read all three scip.proto range encodings, the dump timeout is
  configurable (default 1200s, up from 300s) with timeouts no longer
  mislabeled `toolchain-unavailable`, and ownerless references count as
  unjoined instead of being dropped silently.

## [0.14.0] - 2026-10-21

### Added

- ADR-032 tier 3: opt-in compiler-accurate REFERENCES edges harvested
  from an offline SCIP dump (`enable_precise_index`, default off;
  scip-python detected via path, PATH binary, or module). Harvested
  edges carry `provenance="compiler"` and run last so they dominate
  shared CALLS/INHERITS/IMPLEMENTS keys; absent toolchains degrade with
  one log line. Settings-file graph flags now actually reach every
  construction site (new `graph_flag` read-through; the flat-getattr
  reads previously ignored them).

### Changed

- Graduate ADR-032 tier-2.5 scope binding to default-ON
  (`enable_scope_binding` defaults True everywhere; explicit False opts
  out; victor-codegraph < 0.11.0 degrades via the capability probe).
  Indexing now persists resolved REFERENCES binding edges with resolver
  provenance by default. Evidence: 26.9% vs 21.3% resolution on the
  victor corpus, 12.6% vs 5.7% on proximaDB Rust (2.2x); force rebuild
  required to clear stale edges after an ON-to-OFF flip.

### Fixed

- Tool-calling capability overrides no longer silently reset four
  YAML-configured fields (`thinking_disable_prefix`,
  `exploration_multiplier`, `continuation_patience`,
  `timeout_multiplier`) to defaults when thinking mode or dynamic native
  support is detected — all five override sites now use
  `dataclasses.replace`.
- Gemini text-format parser: a `<ctrl42>`-tagged call no longer parses
  twice (the simple-call pass now skips spans the ctrl42 pass
  consumed), and parse warnings now reach the caller instead of being
  dropped on both empty-result paths — malformed tool-call JSON was
  previously invisible.
- Azure tool-calling adapter no longer normalizes empty warning lists
  to `None` (a latent AttributeError for the callers that append to
  parse-result warnings); warnings are always lists.

### Refactoring

- Deduplication program (jscpd-driven, 13 PRs, ~950 duplicated/dead
  lines removed): shared graph-seed accumulator for the vector-store
  impact/localization pair, one conversation core-table DDL,
  canonical MiddlewareProtocol and RL-runtime re-exports, shared
  benchmark CLI option factories, one gemini ctrl42/simple parse loop,
  and the four hand-rolled native tool-call parse loops unified onto
  `BaseToolCallingAdapter.parse_native_tool_calls` (invalid-name calls
  now kept unconditionally per the OpenAI error-response spec; warning
  text unified to `"{name}: Failed to parse JSON arguments"`).

## [0.13.1] - 2026-10-19

### Fixed

- Fix `FOREIGN KEY constraint failed` on graph node upserts whose emission
  order wrote a child before its parent within one batch: real-world Rust
  files emit impl members before their impl block (proximaDB
  `src/audit/mod.rs` failed to index in isolation), and the indexing
  pipeline's per-file retry cannot recover an ordering defect that lives
  inside the single file batch. `SqliteGraphStore` now clears parent links
  that resolve to neither a batch row nor a persisted node (logged, the node
  counterpart of the dangling-edge guard) and writes node batches
  parents-first, breaking the unorderable parent-cycle case (colliding node
  ids) minimally with one cleared, logged link.

### Added

- Budget-aware compaction: cumulative session prefill volume (real provider
  `prompt_tokens` accumulated per response) crossing
  `max_session_prefill_tokens` (default 400,000; `None` disables) forces
  compaction at the next user-turn boundary regardless of context-window
  utilization. Un-compacted history resend amplification is what burns token
  budgets (measured 2026-10-18: 813k tokens across 24 calls of one turn);
  prefix caching keeps the trigger quiet for cache-stable sessions.

### Changed

- Enforce the KV prefix-stability invariant: within a user turn the
  provider-bound payload is now append-only (byte-extension between
  consecutive calls), so provider prefix caching holds. The session-ledger
  render and the composed turn prefix freeze per user turn; mid-loop failure
  hints append as hidden messages instead of re-rendering sent bytes; the
  tool-result dedup view stubs the newer duplicate instead of rewriting the
  older already-sent message; older-message selection is sticky per turn;
  and background/inline compaction defer to turn boundaries (overflow
  overrides). Observability: `[kv-cache] call=N … extended=… boundary=…`
  per call, `VICTOR_KV_PREFIX_DEBUG=1` for per-message hash chains; the
  invariant is enforced by a hard-assert guard test
  (docs/architecture/kv-prefix-stability.md).
- `max_conversation_history` default 100 → 2000 (single source of truth in
  `victor/config/conversation_settings.py`; was three conflicting defaults).
  A firing history-count trim now logs a sanctioned-boundary line.

## [0.13.0] - release candidate

### Fixed

- Patch the Rust workspace TLS dependency to Rustls 0.23.46 for
  RUSTSEC-2026-0346–0349; this path is used by the edge runtime.
- Preserve failed, paused, cancelled and exhausted outcomes across buffered and
  streaming execution, CLI exits, planning dependencies and response synthesis
  (#1306, #1307, #1336–#1349). Independent plan branches can continue; failed
  prerequisites cannot silently become satisfied.
- Serialize chat replacement, resume and teardown; preserve selected profiles and
  persisted conversation continuity. Ship the shared Victor chat theme and
  packaged workflow graph assets (#1271, #1348, #1350, #1361).
- Keep project instructions in the stable prompt prefix, preserve enabled guidance
  during deduplication, and account for the migrated buffered, streaming, recovery
  and planning model calls (#1353, #1355–#1357). Optional judges, LLM compaction,
  resilience helpers and legacy continuation remain outside that qualification.
  Prompt experiments are measured separately from provider token accounting.
- Settle embedding initialization before publishing model/collection state; retain
  edit undo history and invalidate stale verification reads (#1358–#1360).
- Preserve vector eligibility and exact deletion, settle graph cancellation before
  connection reuse, run scoped atomic schema migrations, and publish WAL-aware
  SQLite backups (#1362–#1366).
- Qualify selected source inventories before force cleanup and bound parser/stream
  ownership (#1367–#1369). Atomic generation publication, cross-store reconciliation
  and cross-process snapshot acceptance remain open; preflight is not a rollback
  guarantee after writes start.
- Preserve bound backend receipts without replay and retain failed member checkpoint
  evidence (#1255, #1258, #1260). Full durable continuation and mixed-team C5
  acceptance remain separate from this release.

### Changed

- Give the refreshed VS Code extension its own 0.5.2 artifact version so it can
  be distinguished from the 0.5.1 VSIX bundled with Victor 0.12.0.

- Prepare the five extracted first-party verticals at suite version 0.13.0;
  their separate publication gate is required for the coding extra update.
- Align the optional ProximaDB SDK extra in core and victor-coding on
  `proximadb>=0.4.1,<0.5`. The native wheel remains a separate optional
  installation; installing it does not replace the managed-server adapter.

- Prepare Victor 0.13.0 with contracts 0.10.0 (Python 3.12+ and packaged typing
  markers) and released `victor-codegraph[treesitter]==0.10.3`. CodeGraph owns the
  shared polyglot parser, ambiguity diagnostics and selected-input contract.
- Require `sandhi-gateway==0.14.0`, published and artifact-verified upstream. It
  retains private-CA support and brings completed Codex tool arguments, bounded
  stream admission and an opt-in allowance calculator; live weighted enforcement
  is not included. The network gateway remains optional.
- Textual is an explicit `tui` extra. Keep the core installation independent of
  optional UI/embedding accelerators; qualify each installed deployment shape.

### Deprecated
- **`FrameworkShim` compatibility surface**, **`TeamNode*` workflow
  compatibility aliases**, and the **`WorkflowGraph` alias from
  `victor.workflows.graph`** remain warning-backed shims — full
  deprecation inventory (replacement paths and target removal dates) in
  the 0.7.0 entries below.

## [0.12.0] - 2026-10-04

- Publish Linux x86_64 standalone binaries with per-artifact smoke tests, portable
  Windows secure paths, and frozen-binary dynamic import fixes (#1231, #1233, #1234).
- Make NumPy optional for the core runtime and add a clean bare-install artifact
  gate (#1224, #1225, #1227, #1230, #1238, #1239).
- Recommend `uv` for isolated CLI installation (#1226). Historical 0.12.0 package
  metadata pins Sandhi 0.10.1; the tap's later documented 0.11.0 private-CA override
  does not rewrite that published metadata.

## [0.11.0] - 2026-10-02

### Added

- Add FEP-0037: `POST /v1/classify` on `victor serve` — one structured
  completion with a pinned contract, bearer auth and per-request
  provider/model/endpoint overrides; additive request fields, explicit
  default-provider fallback and 422-vs-502 triage for policy vs
  infrastructure failures (#1196, #1197, #1213).
- Preserve interrupted tool batch evidence and block unsafe resume; batch
  publication requires complete evidence to fail closed (#1208).
- Add the `ProviderPolicyError` class so policy denials surface distinctly
  from connection and timeout failures.

### Changed

- Route chat planning, task guidance, and tool selection through explicit
  runtime capabilities; route streaming governance, completion, tool-call
  processing, conversation history/accounting, and outcome feedback through
  the enumerated runtime view; and move setup, reporting, and teardown into the
  service-owned chat turn frame.
- Preserve configured request/response governance during runtime bootstrap and
  reject invalid configured-gate results instead of treating them as disabled.
- Preserve gateway identity and terminal policy decisions across the streaming
  path; harden the sandhi transport's TLS trust diagnosis and classify TLS
  trust failures distinctly (#1203, #1205).
- Adopt the Sandhi 0.10.1 accounting contract (`sandhi-gateway`
  0.7.0 → 0.10.1) across the proxy and all requirement files (#1206).
- Withhold denied tool results without replay so denied calls cannot be
  re-executed by a resumed turn (#1204).
- Haskell language plugin: emit the class-declaration query spelling that the
  installed grammar era supports (the upstream `type_synomym` rename) so
  query compilation holds across grammar wheels.

### Fixed

- Finalize buffered task reports on cancellation before turn teardown releases
  the shared chat lock.
- Close nested streaming generators before releasing per-turn locks, metrics,
  and task reports so early client disconnects and cancellations cannot leak
  one turn's state into the next.
- Restore the defensive orchestrator probe in the streaming gateway
  rate-limit check — the direct-access form broke every rate-limit retry
  harness — and ratchet the streaming boundary registry to the reviewed
  counts (probes 14, attributes 56) with the justification recorded
  in-registry (#1215, #1211).

## [0.10.0] - 2026-09-25

- Require Python 3.12+ across core, contracts, codegraph, native extensions and
  verticals. CI tests 3.12/3.13; Python 3.10/3.11 installations must upgrade
  before installing the next release. Previously published releases are unchanged.
- Headless and benchmark turns now use the shared per-turn tool-selection
  runtime (FEP-0034 Stage C), including capability/Q&A gates, intent projection,
  and KV policy. Curated schemas remain stable, pruning remains opt-in, and
  supply traces also cover curated early returns.
- Benchmark traces identify the tool-supply cutover with
  `completion_signals.tool_supply_pipeline=fep-0034-stage-c`.
- Pin the development interpreter to Python 3.12 via `.python-version`.

### Added

- Add opt-in `victor review` and framework review panels with strict required-reviewer
  verdicts, no reviewer tools, bounded input and commit-bound PR provenance. Results
  are advisory; they do not authorize merges or prove model-review accuracy.

- Add 77 code-search tests covering filters, cache invalidation, index integrity
  and recovery, literal retrieval, and semantic dispatch/fallback without model
  downloads or a live vector database.
### Fixed

- Permit the exact Dependabot co-author trailer in the attribution gate while
  retaining checks for altered identities, appended text and other bot trailers.

- Preserve full-path CLI suggestions with Typer's vendored Click runtime and
  keep unrelated usage errors on their native exception path.
- Resolve tool-pruning settings at each policy call and isolate team goals in
  per-execution context so concurrent and hierarchical runs cannot overwrite
  one another's assignments.

The historical `v0.10.0` release was published on 2026-09-26; its artifacts remain unchanged.

### Added

- Add the demand-loaded `gh` tool for GitHub pull requests, issues, workflow
  runs, releases, repository queries and API calls.
- Add `victor mcp add` to verify and register project or user-level stdio MCP
  servers with restricted configuration permissions.

### Changed

- Make InferFlux with `qwen3-coder-30b` the default provider while retaining
  Ollama through the `local` profile and stock llama-server through
  `local-llamacpp`. This is the release's primary breaking default change.
- Advance `sandhi-gateway` to 0.7.0 and recognize chat-contract minor 8,
  preserving reasoning-accounting semantics through the current consumer API.
- Complete FEP-0034 tool-supply consolidation: demand hydration, stable tool
  definitions, member-scoped sessions and per-turn supply traces now align
  served chat, headless and benchmark execution.
- Mount the TUI before background session initialization, keep its prompt live
  during turns with a FIFO queue, and make terminal-native mouse selection the
  default.
- Move the supported extension toolchain to Node 24, TypeScript 6, ESLint 10,
  Vitest 5 and Vite 8. Required CI validates the extension only when relevant
  paths change and reuses the existing aggregate runner job.
- Regenerate the core, API and CPU-embedding deployment locks under shared
  constraints and enforce every lock against project metadata in required CI.
- Require Python 3.12+ for this release candidate and run the primary Python CI
  matrix on 3.12 and 3.13; Python 3.11 wheels are not part of this candidate.
- Raise the default bash command timeout from 60 seconds to 120 seconds and
  route GitHub operations through the dedicated `gh` tool rather than `git pr`.
- Retain existing Victor AI compatibility shims through this release and move
  their announced removal target to 0.11.0. Victor Contracts remains on its
  independent version train and keeps its existing stability schedule.

### Fixed

- Preserve Sandhi's reasoning-inclusion convention and per-call folding through
  provider responses, streaming metrics and session pricing while retaining raw
  completion counts, totals and timing provenance.
- Respect explicit curated tool sets before pruning and demand hydration;
  unavailable curated tools no longer expand to the full registry.
- Isolate concurrent team-member sessions, preserve member goals, and restore
  caller context across streaming yields, early termination and timeout cleanup.
- Bound coding-sandbox startup when Docker is unavailable, remove hardcoded
  provider lookups, and initialize the optional skill matcher outside the
  startup critical path.
- Terminate and reap owned MCP stdio process groups across normal close,
  timeout, cancellation and failed initialization.
- Report `edit(commit=False)` as unapplied instead of returning a misleading
  success result.

## [0.9.5] - 2026-09-18

### Fixed

- Preserve Sandhi's boolean reasoning-inclusion convention through response and
  stream models; fold separate reasoning into billable output once per call before
  aggregation and session pricing.
- Retain raw completion counts alongside billable completion counts, preserve
  reported totals, and include separate reasoning when deriving missing totals.
- Preserve independent origin/boundary timing provenance through transport and
  canonical metrics; reject invalid timing values and stale provenance.

### Changed

- Pin `sandhi-gateway==0.7.0` in package metadata and deployment snapshots and
  recognize chat contract minor 8.
- Keep this release focused on the consumer-accounting correction. The broader
  0.10.0 develop features remain outside the v0.9.5 artifacts.

## [0.9.4] - 2026-09-16

### Security

- Fail closed when approval, safety, sandbox startup or verification setup fails.
  Apply temporary runtime budgets transactionally and reject session reuse if
  restoration fails; preserve inherited tool restrictions.
- Keep SDK vertical validation active under optimized Python, and use SHA-256
  for undo integrity checks while validating legacy histories against their text.

- Bind table-description parameters, quote generated SQLite identifiers, and
  validate update columns, metric groups and learner prefixes before SQL execution.
- Restrict debug/workflow text conditions to bounded plain-data expressions and
  reject coverage XML entities and oversized reports.
- Confine Composer edits to workspace paths, allowlist Git context commands and
  pass terminal working directories through the editor API.
- Launch the VS Code backend with separate executable/argument values, without
  shell interpolation. Render temporary D2 diagrams inside a private directory.
- Replace diskcache and persisted Python-object caches with versioned data-only
  storage. Legacy files are rebuilt rather than deserialized; malformed snapshots
  are rejected before replacing live state.
- Update vulnerable JavaScript and Rust dependencies, patched Python build floors
  and CPU embedding constraints. Remove unused GitPython and Python codecov
  dependencies, and omit pip from runtime containers.
- Calculate real CVSS base scores, retain unknown ratings, follow all OSV pages,
  and reject incomplete advisory/manifest results. Query caches now include exact
  package versions; legacy or malformed offline coverage cannot clear a scan.
- Reject unsupported MCP isolation policies before launch, and abort child exec
  if resource limits or root user demotion fail. Use an external sandbox/container
  for filesystem, network, namespace or seccomp restrictions.
- Match complete OS package versions in Trivy inventories, including release
  and epoch fields, so valid image reports retain their actual blocking findings.
- Preserve complete scan coverage for separately resolved core, API and CPU
  embedding deployments. Unaccepted container high/critical findings still block
  publication. Move runtime images to patched Ubuntu 24.04 packages; lower-severity
  unfixed OS findings remain visible in complete image reports.

### Changed

- Replace the workstation requirements freeze with manifest-derived deployment
  snapshots, remove unused vertical NumPy requirements, and keep RAG storage
  exports lazy during package discovery.
- Consolidate containers into core, MCP, native and full targets with installed
  wheels and separate dependency/build stages. Correct obsolete build paths and
  remove unsupported MCP environment-variable examples.
- Update security/dependency documentation and trim development artifacts from
  the VSIX. The Svelte webview uses the supported mount entrypoint.

### Fixed

- Validate generated SQL backup identifiers and preserve bound candidate values.
- Give the coding native wheel its own package metadata so installing it cannot
  replace the Python coding package.
- Report the installed package version from HTTP and GraphQL health metadata.

Victor AI 0.9.4 was published on 2026-09-16 with Python packages,
native wheels, standalone binaries, checksums, SBOM, Docker image and a GitHub
Release. `victor-contracts` 0.9.2 followed on 2026-09-17. See the
[security batch record](docs/development/security-remediation-0.9.4.md) for the
validated deployment shapes and retained historical-risk evidence.

## [0.9.3] - 2026-09-10

### Security

- Raise the GitPython runtime minimum and requirements pin to 3.1.59 to address
  CVE-2026-78676, which failed the blocking dependency scan on develop (#1050).

### Changed

- Bind chat task requirements to the existing session state owner through
  `ChatRuntimeServices.session`, preserving live state across resets and restores
  (#1049).
- Route streaming chunk generation, sanitization, terminal recovery and garbage
  detection through `ChatRuntimeServices.delivery`, using the existing components
  and preserving chunk metadata, error propagation and policy ordering (#1051).
- Lower chat-runtime boundary ratchets and guard against reintroducing direct
  chunk-generator or sanitizer lookups. These are partial FEP-0031 phase-1 slices;
  turn-frame ownership and the remaining Stage C work are still pending.

Public agent APIs are unchanged. `victor-contracts` remains 0.9.1 and
`victor-native` remains 0.8.0 on their independent release trains.

### Fixed

- Drain subprocess pipes before reaping after output caps, timeouts, cancellation
  or callback errors, preventing full pipe buffers from hanging tool execution.
  Execution timeouts also cover commands that close their output pipes early.

## [0.9.2] - 2026-09-08

### Changed

- Consolidated definition-based workflow execution and streaming onto the single
  `CompiledGraph` engine (ADR-030, #1041–#1043). `WorkflowExecutor` and
  `CompiledWorkflowExecutor` remain import aliases for `StateGraphWorkflowExecutor`.
- Consolidated documentation, replaced stale API examples, refreshed architecture
  diagrams, and validated the GitHub Pages build with the pinned MkDocs toolchain
  and an advisory local link/anchor checker. FEP-0031/0032/0033 diagrams describe
  future ownership and resume changes, not completed implementations.
- Made native/Python parity part of the required CI Success aggregate (#1038).
  `victor-contracts` stays 0.9.1 and `victor-native` stays 0.8.0 on independent
  release trains; this release does not change their source or dependency pins.

### Fixed

- Preserved workflow final state, per-node diagnostics and tool counts across the
  adapter, API and YAML paths. Node and parallel failures now propagate correctly;
  condition routing, compute results and parallel child accounting are aligned.
- Closing or cancelling a workflow stream now cancels and awaits its downstream
  execution. Concurrent runs keep lifecycle observations separate, and configured
  workflow timeouts and checkpointers reach the canonical engine.
- Moved SQLite query-result materialization onto the dedicated store worker (#1039)
  and routed integration agent creation through the framework boundary (#1040).

### Migration notes

This patch includes removals and behavior changes for legacy workflow callers.

- Execute a loaded `WorkflowDefinition` with
  `execute(definition, initial_context=...)`. Executor-only `execute_by_name`,
  cache-stat methods and private BFS helpers are removed. Standalone workflow
  definition/cache infrastructure remains available.
- Legacy executor `cache`/`cache_config`, workflow
  `metadata["continue_on_failure"] = True`, and definition execution with a legacy
  checkpoint ID now fail explicitly. Use a graph checkpointer and `thread_id`;
  old BFS checkpoint payloads are not silently migrated. Handle recoverable
  errors explicitly inside nodes.
- Failed graph streams raise after any earlier successful updates. Handle the
  terminal exception and close the iterator when stopping consumption early.
- Ordinary multi-successor DAGs use breadth-first traversal with shared-descendant
  deduplication and a persisted pending frontier. Combining ordinary fan-out with
  cycles or dynamic `Send` is unsupported. Node-based replay and explicit start-node
  overrides reject sequential-frontier checkpoints; ordinary invocation resumes
  them. Existing single-path cycles and `Send` graphs retain their semantics.
- The deprecated `StreamingChatExecutor.run()` alias and unused
  `AgenticLoop.stream_chat()` wrapper are removed. The canonical chat stream is
  `ServiceStreamingRuntime` → `run_unified()` → `AgenticLoop.run_streaming()`.
- The presence of pause metadata does not implement FEP-0032's general
  interrupt-after/resume contract. Chat runtime inversion, graph resume redesign,
  RL relocation and the remaining Stage C work are outside this release.

## [0.9.1] - 2026-09-07

### Changed

- Published the co-design Waves 1–2 and Wave 3 Stage A implementation, including
  boundary guards, runtime and storage fixes, exact native token counting, and
  removal of unused modules. Stage B added the subsequent execution designs.
- The [Victor AI 0.9.1 release](https://github.com/anvai-labs/victor/releases/tag/v0.9.1)
  published on 2026-09-07. Contracts 0.9.1 followed independently via `sdk-v0.9.1`.
  Post-release #1038–#1043 workflow, integration and CI changes belong to 0.9.2.

## [Rust crates 0.8.0] - 2026-08-21

### Added
- First crates.io release of the modular Rust runtime: `victor-protocol`, `victor-state`,
  `victor-tools`, and the standalone `victor-edge` runtime and binary.
- Independent `rust-vX.Y.Z` release tags with ordered, idempotent crates.io publication.
- Public-registry verification through a clean `cargo install victor-edge` smoke test.

### Security
- Removed the native-tls/OpenSSL dependency chain from `victor-edge` and upgraded PyO3, LRU,
  and rustls-webpki to non-vulnerable releases before publication.

## [0.9.0] - 2026-08-16

### Changed
- **Dependency-chain bump (release train)**: `sandhi-gateway==0.1.6` (was
  `0.1.5`) and `proximadb>=0.3,<0.4` (was `>=0.2,<0.3`) across victor-ai and the
  victor-coding vertical; the `requirements.txt` proximaDB source pin advances
  from a floating SHA to the released `v0.3.0` tag. Completes the
  sandhi 0.1.6 → proximaDB 0.3.0 → victor 0.9.0 dependency-chain releases
  (proximaDB 0.3.0 brings pgwire vector search, DuckDB OLAP routing, DV-aware
  SST compaction, relational ABAC; the SDK relaxes collection-name validation
  and defaults ADR-044 canonical oids on).

### Added
- **Effect-grounded completion gate (ADR-010 / EVR-4, opt-in)**: COMPLETE is now
  conditional on a verifiable effect — a workspace delta or verified check
  recorded in the session's tool results, or (for Q&A/no-mutation tasks) a
  grounded claim. An ungrounded COMPLETE downgrades to RETRY (reason
  `completion-without-effect`; bounded by `max_downgrades=2`, then
  annotate-and-allow). New `victor/framework/effect_gate.py` wraps the single
  EVALUATE seam so the gate preconditions every completion strategy. Enable via
  `AgentSettings.effect_gated_completion` or `VICTOR_EFFECT_GATED_COMPLETION`
  (default off per the flag-graduation policy; strict no-op when disabled).
  Measurement: `TrajectoryDimension.EFFECT_GROUNDING` + `EffectGroundingScorer`
  (excluded from `default_scorers()` to keep battery aggregates stable).

### Removed
- **FEP-0015 Phase 2 — deprecation shim removed**: the `__getattr__` shim in
  `victor/framework/step_handlers.py` that kept the unexported `ExtensionHandler`
  and `CapabilityConfigStepHandler` names importable with a `DeprecationWarning`
  is gone; the one-minor-release deprecation window after Phase 1 has elapsed.
  `from victor.framework.step_handlers import ExtensionHandler` now raises
  `ImportError` — depend on the public `StepHandlerRegistry` /
  `ExtensionsStepHandler` surface instead.

## [0.7.5] - 2026-07-12

Maintenance + capability release: completes the Graph-RAG cross-language import
resolver set, wires two latent-but-inert subsystems, and removes a large body of
verified-dead code found by the F-016 call-graph audit.

### Added
- **Graph-RAG import resolution — full language set**: JS/TS (#485), C/C++
  `#include` (#491), and Java/Scala (#496) resolvers, plus Rust `use` resolution
  and Martin-coupling computed from `IMPORTS` edges (#481). Eager provider
  warm-up on graph build (#485).

### Changed
- **FEP-0016 implemented**: `InitializationPhaseManager` now drives all 9
  orchestrator `_initialize_*` phases via `run_phase`; a guard test prevents a
  phase being silently lost (F-002, #477/#478).
- **Credit-assignment feedback loop wired** (F-016f): `assign_turn_credit` now
  fires at the universal per-turn teardown, so the online tool-guidance loop
  actually runs when credit tracking is enabled (#505; lazy-bootstrap guard
  #509). Off by default.
- `tool_selection` promoted to a package with a size ratchet (F-004, #484).
- Test suites run under 4 pytest-xdist workers; `lang-all` grammar wheels
  included in dev/ci extras (#482, #483).

### Fixed
- Generic-result-cache flag wiring (read the nested value); dropped dead
  `deduplication_strict_mode` + provider-optimization config (F-016e, #501).
- Two cross-test state-pollution defects in sync/async + DB-path plumbing (#490).
- Error-path log content truncation (TD-20 residual, #502).
- `victor-coding` wheel now ships all nested subpackages (#495).
- Narrowed the Laravel storage graph-exclude to specific subdirs (#479).

### Removed
- **F-016 call-graph audit — verified-dead code** (all independently
  re-verified before deletion): orphaned hybrid-compaction subsystem
  (~2,082 LOC, F-015 #487); `HybridDecisionService` + `ExtendedModelSelectorLearner`
  cluster (~4,325 LOC, #493) and its `Extended*Learner` siblings (#500);
  `WorkflowOptimizationComponents` bundle (#494); unwired team credit-attribution
  mixins (#489); unregistered capability step-handlers (#492); orphaned
  post-switch hooks subsystem (#503); the dead `create_step_aware_selector`
  factory, inert provider pool, 8 unread config fields, and the never-supplied
  `ProviderService.health_checker` (F-016j/k/l + h, #508).

## [0.7.3] - 2026-07-05

Patch release: makes the public PyPI install actually work end-to-end.

### Fixed
- **Public install**: `victor-contracts` 0.7.2 is now published to PyPI (Trusted
  Publishing via `release-contracts.yml` + `pypi` environment), so
  `pip install victor-ai` resolves again (#380, #393)
- **Clean-env startup**: declared `packaging` and `numpy` as runtime deps —
  both are imported on the `import victor` / CLI path but were previously only
  satisfied transitively by dev tooling; a bare install crashed at startup (#384)
- **scipy off the CLI path**: `experiments/ab_testing` imports are guarded;
  A/B statistical analysis raises an actionable `pip install "victor-ai[ml]"`
  error instead of breaking CLI startup (#384)
- Stale test-collection class: full-suite `--collect-only` gate on every
  develop PR; quick-tests treats pytest exit 5 (all skipped) as pass (#382)
- `victor-codegraph-v*` tags no longer trigger the victor-ai Release workflow

### Changed
- **`import victor` is lazy (PEP 562)** — drops from importing the whole
  runtime to ~20ms/2 modules; `victor.agent` is a callable module so
  `@victor.agent` and `mock.patch("victor.agent...")` keep working (#390)
- `docker` moved from default dependencies to the `[docker]` extra; all import
  sites already degrade via `DOCKER_AVAILABLE` (#384)
- Deprecation migrations completed internally: no in-repo importers of
  `victor.tools.base` registry re-exports or `victor.workflows.executor`
  remain (shims stay until 0.10.0); `CHAIN_HANDLER_PREFIX` re-homed to
  `victor.workflows.compute_registry` (#387)
- One documented dev-install path: `make install-dev` /
  `pip install -e ./victor-contracts -e ".[dev]"` across README + docs (#383)
- victor-contracts is importable standalone (host-runtime workflow helpers
  resolve lazily; excluded from `__all__` for star-import safety) (#380)


## [0.7.2] - TBD

### ⚠️ Breaking Changes
- **`StreamingChatExecutor.run()` deprecated** — The legacy entry point is now a thin alias to `run_unified()` with `DeprecationWarning` (target removal: 0.8.0). Migrate to `run_unified()` or `AgenticLoop.run_streaming()`. (#199)
- **ToolCategory enum consolidation** — Single authority in `victor.tools.enums`; YAML-derived category views are now validated, not authoritive. (#238)
- **pattern_registry.TaskType collapsed** — 42-member enum migrated to canonical registry; 6 real orphans overridden. (#240)

### Added
- **FEP-0007: Streaming Loop Extraction** — Complete decomposition of `StreamingChatExecutor.run()` into explicit per-turn primitives:
  - Per-turn ACT sub-steps: `_stream_provider_turn`, `_execute_tools_turn` (#180, #181)
  - Per-turn EVALUATE bands: `_evaluate_provider_turn_stops`, `_evaluate_post_tool_turn` (#182, #183)
  - Turn lifecycle: `_extract_task_requirements`, `_apply_run_guidance`, `_initialize_task_intent` (#176-#179)
  - Decision points: `_detect_task_completion_and_mentions`, `_apply_turn_recovery`, `_emit_assistant_turn`, `_handle_continuation_decision` (#184, #185, #187, #189, #190)
  - Single `_stream_turn()` primitive assembling the full per-turn body (#192)
  - Streaming parity battery (14 baseline byte-identical tests) (#186, #188)
  - `AgenticLoop.run_streaming()` over the streaming ACT seam (#195-#197)
  - FEP-0007 cutover: unified streaming onto PPAED loop + legacy removal (#199)

- **Evaluation-Centric Runtime (FEP-0008 + ADR-009-012)**:
  - EVR-1: Trajectory-level evaluation harness (#201)
  - EVR-2: LLM-judge reliability gating (ADR-011) (#202)
  - EVR-3: Rubric-based completion evaluator machinery (ADR-009) (#203, #216, #217)
  - ADR-009: LLM rubric judge + judge-eval harness, completion_strategy threading, A/B harnesses (#216, #217, #219, #223)
  - Vision P6: RL-from-traces with credit-assignment framework (#222)
  - Correlation spine joining offered/invoked/resulted by session+turn (R1) (#224)
  - One canonical default reward; deleted inline reward path (R2) (#225)
  - ToolExperienceStore as projection of durable outcomes (R3) (#226)
  - RL Q-value as sole tool-success signal at Gate 1 (R4) (#227)

- **ADR-013: Unified Temperature Policy**:
  - Single composed TemperatureResolver from 6 scattered paths (Source/Modifier Protocols) (#206)
  - TemperatureSettings group + ProfileConfig.temperatures + settings adapter (#207)
  - Task-hint-aware routing with profile›settings›task-hint›base›0.6 precedence (#208)
  - Streaming cutover + parity regression fixes (#209)
  - Spin-escape ratchet (step 0.05, cap 0.9) + AST boundary guard (#210, #213)
  - Default temperature flip 0.7→0.6 (#243)
  - Recovery ramps folded into shared escalate_temperature (#215)
  - Temperature A/B harness (#214)

- **Tool Supply & RL/Trace Consolidation** (16 PRs: #220-#240):
  - Per-turn tool-supply telemetry (duration, real-output, pruning, follow-ups) (#220)
  - ToolSupplyProfile provider-economics resolver (#228)
  - Cap demotes to STUB instead of dropping tools (#229)
  - Q&A necessity gate returns read-core, not None (#230)
  - Additive session_stable KV tool set (#231)
  - Capability-first prompt-tool split (#232)
  - Single resolve_contract() metadata authority (#234)
  - ACT selection pipeline as explicit ordered stages, on by default (#237)
  - TaskType adapters collapsed onto canonical registry (#236)
  - ToolCategory enum + YAML onto one vocabulary authority (#238)
  - pattern_registry.TaskType collapsed onto canonical registry (#240)

- **FEP-0009: SDK Tool Contract**:
  - Phase 1: `victor_contracts.tools` with ToolContract + trait enums (#245)
  - Phase 2: Duck-typed `resolve_contract()` bridge (version-skew-safe) (#246)
  - Dev story: editable-install local victor-contracts first (#247)
  - Retargeted to 0.7.1 release (#248)

- **ProximaDB CodeGraph Backend (TD-11/12/13)**:
  - Per-repo flag for ProximaDB Code Context Graph backend (#249)
  - Unified ProximaDB correlated collection — vectors + graph on one oid (#250)
  - Optional `proximadb` extra (`proximadb-python>=0.2,<0.3`) (#212)

- **Vertical Tools Extraction**:
  - Core/vertical boundary enforcement via capability_loader (#253)
  - Vertical imports routed through sanctioned pattern; boundary guard test (5/5 pass)
  - ConversationSession extraction + workflow escape hatch purification (#251, #252)

- **Provider Enhancements**:
  - GLM-5.2 registration with context_window() override (fix 128K freeze) (#254)
  - Config-driven provider context windows via `provider_context_limits.yaml` (#255)
  - Z.AI API + KV prefix caching support (#256)
  - Provider cache audit contracts (#257)

- **Tool Improvements**:
  - Grouped command tools (shell-style, standardized args, runtime wiring) (#258-#260)
  - AI attribution logic for commits/PRs + tool folding (#261)
  - Context-aware tool supply stabilization (#258)
  - Lazy load specialty tool schemas (#264)
  - Write-intent guards + action signaling on core tools (#266)
  - STABLE_TOOL_ORDER priority ranking for all 54 session tools (#267)
  - Fold thin CLI wrappers into shell + canonical cwd parameter (#268)

- **Shell & Readonly Hardening**:
  - Expanded readonly validation (bare flags, docker compose, npm, pipdeptree, terraform) (#269)
  - Readonly shell command validation hardening (#270)
  - Browser render mode for web fetch (#271)

- **Chat-UX Polish**:
  - Victor console rendering polish (#265)

- **Auth & Config**:
  - Unified accounts+profiles list, show & env commands, status leak fixes (#168)
  - Convergence-guard thresholds surfaced as ExplorationSettings (#175)
  - Shared TurnEvaluationController — dedupe per-turn guards across agentic loops (#170)
  - Convergence follow-on: FEP-0007, plateau, search-novelty, fulfillment-complete (#173)

- **Docs & CI**:
  - Auto-generate API reference (mkdocstrings) + single-source version stamp (#169)
  - Overview-first tool steering + large-file skeleton hint (#172)
  - Docs-drift gate to ci-fast (#167)
  - Black aligned to 26.3.1 (CI + pre-commit) (#242)
  - ProximaDB Code Context Graph backend documentation (#233)

- **Deps**:
  - typer>=0.15 for click>=8.2 compatibility (#211)

### Fixed
- **Streaming fulfillment criteria** — Now built via real builder API instead of dead code (#184)
- **Tool budget** — Made honestly advisory; removed dead hard-stop (#191)
- **ProximaDB provider lifecycle** — Logs instead of stdout prints (#218)
- **ProximaDB UDS transport** — Respects default transport in embedded config (#273)
- **Conversation semantic search** — Uses correct embedding store method (#277)
- **Bayesian SessionConfig validation** — Asserts threshold ranges (0.0-1.0) with ValueError for out-of-range (#272)
- **Agentic iteration count** — Surfaced correctly for A/B turn metric (#235)
- **Prompt-guard tool names** — Aligned with canonical read surface (grep, not stale code_search hint) (#274)
- **Write/mutation/exploration tool-name sets** — Sourced from unified registry via core_tool_aliases + tool_names (#199)
- **Formatting** — Black applied to proximadb_provider.py (#239)

## [0.7.1] - 2026-06-21

A hardening and chat-UX release: the new `victor ui` chat surface gained controls and
resilience, per-turn cost/token accounting was finished end-to-end, and a sweep of
streaming, tool-selection, governance, and indexing bugs surfaced by live use were fixed.
`victor-contracts` stays at 0.7.0 (no protocol changes).

### Added
- **Chat UI controls & resilience** — stop/cancel an in-flight turn, close-guard so a
  mid-stream disconnect can't tear down the provider, and error recovery (#144)
- **Chat settings panel + session-restore seam** — provider/profile/model/approval controls
  and best-effort replay of a reconnected session's history (#146)
- **`victor ui --profile`** to select an agent profile for the chat surface
- **Informed tool approvals** — the policy ASK prompt now carries tool args and a human
  summary; per-call tool steps, accumulating reasoning, and first-token feedback in the UI (#135)
- **Tool telemetry in chat** — per-call duration, real output, output pruning, and follow-up
  suggestions surfaced as steps (#135)
- **Per-turn cost trace (C0)** — a cost/latency/token footer (L3) and cost-aware model
  routing that biases toward cheaper/faster providers on expensive turns (L4) (#139)
- **Reference-aware tool-result pruning** — context compaction that keeps referenced results,
  enabled by default (L1) (#137, #138)
- **`victor db maintain`** — checkpoint the WAL and reclaim space on the global and project
  databases (#153)
- RL learning-trace merge caching within a task (L2, flag-gated)

### Changed
- **CI split** — lightweight fast checks on `develop`, extensive gating reserved for
  `develop → main`; develop quick-tests scoped to changed-file unit tests (#136, #142)
- Scheduled security scan runs weekly instead of nightly (#145)
- Docs Pages deploy can be triggered on-demand via `workflow_dispatch` (#156)
- New project databases open with `auto_vacuum=INCREMENTAL`; `graph_module_metric_history`
  is capped per module to stop unbounded growth (#153)

### Fixed
- **Streaming cleanup** — the provider SSE stream is now closed on-task across the full
  consumer → resilience → provider decorator chain, eliminating "async generator ignored
  GeneratorExit" / "exit cancel scope in a different task" spam; also corrected the streaming
  `ToolExecutionResult` shape used for plateau/progress accounting (#152, #154, #155)
- **`web_search` reaches the model** — a deliberately-selected web tool is no longer dropped
  by stage pruning or the order-blind post-selection truncation on research-flavored tasks
  (#149, #157)
- **Governance approval in the chat UI** — the interactive ASK approval handler now survives
  the DI container bootstrap (and is registered via the correct API), so ASK-gated tools
  prompt instead of silently denying (#150)
- **No mutation tools forced onto read-only tasks** — write-intent no longer injects
  edit/write/shell on analyze/search/research turns; tool-call loops are detected earlier (#151)
- **`code_search` fails fast** when the semantic indexing provider is absent — falls back to
  literal search in <1s instead of a ~600s index-lock hang (#158)
- **Token accounting** — per-turn input/output tokens are measured end-to-end from provider
  usage and finalized on the service path; Ollama usage surfaced on the streaming done chunk
  (#134)
- **Graph tool** — stop advertising unimplemented modes (`dead_code`, `dynamic_imports`);
  suggest same-file symbols with an unambiguous basename fallback for `file:Symbol` (#125)
- Chainlit 2.x compatibility for chat-UI tool approval

### Security
- Remediated Semgrep SAST findings across workflows, CI, and Docker, and cleared the
  remaining Semgrep OSS findings blocking `develop → main` (#140, #148)
- Upgraded vulnerable dependencies (FastAPI/Starlette, transformers) to clear CVEs (#143)

### Removed
- Superseded standalone Vite UIs (`web/ui`, `ui/`) in favor of the Chainlit `victor ui` (#141)

## [0.7.0] - 2026-06-19

### Added
- **`victor ui`** — a pure-Python Chainlit web chat surface bound to `VictorClient` (streams tokens, renders reasoning and tool calls as steps); behind the optional `chat-ui` extra
- **Human-in-the-loop tool approval** for non-TTY surfaces — `SessionConfig.tool_approval` turns on the governance ASK path for named tools; `register_policy_approval_handler()` registers a surface's approval handler in the DI container
- **Single FastAPI HTTP server** — `/lsp/*` ported onto the core server and new `/credentials/{set,delete,status}` + `/tools/cancel` routes, so the full IDE/extension API surface lives on one server
- **VS Code extension ↔ FastAPI contract test** — guards endpoint drift between the extension client and backend routes
- **VS Code extension** unit harness — vitest + c8 coverage with a `vscode` mock (21 logic-module tests), wired into CI
- **ExtensionManifest** and **ExtensionType** in victor-contracts for vertical capability declaration
- **CapabilityNegotiator** for manifest validation during vertical activation
- **API versioning** (`CURRENT_API_VERSION=2`, `MIN_SUPPORTED_API_VERSION=1`)
- **CallbackCoordinator** for tool/streaming lifecycle callback delegation
- **OrchestratorPropertyFacade** — 37 properties extracted to separate module
- **InitializationPhaseManager** — structured 8-phase runtime initialization
- **ExtensionModuleResolver** — module path resolution extracted from extension loader
- **ExtensionCacheManager** — thread-safe namespaced cache for extensions
- SBOM generation (CycloneDX) in release pipeline
- Tests for `tool_pipeline`, `cqrs`, `workflows/executor`, `sqlite_lancedb` (46 new tests)
- `[embeddings]` optional extra for `sentence-transformers`, `lancedb`, `pyarrow`
- Deprecation warnings on all 5 contrib vertical imports
- TODO/FIXME triage document (81 markers categorized)

### Changed
- **Consolidated to a single HTTP server** — the legacy aiohttp `VictorAPIServer` was removed; `VictorFastAPIServer` is now the sole server and its routers own the full API surface
- **VS Code extension** upgraded to Node 22, TypeScript 5.9, and typescript-eslint 8
- **Orchestrator** reduced from 4,514 to 3,940 LOC via property/callback extraction
- **Extension loader** reduced from 2,049 to 1,897 LOC via resolver/cache extraction
- **protocols.py** decomposed from 3,703 LOC monolith into 9 domain modules with lazy `__getattr__` loading
- **ProviderPool** duplicate removed; wired with `use_provider_pooling` feature flag
- **Settings** flat access now emits `DeprecationWarning` (use nested groups instead)
- Victor-devops: 3 forbidden imports migrated to `victor.framework.extensions`
- 32 documentation files reconciled (provider count 22→24, tool count 33→34)
- Heavyweight dependencies (`sentence-transformers`, `lancedb`, `pyarrow`) moved to `[embeddings]` extra
- `victor chat` no longer exposes `--legacy`; the canonical framework client path is now the only chat path

### Deprecated
- **`FrameworkShim` compatibility surface**
  - Deprecated in: `0.7.0` on `2026-04-30`
  - To be removed in: `v1.0.0`
  - Target removal date: `2027-06-30`
  - Replacement: `Agent.create()` for public callers or `AgentFactory` / `AgentCreationFactory` for internal composition
  - Migration path: `docs/architecture/migration.md`
  - Compatibility shim status: warning-backed shim remains supported through `v1.0.0`
- **`TeamNode*` workflow compatibility aliases** (`TeamNode`, `TeamNodeConfig`, `TeamNodeWorkflow`, `TeamNodeExecutor`)
  - Deprecated in: `0.7.0` on `2026-04-30`
  - To be removed in: `v0.9.0`
  - Target removal date: `2027-03-31`
  - Replacement: `TeamStep*` workflow names
  - Migration path: `docs/architecture/migration.md`
  - Compatibility shim status: warning-backed aliases remain supported through `v0.9.0`
- **`WorkflowGraph` alias from `victor.workflows.graph`**
  - Deprecated in: `0.7.0` on `2026-04-30`
  - To be removed in: `v0.8.0`
  - Target removal date: `2026-12-31`
  - Replacement: `BasicWorkflowGraph` for the simple container or `victor.workflows.graph_dsl.WorkflowGraph` for the typed DSL
  - Migration path: `docs/architecture/migration.md`
  - Compatibility shim status: warning-backed alias remains supported through `v0.8.0`

### Fixed
- **Release pipeline** — the SBOM job installs the in-repo `victor-contracts` before victor-ai (it is not on PyPI), and the native-wheel build passes `--find-interpreter` so maturin finds a CPython under `manylinux: auto`
- Bare `except:` in `experiments.py` → `except (ValueError, TypeError)`
- `_send_rl_reward_signal` test updated for CallbackCoordinator delegation

### Security
- `eval()`/`exec()` audit: only 2 real calls, both sandboxed with `__builtins__: {}`
- SecretStr used for sensitive config fields (17 occurrences)

## [0.6.0] - 2026-03-31

### ⚠️ Breaking Changes
- **NONE** - This release maintains 100% backward compatibility

### Added

**Core Architecture** (Phases 1-7):
- **@register_vertical decorator** for declarative vertical registration with rich metadata
- **VerticalMetadata** dataclass for type-safe metadata extraction (replaces fragile `.replace()` patterns)
- **UnifiedEntryPointRegistry** for single-pass entry point scanning (31x faster: 500ms → 16ms)
- **VersionCompatibilityMatrix** with PEP 440 version constraint checking
- **ExtensionDependencyGraph** with topological sort and circular dependency detection
- **AsyncSafeCacheManager** with lock-per-key caching (eliminates race conditions)
- **OpenTelemetry integration** for production observability and monitoring
- **PluginNamespaceManager** for priority-based plugin namespace isolation
- **VerticalBehaviorConfigRegistry** for dynamic vertical configuration

**Testing & Validation** (Phase 8):
- 224 new tests with 100% pass rate
- 93% average code coverage across all new modules
- Performance benchmarks for entry point scanning and dependency resolution
- Backward compatibility tests validating legacy patterns

**Documentation** (Phase 9):
- Architecture documentation (~1,000 lines) with before/after comparison
- Migration guide (~900 lines) with step-by-step examples
- Complete API reference (~1,200 lines) for all new modules
- Best practices guide (~1,100 lines) for vertical development

**Operations** (Phase 10):
- Rollout plan (~1,500 lines) with 5-stage gradual deployment
- Monitoring dashboards (~1,100 lines) with 4 Grafana dashboards
- Deployment playbook (~1,000 lines) with blue-green procedures
- Legacy deprecation plan (~800 lines) with removal timeline

**Vertical Alignment** (Phase 11):
- All 6 external vertical packages aligned with SDK/framework
- victor-coding, victor-devops, victor-rag updated to v0.6.0
- victor-dataanalysis, victor-research, victor-invest updated to v0.6.0
- Validation script (`scripts/validate_verticals.py`) for automated checking

### Changed

**Performance**:
- Entry point scanning: 9+ independent calls → 1 unified scan (200-500ms startup improvement)
- Scan duration: 500ms → 16ms (31x faster)
- Dependency resolution: 2-5ms for complex graphs
- Cache contention: Single lock → Lock-per-key (enables parallel loading)

**Architecture**:
- Extension loader: 1,897 LOC with SOLID-compliant focused modules
- Vertical metadata: Type-safe extraction replaces fragile string manipulation
- Configuration: Externalized from hardcoded dict to dynamic registry
- Safety rules: Refactored from internal coordinators to framework capabilities

**Code Quality**:
- SOLID principles applied throughout new modules
- Protocol-based design (Interface Segregation Principle)
- Dependency Inversion - depends on abstractions, not concretions
- Single Responsibility - focused, cohesive modules

### Fixed

**Class Name Generation**:
- Fixed: Metadata extraction using pattern matching instead of `.replace()`
- Fixed: Type-safe `canonical_name` extraction preserves naming intent
- Fixed: Display name generation with proper title casing

**Import Boundaries**:
- Fixed: victor-dataanalysis safety rules now use `SafetyRulesCapabilityProvider` from framework
- Fixed: victor-rag safety rules now use `SafetyRulesCapabilityProvider` from framework
- Fixed: Removed forbidden imports from `victor.agent.coordinators.safety_coordinator`

### Deprecated

**Legacy Patterns** (with DeprecationWarnings):
- Class name extraction using `.replace("Assistant", "")` pattern
- Class name extraction using `.replace("Vertical", "")` pattern
- Hardcoded `_VERTICAL_CANONICALIZE_SETTINGS` dict
- Multiple independent `entry_points()` calls (use `UnifiedEntryPointRegistry`)
- Direct `victor.agent.coordinators` imports (use framework capabilities)

**Removal Timeline**: September 2026 (v0.7.0)

### Security

- Safety rule validation prevents dangerous operations
- PEP 440 version checking prevents incompatible framework versions
- OpenTelemetry integration enables security monitoring
- Secret masking in DevOps middleware

### Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Entry Point Scans | 9+ | 1 | 200-500ms saved |
| Scan Duration | ~500ms | ~16ms | 31x faster |
| Startup Latency | Baseline + 500ms | Baseline | 200-500ms faster |
| Dependency Resolution (simple) | N/A | ~2ms | <5ms target ✅ |
| Dependency Resolution (complex) | N/A | ~3-5ms | <10ms target ✅ |
| Cycle Detection | N/A | ~1-2ms | <5ms target ✅ |

### Migration

**For Users**: No action required - all existing verticals work without modification

**For Vertical Developers**: Optional - add `@register_vertical` decorator:
```python
from victor.core.verticals.registration import register_vertical

@register_vertical(
    name="my_vertical",
    version="1.0.0",
    min_framework_version=">=0.6.0",
)
class MyVertical(VerticalBase):
    pass
```

**Documentation**: See `docs/verticals/migration_guide.md`

### Known Issues

1. **Class name construction** in contrib verticals uses lowercase `canonical_name`, causing incorrect camelCase instead of PascalCase (e.g., `ragRLConfig` vs `RAGRLConfig`). External verticals are not affected. Fix scheduled for v0.6.1.
2. **Conversation coordinator** usage in victor-dataanalysis and victor-rag uses internal API. Documented with TODO for future refactoring.

See `docs/verticals/KNOWN_ISSUES_v0.6.0.md` for details.

### Contributors

- Architecture design and implementation: Claude Code AI Assistant
- Testing: 224 tests with 93% coverage
- Documentation: 9,200+ lines across 13 files
- Validation: Automated alignment checking for all 6 vertical packages

---

## [0.5.7] - 2026-03-07

### Added
- Victor-SDK v0.5.7 with zero-runtime-dependency vertical definitions
- Capability negotiation and state externalization (Phase 4)
- Unified victor-ai / victor-contracts versioning with CI enforcement
- Processing/LSP re-export modules for external vertical imports
- CopyOnWriteState thread guard and entry point failure isolation
- Topic-prefix index for O(1) event dispatch
- Collision detection and public extension API for verticals
- Fast CI workflow for quicker feedback

### Changed
- Black formatting applied to 79 files
- FastAPI server hardened against injection, traversal, and data exposure
- SecretStr for API keys to prevent credential leakage

### Fixed
- VS Code extension: handle offline servers in `supportsCapability` check
- Build: victor-contracts built locally in release workflow

## [0.5.6] - 2026-03-01

### Fixed
- Added strawberry-graphql to dev dependencies for GraphQL integration tests

## [0.5.5] - 2026-02-28

### Added
- Initial public release with full framework, 22 providers, 33 tools
- Multi-agent team formations (sequential, parallel, hierarchical, pipeline)
- YAML workflow DSL with StateGraph execution engine
- 9 domain verticals (coding, devops, rag, dataanalysis, research, security, iac, classification, benchmark)
- CLI (Typer) and TUI (Textual) interfaces
- Docker support with multi-arch builds
