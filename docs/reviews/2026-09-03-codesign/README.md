# Victor co-design maintenance review — 2026-09-03

Whole-codebase review of `victor/`, `victor-contracts/`, `verticals/`, `victor-codegraph/`, `rust/`, `vscode-victor/` (~2,300 Python modules, ~930k LOC Python + 13k Rust + 23k TS). Method: graph-index the import structure, cluster it to derive module boundaries, then review each cluster through a co-design lens (NVIDIA HW/SW co-design principles translated to software) with first-principles, data-structure, and design-pattern analyses, informed by hindsight (churn history, concluded programs, A/B verdicts).

**Unit reports (full detail, findings tables with file:line citations):**

| Unit | Scope | Report |
|---|---|---|
| U1 | agentic core (`victor/agent/`, `victor/runtime/`) — 203k LOC | [U1-agentic-core.md](U1-agentic-core.md) |
| U2 | framework public API (`victor/framework/`) — 139k LOC | [U2-framework.md](U2-framework.md) |
| U3 | providers + config — 41k LOC | [U3-providers-config.md](U3-providers-config.md) |
| U4 | tools — 56k LOC | [U4-tools.md](U4-tools.md) |
| U5 | storage, core, state, victor-codegraph — 112k LOC | [U5-storage-core-codegraph.md](U5-storage-core-codegraph.md) |
| U6 | workflows, teams, coordination, protocols — 73k LOC | [U6-workflows-teams.md](U6-workflows-teams.md) |
| U7 | client surface (ui, integrations, vscode, web) — 74k+23k LOC | [U7-client-surface.md](U7-client-surface.md) |
| U8 | context, processing, native, rust — 40k+13k LOC | [U8-context-processing-native.md](U8-context-processing-native.md) |
| U9 | contracts + verticals — 110k LOC | [U9-contracts-verticals.md](U9-contracts-verticals.md) |
| U10 | evaluation, observability, experiments, benchmark, contrib — 80k LOC | [U10-evaluation-observability.md](U10-evaluation-observability.md) |

---

## 1. Method

1. **Graph indexing.** AST import extraction over all in-repo Python packages → directed module graph: 2,315 modules, 6,900 edges, with fan-in/fan-out, LOC, and 12-month git churn per module (builder: `/tmp/victor_graph/build_graph.py`; artifacts `graph.json`, `subpkg.json`, `clusters.json`).
2. **Clustering.** Louvain community detection at file level (50 communities, modularity 0.76) and subpackage level. The subpackage graph is **dense**: best modularity is only 0.16–0.21 — cross-package coupling is high and package boundaries are blurrier than docs suggest (§2). Ten review units were formed from the communities, folded along the documented layering.
3. **Dogfood re-index.** A fresh graph index was run through the production `GraphIndexingPipeline` (not a script) — itself a measurement (§5).
4. **Cross-validation.** The AST graph was validated against victor_codegraph's IMPORTS edges in `project.db` (§5).
5. **Per-unit reviews** by ten parallel reviewers with graph-derived hotspot briefs (churn leaders = hindsight hotspots; fan-in leaders = stability-critical).

**Co-design rubric** (NVIDIA HW/SW co-design translated): (a) *data movement is the enemy* — serialization, copies, round-trips across seams; (b) *roofline* — each hot path optimized at its actual binding constraint (LLM I/O vs disk vs CPU vs locks vs render); (c) *locality & reuse* — caching, batching, prefix/KV-cache friendliness; (d) *amortization* — one-time costs hoisted out of hot loops; (e) *co-designed interfaces* — seams shaped by both sides, not internal convenience.

## 2. What the graph says about module boundaries

- **A 579-module strongly-connected component centered on `victor.agent`.** ~25% of the codebase sits in one import cycle. The god-package is not a style complaint; it is measurable: any module in this SCC can transitively reach any other, so no part of the agentic core can be tested, extracted, or reasoned about in isolation. Secondary cycles: 20 modules across `integrations/api` routes, 10 across `ui/commands`+`ui/slash`, 8 across `framework/coordinators ↔ workflows` (confirming the framework→workflows inversion), plus ~7 smaller pairs.
- **Subpackage-level modularity is only 0.16–0.21.** File-level communities are strong (0.76) — code clusters tightly *within* packages — but the packages themselves are so inter-referenced that clustering barely separates them. The practical unit of cohesion in this codebase is the *sub-system* (agent-services, graph-engine, provider-kernel), not the package.
- **Churn concentrates where coupling is highest**: `victor.agent` (5,265 touching commits/12m) and `victor.framework` (2,882) are also the top fan-in packages (17 and 21) — every change ripples.
- **Stability ranking** (fan-in): `victor.core` (26) > `framework` (21) > `config` (20) > `agent` (17) = `tools` (17) > `contracts` (15) > `providers` (14) > `storage` (13). High fan-in + high churn (`agent`, `framework`) is the classic instability signature.

## 3. Cross-cutting co-design themes

Ten themes emerged independently in 2+ units each. These are the review's real findings — each unit report has the detail; this is the coordinated view.

### T1. Config is data movement (3 units)
Uncached `Settings()` construction — `.env` disk read + double env scan + 155-field pydantic validation, ~4.5–5.5 ms — happens **per LLM request** (providers: sandhi_transport.py:429), **per graph invoke** (framework: 11 call sites incl. graph.py:263), and **per stale file** during reindex (storage: sqlite_store.py delete loop). One process-cached snapshot with invalidation via the existing `Settings.notify_change` listeners eliminates the whole class. *(U3-F1, U2-F3, U5-09 — one small PR, systemic effect.)*

### T2. Hot-path data movement (6 units)
The roofline says every per-turn path is LLM-I/O-bound, yet each carries avoidable CPU/copy work: full-history `model_dump()` per loop iteration for a consumer reading 2 messages (U1-1); up to 4 full-state deep-copies per graph node for a history with zero default-path readers (U2-F2/U6-F3); 3–6 deep-copies of team context per delegate step (U6-F8); `.stat()`+open/append/close per observance event while a `BufferedExporter` sits unused (U10-F2); synchronous SQLite flushes on the event loop under a lock (U5-05); O(history) work per keystroke in the TUI toolbar (U7-F9); unbounded event queues on the SSE/WS bridge (U7-F7).

### T3. Amortization built at the wrong layer (4 units)
The caches exist but not on the paths that run: registry schema cache with zero production callers while `estimate_tool_tokens` regenerates schemas ≥4× per turn (U4-F1); `CachedCompiledGraph` that still revalidates + recompiles per execution (U6-F2); per-tool embedding cache that invalidates wholesale on any description change (U4-F8); `Draft7Validator` constructed per tool call (U4-F6); registrar loop unbatched (U4-F12). Pattern: *amortization machinery lands where it's easy to build, not where the cost is.*

### T4. Dual execution paths (3 units)
Two live workflow engines violating "StateGraph is always the execution engine" (U6-F1); AgenticLoop with four execution paths whose `stream()` lacks the guards `run()` has (U2-F1); `parallel_tool_execution=True` a dead setting over a strictly sequential loop while a parallel executor ships unused (U1-2); three parallel-dataflow implementations with different merge semantics (U6-F6). Every duality is a parity-drift bug factory — U8's findings (below) show what drift costs.

### T5. Guards under-cover their surfaces (5 units — the meta-finding)
The guard-test discipline is real but every layer's guard checks less than its actual import surface: the client guard scans 3 directories and one symbol while 56 `victor.agent.*` import sites live in `victor/ui` and integrations are unguarded (U7-F4); providers/config import `victor.agent.*` with no guard (U3-F2); the facade guard is string-grep (U1-11); three vertical guards define three different boundaries while 15+ lazy host-internal imports evade the audit's prefix list (U9-F1/F6); framework→UI and framework→workflows inversions unguarded (U2-F14, U6). **Fix once, systemically: one boundary-manifest (forbidden/allowed import prefixes) exported from `victor_contracts`, consumed by every guard test and the auditor — AST-based, not string-grep.** (U9 co-design #1.)

### T6. Factory & boundary bypasses (3 units)
"AgentFactory is the single authority" is bypassed by the eval harness's hand-built stack (U10-F1 — eval arms silently miss production middleware), DirectCreationStrategy/protocol adapters (U2-F8), the CLI REPL and `/completions` (U7-F6/F10). Measurement and clients exercising paths production doesn't (or missing paths it does) quietly poisons both A/B results and the client seam.

### T7. Native↔fallback drift where it hurts most (U8)
The Rust/Python split is roofline-correct, but parity has drifted in exactly the budget-guarding functions: disjoint strategy vocabularies silently remapped on each side (U8-F1), three token-count semantics shifting budgets ~±20% by install (U8-F2), a PyO3 type mismatch that silently kills the Rust context-fitter for default inputs (U8-F4), a flush bug dropping tail-of-stream content on the native path only (U8-F7), `panic="abort"` making every Rust panic a process kill despite except-fallback wrappers (U8-F3), and GIL held on 4 of 5 batch FFI paths (U8-F5). A generated native==fallback parity matrix in CI is the structural fix.

### T8. Concluded-program & dead-code aftermath (7 units, ~15k LOC)
Hindsight's biggest maintenance dividend: judge-era machinery ~2.5k LOC (program concluded #884), effect-gate aftermath ~1.4k LOC (A/B negative), third resilience impl 868 LOC (own test only), unused emitters 2,056 LOC, dead stream_adapter 770 LOC, browser_tool_legacy 913 + code_search stub 212, chat_lazy/chat_refactored/optimization/victor.commands 2,821, workflows adapters+metrics 1,355, manager.py 401, incremental_indexing_simple, project_manifest 892 + command_parser 630. Each item is small; together they are ~4–5% of the codebase misleading every future reader and inflating every grep.

### T9. Security: the API auth seam is opt-in (U7)
`_verify_api_key` is called in 3 sites — 1 of 19 route files; `/ws` runs full agent turns without checking `authenticated`; `/terminal/execute` is unauthenticated arbitrary shell with caller-attested approval. Router-level dependency + WS accept-time auth is the fix (U7-F1/F2/F3). Highest-priority items in the whole review.

### T10. Derived-state lifecycle (U5)
The graph DB is defined as derived/rebuildable, yet it is the largest file in the repo (2.64 GB, 66% free pages at churn peak), never reclaims (auto_vacuum=0 in header), starves WAL checkpoints, and the "incremental" path re-parses the whole repo when ≥1 file changes (planner marked 72% of 4,588 files stale after one month; refresh took 966s). Derived state should imply its own reclamation; manifests should prevent re-parsing unchanged files.

## 4. Prioritized maintenance backlog

Sequenced by (severity × leverage ÷ effort). IDs reference unit reports. FEP-gated items marked.

**Wave 1 — immediate, small, high severity (days): DONE 2026-09-03, PRs #994–#998 (+#993 artifacts).**
Each PR passed an adversarial review pass before merge; findings fixed in-PR with negative tests.
Known accepted gaps: terminal blocklist remains substring-based (API-key auth is the real gate);
`/ws` broadcast push unauthenticated on keyed servers (chat is message-gated; header-gating would
break the vscode flow); fallback drops thinking content sharing a chunk with an entering
transition (pre-existing); bootstrap container shares the settings snapshot by design.

| # | Item | Ref | PR |
|---|------|-----|----|
| 1 | API auth on all HTTP routes (per-route split-child-router), WS chat gate + WS event guard, terminal server-side approval, bounded event queues | U7-F1/F2/F3/F7 | #996 |
| 2 | Process-cached Settings snapshot (`load_settings()`, `fresh=True` mutators, lock, session-runner private copy) | U3-F1, U2-F3, U5-09 | #997 |
| 3 | Context-fitter: unified strategy vocabulary + int priority + /100 scoring in f64 on both sides | U8-F1/F2/F4 | #998 |
| 4 | Literal `\n` in background-task messages (7 sites) | U1-3 | #995 |
| 5 | Delete duplicate `context_window` on BaseProvider (dead contract branch) | U3-F5 | #994 |
| 6 | Checkpoint aliasing: deep-copy at save/load/list boundaries | U2-F5, U6-F9 | #995 |
| 7 | Rust streaming flush + `panic="unwind"` (PanicException caught by name in shims) | U8-F7/F3 | #998 |
| 8 | CircuitOpen non-retryable (explicit opt-in preserved); breaker keying by (class, base_url), lazily resolved | U3-F3 | #994 |
| 9 | Tool-selection history projection replaces per-turn `model_dump()` | U1-1 | #995 |
| 10 | (folded into #1/#996) | U7-F7 | #996 |

**Wave 2 — hot-path & structural perf (1–2 sprints): DONE 2026-09-04/05, PRs #1000–#1010.**
Risk-ramped order (mechanical items W1–W4 reviewed after CI green; structural/behavioral items
W5–W10 reviewed before requesting merge — CLAUDE.md rule), one worktree at a time, removed
immediately after merge. Every PR passed an adversarial review pass; W5, W9, and W10 (the
highest-risk items — cache correctness, subprocess process-group handling, and concurrent tool
dispatch respectively) got a dedicated review-agent pass that found and fixed real issues pre-merge
(a callable-identity cache-key collision on W5; a PID-reuse signal race and an unguarded
gather-exception orphan risk on W9/W10; a budget-accounting timing bug and a byte-for-byte
behavior regression for budget-exhausted tail calls on W10, both caught and fixed before merge).
Item 16 shipped as a strict read-only allowlist (idempotent-tools ∩ ¬write-tools, shell hard-excluded)
with **default OFF** per an explicit scope decision — the allowlist makes it sound to enable, but it
still changes execution ordering/timing broadly enough to ship opt-in. Items 17 and 20's
"writer off the event loop" half (listed here as item 20b) are **deferred to Wave 3** — same
explicit scope decision, both are structural changes better suited to a dedicated proposal;
20a (batched staleness/metadata queries) shipped in Wave 2.

| # | Item | Ref | PR |
|---|------|-----|----|
| 11 | state_history opt-in via GraphConfig (default no copy) | U2-F2, U6-F3 | #1006 |
| 12 | Tool schema/token cache keyed by registry version; consume or delete `get_tool_schemas` | U4-F1 | #1001 |
| 13 | CompiledGraph cache keyed by definition hash + factory version | U6-F2 | #1005 |
| 14 | Batched observability sink (usage logger behind BufferedExporter) | U10-F2 | #1007 |
| 15 | Shared subprocess runner: process groups, killpg, capped incremental reads, partial output | U4-F2/F3 | #1009 |
| 16 | Parallel tool execution behind a read-only allowlist gate, default OFF | U1-2 | #1010 |
| 17 | Native↔fallback parity harness in CI (+ BPE ranks for exact counting) | U8-F2/F5 opp.5/6 | deferred → Wave 3 |
| 18 | One shared failure-taxonomy module (classify/retry/health) + TimeoutProfile | U3-F9/F6 | #1008 |
| 19 | Assembler O(n²) fix + per-turn score caching | U1-5 | #1000 |
| 20a | SQLite: batched staleness query, batched node-metadata updates | U5-03/04 | #1004 |
| 20b | SQLite writer off the event loop | U5-05 | deferred → Wave 3 |
| 21 | Lazy slash import (1.7s CLI startup → gate with importtime CI check) | U7-F8 | #1002 |

**Wave 3 — structural, multi-PR (FEPs where noted). Also carries items 17 and 20b deferred from Wave 2:**

| # | Item | Ref | Effort |
|---|------|-----|--------|
| 22 | Single boundary manifest consumed by all guards; AST-based facade guard; ratchet on orchestrator | T5, U1-11, U9-F6 | M |
| 23 | Engine unification: WorkflowExecutor as facade over CompiledGraph; AgenticLoop graduation (FEP-0007) | U6-F1, U2-F1 | L |
| 24 | Interrupt/resume semantics: `interrupted` field, resume-at vs completed-at (FEP) | U6-F4 | M |
| 25 | Dead-code sweep (~15k LOC across 7 units; record negative A/B verdicts in PRs) | T8 | S each |
| 26 | Manifest-aware `parse_repo` + derived-state reclamation (vacuum/WAL) + Tier-A/B on SQLite | U5-01/02/07 | M–L |
| 27 | ChatService inversion: own the turn lifecycle; guard-test upgrade first | U1-4/7 | L |
| 28 | Contrib bases → `victor_contracts.verticals`; de-template the 4 small verticals (~12k LOC) | U9-F5 | M–L |
| 29 | RL/prompt-evolution out of `victor/framework` (FEP-0025 Phase 6) | U2-F4 | L |
| 30 | Coordinator split: WorktreeMergeService + DelegateContractBuilder | U6-F7 | L |
| 31 | One benchmark stack on the BenchmarkRunner protocol; single agent-creation path in evals | U10-F1/F8 | L |
| 32 | REPL/`/completions` through VictorClient; session-aware API contract | U7-F5/F6/F10 | M |

## 5. Dogfood & validation (graph pipeline as measured)

- **Fresh re-index via the production pipeline** (`GraphIndexingPipeline`, incremental, embeddings off): 4,588 files discovered, **3,294 (72%) planned stale after 1 month**, planning 224s, total 966s. Empirically confirms U5-02: the incremental machinery does not amortize parsing — one changed file triggers a full-repo `parse_repo`.
- **AST graph vs victor_codegraph IMPORTS edges** (module-level, in-scope): codegraph 2,884 edges, **2,882 (99.9%) present in the AST graph** — no contradictions; 2 codegraph-only edges. 4,018 AST-only edges are almost entirely `__init__.py` re-export edges (package→submodule) the module-node graph doesn't represent, plus stricter submodule resolution. **Conclusion: both extraction paths validate each other; the AST graph is the richer analytic source, codegraph's IMPORTS are a consistent subset.**
- **Method caveats**: the AST extractor counts TYPE_CHECKING and module-`__getattr__` bridge references as edges (hence `victor_contracts → victor: 40` — largely lazy bridges, see U9); dynamic imports (entry-point plugin discovery) are invisible to static extraction; churn counts file-touching commits, not line volume.

## 6. Keep intact (consolidated load-bearing quirks)

1. KV/prompt-cache stability machinery — frozen system prompt, tool sorting, system-nudge interception (U1). Never "simplify" by allowing system-role appends.
2. Chat-only idempotency-key binding; streams deliberately unbound (U3). Session affinity in neutral metadata, never the request body.
3. No silent sandhi transport fallback — replay after FFI failure can duplicate billed requests (U3).
4. Legacy calls fallback in graph indexing — the only symbol source without an enhanced TSA provider (U5); self-heal schema re-ensure; provider-resolution-before-parse ordering (a documented race).
5. `_NATIVE_AVAILABLE` single flag + benchmark-aware Python-preferred dispatch (U8); `time.monotonic`-style discipline when touched.
6. Default-off `strict_edges`, per-invoke collaborator construction (cached CompiledGraphs stay concurrency-safe), COW-by-default state (U2/U6).
7. run_kind tagging, runtime_feedback trust model, flag-graduation policy, in-container `enhanced` verifier — concluded-program keepers (U10); `effect_gated_completion` stays default-off forever (negative A/B).
8. LiveDisplayRenderer incremental HEAD/TAIL rendering; ObservabilityBus lossy contract; EvalContainer lifecycle (U10/U7).
9. Lazy PEP 562 host bridges in verticals + victor_sdk isinstance-preserving alias (U9).
10. Deliberate busy_timeout divergence and the separate undo.db (U5).

## 7. Artifacts

- Graph builder + outputs: `/tmp/victor_graph/` (`build_graph.py`, `graph.json`, `subpkg.json`, `clusters.md`, `briefs/`, `validate_vs_codegraph.py`, `reindex.py`, `index.log`) — ephemeral; promote `build_graph.py` to `scripts/` if the boundary/cluster analysis should become repeatable.
- Unit reports: this directory (`U1`–`U10`).
