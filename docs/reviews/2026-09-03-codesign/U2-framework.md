# U2 — Framework (`victor/framework/`)

325 modules / 139,329 LOC / 2,882 churn commits (12m). Reviewer: parallel unit review, 2026-09-03.

## Overview
Framework has strong guard-rail culture (FEPs, boundary tests, dated deprecations) but two structural debts dominate: (1) a 4.4k-LOC AgenticLoop god-class with four parallel execution paths (legacy while-loop, StateGraph fork, unwired FEP-0007 streaming seam, Runtime-hosted live streaming), and (2) a 37.8k-LOC RL/prompt-evolution subsystem — 27% of the "stable public API" package — carrying the highest churn. StateGraph itself is cleanly decomposed, but its execution loop pays O(N × state) full deep-copies per node for histories nobody consumes on the default path, and per-invoke Settings reconstruction re-reads YAML/env on every call. Roofline: all framework-path CPU complexity is dwarfed by LLM I/O, yet the copies and config reloads add latency and memory pressure on exactly the wrong axis.

## Findings

| ID | Sev | Location | Finding | Why it matters | Fix sketch | Effort |
|----|-----|----------|---------|----------------|------------|--------|
| F1 | P1 | agentic_loop.py:807-1468, 1767-1960 | AgenticLoop = 4,376 LOC, ~75 methods, run() ≈660 lines; four entry points (run/stream/run_streaming/stream_chat) plus default-off StateGraph fork (feature_flags.py:245-247) | churn=101 (unit max); stream() (1895-1960) re-implements the loop WITHOUT spin guards, verify hook, backslide guard, session ledger — parity drift is a correctness bug for streaming callers | Finish Phase-15: graduate StateGraph executor, delete legacy loop + thin stream(); FEP-0007 amendment | L |
| F2 | P1 | graph_runtime.py:251, 109; graph_execution.py:50-54, 242; graph_state.py:84-87 | Up to 4 full-state copies per node: unconditional deepcopy into state_history, second snapshot in streaming queue, COW deep-copy-on-first-write, deepcopy on invoke entry | O(N²) data movement for conversation-carrying state; only the StateGraph-executor path consumes state_history | Gate snapshotting behind GraphConfig (default off in invoke()); make COW field-level | M |
| F3 | P1 | graph.py:263-266; settings.py:2573-2579 | `_should_use_cow` calls uncached `load_settings()` → fresh pydantic BaseSettings per invoke()/stream(); 11 such call sites in framework (agent.py:250, client.py:327, init_synthesizer ×3) | Per-request config-file I/O and model construction hidden inside the graph engine's hot path | Process-cached Settings snapshot with explicit invalidation via existing change listeners | M |
| F4 | P1 | victor/framework/rl/ (37,826 LOC) | RL/prompt-evolution subsystem is 27% of the "stable public API" package; prompt_optimizer churn=81 | Framework version bumps ship experimental code; external consumers get the whole RL surface | FEP-0025 follow-up: move rl/ behind victor_contracts or its own package | L |
| F5 | P2 | graph_checkpoint.py:66-71; graph_execution.py:241, 244-261 | MemoryCheckpointer stores state by reference; with COW off or BaseModel states, later mutation rewrites every stored checkpoint; `load_initial_state` shallow-copies so resumed runs alias stored history | Corrupts replay/get_state_history exactly when COW is disabled — silent, config-dependent | Deep-copy (or serialize) at save boundary; deep-copy on load | S |
| F6 | P2 | config.py:339-340, 360-364; graph.py:278-313 | GraphConfig exposes `validate_on_entry`/`validate_after_nodes` that the engine never reads; validator rebuilt per node when compile-time and invoke-time configs disagree | Placebo public knobs = contract lies; per-node full pydantic `model_validate` when enabled | Honor the flags or delete them; resolve validator once per invoke | S |
| F7 | P2 | agentic_loop.py:1784-1789, 1802-1806 | `run_streaming()` is an unwired public seam — raises unless `streaming_act_port` is injected "at the FEP-0007 cutover" | Framework advertises an API that cannot run; callers discover at runtime | Wire the port or mark experimental until cutover | S |
| F8 | P2 | integrations/protocol/adapters.py:66; agent/creation_strategies.py:201-214; agent.py:308-370 | "AgentFactory single authority" leaks: DirectProtocolAdapter and DirectCreationStrategy call `AgentOrchestrator.from_settings` directly; Agent.create() duplicates vertical resolution and wires via private attrs (`_te._verifier`, `_te._lsp_context_enabled`) | Two creation paths drift; private-attr poking bypasses factory step ordering | Route strategies through AgentFactory; add factory params for verifier/LSP wiring | M |
| F9 | P2 | agent.py:391-393; agent_factory.py:212-221 | Exception taxonomy classified by substring sniffing ("provider"/"api"/"key" in str(e)) at both creation boundaries | Error type flips on unrelated message wording | Dispatch on exception type/attributes; keep InitializationError.stages as the model | S |
| F10 | P2 | agent_factory.py:104-157 | `_apply_profile_overrides` monkey-patches the caller's Settings: `self._settings.load_profiles = lambda: profiles` (line 156) | Permanently mutates a shared config object the caller may reuse | Pass resolved profile payload explicitly to from_settings | S |
| F11 | P2 | agentic_loop.py:2583-2611, 2554-2581, 3305-3320 | Domain heuristics in generic framework: arXiv-paper counting, research-phase mapping, markdown-header synthesis checks | Violates "root framework stays generic; domain behavior lives in verticals" | Move to capability/vertical extension points (TaskTypeHint seam exists) | M |
| F12 | P3 | agent.py:150-161 | Agent.__init__ type-guards by class NAME string (`orchestrator_type_name != "AgentOrchestrator"`) | Blocks legitimate subclasses and typed test doubles; brittle across renames | isinstance against Protocol, or drop the guard | S |
| F13 | P3 | agent.py:677-736, 1420-1515; shim.py (555 LOC, fan-in=0) | Compat surface bloat: run_oneshot/run_interactive aliases, ChatSession delegate, FrameworkShim retained to 2027-06-30 | 3 parallel names for 2 behaviors | Sunset aliases in the next deprecation window (FEP-0015 pattern) | S |
| F14 | P3 | __init__.py:687 | `discover()` imports `victor.ui.commands.capabilities` — Framework → UI inversion | Layer-2 package importing layer-1 UI; unguarded direction | Move capability manifest to framework; UI imports it | S |

## Co-design opportunities
1. Snapshot policy co-designed with its only consumer: make the StateGraph executor request state_history explicitly; default invoke()/stream() to no copy.
2. Settings as an infrastructure contract: process-cached immutable snapshot + `add_change_listener` invalidation, consumed by graph.py, agent.py, client.py (needs infra buy-in).
3. One loop, one flag graduation: retire the legacy AgenticLoop while-loop + thin stream() (FEP-0007/FEP-0021 amendment).
4. Typed LoopState schema: replace ~20 magic state keys (incl. `*_obj` live-object/dict double representation) with one pydantic state model.
5. RL seam through contracts: expose prompt-evolution as a victor_contracts protocol (FEP-0025 Phase 6 candidate).
6. Factory as the only wiring point: verifier/LSP/budget wiring moves from Agent.create's private-attr pokes into AgentFactory steps.

## Keep intact
- Per-invoke collaborator construction (graph.py:315-354) — makes cached CompiledGraphs safely shareable across concurrent executions; do not "optimize" into instance state.
- Dated deprecation discipline (FrameworkShim → 2027-06-30, FEP-0015 precedent #805) and AST-based boundary guard tests.
- strict_edges opt-in default-off (graph.py:456-468) — fallthrough-to-END is load-bearing for existing graphs.
- RLCheckpointerAdapter (graph_checkpoint.py:85+) — bridges graph recovery onto the RL checkpoint store.
- InitializationError staged taxonomy (stage/suggestions/run_command) — extend it (F9) rather than replace it.

## Quick metrics
- rl/ alone = 37,826 LOC (27% of the package).
- agentic_loop.py: 4,376 LOC, ~75 methods, 44 `hasattr()` dispatch sites, ~20 magic state keys, run() spans 807-1468 (~660 lines); churn=101.
- `load_settings()` = uncached `Settings()` (settings.py:2573-2579); 11 framework call sites.
- Direct `AgentOrchestrator.from_settings` callers outside AgentFactory: 4+ (victor/__init__.py:41, integrations/protocol/adapters.py:66, agent/creation_strategies.py:204, agent/orchestrator_factory.py:514).
- `validate_after_nodes`/`validate_on_entry`: 0 runtime reads (tests only).
- Guard coverage: test_architectural_boundaries.py (18 tests, UI-scope only — framework→UI inversion at __init__.py:687 unguarded).
- FEPs on file: 31 (FEP-0007 loop unification still Draft-status seam).
