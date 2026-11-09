# Seam-duplication audit — 2026-11-09

Read-only audit at develop dcd873d08. Motivation: a consumer campaign
(reporeel) hit four separate config-threading failures and one budget-owner
divergence (a run refused at ~55-60 tool calls against a funded 134) — each
fixed as a separate hand-patch (#1721-#1725). This audit maps the duplicated
seams behind those patches and proposes a smallest-independent-first
consolidation series with behavior-parity gates. Consolidation PRs should
cite their cluster here and land after the budget/gate emergency fixes.

---

# Victor seam-duplication audit — 2026-11-09 (read-only, @ develop dcd873d0)

Ranking (measured damage × breadth): Cluster 1 (budget owners) > Cluster 4
(run-context propagation) > Cluster 3 (termination heuristics) > Cluster 2
(loop-config threading). Estimated 5-7 PRs, smallest-independent-first.

## Cluster 1 — BUDGET OWNERS (rank 1; measured: run refused at ~55-60 calls vs funded 134)

10 owners/envelopes mapped:
1. Pipeline ceiling — tool_pipeline.py:110, default 25 (tool_selection_defaults.py:170);
   enforced :2207 (batch admission), :2451, :2992-3003 (_budget_exhausted_skip).
   COUPLING SURPRISE: same config value doubles as output-aggregator
   max_results (:676, :775-777) — a budget raise silently changes output truncation.
2. ToolService runtime — tool_service.py:944-947 is_budget_exhausted →
   tool_budget_runtime.py:58-68 get_limit() (prefers pipeline/tool_budget else
   BudgetManager.max_budget), default 100. set_tool_budget (:1025) never resets calls_made.
3. Mutator — runtime_overrides.py:43-77 budget_changes(): hasattr-guarded
   fan-out over orchestrator/task_coordinator/_tool_service/pipeline.config;
   does NOT touch UnifiedTaskTracker, StreamingChatContext, ToolContext,
   SessionStateManager. Streaming variant passes fallback=None.
4. Orchestrator/session — orchestrator.py:572 initialize_tool_budget →
   tool_builders.py:666-684 max(adapter.recommended_tool_budget, 50) then
   settings.tool_call_budget (default 10000). SessionStateManager has its own
   is_budget_exhausted (:323-329).
5. TurnExecutor per-turn fallback — turn_execution_runtime.py:754-756,769:
   overrides.get("tool_budget") else self._tool_context.tool_budget
   (adapter mirror, floor 50-60 band).
6. UnifiedTaskTracker — default 250, advisory only.
7. Streaming — StreamingChatContext.tool_budget default 200; caller override
   NEVER reaches it. Enforcement streaming/tool_execution.py:346-390.
8. Loop-level — iteration binding (#1723), topology planning default 10,
   topology_grounder re-injects tool_budget into state overrides (:69-73).
9. BudgetManager/BudgetTracker multipliers (effective_max = base×model×mode×productivity).
10. Team division — budget split per member (learner.py:96).

DIVERGENCE MECHANISMS (all live):
(a) funding scoped per-turn and ROLLED BACK at turn end (:736-737 apply, :958 restore);
(b) cumulative counters vs per-run funding: _calls_used / calls_made accumulate
    across the session; set_tool_budget does not reset usage → remaining =
    134 − prior usage ≈ 55-60 (the observed symptom);
(c) hasattr-conditional application silently no-ops for not-yet-created owners;
    execute_turn fallback reads the adapter-floor value.

Proposed: ToolBudgetLedger (one object per run: ceiling/used/scope); the
override is the only funding write; every enforcement site reads through it;
kill the hasattr fan-out + per-turn budget rollback; decouple max_results.
PARITY: refusal bytes + outcome kinds + reset semantics byte-identical
without overrides. RISK medium; flag-gated fallback reads.

## Cluster 4 — RUN-CONTEXT PROPAGATION (rank 2; streaming drops ALL caller funding/criteria)

Drop points: (1) Agent.stream doesn't accept runtime_context_overrides;
stream_message_events/iter_runtime_stream_events forward neither. (2) chat
stream=True path → _prepare_stream never reads overrides/run_context.
(3) Topology REPLACES instead of merging (stream_ctx.runtime_context_overrides
= dict(topology_overrides)). (4) FEP-0043 (#1721/#1724/#1725) is
BUFFERED-ONLY: StreamingChatContext has no criteria field, streaming config
omits enable_criteria_gate, loop.run_streaming gets no context kwarg.
(5) _execute_agentic_loop signature sniffing silently drops both kwargs on
legacy executors. Proposed: CallerRunContext dataclass threaded as ONE arg
through run/stream → loop; streaming merges topology ONTO it (shared merge
helper). PARITY: buffered funded-run LoopResult/chunks identical; streaming
behavior change gated behind explicit opt-in.

## Cluster 3 — TERMINATION HEURISTICS (rank 3; #1725 suppression duplicated at 3 sites; streaming unsuppressed)

SpinDetector consumers ×2 (legacy evaluator agentic_loop.py:3405-3440 +
shared _record_spin_and_ratchet :2808); criteria-guided suppression
duplicated at THREE sites (agentic_loop.py:3412-3430, :3853+:3900-3906,
enhanced_completion_evaluation.py:385-430); plateau at :3834-3918; enhanced
scoring 'Insufficient progress' retry template duplicated in planning_runtime.py:1679;
_criteria_budget_exhausted set at SIX loop sites; streaming blocked-loop
tracking is a SEPARATE reimplementation (streaming/coordinator.py:333-424,
tool_execution.py:611-615) with NO criteria-guided suppression (divergence:
streaming hard-terminates on blocked streaks while buffered continues).
Proposed: TerminationAuthority owning spin+plateau+criteria+budget state →
single Decision (TERMINATE/GUIDE/YIELD); streaming shares SpinDetector +
the guided predicate at minimum. HIGHEST RISK — do last, golden-transcript
parity (spin-FAIL no-criteria, plateau, guided CONTINUE, yield boundary).

## Cluster 2 — LOOP-CONFIG THREADING (rank 4; the enabler)

Hand-built dicts: buffered turn_execution_runtime.py:1245-1252
{executor, completion_strategy, enable_effect_gate, enable_criteria_gate};
streaming chat_stream_executor.py:1594-1613 — NO enable_criteria_gate, no
criteria/complete_on_key, max_iterations default asymmetry (10 vs 50).
AgenticLoopConfig exists (:502-560) with from_dict passthrough;
criteria_gate_max_downgrades threaded by NO caller. completion_tool_names
travels via ToolPipelineConfig (two independent resolutions of the same
declaration). Mechanism inventory: dict keys vs constructor args vs settings
registry vs override keys vs dual loop-state keys. Proposed: CallerRunContext
dataclass constructed once at Agent.run/stream; AgenticLoop takes it as a
constructor arg. PARITY: from_dict field-identical for every caller shape;
golden buffered run unchanged.

## Cluster 5 — small threading-relevant items
- Two divergent override-application impls (buffered vs streaming) — consolidate.
- Per-turn budget apply/restore vs streaming one-shot — belongs to the ledger PR.
- config.tool_budget doubling as max_results — split.
- Six _criteria_budget_exhausted write sites — fold into one per-iteration hook.

## SUGGESTED SERIES (smallest independent first)
PR 1 budget observability (no behavior change; (owner, ceiling, used) triad
    at every refusal site — pins the 55-60 culprit with the reporeel repro).
PR 2 ToolBudgetLedger (Cluster 1 consolidation; flag-gated fallback).
PR 3 streaming run-context parity (Cluster 4; opt-in for streaming, no-context byte-identical).
PR 4 CallerRunContext (Cluster 2; depends on PR 3).
PR 5 TerminationAuthority evaluation-side (Cluster 3; golden parity).
PR 6 optional: streaming termination unification.
