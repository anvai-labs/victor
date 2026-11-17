# Victor Foundations Plan (FOU)

Status: Active — canonical work ledger for the [Victor Foundations Program](victor-foundations-program.md)
Created: 2026-11-16
Last updated: 2026-11-16

This ledger owns FOU task IDs, dependencies, status, TDD/smoke gates, evidence, and reboot
instructions. It follows the VAS tracker conventions (`victor-agent-service-plan.md`):

- Status vocabulary: `TODO` (no evidence), `RED` (regression demonstrably fails for the
  intended reason; failed evidence retained), `GREEN` (passing with commit hash), `IN REVIEW`,
  `MERGED`, `BLOCKED` (requires prerequisite + owner + unblock condition).
- Never equate local pass, merge, release, and live acceptance. RED is retained, not deleted.
- Specs live in FEPs/ADRs; this ledger records *work*, not decisions.
- One worktree per task off `develop` (`../victor-fou-<task>`), removed at merge.
- Preserve unmerged work outside `/tmp` before stopping.

## Status table

| ID | Task | Tier | Spec | Depends on | Status | Evidence |
|----|------|------|------|-----------|--------|----------|
| FOU-DOC1 | Program vision doc + this tracker + FEP-0047/FEP-0048 + ADR-035/036/037 | — | — | — | MERGED | #1878 |
| FOU-D1 | `GraphSQLiteCheckpointer` adapter (opt-in) + WAL/busy_timeout | D | FEP-0047 | FOU-DOC1 | MERGED | #1879 (RED ModuleNotFoundError → GREEN 7/7 + backend mirror suite; layering ratchet green; 94+17 affected) |
| FOU-D2 | Durable-by-default injection + retention/TTL + opt-out | D | FEP-0047 | FOU-D1 | IN REVIEW | this PR (factory `get_default_graph_checkpointer`; executor seam; settings `workflow_checkpoint_*` + env aliases; per-thread retention + `wf::`-scoped TTL sweep; smoke: cross-process resume zero re-execution + opt-out None) |
| FOU-D3 | FEP-0032 phase 2: loud input discard; FEP-0032 → Accepted | D | FEP-0032 | FOU-D2 | IN REVIEW | this PR (RED 3 tests → GREEN 23/23; Optional invoke/stream signatures + None short-circuit + guard in load_initial_state; resume convention `invoke(None, ...)` migrated across the suite; override = start_node + input) |
| FOU-D4 | Paused result surfacing (executor thread-id → LoopResult.paused → TaskResult status → wire envelope → /chat 202; `CompiledGraph.resume()`) | D | FEP-0032 | FOU-D3 | IN REVIEW | this PR (375 tests across 7 suites; StreamChunk passthrough rides response metadata today — chat_stream surface deferred to the FEP-0031 seam) |
| FOU-D5a | Executor parity: `total_timeout` via LoopGuards wrap | D | — | FOU-D2 | IN REVIEW | this PR (RED 3 → GREEN; same LoopGuards cooperative deadline outside the graph path; LoopSpinError re-raised past the generic graph-error handler; FAIL contract `terminated_by='loop_spin_guard'` matches legacy; verifier refusal retained for FOU-D5b) |
| FOU-D5b | Executor parity: verifier gate on the graph path | D | — | FOU-D2 | IN REVIEW | this PR (RED 3 → GREEN; post-run `_run_verification` on non-paused successful graph runs, same verification dict shape as legacy, bounded re-entry with `_inject_verify_feedback` in context, budget-exhaustion reason surfaced; per-run retry reset; refusal removed) |
| FOU-D5c | Executor parity: token streaming through ACT node | D | — | FOU-D2 | TODO | — |
| FOU-D5f | `USE_STATEGRAPH_AGENTIC_LOOP` default flip | D | ADR-036 | FOU-D5a/b/c + parity suite both modes + soak | TODO | — |
| FOU-S1a | Sandbox backend hardening (bwrap seccomp/egress; Seatbelt deny-default) | S | ADR-035 | FOU-D2 | TODO | — |
| FOU-S1b | Sandbox posture flip to fail-closed + escape hatch | S | ADR-035 | FOU-S1a (qualification green on CI matrix) | TODO | — |
| FOU-S2 | Container `CodeSandbox` quality bar promoted to core | S | ADR-035 | FOU-S1b | TODO | — |
| FOU-S3a–d | Untrusted-content containment (envelope/canary → blocking → classifier → graduation) | S | ADR-037 | FOU-D2 | TODO | — |
| FOU-S4 | Graduation batteries + default flips: `effect_gated_completion`, `per_turn_auditor` | S | ADR-036 | FOU-D2 (stable runs) | TODO | — |
| FOU-A1 | Subagent-spawn tool (Task-tool pattern) | A | ledger §A1 design | FOU-D2 | TODO | — |
| FOU-A2a–d | Core long-term memory (store → write paths → PERCEIVE injection → surfaces) | A | FEP-0048 | FOU-D2 | TODO | — |
| FOU-A3 | SCIP precise tier auto-on + impact-analysis benchmark publication | A | ADR-032 | — | TODO | — |
| FOU-A4 | RBAC/SSO wired into FastAPI middleware; per-user attribution | A | FEP-0020 (unblocks) | — | TODO | — |
| FOU-A5 | Research stack: search API adapters + page-read + browser tool; example cleanup | A | — | — | TODO | — |
| FOU-C1 | Single circuit breaker (merge `circuit_breaker.py` + `resilience.py` breaker) | C | — | — | TODO | — |
| FOU-C2 | Services-layer module map + Stage C continuation (items 24/27/29/31/32) | C | — | — | TODO | — |
| FOU-C3 | Native-parity CI broadened beyond tokenizer/context-fitter | C | — | — | TODO | — |
| FOU-C4 | Governance: FEP-0036 reviewer panels activated; committer rails | C | FEP-0036 | — | TODO | — |

## Per-task gates (expanded)

Each row carries the same gate structure; expand here once and reference by ID:

- **TDD**: write the RED test first (failing for the intended reason), retain the failed-run
  evidence, then GREEN with commit hash. Pin every behavior flip with a positive test (new
  behavior works) AND a negative test (opt-out restores prior behavior).
- **Smoke** (real server per the VAS rule — packaged client, not mocks-only): one runnable
  criterion per task, recorded here at GREEN.
- **Pre-push**: `black --check`, `ruff`, targeted affected tests,
  `pytest tests/ --collect-only -q`.
- **Review**: adversarial review before requesting merge for anything touching security,
  concurrency, shared state, or layer boundaries (CLAUDE.md rule). FOU-S* tasks always
  qualify.

### Key RED anchors (from the 2026-11-16 design pass)

- FOU-D1: `tests/unit/storage/checkpoints/test_graph_adapter.py` — save → fresh adapter →
  load deep-equal with `next_node` intact; ordered `list()`; missing-aiosqlite degradation;
  unknown-thread → `None`; two-writer interleaving (WAL).
- FOU-D2: `tests/unit/framework/test_agentic_loop_executor.py` — compiled graph has
  non-memory checkpointer when enabled, memory when disabled. Smoke: `wf::` row after a real
  chat turn; kill/restart resumes without re-executing the first node; opt-out writes nothing.
- FOU-D3: `tests/unit/framework/test_graph_interrupt_resume.py` — `invoke()` on a
  checkpointed thread with non-None `input_state` currently silently discards; must raise
  `ValueError` naming the thread. FEP-0028 team tests stay green.
- FOU-D4: `tests/unit/framework/test_wire_events.py` +
  `tests/unit/integrations/api/test_chat_resume_routes.py` — paused event carries
  `thread_id`/`next_node` on the `awaiting_approval` envelope; SSE stream terminates with it.
- FOU-D5a: `tests/unit/framework/test_agentic_loop.py` — `run(total_timeout=…,
  executor=STATEGRAPH)` currently raises `NotImplementedError` (agentic_loop.py ~879);
  GREEN = FAIL `LoopResult` with `LoopSpinError` metadata within the deadline.
- FOU-D5b: `tests/unit/framework/test_agentic_loop_verification.py` — identical
  `state["verification"]` metadata shape as the legacy path.
- FOU-D5c: `tests/unit/framework/test_agentic_loop_run_streaming.py` +
  `tests/integration/streaming/test_run_stream_parity.py` green in both executor modes.
- FOU-S3b: canary token appearing in any outbound tool argument blocks the action
  (fail-closed) and records the event — negative corpus test.

### Smoke ladder (run at each tier boundary)

Tier D: durable-resume-across-restart smoke (DONE for D2: two-process resume, zero re-execution, opt-out writes nothing); paused SSE → `/chat/resume` completion smoke;
executor parity transcript diff (`tests/integration/agentic_graph/test_parity.py`) both modes.
Tier S: unsandboxed-refusal smoke (no policy + unavailable backend → actionable error);
opt-out smoke; injection-corpus containment smoke. Tier A: memory recall across two sessions;
subagent spawn smoke; `/memory` surface smoke.

## §A1 Subagent-spawn tool design sketch (FOU-A1)

No FEP required (new tool, not a framework API change). Sketch for implementation:

- LLM-callable `spawn_subagent(prompt, tools?, model?, background?)` registered in
  `victor/tools/`; builder wiring at `victor/agent/factory/tool_builders.py` (same seam as
  `PermissionPolicy`, ~line 257) + `AgentFactory.create()`.
- Substrate: team coordinator member execution + member checkpoints
  (`victor/teams/unified_coordinator.py`) reused for lifecycle; delegate contract
  normalization (`victor/teams/delegate_contracts.py`) reused for hand-off.
- Foreground mode returns the subagent's final result to the calling loop; background mode
  returns a handle + notification via the existing event stream, with results retrievable by
  a `subagent_status`/`subagent_result` companion tool.
- Recursion and budget policy in `victor/config/task_tool_config.yaml` (max depth 1 by
  default; per-task-type tool budgets).
- Permission model: subagent inherits the parent's `PermissionMode`; escalation inside a
  subagent surfaces through the parent's HITL approval path (`victor/agent/hitl.py`).
- Negative tests: depth budget exceeded → refusal; subagent cannot escalate beyond parent
  mode; orphan cleanup on parent cancellation.

## Reboot instructions

1. `git fetch origin develop`; check this table for `IN REVIEW`/`GREEN` rows and verify their
   PRs on GitHub before starting anything (avoid another session's active work).
2. One worktree per task off `origin/develop`; follow the per-task gates above.
3. Update this ledger at each material milestone and at session handoff, citing exact
   commits, test outcomes, review results, and CI runs.
4. Never equate local pass, merge, release, and live acceptance.
