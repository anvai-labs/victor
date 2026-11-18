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
| FOU-D5c | Executor parity: streaming on `run_streaming` | D | — | FOU-D2 | MERGED | #1886 (turn-granularity chunks; terminal verdict contract identical; LoopGuards between events; D5b verification parity on the streamed path) |
| FOU-D5a | Executor parity: `total_timeout` via LoopGuards wrap | D | — | FOU-D2 | MERGED | #1884 (RED 3 → GREEN; LoopSpinError re-raised past the generic graph-error handler; legacy spin-guard FAIL contract) |
| FOU-D5b | Executor parity: verifier gate on the graph path | D | — | FOU-D2 | MERGED | #1885 (post-run `_run_verification`, same dict shape, bounded re-entry with feedback, budget-exhaustion reason; per-run retry reset) |
| FOU-D5f | `USE_STATEGRAPH_AGENTIC_LOOP` default flip | D | ADR-036 | FOU-D5a/b/c + parity + soak | BLOCKED → NEARLY UNBLOCKED (2026-11-17) | completion gap CLOSED live: with S6 (#1888) + a direct-LM-Studio profile (`soak-lmstudio`, added to profiles.yaml — bypasses the default profile's Sandhi routing that 404'd lmstudio), a real chat run COMPLETED (`EvaluateNode decision=COMPLETE score=0.9 reason=Q&A shortcut fired in production; success=True; clean termination`). Two remaining issues before the flip, both found in the 2026-11-17 soak: (a) final response text is empty at the surface ("stream_response returned empty content") — the Q&A answer doesn't reach the TUI (graph→surface content plumbing); (b) the CLI path still hits the inject_execution_context None-guard (services unavailable — S2's run_unified threading covers the ChatService surface, not the CLI direct path). Flip PR = flag default + these two fixes + opt-out negative test. |
| FOU-D5f-S1 | Soak fix: chat wrapper's pre-loop StateGraph refusal removed | D | — | FOU-D5c | this PR | wrapper-level test (mocked loop) + real CLI repro |
| FOU-D5f-S2 | Soak fix: `execution_context` handoff to the streaming loop's graph executor | D | — | FOU-D4 | this PR | constructor field + run_unified threading + inject None-guard; wrapper test asserts handoff |
| FOU-D5f-S3 | Soak fix: ACT node stores response text; evaluation coerces objects | D | — | — | this PR | 3 tests (node shape, default-eval complete, object-defense) |
| FOU-D5f-S4 | Soak fix: evaluate node uses the real `check_fulfillment` API | D | — | — | this PR | real-API + contained-failure tests; phantom-pinning test migrated |
| FOU-D5f-S5 | Soak fix: stale governance/verifier admission refusals removed (`require_supported_loop_executor`) | D | — | FOU-D5b/D5c | this PR | streaming-parity stategraph+gate tests (were the RED evidence) + admission seam test; documented boundary: governance enforced at the chat surface, direct graph-API runs ungated by design |
| FOU-S1a | Sandbox backend hardening (bwrap seccomp/egress; Seatbelt deny-default) | S | ADR-035 | FOU-D2 | IN REVIEW | fou/s1-sandbox-hardening |
| FOU-S1b | Sandbox posture flip to fail-closed + escape hatch | S | ADR-035 | FOU-S1a (qualification green on CI matrix) | TODO | — |
| FOU-S2 | Container `CodeSandbox` quality bar promoted to core | S | ADR-035 | FOU-S1b | TODO | — |
| FOU-S3a–d | Untrusted-content containment (envelope/canary → blocking → classifier → graduation) | S | ADR-037 | FOU-D2 | S3a IN REVIEW (S3b-d TODO) | fou/s3a-containment-envelope |
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

## FOU-S3a evidence (2026-11-17, fou/s3a-containment-envelope)

- **RED**: new containment module + seam tests failed collection on develop
  (module absent; seam unwrapped).
- **GREEN**: `victor/security/containment.py` — canonical provenance envelope
  (`<untrusted source= origin= canary= [flagged=]>`), per-session canary
  contextvar, closing-tag neutralization (envelope closes exactly once,
  negative-tested). Seam: `format_and_prune_tool_output` envelopes the
  model-bound output only; user preview stays raw. Escape hatch:
  `tool_settings.untrusted_content_envelopes` (default on, ADR graduation).
- **Ripple (4 test migrations)**: durable-resume receipt, tool-execution
  runtime publication, tool-service result processing x3 — assertions now
  semantic (payload inside exactly one envelope) instead of raw-equality.
- Selector targets green; affected suites clean; black/ruff clean.

## Reboot instructions

1. `git fetch origin develop`; check this table for `IN REVIEW`/`GREEN` rows and verify their
   PRs on GitHub before starting anything (avoid another session's active work).
2. One worktree per task off `origin/develop`; follow the per-task gates above.
3. Update this ledger at each material milestone and at session handoff, citing exact
   commits, test outcomes, review results, and CI runs.
4. Never equate local pass, merge, release, and live acceptance.

## FOU-S1a evidence (2026-11-17, fou/s1-sandbox-hardening)

- **RED (ADR-035 named)**: egress probe inside the unhardened bwrap wrapping
  (`socket.create_connection(('1.1.1.1', 53))`) returned rc=0 — network was reachable.
  New-test RED run: 15 failed + 5 errors on develop (`test_backends_hardening.py`,
  `test_backends_qualification.py`).
- **GREEN (live, WSL kernel, bubblewrap 0.9.0)**: hardened wrapping → `egress=DENIED
  unshare=EPERM` (network namespace denies egress; classic-BPF seccomp filter enforced);
  plain `echo` and shell-mode `git --version` runs stay green under the same wrapping.
  Seccomp program: 56-insn classic BPF, deny-list over mount/namespace/module/kexec/ptrace/
  bpf/keyring/handle/process_vm syscalls, x86_64 + aarch64 tables, unknown arch fails closed
  (RET ERRNO). Jump-math bugs caught by the live kernel rejection (EINVAL) and fixed against
  a full disassembly.
- **Unit**: 147 passed across sandbox/workflows/policy-snapshot/core-executor suites;
  selector targets green; 11 config failures pre-existing on clean develop (verified via
  stash, byte-identical).
- **Plumbing**: `pass_fds` threaded through `run_managed_process` + sync `subprocess.run`;
  backend `take_pass_fds()`; shell-mode branch covered by a regression test (fd dropped).
- **S1a follow-up (fou/s1a-seccomp-fix)**: post-merge adversarial pass vs the
  Omnigent reference found the shipped cBPF chain had fall-through offsets
  relative to `pc` instead of `pc+1` — the kernel silently skipped alternate
  deny checks (pivot_root enforced nowhere; several numbers plain wrong, e.g.
  x86_64 pivot_root is 155 not 203, kexec_file_load 320 not 294; aarch64
  unshare 97 not 266 — 266 is clock_adjtime, a collateral deny). Fixed with a
  symbolic-offset section emitter + kernel-truth tests: direct seccomp(2)
  load validation, full BPF emulator over the verified deny set (now derived
  from the k8s RuntimeDefault posture, ~60 syscalls/arch), clone(CLONE_NEW*)
  argument filter added, live probes `unshare=EPERM pivot_root=EPERM
  cloneNS=EPERM fork=OK`. ALSO: compat-ABI footgun (int $0x80) covered by
  unknown-arch deny-all.
- **Pending for S1b gate**: macOS CI-matrix qualification of the deny-default Seatbelt
  profile (local host is Linux; profile construction covered by unit tests only).

## FOU-D5f-S8 follow-up: live pickle diagnosis (2026-11-17)

Real graph-executor chat runs still fail flakily with
`cannot pickle 'TextIOWrapper' instances` (some runs complete — Q&A
shortcut verified live; others die). The failure does NOT route through
`copy.deepcopy` (a deepcopy-stack probe printed nothing on failing
runs). Diagnostic probe — drop as `sitecustomize.py` in a dir prepended
to PYTHONPATH; wraps all three entry points and prints the calling
frames for the first 3 failures:

```python
import copy, pickle, traceback, sys
_count = [0]
def _wrap(name, fn):
    def probed(x, *a, **k):
        try:
            return fn(x, *a, **k)
        except Exception as exc:
            if "pickle" in str(exc) and _count[0] < 3:
                _count[0] += 1
                frames = [f"{fr.filename}:{fr.lineno} {fr.function}"
                          for fr in traceback.extract_stack()[-14:-1]]
                sys.stderr.write(f"{name}-FAIL[{_count[0]}] {type(x)}\n")
                for f in reversed(frames):
                    sys.stderr.write(f"  {f}\n")
        raise
    return probed
copy.deepcopy = _wrap("DEEPCOPY", copy.deepcopy)
copy.copy = _wrap("COPY", copy.copy)
pickle.dumps = _wrap("PICKLEDUMPS", pickle.dumps)
```

Soak command: `PYTHONPATH=<probe-dir>:<worktree>
VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=true victor chat --profile
soak-lmstudio "Reply with exactly: SOAK-OK"`. Candidate sites: any
unguarded `copy.copy`/`pickle` in the streaming run plumbing;
`post_loop_recovery.py:117` deep-copies tool-call arguments. The flip
stays BLOCKED until a run completes with delivered content.
