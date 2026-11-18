# Victor Foundations Plan (FOU)

Status: Active — canonical work ledger for the [Victor Foundations Program](victor-foundations-program.md)
Created: 2026-11-16
Last updated: 2026-11-18

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
| FOU-D2 | Durable-by-default injection + retention/TTL + opt-out | D | FEP-0047 | FOU-D1 | MERGED | #1881 (factory `get_default_graph_checkpointer`; executor seam; settings `workflow_checkpoint_*` + env aliases; per-thread retention + `wf::`-scoped TTL sweep; smoke: cross-process resume zero re-execution + opt-out None) |
| FOU-D3 | FEP-0032 phase 2: loud input discard; FEP-0032 → Accepted | D | FEP-0032 | FOU-D2 | MERGED | #1882 (RED 3 tests → GREEN 23/23; Optional invoke/stream signatures + None short-circuit + guard in load_initial_state; resume convention `invoke(None, ...)` migrated across the suite; override = start_node + input) |
| FOU-D4 | Paused result surfacing (executor thread-id → LoopResult.paused → TaskResult status → wire envelope → /chat 202; `CompiledGraph.resume()`) | D | FEP-0032 | FOU-D3 | MERGED | #1883 (375 tests across 7 suites; StreamChunk passthrough rides response metadata today — chat_stream surface deferred to the FEP-0031 seam) |
| FOU-D5a | Executor parity: `total_timeout` via LoopGuards wrap | D | — | FOU-D2 | MERGED | #1884 (RED 3 → GREEN; same LoopGuards cooperative deadline outside the graph path; LoopSpinError re-raised past the generic graph-error handler; FAIL contract `terminated_by='loop_spin_guard'` matches legacy) |
| FOU-D5b | Executor parity: verifier gate on the graph path | D | — | FOU-D2 | MERGED | #1885 (post-run `_run_verification` on non-paused successful graph runs, same verification dict shape as legacy, bounded re-entry with `_inject_verify_feedback` in context, budget-exhaustion reason surfaced; per-run retry reset) |
| FOU-D5c | Executor parity: streaming on `run_streaming` | D | — | FOU-D2 | MERGED | #1886 (turn-granularity chunks; terminal verdict contract identical; LoopGuards between events; D5b verification parity on the streamed path) |
| FOU-D5f | `USE_STATEGRAPH_AGENTIC_LOOP` default flip | D | ADR-036 | FOU-D5a/b/c + parity + soak | MERGED | #1904 (flip + opt-out negative test; superseded #1897/#1898; soak fixes S1–S5 rode the #1887→#1890/#1897 arc; live-soak completion + content delivery proven 2026-11-17; Tier D closed — #1878–#1904 banked, 0.31.0 promoted) |
| MERGED | #1887 arc (wrapper-level test + real CLI repro) |
| MERGED | #1887 arc (constructor field + run_unified threading + inject None-guard; wrapper test asserts handoff) |
| MERGED | #1887 arc (3 tests: node shape, default-eval complete, object-defense) |
| MERGED | #1887 arc (real-API + contained-failure tests; phantom-pinning test migrated) |
| MERGED | #1890/#1897 arc (streaming-parity stategraph+gate tests + admission seam test; documented boundary: governance enforced at the chat surface, direct graph-API runs ungated by design) |
| FOU-S1a | Sandbox backend hardening (bwrap seccomp/egress; Seatbelt deny-default) | S | ADR-035 | FOU-D2 | MERGED | #1907, #1910 (kernel-truth cBPF fix; live probes EPERM, fork=OK) |
| FOU-S1b | Sandbox posture flip to fail-closed + escape hatch | S | ADR-035 | FOU-S1a | MERGED | #1912 (8 adversarial findings fixed pre-merge; suite opt-out via tests/conftest.py). macOS Seatbelt live qualification COMPLETE 2026-11-18: 13/13 probe checks on macOS 26.2/arm64 — one real profile gap found+fixed (`/dev/null` write allow, #1944); probe pattern in tests/unit/tools/sandbox/test_seatbelt_qualification.py |
| FOU-S2 | Container `CodeSandbox` quality bar promoted to core | S | ADR-035 | FOU-S1b | TODO | — |
| FOU-S3a–d | Untrusted-content containment (envelope/canary → blocking → classifier → graduation) | S | ADR-037 | FOU-D2 | S3a+S3b GREEN (4af31f541); S3c-d TODO | #1908, #1911 |
| FOU-S4 | Graduation batteries + default flips: `effect_gated_completion`, `per_turn_auditor` | S | ADR-036 | FOU-D2 (stable runs) | PART-BLOCKED | `effect_gated_completion` graduation LOCKED OUT by the 2026-08-03 negative A/B (on 0.847 vs off 0.864, overlapping CIs — do NOT graduate, codified #884); `per_turn_auditor` battery = executable remainder |
| FOU-A1 | Subagent-spawn tool (Task-tool pattern) | A | ledger §A1 design | FOU-D2 | TODO | — |
| FOU-A2a–d | Core long-term memory (store → write paths → PERCEIVE injection → surfaces) | A | FEP-0048 | FOU-D2 | TODO | — |
| FOU-A3 | SCIP precise tier auto-on + impact-analysis benchmark publication | A | ADR-032 | — | TODO | — |
| FOU-A4 | RBAC/SSO wired into FastAPI middleware; per-user attribution | A | FEP-0020 (unblocks) | — | TODO | — |
| FOU-A5 | Research stack: search API adapters + page-read + browser tool; example cleanup | A | — | — | TODO | — |
| FOU-C1 | Single circuit breaker (merge `circuit_breaker.py` + `resilience.py` breaker) | C | — | — | IN PROGRESS | 2026-11-18 inventory: ≥5 breaker shapes — core/circuit_breaker.py (config+error), providers/circuit_breaker.py (canonical CircuitBreaker+Registry), providers/resilience.py (CircuitOpenError extends CanonicalCircuitBreakerError), agent/resilience.py MultiCircuitBreaker, agent/recovery/fallback.py ModelCircuitBreaker, workflows/data_pipeline.py CircuitBreaker |
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

## FOU-S3b evidence (2026-11-18, fou/s3b-canary-blocking, stacked on S3a)

- **RED**: canary-blocking tests failed collection on the S3a branch
  (find_canary_in_value absent; pipeline unwired).
- **GREEN**: `find_canary_in_value` (deterministic recursive scan, dotted
  path reporting, exact-token match so the bare prefix is not a hit);
  pipeline admission check after the permission gate, before budget:
  canary in outbound arguments -> skip result
  (outcome_kind=canary_exfiltration, block_source=containment,
  retryable=False, zero execution), model-visible reason does NOT echo
  the token, logger.error records the attempt. Escape hatch:
  tool_settings.canary_blocking_enabled (default on per ADR graduation).
- Live-path proof: blocked call never reaches executor
  (executor.assert_not_awaited); clean args + no-session-canary +
  disabled-knob negative tests.
- Selector targets + tool-pipeline ripple suites green (997 passed).
## FOU-A0 evidence: codegraph intent tier (2026-11-18, FEP-0049)

Parallel track to the FOU ladder (adoption/X-axis capability). The original
Cursor-comparison framing ("Victor holds structure, Cursor holds intent") was
stale on arrival and is corrected here: Victor already had semantic search —
the symbol-summary embedding tier (name + signature + docstring, #771, 2026-08)
— it just lacked (a) BODY-level chunk embeddings, (b) a persisted vector index
for them, and (c) incremental re-embedding. What Cursor's architecture actually
contributed was the incremental-sync DISCIPLINE (Merkle-tree-of-hashes), not
intent itself. This program shipped all three missing pieces, then simplified:
the Merkle projection is a derived view over the pre-existing flat
path -> sha256 manifest (mtime+hash discipline retained, root shortcut = O(1)
skip; the flat diff stays the authoritative work set), and the vector store
consolidated SQLite-only (#1943) — chunk vectors co-located in project.db,
lancedb/pyarrow off the required path (numpy brute-force measured sufficient:
0.1 ms @ 14K, 3.4 ms @ 100K). Significance was MEASURED, not assumed: on the
victor corpus (13,973 chunks) intent adds P@3 +13% / MRR +0.165 over keyword,
wins 8/15 queries, with near-disjoint miss sets (complementary legs); on
genmedia (559 chunks) miss rank is 8x closer (26 vs 198). Victor now holds
structure AND intent; the honest remaining gap vs Cursor is indexing UX
(automatic shadow indexing on clone/edit, apply-model polish), not retrieval
capability.

- **FEP-0049** (feps/fep-0049-intent-tier-merkle-sync.md): two-tier embedding
  model (symbol-summary shipped since #771; chunk-source new), Merkle
  discipline derived from the flat manifest, vectors in the graph sqlite
  (`intent_chunks`, #1943 — the FEP's original LanceDB code_chunks table was
  consolidated away), intent_file_state journal, ADR-036 graduation plan.
- **Phase 1 (#1915, victor-codegraph 0.15.0, MERGED)**: merkle.py
  (directory->file->CHUNK leaf granularity; minimal frontier diff; O(1) root
  shortcut; chunk-precise changed_chunks work set) + CHUNK_SOURCE_PROFILE +
  embed_chunks with the SymbolEmbedding staleness discipline. 290 package
  tests green.
- **Phase 2 (#1916, MERGED into the phase-1 branch)**: IntentIndexStore
  (LanceDB code_chunks + intent_file_state in graph sqlite; crash-safety by
  commit ordering; symbol_oid correlation; embedding_ref never written —
  guard test); codebase_intent_indexing_enabled flag. 9 tests.
- **Phase 3 (#1917, MERGED)**: intent_sync.py sync_intent_tier
  (O(1) root shortcut, chunk-precise delete-then-upsert, marker-last crash
  ordering, lazy encoder, profile-change rebuild) + GraphIndexingPipeline
  hook (post-publication, flag-gated) + graph index --intent CLI. 7 work-set
  tests.
- **Phase 4 (#1919, MERGED)**: seeding intent_provider_for/intent_seed_scores
  (chunk hits max-aggregated per symbol, rank-normalized) + hybrid_seeds
  flag-gated intent leg (pairwise RRF; flag-off byte-identical) +
  graph_query_tool mode="intent" + IntentIndexStore.search.
- **FEP-0049 COMPLETE on develop (2026-11-18)**: `victor graph index
  --intent` populates; `graph_semantic_search(mode="intent")` + the hybrid
  intent leg query. Remaining: ADR-036 graduation battery (field data).
- **SQLite-only intent tier (#1943, MERGED 2026-11-18)**: chunk vectors
  co-located in project.db (`intent_chunks` table, float32-LE BLOB) via
  `SqliteIntentChunkStore` — lancedb/pyarrow off the intent tier's
  required path (numpy brute-force: 0.1ms @ 14K, 3.4ms @ 100K rows).
  `IntentIndexStore.search` read-path fixed (was AttributeError on every
  mode="intent" query). `build_intent_index` (seeding.py) is the single
  construction authority for read+write. `intent_file_state` journal kept
  (zero-chunk-file tombstones; marker-last crash contract). LanceDB stays
  for the symbol-summary tier (separate migration).
- **Fan-out byte budgets (#1945, MERGED 2026-11-18)**: see FOU-C1-adjacent
  TD-03 follow-up — `max_fanout_input_bytes` (pre-effect typed rejection)
  + `max_fanout_result_bytes` (observability only).


## FOU-S1b evidence (2026-11-18, fou/s1b-posture-flip)

- **RED**: 11 of 15 new posture tests failed on develop (default was
  disabled/preferred; no env hatch; no posture surface).
- **GREEN**: `effective_sandbox_policy` resolution order = explicit
  settings policy > VICTOR_SANDBOX_POLICY env (invalid values ignored) >
  **required** (default). Legacy `sandbox_enabled` bool no longer maps to
  fail-open preferred. Explicit opt-out warns once per process (ADR-035
  decision 2). Refusal errors name the exact escape hatch.
- Snapshot: `RunPolicySnapshot._effective_policy` now DELEGATES to the
  canonical function (hand-synced copy removed — the F1 drift class).
- API surface: `/system/status` reports `sandbox_posture`
  (StatusResponse.sandbox_posture, current_sandbox_posture() helper).
- Ripple containment: tests/conftest.py sets VICTOR_SANDBOX_POLICY=disabled
  (CI runners cannot enforce OS sandboxes; dedicated flip tests delenv to
  exercise required); 4 legacy tests migrated to the new semantics.
- Wide sweep: 1,931 passed / 0 failed across sandbox/runtime/workflows/
  core/api/integrations/agent-tools; live executor smoke on this host
  (required posture, bwrap + seccomp enforcing): shell runs wrapped, rc=0.
- Follow-ups ALL CLOSED (2026-11-18): #1918 doctor posture check;
  #1920 --dangerously-unsandboxed; #1921 + #1944 macOS live qualification
  (deny-default profile 13/13 on macOS 26.2 arm64; /dev/null write allow
  was the one real gap found).
## Adoption Increment 2 + inventory fix (2026-11-18)

- **Increment 2 (#1922, MERGED 2026-11-18)**: `victor init` quickstart — server
  detection matrix (Ollama/llama.cpp/InferFlux), Ollama pull path reusing
  `OllamaProvider.pull_model`, last-resort llama.cpp daemon (official
  release binary + pinned bartowski Q4_K_M GGUF, detached spawn with
  PID/log files, idempotent PID+health re-entry, path-traversal-rejecting
  extraction), ProfileManager activation, SOAK-OK smoke via the
  OpenAI-compatible transport. Coordinator adversarial pass added
  `expected_min_bytes` truncation guards on both downloads (a partial
  4.4 GB GGUF previously landed silently). 47 tests. Model choice:
  qwen2.5-coder-7b deliberate (stable GGUF, --jinja tool-calling, CPU
  size, Ollama-path consistency); Qwen4 is preview-only — swap is two
  constants when production GGUFs land.
- **Inventory fallback (#1923, MERGED 2026-11-18)**: root cause of 249
  pre-existing test failures — ancestry discovery accepts ANY ancestor
  `.git` including empty debris (stray /tmp/.git), but git then rejects
  it and the walk fallback required zero discovery. Git's verdict is now
  authoritative: not-a-repository -> walk the pinned descriptor dir
  (same trust position; other git failures still fail closed).
  1399 passed / 0 failed in the previously-failing suites.
- **S1b follow-ups ALL CLOSED**: #1918 doctor posture check,
  #1920 --dangerously-unsandboxed (CLI > env > file precedence,
  fail-closed on settings failure), #1921 Seatbelt battery MERGED with
  7/7 PASSED execution evidence on macOS 26.2 arm64 (SSH run on the
  MacBook). S1b gate: QUALIFIED on Linux AND macOS.

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
