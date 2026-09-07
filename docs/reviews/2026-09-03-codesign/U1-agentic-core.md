# U1 — Agentic core (`victor/agent/` + `victor/runtime/`)

412 modules / 202,631 LOC / 5,295 touching commits (12m). Reviewer: parallel unit review, 2026-09-03.

## Overview
`victor/agent/` (+`victor/runtime/`, 6 files) is the execution heart: orchestrator facade → 6 canonical services → 24 `*_runtime.py` helpers → pipeline/store/streaming subsystems. Health: **mixed — effects are genuinely out of the facade (no direct provider/DB calls in orchestrator.py), but the Chat service is a handler-injection shell, so decomposition is ~half real.** Two hot-path data-movement defects (per-iteration full-history `model_dump()`, sequential tool execution behind a dead parallel flag) are the top perf risks; churn concentrates in exactly the files the decomposition was supposed to shrink. Prior backlog F-002/F-005/F-008 confirmed still open.

## Findings

| ID | Sev | Location | Finding | Why it matters | Fix sketch | Effort |
|----|-----|----------|---------|----------------|------------|--------|
| U1-1 | P1 | victor/agent/services/turn_execution_runtime.py:1610-1614 | Every loop iteration does `[msg.model_dump() for msg in self._chat_context.messages]` — pydantic-serializes the ENTIRE history for tool selection; the consumer uses only last 2 messages (victor/storage/cache/generic_result_cache.py:127-160) | O(history) CPU + alloc per model turn on the hottest path; pure data movement across a service seam | Pass `messages[-2:]` (or `(depth, recent2_tuple)`) — selector signature change, internal to agent | S |
| U1-2 | P1 | victor/agent/tool_pipeline.py:2237 (seq loop); victor/config/settings.py:1783 | Canonical loop executes tool calls strictly sequentially; `parallel_tool_execution=True` (default) + `max_concurrent_tools` are dead — `execute_tool_calls_parallel`, `ParallelToolExecutor.execute_parallel`, `AsyncToolExecutor` have zero live-path callers; docstring at :2457 falsely claims internal parallelization | Tool-heavy turns are latency-bound (subprocess/HTTP), wall-clock = sum of N tool latencies; settings promise silently unfulfilled | Route batch through `execute_tool_calls_parallel` gated by read-only/intent classification (action_authorizer already models intents); fix docstring | M |
| U1-3 | P1 | victor/agent/services/turn_execution_runtime.py:442,444,446,457,466 | Background-task completion messages use `\\n` (literal backslash-n) in f-strings — model receives literal `\n` sequences, not newlines | Corrupts model-visible formatting exactly when long-running tasks finish; 5 copy-propagated sites | Replace `\\n`→`\n`; test asserting no literal `\n` in injected content | S |
| U1-4 | P2 | victor/agent/orchestrator.py:766-782; services/chat_service.py:128-181,238-243 | ChatService is nominal: `bind_runtime_components` injects 8 orchestrator handlers; "runtimes" are built with the orchestrator itself and call facade privates (`orch._parse_and_validate_tool_calls`) | Facade-only law holds textually but not structurally; every chat feature re-touches orchestrator+service+runtime (churn=330) | Invert: ChatService owns turn lifecycle, receives frozen per-turn state; move `_prepare/_teardown_chat_service_turn_runtime` into ChatService. Guard-test upgrade (U1-11) first | L |
| U1-5 | P2 | victor/agent/conversation/assembler.py:365 (also :339 per-iteration rescore) | `older_messages.index(msg)` inside selection loop — O(n²) with pydantic `==` per LLM call; full re-score/re-sort every iteration, no cross-iteration reuse | CPU grows quadratically in long sessions exactly when context is largest | Enumerate with indices before sorting; cache older-message scores per turn, invalidate on compaction | S |
| U1-6 | P2 | victor/agent/conversation/store.py:212-213,1142,1386; services/chat_service.py:1410-1425 | Split-lock hazard: async writers take `asyncio.Lock`, sync writers take `threading.Lock` — no cross-domain exclusion on the same sqlite file; live persistence is unordered `run_in_executor(None,…)` fire-and-forget; the properly-locked `add_message_async` has zero production callers | Concurrent turns can interleave session token read-modify-writes and persist messages out of order | Adopt `add_message_async` on the live path or delete it; single lock domain | M |
| U1-7 | P2 | victor/agent/orchestrator.py (whole; :472-687) | God-file persists: 4,225 LOC, ~230 defs, 9 `_initialize_*` methods (F-002), 115 `getattr(self,…)` probes, 322 commits/12m, fan_in=39 | Highest churn×size in the repo; every feature PR risks facade re-growth | Ratchet guard on def-count; move property blocks (~800 lines) to mixins | M |
| U1-8 | P2 | victor/agent/orchestrator.py:791; :3105,3114 | DI init writes a service private (`self._provider_service._current_provider = self.provider`); `add_message` reaches into `conversation._messages` with its own O(n) trim at ceiling 100 vs MessageHistory's 100,000 (message_history.py:130) | Two divergent trim invariants; private reach-ins defeat the protocol layer | Add ProviderService.bind; expose `MessageHistory.trim(max)` | S |
| U1-9 | P2 | victor/agent/message_history.py:36-111,196-208; orchestrator.py:2096-2117 | Triple shipped defensive type-scanning: `_TrackedList` ("temporary instrumentation") + full-history isinstance scan after every add + re-normalization on every assembled request | O(history) waste per message and per LLM call paying forever for a one-time historical corruption | Assert Message type at the 2 write boundaries; delete all three scans | S |
| U1-10 | P2 | chat_stream_helpers.py (1,994, fan_in=1) + chat_stream_executor.py (1,657) + chat_stream_runtime.py (547) + streaming/handler.py (1,652) | Streaming ACT spread over 4 files as a mixin-with-orchestrator cluster; combined churn 170+/12m | Primary UX surface is the least coherent module family | Fold helpers into executor as explicit functions taking a context struct | M |
| U1-11 | P3 | tests/unit/agent/services/test_service_layer_validation.py:116-206 | Facade guard is string-grep: counts `_chat_service` occurrences, asserts protocol names in source | Cannot detect handler-injection (U1-4), service-private writes (U1-8), or effectful additions | AST/import-based guard: forbid orchestrator imports of DB/provider effect modules; ratchet handler-injection kwargs | M |
| U1-12 | P3 | victor/agent/* (132 hits `getattr(self.settings,…)`) | Settings read defensively with inline defaults everywhere | Typo'd flag names silently take the default — flag-drift bugs invisible (a real flag was deleted in #806) | Typed accessor helpers on services; lint rule for new raw getattrs | M |
| U1-13 | P3 | victor/agent/conversation/assembler.py:397 | `result.insert(len(result), msg)` — obfuscated append | Readability | `result.append(msg)` | S |

## Co-design opportunities
1. Kill per-iteration serialization at the tool-selection seam (U1-1 + U1-5): selector wants `(query, last-2, depth, stage)`; give it exactly that and cache scores per turn.
2. Parallelize the ACT tool batch behind intent classification — reuse `action_authorizer` (fan_in=23) as the parallel-safety gate for U1-2.
3. Single per-turn state handoff: one frozen `TurnState` replacing the setup/teardown handler pair and per-call dependency gathering.
4. Assembly memoization keyed on `(history_len, budget, query)`, invalidated on compaction/edit — amortized O(new messages).
5. Guard-test upgrade (U1-11) as the ratchet for U1-4/U1-7 before the L refactor.
6. Streaming ACT consolidation (U1-10).

## Keep intact
1. KV/prompt-cache stability machinery: frozen system prompt, `_sort_tools_for_kv_stability`, system-nudge interception into the user prefix (orchestrator.py:3103-3109, 3447-3465) — load-bearing for byte-identical prefixes.
2. `enhanced` completion default + judge pinning gate (judge_calibration_gate.py:188-213); effect-gated completion default OFF — concluded programs; no graduation proposals.
3. LRUToolCache with edit-driven invalidation (tool_pipeline.py:305-421) — bounded, TTL'd, file-state-aware.
4. `_ProgressiveToolsProxy` lazy registration (orchestrator.py:348-407) — amortizes registry cost out of the turn path.
5. Deprecation warnings on orchestrator.chat/stream_chat (orchestrator.py:3171, 3688) — deliberate v2.0 removal path.

## Quick metrics
- services/: 57 files, 36,537 LOC; 24 are `*_runtime.py` twins (F-005 said 23 — grew); 28/57 reference "orchestrator"; orchestrator attr-derefs inside services ≈ 218.
- Orchestrator: 4,225 LOC, ~230 defs, 9 `_initialize_*`, 115 `getattr(self,…)`, 18 `hasattr`; zero direct `provider.chat`/DB calls.
- Dead-on-live-path: `execute_tool_calls_parallel`, `AsyncToolExecutor`, `add_message_async`, `parallel_tool_execution`/`max_concurrent_tools` settings.
- Bounded collections verified: usage_analytics (1,000/tool, 500/provider), streaming session history (100), LRUToolCache (50+TTL).
- Brief correction: `victor.agent.streaming.pipeline` (churn=60, loc=11) is a tombstone raising ImportError — historical-churn artifact, not a live hotspot.
