# Victor Foundations Plan (FOU)

Status: Active — canonical work ledger for the [Victor Foundations Program](victor-foundations-program.md)
Created: 2026-11-16
Last updated: 2026-11-19 (FOU-A4 design landed)

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
| FOU-S2 | Container `CodeSandbox` quality bar promoted to core | S | ADR-035 | FOU-S1b | IN REVIEW | #1976: core owns the bar at `victor/tools/sandbox/container_sandbox.py` — `ContainerSandbox` (lazy acquire / owned finalizer / idle-TTL self-expiry with `auto_remove` / network-disabled default / `mem_limit`+`cpu_shares` caps on the docker-run invocation / owner labels + reapable) + capability gating `resolve_container_sandbox(settings, required=)` mirroring `resolve_sandbox_backend` (None when Docker unavailable — backend simply not offered; typed refusal reason `binary_missing`/`daemon_unreachable` under required) + `daemon_available()` bounded probe + `sandbox_kwargs_from_settings` (reads `code_executor_*` knobs incl. `subprocess` group) + shared `execute_code`/`upload_files` async helpers + the owner-aware reaper. Capability seam: `service_provider._create_code_execution_manager` now constructs the CORE class directly — Docker-available installs get container isolation without the vertical; loader-based consumers (`cleanup-containers`, `container_eval`) unchanged. Vertical delegation: victor-coding `code_executor_tool.py` is a shim — `CodeSandbox`/`ResourceError` are identity aliases of the core types (not subclasses), historical names re-exported, `@tool` surface intact. Boundary resolution: `victor.tools.*` is not in `FORBIDDEN_RUNTIME_IMPORT_PREFIXES` and the vertical already imported `victor.tools.base/sandbox_ownership`, so `victor.tools.sandbox.container_sandbox` is the sanctioned import path (check_extracted_vertical_boundaries PASSED, errors=0). Tests RED-first: 45-test core suite (`tests/unit/tools/sandbox/test_container_sandbox.py` — lifecycle, network-default, caps-at-argv, 0-leak finalizer, gating, reaper), 14-test vertical delegation battery (identity + compat + tool contract; full vertical suite 1371 passed), bootstrap tests re-pinned to the core seam (unresponsive-daemon <1s, no-vertical construction, inert no-docker, loud construction failure), docker-gated integration egress-denied + caps tests appended to `test_code_sandbox_lifecycle.py` (skip without daemon). Gates: sandbox+security 1028 passed; architecture/contracts guards 210 passed; collect-only 41,889 clean; black/ruff clean; RELATED_TESTS entry added |
| FOU-S3a–d | Untrusted-content containment (envelope/canary → blocking → classifier → graduation) | S | ADR-037 | FOU-D2 | S3a+S3b GREEN (4af31f541); S3c GREEN (injection screen: annotation-only, in-envelope highlighting, corpus-pinned false-positive discipline); S3d MERGED (#1959) + S3d FOLLOW-UP MERGED (fix/injection-homoglyph-normalize): homoglyph pre-pass + quote-guard abuse carve-out closed both designed FN classes → battery re-run verdict = GRADUATE_HARD_FAIL_CANDIDATE (0 FP / 155 benign, 35 TP / 0 FN / 35 adversarial); **hard-fail WIRING SHIPPED flag-gated default-OFF (`ToolSettings.untrusted_content_hard_fail`, False): a tool call whose arguments echo content from a family flagged earlier in the session hard-fails through the same containment gate as canary blocking — flip = field data per ADR-036**; **S3b canary binding RESTORED (Codex fix wave 1, #1983): session-owned canary bound around every turn — the dispatch floor now fires on real sessions, not only tests** | #1908, #1911, #1983 |
| FOU-S4 | Graduation batteries + default flips: `effect_gated_completion`, `per_turn_auditor` | S | ADR-036 | FOU-D2 (stable runs) | PART-BLOCKED | `effect_gated_completion` graduation LOCKED OUT by the 2026-08-03 negative A/B (on 0.847 vs off 0.864, overlapping CIs — do NOT graduate, codified #884). `per_turn_auditor` deterministic battery SHIPPED (`victor/evaluation/auditor_ab.py` + 13-test suite): verdict **ESCALATE_LIVE_AB_REQUIRED** — wiring verified (flag resolves + edge-model judge injected on BOTH execution paths; a buffered-path wiring gap — `turn_execution_runtime` resolved the effect gate but never the auditor — found and fixed in the same change; OFF arm strict no-op pinned by negative test; placement audit guards the dormant-flag class) and cost known (max 1 edge-model call per COMPLETE turn, ≤1780-char bounded prompt, 128-token output cap, 0 main-loop token overhead; the deterministic heuristic alarm skips the judge; `max_alarms` bounds downgrades then annotate-and-allow). Judge QUALITY (premature-completion catch vs false-downgrade) is undecidable offline → default stays OFF; graduation requires the documented live A/B: `python -m victor.evaluation.flag_ab --flag per_turn_auditor --base-url <ollama> --model qwen3-coder-tools:30b --corpus effect-gate --variants 2` then `assess_graduation()` (EVR-5 oracle) + catch/false-downgrade rates. No default flipped. **Field-data harness SHIPPED (#2002)**: `victor/evaluation/field_data.py` + `victor evaluation field-data summary` — opt-in per-session recorder (`evaluation.field_data_enabled`, DEFAULT OFF, env `VICTOR_FIELD_DATA_ENABLED`) wired on the REAL streaming session path (`run_unified`; guarded import keeps `victor.evaluation` off the default chat cold path; placement + never-imports tripwire tests per the dormancy lesson). Records per-session: `per_turn_auditor` + `codebase_intent_indexing_enabled` exposure (production resolvers) + outcome markers (status/turns/duration/tool-calls, error CLASS only) — NO conversation content/code/identifiers (session id → truncated sha256, grep-pinned); `effect_gated_completion` deliberately not tracked (LOCKED OUT). Append-only JSONL under `~/.victor/evaluation/` (daily segments, 5 MiB rotation). **Collection starts with 0.32 sessions once operators enable the flag (documented default-off)**; aggregate via `victor evaluation field-data summary` |
| FOU-A1 | Subagent-spawn tool (Task-tool pattern) | A | ledger §A1 design | FOU-D2 | DONE — phase-1 MERGED (#1977, foreground); phase-2 MERGED (#2018, background mode + subagent_status/subagent_result companions + per-session bounded run registry + hallucinated-id refusal). Shipped in 0.34.0 | #1977, #2018 |
| FOU-A2a–d | Core long-term memory (store → write paths → PERCEIVE injection → surfaces) | A | FEP-0048 | FOU-D2 | A2a MERGED (`PreferenceEpisodicStore` + `MemoryType.PREFERENCE/EPISODE` + two-database schema (memory_preference/memory_episode) + adapter lanes; 33-test suite: round-trip, scope isolation (negative), secret/oversize refusal, forget-by-id/content, thread queries, coordinator interop, db placement). A2b MERGED (#1952: `memory_set`/`memory_forget` tools, workspace-write gated + `DEFAULT_TOOL_PERMISSIONS`; `MemoryConsolidator` propose/commit/discard hook in Session service, confirm-by-default with `memory_auto_capture` default off; secret-scanner word-boundary fix — generic keyword patterns now require a `[=:]` value-bearing delimiter, benign prose no longer refused, dual-path (native+Python) parity tests). A2c MERGED (#1954: `MemoryBlockBuilder` PERCEIVE injection via PerceptionIntegration `memory_injector` collaborator — scoped coordinator queries, 500-token hard cap by rank exclusion, off-when-empty byte-identical negative, flag + `VICTOR_MEMORY_ENABLED` env re-resolved at every read; BudgetManager additive context-block registration so memory competes by priority (evict weaker-only, refit, starve); `victor/evaluation/memory_ab.py` deterministic A/B battery (flag_ab.py documented as too LLM-heavy for the wiring gate) over seeded fixtures → snapshots + ADR-036 verdict `KEEP_DEFAULT_ON` (usefulness 1.0, max 20 tokens, 0 false surfaces, baseline 0 injections); 48-test suite; production container wiring deferred per the FEP's additive-optional hook parameter). A2d MERGED (#1956: `/memory` slash command — list preferences with scope/source/confidence + recent episodes + staged proposals, set with user-scope default/`--project`, forget by id/content, confirm/discard; API routes GET/POST/DELETE `/memory` + `/memory/proposals` list/confirm/discard with sibling management-route auth posture and structured 4xx refusals that never echo the matched secret; shared consolidator handle on `MemoryConsolidator` reached from the client layer via the new `victor.framework.memory` seam, bootstrap SessionService now binds it (confirm-by-default, auto-capture only from `settings.memory.memory_auto_capture`) so end_session stages onto the handle the surfaces confirm; `PreferenceEpisodicStore.recent_episodes` public listing read; 45-test suite; docs in quickstart + http-api). **Codex review wave 1 (2026-11-20): enforced single-trusted-operator posture over /memory (typed 409 on a second credential) + store-level cross-principal guard (`principal_id`/`on_behalf_of` → `CrossPrincipalAccessRefused`) + `EmptyContentMatchRefused` for empty/whitespace `content_match` forgets (pinned from store, API and tool); full ownership remains A4a+ paused pending D4 — see §Codex review fix wave 1). **Codex review wave 2 (2026-11-20): findings 6/19 FIXED — the PERCEIVE block now REACHES provider messages (enveloped `wrap_untrusted` MEMORY source class, exactly once, both buffered `_execute_model_turn` and streamed `assembled_messages_for_call` seams, turn-scoped candidate, session-canary envelope); binding supplies the actual project path + the active context-block `BudgetManager` (lazy factory — registration observed, exhausted allowance → byte-identical prompt); adapter reads are off-loop (`asyncio.to_thread`) and lane-selected before reading — see §Codex review fix wave 1 — memory delivery + off-loop reads** | #1951; A2b MERGED #1952; A2c MERGED #1954; A2d MERGED #1956 |
| FOU-A3 | SCIP precise tier auto-on + impact-analysis benchmark publication | A | ADR-032 | — | FUNCTIONAL + OPT-IN (ADR-032 rev 1.8/1.9) | Tier-3 SCIP harvesting works end-to-end and stays opt-in by the ADR-032 decision (`--precise`/`--no-precise`/`--precise-timeout` on `victor index graph index`; polyglot providers shipped — scip-python + scip-typescript + rust-analyzer, rev 1.6 corpus: 146,211 compiler refs @ 76.9%/67.4%/89.2% join rates; CLI drivability fixed rev 1.8 — settings resolve through `Settings.from_sources()` so config files can enable tier 3, per-provider harvest outcomes visible in index summary/`graph stats`/watch manifest; rev 1.9 hardening). REMAINING = two owner decisions, not code: auto-on flip of `enable_precise_index` and impact-analysis benchmark publication | — |
| FOU-A4 | RBAC/SSO wired into FastAPI middleware; per-user attribution | A | FEP-0020 (unblocks) + [A4 design doc](fou-a4-rbac-sso-design.md) | — | A4a+A4b MERGED (#1993, commit f0d85b2f6 on develop); D4 REVISED 2026-11-21 (owner-directed: Sentinelpass-primary credentials + TLS 1.2/1.3 + passkey — design doc §5, decisions D4-revised/D11/D12); A4c-1/-2/-3 DESIGNED (default-off phases, docs-only evolution PR); A4c OIDC pending D1/D2/D6; A4d pending D3 | Design doc: inventory (live auth seam `_include_with_auth`/`_make_auth_dependency` at `fastapi_server.py:582-686`; VAS-05j/k owner scoping; FEP-0020 attribution join subject-only today, `group_id` never bound; VAS-05a contracts design-only — no `VerifiedPrincipal`/OIDC/policy code exists) + gap analysis + two dormancy findings (`api/middleware.py` aiohttp stack zero callers; `ServerSettings` session-token knobs never wired). Phases: A4a principal resolution at the auth seam (behavior-neutral), A4b verified subject+group into FEP-0020 events, A4c Kanidm OIDC entry (`server.auth.mode` default `api_key`), A4d strict policy/role enforcement (`server.auth.policy_path` default off). Decision register D1–D10 with recommended defaults; D1/D2/D6 block A4c only; A4d needs D3/D4. **A4a implemented**: `victor/integrations/api/auth/principal.py` — `VerifiedPrincipal` (principal_id/kind/auth_method/credential_id=sha256 key digest/issuer/subject/groups/expires_at/scopes, `client_id` compatibility property) + `legacy_api_key_principal` synthesis (principal_id = key label, zero migration); resolved inside `_make_auth_dependency` (yield-dependency) / `_make_ws_auth_dependency` / the `/ws` auth frame → `state.verified_principal` + `state.client_id` shim; keyless/exempt byte-identical negatives. **A4b implemented**: centralized `bind_principal_attribution` (best-effort, FEP-0020) inside the HTTP dependency + `/ws` turn binding upgraded to principal subject+group; per-route `_verify_api_key` re-verification and binds removed from chat/classify routes (the #1981 posture claim now consumes the dependency-set shim; posture semantics unchanged); group producer = principal groups (legacy keys bind None); dormancy-guarded by orchestrator/provider-observed identity tests incl. the streaming-generator span. **D4 registry implemented**: `auth/principals_registry.py` strict YAML parser (revision; per-entry key/principal_id/kind/subject_label/groups/expiry; uniqueness + group dedup + unknown-field rejection), default `/etc/victor/api_principals.yaml`, override `server.auth_principals_path`, startup invariant refuses (typed `PrincipalsRegistryPathError`) any path inside the REAL sandbox writable-surface list (`writable_surface_roots`); expired entries fail closed 401; activation = restart or explicit `reload_principals_registry()` (operators publish via atomic rename). Tests: 27-test registry suite + 20-test principal/attribution suite (RED evidence retained vs origin/develop extract); gates: api/security/core/architecture/config suites green delta vs develop (11 config env-noise failures verified pre-existing); collect-only clean; black/ruff/mypy clean on changed files; RELATED_TESTS entries added. + **2026-11-21 design evolution (docs-only PR)**: design doc §5 added — A4c-1 Sentinelpass vault as PRIMARY credential store (exact-entry service grants, digest index, typed degradation matrix: infra-unavailable degrades to YAML snapshot, grant/token deny fails closed), A4c-2 TLS 1.2/1.3 with vault-held certs (uvicorn/OpenSSL SSLContext, pinned AEAD TLS1.2 suites, HSTS, restart-bounded rotation; rustls rejected), A4c-3 passkey WebAuthn RP (python-fido2 selected over py_webauthn on maintenance evidence; no resident-key discovery v1; server-side credential-id registration; Sentinelpass passkey custody = upstream ROADMAP item, seam reserved via PasskeyCredentialStore); all claims verified against ~/code/sentinelpass@52079eb; startup-invariant-now-moot security upgrade recorded; D4 revised, D11/D12 added | — |
| FOU-A5 | Research stack: search API adapters + page-read + browser tool; example cleanup | A | — | — | A5 CLOSED — phase-1 MERGED (#1979 — search adapters + page-read); phase-2 MERGED (#2016 — browser tool reworked to the agentbrowser substrate per owner direction, see §FOU-A5 phase-2 evidence below); example cleanup MERGED (#2017 — web_search_demo rewritten to the shipped `provider=` / page-read / browser_tool API, superseded 825-line mcp_playwright_demo deleted in favor of browser_tool + the existing agentbrowser MCP demo, toolkit-demo research list pinned to real tool names, research-vertical doc reconciled to the shipped providers) | #1979, #2016, #2017 |
| FOU-A5 evidence | Phase 1 extends the EXISTING web surface instead of duplicating it: (1) `victor/tools/search_providers/` — Tavily/Brave/Serper thin httpx adapters behind one `SearchProvider` protocol (`search(query, max_results) → SearchResult(title,url,snippet)`), key-gated via env var → `victor keys --set-service` service chain (tavily/brave/serper added to `api_keys_registry.yaml` + `SERVICE_ENV_VARS` fallback); absent key = provider not offered. (2) `web_search(provider=...)`: `auto` = first configured API provider else keyless DuckDuckGo (pre-existing DDG scraping reused — NO new scraping); explicit provider + missing key → structured `provider_configured: False` result (negative test proves no unauthenticated request fires); API errors → structured error, never raise. (3) Page-read = `web_fetch` upgraded: bounded `_fetch_page` httpx streaming download cap (2 MB, oversized → structured error), content-type guard (PDF/binary rejected), title extraction (<title>/og:title) in the payload. Browser tool + example cleanup = phase-2 follow-ups. Tests RED-first, respx-only (no real network): 21 adapter tests (`test_search_providers.py`), 13 dispatch tests (`test_web_search_provider_dispatch.py` incl. registration/discovery), 12 page-read tests (`test_web_fetch_page_read.py` incl. cap/timeout/content-type); existing `test_web_search_tool_unit.py` re-pinned to the `_fetch_page` seam; full `tests/unit/tools/` 3866 passed. Docs: tool-reference, user-guide/tools, api-keys.md. | #1979 |
| FOU-A5 phase-2 evidence | SUBSTRATE DECISION (owner direction, mid-PR rework): the browser tool is a CLIENT of the owner's agentbrowser product (~/code/agentbrowser @ origin/develop, HTTP API 127.0.0.1:5709 per ADR-017, `AGENTBROWSER_BASE_URL`/`AGENTBROWSER_API_KEY` overrides, Cedar capability authorizer gates `session.headed` — Victor always creates sessions with `headless: true`); Playwright is DEMOTED to a documented local fallback (`engine="playwright"`, optional `browser-playwright` extra, `victor/tools/browser_playwright_fallback.py`) and is NOT the primary. Policy division: Victor enforces the `tools.web_fetch_allowed_hosts` destination allowlist on the INITIAL URL before any call (and re-checks the final URL — belt and braces); agentbrowser's Cedar admission stack enforces navigation admission (its `blocked` status + reason + scopeDenials surface as structured `navigation_blocked`). Auth detail verified against the LIVE service: with a bearer key the tenant is server-derived (SHA-256(key) → tenantId), so Victor omits the body tenantId — a disagreeing body tenant is refused FORBIDDEN; unauthenticated local deployments declare `tenantId: victor`. Per read: session → page → navigate → extract(text+links) → title → session closed in finally (failed close never masks the result; explicit `session_id` reuses a caller-owned session and is never closed). Service unreachable → structured `agentbrowser_unreachable` naming the startup command (`node packages/api/dist/bin.js`, HOST/PORT, AGENTBROWSER_POLICY_FILE) and the playwright escape hatch. Contract tests mock the service with respx against the openapi.ts shapes AND validate the client's assumptions against the real ~/code/agentbrowser/openapi.json when the sibling checkout exists (skip on CI). Fallback engine retains the full guarantee battery below (manual per-hop redirect guard, 2 MB download cap, hard process-group kill probed against the live process table). Browser tool = NEW core read surface (no vertical duplication: the victor-coding browser capability is the external `agentbrowser` CLI/MCP skill; `web_fetch render=browser` returns plaintext only). `victor/tools/browser_tool.py` renders JS-required pages in headless Chromium (playwright, the existing optional `web` extra — no new dependency group) and returns structured title/url/content/links. Policy = the #1990 class extended to the browser's whole request stream: allowlist checked on the initial URL BEFORE any browser launches; redirects followed MANUALLY inside a context-scoped route guard (`route.fetch(max_redirects=0)` + Location resolution, the `_fetch_page` structure) because a fulfilled 3xx is followed by the network stack WITHOUT firing another route event (verified empirically — per-request route events alone would let a redirect chain escape the allowlist); every hop and every subresource request policy-checked; denied documents settled with a synthetic page so the denial maps deterministically (aborting raced ERR_ABORTED against the denial bookkeeping); popup pages cannot bypass (context-level routing, negative test pinned); final URL re-checked post-navigation; 2 MB download cap counted per admitted body (`file://` bypasses interception — documented residual); per-navigation timeout → structured `timeout: true`. Cleanup = container-sandbox placement discipline: browser launched detached with a unique per-session cmdline/env token; teardown closes page/context/browser/driver under per-step timeouts then SIGKILLs the session's process group via /proc scan — cancellation-safe (cancelled aclose reaps inline; detached task reaps again idempotently) and precise (only the session's token matches; concurrent sessions/user chrome untouched); live probe test hangs the graceful close, cancels, and polls the real process table to zero. Structured errors only: playwright_missing / browser_binary_missing / destination_policy_denied / redirect_budget_exceeded / download_cap_exceeded / timeout. Registration = demand-load spec + hydration keywords (optional dep → no bootstrap schema space, `gh` precedent); `web browser <url>` umbrella subcommand; READ_ONLY in DEFAULT_TOOL_PERMISSIONS; WEB_TOOL_NAMES airgap entry pre-existing; RELATED_TESTS entry added. Tests RED-first (25, `test_browser_tool.py`): pre-launch refusal incl. file://-under-allowlist, per-hop guard + subdomain semantics, document-vs-subresource denial settle, optional-dep structured errors (monkeypatched import + fake launch), timeout mapping + cleanup, download cap (guard unit + real loopback server), happy-path file:// extraction, REAL redirect chain positive + denied-hop negative vs a local HTTP server, hard-kill probes (cancelled teardown process-table, failed-close reap, idempotency), registration/discovery/demand keywords. Gates: black 26.10.1 + ruff clean; tests/unit/tools 3,999 passed / 113 skip; collect-only 42,443 clean; related suites green (web_search unit/dispatch/page-read/destination-policy, web_tool, permissions, select_changed_tests). | #2016 |
| FOU-C1 | Single circuit breaker (merge `circuit_breaker.py` + `resilience.py` breaker) | C | — | — | IN REVIEW | canonical `CircuitBreaker`+`CircuitBreakerRegistry` moved to `victor.core.circuit_breaker` (importable by all layers); `providers/circuit_breaker.py` = re-export shim; `MultiCircuitBreaker` (agent) rewritten as delegation facade over the core breaker (+ `try_acquire` slot reservation); agent `CircuitOpenError` subclassed onto canonical `CircuitBreakerError`; distinct-semantics `ModelCircuitBreaker` (per-model, decrement-on-success) and stateless `RecoveryService.get_circuit_state` left as classified; shim identity + delegation + half-open-slot tests; layering ratchet green; 178 breaker tests + 204 architecture + 550 agent-selective pass | #1947 |
| FOU-C2 | Services-layer module map + Stage C continuation (items 24/27/29/31/32) | C | — | — | IN REVIEW — item 24 LANDED (shipped in 0.34.0, #2000); 27/29/31/32 remain | Item-24 (FEP-0032 partial repair) landed ground-truth-first: RED evidence — with executor-style `checkpoint_after_nodes` (`AgenticLoopGraphExecutor` compiles with `{"act"}`), a *completed* run's last durable checkpoint sat mid-graph, so re-invoking the thread re-executed the completed tail (RED 4 across the durable seam). Fix: the runtime loop records an END-boundary completion checkpoint when the last completed node was not already durably written (`GraphCheckpointManager.save_checkpoint` now returns whether it wrote; full checkpointing gains no extra row — pinned). Acceptance added at the real seam: interrupt-after resume end-to-end on `ProjectDatabaseGraphCheckpointer` + `GraphSQLiteCheckpointer` (previously Memory-only), `compiled.resume()` of a completed sparse thread executes nothing, crash-resume-then-reinvoke stops at the boundary, Send fan-out completion pins a real node + the documented fan-out fallback. Framework (6769+182) + storage (610) + agentic_graph (236) green; collect-only clean. Remaining Stage C: item 27 FEP-0031 (six binding kwargs, raw state, factories/shims); item 32 client routing (REPL streams through an agent, `/completions` calls the provider directly); items 31→29 benchmark/RL consolidation (FEP-0033); streaming-driver paused surfacing rides the FEP-0031 stream-runtime inversion (FOU-D4 deferral). | #2000 |
| FOU-C3 | Native-parity CI broadened beyond tokenizer/context-fitter | C | — | — | IN REVIEW | parity/benchmark inventory of all 130 exported victor_native ops written into native-acceleration-strategy.md (backlog item 3 parity half); 5 new `native_parity` suites in tests/unit/native — embeddings (NumPy-f32 spec reference: quantize bit-parity, integer cosine, LCG projection reproduced bit-exactly, pairwise/KNN), secrets (full agreement + benign corpora through both dispatch paths + pinned registry-divergence ledger), graph_algo (1e-6 pagerank tolerance per the ahash finding, exact structural parity), trace_scanner (plain/gzip/malformed/filtering corpora), dedup/hashing/classifier-keywords/json_repair (structure-level; xxHash3-vs-MD5 digests pinned divergent by design); CI needs NO change (the native-parity job selects `-m native_parity` across tests/unit/native); findings pinned, then resolved: Python repair_json string-content corruption (FIXED #1960), coerce_string_type install-shape split (FIXED #1962), scan_usage_files same-session aggregation vs concatenation (FIXED #1975 — Python fallback adopts native cross-file aggregation), secret-registry gaps both sides (FIXED #1975 — union coverage; Rust gained bearer/PGP/bare-AWS, Python gained the vendor families); xxHash3-vs-MD5 digests remain divergent BY DESIGN (structure-level contract); remaining uncovered differentials (EmbeddingIndex, batch_score_messages, streaming, thinking, pattern_match, extractor, sanitizer, yaml_loader, chunking/arg_normalizer batch ops) recorded in the inventory for a future increment | #1957 |
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

## FOU-A1 phase-1 evidence (2026-11-19, fou/a1-subagent-spawn foreground)

- **RED**: `tests/unit/tools/test_spawn_subagent.py` failed collection on develop
  (module absent).
- **Substrate choice**: the team coordinator's member-execution path
  (`victor.agent.subagents.SubAgentOrchestrator`) over raw `AgentFactory.create()` —
  it already provides constrained per-member tool registries, per-role tool budgets
  and context sizing, member session-id derivation, timeout wrapping, and the
  `SubAgentResult` hand-off shape the foreground result needs. No new layering
  edge: the tool stays runtime-import-free; the factory binds a runner callable
  into the session's tool-executor context (`build_spawn_subagent_runner` in
  `victor/agent/factory/tool_builders.py` + the `ComponentAssembler.assemble_tools`
  call site — the wiring IS the deliverable, per the FEP-0041/0048 dormancy lesson).
- **Depth-1 guard, mechanical**: `max_depth != 1` refused (not clamped);
  `sanitize_subagent_tool_list` strips `spawn_subagent` from any requested child
  tool set at the tool boundary AND again in the runner; the runner hard-codes
  `can_spawn_subagents=False`; no role's default tool set authorizes the spawn
  tool (asserted for every role).
- **Budget**: verified `task_tool_config.yaml` carries only exploration-iteration
  budgets and stage tool lists — no per-tool budget mechanism — so the live
  `ToolSettings.spawn_subagent_tool_budget` path is used (None = substrate role
  default); the spawn call itself counts against the parent's live tool-call
  budget. Budget knob flows to the executor context (wiring-tested with a
  non-default value).
- **Permissions**: `spawn_subagent` registered in `DEFAULT_TOOL_PERMISSIONS` at
  the write tier (`WORKSPACE_WRITE`); read-only parents deny, workspace-write and
  full-access allow (both directions tested via `PermissionPolicy.authorize`).
- **Wiring (placement-not-existence)**: `tests/unit/agent/test_spawn_subagent_wiring.py`
  pins the REAL composition root: production-path `AgentOrchestrator` construction
  binds the runner + budget into `tool_executor.context`; the turn path
  (`ToolService` → `registrar.get_all_tools()`) hydrates and registers the tool;
  `subagent_orchestration_enabled=false` degrades to a runner that returns None
  through the same real path (tool refusal, session start unaffected).
- **Latent develop bug found and fixed**: the lazy `subagent_orchestrator`
  property called `SubAgentOrchestrator(parent=self)` but the constructor
  parameter is `parent_orchestrator` — every lazy init raised TypeError and
  silently disabled subagent orchestration on develop (caught by the dormancy
  test; fixed + positive-tested: the property now yields a real
  `SubAgentOrchestrator`).
- **Gates**: black 26.10.1 + ruff clean; `tests/unit/tools/ tests/unit/agent/`
  green; `pytest tests/ --collect-only -q` clean; layering ratchet 95/95 pairs
  (0 new — injection, no new upward import); RELATED_TESTS entries added for
  the tool, builders, assembler, orchestrator_properties, permissions and
  tool_settings sources.
- **Follow-ups (explicitly deferred)**: background mode + `subagent_status` /
  `subagent_result` companions; depth>1 budget config in the §A1 sketch's
  `task_tool_config.yaml` slot; orphan cleanup on parent cancellation (the
  substrate's `active_subagents` bookkeeping is the anchor); smoke criterion
  for the Tier A pre-push gate.

## FOU-A1 phase-2 evidence (2026-11-24, feat/fou-a1-phase2-background-subagents)

- **RED**: `tests/unit/tools/test_spawn_subagent_background.py` +
  `tests/unit/agent/test_spawn_subagent_background_wiring.py` failed
  collection on the phase-1 baseline (`victor.agent.subagents.run_registry`
  absent; `background` kwarg and companions unresolved).
- **Background spawn**: `spawn_subagent(..., background=False)` — when True
  the tool returns IMMEDIATELY with the structured handle
  `{success: true, background: true, subagent_id, child_session_id,
  status: "running"}` while the child runs detached through the SAME
  substrate path. The foreground path is byte-for-byte the phase-1 behavior
  (same tool, same enforcement, same intersection; pinned by
  `test_foreground_mode_does_not_touch_registry` and the unchanged phase-1
  suite). The runner (`build_spawn_subagent_runner`) gained a `background`
  kwarg: registration happens BEFORE detachment, then the identical
  `spawn_kwargs` run as an asyncio task wrapped in the run-policy snapshot
  captured at spawn time (BackgroundAgentManager parity).
- **Registry**: `victor/agent/subagents/run_registry.py` — per-session
  `SubAgentRunRegistry` (default cap 100, oldest evicted;
  `threading.RLock`; asyncio-task strong refs held via `attach_task` and
  dropped at terminal). Record = subagent_id / status
  (running|completed|failed|cancelled, one-way transition) /
  child_session_id / prompt / parent_session_id / started_at / finished_at /
  the SubAgentResult fields (success/content/tool_calls_used/
  duration_seconds/error) / owner_client_id. Double-terminal reports are
  ignored (first state wins). Created by
  `ToolBuildersMixin.build_subagent_run_registry` and bound into the tool
  executor context as `subagent_run_registry` — the SAME instance the
  runner detaches into (assembly-time identity, wiring-tested).
- **Companions**: `subagent_status(subagent_id)` (full run metadata, both
  states) and `subagent_result(subagent_id)` (structured "still running"
  until terminal, then the SubAgentResult payload), same module as the
  spawn tool. The registry is the ONLY source of run ids: hallucinated ids
  resolve to structured not-found (`found: false`), never another run's
  data — negative-tested through the production executor. Both registered
  in `DEFAULT_TOOL_PERMISSIONS` at the spawn tool's WORKSPACE_WRITE tier;
  read-only denies / workspace-write and full-access allow (both tools).
- **Terminal notification**: on a terminal transition the registry fires the
  BackgroundAgentManager-convention event callback
  (`subagent_completed`/`subagent_failed`/`subagent_cancelled`, owner-
  tagged, callback errors swallowed) AND a fire-and-forget
  `subagent.terminal` event on the observability event bus
  (`emit_event_sync`/running-loop task; notification failures never break
  run bookkeeping). Wiring test observes the callback through the real
  composition root.
- **Enforcement unchanged**: background children pass through the same
  `compute_child_tool_set` intersection, depth-1 strip + admission guard,
  minted child session ids, `can_spawn_subagents=False` and permission tier
  — `test_background_child_tool_set_intersects_and_excludes_spawn` pins
  spawn-tool absence from the child set on the background path; phase-1
  tests green unchanged.
- **Wiring (placement-not-existence)**: `test_spawn_subagent_background_wiring.py`
  drives `spawn_subagent(background=true)` through the REAL
  `orch.tool_executor.execute` after the production turn-path hydration
  (`registrar.get_all_tools()`), then reaches the same run via
  `subagent_status`/`subagent_result` through the production executor
  context; disabled orchestration refuses while the still-bound registry
  answers not-found; concurrency test interleaves three background
  children with one blocking foreground spawn.
- **Gates**: black 26.10.1 + ruff clean; `tests/unit/tools/ +
  tests/unit/agent/` green; `pytest tests/ --collect-only -q` clean;
  RELATED_TESTS entries added for spawn_subagent.py, the new
  run_registry.py, tool_builders.py, component_assembler.py and
  permissions.py.

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

## FOU-S3d evidence (2026-11-19, fou/s3d-fp-battery, #1959)

- **RED**: `tests/unit/security/test_injection_fp_battery.py` failed
  collection on the S3c branch (battery + corpus modules absent).
- **Battery**: `victor/evaluation/injection_fp_battery.py` +
  `injection_fp_corpus.py` — mirrors memory_ab conventions (two JSON
  snapshots `injection_fp_{benign,injection}.json`, verdict dict, CLI,
  deterministic offline). Corpus: **155 realistic benign samples across 8
  categories** (python/ts/rust source mentioning prompt/token vocabulary;
  MCP docs quoting `<tool_call>` syntax; logs/config dumps with
  password-policy prose; shell transcripts; review comments DISCUSSING
  injection; policy text) + **35 adversarial injection samples** (S3c base
  + split/padded/paraphrased/encoded variants).
- **Verdict: CONDITIONAL** — annotation-level FP rate **0/155** (each
  tightening keeps its benign counterpart pinned in
  test_injection_screen.py `S3D_TIGHTENING_BENIGN` + the corpus);
  adversarial **33/35** TP with 2 DESIGNED FNs (inj-034 quote-guard abuse,
  inj-035 homoglyph U+0406 — reported, NOT chased by pattern expansion).
  Per-family: impersonation 12 TP/2 FN, forgery 10/0, exfil 11/0.
- **9 tightenings applied to injection_screen.py** (probe: initial corpus
  run had 29 FPs → 0): quoted-mention guard (odd-quote-parity per line;
  documented trade-off = the quote-abuse FN), imperative-continuation
  lookaheads, model-directed-verb requirement on "New instructions:",
  constraint-removal cue on "You are now a X", imperative cue on line-start
  override headers, suspicious-body requirement on `<tool_call>{`,
  letter-start on `<function_results>`, dotted-method routing tokens,
  HIGH-RISK payload (pipe-to-shell/metadata IP/credential files) on
  "executed the command". Plus one exfil pattern ADDED
  (tell/give/show...+the/all forms — closed 3 adversarial FNs at no FP
  cost).
- **Hard-fail stays GATED**: battery REPORTS only (`hard_fail_wired: False`
  pinned by test); no production flag flipped. CONDITIONAL verdict means a
  follow-up may close the 2 FN classes (NFKC homoglyph pre-pass; quote-guard
  policy) before re-running for a GRADUATE_HARD_FAIL_CANDIDATE.
- Gates: black 26.10.1 + ruff clean; tests/unit/security/ +
  tests/unit/evaluation/ = 1592 passed, 7 skipped (known env-noise warning
  only); `pytest tests/ --collect-only -q` clean (41434 tests).

## FOU-S3d follow-up evidence (2026-11-19, fix/injection-homoglyph-normalize)

Closes the two DESIGNED FN classes from the S3d battery; **battery re-run
verdict: GRADUATE_HARD_FAIL_CANDIDATE** (0 FP / 155 benign, 35 TP / 0 FN /
35 adversarial; `hard_fail_wired: False` still pinned — the battery
REPORTS, it never flips).

- **Homoglyph pre-pass** (`normalize_for_matching` in
  `victor/security/injection_screen.py`): patterns run over a NORMALIZED
  COPY — per-character NFKC (folds fullwidth/mathematical-alphanumeric
  forms) plus a curated Cyrillic/Greek look-alike→Latin map (а е о р с х у
  і ѕ ј һ ԁ … + uppercase + Greek ο ι κ ρ τ υ ν α ε …). The screened
  content handed onward is BYTE-IDENTICAL (detection-only normalization).
  Span handling: normalization builds a per-character index map and
  `_map_span` maps matches back to ORIGINAL-text coordinates exactly —
  no whole-line approximation, no misaligned highlight offsets (positive
  test pins the span cut from the original text with the homoglyph
  intact). ASCII input short-circuits to the identity map.
- **Quote-guard abuse carve-out**: the odd-quote-parity guard (already
  PER-LINE, never cumulative) still skips discussion OF a payload; a
  quoted run whose DIRECTIVE continues after its closing quote (imperative
  verb — do/dump/delete/send/… — in the 160-char tail window past the
  close) now flags. `_IMPERATIVE_CONTINUATION` and the exfil direction
  lookahead accept a closing quote so the candidate match is found across
  the quote boundary; the guard + carve-out then decide discussion vs.
  directive. Quoted discussions (py-020/docs-004/review-001/review-003)
  stay guarded — pinned as new benign counterparts alongside real
  Cyrillic/Greek prose (normalization must not create FPs).
- **Positive tests added** (`test_injection_screen.py`
  `TestS3dFollowUpAdversarialClasses`): inj-034/inj-035 verbatim +
  variants (Cyrillic о/е/а, Greek omicron, fullwidth via NFKC; quote
  abuse on impersonation and exfil phrasing), content-immutability,
  span-mapping, and discussion-vs-abuse discrimination. Battery suite:
  `EXPECTED_FNS = set()`, verdict assertions flipped to
  GRADUATE_HARD_FAIL_CANDIDATE.
- **Hard-fail wiring is the now-unblocked next increment** (consumed via
  ADR-036): evidence complete, no production flag touched here.
- Gates: black 26.10.1 + ruff clean; tests/unit/security/ +
  tests/unit/evaluation/ = 1610 passed, 7 skipped;
  `pytest tests/ --collect-only -q` clean (41745 tests).

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
  intent leg query. Remaining: ADR-036 graduation battery (field data) — the
  collection harness landed (#2002, FOU-S4 row); enable
  `evaluation.field_data_enabled` to record sessions.
- **ADR-036 graduation battery — controlled phase RUN (2026-11-25)**:
  `tests/unit/evaluation/test_intent_tier_graduation_battery.py` green on the
  0.34 tree (A/B off-vs-on over the fixture corpus with known symbol
  positions; intent-over-keyword improvement pinned). Field-data collection
  verified end-to-end live: recorder fires per session (start+end records
  with flag-arm capture, duration, outcome) into `~/.victor/evaluation/`.
  Enablement note corrected in-repo: CLI chat sessions load env/.env only —
  the effective switch is `VICTOR_FIELD_DATA_ENABLED`; the settings.yaml
  evaluation group binds only via `from_sources` consumers. Graduation of
  `codebase_intent_indexing_enabled` now waits on accumulated field data
  (both arms) per ADR-036; `victor evaluation field-data summary` reads the
  evidence table.
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

## FOU-S3d hard-fail wiring evidence (2026-11-19, feat/release-chain-0.32)

The GRADUATE_HARD_FAIL_CANDIDATE battery verdict unblocked the MECHANISM;
it ships BEHAVIOR-GATED — `ToolSettings.untrusted_content_hard_fail` =
False (DEFAULT OFF; zero behavior change pinned by negative tests). The
flip is a field-data decision per ADR-036 and is NOT taken here.

- **Mechanism**: the S3c screen's tool-output verdicts record their
  families into session-scoped state
  (`victor/security/injection_screen.py`: `record_flagged_families` /
  `current_flagged_families` — same ContextVar session pattern as the
  canary binding). The recording point is `format_and_prune_tool_output`
  in `tool_service.py`, right where the screen highlights spans (recording
  is inert on its own). The enforcement point is the tool-call pipeline
  (`tool_pipeline.py`, immediately after the FOU-S3b canary floor): when
  the flag is ON and the session has flagged families, a tool call whose
  ARGUMENTS echo content matching one of THOSE flagged families
  (`find_flagged_content_in_value` — pattern match AND family
  intersection, so arguments matching an unflagged family pass) hard-fails
  through the same `_build_skip_result` containment gate: `outcome_kind =
  "flagged_content_echo"`, `block_source = "containment"`, `retryable =
  False`, model-visible refusal naming the envelope-as-data rule — the
  same conventions as the canary-blocking result. Contract per ADR-037
  §decision 2-3: canary hits (exact token) remain the deterministic floor;
  the screen adds the probabilistic escalation layer (model obeying the
  injected instruction) on top.
- **Tests** (`tests/unit/security/test_injection_hard_fail.py`, RED-first —
  collection ImportError before the mechanism existed): session-state
  accumulate/reset/default-empty; `find_flagged_content_in_value` top-level
  + nested + family-intersection-required + clean-args + scalar-ignoring;
  screen-records-families through `format_and_prune_tool_output` (+ clean
  output records nothing); pipeline-level ON-blocks-echo (positive), OFF
  default never blocks (negative), ON + no recorded families executes,
  ON + family mismatch executes, canary floor unaffected by hard-fail
  state in BOTH directions (ON and OFF), benign args execute with flag ON.
  Full adjacent suites green: tests/unit/security/ + tests/unit/evaluation/
  + tests/unit/tools/ = 5389 passed, 120 skipped, 1 xfailed, 3 xpassed.
- Gates: black 26.10.1 + ruff clean on changed files.

## 0.32 release train (2026-11-19)

Wave PRs merged to develop since 0.31.1, forming the 0.32 promotion
candidate (this ledger's rows above carry the detail):

- **MCP code-intel product**: #1924–#1929 (unified inventory + one-liners).
- **Provider/transport**: #1940 (HTTP transport), C1 breaker consolidation
  #1947 (canonical `victor.core.circuit_breaker`; IN REVIEW → merged on
  this branch's base).
- **Memory (FEP-0048)**: #1951 (A2a `PreferenceEpisodicStore`),
  #1952 (A2b `memory_set`/`memory_forget` + confirm-by-default
  consolidation hook), #1954 (A2c PERCEIVE injection + budget competition
  + A/B battery), #1955 (wiring), #1956 (A2d `/memory` + API routes,
  IN REVIEW at train-open).
- **Intent tier / durability**: #1943 (SQLite intent tier), #1945 (byte
  budgets).
- **Security (ADR-037)**: #1948 (S3c screen), #1957/#1959 (C3 parity
  inventory + S3d battery), #1960/#1963 (parity fix + homoglyph/quote-abuse
  follow-up) — S3d verdict **GRADUATE_HARD_FAIL_CANDIDATE** (0 FP / 155
  benign, 35/35 TP / 0 FN); S4 verdict **ESCALATE_LIVE_AB_REQUIRED**
  (`effect_gated_completion` stays locked out per the 2026-08-03 negative
  A/B; `per_turn_auditor` battery remains the executable remainder).
- **Attribution + native**: #1949–#1950 (attribution + `Python::detach`
  audit), #1964 (maturin floor 1.16).
- **Fix wave**: #1960–#1964.
- **This PR**: S3d hard-fail wiring shipped flag-gated default-OFF (above);
  FOU-S3a–d row updated accordingly; FOU-A3 row corrected to
  FUNCTIONAL + OPT-IN (ADR-032 rev 1.8/1.9) with the two owner-decision
  remainders (auto-on flip + impact-analysis benchmark publication).

## Codex review fix wave 1 — delegation + tool contracts (2026-11-20, #1982)

Fixes Codex-review findings 5, 10 and 14 (all HIGH) from the v0.32.0
candidate review (`~/code/victor-review-findings.md`).

- **Finding 5 — delegation grant widening.** The `spawn_subagent`
  restrict-list used to REPLACE the researcher role list, a tool missing
  from the parent could be instantiated from shared discovery, and depth
  was never checked on the final tool set. Now: `compute_child_tool_set`
  intersects requested ∩ effective role grants ∩ parent-admitted after
  alias resolution (parent-admitted read from the enabled+folded session
  registry, failing CLOSED when not enumerable — `tool_builders.py`);
  depth-1 enforced at child ADMISSION on the final set (`SubAgentOrchestrator.spawn`
  typed refusal, member never constructed; `stream_spawn` strips);
  `_configure_allowed_tools` consults the copied `can_spawn_subagents`
  boolean before registering anything. Depth-1 phase-1 enforcement from
  #1977 intact (A1 battery + wiring tests unchanged and green).
- **Finding 10 — code executor context contract.** The vertical sandbox
  tool expected an ordinary `context` argument the executor never
  supplies, so live dispatches returned SUCCESS with "Error: Context not
  provided." and a mock manager received zero calls. The tool now accepts
  the reserved `_exec_ctx` injection (schema advertises no context
  parameter) and raises `ToolExecutionError` (category `tool_execution`
  + recovery hint) for missing context/`code_manager`/operation
  arguments — failures are never success-shaped. Tests dispatch through
  the adapter's normal injection and assert the manager actually executed.
- **Finding 14 — spawn session ids.** Every foreground spawn passed the
  same member id (`spawn_subagent`), so the substrate derivation
  collapsed to one shared upstream session/cache key and the returned
  `child_session_id` read an optional input that was never supplied. Each
  spawn now mints a unique child session id (`mint_child_session_id`)
  threaded to provider binding (derivation precedence 1), the
  `SubAgentResult` and the identity/audit metadata; `_identity_metadata`
  reports the RESOLVED session id. Pinned by two overlapping spawns +
  one retry through the real wiring (three distinct ids, binding ==
  result).

Evidence: `tests/unit/tools/test_spawn_subagent.py`,
`tests/unit/agent/test_spawn_subagent_wiring.py`,
`tests/unit/agent/subagents/test_orchestrator.py`,
`verticals/victor-coding/tests/tools/test_code_executor_tool.py`.
Gates: tests/unit/tools + tests/unit/agent 13,542 passed; victor-coding
vertical suite 1,373 passed; collect-only 42,004 clean; black 26.10.1 +
ruff clean; RELATED_TESTS entries added for
`victor/agent/subagents/{orchestrator,base}.py`.

## Codex review fix wave 1 — containment enforcement at dispatch (2026-11-20, fix/containment-dispatch-enforcement)

Codex v0.32.0-candidate review findings 1/2/16 (HIGH/HIGH/MEDIUM), ADR-037
containment. All three fixed in one PR (#1983), each with positive + negative
tests:

- **Finding 1 — canary never bound to the serving session (placement-not-existence
  confirmed)**: `new_session_canary` was reachable only as the ephemeral fallback in
  `wrap_untrusted`; `set_session_canary` had zero production callers, so the S3b
  dispatch floor never fired on a real session. Fix: `SessionContainmentState`
  (`victor/security/containment.py`) owns ONE stable canary + bounded flagged-family
  evidence per session; `RuntimeIdentity.containment_state()` mints it lazily per
  resolved session id and regenerates on session transition (member bind / restored
  id), so child sessions stay distinct and old canaries retire. `ChatService.chat`
  and `stream_chat` bind the owner state around EVERY turn (and each resumed stream
  step / aclose) via `bind_containment_state`, restoring the prior binding on exit.
  Acceptance: real `ChatService` chat turn from a fresh request task — enveloped
  tool output publishes the session canary, an echo of that token in a follow-up
  call is blocked with ZERO executor calls; a second session holds a distinct
  canary; binding restored after the turn.
- **Finding 2 — admission checks fail-open (three sub-defects)**:
  (a) model-bound ERROR results bypassed the envelope — the display formatter or
  the FIRST `format_and_prune_tool_output` return value (the preview) entered
  history. Fix: the error branch in `process_tool_results_with_context` publishes
  through the SAME formatter; `content` (persisted tool message) is the enveloped
  `llm_output`, display fields keep the user-facing text.
  (b) the envelope wrapper swallowed failures (`except: pass`). Fix:
  fail-closed when enforcement is configured — an envelope failure publishes the
  inert `CONTAINMENT_FAILURE_PLACEHOLDER` (raw content never reaches model
  context); with `untrusted_content_envelopes` explicitly OFF, delivery stays
  raw as before. The injection screen remains advisory (separate block).
  (c) the canary walker checked mapping values only. Fix: `find_canary_in_value`
  covers mapping KEYS (reported via a redacted `<key>` path component so the
  model-visible refusal never echoes the token) and the `str()` form of
  non-JSON-able leaves; `find_flagged_content_in_value` screens keys too.
  Dispatch gates are fail-closed on containment-infrastructure failure: with
  `canary_blocking_enabled` on (default) — or settings unreadable while a canary
  is bound — a throwing checker returns a typed no-execution result
  (`outcome_kind=containment_failure`, zero execution); with the floor explicitly
  OFF the check is dormant (logs only). The S3d hard-fail gate gets the same
  treatment when its flag is ON; unreadable settings keep it dormant (unknown
  never enables blocking).
- **Finding 16 — hard-fail evidence task-local**: flagged families lived in a
  task-local ContextVar whose reset had no production caller. Fix: evidence is
  stored in the session-owned `SessionContainmentState` (bounded,
  `MAX_FLAGGED_FAMILIES=32`), written through by `record_flagged_families`,
  rebound per turn, restored on turn exit — a fresh request task recovers the
  session's evidence; a different session sees none; a long-lived task cannot
  carry stale flags across sessions. Family-overlap semantics documented
  honestly in `injection_screen.py`: overlap is a NECESSARY signal, not proof of
  exact echo or causation — which is why hard-fail stays flag-gated default-OFF
  per ADR-036.

Evidence: `tests/unit/security/test_containment_session_binding.py` (16),
canary-blocking walker + fail-closed gates (+11), tool-output containment
fail-closed envelope (+2), hard-fail fail-closed (+2), result-processing error
envelope (+2), chat-turn acceptance (3). Gates: security 884+ green; agent
services + tools suites green vs the pre-change baseline; black 26.10.1 +
ruff clean; collect-only clean.


## Codex review fix wave 1 — intent shim + seccomp fd (#1984, 2026-11-20)

Fixes findings 7 and 8 (HIGH) and finding 21 (MEDIUM) from the v0.32.0
candidate Codex review (`~/code/victor-review-findings.md`), branch
`fix/intent-shim-seccomp-fd` off `origin/develop` @ `04b3efa88`.

- **Finding 7 (HIGH) — intent tool shim**: `graph_query_tool._intent_provider_for`
  was an async factory against the seeding core's synchronous factory
  contract — with the intent flag on, explicit `mode="intent"` and the
  hybrid intent leg reported "unavailable leg: 'coroutine' object has no
  attribute 'search'" before touching the index. The shim is now sync.
  Acceptance: the REGISTERED `graph_semantic_search` tool driven end to end
  over one indexed SQLite fixture (real graph store + real `intent_chunks`
  table + production `build_intent_index`; deterministic encoder stand-in)
  in BOTH modes, known symbol contributes a score
  (`tests/unit/tools/test_graph_query_tool_intent.py`).
- **Finding 21 (MEDIUM) — intent switch in three settings groups**: ONE
  effective field chosen: `search.graph.codebase_intent_indexing_enabled`.
  All supported spellings (flat legacy key/env, `search.`/`codebase.`/
  `embedding.` mirrors, new-style group env) migrate onto it explicitly
  (`Settings._migrate_intent_indexing_spellings`); conflicting spellings
  are rejected at load. CLI help now advertises the effective field.
  Asserted through the graph CLI (every spelling reaches
  `GraphIndexConfig`) and the registered query tool
  (`tests/unit/config/test_intent_flag_migration.py`,
  `tests/unit/ui/commands/test_graph_index_cli.py`).
- **Finding 8 (HIGH) — seccomp fd forwarding**: the direct terminal, git
  and search launchers forwarded the wrapped argv WITHOUT the backend's
  seccomp filter descriptor (bwrap refused: "Can't read seccomp data") and
  leaked one open fd per launch. New shared seam
  `victor/tools/sandbox/launch.py`: `prepare_launch` returns argv AND
  owned descriptors together; `run_launch`/`run_launch_async` forward
  `pass_fds` and close them on every exit. Terminal/git/search routes and
  the canonical executor (sync + async sites) are all consumers. Tests:
  fd-count stability across repeated failed launches AND repeated
  admission refusals, route `pass_fds` regressions, live bwrap
  qualification (harmless subprocess green; unshare probe denied rc=7
  proving the filter is attached; argv-only negative control fails
  loudly), skip-with-reason on unqualified hosts
  (`tests/unit/tools/sandbox/test_launch_helper.py`).
- **Residual (wave-2 candidates)**: FIXED in wave 2 — see the next section.
- **Gates**: unit tools/graph_rag/security 6128 passed; api/storage/
  config/ui-command suites green; collect-only clean (42029); black
  26.10.1 + ruff clean on changed files. The seccomp/bwrap live tests run
  on qualified Linux only.
## Codex review fix wave 2 — seccomp residual at every wrap-then-spawn site (FIXED, 2026-11-20)

Closes the finding-8 residual left by #1984: all nine wrap-then-spawn
launchers now go through the shared `victor/tools/sandbox/launch.py` seam
(`prepare_launch` + `run_launch`/`run_launch_async`), so under the fail-closed
required posture each carries its seccomp filter descriptor and no parent
descriptor leaks. Branch `fix/w2-seccomp-residual` off `origin/develop` @
`fcb1f8f3a`. PR: #1988.

Converted sites (behavioral parity where the site was correct: `None`/none
backend → raw argv, zero descriptors; refusal paths unchanged, pre-wrap):

1. **bash tool** (`victor/tools/bash.py`) — wraps `["/bin/sh", "-c", cmd]`
   via `prepare_launch`; `run_managed_process` receives `argv` AND
   `pass_fds`; owned fds closed in a `finally` around the runner.
2. **workflow sandbox executor** (`victor/workflows/sandbox_executor.py`)
   — `_execute_process` wraps via `prepare_launch`;
   `SandboxedProcess.start` gained a `pass_fds` parameter forwarded to its
   `Popen`; fds closed in the existing `finally`.
3. **MCP stdio client** (`victor/integrations/mcp/client.py`) —
   `_start_process` wraps via `prepare_launch`; both owners (the
   rlmit `SandboxedProcess` and the plain `Popen`) receive `pass_fds`; fds
   closed in a `finally` covering the cancelled-startup path.
4. **filesystem write enhancer** (`victor/tools/lsp_write_enhancer.py`) —
   formatter subprocess now spawns through `run_launch_async` (was inline
   `asyncio.create_subprocess_exec`); error handling and temp-file cleanup
   preserved.
5. **merge manager** (`victor/processing/merge/manager.py`) — `_run_git`
   spawns via `run_launch` (raised `CalledProcessError` contract kept).
6. **temporal analyzer** (`victor/tools/verification/temporal_analyzer.py`)
   — `_run_git` spawns via `run_launch`; degrade-to-empty-string paths
   unchanged.
7. **auto-commit** (`victor/agent/auto_commit.py`) — `_run_git` spawns via
   `run_launch` with `check` forwarded (typed refusals unchanged).
8. **snapshot store** (`victor/agent/snapshot_store.py`) — `_run_git`
   spawns via `run_launch`.
9. **navigation slash command** (`victor/ui/slash/commands/navigation.py`)
   — `_run_git` spawns via `run_launch`.
10. **(sweep find, same defect class)** filesystem read-root discovery
    (`victor/tools/filesystem.py:_discover_read_roots`) — was wrapping
    `git worktree list` without descriptors; now through the seam.

**Structural guard**: `test_no_call_site_wraps_argv_outside_the_shared_seam`
(`tests/unit/tools/sandbox/test_launch_call_sites.py`) asserts no module
outside `victor/tools/sandbox/` contains a `.wrap_argv(`/`take_pass_fds(`
call — a wrap anywhere else is by construction a leak. The only sanctioned
consumer is `prepare_launch` itself.

**Tests** (`tests/unit/tools/sandbox/test_launch_call_sites.py`, mirrors of
`test_launch_helper.py`): per-site pass_fds forwarding under an enforcing
fake backend, raw-argv + zero-descriptor parity without one, descriptor
drain + close on every exit (including failing spawns), parent fd-table
stability across repeated failing launches, a live bwrap qualification
through the snapshot-store site (skip-with-reason on unqualified hosts).
Updated call-shape assertions in `test_mcp_factory_lifecycle.py`,
`test_failure_boundaries.py` (`start(..., pass_fds=())`) and the workflow
`_RecordingBackend` stub (now honors the `take_pass_fds` contract).

**RELATED_TESTS** entries added for all ten touched production modules.

**Gates**: `tests/unit/tools/ tests/unit/security/ tests/unit/integrations/`
green (5179+ passed; the 8 pre-existing `MCP factory lifecycle` call-shape
failures fixed in-PR); sandbox/workflows/agent/processing/commands suites
green; black 26.10.1 + ruff clean.

## Codex review fix wave 1 — owner isolation (2026-11-20)

Near-term fixes for Codex v0.32-candidate review findings 3, 4 and 13
(HIGH, owner isolation + destructive forget). Full per-principal ownership
is the PAUSED A4a+ design (FOU-A4 row; VAS-06 context) pending decision D4 —
this wave enforces the honest single-trusted-operator posture instead of
building ownership.

- **Finding 3 (shared server-wide conversation client)**: new
  `victor/integrations/api/operator_posture.py` + `VictorFastAPIServer.claim_shared_state`.
  SAFE default: with MORE THAN ONE configured credential (`api_keys` /
  `VICTOR_API_KEYS`) and no explicit declaration, the FIRST authenticated
  principal to use a shared surface (chat conversation addressing
  `/chat`, `/chat/stream`, `/chat/wire-stream`, `/chat/resume`;
  `/conversation/export`; `/conversation/reset`) becomes its sticky owner
  for the server session (reset on `update_session_config`), and requests
  under a DIFFERENT `client_id` label are refused with the typed
  `409 MultiPrincipalRefused` payload naming the limitation and the escape.
  Posture-active + unresolved principal fails closed. Unresolved-credential
  count is credentials, not labels; keys aliased to one `client_id` label
  count as one principal. Operator escape: `single_operator_mode=True` /
  `VICTOR_SINGLE_OPERATOR_MODE=1` attests one trusted operator (documented
  in `docs/user-guide/cli-reference.md` §victor serve). Single-credential
  and keyless servers behaviorally unchanged (pinned negatives).
- **Finding 4 (shared memory store/routes)**: same enforced posture over
  every `/memory` + `/memory/proposals` surface (typed 409 on a second
  principal; store untouched). PLUS a store-level guard:
  `PreferenceEpisodicStore(principal_id=...)` binds a principal context;
  `set_preference`/`get_preferences`/`forget`/`record_episode` accept
  `on_behalf_of=`, and cross-principal addressing is refused with
  `CrossPrincipalAccessRefused` BEFORE any transaction opens (rows
  preserved). Store unbound + call unnamed = legacy single-operator
  behavior unchanged.
- **Finding 13 (empty forget wipes a scope)**:
  `EmptyContentMatchRefused` raised in `PreferenceEpisodicStore.forget`
  for empty/whitespace-only `content_match` before any transaction; pinned
  from ALL THREE layers (store, DELETE `/memory` → 400 + rows preserved,
  `memory_forget` tool → structured error + rows preserved) plus positive
  controls (non-empty match still works; typed class named in the API
  ValueError response).
- **Known residual (out of this wave's contract)**: the `/ws` chat socket
  (message-level auth, Codex finding 1 scope) does not participate in the
  sticky claim; the completions endpoint is stateless and unaffected.
- **Follow-up commit (1429a8f9c)**: jscpd ratchet flagged the duplicated
  route-file `_posture_refusal` helper + stream-handler prelude — deduped
  onto shared `posture_json_response`/`request_principal` +
  `_admit_chat_turn` (ratchet back to baseline, delta +0); plus a
  real-server two-credential `/memory` test (auth dependency resolves
  `request.state.client_id` into the sticky claim; store rows preserved).
- Tests: 26-test refusal/isolation battery
  (`tests/unit/integrations/api/test_multi_principal_refusal.py` — RED
  first: two-credential interleaved chat/export/reset/stream/resume →
  second credential 409 + first user's data intact; single-credential and
  keyless negatives; escape-hatch positive), store cross-principal +
  empty-match pins in `test_preference_episodic.py`, API pins in
  `test_memory_routes.py` (incl. refusing-stand-in posture fixture —
  addresses the "one router/one store fixture" note), tool pins in
  `test_memory_write.py`. RELATED_TESTS entries added for
  `operator_posture.py`, `conversation_routes.py`, `memory_routes.py`,
  `chat_routes.py`. Gates: black 26.10.1 + ruff clean;
  tests/unit/integrations/api + tests/unit/storage/memory + tests/unit/tools
  4235 passed; collect-only 42,018 clean.

## Codex review fix wave 1 — memory delivery + off-loop reads (#1987, 2026-11-20)

Codex v0.32.0-candidate review findings 6 (HIGH) and 19 (MEDIUM): the
PERCEIVE memory block was rendered into perception metadata with no named
production consumer, the binding supplied neither the session project path
nor the active budget manager, and both memory lane adapters ran synchronous
SQLite scans + Python ranking ON the event loop.

- **Finding 6 FIXED (#1987) — provider-message delivery, provenance-tagged, exactly
  once**: `PerceptionIntegration.perceive` now records a turn-scoped
  `MemoryDelivery` (query, block) — overwritten on EVERY perceive, so a
  previous turn's block can never leak into a later turn. The named
  production consumer is `memory_injection.insert_memory_delivery_message`,
  called from BOTH provider-message assembly seams: `assembled_messages_for_call`
  (streaming ACT via `get_assembled_messages`) and `TurnExecutor._execute_model_turn`
  (buffered turns). The block enters as an ADR-037 canonical envelope
  (`wrap_untrusted`, new `UntrustedSourceClass.MEMORY`, origin
  `long_term_memory/<scopes>`), canary from the session containment state
  ChatService binds around every turn — provenance-tagged data, never raw
  system-prompt input. Exactly once: structural delivery marker skips
  already-delivered assemblies, insertion is transient (conversation history
  never mutated, no accumulation across turns/intra-turn calls), positioned
  after the leading system message(s) before history. The history-serialization
  projection in `chat_stream_executor._get_conversation_history` opts OUT
  (`include_memory_block=False`) — analysis context stays memory-free.
  Binding fix: `bind_orchestrator_memory_injector` now supplies the ACTUAL
  session project path (explicit session workspace or composition cwd) and
  the ACTIVE context-block `BudgetManager` (container `IBudgetManager`,
  resolved lazily via `budget_manager_factory` because the service layer
  initializes after the composition-root binding).
- **Finding 19 FIXED (#1987) — off-loop, lane-selected reads**: `PreferenceEpisodicAdapter.search`
  (both lanes) pushes its lane into the query (`filters["kind"]`) so the
  store selects the lane BEFORE reading (the PREFERENCE lane never opens the
  episode table, and vice versa — each scope's scan is bounded to its lane),
  and offloads the synchronous SQLite scan + Python ranking through
  `asyncio.to_thread`; `PreferenceEpisodicStore.get` factors its sync lookups
  into `_get_by_id_sync` behind the same hop. The event loop never blocks on
  a memory read.
- Tests RED-first: `tests/unit/agent/test_memory_delivery_provider_messages.py`
  (11) — buffered AND streamed turns capture the final provider messages and
  pin the enveloped block EXACTLY ONCE (position after system before
  history, envelope + origin + session-bound canary, transient insertion);
  negatives (memory disabled, empty store, refused wrong-owner write, wrong
  workspace, exhausted context-block allowance, stale-turn block) each yield
  NO envelope and a byte-identical assembly (pre-/post-perceive snapshot);
  budget registration observed on the resolved active manager; stream
  history-serialization projection excludes the envelope.
  `tests/unit/storage/memory/test_preference_adapters_offloop.py` (5) —
  scan thread ≠ event-loop thread, differential parity of the hop, lane
  selected before reading (monkey-pinned table exclusivity), off-loop point
  lookups. Dormancy test `test_memory_injection_production_wiring.py`
  re-verified green in this environment (22.8s, no timeout hardening needed).
  RELATED_TESTS entries added/extended for `memory_injection.py`,
  `context_assembly.py`, `orchestrator_context.py`, `perception_integration`
  group, `turn_execution_runtime` group, `chat_stream_executor.py`,
  `containment.py`, `adapters.py`, `preference_episodic.py`.
  Gates: black 26.10.1 + ruff clean; tests/unit/agent (5,924 passed),
  tests/unit/storage + tests/unit/security and tests/unit/evaluation green;
  collect-only clean.


## Codex review fix wave 2 — consolidation wiring, dormancy cleanup, EventBridge owner binding (#1986, 2026-11-20)

Fixes findings 15 and 22 (MEDIUM) and finding 9 (HIGH) from the v0.32.0
candidate Codex review (`~/code/victor-review-findings.md`), branch
`fix/w2-api-dormancy-consolidation` off `origin/develop` @ `fcb1f8f3a8`.

- **Finding 9 (HIGH) — EventBridge owner binding**: the /ws/events socket
  authenticated a key but registered a RANDOM connection id with no
  principal; delivery checked only type + correlation. The WS auth
  dependency stamps the verified principal onto `websocket.state.client_id`
  (constant-time compare; /ws keeps its message-level auth); `/ws/events`
  binds it to each subscription record at registration
  (`ClientConnection.owner_client_id`, immutable — subscribe frames cannot
  rebind); `BridgeEvent.owner_client_id`/`resource_owner` (also honored
  from `data["owner_client_id"]`) marks protected events and `accepts()`
  RECHECKS at delivery time against the live record — fail closed for
  unbound clients, ended leases (removed connections) are never consulted.
  Replay is owner-scoped (`get_recent_events(owner_client_id=…)`,
  `/events/recent`). `EventBusAdapter` maps internal owners into protected
  bridge events. Two principals receiving simultaneous events on BOTH WS
  transports (real TestClient /ws/events sockets with two keys +
  cross-owner negative; /ws `_broadcast_ws` two-principal positive).
  #1980 bounded-queue drop-oldest + single-writer guarantees untouched.
  Evidence: `test_event_bridge_owner_binding.py` (10).
- **Finding 15 (MEDIUM) — consolidation wiring**: stable proposal IDs
  (uuid5 per session+summary) with session/owner membership; staging is
  per-SESSION on the shared consolidator (was one process-wide list) with
  a retention bound (oldest evicted); confirmation commits bucket-head-
  first so a mid-commit DB error leaves only the uncommitted remainder
  staged and a retry never duplicates the committed prefix (committed-ID
  set makes re-offers no-ops; refused proposals dropped, not retried
  forever); session-scoped confirm/discard on the Session-service seam
  (incl. after end_session and after a newer session started — negative
  guard); the ACTUAL conversation source is wired at session end
  (memory-manager context messages when no ledger was registered);
  `get_shared_memory_consolidator` no longer freezes auto-capture at
  first construction. Confirm / discard / retry / opt-out across TWO
  sessions on one shared consolidator.
  Evidence: `test_memory_consolidation_sessions.py` (14).
- **Finding 22 (MEDIUM) — dormancy cleanup**: external compatibility
  audit found ZERO consumers of the aiohttp-era middleware names in core,
  vscode-victor, web/, docs or tests — shims kept, every access emits
  `DeprecationWarning`. CORRECTION (post-CI): the session-token settings
  are NOT dead — the web UI serving path (`web/server/main.py`) consumes
  `server_session_secret` (HMAC token signing/verification),
  `server_session_ttl_seconds`, `server_max_sessions` and
  `server_max_message_bytes`; the initial audit scope (victor/ +
  vscode-victor) missed `web/`, and the first commit's autogeneration
  removal broke the web server's fail-fast (caught by
  test_vscode_extension_contract on CI). Disposition as merged:
  autogeneration RESTORED as a visible dev/test default — generation is
  recorded on `server_session_secret_auto_generated`, the web server
  warns loudly at startup when the secret is auto-generated (tokens will
  not verify across workers), and the posture reports the secret's
  provenance; a multi-worker production deployment must set it
  explicitly (per the D6 direction). `server_api_key` WIRED at its
  actual owner (merged into the FastAPI auth credential map; precedence
  pinned); `server_max_message_bytes` WIRED on the /ws + /ws/events
  receive paths (typed refusal; normal-frame negative guard);
  `rate_limit_rpm` declared-but-not-enforced with a disposition note;
  NEW effective-posture surface: `VictorFastAPIServer.effective_posture()`
  reports enforced vs declared-not-enforced for EVERY advertised control,
  logged at lifespan startup and attached to `/status`
  (Evidence: `test_server_effective_posture.py` (14)).
- RELATED_TESTS entries added for `event_bridge.py`, `middleware.py`,
  `websocket_routes.py`, `fastapi_server.py`, `server_config.py`,
  `system_routes.py`, `memory_consolidation.py`, `session_service.py`.
- Gates: black 26.10.1 + ruff clean; tests/unit/integrations/api +
  tests/unit/agent/services + tests/unit/config 2643 passed (only 9
  failures = pre-existing `ANTHROPIC_API_KEY` env noise, byte-identical
  on the untouched baseline tree); collect-only clean (42,133).

## Codex review fix wave 2 — containment/sandbox mediums (2026-11-20)

Fixes Codex v0.32-candidate review findings 17, 18 and 20 (MEDIUM,
containment/sandbox) in one PR (#1985, `fix/w2-containment-mediums` off
`origin/develop` @ `fcb1f8f3a`), each with positive + negative tests:

- **Finding 17 — homoglyph normalization offset map**
  (`victor/security/injection_screen.py`): the ASCII fast path allocates
  a pair of offsets per character (4,205,808 traced bytes peak for a
  32,768-char tool output) and the non-ASCII path records one pair per
  ORIGINAL character even when normalization emits several, so later
  annotations can point at wrong source text. Fix: ASCII short-circuits
  to `(text, ())` — identity, no map allocation, `_map_span` treats an
  empty map as identity; non-ASCII records ONE entry PER EMITTED
  normalized character so `len(index_map) == len(normalized)` and any
  match after an expansion (ligature → two chars) maps onto the exact
  source text. Measured and recorded in-test (tracemalloc +
  perf_counter): ASCII 32,768 peak 48 bytes (<100K bound); 512KB
  realistic ASCII full screen ~90 ms (<2 s bound); 32,768-char
  typography-heavy non-ASCII normalize peaked ~4.34 MB with an aligned
  per-emitted-char map (<8 MB bound, linear by construction).
- **Finding 18 — container daemon-error ownership**
  (`victor/tools/sandbox/container_sandbox.py`): any Docker reload
  error detached the existing finalizer and permitted replacement
  (live container unowned until TTL, duplicate-sandbox risk); upload to
  an unavailable container returned normally and the async helper
  formatted it as success; `cpu_shares` is a relative weight, not a
  ceiling. Fix: reload distinguishes CONFIRMED deletion (`NotFound` →
  detach + replace) from AMBIGUOUS daemon errors (retain ownership +
  raise, retry next use — #1563 zero-leak guarantees preserved);
  `put_files` validates local paths before any Docker contact and
  raises the typed `ContainerSandboxUnavailableError`
  (`binary_missing`/`daemon_unreachable`) when no container can be
  acquired; explicit hard CPU ceiling `cpu_quota` (CFS microseconds per
  100 ms period) reaches the container-run invocation only when
  configured, threaded through `sandbox_kwargs_from_settings` +
  `code_executor_cpu_quota` (subprocess/security settings groups +
  flat alias + configuration docs, default None = no quota promised).
- **Finding 20 — policy-eval fail-open** (`victor/tools/sandbox/backends.py`
  + `victor/tools/subprocess_executor.py`): an exception INSIDE
  effective-policy evaluation returned an inactive, non-refusing
  decision (the settings-LOAD failure was already handled correctly —
  this is the eval-time gap), and the canonical executor's remaining
  fallback could return the passthrough backend. Fix: tri-state policy
  resolution — `resolve_effective_policy` → `PolicyResolution`
  (known | unknown; the `required` default applies only on a successful
  read); `resolve_required_launch` REFUSES on unknown before launch
  (typed reason); `_resolve_default_sandbox` drops the final
  `except: NoneSandbox()` fallback — a backend-selection fault under a
  known non-required policy raises the typed
  `PolicyUnavailable("backend_selection_failed")`. Independently
  tested: settings object loads, policy FIELD faults, inert launcher
  (spawn trap) — separate from the existing settings-load and
  missing-backend tests (`TestPolicyEvalFaultRefusesBeforeLaunch`,
  `TestPolicyResolutionTriState`).

Tests: `TestNormalizationOffsetMap` (6, incl. the typography-before-a-
later-match span-exactness positive and the map/emitted-length
misalignment negative), container sandbox +6 (ownership retention with
the finalizer still targeting the container, typed upload failure via
`put_files` AND the `upload_files` helper, quota-present/absent run
kwargs, settings threading; `test_degraded_lifecycle_is_inert` updated
to the typed-failure contract), executor + posture tri-state +8. All
#1976 lifecycle/gating/reaper tests unchanged and green. RELATED_TESTS
entries added for `backends.py`, `subprocess_executor.py`,
`subprocess_config.py`, `security_settings.py` (+ the two config suites
onto the `settings.py` entry). Gates: security + sandbox + tools 1084
passed (16 skipped); no new env-noise vs origin/develop (9 known
`tests/unit/config` account-credentials failures verified pre-existing
on a pristine develop worktree); collect-only 42,116 clean; black
26.10.1 + ruff clean on changed files.

## Codex review fix wave 2 — container cancellation + research destination policy (#1990, 2026-11-20)

Fixes Codex v0.32-candidate review findings 11 and 12 (HIGH, the last
unassigned findings) in one PR (#1990, `fix/f11-f12-container-cancellation-research-policy`
off `origin/develop` @ `0950fc861`), each with positive + negative tests:

- **Finding 11 — container cancellation protocol + host-boundary transfers**
  (`victor/tools/sandbox/container_sandbox.py`): owned-handle protocol —
  `close()` stops admission (typed reason `closed` on `execute`/`put_files`/
  `get_file`/`start`/acquisition), force-removes the container (terminating
  the exec/transfer a worker is blocked on) and joins in-flight workers under
  a bounded timeout (`close_join_timeout_seconds`, env
  `VICTOR_SANDBOX_CLOSE_JOIN_TIMEOUT`); every blocking op is admitted through
  an op counter, and `_ensure_container` re-checks the closed flag under the
  acquisition lock, discarding a container that raced teardown — nothing is
  acquired after the caller gave up. Async helpers terminate the in-flight
  Docker work on caller cancellation (`terminate_in_flight`); async context
  entry/exit run lifecycle I/O off-loop and `__aexit__` completes the
  teardown through delivery of further cancellation. Session shutdown (both
  `lifecycle_manager` release paths) prefers `close()` for a real
  `ContainerSandbox`; other managers keep `stop()`. Uploads are authorized
  at the host boundary — paths must resolve inside the admitted
  `code_executor_workspace_root` (symlink escapes refused; unset = local
  single-operator posture) — and BOTH directions are byte-budgeted
  (`code_executor_transfer_byte_limit_bytes`, default 64 MiB): pre-flight
  stat + budget-checked tar build before any Docker contact; downloads check
  the daemon-reported size before reading and cap the chunk stream
  cumulatively. New `SandboxTransferRefused` typed refusal
  (`path_outside_workspace` / `transfer_limit_exceeded`); directory uploads
  keep recursive-tar behavior under the budget.
- **Finding 12 — research destination policy**
  (`victor/tools/web_search_tool.py`): `tools.web_fetch_allowed_hosts`
  (Settings → ToolConfig → `_get_web_config`; empty = allow-all default so
  local single-operator behavior is unchanged; non-empty = exact-or-subdomain
  host match, case-insensitive, non-http(s) refused). Enforced at the
  `web_fetch` entry point BEFORE the result cache, re-checked at EVERY
  redirect hop (the HTTP path follows redirects manually under a bounded hop
  budget instead of `follow_redirects=True`) and on the browser render path's
  final URL (`page.url`); violations return a structured error naming the
  destination (`destination_policy_denied` / `denied_host` / `denied_url`).

Tests: `TestCancellationProtocol` (fake docker module pattern, gated
workers — cancel during acquisition/execution/close, idempotent close,
async-context close, off-loop lifecycle proven by a loop-spin counter),
`TestUploadAuthorization` (workspace admits/outsides/symlink escape,
upload budget pre-Docker, download size refusal without reading, stream
cap, recursive directory upload under the budget, settings threading),
`test_web_fetch_destination_policy.py` (19 respx tests — default posture,
denied entry host with route-not-called, cache-bypass, browser
pre-Playwright refusal, denied redirect hop, allowed chain, relative
Location resolution, hop budget, settings wiring), lifecycle-manager
close-preference and bootstrap knob tests; vertical delegation battery
14/14 green. #1985 typed-failure and #1563 zero-leak guarantees
preserved; the #1989 `get_file` typed download refusal is carried into the rewritten `get_file` (typed refusal replaces the historical `b""`).
RELATED_TESTS entries added/extended (`web_search_tool.py`,
`container_sandbox.py`, new `lifecycle_manager.py` entry, `base.py`,
`subprocess_config.py`, `tool_settings.py` + selector-test expectation);
`code_executor_workspace_root` / `code_executor_transfer_byte_limit_bytes`
documented in configuration-options. Gates: sandbox + tools + affected
agent/config suites green (4,000+ passed); collect-only 42,166 clean;
black 26.10.1 + ruff clean on changed files.
**Residual follow-up (#1989, `fix/container-getfile-typed-failure` off
`origin/develop` @ `0950fc861`)**: the documented `get_file` residual of
finding 18 — the download side still returned the historical silent
`b""` when no container could be acquired, the same
silent-empty-as-success shape the upload fix closed. Fix: `get_file`
raises the typed `ContainerSandboxUnavailableError` with the matching
reason (`binary_missing`/`daemon_unreachable`), never `b""` as success;
the success path is unchanged. No production caller consumed the
`b""` (grep: zero `get_file` call sites on the sandbox in `victor/` +
`verticals/`); the caller contract is pinned by a structured-error test
(typed refusal → `{ok: false, reason, error}`, no `data` field, not a
crash). Tests RED-first mirroring the `put_files` typed-failure pair:
no-docker → `binary_missing`, daemon-unreachable → `daemon_unreachable`,
caller handling, success path re-pinned unchanged;
`test_degraded_lifecycle_is_inert` moved to the typed contract; the
`test_docker_availability.py` integration battery (via the vertical's
`CodeSandbox` alias) updated — its `put_files` sibling had been stale
since #1985 (asserted a silent early return). RELATED_TESTS
`container_sandbox.py` entry extended with the docker-availability
integration file. Gates: sandbox unit suite 180 passed / 9 skipped;
selector-mapped suites green; collect-only 42,119 clean; docker-
availability degraded battery 10 passed with the vertical on
PYTHONPATH; black 26.10.1 + ruff clean on changed files.

## Codex review fix wave 2 — attribution-group rule, finding 23 (#2001, 2026-11-21)

Fixes Codex v0.32.0-candidate review finding 23 (MEDIUM, attribution-group
rule; branch `fix/m23-attribution-group-rule` off `origin/develop` @
`1bd409eeb`): the FOU-A4a/A4b join forwarded the principal's group per
event, but `SandhiMeter._ensure_key` registered the gateway virtual key
ONCE PER SUBJECT capturing its FIRST group — a verified group change
never reached any later usage event, and the frozen binding was racy
under overlapping HTTP/SSE/WS calls.

**Chosen rule** (D5 + Codex acceptance): the stable metering identity is
the principal SUBJECT (legacy key label today, issuer-qualified principal
ID once A4c lands); the billing GROUP is recorded PER EVENT, realized as
the **versioned metering key** — `victor:{subject}` (no verified group)
or `victor:{subject}:{group}`. API evaluation: the gateway's Python
binding admits NO per-event group (`meter_tokens` is key-authoritative —
sandhi ADR-0004 D4 resolves subject/group from the virtual key; only
session/route are caller-supplied), and same-id re-registration
(`add_virtual_key` → store overwrite) would mutate a shared binding —
racy exactly in the overlap scenario the finding asks to verify. The
versioned key is the sanctioned alternative and is race-free: each
event's key encodes its own event-time group, so a group change meters
through a NEW key — subsequent events carry the new group, historical
events keep their original group, and the per-subject key registry stays
bounded (returning to a prior group reuses its key). Key-only principals
bind `group=None` → the unversioned key-only shape, pinned
behavior-neutral. Tracker-side wiring was already per-event (A4a #1993);
no tracker change. The legacy per-route `bind_attribution` cited at
`routes/chat_routes.py:155` no longer exists (removed by A4a/A4b) —
reconciliation recorded in the A4 design doc (§A4b + §4.1), which now
documents the rule.

Tests RED-first (stale-first-group failure evidence retained vs
develop): group change between calls → subsequent real gateway events
carry the new group; overlapping HTTP/SSE/WS attribution bindings on one
shared session tracker → all 9 recorded events carry their own channel's
group; return-to-prior-group key reuse; legacy key-only principal binds
no group; `VerifiedPrincipal` via `bind_principal_attribution` end to
end. One stale key-id pin updated (`test_end_to_end_neutral_event` →
`victor:{subject}:{group}`). RELATED_TESTS entries added
(`victor/observability/sandhi_meter.py` — new;
`victor/agent/session_cost_tracker.py` — new). Gates: agent + api +
observability suites 10,713 passed / 5 skipped; collect-only 42,345
clean; selector suite 117 passed; black 26.10.1 + ruff + mypy clean on
changed files.

## 0.34.0 release train (2026-11-25)

Joint promotion with Sandhi v0.26.1 (train order: sandhi tagged → all four
registries verified → private edge upgraded → Victor promoted). Victor main
`ce520edd6`, tag `v0.34.0`, PyPI verified (victor-ai + victor_native 8-wheel
matrix incl. macosx arm64; fresh-install smoke clean from a neutral cwd).

Program content shipped on this train:

- FOU-A1 complete: background subagent mode + companions + bounded run
  registry (#2018, with #1977 foreground). Row flipped DONE above.
- FOU-A5 closed: browser_tool on the agentbrowser substrate (#2016) +
  example-surface reconciliation (#2017).
- FOU-C2 item-24: END-boundary completion checkpoint — FEP-0032 completed-
  thread resume guarantee holds when the last node's write raced completion
  (#2000).
- Sandhi endpoint dialect configuration: `endpoint_profile` / `protocol`
  admitted on `ProviderConfig`, configured protocol wired through the direct
  typed transport (was silently dormant), bindings fail closed on unknown
  profiles (#2019).
- Sandhi consumer floor contract: `sandhi-gateway>=0.26.0` in pyproject with
  exact deployment locks, runtime errors and docs naming the floor, alignment
  guards enforcing the split (#2021). Verified live: a clean 0.34.0 install
  resolves sandhi-gateway 0.26.1 with no Victor edit.

Operational notes: the promotion battery needed one rerun — the
core-integration shard hung at 99% on a rich-Live TUI test (non-tty xdist
flake; passes locally and in 0.33.2); bounded per-test by #2025. Sandhi
v0.26.1's release run surfaced three infrastructure failure classes (GitHub
API 500; rerun output-binding loss for previously-skipped jobs; gzip-mtime
asset byte-differ) — the asset-idempotency class is fixed on sandhi develop
(content-tree comparison).
