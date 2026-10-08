---
plan_id: VAS-2026-10
status: active
updated: 2026-10-08
baseline_develop: a5fa47e444e6f973ad95149ae0677327df060dc7
next_task: VAS-11a
---

# Victor shared agent-service implementation plan and tracker

**Canonical task-status ledger.** Browser UI, VS Code and external API consumers
will converge on one Victor service and contract family. The user authorized this
plan after the [API audit](../development/victor-agent-service-audit.md).
[FEP-0039](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0039-unified-agent-service-api.md)
owns the proposed architecture and remains Draft. This file owns execution status;
the [formation ledger](multiagent-formation-coverage-handoff.md) owns historical
formation results and G-numbered gaps. Do not duplicate status in another roadmap.

This is a dated checkpoint. On every resume, reconcile its rows with GitHub PR
state and fetched source before choosing work. Completed historical work stays
completed unless new evidence invalidates it. A missing `/tmp` directory is not
permission to redo a merged task.

## Outcome and constraints

A user can start a permitted single-agent, team or workflow run from web or VS Code,
observe the same recorded progress, approve the exact action, disconnect/restart,
and recover the true outcome through the same API. Unknown writes are reconciled
before continuation; stop controls distinguish requested from completed cancellation.

- Victor owns reusable OSS building blocks; AnvaiOps owns commercial product
  integration, managed operations, billing/entitlements and packaging. Keep OSS
  auth/security, authoring components and standalone use independent of AnvaiOps.
- One canonical server composition root, service/dispatch owner and identifier
  derivation; structured contracts and explicit errors. Preserve unchanged legacy
  defaults through opt-in/versioned surfaces and adapters.
- Reuse typed web session storage, existing wire events, framework client/services,
  paused-run/action journal and team/member owners. Preserve existing auth repairs.
- Required Sandhi Python binding, optional separately deployed proxy; InferFlux
  owns inference/GPU placement. No service cutover or credential transfer implied.
- TDD in existing test owners; remove duplicate tests only with invariant/coverage
  evidence. Smoke tests use a real server/packaged client, not only mocks.
- Linked worktrees from `origin/develop`, conventional prefixes, no force-push.
  Squash feature PRs only after independent review and every applicable CI gate,
  including Vertical Py3.12. Main promotion/release is a separate gated milestone.

## Status semantics

| State | Meaning / required evidence |
| --- | --- |
| TODO | No implementation/evidence accepted yet |
| ACTIVE | Named owner, branch, baseline SHA and exact next action recorded |
| RED | Relevant regression demonstrably fails for the intended reason; expected failed evidence retained |
| LOCAL_PASS | Final local diff passes cited tests/smokes; independent review/CI/merge may remain |
| MERGED | Exact PR, merge SHA and applicable green checks verified; not release/runtime acceptance |
| ACCEPTED | All row-specific acceptance gates passed, including release/live gates when required |
| BLOCKED | Concrete external prerequisite, owner, evidence and unblock condition recorded |
| SUPERSEDED | Replacement ID/PR and preserved invariant identified; never use for an unexplained omission |

Mark ✅ only for MERGED/ACCEPTED implementation rows or completed evidence-only
rows. Preserve the distinction between source, release and live acceptance. A
failed/timeout/interrupted run never becomes passed through a later documentation
edit. A mock test is not live provider or C5 evidence.

## Current checkpoint

- Fetched develop: `a5fa47e444e6f973ad95149ae0677327df060dc7` (#1252 merged).
- Owner: current durable-recovery session. Worktree:
  `/private/tmp/victor-action-recovery`; branch `fix/durable-action-recovery-guards`.
  Next action: complete final validation and exact-commit review, publish the
  admission repair, then verify every applicable CI gate before squash merge.
- Root checkout remains on main; another session's `fix-inferflux-codesign`
  worktree is not owned by this plan. Do not modify or remove it.
- Local default Node upgraded to 24.21.0. This is the build/tooling runtime;
  VS Code supplies its own extension-host runtime.
- Original WS-A–WS-I: **9/9 landed**. C5, G61/G62/G70 and other recorded lifecycle
  gaps remain open; this plan adds no formation passes.
- Durable local recovery copy: `var/session-closeout-2026-10-07/vscode-api-audit/`
  in the root checkout; this increment uses
  `var/session-closeout-2026-10-08-stream-contract/`. These contain patches,
  reports, logs and hashes; the archives are ignored
  by Git and machine-local. Remote Git/PR records are the cross-machine authority.

## Work ledger

Planning/evidence: VAS-00 is accepted; VAS-01 is merged.
Delivery: VAS-02 and VAS-03a are merged; VAS-11a has local passes; other implementation rows remain TODO. These counts are not
an effort-weighted completion percentage. The original formation denominator is
independent. Update this paragraph and the rows together at each checkpoint.

| ID | Milestone / bounded scope | Depends on | State | Evidence / next action |
| --- | --- | --- | --- | --- |
| VAS-00 | Audit current API shapes and reproduce boundary defects | — | ✅ ACCEPTED | Audit: real core router returns 422; compiled TS stub probes show approval loss, EOF success and false cancellation; independent review corrected binding/ownership details |
| VAS-01 | Publish FEP, plan, roadmap/gap links and reboot handoff | VAS-00 | ✅ MERGED | [#1251](https://github.com/anvai-labs/victor/pull/1251), `470c9ccdcb222dcfd8acce5b4c52fb27321ac952`; all applicable checks green; FEP remains Draft |
| VAS-02 | Node 24, dependency remediation and real activation/package smoke | — | ✅ MERGED | [#1251](https://github.com/anvai-labs/victor/pull/1251), `470c9ccdcb222dcfd8acce5b4c52fb27321ac952`; 999 host + 50 unit tests; audits zero; packaged VSIX activation/106 commands passed |
| VAS-03 | Agree contract/FEP and consumer inventory; repair streaming request parity | VAS-01, VAS-02 | TODO | Accept FEP before new public API; preserve FEP-0037/0038 classify/model-effort contracts; extend existing contract owner, RED actual TS body vs real router, then explicit compatibility adapter |
| VAS-03a | Repair existing VS Code streaming request compatibility | VAS-01, VAS-02 | ✅ MERGED | [#1252](https://github.com/anvai-labs/victor/pull/1252), `a5fa47e444e6f973ad95149ae0677327df060dc7`; clean exact-head review, 41 applicable checks passed including Vertical Py3.12; parent/FEP acceptance remains open |
| VAS-04 | Typed client outcomes: paused run, terminal EOF, cancellation acknowledgement | VAS-03 | TODO | Preserve status/run/approval; reject incomplete streams; consume negative cancellation body; no new approval store |
| VAS-05 | Shared authentication and per-resource principal authorization | VAS-03 | TODO | HTTP/events parity, no placeholder tokens, OIDC hosted posture and explicit scoped machine/local modes; reject cross-owner access |
| VAS-05a | Strict verified-principal and resource/action policy contracts | VAS-03 | TODO | [Kanidm/API-key policy design](victor-agent-service-auth-policy.md); deny overrides, credential-scope ceiling, ownership and no cross-grant widening; design only |
| VAS-05b | Kanidm access-token and scoped key authentication | VAS-05a | TODO | Dedicated verified registration/discovery; no alternate-auth fallback; expiry/revocation/rotation, group provenance; same principal contract |
| VAS-05c | Enforce authorization through every resource/transport | VAS-05b, VAS-06 | TODO | Owner/workspace/tenant checks on HTTP/SSE/WS and delegation; existing tool/approval owners remain authoritative |
| VAS-05d | Login UX, Kanidm provisioning and deployment acceptance | VAS-05c | TODO | Browser/IDE login and service-key positive/negative smokes; verified DS3 setup, roles, recovery and documented revocation bounds |
| VAS-05e | Victor/Sandhi identity propagation and delegation contract | VAS-05a | TODO | Co-design separate initiator/actor/provider identity; verify Kanidm exchange capability, scoped delegation, audience and revocation; shared workload attribution is intermediate only |
| VAS-05f | Implement immutable per-run gateway credential binding and delegation | VAS-05c, VAS-05e, VAS-06 | TODO | Two-user concurrency, no shared credential mutation, grant/model denial, rotation/revocation and actual identity/accounting joins |
| VAS-06 | Canonical session ownership and bounded admission | VAS-05b | TODO | Port/reuse web store semantics; scope by principal/workspace, concurrent-turn policy, pre-initialization admission and restart contract |
| VAS-07 | Durable run admission, identity and status | VAS-06 | TODO | Reuse existing durable owners; same-key dedup/content conflict, reliable dispatch and honest recorded outcomes |
| VAS-08 | Canonical FastAPI composition and web compatibility entry point | VAS-03, VAS-06, VAS-07 | TODO | One route/service owner; port web wire/session behavior, preserve supported old requests and startup entry points |
| VAS-09 | Authoritative schemas and generated shared TS SDK | VAS-08 | TODO | OpenAPI/event schema generation, runtime boundary validation, deterministic generation and compatibility tests |
| VAS-10 | Unified run events, bounded parsing and replay | VAS-07, VAS-09 | TODO | UTF-8/SSE framing, IDs/cursors, explicit retention gaps, durable terminal state, bounded slow-client handling |
| VAS-11 | Exact approval and verified action reconciliation through existing framework owners | VAS-00, VAS-02 | TODO | G61/G62/G70: bind owner/payload/version/expiry; backend commit then response loss; receipt lookup or safe same-key guarantee; unknown blocks replay |
| VAS-11a | Keep durable approval admission off the async event loop | VAS-00, VAS-02 | LOCAL_PASS | Existing store owns expiry/load/single-use claim; real SQLite contention, cancellation and competing claim tests; no receipt verification or claim reopening |
| VAS-12 | Complete member continuation and cancellation lifecycle | VAS-11 | TODO | G61/G63/G70: restore completed batches, no whole-member replay; own cancellation and preserve partial effects |
| VAS-13 | VS Code migrates to shared SDK/state; secure webview and remote workspace | VAS-09, VAS-10, VAS-11, VAS-12 | TODO | Real ephemeral server + installed VSIX smoke; auth expiry, reconnect, approval, cancellation, capabilities and workspace boundaries |
| VAS-14 | Web UI migrates to the same API and state semantics | VAS-09, VAS-10, VAS-11, VAS-12 | TODO | Browser smoke against the same server/fixtures as VS Code; embedded Chainlit retains shared service path; remote mode uses API |
| VAS-14a | Workflow/formation authoring contract and consumer inventory | VAS-03, VAS-05a | TODO | Existing visualizer is not an authoring UI; reuse canonical compiler, coordinator and formation registry; typed validation and versioned save/load |
| VAS-14b | Drag/drop plus accessible workflow and formation editor | VAS-14a, VAS-09 | TODO | Node/edge editing, all registered formation choices, bounded workflow styles, undo/redo, stable definition hash, reload and keyboard parity; no second runtime |
| VAS-14c | AgentBrowser drag and transport co-design capability | VAS-14a | TODO | Current 1.15.2 catalog has no drag primitive and policy restricts WS/SSE; add supported capability or record blocker, never silently substitute pointer/stream proof |
| VAS-14d | Headed OIDC workflow→Victor→Sandhi→provider demo | VAS-14b, VAS-14c, VAS-14e, VAS-05d, VAS-05f, VAS-11, VAS-12 | TODO | AgentBrowser snapshots + exact definition + verified execution/approval/cancel + human/actor/accounting joins; deterministic then released-provider acceptance |
| VAS-14e | Victor adapter for AnvaiOps/Sandesha/Sandhi family theme and bundled UI assets | VAS-14a, VAS-21a | TODO | Pinned semantic token source/adapters and drift guard; ink/teal, light/dark/high-contrast, responsive/keyboard/reduced motion; headed visual/functional snapshots; OSS/commercial boundary |
| VAS-15 | GraphQL/MCP/legacy adapter convergence and deprecation | VAS-08, VAS-13, VAS-14 | TODO | Inventory external consumers, policy/attribution parity; one owner, compatibility window; retire only evidenced duplicates |
| VAS-16 | Measured performance and broader lifecycle acceptance | VAS-12, VAS-13, VAS-14, VAS-15 | TODO | Same-workload baseline/comparison; cold/warm, 1/8/32 concurrency, memory/backpressure, deadlines and crash/recovery |
| VAS-17 | Released foundation deployment and lifecycle acceptance for C5 | VAS-02, VAS-11, VAS-12 | TODO | Full green promotion/release CI, artifact/binary/source/config IDs and rollback; verify released Sandhi/InferFlux readiness without clearing cache |
| VAS-18 | Full mixed-team C5 verdict | VAS-17 | TODO | Six-Qwen/one-ZAI harness; unchanged deliverable/pytest/session/accounting gates; reviewed verdict on InferFlux #184 |
| VAS-19 | Matched formation cohorts and remaining semantics | VAS-18 | TODO | Preserve ZAI reference, simpler explicitly labelled local tasks, all 12 formations + 3 policies; G72 opt-in strict hierarchy separately |
| VAS-20 | OSS shared API/UI main promotion and release | VAS-16, VAS-17, VAS-14d | TODO | Full green promotion/release CI; publish server/SDK/VSIX/docs and compatibility matrix; verify installed artifacts |
| VAS-21 | Cohesive product-family UI across Victor, Sandhi, Sandesha and AnvaiOps | VAS-00 | TODO | Parent stays open until shared token ownership, product adapters and cross-product acceptance pass; existing standalone OSS deployment remains supported |
| VAS-21a | Review/version shared semantic token contract and publication boundary | VAS-00 | TODO | AnvaiOps source, Sandesha alignment, licenses/provenance, generated CSS adapters and drift check; no private checkout dependency |
| VAS-21b | Align Sandhi dashboard with the family theme | VAS-21a | TODO | Sandhi-owned linked worktree/PR; adapt existing dashboard CSS/JS; preserve OIDC/key/public-read modes, protected controls, real data/empty/error states, responsive accessibility and standalone packaging |
| VAS-21c | AnvaiOps-owned cross-product visual, identity and commercial-shell acceptance | VAS-14e, VAS-21b, VAS-05d | TODO | Headed AgentBrowser snapshots and actions on released artifacts; same brand/navigation, distinct audience/roles, no session/credential leakage; verify AnvaiOps/Sandesha current owner changes before adoption |
| VAS-22 | Technology allocation and measured compute qualification | VAS-00 | TODO | [Decision](victor-agent-service-technology.md): TS clients, async Python orchestration, existing Rust compute path; document baseline/SLOs in VAS-16 and qualify only measured hotspots; no rewrite assumed |

**Two delivery paths:** VAS-11 → VAS-12 → VAS-17 → VAS-18 preserves the original
framework/C5 mission and does not wait for GraphQL deprecation or either UI
migration. VAS-03–VAS-10/VAS-13–VAS-16 → VAS-20 delivers the shared API platform.
VAS-11 reuses existing framework identity/pause/action owners and does not wait
for new HTTP run endpoints; VAS-05/07 later expose those same contracts. The C5
harness calls the framework/providers directly. New API event work is not a
technical prerequisite for its verdict. Parallel sessions can claim independent
ready rows; the default remains the fewest worktrees and one accountable owner.

VAS-21c is commercial integration evidence owned by AnvaiOps. It does not gate
Victor OSS release VAS-20, standalone functionality or the original C5 verdict.
Generic reusable UI/auth/API work stays in Victor; commercial orchestration of
product deployments and entitlements stays in AnvaiOps, consuming public APIs.

Each row is a milestone, not a requirement to combine all its work into one PR.
If scope exceeds a reviewable increment, add child IDs (for example VAS-05a/b)
with explicit acceptance before coding; the parent stays open until all children
pass. Avoid cross-row feature bundles. Independent work may overlap only with
separate file ownership and recorded coordination, never shared mutable branches.


## Scope reconciliation and repository ownership

This matrix maps the requests accumulated across sessions to delivery owners. It
is an index into the ledger and existing handoffs, not a second status table.
Historical claims must be reverified against exact source, release and runtime
identities before they satisfy a current gate. Repository ownership names below
are responsibilities; an ACTIVE claim still needs a named session and branch.

| Requested outcome / insight | Authoritative task or evidence owner | Completion test / boundary |
| --- | --- | --- |
| WS-A through WS-I, nomenclature, orphan integration and unique formations | Formation handoff, VAS-19 | Preserve landed PR references; audit 12 formations/3 policies and duplicate semantics before additions; no invented new denominator |
| Exact approvals, durable action reconciliation and member continuation | VAS-11/12, G61/G62/G70 | Crash/commit-with-lost-response, stale approval and whole-member replay negative cases; verified effects before success |
| Every API/client tells the same truth | VAS-03/04/06–10/13–15 | Same principal/session/run/outcome semantics; stream EOF is not success; explicit compatibility/deprecation |
| Agent-as-service with Kanidm, scoped keys and authorization policy | VAS-05a–d | Browser/native/workload registrations, denial/expiry/revocation, workspace ownership and operator setup documentation |
| User → Victor → Sandhi → provider identity | VAS-05e/f; Sandhi auth owner | Immutable initiator/actor binding and scoped delegation; workload attribution alone is not human authorization |
| Sandhi embedded library and optional external gateway | VAS-05e/f/17; Sandhi owner | Explicit capability/profile matrix, identical accounting contracts, no failure-driven bypass |
| Shared OIDC mechanism reuse | VAS-05a/b/e; Sandesha/Sandhi candidate owner | Review candidate HOLD/release/security constraints before reuse; product authorization stays distinct |
| Theme and coherent commercial product experience | VAS-14e/21a–c; each product owner | Ink/teal semantic tokens, provenance/drift, accessibility and headed functional snapshots; no OSS dependence on private product code |
| Actual drag/drop workflow and formation authoring | VAS-14a/b/d | Versioned canonical graph, save/reload/hash, keyboard parity and actual execution; layouts alone do not qualify |
| AgentBrowser headed actions and evidence | VAS-14c/d/21c; AgentBrowser owner | Supported drag primitive and approved stream transport, isolated users, screenshots plus semantic outcome assertions |
| Python vs TypeScript vs Rust and compute placement | VAS-22/16, native strategy | Keep existing owners; bounded async waits, no event-loop blocking, measured native/IPC trade-off and packaging parity |
| Dual AMD/NVIDIA models and embeddings | VAS-17; InferFlux deployment/placement owner | Released build and pinned model/device/config identity; concurrent generation/embeddings and readiness on actual consolidated endpoint |
| Endpoint admission/rate limits and gateway deadlines | VAS-17/16; Sandhi/InferFlux owners | Measure passages/tokens/concurrency, not guessed requests/second; document model/global deadlines, timeouts as failures; preserve rollback |
| Cache counts, tokenizer units, session/request tracing and conservation | VAS-17/18; TD-0028, InferFlux #184 | Wire/SQLite/C4/dashboard joins, reporting coverage distinct from executed reuse; preserve original failed and passing evidence |
| Sandhi terminal observations, settlement and lifecycle recovery | VAS-17; Sandhi owner and latest durable-settlement handoff | Reverify released behavior for streaming, cancellation, process death and bounded recovery; source merge is insufficient |
| Six-Qwen/one-ZAI full mixed-team C5 | VAS-18, InferFlux #184 | Unchanged harness, member artifacts + task pytest + distinct sessions + accounting; no replay of passing five-call probe just to regenerate it |
| Better model first, then simpler local-model cohorts | VAS-19 | ZAI reference then labelled simpler Qwen tasks; model swaps require measured task evidence and InferFlux placement acceptance |
| Research evidence and best-practice audit | VAS-03/11/12/22; formation gaps | Reconcile existing audit/PDF findings, use supplied arXive API for relevant unanswered design questions; record paper/version, applicability and rejected recommendations, no speculative framework proliferation |
| Node/dependency/VS Code release health | VAS-02/13/20; dependency PR owner | Installed VSIX, supported runtime, audits and packaged smoke; fetch/review outstanding PRs before superseding them |
| MkDocs/GitHub Pages, main promotion and consistent releases | VAS-17/20; release owners | Documentation build/link/deployment, matching package/binary/source versions, post-release installed smoke; do not equate develop merge with published release |
| Reboot-safe work, upstream freshness and parallel-session reconciliation | Checkpoint protocol below, every ACTIVE owner | Remote PR evidence plus sanitized durable archive; fetch first, FF where possible, rebase only unpublished work or merge published ancestry without force-push |

## Delivery order and progress accounting

1. Finish the isolated VAS-03a repair and preserve its regression evidence.
2. Close the remaining contract/identity decisions and framework recovery gates
   (VAS-03/05a/e and VAS-11/12). Shared token review VAS-21a and Sandhi theme work
   VAS-21b can proceed independently; theme work does not delay C5 foundations.
3. Verify released Sandhi/InferFlux lifecycle/admission/settlement and execute
   C5 through VAS-17/18. Maintain timeout and failed-evidence semantics.
4. Build authenticated ownership/durable admission and the shared SDK/API, then
   migrate editor/browser clients and add actual workflow authoring. Use VAS-14a's
   small TypeScript/Svelte reuse spike before selecting graph UI dependencies.
5. Complete headed identity/authoring/product-family acceptance, measured capacity,
   formation cohorts, docs, main promotion and installed release verification.

Critical controls precede UI convenience and speculative acceleration. Independent
work need not wait for unrelated roadmap milestones, but its own gates still apply.
There are no calendar or effort estimates yet: the evidence does not support a
credible total-project percentage. Report separately (a) original WS 9/9 landed,
(b) leaf tasks merged, (c) released and (d) live acceptance cases passed/total with
failed/blocked/not-run counts. Never count both a parent and its children or use
source completion as a proxy for C5/model quality. Before claiming full closure,
all requested outcomes above must have accepted evidence or an explicit,
user-approved scope change; a TODO plan entry is not delivery.

## TDD and test ownership

Before a runtime edit, inspect the existing owner below and list the invariant
being changed. Run the minimal regression RED on the baseline, implement the
smallest repair, run GREEN, then the affected suite and applicable smoke. Preserve
RED and final GREEN evidence with SHAs. Do not weaken assertions to match defects.

| Work | Existing test owner / initial regression | Required negative and smoke cases |
| --- | --- | --- |
| VAS-02 activation | `vscode-victor/src/test/suite/extension.test.ts`, `codeActionProvider.test.ts` | Actual activation/all advertised commands; explicit/cursor/serialized targets and malformed rejection; VSIX package |
| VAS-03/08 HTTP parity | `tests/unit/api/test_vscode_extension_contract.py`, `tests/unit/integrations/api/test_fastapi_chat_request_correlation.py`, `tests/unit/api/test_consolidated_routes.py` | Execute real TS request fixtures against real router, not just regex paths; legacy and new payloads, malformed requests, no duplicate routes |
| VAS-04/09/10 TS wire | `vscode-victor/src/test-unit/victorClientStream.unit.test.ts`, `victorClient.unit.test.ts` | Paused202 fields, EOF/error terminal distinctions, byte fragmentation, extra/unknown versions, false cancellation acknowledgement |
| VAS-05 auth | `tests/unit/integrations/api/test_api_auth_boundary.py`, `test_ws_auth_gate.py`, `test_chat_resume_routes.py` | HTTP/WS same identity; expired/wrong issuer/audience tokens, cross-owner IDs, machine scopes, local mode isolation |
| VAS-06 sessions | `tests/unit/web/test_session_store.py`, `tests/unit/framework/test_client_close_guard.py` | Admission races, init failure, bounded sessions, close/disconnect; no lock held over slow initialize/shutdown |
| VAS-07/11 durability | `tests/unit/agent/test_paused_run_persistence.py`, `test_paused_run_expiry.py`, `test_durable_resume.py`; `tests/unit/framework/test_client_resume.py` | Commit-then-timeout, restart, duplicate events, changed-key payload, stale/expired approval, unknown effects; authoritative receipt |
| VAS-10 events | `tests/unit/framework/test_wire_events.py`, `test_wire_events_member.py`; `tests/unit/integrations/api/test_event_bridge_queue_bounds.py` | Replay gaps, slow consumers, bounded queues, duplicate/out-of-order events, durable terminal status distinct from lossy telemetry |
| VAS-12 members | `tests/unit/teams/test_member_pause_resume.py`, `tests/unit/agent/subagents/test_member_approval.py`, `test_member_pause_trigger.py` | Completed members/tools never replay; ownership mismatch, interruption between checkpoints, partial effects and cancel propagation |
| VAS-13/14 UI | Existing VS Code test owners; `tests/unit/ui/chat_app/test_app_resume.py`, `test_approval.py` | Same backend through both UIs, real network boundary, reload/resume, login expiry, malicious messages, remote workspace and accessibility |
| VAS-15 adapters | `tests/unit/integrations/api/test_graphql_schema.py`, `tests/integration/integrations/mcp/test_mcp_server_client_e2e.py`, `tests/unit/protocols/test_protocol_adapters.py` | Same policy/outcome/error across adapters; supported old consumers; no bypass from alternate transport |
| VAS-18/19 live | `scripts/validation/multiagent_gateway_live.py`, `formation_gateway_matrix.py` and their existing tests | Actual members, artifacts + passing task pytest + distinct sessions; wire/SQLite/C4/dashboard conservation and bounded deadlines |

Only add a new test module when no existing owner covers the boundary (for example
cross-language HTTP consumer/provider execution). Document why in the PR. When
removing tests, record invariant mapping and before/after affected line/branch
coverage; file counts or superficially similar names do not prove duplication.
Known weak path-only/synthetic tests are replaced once the stronger test covers
their invariant, not simply removed to make the suite green.

## Smoke ladder and completion gates

| Gate | Runs when | Passing evidence |
| --- | --- | --- |
| S0 deterministic contracts | Every touched contract | Schemas/fixtures, malformed inputs and boundary guards; no network/provider required |
| S1 real local HTTP | Every transport/server/SDK increment | Ephemeral canonical server, deterministic provider, actual client bytes; auth/session/run/events/approval/cancel outcomes |
| S2 packaged clients | UI/packaging increment | Installed isolated VSIX and browser UI on same API; no personal profile, real user actions, UI/terminal state agreement |
| S3 lifecycle/fault injection | Durable/recovery/event increment | Worker crash/restart, client loss, stream truncation, stale approval, commit-with-lost-response; no duplicate effect |
| S4 performance | Before accepting a performance claim | Reproducible same-workload p50/p95, throughput, queue wait, CPU/RSS and resource bounds; no guessed capacity |
| S5 released services | Before live model acceptance | Release + source + binary + config identities, model readiness, routes/deadlines, auth and rollback recorded |
| S6 C5 and cohort evidence | After S5 and required lifecycle gates | Reviewed actual-member results, complete accounting joins, artifacts/test results and unique sessions |

S0–S3 use deterministic doubles only at provider/business-system boundaries;
label them as such. They prove control/transport behavior, not model quality.
S5 startup/readiness is not S6. Timeouts are failures. Retain original failed and
interrupted evidence. Do not replay a passing five-call experiment merely to
regenerate evidence, clear shared caches or transfer credentials.

From the worktree root, use `.venv-codesign/bin/python -m pytest <cited-owner>
--no-cov -q`, and before push run `VICTOR_ENABLE_MLX_PROVIDER=0
.venv-codesign/bin/python -m pytest tests/ --collect-only --no-cov -q`.
Use the MLX flag only for unrelated control tests, never to claim MLX acceptance.
Run Black/Ruff/MyPy for touched Python, TS lint/compile/unit/real-host tests for
extension changes, webview check/build, `npm ci` for both graphs and production
VSIX packaging. Docs changes require MkDocs/link checks. FEP changes also require
`.venv-codesign/bin/victor fep validate <proposal-path>`; a passing docs build
does not establish FEP structure validity. Hosted CI is not an
edit/test loop; validate locally and independently review the exact candidate first.

C5 retains the confirmed 120-second buffered gateway deadline unless a separately
reviewed, versioned plan changes it. A larger client timeout does not override it.
Separate explicit cache counts, reporting coverage and executed reuse; zero
reported cache does not prove zero reuse. Discover current routes after restart;
do not blindly reuse old 8081/18081 or dual-GPU deployment instructions.

## Checkpoint record and handoff protocol

At each material boundary (RED, GREEN, PR, merge, release, blocked state or end of
session), update the row and append a checkpoint. Every implementation PR carries
its tracker update. Record at least:

```yaml
checkpoint: YYYY-MM-DD-task-id-N
id: VAS-XX
state: LOCAL_PASS
owner: session or accountable maintainer
branch: fix/example
worktree: absolute path, disposable after merge
base_sha: full fetched develop SHA
tested_sha: exact commit, or explicitly uncommitted tree
review: reviewer, commit, verdict, unresolved findings
pr: URL or not opened
merge_sha: null until verified
checks: command, exit code, count, evidence path or CI URL
runtime: source/binary/config/release IDs only when exercised
remaining: concrete unmet gates
next_action: one executable next step
```

No secrets, bearer tokens or private request bodies in records. Use sanitized
artifacts, hashes and operation/request IDs. Keep committed evidence compact;
large logs belong in durable CI artifacts or a declared local archive with hashes.
A local archive is recovery assistance, not cross-machine proof or a deployed
service. Do not require ephemeral files to understand the next action.

### Checkpoint 2026-10-07-VAS-00-02

- Owner: current Victor API/dependency session; branch/worktree above.
- Base: `b8e195cc68e009c68b2cfc9b96ac8cf5b74bef84`; implementation evidence was
  gathered on the uncommitted candidate, not an immutable release.
- VAS-00: audit accepted. Actual router 422 without runtime init; compiled-client
  transport stubs reproduce lost approval fields, premature EOF and false cancel.
  Existing API/auth/resume tests: 23 passed; wire/session/boundary: 33 passed.
- VAS-02: RED actual extension activation failed on duplicate symbol commands;
  GREEN 999 real-host tests after one registration owner, target decoding and
  six inactive advertisements pruned. Unit tests: 50 passed; both npm audits zero;
  VSIX built; 83 repository contracts passed; 33,883 Python tests collected.
  Existing lint/bundle warnings and low whole-extension coverage remain disclosed.
- Node: 24.21.0 nvm default; `.nvmrc` is canonical for CI. No provider deployment.
- VAS-01: authored plan/FEP/links; formal FEP acceptance and API implementation
  remain outstanding. Independent review and exact candidate/PR evidence must be
  appended below or linked from the publication PR before source acceptance.
- Next action: finish current candidate review, validate final docs/source, commit,
  push and drive its PR to green. Then reconcile VAS-01/02 from the actual merge
  record and start VAS-03; never resume by rerunning the original WS-A.

### Checkpoint 2026-10-07-publication-review

- PR: [#1251](https://github.com/anvai-labs/victor/pull/1251), targeting develop.
- First candidate: `32f969ac3d847314b6987329ddc570b192251ab7`, independently reviewed;
  code/local tests passed. The two FEP-specific CI gates failed because the first
  proposal lacked required canonical sections; a passing MkDocs build did not
  exercise that validator. Preserve that failure, not a full-green claim.
- Correction: restructure the same design into required sections, record unresolved
  decisions/compatibility/implementation and validate the actual FEP locally.
  The publication PR's latest head/review/CI is authoritative for the correction;
  never merge using the earlier commit's attestation. No runtime code changed.
- VAS-01/02 stay LOCAL_PASS until the actual merge is verified. On the next
  checkpoint, replace their states with MERGED and cite #1251's merge SHA if green;
  do not repeat the dependency/activation work merely because this snapshot predates
  its merge. VAS-03 and independent framework VAS-11 are the next ready candidates.

## Resume after a session change or reboot

1. In `~/code/codingagent`, fetch origin with prune; inspect status, worktrees,
   current PRs and this file from `origin/develop`. If not yet merged, find the
   `fix/vscode-security-locks` PR and read the plan from its exact head. Do not
   assume a local main checkout contains the latest plan.
2. Compare remote PR/CI/merge state with the ledger. Mark newly verified merges
   and supersessions with links and SHAs before choosing work. If develop advanced,
   fast-forward an unmodified base. Merge develop into a published branch to avoid
   force-push; rebase only an unpublished branch after preserving its work. Re-run
   affected checks and commit-bound review after reconciliation.
3. Reuse an existing owned worktree if intact. Otherwise create a linked worktree
   from the recorded remote branch or fresh origin/develop; never overwrite dirty
   files or another session's worktree. Restore archive patches only after diffing
   against Git to avoid double application. Recreate venv/dependencies; temp
   processes, tunnels and test hosts are not durable infrastructure.
4. Select the first ready, unaccepted row and claim it with owner/branch/baseline
   and next action. If another session owns it, coordinate explicitly; do not start
   duplicate implementation. A stale timestamp alone does not transfer ownership.
5. Verify credentials/services privately only when the selected row needs them.
   Continue the TDD/smoke ladder; update this tracker and the relevant G-gap record.
6. Before stopping, commit/push reviewed work where gates allow, record incomplete
   gates honestly, preserve pending changes outside `/tmp`, and update the root
   archive restart pointer. Remove only this task's worktree after its PR merges.

Resume prompt:

> Read docs/architecture/victor-agent-service-plan.md and FEP-0039. Fetch origin and
> reconcile the ledger with merged/open PRs before acting. Continue the first ready
> VAS item using TDD in the existing test owner, real-server/packaged-client smokes,
> independent review and all CI gates. Update the ledger/evidence each milestone.
> Preserve G61/G62/G70/C5 limits, credentials, shared caches and other sessions.

### Checkpoint 2026-10-08-VAS-03a-claim

- Verified #1251 merged into develop at `470c9ccdcb222dcfd8acce5b4c52fb27321ac952`;
  41 SUCCESS, Trivy NEUTRAL, PR Pages deploy SKIPPED; no failed/pending checks.
- No open PRs at claim time. Other session worktree is preserved.
- VAS-03a is a defect repair to the existing client/server contracts, permitted
  independently of Draft FEP-0039 acceptance. VAS-03 as a whole stays open.
- Duplicate-test audit: the majority-of-endpoints assertion is implied by exact
  missing-route equality; replace that redundant assertion with executable payload
  coverage while retaining source extraction/route-presence guards.
- No provider calls, service deployment or C5 acceptance is planned in this increment.

### Checkpoint 2026-10-08-authentication-direction

- User selected Kanidm OIDC plus scoped API keys and explicit authorization policy.
  [Policy design](victor-agent-service-auth-policy.md) records concrete proposal,
  current local Sandhi/InferFlux source evidence and standards. VAS-05a–d are
  tracked separately; no authentication implementation/deployment is claimed.
- VAS-03a smoke also reproduced web import failure after nested settings migration
  (G74). Repair reads ServerSettings and unwraps SecretStr; no second secret owner.

### Checkpoint 2026-10-08-VAS-03a-local-validation

- Baseline: `470c9ccdcb222dcfd8acce5b4c52fb27321ac952`; tested tree uncommitted.
- RED: three TS payload assertions failed; actual compiled TS → core HTTP returned
  422. Web router import and two configured/default settings tests failed on
  removed flat settings. Original failures are retained in the local evidence archive.
- GREEN: one compatibility envelope derives newest-user content once and dispatches
  once. Existing web message/session behavior and core messages contract remain;
  no server aliases, retries or session-ownership claim added.
- Executable HTTP/config smoke: 13 passed, including clean minimal `[api]` environment;
  two real client turns per server, core correlation/web reuse, invalid auth and
  malformed request rejection. Agent execution is a deterministic double; startup
  lifecycle and live models are not exercised. CI requires a compiled client and
  these tests when related source/contracts change, rather than accepting a skip.
- Affected API/web/settings/workflow-guard suites: 352 passed. TS: 54 unit and
  999 isolated VS Code host tests passed. Whole-extension statement/line coverage
  remains low (5.38%/5.46%), not a claim of full API coverage. Lint has nine existing
  warnings; webview build has its existing bundle-size warning. MyPy/Black/Ruff,
  FEP validation, MkDocs/build-site links and repo hygiene passed.
- Removed only the redundant majority-route assertion: before/after route coverage
  is identical across 19 modules, 564 executed lines and zero executed branches.
  Exact route drift and extraction guards remain; actual payload tests add coverage.
- Review found an HTTP fixture singleton leak; corrected before final validation.
  Auth design now requires separate browser/native Kanidm registrations and explicit
  verified account linking for cross-client ownership. FEP remains Draft.
- Remaining: final commit-bound review, hosted CI and squash merge;
  source merge is not a release, OIDC deployment, session isolation or C5 acceptance.

### Checkpoint 2026-10-08-workflow-ui-and-Sandhi-scope

- User explicitly requested Kanidm across workflow UI, agent-as-service and Sandhi;
  scoped API keys remain supported under the same authorization policy. They also
  requested actual drag/drop formation/workflow authoring and a headed AgentBrowser
  demo. VAS-05e/f and VAS-14a–d make those delivery gates explicit.
- Independent read-only review of `../sandhi` (fetched develop equals local SHA
  `1a7e6ed3f35c001892fcac28333e74c1717108f8`) found strict gateway identity/grants
  and vault mapping, but no automatic Victor-user→Sandhi delegation. See the
  [co-design evidence and acceptance matrix](victor-agent-service-auth-policy.md).
- Scope order: finish VAS-03a defect repair; agree strict identity/policy and
  authoring contracts; establish ownership and auth/delegation; implement editor
  and supported browser actions; run deterministic UI/HTTP lifecycle tests; then
  released-service headed acceptance and C5. Independent framework VAS-11 remains
  eligible; no new UI requirement silently certifies or closes the original C5.
- Browser baseline uses an isolated headed session with fixture data only. Current
  visualizer/layout interactions are not drag/drop authoring or OIDC acceptance.

### Checkpoint 2026-10-08-shared-theme-and-runtime-shape

- User requires a cohesive AnvaiOps/Sandesha/Victor service design. Fetched both
  sibling origins; dirty in-flight branches remain untouched. VAS-14e records
  exact source/token baselines and visual/accessibility acceptance.
- Sandhi embedded library and external gateway remain explicit supported shapes.
  External gateway is preferred for the hosted multiuser demo; embedded support
  needs control/capability parity rather than an assumed OIDC middleware feature.
- Sandesha has existing shared OIDC extraction work on adoption HOLD. Reconcile
  that source, dual-consumer/security/performance proof and accessible package
  before creating another verifier or treating its prototype as production ready.
- Headed AgentBrowser baseline captured actual visualizer failure (`cytoscape is
  not defined`) and absent authoring controls; preserved as failed/limited UI
  evidence, not a drag-and-drop, OIDC, streaming or formation pass.
- Full collection initially failed because the new local venv lacked the pinned
  Sandhi binding. Installed declared `sandhi-gateway==0.11.0`; final collection
  passed: 33,888 tests. This was an environment repair, not a dependency upgrade.

### Checkpoint 2026-10-08-family-and-technology-scope

- User explicitly added Sandhi dashboard alignment to the AnvaiOps/Sandesha family.
  VAS-21 now owns the cross-repository theme delivery and acceptance dependency;
  VAS-14e remains Victor's adapter. No sibling UI implementation is claimed here.
- Recorded the bounded polyglot decision: TypeScript/Svelte reuse for rich clients,
  typed async Python for orchestration, existing Rust/PyO3 for measured compute,
  InferFlux for GPU placement. VAS-22/16 own measurement, not an assumed rewrite.
- Added request-to-owner coverage and delivery order, including release/lifecycle,
  full C5, research applicability, shared identity, real authoring and reboot
  recovery. This reconciles scope without reopening landed WS-A–WS-I.
- Sandhi's baseline dashboard uses standalone CSS/JS with blue accent tokens;
  theme migration can stay within those assets. Its local untracked handoff files
  and the dirty AnvaiOps/Sandesha work remain untouched.

- Final review corrected the delivery dependency: VAS-06 establishes ownership
  after verified authentication (VAS-05b), then VAS-05c proves authorization across
  every owned resource/transport. The CI compiled-client trigger now includes the
  shared wire-event owner. No authorization-complete claim precedes ownership.

### Checkpoint 2026-10-08-oss-commercial-boundary

- User confirmed reusable OSS building blocks belong in Victor and the commercial
  offering belongs in AnvaiOps. The technology decision now gives an explicit
  repository ownership table; Sandhi/InferFlux retain their own OSS responsibilities.
- Removed commercial-shell VAS-21c from the OSS VAS-20 release prerequisites.
  Generic security, durable execution, APIs and reusable workflow UI remain OSS;
  commercial packaging/billing/entitlements and managed operations remain AnvaiOps.
- PR [#1252](https://github.com/anvai-labs/victor/pull/1252) publishes the stream
  repair and this plan. Initial head `bcfb331e51f3e4aa1f59e08de30686db83c063d3`
  passed independent review; hosted quick-test failure requires correction and
  a fresh exact-head review/CI verdict. Preserve failed evidence; not merged yet.

- CI failure diagnosis: changed-file testing installs the dev extra, which lacked
  Uvicorn, and invoked the pytest executable without the repository root on the
  import path for the source-only `web` namespace. Add the actual HTTP test runner
  to dev dependencies and invoke that job with `python -m pytest`; keep the real
  HTTP assertions and dedicated required compiled-client gate intact.

- Corrected candidate local validation: 96 real HTTP/config/CI-policy tests passed,
  including the required compiled-client cases; formatting, lint, docs/link/FEP
  checks passed. The first hosted run's only underlying failure was quick tests
  (the aggregate also failed). Do not reuse its green jobs as the new head verdict.

### Checkpoint 2026-10-08-VAS-11a-claim

- Reconciled PR1252 from GitHub: merged `a5fa47e444e6f973ad95149ae0677327df060dc7`,
  41 successful applicable checks including Vertical Py3.12, one deployment skip
  and one neutral check; independent review clean. VAS-03a is MERGED, not released.
- No open Victor PRs at claim; the other existing local worktree remains untouched.
- VAS-11a targets synchronous SQLite expiry/load/claim in `VictorClient.resume()`.
  Journal writes already run off-loop after #1249; approval admission still can
  block unrelated async work for the database lock timeout. Audit the existing
  client/store tests for duplicate invariants before adding regression cases.
- Preserve existing single-use/default-off/error behavior and existing store
  ownership. Cancellation or a failed claim must never reach tool execution.
  Verified backend receipts, principal ownership and member continuation remain open.

### Checkpoint 2026-10-08-VAS-11a-local-repair

- RED: both real-SQLite contention cases blocked the loop until the five-second
  watchdog released the write lock. GREEN: 96 client/store/durable-resume tests
  passed after offloading expiry/load/claim for the built-in persistent backend.
- One store-owned helper receives the captured store; only admission runs in a
  worker. Session hydration and canonical runtime dispatch require successful
  await. Concurrent resumes still claim once; a committed claim with lost
  acknowledgement dispatches nothing and remains consumed.
- Cancellation cannot stop an already-started SQLite worker. It may consume the
  approval after the waiter cancels, but never dispatches or reopens that claim.
  Recovery for consumed-but-undispatched approvals remains unresolved.
- Injected non-ProjectDb stores retain caller-thread behavior: no new protocol
  thread-safety requirement or failure fallback. ProjectDb subclasses must honor
  the existing thread-local/locking contract; blocking custom stores retain their
  own responsiveness responsibility. The synchronous status API is unchanged.
- Duplicate-test audit: client admission concurrency/loop tests cover a distinct
  boundary from store persistence and action-journal dispatch tests. No existing
  test was removed; sequential single-use, expiry, hydration, default-off and
  opt-in contract cases remain. No extra test module or public API was added.
- G77 records this newly discovered gap. G61/G62/G70, backend receipt verification,
  full member continuation and C5 remain open. Independent exact-head review,
  affected suites, collection and hosted CI still precede merge.

- Broader validation: 216 affected tests passed (framework clients, durable stores,
  resume, member/API/UI resume and guards). Independent review strengthened the
  existing race test with a bounded barrier so both workers read pending before
  either claims. Final focused client/CI-selector suite: 55 passed; collection:
  33,894 tests (collection only). Mypy on both production modules passed.
- Extended the existing changed-test selector/parametrized regression so a future
  paused-store-only edit executes the client admission tests. Its missing mapping
  failed RED before the one-line mapping correction; no separate CI job was added.

- Final local CI-selection run: 174 tests passed with coverage enabled; Black/Ruff
  and mypy for the two production modules plus selector passed. Documentation
  build and built-site link checks passed. Cumulative independent review is clean;
  the checkpoint now points at publication, not repeating the completed repair.
