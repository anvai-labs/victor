---
plan_id: VAS-2026-10
status: active
updated: 2026-10-09
baseline_develop: 13a3c8261ab27c69afd7c45a15073aa37229c0d9
next_task: VAS-17
---

# Victor shared agent-service implementation plan and tracker

**Canonical task-status ledger.** Browser UI, VS Code and external API consumers
will converge on one Victor service and contract family. The user authorized this
plan after the [API audit](../development/victor-agent-service-audit.md).
[FEP-0039](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0039-unified-agent-service-api.md)
owns the proposed architecture and is in Review, not Accepted. This file owns execution status;
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

- Fetched develop: `13a3c8261ab27c69afd7c45a15073aa37229c0d9` (#1269 merged).
- VAS-15a merged in [#1268](https://github.com/anvai-labs/victor/pull/1268),
  `e9004a5d8b483f653e258431b0cbd32c1bcad49d`, from baseline `07b076726`.
  The retained Python HTTP adapter preserves paused metadata and rejects invalid
  or incomplete streams without automatic POST replay. Existing protocol test owners
  reproduced 37 initial failures. Independent review found delayed CR terminal
  dispatch (two additional RED cases), quadratic suffix copying and malformed
  envelope fallback (four additional RED cases); all are fixed.
  Focused GREEN: 144 tests and a real authenticated core HTTP smoke using synthetic
  execution, including tool-call lists, lost terminator and denied credentials.
  Final affected GREEN: 696 protocol/API/web tests (including the compiled VS Code
  client over HTTP), 81 documentation tests, Black (3,917 files), Ruff, touched-module
  MyPy, strict MkDocs/internal links and repository hygiene; 34,082 tests collected
  (final collection without coverage instrumentation).
  [#1268](https://github.com/anvai-labs/victor/pull/1268) is merged. Its initial
  changed-file CI gate found a missing selector mapping, before running tests.
  Three additional selector RED cases now map the package, interface and adapter
  to existing canonical suites, instead of unrelated flattened-name matches.
  All 44 selector tests and 199 exactly selected tests pass, with 95% changed-line
  coverage. Exact candidate `e5420da0fd1e0a73840f221fbd0304207a0d2638`
  passed independent review and all applicable CI, including Vertical Py3.12:
  40 successful checks, one nonapplicable deployment skip and neutral Trivy.
  The squash-merge tree matches the reviewed candidate; its worktree is removed.
- VAS-15a test audit: strengthened the existing serialization assertion and replaced
  the mocked HTTP happy path with parameterized real HTTPX transport cases. Added
  malformed/framing/ownership cases to the existing adapter owner and reused the
  production-route fixture. No equivalent regressions were found to remove. The
  direct adapter remains separately unqualified (G82/VAS-15); no adapter removal,
  new service contract, durable continuation, release or C5 pass is implied.
- VAS-05g merged in [#1267](https://github.com/anvai-labs/victor/pull/1267),
  `07b076726de88a50d37df766e72a0945e2435496`, with clean exact-commit review
  and all 39 applicable CI checks, including Vertical Py3.12. Local evidence:
  113 Vitest, 1,012 actual host, 41 API/auth/HTTP and 76 documentation tests;
  34,030 collected. EventBridge now uses the VictorClient credential owner,
  fences replaced connections and releases owned subscriptions. The undiscovered
  placeholder test file was replaced by executable host cases. Server-side live
  revocation (G81), OIDC and C5 acceptance remain separate.
- VAS-03b merged in [#1266](https://github.com/anvai-labs/victor/pull/1266),
  `35e2d0aaf22760dfe8e084b9e21f79e37376e458`, with 40 successful checks.
  FEP review began 2026-10-09 02:15:20 UTC; minimum period ends
  2026-10-23 02:15:20 UTC. Maintainer consensus remains required; green CI alone
  does not accept the design. The source inventory and synthetic probes remain
  evidence only, not closure of parent VAS-03.
- VAS-04b is complete on develop: [#1265](https://github.com/anvai-labs/victor/pull/1265)
  passed clean exact-commit review and all 39 applicable CI checks, including
  Vertical Py3.12. Local evidence: 12 client/five host RED cases and one real HTTP
  RED, then 88 client, 1,006 actual host, 239 API and 76 doc tests; 34,030 collected.
  Paused metadata, guarded content consumers and late Composer session ownership
  are repaired. Ordinary transport-error paste fallback is unchanged; volatile UI
  retention is not restart-safe continuation or a complete approval experience.
- VAS-17 deployment preflight (2026-10-09): Mac Homebrew CLI/proxy are Sandhi
  0.11.0, released source `68580981a0008f27d17260cddc11a25020844488`.
  Loopback gateway and registered HTTPS edge respond; anonymous admin access
  is denied. The official Linux 0.11.0 artifact passed checksum verification and
  isolated copied-state TLS/auth/usage checks on aiserver1. The host owner then
  installed the staged unit and restarted it. Actual PID 2969496 uses the official
  binary, SHA-256 `5c804c4a6a9431a1be58e046b6541d974dc4f59befd2f564258a96db23a1ec56`.
  The unit changed only the executable; private state, credentials and TLS persist.
  Readiness alone does not establish provider credential availability.
- Existing DS3 replacement brokers passed token exchange/introspection for distinct
  member/accounting identities. Five isolated authorization checks passed with an
  empty store: anonymous admin denial, member admin denial, accounting version and
  usage reads, and accounting inference denial. The first copied-state startup
  exceeded its 60-second bound in macOS Keychain credential loading. A later
  attempt started and passed authorization but hit HTTP 502 on the obsolete 18080
  route. Preserve both failures. The explicit 18081 route passed both Qwen models
  and ZAI in staging; the old route and private configuration remain for rollback.
  Exact issuer/subject mappings and replacement broker commands are now active
  on the restarted local gateway. No credentials or Keychain ACLs were changed.
  Do not reuse superseded UUIDs or regenerate already-provisioned service accounts.
- Sandhi still uses Kanidm. [Anvai Identity's tracker](https://github.com/anvai-labs/anvai-identity/blob/7da3c01fb62ad778caf9298036da87e84cd7db6b/docs/planning/tracker.md)
  leaves T211 consumer qualification and T217 migration/rollback open. The 0.1.0
  local preview is not a hosted identity replacement. Follow Sandhi's
  [identity migration contract](https://github.com/anvai-labs/sandhi/blob/1a7e6ed3f35c001892fcac28333e74c1717108f8/docs/operator/identity-groups.md)
  for the current issuer and exact subject bindings.
- Local HTTPS OIDC deployment smoke passed ZAI `glm-5.3` and both Qwen models:
  three buffered calls, 49 input tokens, five output tokens, and explicitly
  reported zero cache tokens (3/3). Wire, SQLite and per-run totals reconcile;
  the dashboard delta matches exactly. Distinct run/session IDs and nonempty
  request IDs join to the new member subject. The initial probe used an incorrect
  session header and failed its assertion; retain that evidence separately from
  the corrected `x-sandhi-session` run. These are deployment probes, not Victor
  members. Zero reported cache does not prove zero executed reuse; browser login,
  restart durability, streaming/cancellation and full lifecycle remain unqualified.
- Remote Sandhi post-cutover smoke passed trusted TLS on 18788, anonymous admin
  denial, authenticated usage reads, existing client-credential exchange, Qwen3
  chat and two finite 384-dimensional BGE embeddings. Wire/SQLite/run accounting
  and dashboard delta reconcile two calls, 22 input tokens and one output token.
  Chat reports explicit zero cache; embeddings omit cache reporting (absent, not
  explicit zero). Remote auth remains its existing `tokens` mode; this is not an
  OIDC acceptance claim. Missing `openai:codex` and `zai:coding` credentials produce
  the same warnings as before the upgrade and remain unavailable there. No secrets
  were transferred or unrelated provider routes silently disabled. Mac ZAI remains
  the independently verified cloud route.
- VAS-17 released-origin checkpoint (2026-10-09): InferFlux
  [v0.5.0](https://github.com/anvai-labs/inferflux/releases/tag/v0.5.0) is published
  from `bfdf32c00ccf40c7af73b86df2eedabc20a7ae85` after main/tag CI, native package
  smokes and trusted CUDA/ROCm/same-process GPU gates passed. The deployed immutable
  mixed-GPU build uses that source; standard Linux release archives are CPU-only.
  Binary SHA-256: `66f2fa90fef39d7bea8a67b18c275745c9961c12add9da7470eded7cf2c46d71`.
  Config SHA-256: `fe799c5895b7e32f77704936e6b61ccc047771225b9a2407dea6122c20e6d5d6`.
  PID 152439 at acceptance serves loopback 8080 and TLS 8443; Qwen3 remains on AMD,
  Qwen2.5-Coder-14B and BGE on NVIDIA. All three models are ready without fallback.
  Private configuration, credentials, persisted caches and the old runtime/unit
  rollback snapshot are preserved. This replaces the earlier unqualified binary
  `8b10c5cdc40a7d44325735747a5bcf120406b294b650126235db0bd7fbd10d06`.
- [New synthetic release wire evidence](evidence/released-foundation-wire-2026-10-09.json):
  **16/16 passed**, twelve direct and four through released Sandhi 0.11.0/TLS 18788.
  Both Qwen models pass legacy buffered/SSE completions directly; BGE passes float
  and base64 batches of two finite 384-dimensional vectors. The existing gateway
  grant covers Qwen3 and BGE; permissions were not widened for Qwen14.
  Four gateway wire/SQLite/C4 joins and the dashboard delta reconcile **4 calls,
  48 input, 2 output and 50 billable tokens**. Legacy completions explicitly report zero cache
  (2/2); embeddings omit cache reporting (2/2), not an explicit zero assertion.
  Gateway request IDs echoed by InferFlux match SQLite; provider completion IDs
  are distinct identifiers. Acceptance retains the 120-second buffered gateway
  deadline and 125-second client bound; timeouts fail acceptance.
- Two failed harness attempts remain preserved: sending `stream_options` on a
  buffered request was correctly rejected; the next attempt incorrectly equated
  a provider response ID with a gateway request ID. Both restored and verified the
  old runtime before the corrected run. The passing run does not erase them.
  These are new synthetic wire probes, **not actual Victor member replay or C5**.
  Reported cache counts do not establish executed reuse. Tokenizer units, session
  leases, origin cancellation, mixed-load capacity and full lifecycle remain open.
- Sandhi [#336](https://github.com/anvai-labs/sandhi/pull/336) merged owned prepared
  admission as `21adf4b3165c2797a2f5321bfecb99ae343c81e2`: clean exact-commit review,
  all 11 applicable CI checks, 843 Rust tests (six ignored) and 84 selected SDK cases.
  This is source evidence only; serving Sandhi remains 0.11.0. It does not activate
  durable HTTP settlement or close the foundation lifecycle gate.
- Next VAS-17 owner: this co-design session, Sandhi branch
  `feat/buffered-accounting-ownership`, baseline `21adf4b3165c2797a2f5321bfecb99ae343c81e2`.
  Local WIP is not accepted: review found stale-ticket double transfer, missing
  single-admission/phase binding and terminal evidence loss when no runtime exists.
  The three findings now have RED/GREEN regressions; a further cutoff-cleanup
  regression is fixed locally (12 job-owner tests pass). Cumulative review and
  HTTP integration remain open. Next complete opt-in buffered
  HTTP accounting and bounded ReadyToSettle recovery. Keep one obligation across
  dispatch; preserve actual usage through cancellation/refusal; never replay the
  provider or invent zero usage. Release/deploy and qualify lifecycle before C5.
  FEP-0039 remains in Review; VAS-11c still needs a qualified production receipt
  operation. Do not restart merged repairs or create a new public Victor API.
- VAS-02a merged evidence: Linux Python 3.12/pip-tools 7.6.1 regenerated the CPU
  embeddings lock with `multidict==6.9.1` and removed stale Textual-only dependencies
  absent from current deployment metadata. Rust edge TLS resolves `rustls==0.23.45`
  and `rustls-webpki==0.103.15`. Original RustSec/embeddings audits each found one
  vulnerability; updated RustSec, Linux lock and complete installed-environment
  audits found zero. The installed local Victor/contracts wheels passed real
  HTTP multivalue header/query, deterministic CPU embedding pipeline and CLI smokes.
  No pretrained model download/quality or provider/C5 claim is made. All 28 edge
  tests/doctests, 79 lock/doc tests and 34,029 collection cases passed. Clean exact-
  commit review and all 39 successful checks, including Vertical Py3.12, preceded
  [#1263](https://github.com/anvai-labs/victor/pull/1263) squash merge. Existing
  default-branch alerts still require main promotion; no alerts were dismissed.
- VAS-11c qualification recheck: TaskStore mutates memory and saves JSON; database
  tools execute arbitrary caller SQL under process-local connection IDs; Jira/Slack
  return backend IDs after writes, without an action-key receipt lookup. None can
  be enabled as a production receipt adapter on this evidence. A named deployed
  business operation, stable backend identity, authenticated lookup and atomic
  effect/receipt contract must be supplied/qualified first. Preserve unknown and
  unsupported effects without replay. The existing SQLite fixture is conformance
  evidence only. No production adapter or member continuation is claimed.
- VAS-04a merged evidence: 23 client RED failures, six actual extension-host RED
  failures and a further UTF-8-after-terminal RED; repaired through existing owners.
  Both production HTTP entry points pass normal and lost-terminator smokes using the
  compiled client. Pending-approval cancellation is checked over authenticated HTTP.
  Final local checks: 76 Vitest, 1,000 actual VS Code host, 71 API/web tests;
  34,029 collected; Black/Ruff/MyPy and strict docs/link/hygiene checks pass.
  Independent review passed 76 Vitest and nine HTTP contract cases. Exact-commit
  review and all 39 successful CI checks, including Vertical Py3.12, passed before
  [#1262](https://github.com/anvai-labs/victor/pull/1262) squash-merged. No release claim.
- Limits: stream termination is not business success. Recognized event payloads still
  have legacy per-field defaults; generated schema/result validation, paused-run and
  approval fields, durable terminal state, remote cancellation and full VAS-04 stay open.
  No foundation deployment, released artifact or C5 acceptance is added.
- Upstream release audit (2026-10-08): Sandhi's latest published GitHub release is
  [v0.11.0](https://github.com/anvai-labs/sandhi/releases/tag/v0.11.0); Victor metadata
  and all three deployment locks already pin `sandhi-gateway==0.11.0`. InferFlux's
  latest is [v0.4.0](https://github.com/anvai-labs/inferflux/releases/tag/v0.4.0), a
  separately deployed service. No dependency bump is needed for these release pins.
  VAS-17 still owns deployed binary/config identities, release compatibility and
  lifecycle evidence; dependency installation alone does not enable or accept every
  upstream feature. Future upgrades need release-note/security review, synchronized
  metadata/locks and contract/installed-artifact smokes before changing the baseline.
- Documentation maintenance uses the [repository map](../development/repository-map.md)
  and compact [completed-work record](../development/completed-work.md). Superseded
  interim handoffs are removed; unique failures, FEPs/ADRs and current gates remain.
- Root checkout remains on main; another session's `fix-inferflux-codesign`
  worktree is not owned by this plan. Do not modify or remove it.
- Local default Node upgraded to 24.21.0. This is the build/tooling runtime;
  VS Code supplies its own extension-host runtime.
- Original WS-A–WS-I: **9/9 landed**. C5, G61/G62/G70 and other recorded lifecycle
  gaps remain open; this plan adds no formation passes.
- Durable local recovery copy: `var/session-closeout-2026-10-07/vscode-api-audit/`
  in the root checkout; this increment uses
  `var/foundation-upgrade-2026-10-09/` and `var/inferflux-foundation-2026-10-09/`;
  Sandhi admission/WIP evidence is in `var/sandhi-owned-admission-2026-10-09/`;
  protocol repair evidence remains in
  `var/session-closeout-2026-10-08-protocol-outcomes/` (EventBridge evidence remains
  in `var/session-closeout-2026-10-08-eventbridge-auth/`; API review evidence remains
  in `var/session-closeout-2026-10-08-api-contract-review/`; earlier client outcomes
  remain in `var/session-closeout-2026-10-08-vscode-outcomes/`; paused-chat evidence
  remains in `var/session-closeout-2026-10-08-paused-chat/`; checkpoint cancellation evidence
  remains in `var/session-closeout-2026-10-08-checkpoint-cancellation/`; storage-failure evidence
  remains in `var/session-closeout-2026-10-08-checkpoint-failures/`; pause evidence remains
  in `var/session-closeout-2026-10-08-pause-persistence/`; receipt evidence remains
  in `var/session-closeout-2026-10-08-action-receipts/`; prior stream evidence remains
  in `var/session-closeout-2026-10-08-stream-contract/`). These contain patches,
  reports, logs and hashes; the archives are ignored
  by Git and machine-local. Remote Git/PR records are the cross-machine authority.

## Product launch slice: author, run and audit (2026-10-09)

The next customer-visible milestone is a versioned two-provider workflow that a
user creates in the editor, saves/reloads, executes through either UI or CLI, and
inspects through one run/step/member/request audit trail. Foundation checks are
necessary but do not constitute this product acceptance. Existing source contains
a workflow **visualizer**, not a completed drag/drop authoring editor. Keep this
slice under VAS-14a–e, VAS-05e/f and VAS-18; do not create a second execution engine.

| Launch gate | Required evidence |
| --- | --- |
| Qualified foundations | VAS-17 settlement/recovery/lifecycle release acceptance, then reviewed C5; preserve current release wire evidence |
| Authoring | Canonical graph/formation validation, versioned save/load, stable definition hash, undo/redo, drag/drop and keyboard parity |
| Same UI/CLI execution | Identical saved definition and provider bindings use the same compiler/coordinator/API; no browser-only execution path |
| OpenAI plus ZAI | User-requested economical GPT-5-series model, selected after actual entitlement/capability qualification; ZAI via existing Sandhi credential owner; no silent provider/model fallback |
| Audited run | Distinct member sessions; definition/run/step/request correlation; usage reporting coverage; deliverables and verified terminal status; approval/cancellation/recovery errors visible |
| Headed adoption smoke | AgentBrowser sign-in, create/connect/configure/save/reload/run/inspect actions plus semantic assertions and screenshots; family theme, responsive layout, keyboard and error states |

Authentication and model qualification are separate from application SSO.
[Official OpenAI guidance](https://developers.openai.com/siwc/quickstart) describes
eligible ChatGPT-plan usage through an OSS OAuth registration. Its
[Codex app-server guidance](https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server)
requires application-owned tokens and a successful inference turn to establish
model access; catalog membership alone is insufficient. Do not export an existing
Codex session token into the browser or assume it grants arbitrary API/model access.
The [GPT-5 nano model page](https://developers.openai.com/api/docs/models/gpt-5-nano)
lists paid API pricing, no free tier and deprecation; it is an evaluation candidate,
not a new long-lived default. Qualify the user's available supported GPT-5-series
route before pinning. Do not substitute a paid key or another model silently.

AgentBrowser 1.15.2 is running in a headed-capable local deployment. Its advertised
policy still blocks WebSockets and buffers SSE; no drag primitive is advertised.
Snapshot/REST checks cannot establish drag or streaming acceptance. VAS-14c owns
those capability gaps. A visual audit may proceed while foundation work completes;
new public API implementation still follows FEP-0039's acceptance gate.

The first headed audit reproduced G75: the pinned Cytoscape script returned
HTTP 404/HTML; the graph remained blank with `cytoscape is not defined` and Loading
status. Refresh retrieved execution status but did not recover graph initialization.
The production route served a clearly labeled synthetic graph, with no model calls.
Screenshots, snapshots and network events are preserved in
`var/workflow-editor-audit-2026-10-09/`. Bundle verified assets and expose actionable
load/retry state before claiming editor readiness; theme-only changes are insufficient.

## Work ledger

Planning/evidence: VAS-00 is accepted; VAS-01 is merged.
Delivery: VAS-02, VAS-03a, VAS-11a, VAS-11b, VAS-11d, VAS-12a and VAS-12b are
merged. VAS-04a, VAS-04b and VAS-02a are merged. VAS-03b is merged evidence only; VAS-05g and VAS-15a are merged.
VAS-17 released InferFlux provenance and bounded wire acceptance now pass; durable gateway HTTP settlement/recovery and broader lifecycle qualification remain open. Parent VAS-03 and formal FEP acceptance remain open. Other implementation rows remain TODO. These counts are not
an effort-weighted completion percentage. The original formation denominator is
independent. Update this paragraph and the rows together at each checkpoint.

| ID | Milestone / bounded scope | Depends on | State | Evidence / next action |
| --- | --- | --- | --- | --- |
| VAS-00 | Audit current API shapes and reproduce boundary defects | — | ✅ ACCEPTED | Audit: real core router returns 422; compiled TS stub probes show approval loss, EOF success and false cancellation; independent review corrected binding/ownership details |
| VAS-01 | Publish FEP, plan, roadmap/gap links and reboot handoff | VAS-00 | ✅ MERGED | [#1251](https://github.com/anvai-labs/victor/pull/1251), `470c9ccdcb222dcfd8acce5b4c52fb27321ac952`; all applicable checks green; FEP was Draft at that checkpoint |
| VAS-02 | Node 24, dependency remediation and real activation/package smoke | — | ✅ MERGED | [#1251](https://github.com/anvai-labs/victor/pull/1251), `470c9ccdcb222dcfd8acce5b4c52fb27321ac952`; 999 host + 50 unit tests; audits zero; packaged VSIX activation/106 commands passed |
| VAS-02a | Patch remaining embeddings and Rust edge dependency alerts | VAS-02 | ✅ MERGED | [#1263](https://github.com/anvai-labs/victor/pull/1263), `9c281c57236aaf2cfa937e0114fadb6992531056`; clean exact-commit review and 39 successful CI checks including Vertical Py3.12; patched TLS/embeddings locks, zero RustSec/Linux lock/installed audit findings, installed-wheel HTTP/CPU/CLI smokes; main release/deployment separate |
| VAS-03 | Agree contract/FEP and consumer inventory; repair streaming request parity | VAS-01, VAS-02 | TODO | Accept FEP before new public API; preserve FEP-0037/0038 classify/model-effort contracts; extend existing contract owner, RED actual TS body vs real router, then explicit compatibility adapter |
| VAS-03a | Repair existing VS Code streaming request compatibility | VAS-01, VAS-02 | ✅ MERGED | [#1252](https://github.com/anvai-labs/victor/pull/1252), `a5fa47e444e6f973ad95149ae0677327df060dc7`; clean exact-head review, 41 applicable checks passed including Vertical Py3.12; parent/FEP acceptance remains open |
| VAS-03b | Reconcile actual consumers and prepare FEP review gates (evidence only) | VAS-03a, VAS-04a, VAS-04b | ✅ MERGED (evidence only) | [#1266](https://github.com/anvai-labs/victor/pull/1266), `35e2d0aaf`; clean exact-commit review, 40 successful checks; independently reproduced synthetic probes; formal FEP acceptance and parent VAS-03 remain open |
| VAS-04 | Typed client outcomes: paused run, terminal EOF, cancellation acknowledgement | VAS-03 | TODO | Preserve status/run/approval; reject incomplete streams; consume negative cancellation body; no new approval store |
| VAS-04a | Repair existing stream termination and cancellation acknowledgements | VAS-03a | ✅ MERGED | [#1262](https://github.com/anvai-labs/victor/pull/1262), `bdb10c083951426a14209b27c44a5073263a4a90`; clean exact-commit review, 39 successful CI checks including Vertical Py3.12. 76 Vitest, 1,000 host, 71 API/web tests; bounded transport/pending-approval acknowledgement only; no durable result/approval/continuation contract |
| VAS-04b | Preserve existing paused-chat response fields in extension consumers | VAS-03a | ✅ MERGED | [#1265](https://github.com/anvai-labs/victor/pull/1265), `b19f0f6730a04b52974fae82829c7ed0152b9d01`; clean exact-commit review and 39 applicable CI checks; 88 client/1,006 host/239 API tests, 34,030 collected; existing response metadata, shared completion guard and Composer ownership; release/continuation/C5 separate |
| VAS-05 | Shared authentication and per-resource principal authorization | VAS-03 | TODO | HTTP/events parity, no placeholder tokens, OIDC hosted posture and explicit scoped machine/local modes; reject cross-owner access |
| VAS-05a | Strict verified-principal and resource/action policy contracts | VAS-03 | TODO | [Kanidm/API-key policy design](victor-agent-service-auth-policy.md); deny overrides, credential-scope ceiling, ownership and no cross-grant widening; design only |
| VAS-05b | Kanidm access-token and scoped key authentication | VAS-05a | TODO | Dedicated verified registration/discovery; no alternate-auth fallback; expiry/revocation/rotation, group provenance; same principal contract |
| VAS-05c | Enforce authorization through every resource/transport | VAS-05b, VAS-06 | TODO | Owner/workspace/tenant checks on HTTP/SSE/WS and delegation; existing tool/approval owners remain authoritative |
| VAS-05d | Login UX, Kanidm provisioning and deployment acceptance | VAS-05c | TODO | Browser/IDE login and service-key positive/negative smokes; verified DS3 setup, roles, recovery and documented revocation bounds |
| VAS-05e | Victor/Sandhi identity propagation and delegation contract | VAS-05a | TODO | Co-design separate initiator/actor/provider identity; verify Kanidm exchange capability, scoped delegation, audience and revocation; shared workload attribution is intermediate only |
| VAS-05f | Implement immutable per-run gateway credential binding and delegation | VAS-05c, VAS-05e, VAS-06 | TODO | Two-user concurrency, no shared credential mutation, grant/model denial, rotation/revocation and actual identity/accounting joins |
| VAS-05g | Repair existing EventBridge credential and connection ownership | VAS-04a | ✅ MERGED | [#1267](https://github.com/anvai-labs/victor/pull/1267), `07b076726`; clean exact-commit review and all 39 applicable CI checks including Vertical Py3.12; 113 Vitest, 1,012 actual host, 41 API/auth/HTTP tests, 34,030 collected. Header-only key, owned rotation/cleanup and no redirect/anonymous downgrade; server-side live revocation remains G81 |
| VAS-06 | Canonical session ownership and bounded admission | VAS-05b | TODO | Port/reuse web store semantics; scope by principal/workspace, concurrent-turn policy, pre-initialization admission and restart contract |
| VAS-07 | Durable run admission, identity and status | VAS-06 | TODO | Reuse existing durable owners; same-key dedup/content conflict, reliable dispatch and honest recorded outcomes |
| VAS-08 | Canonical FastAPI composition and web compatibility entry point | VAS-03, VAS-06, VAS-07 | TODO | One route/service owner; port web wire/session behavior, preserve supported old requests and startup entry points |
| VAS-09 | Authoritative schemas and generated shared TS SDK | VAS-08 | TODO | OpenAPI/event schema generation, runtime boundary validation, deterministic generation and compatibility tests |
| VAS-10 | Unified run events, bounded parsing and replay | VAS-07, VAS-09 | TODO | UTF-8/SSE framing, IDs/cursors, explicit retention gaps, durable terminal state, bounded slow-client handling |
| VAS-11 | Exact approval and verified action reconciliation through existing framework owners | VAS-00, VAS-02 | TODO | G61/G62/G70: bind owner/payload/version/expiry; backend commit then response loss; receipt lookup or safe same-key guarantee; unknown blocks replay |
| VAS-11a | Keep durable approval admission off the async event loop | VAS-00, VAS-02 | ✅ MERGED | [#1253](https://github.com/anvai-labs/victor/pull/1253), `ec41c97b4bf6b62e31168cbba0153d850d133ae3`; clean exact-head review and 39 successful checks including Vertical Py3.12; no receipt verification or claim reopening |
| VAS-11b | Bound receipt provenance and local transaction reconciliation | VAS-11a | ✅ MERGED | [#1255](https://github.com/anvai-labs/victor/pull/1255), `76384b0e455bcaea29c4046a15ff0092cfe6df1d`; exact-head review clean, 40 applicable checks passed including Vertical Py3.12; readonly backend lookup and immutable local receipt, no tool replay/result publication/member continuation; production adapters remain VAS-11c |
| VAS-11c | Qualify production backend receipt adapters | VAS-11b | TODO | G78: no built-in tool currently qualifies; choose a concrete backend with atomic effect/receipt correlation and stable account/environment identity; no automatic retrofit to arbitrary SQL, shell, Jira or Slack; adapter and deployment failure evidence required |
| VAS-11d | Keep initial and chained approval pause persistence off the event loop | VAS-11a | ✅ MERGED | [#1257](https://github.com/anvai-labs/victor/pull/1257), `440b8f46d7400ac79d1ca86a410b881ac83b80d1`; exact-head review clean, 39 applicable CI checks passed including Vertical Py3.12; 368 local tests and 33,965 collected; preserves ownership and custom-store affinity |
| VAS-12 | Complete member continuation and cancellation lifecycle | VAS-11 | TODO | G61/G63/G70: restore completed batches, no whole-member replay; own cancellation and preserve partial effects |
| VAS-12a | Stop on member checkpoint I/O failure without unsafe fresh restart | VAS-11d | ✅ MERGED | [#1258](https://github.com/anvai-labs/victor/pull/1258), `dcba943b61b970b087bc9b9f8db7fee3e9bc70a9`; clean exact-head review, 40 successful checks including Vertical Py3.12; 551 affected tests and 33,995 collected. G79 bounded storage-failure repair; full continuation remains separate |
| VAS-12b | Own checkpoint cancellation and preserve member recovery evidence | VAS-12a | ✅ MERGED | [#1260](https://github.com/anvai-labs/victor/pull/1260), `742b1a4b025f55e00bef1d92f1c67745bc951ae5`; clean exact-head review, 40 successful checks including Vertical Py3.12; 583 affected tests, 34,027 collected, 67 independent review tests. Owned coroutine/retained workspace and pending-cancellation boundaries only; whole-member continuation remains open |
| VAS-13 | VS Code migrates to shared SDK/state; secure webview and remote workspace | VAS-09, VAS-10, VAS-11, VAS-12 | TODO | Real ephemeral server + installed VSIX smoke; auth expiry, reconnect, approval, cancellation, capabilities and workspace boundaries |
| VAS-14 | Web UI migrates to the same API and state semantics | VAS-09, VAS-10, VAS-11, VAS-12 | TODO | Browser smoke against the same server/fixtures as VS Code; embedded Chainlit retains shared service path; remote mode uses API |
| VAS-14a | Workflow/formation authoring contract and consumer inventory | VAS-03, VAS-05a | TODO | Existing visualizer is not an authoring UI; reuse canonical compiler, coordinator and formation registry; typed validation and versioned save/load |
| VAS-14b | Drag/drop plus accessible workflow and formation editor | VAS-14a, VAS-09 | TODO | Node/edge editing, all registered formation choices, bounded workflow styles, undo/redo, stable definition hash, reload and keyboard parity; no second runtime |
| VAS-14c | AgentBrowser drag and transport co-design capability | VAS-14a | TODO | Current 1.15.2 catalog has no drag primitive and policy restricts WS/SSE; add supported capability or record blocker, never silently substitute pointer/stream proof |
| VAS-14d | Headed OIDC workflow→Victor→Sandhi→provider demo | VAS-14b, VAS-14c, VAS-14e, VAS-05d, VAS-05f, VAS-11, VAS-12 | TODO | AgentBrowser snapshots + exact definition + verified execution/approval/cancel + human/actor/accounting joins; deterministic then released-provider acceptance |
| VAS-14e | Victor adapter for AnvaiOps/Sandesha/Sandhi family theme and bundled UI assets | VAS-14a, VAS-21a | TODO | Pinned semantic token source/adapters and drift guard; ink/teal, light/dark/high-contrast, responsive/keyboard/reduced motion; headed visual/functional snapshots; OSS/commercial boundary |
| VAS-15 | GraphQL/MCP/legacy adapter convergence and deprecation | VAS-08, VAS-13, VAS-14 | TODO | Inventory external consumers, policy/attribution parity; one owner, compatibility window; retire only evidenced duplicates |
| VAS-15a | Preserve outcomes through the retained Python HTTP adapter | VAS-00 | ✅ MERGED | G80; [#1268](https://github.com/anvai-labs/victor/pull/1268), `e9004a5d8`; exact-candidate independent review and all applicable CI green. 144 focused/696 affected/81 docs/44 selector/199 selected tests, 34,082 collected, 95% changed-line coverage; no POST replay, adapter removal or durability claim; direct adapter remains G82/VAS-15 |
| VAS-16 | Measured performance and broader lifecycle acceptance | VAS-12, VAS-13, VAS-14, VAS-15 | TODO | Same-workload baseline/comparison; cold/warm, 1/8/32 concurrency, memory/backpressure, deadlines and crash/recovery |
| VAS-17 | Released foundation deployment and lifecycle acceptance for C5 | VAS-02, VAS-11, VAS-12 | ACTIVE (partial foundation accepted) | InferFlux v0.5.0 released/deployed; 16 synthetic wire checks pass through direct HTTP/TLS and Sandhi 0.11.0; gateway reconciliation 4 calls/48 in/2 out. Sandhi #336 admission merged, not released. Next owner/branch above: fix ownership review findings, integrate bounded HTTP settlement/recovery, release/deploy and qualify lifecycle. Browser, cancellation, executed reuse/session behavior, remote cloud credential availability and full C5 remain open |
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

1. Preserve the merged child milestones, including VAS-03a/04a/04b and
   VAS-11a/b/d/12a/b; continue their open parents without repeating repairs.
   VAS-05g and VAS-15a are merged existing-contract repairs. Continue foundation
   qualification while the shared public API undergoes FEP review; neither waives acceptance.
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
| VAS-05g event credentials | `vscode-victor/src/test/suite/eventBridgeClient.test.ts`, controlled race tests in `src/test-unit/eventBridgeClient.unit.test.ts`, `tests/integration/integrations/api/test_fastapi_event_bridge_e2e.py` | Actual client/server auth, credential/server changes, no credential URL/log exposure, auth denial without downgrade, late callbacks and disconnect ownership; placeholder file replaced by executable cases in the discovered suite; set `VICTOR_EVENTBRIDGE_SMOKE_URL` to an isolated production router using synthetic `contract-test-key` for the additional host-to-Victor smoke |
| VAS-15a legacy outcomes | `tests/unit/protocols/test_protocol_adapters.py`, `test_protocol_interface.py` | Real adapter paused/error metadata, malformed UTF-8/JSON/SSE and EOF, bounded frames, no duplicate POST; production-route smoke before completion |
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

Historical claim/local-pass/publication checkpoints are condensed in the
[completed-work record](../development/completed-work.md#validation-provenance).
Their full original text is immutable history, not current restart instructions.

## Resume after a session change or reboot

1. In `~/code/codingagent`, fetch origin with prune; inspect status, worktrees,
   current PRs and this file from `origin/develop`. Do not
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

## Latest verified delivery

- [#1255](https://github.com/anvai-labs/victor/pull/1255) merged receipt capability
  as `76384b0e455bcaea29c4046a15ff0092cfe6df1d`; independent review clean and all
  40 applicable checks passed. Production adapter/continuation/C5 limits remain.
- [#1256](https://github.com/anvai-labs/victor/pull/1256) merged documentation
  consolidation as `b6d936d88fa99c1f467266b2fdbc62fad2b80833`; independent review
  clean and 39 applicable checks passed, including Vertical Py3.12. Inventory and
  compact history preserve unresolved tasks and immutable originals.
- VAS-11d RED: four real SQLite contention regressions failed on the baseline
  (buffered/streamed, commit/cancel), plus chained persistence ran on the caller
  thread. Four additional session/agent-switch regressions exposed publication after
  ownership changed during the new await; the post-save guard now rejects it. GREEN: 368 affected/boundary/documentation tests. Inputs and store are captured before yielding;
  save must return and runtime ownership must still match before publishing approval. No retry, fallback or tool dispatch
  is added. Existing test owners extended; similar storage/presentation cases prove
  distinct invariants, so none were removed without coverage evidence.
- Cancellation may leave one pending pause after a late worker commit; lost response
  discovery and bounded worker admission remain future VAS-07/16 requirements.
  Unknown production effects still require VAS-11c. No released service, provider,
  shared cache or C5 result changed. #1257 is now verified merged; its 39 applicable
  checks and exact-head review are clean. VAS-12a subsequently landed as #1258;
  do not repeat either increment.

- VAS-12a RED: 15 read/save regressions exposed fresh replay and swallowed storage
  failures; six lane-event cases exposed publication before pause acknowledgement.
  Independent review found forced cleanup deleting completed work; two real git
  worktree cases reproduced that loss. Final GREEN: 551 team/formation tests,
  33,995 tests collected; Black/Ruff/MyPy (2,000 source files), FEP validation,
  strict MkDocs, internal links and repository hygiene pass. Independent review
  passed 64 tests after the workspace fix. Extended existing checkpoint, concurrent, hierarchical, iterative and
  isolation owners; no equivalent regression existed to remove. Checkpointer-free
  defaults remain covered. Commit-before-lost-ack is memory-store fault injection,
  not backend acceptance. G79 records external cancellation, in-flight receipts and
  safe whole-member continuation limits. #1258 merged as
  `dcba943b61b970b087bc9b9f8db7fee3e9bc70a9` after 40 successful CI checks and clean review of `f3b3312329e95b6e75c5f28a24c90b90a46fd567`.
  Trivy was neutral; PR-only documentation deployment and helper publication were
  non-applicable skips. No release, provider runtime or C5 result changed.

- VAS-12b RED: 16 regressions reproduced deleted worktrees and cancellation retry/join
  defects; ten late-save cases and four suppressed load/member/store-error cases
  reproduced continuation after a pending stop. Existing three test owners extended;
  workspace cases parameterize the prior storage-failure test instead of duplicating
  the git fixture or creating another suite. Independent review requested the load
  boundary guard as well as saves. Owned coroutine joins and best-effort cancellation
  telemetry do not establish durable status, remote cancellation, bounded shutdown
  for callbacks that never finish or process-crash recovery.

- VAS-12b final local validation: 583 affected tests, 34,027 collected; Black checks
  3,917 files, Ruff passes, MyPy checks 2,000 source files, strict MkDocs/internal
  links/hygiene/FEP validation pass. Independent runtime/test review is clean
  (67 tests). Review also corrected the tracker summary to match its row. No
  provider runtime, release or C5 acceptance changed.

- VAS-12b merged in [#1260](https://github.com/anvai-labs/victor/pull/1260) as
  `742b1a4b025f55e00bef1d92f1c67745bc951ae5` after 40 successful checks including Vertical Py3.12 and clean
  review of `741c61fc3c98122c05e843dcad87995a905388b2`. Trivy was neutral;
  documentation deployment/helper publication were non-applicable skips. Original
  RED and final GREEN evidence remain in the declared archive. Release and C5
  acceptance are separate; continue VAS-11c/12 without repeating VAS-12a/12b.
