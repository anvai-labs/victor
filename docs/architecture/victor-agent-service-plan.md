---
plan_id: VAS-2026-10
status: active
updated: 2026-10-07
baseline_develop: b8e195cc68e009c68b2cfc9b96ac8cf5b74bef84
next_task: VAS-02
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

- Fetched develop: `b8e195cc68e009c68b2cfc9b96ac8cf5b74bef84` (#1250 merged).
- Worktree: `/private/tmp/victor-vscode-dependencies`; branch
  `fix/vscode-security-locks`. Source publication status comes from that branch's
  PR; see the append-only checkpoint/evidence records below.
- Root checkout remains on main; another session's `fix-inferflux-codesign`
  worktree is not owned by this plan. Do not modify or remove it.
- Local default Node upgraded to 24.21.0. This is the build/tooling runtime;
  VS Code supplies its own extension-host runtime.
- Original WS-A–WS-I: **9/9 landed**. C5, G61/G62/G70 and other recorded lifecycle
  gaps remain open; this plan adds no formation passes.
- Durable local recovery copy: `var/session-closeout-2026-10-07/vscode-api-audit/`
  in the root checkout. It contains a patch, reports, logs and hashes; it is ignored
  by Git and machine-local. Remote Git/PR records are the cross-machine authority.

## Work ledger

Planning/evidence: VAS-00 is accepted; VAS-01 is locally authored pending publication.
Delivery: VAS-02 has local passes; VAS-03–VAS-20 remain TODO. These counts are not
an effort-weighted completion percentage. The original formation denominator is
independent. Update this paragraph and the rows together at each checkpoint.

| ID | Milestone / bounded scope | Depends on | State | Evidence / next action |
| --- | --- | --- | --- | --- |
| VAS-00 | Audit current API shapes and reproduce boundary defects | — | ✅ ACCEPTED | Audit: real core router returns 422; compiled TS stub probes show approval loss, EOF success and false cancellation; independent review corrected binding/ownership details |
| VAS-01 | Publish FEP, plan, roadmap/gap links and reboot handoff | VAS-00 | LOCAL_PASS | This document + Draft FEP-0039; verify docs, review and publish with current coherent PR; update merge proof at next checkpoint |
| VAS-02 | Node 24, dependency remediation and real activation/package smoke | — | LOCAL_PASS | 999 host + 50 unit tests; audits zero; production VSIX built; finish exact-candidate review/CI/merge |
| VAS-03 | Agree contract/FEP and consumer inventory; repair streaming request parity | VAS-01, VAS-02 | TODO | Accept FEP before new public API; preserve FEP-0037/0038 classify/model-effort contracts; extend existing contract owner, RED actual TS body vs real router, then explicit compatibility adapter |
| VAS-04 | Typed client outcomes: paused run, terminal EOF, cancellation acknowledgement | VAS-03 | TODO | Preserve status/run/approval; reject incomplete streams; consume negative cancellation body; no new approval store |
| VAS-05 | Shared authentication and per-resource principal authorization | VAS-03 | TODO | HTTP/events parity, no placeholder tokens, OIDC hosted posture and explicit scoped machine/local modes; reject cross-owner access |
| VAS-06 | Canonical session ownership and bounded admission | VAS-05 | TODO | Port/reuse web store semantics; scope by principal/workspace, concurrent-turn policy, pre-initialization admission and restart contract |
| VAS-07 | Durable run admission, identity and status | VAS-06 | TODO | Reuse existing durable owners; same-key dedup/content conflict, reliable dispatch and honest recorded outcomes |
| VAS-08 | Canonical FastAPI composition and web compatibility entry point | VAS-03, VAS-06, VAS-07 | TODO | One route/service owner; port web wire/session behavior, preserve supported old requests and startup entry points |
| VAS-09 | Authoritative schemas and generated shared TS SDK | VAS-08 | TODO | OpenAPI/event schema generation, runtime boundary validation, deterministic generation and compatibility tests |
| VAS-10 | Unified run events, bounded parsing and replay | VAS-07, VAS-09 | TODO | UTF-8/SSE framing, IDs/cursors, explicit retention gaps, durable terminal state, bounded slow-client handling |
| VAS-11 | Exact approval and verified action reconciliation through existing framework owners | VAS-00, VAS-02 | TODO | G61/G62/G70: bind owner/payload/version/expiry; backend commit then response loss; receipt lookup or safe same-key guarantee; unknown blocks replay |
| VAS-12 | Complete member continuation and cancellation lifecycle | VAS-11 | TODO | G61/G63/G70: restore completed batches, no whole-member replay; own cancellation and preserve partial effects |
| VAS-13 | VS Code migrates to shared SDK/state; secure webview and remote workspace | VAS-09, VAS-10, VAS-11, VAS-12 | TODO | Real ephemeral server + installed VSIX smoke; auth expiry, reconnect, approval, cancellation, capabilities and workspace boundaries |
| VAS-14 | Web UI migrates to the same API and state semantics | VAS-09, VAS-10, VAS-11, VAS-12 | TODO | Browser smoke against the same server/fixtures as VS Code; embedded Chainlit retains shared service path; remote mode uses API |
| VAS-15 | GraphQL/MCP/legacy adapter convergence and deprecation | VAS-08, VAS-13, VAS-14 | TODO | Inventory external consumers, policy/attribution parity; one owner, compatibility window; retire only evidenced duplicates |
| VAS-16 | Measured performance and broader lifecycle acceptance | VAS-12, VAS-13, VAS-14, VAS-15 | TODO | Same-workload baseline/comparison; cold/warm, 1/8/32 concurrency, memory/backpressure, deadlines and crash/recovery |
| VAS-17 | Released foundation deployment and lifecycle acceptance for C5 | VAS-02, VAS-11, VAS-12 | TODO | Full green promotion/release CI, artifact/binary/source/config IDs and rollback; verify released Sandhi/InferFlux readiness without clearing cache |
| VAS-18 | Full mixed-team C5 verdict | VAS-17 | TODO | Six-Qwen/one-ZAI harness; unchanged deliverable/pytest/session/accounting gates; reviewed verdict on InferFlux #184 |
| VAS-19 | Matched formation cohorts and remaining semantics | VAS-18 | TODO | Preserve ZAI reference, simpler explicitly labelled local tasks, all 12 formations + 3 policies; G72 opt-in strict hierarchy separately |
| VAS-20 | Shared API/UI main promotion and release | VAS-16, VAS-17 | TODO | Full green promotion/release CI; publish server/SDK/VSIX/docs and compatibility matrix; verify installed artifacts |

**Two delivery paths:** VAS-11 → VAS-12 → VAS-17 → VAS-18 preserves the original
framework/C5 mission and does not wait for GraphQL deprecation or either UI
migration. VAS-03–VAS-10/VAS-13–VAS-16 → VAS-20 delivers the shared API platform.
VAS-11 reuses existing framework identity/pause/action owners and does not wait
for new HTTP run endpoints; VAS-05/07 later expose those same contracts. The C5
harness calls the framework/providers directly. New API event work is not a
technical prerequisite for its verdict. Parallel sessions can claim independent
ready rows; the default remains the fewest worktrees and one accountable owner.

Each row is a milestone, not a requirement to combine all its work into one PR.
If scope exceeds a reviewable increment, add child IDs (for example VAS-05a/b)
with explicit acceptance before coding; the parent stays open until all children
pass. Avoid cross-row feature bundles. Independent work may overlap only with
separate file ownership and recorded coordination, never shared mutable branches.

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
