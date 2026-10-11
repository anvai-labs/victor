---
plan_id: VAS-2026-10
status: active
updated: 2026-10-10
baseline_develop: decfc3ad5d8200a937e7ab156c5493d5bf74c464
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
- Required Sandhi Python binding, optional separately deployed proxy. The embedded
  library connects directly to InferFlux, ZAI and OpenAI/Codex subscription routes;
  the network gateway is a separately selected deployment mode, not a prerequisite.
  No failure-driven bypass, second transport or credential transfer. InferFlux
  owns inference/GPU placement; account/model entitlement is qualified separately.
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

- VAS-17 release alignment merged in [#1276](https://github.com/anvai-labs/victor/pull/1276)
  as `9ff5b9862a17950f70d9d3c850eee03dcca075af`, after clean independent
  review and all 39 applicable CI checks passed (including Vertical Py3.12).
  Deployment evidence is owned by this co-design session on
  `docs/gateway-012-deployment`, based on that merge.
  Sandhi's version preparation [#346](https://github.com/anvai-labs/sandhi/pull/346)
  merged with 13 applicable checks green; promotion
  [#347](https://github.com/anvai-labs/sandhi/pull/347) merged as
  `fc77b537eabbefa0f179769c8953827d776017d1` after clean cumulative review,
  promotion CI and exact-develop push CI. Main and develop content matches;
  exact-main push CI also passed. [Sandhi 0.12.0](https://github.com/anvai-labs/sandhi/releases/tag/v0.12.0)
  published all required targets in run `38025435213`; independent verification
  confirmed four Python wheel platforms, four crates, three npm packages and both
  binary archives. Victor's metadata now pins the published 0.12.0 wheel.
  Homebrew [#99](https://github.com/anvai-labs/homebrew-tap/pull/99) merged after
  all four checks passed; local upgrade and both 0.12.0 binary version tests pass.
  Local validation covers 841 distinct existing tests: 824 passed in the restricted
  run, with 17 socket-denied cases passing in the 37-test HTTP/TLS rerun with
  loopback access. Full collection found 34,253 tests; strict documentation,
  repository hygiene and installed CLI/import checks pass. The isolated copied
  environment uses the published Sandhi wheel and current declared aiohttp/cvss
  requirements. Its inherited optional-package conflicts keep broad `pip check`
  non-green; no shared environment was modified and this is not a clean full-extras
  installation claim. No new tests or duplicate test copies were added.
  Deployment recheck (2026-10-10): the Mac managed gateway now executes 0.12.0;
  the remote managed gateway also executes 0.12.0 after the host owner’s sudo cutover. The current
  validation profile targets `https://sso.singh.local:8444`, forwarding to Mac
  loopback `18789` (OIDC), with ZAI `glm-5.3` and both Qwen models. InferFlux uses
  the existing Mac `18081` SSH tunnel to aiserver1 loopback `8080`. The separate
  remote Sandhi TLS service on `18788` is not a prerequisite for this path.
  Mac activation used graceful SIGTERM/launchd restart to PID 88809, verified the
  exact 0.12.0 executable and SHA-256
  `b138480cd96f4c5a1d830a1f9bf1615ba5dd7519362531323d1c37619ed00e34`, and retained
  all 1,068 existing usage events with database integrity `ok`. OIDC and route
  configuration are byte-identical. Five copied-state authorization checks passed.
  Three new HTTPS/OIDC provider probes passed: ZAI `glm-5.3`, Qwen3 and Qwen2.5.
  Wire, SQLite, per-run accounting and dashboard delta reconcile exactly: **3 calls,
  49 input tokens, 5 output tokens, explicit zero cache reporting 3/3**. Distinct
  sessions/run IDs and nonempty request IDs join to the configured member subject.
  These are new deployment probes, not actual-member/C5 evidence; reported zero
  cache does not establish zero executed reuse.
  Remote official 0.12.0 CLI links are updated. Its new service binary SHA-256 is
  `47b6b9d5f7d040917645b3af3d2b6b1acb824a752d301a09228270beac589786`; copied-state
  TLS, anonymous-admin denial and authenticated version/key/usage/dashboard reads
  passed. The host owner installed the validated unit; live PID 2231370 matches
  the official binary hash. The unit changes only `ExecStart`, with rollback unit,
  config and SQLite backup preserved under the host’s
  `.local/state/sandhi-upgrade-012-20261010/`. Live trusted TLS, anonymous admin
  denial, authenticated reads and client-credential exchange pass. Qwen3 chat and
  two finite 384-dimensional BGE embeddings reconcile wire/SQLite/run/dashboard
  totals: **2 calls, 22 input tokens, 1 output token**. Chat explicitly reports zero
  cache; embeddings omit cache reporting (absent, not explicit zero). The remote
  service retains its explicit token-compatibility mode; this is not OIDC evidence.
  Database integrity is `ok`; all 235 prior usage rows retain their checked request,
  session, step, model and token fields, plus the two new rows. The symlink-skipping
  warnings concern the already-unavailable OpenAI/Codex and ZAI credential entries;
  no guard was weakened and no credentials were transferred. Mac ZAI remains the
  qualified cloud route. The isolated tracked OIDC qualification is recorded below; next are managed tracked deployment and streaming lifecycle acceptance,
  then the full mixed-team C5 run against accepted released runtimes. Codex is absent from the Mac validation profile and requires
  authentication/model qualification. Sanitized reports and credential-free scripts
  are retained in `var/gateway-upgrade-012-20261010/`; private backups stay local.
  This release/adoption increment
  does not close tracked TLS/OIDC, streaming lifecycle or mixed-team C5 acceptance.
- Previous checkpoint baseline: `6762481fef7d77ac1978f43d36f8ecb0e3dce317` (#1274 merged).
- VAS-14e's bounded existing-visualizer repair merged in
  [#1271](https://github.com/anvai-labs/victor/pull/1271): packaged assets,
  safe inspection, error/retry states and light/dark/system appearance. Final
  candidate `7786dfa2947d6bdd6093658e3b4923f41a0ec0f3` passed independent
  review and 40 successful checks (plus neutral Trivy and a nonapplicable
  documentation deployment skip); merge tree matched. VAS-14e remains open for
  full editor and cross-product adoption; the evidence is synthetic, not OIDC,
  actual execution, drag/drop or C5 acceptance.
- User expanded reuse to ProximaDB and AnvaiOps's existing workspace editor and
  notebook. The [source inventory and staged reuse decision](victor-agent-service-technology.md#reuse-with-proximadb-and-the-anvaiops-workspace)
  pins fetched ProximaDB `9d6acc7f4f08abec57b4c54b3a0bc8d5ee6ddb6c` and
  AnvaiOps `91d731e8203e6c0b8393c8176031aae79d3d0557`. VAS-21d records
  source inspection only; dependency/runtime qualification, extraction and adoption
  remain open. No sibling code, deployment or unrelated dirty file was changed.
  Commercial workspace specs now drive the generic requirement/contract/adapter acceptance map; TypeScript/React is the direction for new rich components, subject to compatible build/peer qualification. Keep VAS-17 as the next implementation task.
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
- Sandhi [#337](https://github.com/anvai-labs/sandhi/pull/337) merged buffered
  usage qualification as `98aa830388f463d16175b8dac6cb6d0d45515985`: missing or
  malformed usage is distinct from explicit zero, without provider replay.
- Sandhi [#338](https://github.com/anvai-labs/sandhi/pull/338) merged as
  `7bd259a46b4c42891d418ee0b5471dcdc2238caa` after clean exact-commit independent
  review and all 13 applicable CI checks passed. The merge tree matches tested
  candidate `647a43ea1bf705d0d7e730d8f6598d163b4a78fa`. It connects opt-in buffered HTTP
  admission, dispatch, terminal evidence and settlement through one retained owner
  and the original ledger. Gateway correlation is persisted atomically with intent;
  disconnect does not abandon ownership. Bounded recovery advances across removed
  and unbudgeted scopes, never replaying inference or telemetry. Shutdown checks
  durable uncertainty even when no in-memory workers remain.
  Local validation: 861 Rust tests (six ignored), 95 Python and 56 Node binding
  tests, ten actual-process HTTP/TLS/SIGTERM smoke tests, 90.27% line coverage,
  strict lint and clean independent review. Review findings for terminal evidence
  loss on telemetry panic and skipped recovery slot zero have RED/GREEN regressions.
  Two earlier CI attempts failed on Docker Hub's anonymous pull quota, before the
  container-resource test. The final CI-only repair uses Google's mirror with the
  same Dockerfile-derived image digest; all container and optional-feature gates
  passed afterward. Failed evidence remains preserved; no gate was waived.
  This supersedes the earlier helper-only WIP; the release/adoption checkpoint
  above owns current publication status. Source acceptance is not deployment.
- Sandhi [#339](https://github.com/anvai-labs/sandhi/pull/339) merged as
  `fb8c02fb3a8736d6966e4658ddbfb76d1ea3ecf9` after clean independent review and
  all six applicable CI checks, including Linux SDK conformance, passed. The merge
  tree matches candidate `691d9e0d434caa8c0ea8a3d3454111a8e9a5044d`. Two actual-process
  crash/restart cases now pass, using the same copied and hashed binary:
  unknown post-origin dispatch retains liability and incomplete shutdown; persisted
  terminal usage recovers one receipt after a synthetic receipt-write failure.
  A second restart preserves receipt identity, time and spend, with no provider or
  usage-event replay. Existing fixtures are extended; no production hooks or code
  changes are added. Recovery/shutdown suites pass 17 tests; other fixture consumers
  pass 27, including headless AgentBrowser restored-dashboard and negative oracles.
  Two temporary restart-only legacy-mode controls fail at the intended assertions,
  then restored tracked-mode tests pass. This is HTTP/token-mode synthetic
  acceptance on Mac and Linux, not tracked TLS/OIDC, streaming, a released deployment or C5.
- Sandhi [#340](https://github.com/anvai-labs/sandhi/pull/340) merged as
  `f7b144b7b85fcb66f84e856d56dbcb71ca532cf6` after all 11 applicable CI checks passed;
  its tree matches reviewed candidate `f4a4d49548445e1d08066f0ba940ee2bbc45ed20`.
  It qualifies terminal publication under SQLite contention and process death. Two synthetic HTTP/token-mode
  cases gate the origin response, hold an external write lock, witness qualified
  gateway usage (7 fresh-input, 4 cache-read, 3 output; 14 billable), and wait for
  the original accounting worker to end before releasing the lock. Survival retries
  retained usage into one receipt; death before persistence retains unknown liability
  and incomplete shutdown. Both preserve correlation and avoid inference replay.
  Optional events may be absent but cannot duplicate or change attribution/counts.
  All 46 affected process/fixture-consumer tests, 12 metrics tests and repeated
  focused cases pass. Missing-usage and disabled-retry negative controls fail at
  their intended assertions; both mutations were restored. Independent review is clean.
  Existing fixtures/polling are reused, and metrics documentation now distinguishes
  observed usage from durable spend. Aggregate metrics prove association only in
  this isolated single-request fixture. Shared runtimes remain unchanged.
  The first CI run failed an unchanged optional-feature text-policy test (503 versus
  expected 403). Its exact 14-test module passed locally, then one independently
  justified failed-job rerun passed. The cause remains unproven; original failure
  evidence is preserved. No deadline or assertion was weakened.
- Sandhi [#350](https://github.com/anvai-labs/sandhi/pull/350) merged as
  `c66f2fd5f92b13afc8dc106797cbfd4e95085f08` after clean independent review and
  all seven applicable CI checks, including Linux SDK conformance, passed. Its tree
  matches reviewed candidate `24fbfdcfe51fe38fbf1942215e97b1c3cc6e8393`.
  The existing tracked buffered shutdown test now verifies CA-backed TLS and HTTP,
  including settled/unresolved usage and exit 0/124. The existing SSO browser role
  matrix runs in legacy and tracked modes. A scoped OIDC bearer call joins the
  returned request ID, intent, session/run/step-attributed event and receipt, with
  exactly 14 charged tokens against its bound Block budget. Invalid bearer,
  disallowed model and subject spoof leave origin requests, durable record counts
  and dashboard budget spend unchanged. No production behavior changed.
  All 149 unique affected tests are verified: 144 passed initially; five existing
  broker fixture setup errors under macOS's long temporary socket paths passed
  with a short temporary directory. Original failure JUnit is preserved. The final
  three denial cases also pass with the review's stronger post-denial budget check;
  314 SDK tests collect, and strict docs/rustfmt pass. Test owners were reused;
  no redundant suite was removed at the cost of distinct failure coverage.
  The fixture authority maps inference tokens only to admin; signed browser roles
  are tested separately. HTTP clients and the gateway-to-IdP path verify CA trust,
  while browser contexts ignore the disposable fixture certificate. This is
  synthetic source acceptance, not deployed tracked OIDC, OIDC crash/restart
  composition, streaming lifecycle or C5. Shared runtimes and cache are unchanged.
  Evidence: `var/sandhi-tracked-tls-oidc-2026-10-10/`.
- Sandhi [#351](https://github.com/anvai-labs/sandhi/pull/351) merged as
  `a37ad3315469d266bcfa8d0b829c10d4fbaa5b86` after clean independent review and
  all seven applicable CI checks, including Linux SDK conformance. It extends the two
  existing tracked SIGKILL tests to CA-verified TLS/OIDC with separate admin/member
  identities. Unknown liability keeps its original bound scope and denies new work
  with budget-exhausted 429. Persisted final usage recovers one original 14-token
  receipt after inference-grant revocation; new inference returns 403 without
  changing spend, events, receipts or the single origin request. A second restart
  preserves receipt identity/time. Token-mode cases and existing fixture consumers
  retain their defaults; no duplicate parser/auth/transaction suite is added.
  All 63 recovery/dashboard/shutdown tests and the AgentBrowser recovery smoke
  pass; 316 SDK tests collect. Initial RED checks reject the old HTTP fixture;
  a negative control retaining the grant fails 200 versus required 403. Final four
  SIGKILL variants, strict MkDocs and rustfmt pass. Independent review is clean for
  `845887b28b949e5e9e2e308b46384967ffa001ac`. Source fault injection remains synthetic.
- New isolated released-binary acceptance uses official macOS Sandhi 0.12.0
  (`b138480cd96f4c5a1d830a1f9bf1615ba5dd7519362531323d1c37619ed00e34`),
  the deployed AnvaiIdentity policy and existing local credential references.
  A private loopback TLS tracked candidate starts from an online SQLite backup;
  the managed gateway stays in default accounting mode. Both InferFlux Qwen models
  and ZAI `glm-5.3` return HTTP 200/nonempty content. Request/member/group/model and
  distinct session/run joins reconcile wire, SQLite, receipts, run API and dashboard:
  **3 calls, 46 fresh input + 5 output = 51 charged tokens**, explicit zero cache
  reporting 3/3. Candidate shutdown exits 0, database integrity passes and all
  1,071 historical events/configuration remain unchanged. Gateway/client deadlines
  stay 120/125 seconds; no inference retries or shared cache clearing.
  This is new provider smoke, not actual-member evidence, live crash recovery,
  managed cutover or C5. Two helper-import failures occurred before server launch
  or calls and were fixed before the successful run. Evidence and private backups:
  `var/sandhi-oidc-recovery-2026-10-10/`. The source recovery and isolated released
  integration gates are now distinct accepted evidence; managed tracked deployment
  and broader lifecycle remain open.
  Victor tracker validation: repository hygiene, strict MkDocs/internal links and
  34,253-test collection pass. The first collection had six import errors because
  a sibling editable install exported a conflicting `scripts` package. A private
  virtualenv copy excluding that editable path collects cleanly; shared environment
  and repository code remain unchanged. Both failure and successful rerun are retained.
- Sandhi [#352](https://github.com/anvai-labs/sandhi/pull/352) merged as
  `eeb61d270375dfa9e6743e3b5d587a5b6d9db751` after clean independent review of
  `fbb73f26f80c394f59c479637ac924cc8a9dc868` and all 13 applicable CI checks,
  including Rust feature tests, coverage, bindings/ARM64 wheel and SDK conformance.
  This completes the strict OpenAI Chat stream-usage core prerequisite. Its API qualifies
  request-total usage only from complete decoded empty-choices Chat chunk events,
  sharing the buffered counter validator and existing parser. Missing/null usage
  remains unmeasured; explicit zero is valid; malformed/nonfinal usage is rejected.
  A valid measurement precedes `[DONE]`; later delivery failure cannot erase it,
  and EOF/finish markers alone cannot create it. Existing numeric test matrices
  cover both paths; one new envelope test avoids duplicate parser suites.
  Local validation: 862 Rust tests pass (six existing ignored), 18 SDK regressions,
  strict MkDocs, rustfmt and workspace clippy pass; both numeric/envelope guard-bypass
  negative controls fail as intended and are reverted. Independent exact-commit
  review is clean. No transport activation, live inference, deployment or C5 evidence
  is added. Logs and restart state: `var/sandhi-stream-qualification-2026-10-10/`.
- Sandhi [#353](https://github.com/anvai-labs/sandhi/pull/353) merged as
  `bae2226800db20223c01dfe526a6b8d31579cbe7` after clean independent review of
  `0229b393407c3869d63065e7ca2d209003f250e7` and all 13 applicable CI checks.
  The additive provider observer uses the existing splitter's opt-in SSE mode and
  core qualifier for complete bounded events. CR/LF/CRLF, initial BOM, multiline
  data, complete/pending line limits and aggregate event limits are covered.
  The first usage observation is immutable; identical repeats are idempotent and
  conflicts or later failure preserve evidence while exposing a sticky error.
  Usage, protocol completion and failure remain independent; EOF cannot promote
  partial events and `[DONE]` alone cannot manufacture measurement.
  Five new observer tests extend existing splitter boundary/linearity matrices;
  no duplicate numeric parser/storage suite. All 867 Rust tests pass (six existing
  ignored), 18 SDK regressions, strict MkDocs, rustfmt and clippy pass. Four reverted
  guard-bypass controls fail as intended. This is an unwired source prerequisite,
  not request binding, durable settlement, deployed streaming or C5 acceptance.
  Read-only checks find Mac Sandhi on `127.0.0.1:18789` (the installed binary reports
  0.12.0), aiserver1 Sandhi on `18788` and InferFlux on loopback `8080`. These checks establish process/listener
  liveness only; no provider calls or runtime changes. Evidence and restart state:
  `var/sandhi-stream-observation-2026-10-10/`.
- Direct-library acceptance clarification (2026-10-10): the user confirmed the
  embedded Sandhi runtime stays required; only the network hop is optional.
  The official installed Sandhi 0.12.0 binding passes 55 fixture checks: existing direct/
  gateway construction now includes InferFlux and ZAI; existing real HTTP parity,
  tool-call, error, timeout and no-double-request tests cover both providers.
  Two new cases exercise Victor's Codex OAuth selection through the real binding
  and local Responses server for complete/stream calls, with current bearer and
  account headers, required instructions and exactly one POST. Credential acquisition
  alone is stubbed; actual token refresh is not tested and no real token is read or
  copied. All 197 affected provider/auth/
  integration tests pass. The initial copied venv contained 0.11.0; only the private
  worktree venv was upgraded to the pinned official 0.12.0 wheel and the tests rerun.
  The initial missing-instruction fixture failure is now an explicit no-dispatch
  assertion. A duplicate one-POST test was removed because the existing completion
  parity case asserts the same invariant for both handles and all three providers.
  This does not establish
  live cloud access, model entitlement, released direct cancellation or C5 acceptance.
  No production provider/registry change or replacement transport is needed.
- Sandhi [#354](https://github.com/anvai-labs/sandhi/pull/354) merged as
  `872fa76b057c0139cd01a992cbcbe01281d665b3` after clean independent review of
  `0b19bf4fabe9d0bb0600afa12457adb1f85b917c` and all 13 applicable CI checks.
  One opt-in raw OpenAI Chat stream now owns the complete-event observer and a
  copied snapshot surviving drop. Usage, protocol DONE and delivery outcome are
  separate; post-DONE errors remain observable. Accepted usage remains Final on
  cancellation/timeout/transport failure, while malformed/conflicting reports stay
  explicit for reconciliation. Shared setup/send/idle, canonical attempt/session/
  request correlation and unchanged response bytes preserve existing defaults.
  Local validation: 871 Rust tests pass (six existing ignored), 261 provider tests
  after the final resource-release assertion, 18 SDK/shutdown tests, strict docs,
  rustfmt and clippy. Two lifecycle matrices and existing HTTP/deadline fixtures
  cover drop-before-poll, termination, resource release and diagnostic-channel loss;
  parser/numeric matrices are reused. No tracked HTTP activation, durable streaming
  publication, deployment or C5 pass is claimed.
  Evidence and restart state: `var/sandhi-stream-binding-2026-10-10/`.
- VAS-17 tracked streaming bridge (2026-10-10): Sandhi [#355](https://github.com/anvai-labs/sandhi/pull/355)
  merged as `4a6939c59bc06ebb875619d44bc84255d2ee310f` after all 13 applicable CI checks
  passed. It connects the
  existing correlated admission, raw response snapshot, bounded body producer and
  durable settlement jobs. Only retry-free, bounded transparent OpenAI Chat streams
  are admitted; defaults and the shared Chat/Responses adapter ownership stay intact.
  Qualified counts survive disconnect/body deadline/shutdown; conflicting or invalid
  evidence retains first counts as Partial, missing usage stays Unavailable, and
  neither can settle. Normal HTTP EOF waits for a receipt. Failed terminal publication
  retains accounting-only recovery without another model call. Shutdown reserves its
  existing accounting wait inside the original grace; expired grace retains unknown
  liability. Metrics/events conserve observed totals without claiming committed spend.
  Existing refusal and sink-panic matrices were extended; the numeric/framing/storage
  suites remain their own test owners. Review found missing-usage success labels and
  partial-total disagreement; both are fixed with regression assertions. The metric
  negative control failed 0 versus 12 before the fix.
  Validation: 875 Rust workspace tests pass (six existing ignored), 11 final owned-HTTP
  tests, 18 SDK/shutdown and 11 existing crash/restart regressions, rustfmt,
  workspace clippy and strict MkDocs.
  Independent source review is clean for `9f7b90cd265f1ee4c7f6128b838f0983a0d01b42`;
  The merged tree matches the reviewed candidate. Exact source/review/CI evidence lives in
  `var/sandhi-owned-stream-2026-10-10/`.
- VAS-17 standalone streaming acceptance (2026-10-10): Sandhi [#356](https://github.com/anvai-labs/sandhi/pull/356)
  merged as `da4002a389d28bdd5c298e60e57ca15c309e7ed3` after all seven applicable CI checks
  passed. Existing
  shutdown/recovery fixtures now cover buffered/streaming requests with token and
  CA-verified TLS/OIDC authentication. SIGKILL before terminal persistence retains
  unknown liability; persisted final usage recovers one receipt after restart,
  including after inference-grant revocation. SQLite contention preserves retained
  recovery while alive and unknown liability after process death. The OIDC streaming
  completion/disconnect/SIGTERM matrix verifies final usage, termination outcome,
  request/session/run/step correlation and 7 fresh + 4 cache-read + 3 output tokens
  in one 14-token receipt, with one origin request. Legacy fixture defaults remain
  unchanged; numeric/framing and storage suites retain their existing ownership.
  Validation: 40 affected process tests and three strengthened lifecycle cases pass;
  rustfmt, Python compilation, strict MkDocs and independent exact-commit review
  are clean for `db53875448785fd3cf0e3498936fcea8549c5450`. The unconfigured-policy
  negative control rejects streaming with HTTP 400. This is synthetic built-source
  acceptance, not packaged release, deployed lifecycle, tokenizer/cache/session or
  C5 evidence. Logs, review binding and restart state:
  `var/sandhi-stream-acceptance-2026-10-10/`.
- VAS-17 release and consumer qualification (2026-10-10):
  [Sandhi 0.13.0](https://github.com/anvai-labs/sandhi/releases/tag/v0.13.0)
  published from reviewed main `8aea5492fa4a5e5e152fa4b14125082f917be7b2` after
  preparation [#357](https://github.com/anvai-labs/sandhi/pull/357), promotion
  [#358](https://github.com/anvai-labs/sandhi/pull/358) and exact-main push CI passed.
  All GitHub binaries, four wheel platforms, four crates and three npm packages
  passed independent publication verification. The promotion's first SDK attempt
  encountered a fixture port collision (330 passed, three skipped, one setup error);
  the targeted retry passed. Original failure evidence is retained.
  Victor's project requirement and all three deployment locks now pin 0.13.0.
  The official macOS wheel is byte-identical to the PyPI download; 1,712 affected
  provider, metering, dependency-lock and HTTP parity tests pass (five existing
  skips), and all 34,330 tests collect. Strict docs, links and hygiene pass.
  The copied development environment's ProximaDB/protobuf mismatch was repaired
  only in the private test environment; no unrelated dependency pin changed.
  Homebrew's local 0.13.0 installation passes both executable versions, HTTP
  identity, health/readiness and graceful shutdown; the formula update has separate
  review/CI. At release qualification, both shared gateways remained on 0.12.0;
  the subsequent remote upgrade is recorded below.
  The embedded binding retains its facade and does not acquire proxy ledger
  enforcement. Evidence and restart state: `var/sandhi-release-013-2026-10-10/`.
- Next VAS-17 owner: this co-design session. Consumer/formula gates completed
  in [Victor #1285](https://github.com/anvai-labs/victor/pull/1285) and
  [Tap #101](https://github.com/anvai-labs/homebrew-tap/pull/101). Qualify the released
  runtime with preserved state and deployed
  TLS/OIDC/provider routes. Record rollback identities before managed rollout,
  broader lifecycle acceptance, actual-member C5 and formation cohorts. Preserve
  failed evidence, shared caches and credentials. Source and package smoke do not
  prove browser SSO, origin cancellation, tokenizer units or executed cache reuse.
  **Release boundary:** the scoped streaming source milestone and all-target
  publication are complete; managed tracked deployment and C5 remain open.
  Amendments, export and fleet work remain separate capabilities.
  The first modes are single-file, retry-free buffered or bounded transparent
  streaming OpenAI-compatible transport with explicit output bounds and Block policy; logical dedup, amendments, atomic
  telemetry export and tracked threshold alerts remain separate open capabilities.
  FEP-0039 remains in Review; VAS-11c still needs a qualified production receipt
  operation. Do not restart merged repairs or create a new public Victor API.
- VAS-17 preserved-state deployment preflight (2026-10-11): the official Linux
  Sandhi 0.13.0 candidate passes CA-verified TLS readiness, denied anonymous admin
  access, authorized admin/dashboard access and clean shutdown against a SQLite
  online backup. Real Qwen3 chat and BGE embeddings reconcile two calls, 22 input
  tokens and one output token across wire usage, SQLite, per-run totals and
  dashboard totals. Chat reports cache explicitly; embeddings omit it. These are
  new deployment probes, not actual-member/C5 evidence. Default accounting mode
  does not establish the tracked response-header/intent/receipt correlation join.
  Original passing reports were rechecked without repeating inference.
  The host owner completed remote activation: aiserver1 now runs the official
  0.13.0 binary (SHA-256 `e6018a8f22b999bfd37f35fb936f4a24fc5fcbca5b52880ad018938565beb624`).
  Post-cutover TLS/auth, Qwen3/BGE probes and conservation pass again (two calls,
  22 input / one output token); all 248 prior usage rows remain intact. Independent
  read-only verification confirms the running binary, unit, environment/config
  hashes and effective settings match the reviewed deployment; no model call was
  repeated for this verification. Token compatibility and default accounting
  (`off`) remain unchanged; remote cloud credentials remain unavailable.
  Rollback unit/binary and online backups are retained. The Mac remains on 0.12.0.
  The Mac 0.13.0 tracked streaming candidate stopped at credential-vault startup
  at both 15- and 60-second readiness bounds, before any provider calls. A macOS
  SecurityAgent process appeared; Keychain interaction is suspected, not proven.
  Do not restart the working Mac gateway until credential access and qualification
  pass. No credential transfer, shared-cache clear or database rollback occurred.
  Private restart evidence and guarded deployment scripts are retained under
  `var/gateway-upgrade-013-20261011/`; preserve both failed Mac reports. Remaining:
  Mac credential access and qualification, deployed tracked
  lifecycle and the reviewed full C5 verdict. No source/release tests need repeating.
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

### Embedded library and optional gateway acceptance

Select the route before dispatch through the same provider registry and typed
Sandhi runtime. Converge compatible endpoints on one OpenAI Chat Completions
adapter; keep one Responses adapter for subscription endpoints requiring that
protocol. Provider differences belong in validated capability/configuration data,
not vendor-specific execution loops. These are distinct wire specifications behind
one neutral contract, not competing Victor transports. Do not translate subscription
OAuth into API-key Chat Completions or silently guess a protocol after failure.
A network gateway outage/auth failure never switches to direct upstream credentials.
Preserve one session/request lineage through either route.

| Route | Verified evidence | Remaining acceptance |
| --- | --- | --- |
| Victor → embedded Sandhi → InferFlux | Installed-binding construction; real local HTTP completion, tools, SSE, errors, timeout and one-POST fixtures; earlier released-origin checks below | Released direct lifecycle/cancellation and member tasks against both deployed model IDs |
| Victor → embedded Sandhi → ZAI | Same existing HTTP matrix and installed-binding construction; provider catalog/config policy suites | Live direct authenticated complete/stream/tool turn with approved model and usage/session correlation |
| Victor → embedded Sandhi → OpenAI Codex subscription | Real binding/HTTP Responses fixture for complete/stream, required instructions, current bearer/account headers, no dispatch for invalid request | Current authorized subscription, supported endpoint/model entitlement, live lifecycle and member outcome; no silent paid-key substitution |
| Victor → embedded Sandhi → network Sandhi → provider | Existing direct/gateway handle construction and gateway-only auth isolation; released gateway evidence remains in VAS-17 | Durable streaming terminal/settlement, released managed lifecycle, then reviewed C5 |

Fixture conformance is not cloud availability. Direct mode needs no gateway process
or gateway identity, but retains the required embedded library. Gateway mode uses
its own OIDC/virtual-key authority; provider credentials stay with the gateway owner.
Reuse these matrices for later acceptance rather than adding a second transport or
copying parser tests. The retained Python-adapter work in VAS-15 is separate from
this deployment-mode requirement.

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

## UI library and theme decisions (2026-10-09)

The launch surface is an authoring canvas, a typed logic inspector and a run/audit
view backed by the same Victor service. A chart library cannot replace those
contracts. These are evaluated candidates, not new installed dependencies or
claims of measured performance:

| Responsibility | Candidate / decision | Rationale and acceptance gate |
| --- | --- | --- |
| Existing execution graph | Keep locally packaged Cytoscape + Dagre | Repair the shipped visualizer now; do not build a second authoring engine inside it. Verify wheel assets, node inspection and honest load/retry states. |
| Workflow/formation editor | Prefer React Flow in the planned TypeScript UI | Native node/handle interactions and accessibility hooks; server remains authoritative for typed connections, bounded loops, conditions, permissions and definition hashes. Prototype save/reload, undo/redo, keyboard parity and actual drag before adoption. |
| Complex automatic layout | Compare Dagre with ELK for compound/multi-port graphs | Measure representative 50/250/1,000-node workflows; record latency, memory, rendering responsiveness and layout stability. Do not pick from library size alone. |
| Logic / prompt / YAML / JSON editor | Compare Monaco and CodeMirror 6 | Monaco fits the existing VS Code ecosystem and language-service UX; CodeMirror offers composable cell editors. Measure startup/bundle/memory and accessibility; choose one primary editor after the spike. No browser eval of user logic. |
| Audit charts | Prefer an existing lightweight chart surface; evaluate D3/Observable Plot if necessary | D3 supplies low-level visualization primitives; Rickshaw targets interactive time-series graphs, not workflow authoring. Introduce neither merely for visual novelty. Charts use recorded outcomes and neutral usage, with accessible tables. |
| Notebook-style authoring | Begin with ordered Markdown, input, typed operation and result cells | Cells reference the canonical workflow definition and recorded run/step IDs. Editing/import does not execute code. Bound, sanitize and label output provenance. |
| Full notebook compatibility | Optional JupyterLab/nbformat adapter after core editor | A kernel and server are separate execution/authorization surfaces. Require isolated per-user execution, quotas, cancellation, retention, explicit trust and existing action/approval gates; never silently execute imported cells. |

Primary references: [React Flow accessibility](https://reactflow.dev/learn/advanced-use/accessibility),
[connection validation](https://reactflow.dev/examples/interaction/validation),
[D3 scope](https://d3js.org/what-is-d3), [Rickshaw](https://github.com/shutterstock/rickshaw),
[Monaco](https://github.com/microsoft/monaco-editor), and
[JupyterLab notebook model](https://jupyterlab.readthedocs.io/en/latest/extension/notebook.html).
Databricks publicly documents its adoption of
[Monaco](https://www.databricks.com/blog/2023/01/30/introducing-upgrades-databricks-notebooks-new-editor-python-formatting-and-more).
[SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/machine-learning-environments.html)
offers JupyterLab and a separate Code-OSS editor. Snowflake documents
[SQL/Python/Markdown cells](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-develop-run),
but the inspected source does not establish its underlying editor library. Do not
infer a proprietary application's stack from similar appearance or file formats.

### Product-family visual contract

Fetched references: AnvaiOps `origin/develop`
`91d731e8203e6c0b8393c8176031aae79d3d0557`, `design/tokens.css`; Anvai Identity
`origin/main` `724132c4234eebaeaa36423d02414cdb9eaf3eaa`,
`crates/idm/assets/theme.css` and `docs/specs/SPEC-UI-001-anvai-experience.md`.
AnvaiOps has unrelated local edits, so only remote refs were refreshed; Identity
was inspected in a temporary reference clone because the named sibling was absent.

Use local OSS-owned semantic values, not copied private application styles or a
runtime import from AnvaiOps. Shared light palette: background `#f6f7fb`, surface
`#ffffff`, text `#101828`, accent `#0f766e`, ink chrome `#0f172a`. Identity provides
the dark mapping: background `#0b1220`, surface `#111d2e`, text `#edf2f7`, accent
`#5eead4`. Preserve Identity's stronger control borders and readable secondary text.
Use system fonts, 12px cards, 8px controls, visible focus and reduced motion.

Dark/light/system is an appearance preference scoped to each origin; it does not
share identity cookies or imply cross-service authorization. System follows OS
changes; explicit choices persist when storage is available, and blocked storage
must still allow in-page switching. Repaint canvas colors from the same semantic
tokens without accumulating stylesheet contexts or resetting execution state.
On a reference update, compare selected light/dark tokens and accessibility
contrast, review intentional differences, update the pinned commits, and capture
headed desktop/mobile light/dark/system evidence. A shared versioned token package
is a later cross-repository decision if measured drift justifies it.

The current visualizer increment removes the failing CDN dependency, adds safe
node inspection and retry states, and implements this appearance baseline. It does
not complete VAS-14: authoring, authenticated execution, browser drag/stream support,
whole-member continuation and C5 remain open. The existing legacy execute route
also creates a record without graph metadata: an actual router fixture returns
200 on execute, then 404 on its graph URL. VAS-14a/14d must repair and test this
projection seam before claiming the UI displays a real executed workflow. The notebook/editor spike belongs to
VAS-14f/g below and must obey the existing FEP and foundation gates.

Candidate validation: 180 affected tests, 25 final focused tests and 34,173 collected;
Black/Ruff/MyPy, strict documentation build and repository hygiene pass. Installed
wheel HTTP/assets and source-distribution membership checks pass. Headed screenshots
cover desktop light/dark/system and a corrected narrow-screen canvas. Independent
review found stylesheet growth; the canonical style owner and 100-switch regression
fix it. [Evidence manifest](evidence/workflow-visualizer-ui-2026-10-09.json) records
hashes and the explicit synthetic, authentication and streaming limits.

## Work ledger

Planning/evidence: VAS-00 is accepted; VAS-01 is merged.
Delivery: VAS-02, VAS-03a, VAS-11a, VAS-11b, VAS-11d, VAS-12a and VAS-12b are
merged. VAS-04a, VAS-04b and VAS-02a are merged. VAS-03b is merged evidence only; VAS-05g and VAS-15a are merged.
VAS-17 released InferFlux provenance and bounded wire acceptance now pass; tracked buffered gateway HTTP settlement/recovery merged in Sandhi #338 and synthetic TLS/OIDC composition in #350. The Mac gateway remains on Sandhi 0.12.0; aiserver1 now runs verified 0.13.0 in token/default-accounting mode with all 248 prior usage rows preserved. OIDC source crash/restart and isolated released-binary/deployed-policy buffered qualification pass (#351). Strict OpenAI Chat stream usage qualification merged in #352; bounded complete-event framing and immutable observation retention merged in #353. Sandhi #354 merged opt-in raw response ownership after clean review and all 13 applicable CI gates; Sandhi #355 merged opt-in bounded streaming HTTP terminal/settlement and retained recovery. Sandhi #356 qualifies standalone streaming crash/restart, contention and TLS/OIDC completion/disconnect/shutdown using the existing process fixtures. Sandhi 0.13.0 is published with every artifact target verified; official-wheel consumer and local Homebrew smoke pass. Consumer/formula merge gates are complete; managed tracked deployment remains next; broader lifecycle remains open. Parent VAS-03 and formal FEP acceptance remain open. VAS-14e has a merged existing-visualizer baseline (#1271); its parent and other UI implementation rows remain open. These counts are not
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
| VAS-14e | Victor adapter for AnvaiOps/Sandesha/Sandhi/Identity family theme and bundled UI assets | VAS-14a, VAS-21a | IN PROGRESS | Existing visualizer baseline; full editor/cross-product acceptance remains.  Pinned semantic token source/adapters and drift guard; ink/teal, light/dark/high-contrast, responsive/keyboard/reduced motion; headed visual/functional snapshots; OSS/commercial boundary |
| VAS-14f | Editor library spike and canonical logic inspector | VAS-14a, VAS-21d | TODO | Reuse-qualified comparison of existing AnvaiOps React Flow/Monaco and ProximaDB Monaco/Cytoscape, graph layouts and CodeMirror on representative workflows; measure load/memory/interaction, keyboard access, packaging/licenses and typed save/reload parity; no client execution engine |
| VAS-14g | Notebook-style cells and optional Jupyter interoperability | VAS-14f, VAS-21e, VAS-05c, VAS-11, VAS-12 | TODO | Canonical definition/run linkage, bounded safe outputs, non-executing import/export; separately authorize isolated kernels if required; retain exact approval and recovery semantics |
| VAS-15 | GraphQL/MCP/legacy adapter convergence and deprecation | VAS-08, VAS-13, VAS-14 | TODO | Inventory external consumers, policy/attribution parity; one owner, compatibility window; retire only evidenced duplicates |
| VAS-15a | Preserve outcomes through the retained Python HTTP adapter | VAS-00 | ✅ MERGED | G80; [#1268](https://github.com/anvai-labs/victor/pull/1268), `e9004a5d8`; exact-candidate independent review and all applicable CI green. 144 focused/696 affected/81 docs/44 selector/199 selected tests, 34,082 collected, 95% changed-line coverage; no POST replay, adapter removal or durability claim; direct adapter remains G82/VAS-15 |
| VAS-16 | Measured performance and broader lifecycle acceptance | VAS-12, VAS-13, VAS-14, VAS-15 | TODO | Same-workload baseline/comparison; cold/warm, 1/8/32 concurrency, memory/backpressure, deadlines and crash/recovery |
| VAS-17 | Released foundation deployment and lifecycle acceptance for C5 | VAS-02, VAS-11, VAS-12 | ACTIVE (partial foundation accepted) | InferFlux v0.5.0 released/deployed; 16 synthetic wire checks pass through direct HTTP/TLS and Sandhi 0.11.0; gateway reconciliation 4 calls/48 in/2 out. Sandhi #336 admission and #337 usage qualification merged; #338 tracked buffered HTTP/recovery merged after 13 applicable green CI checks; released in Sandhi 0.12.0 and deployed on both hosts; tracked-mode lifecycle qualification pending. Two tracked crash/restart cases merged in #339 after all six applicable CI gates passed. Terminal-publication contention/death cases qualified in Sandhi #340 with 46 affected tests, 12 metrics tests and two negative controls. Synthetic buffered TLS/OIDC source acceptance merged in Sandhi #350 with 149 unique local tests and 7 applicable CI checks. OIDC crash/restart composition and isolated released 0.12.0/deployed-policy qualification pass in #351 (64 affected tests, 3 live provider calls / 51 charged tokens); managed gateway unchanged. Strict OpenAI Chat stream-usage core qualification merged in #352 after 862 Rust tests, 18 SDK regressions and 13 applicable green CI checks; no transport activation. Bounded complete-event framing/immutable observation merged in #353 after 867 Rust tests, 18 SDK regressions and 13 applicable green CI checks. Sandhi #354 merged opt-in raw response ownership with Final usage retained through drop/timeout after clean exact-commit review, 871 Rust tests, 18 SDK regressions and all 13 applicable CI checks. Sandhi #355 merged bounded transparent streaming HTTP ownership, durable terminal/settlement, shutdown headroom and retained recovery. Sandhi #356 merges standalone streaming process/TLS/OIDC acceptance (40 affected process tests; clean exact-commit review). Sandhi 0.13.0 published through #357/#358 with all artifact targets verified; 1,712 official-wheel consumer tests and local Homebrew binary smoke pass. Consumer/formula gates completed in Victor #1285 and Tap #101; remote 0.13.0 default-mode deployment passes TLS/auth/provider/accounting checks with 248 prior usage rows preserved. Mac credential access/upgrade remains pending. Next: preserved-state managed tracked deployment and broader lifecycle acceptance. Browser deployment, cancellation, executed reuse/session behavior, remote cloud credential availability and full C5 remain open |
| VAS-18 | Full mixed-team C5 verdict | VAS-17 | TODO | Six-Qwen/one-ZAI harness; unchanged deliverable/pytest/session/accounting gates; reviewed verdict on InferFlux #184 |
| VAS-19 | Matched formation cohorts and remaining semantics | VAS-18 | TODO | Preserve ZAI reference, simpler explicitly labelled local tasks, all 12 formations + 3 policies; G72 opt-in strict hierarchy separately |
| VAS-20 | OSS shared API/UI main promotion and release | VAS-16, VAS-17, VAS-14d | TODO | Full green promotion/release CI; publish server/SDK/VSIX/docs and compatibility matrix; verify installed artifacts |
| VAS-21 | Cohesive product-family UI and reusable editor components across Victor, ProximaDB, Sandhi, Sandesha and AnvaiOps | VAS-00 | TODO | Parent stays open until shared token ownership, product adapters and cross-product acceptance pass; existing standalone OSS deployment remains supported |
| VAS-21a | Review/version shared semantic token contract and publication boundary | VAS-00 | TODO | AnvaiOps source, Sandesha alignment, licenses/provenance, generated CSS adapters and drift check; no private checkout dependency |
| VAS-21b | Align Sandhi dashboard with the family theme | VAS-21a | TODO | Sandhi-owned linked worktree/PR; adapt existing dashboard CSS/JS; preserve OIDC/key/public-read modes, protected controls, real data/empty/error states, responsive accessibility and standalone packaging |
| VAS-21c | AnvaiOps-owned cross-product visual, identity and commercial-shell acceptance | VAS-14e, VAS-21b, VAS-05d | TODO | Headed AgentBrowser snapshots and actions on released artifacts; same brand/navigation, distinct audience/roles, no session/credential leakage; verify AnvaiOps/Sandesha current owner changes before adoption |
| VAS-21d | Cross-repository component, contract and test inventory | VAS-00 | IN PROGRESS | [Pinned source inventory](victor-agent-service-technology.md#reuse-with-proximadb-and-the-anvaiops-workspace) inspected; map pinned commercial spec revisions/journeys to generic OSS requirements and both acceptance fixtures; qualify actual consumers/tests, React/build compatibility, notebook import/output/auth semantics and equivalent behavior before extraction |
| VAS-21e | Minimal OSS UI contracts and two-consumer package qualification | VAS-21a, VAS-21d, VAS-14a | TODO | Victor-owned public contracts with ProximaDB fixture adapter; optional editor/canvas/notebook entries, no domain runtime/credentials; license/release owner, offline workers/CSP, peer matrix, measured performance and save/reload parity before publishing |
| VAS-21f | ProximaDB-owned adoption and standalone acceptance | VAS-21e | TODO | Repo-owned PR consumes pinned OSS release; retain database query/auth/graph adapters; packaged real-endpoint smoke and rollback; remove duplicate components only after parity, no forced graph/workflow model unification |
| VAS-21g | AnvaiOps-owned notebook/workspace adoption | VAS-21e, VAS-14g | TODO | Reuse qualified cell/editor/output components through product adapters; retain workspace kernel ownership, scoped auth, entitlement and commercial orchestration; non-executing import, malicious outputs, disconnect/interrupt/unknown outcomes and headed real-kernel acceptance |
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
| Work backward from commercial specs while preserving OSS independence | VAS-21d/e, VAS-22; AnvaiOps spec owners | Trace journey → generic public contract → standalone and commercial adapter tests; TS/React rich UI direction, deliberate divergence for unsupported/unsafe private-only requirements |
| Shared workflow/code editor/notebook components with ProximaDB and AnvaiOps | VAS-21d–g, VAS-14f/g; product-owned adapters | Public OSS package independent of private checkout; real second consumer, measured compatibility, retained execution/auth owners and commercial shell adoption without forks |
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
   migrate editor/browser clients and add actual workflow authoring. Use VAS-14a/14f
   TypeScript/React shared-editor and host-adapter qualification, preserving
   supported Svelte clients, before selecting graph UI dependencies.
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
