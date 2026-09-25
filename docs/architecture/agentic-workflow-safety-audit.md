# Agentic workflow safety audit

Audit date: 2026-09-24. Source: Victor `develop` at
`7efb245c753048ab19ac1bc7c6fde31cf70c13c4` (PR #1178).
This is an assessment of the supplied **Agentic Workflow Automation** principles,
not a claim that the recommended repairs have shipped. Line references below use
that revision. The [formation handoff](multiagent-formation-coverage-handoff.md)
remains the acceptance record for formations and C5.

## Verdict and applicability

**Victor partially aligns. It has useful orchestration and validation primitives,
but the inspected execution paths do not establish the supplied durable,
policy-controlled external-write contract.** Three foundational gaps are confirmed:
a configured policy exception can allow execution; resumed execution is not bound
to the approved payload; and an effectful tool can enter generic timeout retry
without first establishing whether the effect committed.

The principles apply directly to multiagent execution. More members increase the
number of dispatch, approval, budget and recovery boundaries; they do not transfer
authority to a model. Keep deterministic coordination in software and use a single
agent unless specialization, separate permissions or measured quality justify
additional members. Parallel API reads alone do not justify another formation.

Apply the controls proportionately:

| Usage profile | Required guarantees |
|---|---|
| Trusted, local, read-only answer or draft | Bounded execution, explicit errors, scoped evidence and verified claims; a durable queue need not be mandatory. |
| Local coding tools and approved writes | Canonical checked dispatch, exact approval binding, explicit committed/unknown outcomes and effect-aware recovery. Shell and filesystem operations are effects too. |
| Long-running external business workflow | Durable admission, action intent/receipts, restart-safe approval and reconciliation, reliable dispatch, durable status. |
| Shared enterprise service | All applicable guarantees above plus authenticated run ownership, tenant/resource authorization, quotas and access-filtered retrieval. Optional local API-key configuration is not a multi-tenant isolation proof. |

Sandhi provides gateway identity, routing and usage evidence; InferFlux serves
models. Neither can bind a Victor tool approval to its business payload or deduplicate
an external business write. Those guarantees belong to Victor's execution boundary
and the authoritative API adapter. A stronger ZAI or Qwen model cannot repair them.

## Alignment with the supplied principles

"Partial" means concrete mechanisms exist but do not establish the whole contract.
There is deliberately no aggregate completion percentage: a missing approval or
recovery guarantee cannot be averaged away by unrelated successful tests.

| Principle | Assessment | Evidence and limit |
|---|---|---|
| Software controls model proposals | Partial | Structured policy events, schema validation and executor hooks exist. Policy/middleware exception handling and alternate resume dispatch weaken the boundary (G60–G61). |
| Prefer one bounded agent; justify specialists | Supported architecture; effectiveness remains workload-specific | Existing coordinator/preset dispatch and single-agent operation are usable. Formation count is not evidence of benefit; compare verified outcomes with a single-agent baseline. |
| Durable admission and reliable dispatch | Gap in workflow HTTP surface | `workflow_routes.py:148–171,254–262` creates an in-memory record and background task before returning success; no durable admission/deduplication is established there (G63). |
| Saved progress and approval waits | Partial | Paused-run persistence, expiry, single-use claim and graph checkpoints exist. Node completion can precede checkpoint persistence; approval claim precedes execution (G62). |
| Checked tool boundary | Partial | Stateful tools elevate lenient validation to strict; safety and optional RBAC run. Resume bypasses policy middleware; final dispatch is not approval-bound (G61). |
| Exact sensitive-action approval | Gap | Pending tool arguments are not compared to the executed conversation arguments; name/position selects the call (G61). |
| Recovery before retries | Gap for generic effectful tools | Bounded retry exists, but an early timeout can trigger another attempt without effect classification or reconciliation (G62). |
| Current facts and scoped retrieval evidence | Partial | Retrieval adapters and session filters exist. The inspected gateway contract lacks authenticated scope, source version/time, and explicit unavailable status (G65). This is not a demonstrated data leak. |
| Runtime limits and backpressure | Partial | Tool counters, timeouts and cost policy exist; atomic reservation across concurrent dispatch and a complete hard spend contract are not established (G64). |
| Tenant isolation and secrets outside model context | Partial/configuration-dependent | Optional RBAC and member tool restrictions help. Workflow run ownership is not enforced in the inspected routes, and argument logging needs a consistent redaction contract (G63–G64). |
| Validate before claiming success | Strong formation-test mechanisms; partial general runtime guarantee | Matrix checks artifacts, pytest, domain oracles, member/session identity and accounting. These do not prove arbitrary external business outcomes or exactly-once effects. |
| Trace and evaluate verified outcomes | Partial | Run/session/request evidence is available in the validation harness. Durable action identity, receipts and unknown-outcome reconciliation remain distinct gaps (G62). |

## Findings, in dependency order

### Follow-up: pre-dispatch enforcement repair

Landed in [#1180](https://github.com/anvai-labs/victor/pull/1180), squash commit
`3a2881a7b7f9f6640e981affd05813ae17c5577a`, after clean exact-source review and
all CI gates, including Vertical Py3.12. The findings below retain the original
audit baseline; this follow-up supersedes its pre-dispatch failure behavior only.

The first repair changes configured enforcement failures into explicit stops:
policy selection/evaluation and malformed verdicts deny; middleware selection,
execution and malformed results block dispatch; the outer tool pipeline no longer
catches those failures and proceeds. A blocked middleware result is non-retryable.
Configured context errors no longer become an empty/zero snapshot, broken governance
wiring prevents initialization, and invalid content-policy regexes are rejected.
Configured approval-handler resolution/invocation failures cannot use an explicit
allow-on-absence fallback. Errors presented to callers omit exception text.

The feature remains opt-in. No-policy operation and successful evaluations retain
their contracts; absent optional context still uses the documented default, and
absent approval handlers retain the configured fallback. Best-effort observers stay
separate from enforcement. Cancellation and durable approval pauses propagate.

This does **not** close all of G60: the TOOL_RESULT adapter/after-middleware path
still needs an explicit result-withholding contract after a tool has executed.
Streaming checks cannot retract tokens already emitted. G61–G65 are unchanged;
missing pricing remains separate from a configured context source raising an error.
The original findings below remain the historical audit of the pinned base.

Verification reused existing owners: 20 failing regressions on the original code,
followed by separate review regressions (8 and 4 failures) for malformed decisions,
scope, context and falsey approval handlers. The final targeted and affected cohort
passed **310 tests**, including durable/streaming pause and member approval suites.
Existing fail-open expectations were replaced, not duplicated; normal denial,
explicit absence, pause and cancellation tests remain. The pipeline fault cases
assert that the executor was never called. Formatting, lint and configured type
checks passed; **33,472 tests collected**. One redundant falsey-handler parameter
was removed after coverage comparison retained exactly **79 executed middleware
lines and 18 executed branches** (23 cases before, 22 after). The original failed
collection attempt encountered a concurrent coverage-database error; the isolated
collection rerun passed. These are offline controls tests, not model or C5 acceptance.

The final CI repair mapped three modules to their existing test owners and replaced
the coding vertical's stale fail-open expectation. It added no duplicate suites.
The selected cohort passed **250 tests with 94% changed-line coverage**, the vertical
middleware owner passed **26 tests with two existing skips**, and the final candidate
collected **33,475 tests**. Initial selector and Vertical failures remain recorded
in the PR alongside their repairs.

### G60 — configured policy failures can allow execution (high priority)

`victor/framework/policies/engine.py:99–105` catches an ordinary policy exception,
logs it and continues; with no denying policy the engine returns ALLOW.
`victor/agent/middleware_chain.py:227–235` similarly continues after middleware
exceptions. A mocked required policy raising `RuntimeError` reproduced ALLOW.

This does **not** mean every approval is bypassed: normal DENY blocks, ASK defaults
to refusal without an approval handler, and `ApprovalPause` inherits `BaseException`
so these ordinary exception handlers do not swallow a durable pause.

Repair: required enforcement failures must stop dispatch with a structured reason.
Only explicitly non-enforcing observers may remain best-effort. Missing required
context or failed policy installation must not silently remove enforcement.
Extend the existing policy engine/middleware test owners, including a failing guard
followed by an otherwise allowed action and confirmation that dispatch never occurs.

### G61 — durable approval is not bound to the dispatched payload (high priority)

`victor/agent/durable_resume.py:129–141` selects an unresolved call by tool name or
defaults to the first. At `:204–213` it executes arguments from the conversation via
raw `ToolService.execute_tool`, without comparing them to `pending_tool.arguments`
and without the policy middleware. A mocked resume approved one disposable argument
but dispatched a different one. No shell command or real tool was executed.

Lower controls still apply: tool budget, schema validation, safety and optional
RBAC. Stateful tools receive strict validation under the normal lenient default.
They do not check that this is the exact payload the reviewer approved. Single-use
pause claiming and a 24-hour expiry reduce replay risk but do not solve binding.

Repair: normalize once before approval; bind the exact call/action ID, canonical
payload hash, tool/schema version, principal/scope, relevant data preconditions and
expiry. At dispatch, recheck current authority and versions through the canonical
tool pipeline; satisfy only the matching ASK, never skip DENY or other guards.
Reject changed or ambiguous calls and require renewed approval. Preserve explicit
legacy-record behavior rather than guessing which call was approved.

Existing owner: `tests/unit/agent/test_durable_resume.py`. Its current approval test
already supplies mismatched pending/conversation arguments and expects conversation
arguments to execute: correct that expectation when implementing the repair, instead
of adding a contradictory duplicate suite. Extend client-resume and expiry owners
for stale approvals, competing claims and interrupted execution.

### G62 — durable external-action recovery is incomplete (high priority)

`victor/agent/tool_executor.py:1242–1262` allows timeout exceptions into the configured
retry strategy without checking effect or backend idempotency support. Explicit
`ToolResult(success=False)` is not retried, attempt limits apply, and an outer timeout
may stop execution first. Separate path-recovery retries are already read-only.
The finding concerns an early timeout on a potentially committed effect, not every
timeout or every recovery path.

`victor/framework/client.py:877–884` atomically marks a pause resumed before restoring
the session and executing. That prevents a second claim, but a crash can consume the
claim without a recorded execution outcome. Graph execution also persists node
checkpoints after execution (`graph_runtime.py:265,320`); restarting a node does not
by itself reconcile effects already committed inside it. Existing tool observability
is not a transactional intent/receipt ledger.

Repair in this order:

1. Classify tools/adapters by effects and explicit retry capabilities. Default
   unknown effectful outcomes to reconciliation, not blind replay.
2. Persist action identity, canonical request hash and intent before dispatch.
   Record attempts and `pending`, `unknown`, `confirmed` or known failure separately.
3. Use backend operation lookup or a documented same-key deduplication contract to
   reconcile. If neither exists, expose unknown status and require intervention.
4. Integrate checkpoint/approval recovery with that action owner and record verified
   receipts. Report partial committed effects; compensate only through a defined
   business operation, whose failure must also be visible.

Extend existing tool-executor retry tests with a fake backend that commits once and
loses the response. Test restart after dispatch/before receipt, repeat events and
receipt recovery. Queue delivery or a checkpoint alone must never be described as
exactly-once external execution. Cross-reference formation durability G17/G21 rather
than introducing another formation/checkpoint implementation.

### G63 — workflow API admission, ownership and cancellation need durable semantics

`victor/integrations/api/routes/workflow_routes.py:148–171,254` stores execution state
in `server._workflow_executions` and launches `asyncio.create_task`. Restart loses that
record. The status/list/cancel handlers at `:270–317` do not filter by run owner.
Cancel changes a status field without cancelling the task; continued execution can
later overwrite it with completion.

The API can require configured keys (`fastapi_server.py:590–603,663`, covered by
`tests/unit/integrations/api/test_api_auth_boundary.py:133`); this is **not** a claim
that every route is unauthenticated. The narrower gap is authenticated ownership and durable lifecycle
semantics in these routes. The chat resume route also authenticates and attributes
the caller, which is separate from checking paused-run ownership.

For the durable service profile, accept only after persistent admission, bind
request key to caller/tenant and content, and reliably dispatch through a durable
engine or outbox. Use versioned transitions and owned status lookups. Cancellation
must distinguish requested, stopped and effects already committed. Keep simpler
local behavior explicitly scoped. Extend `test_workflow_routes.py` with admission
failure, duplicate key/content conflict, cross-owner refusal, restart and cancellation
race cases. Do not add a second dispatcher.

### G64 — budget reservations and redaction need consistent boundary ownership

`victor/agent/services/tool_service.py:923–941` checks a budget before awaiting
execution and consumes afterward; that sequence alone is not an atomic reservation
for concurrent calls sharing that tool service. `tool_budget_runtime.py` provides counters, but
logical calls do not account for every internal retry. `CostBudgetPolicy`
(`victor/framework/policies/builtins.py:41–89`) is a tool-phase cost gate and treats
unavailable pricing as zero; it is not a universal hard model-spend limit.

`victor/agent/tool_pipeline.py:3142,3193,3220` logs or retains tool arguments in
execution/error paths. No universal sensitive-field projection is established at
those sites. Member allowlists and optional RBAC are useful restrictions, not OS
sandboxing or resource-level tenant authorization.

Repair: reserve shared capacity atomically before dispatch, charge attempts and
reconciliation explicitly, retain unknown usage as unknown, and fail closed when a
configured hard limit cannot be evaluated. Use one schema-aware redaction projection
for logs, traces and errors; keep runtime credentials outside prompts. Extend the
existing budget/concurrency and pipeline owners rather than copying their fixtures.

### G65 — retrieval contract does not establish authorized, versioned evidence

`victor/storage/retrieval/gateway.py:34–52` carries query/session parameters and
message/score/source/snippet results, without an authenticated access scope, evidence
version or retrieval timestamp. Vector/FTS paths pass session filters; the hybrid
query does not carry that field. Those paths may serve different corpora, so this
observation is not proof of cross-tenant disclosure. Backend exceptions can become
an empty result (`:88–100`), obscuring unavailable evidence versus no match.

For protected retrieval, enforce access before text reaches the model, preserve
source/version/time and distinguish empty, unavailable and denied results. Treat
retrieved text as untrusted evidence; it cannot grant tool authority. Use live
authoritative APIs for changing business facts and surface unresolved conflicts.
Extend `tests/unit/storage/test_retrieval_gateway.py`; its exception-to-empty test
documents behavior that must change for an explicit-error contract. Audit each
vertical adapter before claiming platform-wide tenant isolation.

## Evidence, tests and limits

This audit combines source tracing, existing test inspection, two mocked
counterexamples (policy exception and approval mismatch), and independent adversarial
review. It does not include a penetration test, crash-recovery campaign, production
write, new model run or completed fix. Test passes on the current code describe its
current behavior; some existing expectations encode the gaps above.

The existing formation matrix (`scripts/validation/formation_gateway_matrix.py`,
`formation_matrix_cases.py`, `formation_matrix_oracle.py`) independently checks
deliverables, pytest, domain results, distinct member sessions and request/usage
reconciliation. Preserve those assertions. They are stronger evidence than a model's
self-report, but do not certify arbitrary tool effects, tenant isolation or recovery.

Do not remove tests merely for similar names or assertions. Reuse the owners listed
above; consolidate only when before/after branch coverage and distinct failure cases
remain preserved. This audit increment adds no overlapping test suite and
removes no runtime tests.

Validation against the audited base plus the fixture isolation repair:

- **260 existing tests passed** across policy tests, durable resume, client resume,
  pause expiry, workflow routes, retrieval gateway, tool-budget runtime, tool executor
  and API authentication boundary owners.
- **33,440 tests collected** with `.venv-codesign/bin/python -m pytest tests/
  --collect-only -q`; Black, Ruff and `git diff --check` passed for the change.
- Both mocked counterexamples reproduced. The fake required policy returned ALLOW
  after its exception; the fake tool service recorded conversation arguments that
  differed from the pending approval. Real tool calls: **zero**.
- The first combined executor/authentication run had **126 passed / 1 failed**:
  the existing HITL fixture opened the developer's default approval database and
  encountered a read-only write error. The fixture now redirects the existing
  `get_default_hitl_db_path` lookup to `tmp_path / "hitl.db"`; real SQLite and the
  authentication assertions remain. Its affected API suites passed **13/13**, then
  the complete selected set passed **260/260**. No production database change or
  runtime security repair is included.

Tests used `.venv-codesign/bin/python -m pytest` from the linked worktree with
`VICTOR_ENABLE_MLX_PROVIDER=0` because this audit does not exercise local MLX.
The failures above are recorded separately from the passing reruns; neither the
fixture repair nor the baseline suite resolves G60–G65.

## Implementation and acceptance sequence

1. Finish released Sandhi/InferFlux foundation deployment and liveness acceptance;
   retain the current hold on formation/C5 runs. Their release status is independent
   of this Victor audit.
2. Repair G60 and G61 through failing regressions in existing owners, canonical
   enforcement, and independent review. These are model-independent safety controls.
3. Establish G62 action ownership and effect-aware recovery before durable write
   automation; then connect G63 admission, approval recovery and truthful status.
4. Complete G64/G65 for the selected deployment profile. Avoid introducing an
   enterprise queue or tenant system into every local read-only use case.
5. Resume ZAI reference then appropriately sized InferFlux formation tasks only
   after foundation readiness. Report formation quality separately from control
   guarantees. Measure verified completion, unintended/duplicate actions, unknown
   outcomes, intervention, latency and cost per successful task against a single-agent
   baseline; no additional formation is justified solely by this audit.

Close each gap with source/PR identity, positive and negative tests, tested recovery
boundaries and remaining limitations. Preserve original failed evidence. C5 remains
open until its full mixed-team verdict is reviewed.
