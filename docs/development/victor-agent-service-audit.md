# Victor agent-service and client API audit

**Decision report / proposal only — October 7, 2026 (October 8 UTC).** No API
consolidation is implemented by this report. Audited `origin/develop`
`b8e195cc68e009c68b2cfc9b96ac8cf5b74bef84` plus the pending Node/dependency and
activation fixes in `fix/vscode-security-locks`. A fresh fetch found develop
unchanged. Do not count this document as an accepted FEP or completed milestone.

The [implementation tracker](../architecture/victor-agent-service-plan.md) owns
current task statuses and restart instructions. The original report below remains a dated
finding/evidence snapshot; its proposed sequence does not supersede the tracker.

## Consumer reconciliation — October 8, 2026

Rechecked `origin/develop` at `b19f0f6730a04b52974fae82829c7ed0152b9d01`
after [#1265](https://github.com/anvai-labs/victor/pull/1265). The original report
below is preserved as audit-time evidence, not a list of still-unfixed defects.
Delivery status remains in the canonical tracker; this section records the
consumer relationships and compatibility decisions required by VAS-03b.

The stream request mismatch was repaired in [#1252](https://github.com/anvai-labs/victor/pull/1252).
Transport termination and cancellation acknowledgement were repaired in
[#1262](https://github.com/anvai-labs/victor/pull/1262); paused nonstream responses
and their three content consumers were repaired in #1265. Those bounded repairs
are not shared authentication, durable run results, approval/resume UI, remote
cancellation, a release, or C5 acceptance. The API audit's historical RED probes
must not be used to claim those exact VS Code defects still exist.

| Consumer / actual call path | Present contract and ownership | Migration decision / remaining evidence |
| --- | --- | --- |
| VS Code chat panel: `chatViewProvider.ts` → `victorClient.ts::streamChat` | Extension host owns HTTP; compatibility body serves core `messages` and web `message`/optional `session_id`; core `[DONE]` or web v1 `stream_end` required | Keep the explicit compatibility behavior until both server entry points pass the shared contract. A terminator establishes transport completion only |
| Composer, terminal suggestions, Smart Paste → `VictorClient.chat` | Core `POST /chat`; preserves `status`, `run_id`, `approval_request`; one completion guard prevents consuming paused content | Preserve #1265 ownership/no-paste regressions. No automatic resume or restart-safe UI pause retention; ordinary transport-error paste fallback remains legacy behavior |
| Svelte webview: `webview-ui/src/stores/chat.ts` → VS Code `postMessage` → `ChatViewProvider` | Embedded webview bridge, not a standalone network client of `web/server`; host owns credentials and HTTP | Share future SDK/state at the host boundary; validate bridge messages separately. Do not invent a second browser REST owner based on the directory name |
| Active EventBridge: `ChatViewProvider._ensureEventBridgeConnected` → `eventBridgeClient.ts` | Separate `ws` connection to `/ws/events`, subscription/correlation filters; constructor currently sends no authorization | Existing-key protected server rejects it. VAS-05g owns the planned credential/connection repair; correlation filtering is not resource authorization |
| Main `VictorClient.connectWebSocket` | `/ws` uses first-message API-key auth; optional prefetched session token in URL. No production caller of `connectWebSocket()` found under `vscode-victor/src` | Inventory as a retained client API, not the active EventBridge transport. Core `/session/token` is still a placeholder; neither random token nor WS open means authenticated/resumable session |
| Browser Chainlit: `victor/ui/chat_app/app.py` | Per-browser-session framework `VictorClient`; calls Python in process, including existing session restore and UI handlers | Retain embedded mode. Hosted remote mode needs the shared adapter; session history restore is not durable action reconciliation |
| Separate `web/server/main.py` | Server entry point with singular-message SSE, `X-Session-Id`, signed session tokens and bounded in-memory sessions; several IDE endpoints are placeholders | No independent browser HTTP client was found in checked-in `web/`; external consumers remain unknown. Preserve compatibility until usage inventory/parity/deprecation are approved |
| GraphQL: `graphql_schema.py` | Mounted under the core auth gates when installed; chat resolver calls the orchestrator directly, subscriptions use event infrastructure | Retain while auditing external clients; shared authentication does not prove principal/resource policy parity. Delegate to canonical services before claiming equivalent execution semantics |
| MCP: `integrations/mcp/server.py` | Registry tool execution with an empty context in `call_tool`; stdio/JSON-RPC shape | Keep interoperability; qualify principal/policy context before hosted use. Do not rename it REST or remove it to reduce endpoint count |
| Public Python protocol adapters: `integrations/protocol/{interface,adapters}.py` | Exported direct/HTTP wrappers; no non-example production consumer imports found elsewhere in the repository | Package docstrings claim CLI/VS Code usage not supported by the current call sites. Preserve public imports pending external inventory; VAS-15a covers the reproduced outcome gaps below |
| Embedded Python / teams / workflows | Existing framework client, coordinator and workflow owners | No network hop for embedded calls. Shared run envelopes retain formation-specific execution; peer formations do not require a supervisor |
| Sandhi / InferFlux | Gateway/provider and model-serving APIs, independent of Victor's agent-run API | Preserve separate token audiences and deployment ownership. Do not make gateway login an approval or expose provider credentials to the UI |

**Newly isolated legacy adapter gap (G80).** Synthetic compatibility probes against
the actual `ChatResponse.from_dict` convert a paused response with run/approval
fields into `{content, tool_calls, finish_reason: "stop", usage}` and discard the
pause identity. A real `HTTPProtocolAdapter.stream_chat` with an `httpx.MockTransport`
response containing one complete `data` frame but no terminal marker yields the
partial content and exhausts normally; exactly one POST was observed. These are
local adapter probes, not live server/member/model evidence. Existing protocol
tests do not establish the lost-outcome invariant. Repair in the existing test
owners; do not publish a second result registry or delete the public adapter
because an internal caller was not found.

The existing `src/test/eventBridgeClient.test.ts` contains placeholder assertions
and sits outside the host runner's `test/suite` discovery root. It is not current
client authentication evidence; VAS-05g must exercise the real owner in a discovered
suite and remove superseded placeholders only after preserving useful invariants.

**Inventory limits.** This is checked-in source reachability, not deployed usage
telemetry. External clients, installed extensions and generated/dynamic imports
need explicit compatibility evidence before removal. Existing contract tests are
reused; this documentation increment adds no mirrored parser or inventory-only
runtime test. The review gates and unresolved design choices are in
[FEP-0039](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0039-unified-agent-service-api.md#review-process-and-acceptance-gates).

## Recommendation

Expose one authoritative Victor agent service to the server web UI, VS Code and
external API consumers. Share request/response schemas, a TypeScript SDK, event
parsing and run-state semantics. Keep UI components and authentication storage
appropriate to their host. Preserve the Python framework and definition SDK;
MCP and any retained GraphQL become thin adapters to the same application services.

Use `victor serve` as the canonical composition root. Migrate the useful session
store and v1 event work from `web/server` into framework-owned services and the
existing FastAPI routers. Keep a temporary compatibility entry point for the old
web app. Do not add an HTTP-to-HTTP proxy inside the same deployment merely to
share code. The Sandhi Python binding is required by Victor (`sandhi-gateway==0.11.0`);
routing through a separately deployed Sandhi proxy is optional. InferFlux remains
the inference service. Neither owns Victor run approval or state.

## Surface inventory

| Surface | Current purpose/evidence | Disposition |
| --- | --- | --- |
| Python framework | `victor/framework/{_api,agent,client,session_config}.py`: Agent/TaskResult, VictorClient, StateGraph/workflows/teams | Retain; application execution owner behind transport adapters |
| Definition/plugin SDK | `victor-contracts/victor_contracts`; plugins and vertical definitions | Retain independent boundary; do not put HTTP clients or runtime state here |
| Core FastAPI server | `victor/integrations/api/fastapi_server.py`, `routes/`: 102 HTTP operations counted in core router factories, excluding separately mounted HITL/GraphQL/plugins | Canonical HTTP service; cohesive routers, one composition root |
| Separate web backend | `web/server/main.py`: separate FastAPI app, sessions, signed tokens, v1 SSE, rendering and placeholder compatibility routes | Consolidate useful implementations; deprecate duplicate app after parity |
| Chainlit web chat | `victor/ui/chat_app/app.py`: per-session VictorClient and approval UI, in process | Preserve UI; use remote service adapter for hosted mode, same application service in embedded mode |
| VS Code HTTP client | `vscode-victor/src/victorClient.ts`: 77 distinct literal HTTP method/path calls; handwritten types and SSE parser | Shared generated contract SDK plus thin IDE adapter |
| Streaming/events | `/chat/stream`, `/ws`, `/ws/events`, workflow WS; framework wire events and EventBridge | One execution event contract over SSE; keep WS only for demonstrated duplex/subscription needs |
| Webview bridge | Svelte/legacy HTML ↔ extension host `postMessage`, plus command URI/CodeLens callbacks | Validated, discriminated UI messages; network/auth in extension host |
| GraphQL | `graphql_schema.py`: optional dependency, enabled when available; separate resolvers | Freeze expansion; retained resolvers delegate to same services, deprecate only after consumer inventory |
| MCP | `integrations/mcp`: JSON-RPC tools/resources, including stdio | Retain interoperability; align authorization/execution policy, not REST envelope format |
| Legacy protocol adapters | `integrations/protocol/{interface,messages,adapters}.py`: another DTO family and direct/HTTP adapters | Deprecation candidate; no production imports found outside its own package/docs, but external consumers require notice |
| Inference/provider API | Provider adapters, required Sandhi Python binding, optional standalone proxy, InferFlux/OpenAI-compatible models | Separate model API from agent-run API; preserve provider abstraction |

Counts describe source operations, not successful end-to-end coverage, deployed
routes, or distinct business capabilities. Core API routes include tools, agents,
teams, plans, workflows, configuration, search, LSP, Git, terminal and observability.

## Findings and priorities

P0 blocks reliable shared-service exposure; P1 addresses boundary robustness;
P2 is measured simplification/performance. Findings are source observations unless
marked reproduced. Existing tests passing does not negate a missing contract test.

| Priority | Finding | Evidence and user impact | Recommended change |
| --- | --- | --- | --- |
| P0 | Two incompatible chat APIs | VS Code `streamChat()` sends `{message, session_id?}`; ordinary `chat()` already sends `messages`. Streaming core `ChatRequest` requires `messages`. **Reproduced HTTP 422**, missing `body.messages`, before runtime initialization. The separate web backend accepts the VS Code shape. | One versioned chat/run contract and executable cross-language contract fixtures; compatibility adapter for existing callers |
| P0 | Session ownership differs | Core caches one `_victor_client`; separate web uses a bounded per-session in-memory store. Core does not expose the session-header contract expected by VS Code. Web session reuse/token reissuance checks shared API-key access and session existence, not per-principal ownership; an HMAC is not authorization. | Explicit principal/workspace/session/run ownership, concurrent-turn policy, durable state where restart is promised |
| P0 | Approval semantics lost in clients | REST `/chat` returns 202 + `awaiting_approval`, run ID and request; TS `chat()` discards those fields. `/tools/approve` uses a separate in-memory map; `/chat/resume` and `/hitl` are other surfaces. | One approval lifecycle and exact payload/version binding; retain domain adapters, expose resumable run state in both UIs |
| P0 | EOF can appear successful | TS stream parser resolves on `end` without requiring completion; malformed frames are logged then ignored; core and web use different terminators. | Bounded UTF-8/SSE parser, explicit terminal run status, interruption/unknown state on premature EOF; stream end alone is not business success |
| P0 | Authentication is inconsistent | Core HTTP/WS gates exist and tests pass. EventBridgeClient sends no auth to guarded `/ws/events`. Core `/session/token` returns random placeholder IDs; web app signs tokens. VS Code reads server key from ordinary configuration. | One authenticated principal resolver, scoped access and protected event subscriptions; host secret storage; remove placeholder security contracts |
| P0 | Cancellation acknowledgement is overstated | TS returns true after WS send; core handler has no `cancel_tool` branch. HTTP returns true client-side for 200 even when body says `cancelled:false`. | Requested/acknowledged/terminal cancellation states; verify runtime/provider effect, preserve already committed effects |
| P1 | Multiple execution seams | Core chat uses framework client; completions calls provider directly; GraphQL chat calls orchestrator; MCP executes registry tools with empty context. | Route through existing application services with shared policy and attribution; completions need a bounded fast path, not a full agent loop |
| P1 | Duplicated DTOs and weak schema checking | Pydantic, protocol dataclasses and handwritten TS overlap; many responses are `unknown` or cast records. Endpoint test verifies paths only. | Publish OpenAPI and event JSON Schema from one authoritative contract owner; generate TS types/client and validate at network boundaries |
| P1 | Capabilities and errors can mislead | Empty capabilities implies supported; several catches return empty data on failures. | Explicit supported/unavailable/unknown states and typed errors; no silent auth/network-to-empty-success conversion |
| P1 | Webview/Markdown boundaries need tightening | Existing CSP is useful; incoming message fields are unchecked; hover trusts combined Markdown broadly. | Validate messages and target scope, allowlist trusted command IDs, escape untrusted content, preserve workspace trust and remote path semantics |
| P2 | Parallel event/rendering paths | EventBridge has useful bounded queues/drop counters, while chat also streams directly; UI posts cumulative content per chunk. | Durable run events authoritative, telemetry separate; sequence/deduplicate events, bounded UI batching and retention after measurement |
| P2 | Inactive command and protocol surface | Activation smoke found duplicate symbol IDs and six unregistered advertisements. Local pending dependency branch repairs/prunes them. | Keep one command owner; restore dormant features only with real invocation tests; deprecate unused adapters after consumer audit |

The September U7 review and UX action plan are useful historical records, but
some assertions are stale: web/server now uses VictorClient, session storage is
typed/bounded, and core HTTP/WS auth gates were repaired. Preserve these changes.
Do not repeat already completed work or claim all historical findings remain open.

## Proposed public agent-service contract

Illustrative routes, subject to FEP and compatibility review; not existing API claims.

| Resource / operation | Contract | Customer-visible behavior |
| --- | --- | --- |
| `GET /v1/capabilities` | Schema versions, enabled features, limits, auth mode | Both UIs show only supported actions and explain incompatibility |
| Agent definitions | Approved single-agent/team/workflow definition and configuration references | Expose an agent as a service without arbitrary module loading or client-supplied credentials |
| `POST /v1/sessions` | Owner, workspace scope and immutable configuration reference | Separate tabs/workspaces cannot silently share conversation state |
| `POST /v1/runs` | Session/definition ID, typed input, deadline/budget and idempotency key | Return 202 + stable run ID only after durable acceptance in durable mode |
| `GET /v1/runs/{id}` | Recorded status, result/artifacts, approvals, errors and timestamps | Reconnect/refresh obtains the same authoritative outcome |
| `GET /v1/runs/{id}/events` | Versioned events with event ID/sequence, run/session/member/action correlation | SSE replay from a cursor; explicit retention gaps, no re-execution to recover UI |
| Approval decision | Approval ID, decision, exact payload/version hash, expiry; actor from authenticated context | Reject stale/unauthorized decisions before dispatch; continuation uses existing owner |
| Cancellation request | Stable run/action identity; cancellation requested versus completed | Stop controls never claim that committed writes were undone |
| `/v1/completions`, search/LSP | Bounded read-oriented requests with cancellation and feature gates | Low latency editor operations without agent orchestration overhead |
| Health/readiness/admin | Separate liveness, capacity/readiness, and privileged configuration | Useful diagnostics without exposing credentials or unrestricted administration |

Single agents, teams and workflows share run lifecycle/result envelopes, with
typed execution-specific definitions. Do not flatten their semantics or mandate
a supervisor for peer formations. Reuse existing run, pause, journal and team
owners rather than inventing a parallel agent-service state machine.

## Shared client architecture and security

| Layer | Share | Keep host-specific |
| --- | --- | --- |
| Contracts | OpenAPI DTOs, event schemas, compatibility rules, error/state enums | No environment-specific credentials |
| TypeScript SDK | Transport interface, auth injection, validated response/event decoder, cancellation, replay cursor | Browser fetch versus extension-host fetch/HTTP adapter |
| State layer | Run reducer, event deduplication, pending approval and terminal-state semantics | Browser tabs versus VS Code panels/workspaces |
| UI | Optional reusable presentation components if useful | VS Code commands, editor APIs, theme/accessibility and file application |
| Authentication | Verified principal, role/scope policy, issuer/audience/expiry checks | Browser OIDC session/cookies with CSRF controls; VS Code interactive login + SecretStorage; scoped noninteractive credentials for automation |
| Deployment | One service implementation and route composition | Embedded/direct mode versus remote HTTP; optional same-origin UI hosting |

OIDC should be the hosted-user posture where configured; a gateway login alone
does not prove Victor endpoint authorization. Retain explicit scoped machine-auth
and tightly constrained local-development modes. Never put reusable bearer
credentials into query strings or webview messages. Browser SSE can use a
same-origin authenticated session or an authenticated fetch stream; native
EventSource cannot attach arbitrary Authorization headers.

## Performance decisions and validation

No comparative throughput or latency benchmark was run during this audit. These
are acceptance proposals, not achieved performance claims.

| Concern | Preferred design | Required evidence |
| --- | --- | --- |
| Network overhead | One UI-to-service hop; pool/reuse connections | Compare p50/p95 latency and CPU at 1/8/32 concurrent runs, same model/tasks |
| Editor responsiveness | Bounded completion request, existing cancellation, measured debounce | Keystroke-to-suggestion p95, stale-result rate and cancellation latency |
| Streaming | Incremental UTF-8 decoder, full SSE framing, bounded frame/buffer sizes | Split every byte boundary, CRLF/multiline, oversized frame, truncated stream and slow consumer tests |
| UI rendering | Deltas plus measured 16–50ms batching as an initial experiment | Frame time, first-token visibility and heap growth on long streams; retain only if better |
| Admission and deadlines | Per-workload concurrency/budgets and propagated remaining deadline | Saturation, fairness, queue wait and timeout behavior; no arbitrary one-user RPS cap |
| Durability | Persist logical actions, approvals, checkpoints and terminal outcomes; define token-event retention separately | Crash after acceptance/commit, reconnect and duplicate requests; no unverified action replay |
| Resource bounds | Preserve bounded session/event queues; partition by owner | Slow-client isolation and bounded RSS/task/socket counts after repeated disconnects |
| Retry policy | Retry safe reads/transient transport only within budgets; writes require proven same-key semantics or reconciliation | Commit-then-timeout recovery, duplicate-key/different-payload conflict, Retry-After handling |

Suggested regression gate: no more than 10% p95 adapter-overhead regression against
a recorded same-workload baseline, plus agreed absolute UX limits after baseline
measurement. No language rewrite, new broker, mandatory Redis, GraphQL expansion
or gRPC migration without a measured requirement. One-process deployments can use
an appropriate SQLite-backed durable store; distributed execution requires a
separately reviewed shared-store/lease/admission design.

## PR-sized sequence and adversarial gates

| Order | Milestone | Exit criteria |
| --- | --- | --- |
| 0 | Finish isolated Node/dependency/activation repair | Clean audits, real activation/package smoke, reviewed candidate and all CI; no claim of chat-service acceptance |
| 1 | FEP + current contract inventory; TDD failing consumer-provider fixtures | Reproduce 422, approval-field loss, EOF and false cancellation; track consumer compatibility and deprecation plan |
| 2 | Authoritative session/run/auth ownership | Cross-principal/workspace denial, concurrent-turn policy, durable acceptance/restart, exact approval binding |
| 3 | Canonical FastAPI composition and shared schemas/SDK | Port existing web session/wire behavior; both clients use same route/service; legacy adapter parity; no second dispatch |
| 4 | Shared event and result lifecycle | UTF-8/framing, replay, terminal outcomes, verified cancellation, unknown-write reconciliation and UI reconnect pass |
| 5 | UI migration and obsolete-surface removal | Web + VS Code against real ephemeral server; both same API; remote workspace, auth expiry, disabled capabilities, accessibility and packaged VSIX smoke |
| 6 | Measured performance and shared API release | Benchmarks, independent adversarial review, green full CI, released server/SDK/VSIX compatibility |
| Independent framework path | Existing G61/G62/G70 lifecycle fixes, released foundations, then C5 | The framework harness need not wait for UI migration or GraphQL deprecation; see VAS-11/12/17/18 in the tracker |

Incremental API changes require a FEP under repository rules. Migrate compatibility
adapters with a deprecation window; do not silently change existing public payloads
or remove external interfaces merely because internal references are absent.
Extend existing tests by ownership; replace vacuous/duplicate fixtures only when
their invariant is demonstrably covered. Do not retain both a legacy fake-only
contract suite and an equivalent generated contract suite indefinitely.

## Evidence and limits

- Fresh origin/develop: `b8e195cc68e009c68b2cfc9b96ac8cf5b74bef84`.
- Reproduced current TS stream body against the real FastAPI router: HTTP 422,
  missing `messages`; runtime initialization was not called (no model traffic).
- Compiled TS client with controlled transport stubs (not live-agent evidence):
  a paused 202 response returns only role/content/toolCalls; a stream containing
  partial content then EOF resolves; HTTP `cancelled:false` returns true to the UI.
- Existing route-presence, HTTP/WS auth and resume tests: **23 passed**.
- Existing wire-event, web session-store and web-boundary suites: **33 passed**.
- Pending dependency branch: Node 24.21.0; **999 real-host integration tests**, 50
  Vitest tests, zero reported npm vulnerabilities in both graphs, production VSIX
  built. These tests do not prove real-server chat works; this audit found that gap.
- Existing full Python collection: **33,883 collected**; collection is not execution.
- No provider benchmark, OIDC deployment, public API migration, live mixed-team
  run, release or CI merge was performed as part of this audit.

## Standards consulted

- [OpenAPI 3.1.2](https://spec.openapis.org/oas/v3.1.2.html): published machine-readable HTTP contracts.
- [WHATWG SSE](https://html.spec.whatwg.org/multipage/server-sent-events.html): framing, UTF-8 and event IDs/reconnection.
- [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html): method semantics and constraints on retrying non-idempotent requests.
- [VS Code webview guide](https://code.visualstudio.com/api/extension-guides/webview): message passing, CSP and content security.
- [VS Code remote extensions](https://code.visualstudio.com/api/advanced-topics/remote-extensions): extension-host and webview deployment differences.

Independent adversarial review confirmed the principal findings and operation
count. Corrections incorporated: required Sandhi binding versus optional proxy,
stream-only request-shape mismatch, and HMAC versus session ownership distinction.
