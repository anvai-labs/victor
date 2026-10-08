# Agent-service technology and product-family decision

Design recommendation, 2026-10-08. Implementation and acceptance status live only
in the [canonical tracker](victor-agent-service-plan.md). This extends the existing
[native acceleration policy](native-acceleration-strategy.md); it does not approve
a new orchestration engine or replace the Draft FEP-0039 acceptance gate.

## Decision and evidence

Use a bounded polyglot architecture: TypeScript for interactive clients, typed
async Python for Victor's service and orchestration, and the existing Rust/PyO3
path for qualifying compute. Keep InferFlux's existing native inference runtime
responsible for CUDA/ROCm and placement. Share contracts, visual tokens and tested
behavior across products rather than requiring all products to use one framework.

Victor already has this split: `vscode-victor/src` is TypeScript, its webview uses
Svelte/TypeScript/Vite, HTTP composition uses FastAPI, and `rust/` provides native
operations. The standalone workflow visualizer is currently an HTML/JavaScript
monitor, not a drag/drop authoring application. Sandhi's dashboard is a small
JavaScript/CSS application embedded in its Rust proxy. A common commercial brand
does not make a whole-stack rewrite necessary.

| Responsibility | Recommended technology / owner | Why and limits |
| --- | --- | --- |
| Workflow editor, run timeline, account/approval UI | TypeScript; reuse Victor's Svelte/Vite tooling and generated SDK by default | Rich browser interactions and compile-time contracts. Extract transport-neutral state/components with thin browser/VS Code host adapters. Prove standalone mounting, CSP, accessibility and graph-library support before committing to component reuse. |
| VS Code extension host | Existing TypeScript | Native editor APIs, SecretStorage and remote workspace integration. Never put provider credentials or execution authority in the webview. |
| Victor API, workflows and agent coordination | Existing typed async Python/FastAPI/framework services | Reuses plugins, durable state and tool policy. Async I/O can suspend while awaiting models; prevent synchronous blocking and bound concurrency. A faster language does not reduce upstream model latency by itself. |
| Stable CPU-heavy transforms | Existing Rust/PyO3, or an already supported vectorized library | Batch parsing/tokenization/scanning/numeric work after profiling. Include serialization, allocations, FFI and interpreter detachment in measurements. Keep a reference contract and differential tests. |
| Untrusted, long or crash-prone compute | Isolated owned worker/process where justified | Resource limits, cancellation and failure isolation. Language follows the workload/library; a Rust rewrite is not a substitute for isolation. Persist intent before effects and reconcile unknown outcomes. |
| Model serving and GPU scheduling | InferFlux's native runtime | One model-ID contract, explicit hardware placement and measured admission; Victor does not acquire a second GPU scheduler. Preserve dual-GPU acceptance and released binary/config identities. |
| Gateway policy, credential mapping and metering | Existing Sandhi Rust service or explicit embedded binding | External service for hosted multiuser policy/vault; embedded for supported local profiles. Same accounting/failure contracts, explicit capability gaps, no policy-bypass fallback. |
| Sandhi dashboard theme | Existing CSS/JS initially | Semantic token adoption needs no React/Svelte rewrite or production Node server. Consider TypeScript incrementally when state/contract maintenance demonstrates a benefit, using generated types plus runtime validation. |
| Commercial AnvaiOps integration | Product shell and supported OSS APIs | Family navigation/SSO and visual consistency without importing private product logic into Victor/Sandhi or coupling OSS startup to the control plane. |

Svelte is the reuse preference, not a claim of proven superiority over React.
VAS-14a must record a small functional spike covering accessible node/edge editing,
canonical definition round trips, package/license constraints, bundle size and
the VS Code/browser host boundary. If its required graph tooling fails those
criteria, select a maintained TypeScript alternative in a reviewed decision with
migration cost. Do not maintain two authoring applications after that decision.
Existing Chainlit consumers remain supported through the migration window.

## Where performance work pays

Measure user-visible queue wait, policy/auth latency, first-token visibility,
model prefill/decode, tool execution, checkpoint/settlement and UI rendering
separately. Compare cold/warm runs and 1/8/32 concurrency with fixed tasks/models;
record errors, memory, event-loop lag, tail latency and completed useful outcomes.
Small local models and cached responses can make orchestration overhead material,
so the initial recommendation must remain falsifiable.

First remove duplicate calls, context inflation, blocking I/O, repeated parsing,
unbounded fan-out and lock contention. Use bounded async waits for I/O and durable
timers/events for long approval waits; do not retain a worker solely to wait hours.
Parallelize independent reads without adding specialist agents unless different
permissions, expertise or measured outcome improvements justify them.

For compute, apply the existing native admission gates: a stable batchable hotspot
accounts for at least 10% of end-to-end time or a missed service target; a prototype
must improve the operation at least 2x and the user path at least 10%, including
conversion costs. Correctness, wheel support and maintainability still gate adoption.
These are repository decision thresholds, not newly measured performance results.

Do not add a Node backend, new RPC broker, Cython/Numba toolchain, WASM execution
engine or Rust service solely to unify language or chase assumed speed. A real
need for crash isolation, independent scaling or a missing latency target can
justify a process boundary, with its ownership/version/recovery costs recorded.

## Contracts at every language boundary

- Generate the TypeScript HTTP SDK and event validators from authoritative
  OpenAPI/event schemas. TypeScript checks source types; it does not validate
  untrusted JSON at runtime. Validate requests, events and stored definitions.
- Keep authentication, per-resource authorization, approvals, durable admission
  and effect reconciliation on the server. UI gating improves UX but grants no
  permission. Do not duplicate workflow scheduling or policy in a browser store.
- Persist a canonical workflow definition independently of layout coordinates;
  execute through the existing compiler/coordinator. Share model/formation
  identifiers and schemas rather than another client-side registry.
- Bound queues and payloads; propagate correlation, cancellation and deadlines.
  FFI failure must not replay network/tool/billing effects. Test partial and
  unknown outcomes across process and language boundaries.
- Pin Node/build tools and dependencies, ship compiled static UI assets, and
  verify packaged/offline assets with a real browser. Do not require a running
  frontend development server or third-party CDN for the released dashboard.

## One product family, several deployable services

AnvaiOps owns the reviewed family token baseline. VAS-14e adapts it for Victor;
VAS-21 covers Sandhi and cross-product evidence. Use a small semantic CSS token
contract with provenance and drift checks, including light/dark, focus, contrast,
reduced motion, spacing, typography, status and responsive layout. Preserve IDE
theme/high-contrast overrides. Avoid rewriting Sandesha, Sandhi or AnvaiOps just
to match Victor's frontend framework.

Prefer bundled generated token adapters until a shared asset package has a
reviewed public license, release owner and immutable version. A private sibling
checkout is not an installable OSS dependency. The dashboard remains usable
standalone; the commercial shell may link/mount the supported application through
reviewed routes without weakening CSP, frame protections or product authorization.

Kanidm provides shared identity; each product still verifies its own audience,
uses its own approved client/session configuration and checks its own grants.
Review Sandesha/Sandhi's shared OIDC candidate before adding another verifier;
its current HOLD means reuse is a dependency decision, not an accepted security
implementation. See the [identity and deployment design](victor-agent-service-auth-policy.md).

## Acceptance and reconsideration

The implementation must pass contract/TDD, duplicate-test review, real HTTP,
packaged UI, headed AgentBrowser, accessibility, lifecycle and released-provider
gates recorded in the tracker. Compare end-to-end performance before/after; a
microbenchmark, screenshot or faster language alone is insufficient. Reconsider
the framework or compute boundary only when these measurements or a documented
maintenance/compatibility limitation falsify the choice.

Primary references: [Python asyncio](https://docs.python.org/3/library/asyncio.html)
describes the I/O concurrency model; the [TypeScript handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
defines static type checking; [PyO3 parallelism](https://pyo3.rs/main/parallelism)
describes interpreter attachment and native concurrency. The allocation of these
technologies to Victor is a repository-specific engineering recommendation.
