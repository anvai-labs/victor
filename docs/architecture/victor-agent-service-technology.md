# Agent-service technology and product-family decision

Design recommendation, updated 2026-10-09. Implementation and acceptance status live only
in the [canonical tracker](victor-agent-service-plan.md). This extends the existing
[native acceleration policy](native-acceleration-strategy.md); it does not approve
a new orchestration engine or replace the FEP-0039 acceptance gate (currently Review).

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
| New shared workflow/code/notebook UI | TypeScript/React; Vite host qualification and generated product SDKs | Align with the existing AnvaiOps/ProximaDB rich UI investments. Keep transport-neutral contracts and thin host adapters; prove compatible peers, standalone mounting, CSP and accessibility before package adoption. Existing Svelte webviews remain supported. |
| VS Code extension host | Existing TypeScript | Native editor APIs, SecretStorage and remote workspace integration. Never put provider credentials or execution authority in the webview. |
| Victor API, workflows and agent coordination | Existing typed async Python/FastAPI/framework services | Reuses plugins, durable state and tool policy. Async I/O can suspend while awaiting models; prevent synchronous blocking and bound concurrency. A faster language does not reduce upstream model latency by itself. |
| Stable CPU-heavy transforms | Existing Rust/PyO3, or an already supported vectorized library | Batch parsing/tokenization/scanning/numeric work after profiling. Include serialization, allocations, FFI and interpreter detachment in measurements. Keep a reference contract and differential tests. |
| Untrusted, long or crash-prone compute | Isolated owned worker/process where justified | Resource limits, cancellation and failure isolation. Language follows the workload/library; a Rust rewrite is not a substitute for isolation. Persist intent before effects and reconcile unknown outcomes. |
| Model serving and GPU scheduling | InferFlux's native runtime | One model-ID contract, explicit hardware placement and measured admission; Victor does not acquire a second GPU scheduler. Preserve dual-GPU acceptance and released binary/config identities. |
| Gateway policy, credential mapping and metering | Existing Sandhi Rust service or explicit embedded binding | External service for hosted multiuser policy/vault; embedded for supported local profiles. Same accounting/failure contracts, explicit capability gaps, no policy-bypass fallback. |
| Sandhi dashboard theme | Existing CSS/JS initially | Semantic token adoption needs no React/Svelte rewrite or production Node server. Consider TypeScript incrementally when state/contract maintenance demonstrates a benefit, using generated types plus runtime validation. |
| Commercial AnvaiOps integration | Product shell and supported OSS APIs | Family navigation/SSO and visual consistency without importing private product logic into Victor/Sandhi or coupling OSS startup to the control plane. |

The 2026-10-09 cross-product inventory supersedes the earlier Svelte-first
preference for **new shared rich editors**. React is a compatibility direction,
not a measured performance winner or a mandate to replace existing Svelte
webviews. VAS-14a/14f must qualify accessible node/edge editing, canonical definition
round trips, bundle/worker costs and the browser/VS Code host boundary. Keep one
authoring implementation behind supported host adapters; do not duplicate it in
Svelte and React. A failed qualification requires a reviewed alternative and
migration cost. Existing Svelte and Chainlit consumers retain compatibility until
an evidenced migration; Node 24 remains the established Victor build baseline.

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

## Database transaction and external-effect boundary

Use the database as the authority for durable local state. Its constraints,
conditional updates and transactions must enforce uniqueness and legal state
transitions; an in-process lock alone is insufficient across workers. Group the
local changes that establish one transition in one transaction. Keep model calls,
human waits and external tool requests outside that transaction.

| Boundary | Required guarantee | Recovery rule |
| --- | --- | --- |
| Approval and action intent | Verify the exact binding and current state; atomically commit the local records required by an authorized transition | A losing claimant cannot dispatch. A consumed claim with missing intent must not be reopened by assumption. |
| Durable admission and dispatch | Commit accepted work and its dispatch record together, or use an existing durable engine with equivalent guarantees | Recover delivery from committed state. Outbox delivery can repeat; it does not make an external effect exactly once. |
| External tool effect | Use a stable action key only where the backend actually guarantees its semantics; preserve backend identity and request binding | A timeout after possible dispatch leaves the outcome unknown unless authoritative evidence establishes it. Query authoritative status/receipt before retry; without a safe backend contract, stop for reconciliation. |
| Receipt and continuation | Validate receipt provenance and exact action identity; commit monotonic outcome/checkpoint changes through the existing store owner | Receipt lookup must not execute the effect. Crash or duplicate reconciliation must not publish duplicate results or rerun completed work. |

Current code already provides a conditional single-use approval claim and a
transactional action intent/observation update in `ProjectDbPausedRunStore` when
`durable_actions=True`; default resume does not write that action journal.
Approval claiming, intent creation and transcript publication are separate
commits today. Moving blocking SQLite work to a worker thread changes event-loop
responsiveness, not the transaction's atomicity. Cancellation can leave a consumed
claim, and a returned invocation is still not a verified backend receipt.

VAS-11b merged bounded local receipt provenance/reconciliation in #1255, as
recorded in the tracker. VAS-11c still must qualify a production backend adapter
and its transaction/effect boundary; VAS-12 owns complete continuation. Verify the
backend durability settings and failure model, including process crash versus
power loss. Define absent/negative receipt semantics: a not-found response from
an eventually consistent lookup must not authorize replay. Do not invent a
second action ledger or a generic retry endpoint. Include backend commit with
lost response, wrong/stale/conflicting receipt, concurrent reconciliation, and
crash-before-publication tests in the existing owners. VAS-12 separately owns
complete member continuation. The bounded local implementation does not establish
production receipt-adapter, full continuation or live acceptance.

[SQLite's transaction guarantee](https://www.sqlite.org/transactional.html) covers
changes within its transaction. The [transactional outbox pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html)
addresses durable state plus message publication and still requires handling
duplicate delivery. Neither automatically includes an arbitrary remote API in
the local transaction. Keep the existing database until measured concurrency,
deployment or availability requirements justify another backend; changing the
database product alone does not resolve external-effect uncertainty.

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

**Repository/product boundary:** Victor owns reusable OSS agent-service building
blocks. AnvaiOps owns the commercial offering built from those blocks. This is
an ownership decision, not a new licensing change or a requirement to move other
projects into Victor.

| Repository | Owns | Must not require |
| --- | --- | --- |
| Victor (OSS) | Generic agent/workflow/formation runtime, durable execution, service API, SDKs, generic auth/policy adapters, reusable authoring/UI components and a functional standalone experience | AnvaiOps account, private checkout, commercial entitlement service or product billing to use the OSS capabilities |
| ProximaDB (OSS) | Database/query/graph execution, database authorization, its SDK and standalone query/graph UI; adapter to reusable presentation components | Victor owning its query engine, or an AnvaiOps runtime dependency for standalone queries |
| Sandhi / InferFlux (their OSS projects) | Gateway/metering/provider contracts and inference/hardware runtime respectively; their standalone admin UIs and reusable integration APIs | Victor owning a duplicate gateway/inference implementation, or private AnvaiOps services for standalone operation |
| AnvaiOps (commercial product) | Integrated product shell/navigation, managed deployment/fleet/customer operations, billing/subscriptions/entitlements, commercial packaging and product-specific workflows | A fork or privileged bypass of Victor/Sandhi policy, a second execution engine, or proprietary code copied into OSS |
| Sandesha (its product owner) | Messaging/product functionality and its own UI adapter | Victor adopting its application code or backend framework just to share a theme |

Use public versioned contracts and generic extension/configuration points in OSS.
Keep commercial policy data and integrations in AnvaiOps. Authentication,
authorization, auditability and safe recovery remain OSS security building blocks;
they must not become commercial-only correctness fixes. Shared assets must have
compatible publication/license provenance and an offline standalone fallback.
AnvaiOps may apply branding/entitlements to its offering without granting extra
runtime authority. Its commercial acceptance/release is separate from Victor's
OSS release; a private control-plane outage must not break a standalone install.


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

## Reuse with ProximaDB and the AnvaiOps workspace

The workflow canvas, source editor, notebook document and result presentation
should share qualified components, while each service retains its execution,
authorization and persistence contracts. Build the reusable OSS surface in Victor
first, with explicit adapters for ProximaDB and AnvaiOps. This is a proposed
packaging boundary, not an extracted or published package. VAS-21d–g in the
[canonical tracker](victor-agent-service-plan.md) own qualification and adoption.

### Source inventory and compatibility risks

Inspected fetched source on 2026-10-09: Victor `8a4b6ad0e4732114fe609461d143e4f8c343d886`,
ProximaDB `9d6acc7f4f08abec57b4c54b3a0bc8d5ee6ddb6c`, AnvaiOps
`91d731e8203e6c0b8393c8176031aae79d3d0557`. Sibling worktrees were not changed;
AnvaiOps has unrelated local work. This inventory establishes source surfaces,
not runtime acceptance or permission to copy private implementation into OSS.

| Surface | Existing owner and inspected paths | Reuse decision / evidence still needed |
| --- | --- | --- |
| Workflow/run graph | Victor `victor/ui/visualizer/`; bundled Cytoscape visualizer | Preserve the shipped viewer; repair the actual execution-to-graph projection in VAS-14a. Authoring must use Victor's canonical compiler/formation registry, with layout stored separately. |
| Database graph | ProximaDB `ui/src/components/GraphVisualizationTab.tsx`; Cytoscape | Reuse tokens, selection/inspector and accessibility patterns where contracts match. A property graph is not an executable workflow; keep its node/edge semantics and database adapter distinct. |
| Commercial code graph | AnvaiOps `apps/control-plane/src/react/codegraph/GraphCanvas.tsx`; React Flow | Inspect existing interaction/tests before introducing a second canvas. React Flow is a candidate for authoring, not evidence that Victor workflows or database graphs already share a model. |
| Code/SQL editor | ProximaDB `ui/src/components/SqlQueryTab.tsx` uses Monaco React; AnvaiOps `apps/control-plane/src/main.ts` and `react/codebrowser/ReadOnlyMonaco.tsx` use Monaco loader | Qualify one generic document/editor lifecycle with language, diagnostics, read-only and execute callbacks. Domain completions and execution stay in adapters. AnvaiOps currently documents CDN loading: bundle pinned workers/assets before claiming offline/CSP compatibility. |
| Notebook document and kernel UI | AnvaiOps `apps/control-plane/src/notebook.ts`, `apps/api/src/routes/workspace_notebook.py` | Existing cell/import/export and gated kernel bridge are reference contracts, not production acceptance. Current importer can omit unsupported cells; require explicit diagnostics/preservation policy. Browser bearer query-token support needs review before reuse; do not propagate it as the shared security design. |
| Embedded notebook compute | ProximaDB `clients/python-embedded/src/proximadb_embedded/notebook.py` | Lazy Python plan builder over existing Rust database execution, not a browser notebook editor or kernel manager. Consume through the ProximaDB adapter; do not move this compute path into Victor. |
| Theme and build | ProximaDB `ui/package.json` declares React 17/TypeScript 4/CRA/Material UI 4; AnvaiOps control-plane declares React 18/TypeScript 5/Vite; Victor viewer is plain HTML/JS | A shared React bundle cannot be assumed compatible. Start with semantic tokens and framework-neutral document contracts; qualify peer dependencies, worker paths, CSS isolation and standalone builds before selecting component packaging. ProximaDB's current theme resolves system preference initially but does not expose a persistent system mode. |

These are source observations and migration constraints, not a claim that a
library version caused a production defect. No library or framework upgrade is
approved solely by this comparison; use the existing dependency/security process
and representative compatibility tests.

### Work backward from commercial requirements

AnvaiOps is the integration reference consumer: start with its customer journeys,
then derive public OSS capabilities and acceptance fixtures. At the pinned source,
`docs/HLD_WORKSPACE_UX.md` describes workspace files, SQL and notebooks;
`docs/adr/0019-workspace-coding-platform.md` (Proposed, Revision 1 takes precedence)
describes code navigation and the staged coding workspace;
`docs/adr/0006-per-workspace-persistent-runtime.md` (Accepted) owns isolated runtime
requirements. `docs/adr/0010-tiered-console-architecture.md` is Proposed with
partially implemented deployment phases. These are requirements references,
not proof that every topology, identity choice or runtime is deployed today.
Reconcile later decisions and actual behavior before implementing a requirement;
do not copy old dates, version pins, egress exceptions or performance estimates
into an OSS guarantee.

| Commercial journey | OSS capability to qualify first | AnvaiOps integration responsibility |
| --- | --- | --- |
| Open a workspace, navigate file ↔ symbol, edit and save | Stable document/navigation interfaces, version/conflict handling, read-only/editor capability inputs | Workspace resolution, repository/file adapter, membership and commercial shell |
| Author a workflow, select models, run, approve and inspect audit | Canonical Victor definition/run SDK, accessible editor and recorded outcomes; typed provider/formation capabilities | Product templates, authorized deployment selection and customer operations; no alternative scheduler |
| Query data or run notebook cells and inspect results | ProximaDB query adapter; shared safe document/output components; explicit cancellation/unknown semantics | Scoped data access, isolated workspace execution, resource/egress policy and entitlement enforcement |
| Use the product across standalone and hosted installations | Versioned packages/assets, auth adapter interfaces, offline packaging and observable errors | Managed rollout, tier presentation, billing and integrated product support |

Directionally align **new rich UI components on TypeScript/React**, using the
existing AnvaiOps Vite build as the first host qualification target. Prefer the
existing Monaco investment and React Flow authoring candidate subject to the
measured spike; keep Cytoscape for existing graph viewers unless evidence warrants
replacement. Select compatible supported React/TypeScript/Node versions through
the peer/build matrix, rather than making current React 17 or 18 pins a permanent
standard. Keep tokens and document contracts framework-neutral, lightweight admin
UIs free to retain their existing framework, orchestration in typed async Python
and database/measured native compute in the existing Rust paths. No backend
rewrite or universal frontend rewrite follows from choosing a common editor host.

Every proposed extraction should record: commercial specification/revision and
journey, generic OSS requirement, owning public contract, standalone acceptance,
AnvaiOps adapter acceptance and any intentional divergence. Reject a private-only
requirement from OSS core when a supported adapter serves it. Where an old spec
conflicts with security or measured behavior, update the owning commercial spec
and contract decision before migration. Public FEP/review and licensing gates
still apply; commercial urgency cannot bypass runtime authorization or correctness.

### Component boundaries

| Reusable OSS piece | Contract and owner | Product-specific responsibility |
| --- | --- | --- |
| Semantic theme and basic controls | Reviewed public token schema, accessible controls, light/dark/system/high-contrast adapters; license/provenance and immutable release | AnvaiOps branding/navigation/entitlements; local product appearance preference and supported IDE overrides |
| Document/editor components | Versioned source document, language, diagnostics, read-only state, dirty/save/conflict callbacks; optional Monaco adapter loaded only when needed | Victor logic schemas; ProximaDB SQL completion/query validation; AnvaiOps workspace selection and persistence |
| Workflow canvas and inspector | Controlled nodes/edges/layout plus typed edit/selection/validation callbacks; no browser scheduler or independent formation registry | Victor owns workflow meaning and server validation. Database and code graphs keep domain models; share only proven presentation primitives. |
| Notebook document and safe output renderer | Cells, source, bounded MIME outputs and explicit import/export diagnostics; imported content never executes; HTML/SVG require sanitization or isolation under documented CSP | Kernel creation, credentials, execution permissions, quotas, interrupts and recovery remain in the authorized backend adapter. Victor run cells need not start a Python kernel. |
| Run/query status and audit views | Typed presentation adapters preserve pending, partial, unknown, failed and confirmed outcomes, source IDs and trace links | Each backend owns receipts, cancellation acknowledgement, replay/cursors and tenant/resource checks. A UI completion animation cannot establish committed success. |

Use the generated Victor SDK for Victor operations and the ProximaDB SDK/API
for database operations. Do not create a universal execute endpoint or a second
shared run store to make unrelated semantics look identical. Supply authenticated
API clients from each host; reusable components must not discover tokens, read
cross-origin storage or receive provider credentials. Server capability responses
guide UI affordances but never replace authorization on every operation.

### Incremental adoption and acceptance

1. **Inventory before extraction (VAS-21d):** map actual imports, tests and live
   consumers in both products; distinguish production code from demos. Record
   equivalent behavior and gaps before deleting anything. Retain the current
   viewer and public routes throughout this qualification.
2. **Contracts before package (VAS-21a/21e, VAS-14a):** review the smallest common
   seams with both consumers. Keep framework-neutral contracts/tokens and optional
   editor/canvas/notebook entry points separately loadable. Select the package
   location/name, public license, release owner, API compatibility policy and
   supported peer versions before publication. No sibling filesystem imports,
   private registry dependency, automatic credential sharing or runtime federation.
3. **One proven vertical slice (VAS-14f/21e):** source document → edit → validate
   → save/reload in a Victor fixture and a ProximaDB adapter fixture. Compare
   existing Monaco with CodeMirror only where measured payload, accessibility or
   lifecycle costs justify an alternative; compare canvas libraries on authoring
   needs separately. No wholesale framework migration to obtain visual consistency.
4. **Adopt without forks (VAS-21f/21c):** consume the same immutable OSS release
   in ProximaDB standalone and the AnvaiOps shell. Their repository-owned PRs keep
   product execution/auth adapters and test real endpoint contracts. Remove an
   old component only after parity, upgrade/rollback and supported consumer checks.
5. **Notebook execution last (VAS-14g/21g):** first qualify non-executing notebook
   documents and safe outputs; then opt-in isolated execution, capability/role
   denial, resource bounds, lost connections and truthful interrupt/unknown states.
   Commercial workspace scheduling and entitlements remain AnvaiOps-owned.

Before extraction, freeze representative fixtures and agreed budgets for cold
load, transferred JS, peak heap, editor mount/dispose, graph pan/edit latency and
large output rendering. Measure small/medium/large graphs and notebooks based on
observed customer workloads; publish fixture sizes and hardware, not invented
throughput claims. Reuse existing tests by invariant: shared component behavior
once, each product's API/auth adapter separately, and a small installed integration
matrix. Avoid copying entire test suites into each repository.

Acceptance includes standalone packaged assets with CDN access disabled, worker
and CSP paths, compatible peer dependencies, storage denied, all theme modes,
keyboard-only editing, malicious notebook outputs, bounded output/graph sizes,
mount/dispose leaks, save conflicts and no automatic execution on import. Headed
AgentBrowser evidence must cover real save/reload/run/audit with authorized and
denied roles; synthetic component checks do not close provider, kernel or C5 gates.
Preserve any unsupported browser automation capability as an explicit blocker.

Shared planning and adapter qualification may proceed now. Implementation of new
Victor public contracts still follows FEP-0039 acceptance. Sandhi ownership and
settlement/recovery remain the next execution priority; cross-product UI adoption
does not delay that foundation, C5 or standalone OSS release.

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
