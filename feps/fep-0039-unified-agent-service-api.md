---
fep: "0039"
title: "Unified agent-service API for web, IDE and automation clients"
type: Standards Track
status: Draft
created: 2026-10-07
modified: 2026-10-07
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
---

# FEP-0039: Unified agent-service API

## Summary

Use `victor serve` as the single HTTP composition root for browser, VS Code and
external agent-service consumers. Existing framework services own execution,
authorization, sessions, approvals and recovery. Generate TypeScript contracts
and a shared SDK from authoritative HTTP/event schemas; keep host-specific UI
and secret storage separate.

This proposal is **Draft**. The user authorized planning and tracking; that does
not record formal FEP acceptance or implementation. The [execution tracker](../docs/architecture/victor-agent-service-plan.md)
is the only task-status ledger. The [audit](../docs/development/victor-agent-service-audit.md)
contains the reproduced defects and source inventory. Approval of this design
must be recorded before new public contracts ship. Ordinary defect repairs may
proceed independently through the normal reviewed PR gates.

The primary customer journey is a run that can begin in an editor, pause for an
exact approval, survive disconnection, and resume from the browser without losing
ownership or duplicating an external action. Both clients render recorded service
state. They do not infer business success from an HTTP connection closing, a
model assertion, or a cancellation message being sent.

## Motivation

`victor serve` and `web/server/main.py` expose different chat requests, streams,
session/token behavior and supported operations. VS Code's `streamChat()` matches
the latter, while its broad IDE API targets the former; the current stream body
returns HTTP 422 from the core router. Existing endpoint-presence tests do not
validate payload compatibility. The client also loses paused-run fields, accepts
premature EOF as completion, and overstates cancellation acknowledgement.

The web backend already has useful typed bounded session storage and v1 event
serialization; core already has broad routers and repaired HTTP/WS auth gates.
Preserve those investments. Neither implementation alone establishes safe,
durable, multi-principal agent-as-a-service behavior under concurrent use.

## Boundaries and ownership

| Owner | Responsibility |
| --- | --- |
| Framework application services | Agent/session creation, run transitions, policy, approval and recovery; reuse existing owners |
| Canonical FastAPI application | Transport validation, authentication, owner checks, HTTP status/error mapping, route composition |
| Schema owner | Authoritative HTTP DTOs and event schemas; generated artifacts cannot become competing definitions |
| Shared TypeScript SDK | Validated transport, auth injection, errors, event decoding, cursor/reconnect and cancellation |
| Browser/VS Code adapters | Login/secret storage, workspace integration, rendering and local UI messages |
| Provider layer | Required Sandhi Python binding; optional standalone proxy; inference via InferFlux/ZAI/etc. |

`victor-contracts` remains the definition/plugin package, without HTTP runtime
state. Keep Python's embedded API; it calls the same application services without
an unnecessary network hop. MCP and any retained GraphQL resolve through those
services rather than introducing independent execution/policy owners. Completions
retain a bounded fast path instead of invoking a full agent loop.

Preserve the existing [FEP-0037 classification contract](0037-v1-classify-endpoint.md)
and [FEP-0038 model/reasoning-effort contract](0038-model-reasoning-effort.md).
Client migration must retain explicit model identity, supported reasoning controls
and structured classification semantics rather than creating competing definitions.

## Proposed Change

A reviewed `/v1` resource family exposes capabilities, approved agent definitions,
sessions, runs, run status/results, run events, approval decisions and cancellation.
An agent definition may refer to a single agent, team or workflow; typed variants
retain their semantics. Never accept arbitrary Python import paths from callers.
Peer formations do not require a supervisor. Existing identifiers have one
canonical derivation, and transport IDs remain distinct from logical action IDs.

- Create a durable run with a caller-scoped idempotency key and canonical request
  hash. Matching repeats return the same run; changed content conflicts. Return
  202 only after durable acceptance. Store-and-dispatch uses the existing engine's
  durable start or a transaction/outbox design, not a success response before commit.
- Bind sessions/runs to authenticated principal and workspace scope; constrain
  concurrent turns. Bound admission before costly agent initialization. An HMAC
  or known session ID does not authorize another principal's session.
- Expose recorded run status, result, artifacts, evidence and errors. Preserve
  paused, partial, failed, unknown and cancellation-requested states. Success
  requires validated outcomes, not transport EOF or model confidence.
- Event contracts include version, sequence/event identity and applicable
  run/session/member/action correlation. Reuse existing v1 serialization via
  explicit adapters; version incompatible changes. Replay a retained cursor or
  report a retention gap and recover a snapshot. Telemetry may be lossy; terminal
  run state is durable. Token deltas need an explicit retention policy, not an
  accidental promise to persist every token forever.
- Approval binds the exact action, payload hash, versions and expiry; derive actor
  identity from verified authentication. Recheck permissions/preconditions at
  dispatch. Reconcile unknown effects before continuation. Preserve FEP-0029 and
  G61/G62/G70 ownership rather than building another approval store.
- Cancellation acknowledgement is distinct from completion and from undo.
  Propagate cancellation/deadlines to owned tasks and providers; record effects
  already committed. No blind replay of writes on timeout or process restart.

## Authentication and deployment

Hosted users use configured OIDC with verified issuer, audience, signature and
expiry; principal/role checks apply to HTTP, events and adapters. Browser cookies
need CSRF controls; VS Code uses extension-host authentication and SecretStorage.
Support explicitly scoped machine credentials and an explicit constrained local
mode. Never infer Victor authorization from a Sandhi dashboard login or trust
caller-supplied identity headers from an untrusted proxy.

No reusable credentials in URL queries, webviews, logs or committed evidence.
Validate webview messages, URI/workspace targets and trusted Markdown commands.
Preserve remote-development behavior: the extension host's workspace may not be
on the browser/UI machine. Keep liveness, readiness and privileged administration
separate.

## Migration Path

Migrate the web session/wire implementations into existing canonical routers and
framework owners. Keep `web/server` as a temporary compatibility entry point using
the same implementation; preserve documented old payloads via explicit adapters.
Do not silently switch old callers to a different schema or authentication mode.
Publish capability/schema versions and a client/server compatibility matrix.

Generate HTTP client types from OpenAPI and event types/validators from the same
event schemas. Validate untrusted boundaries; do not spread independent JSON
casts and per-view parsers. Additive events remain forward-compatible; unsupported
major versions fail clearly. Remove legacy protocol adapters, GraphQL surfaces
or dormant commands only after consumer inventory, parity and a deprecation plan.

## Validation and performance

TDD extends existing owners with consumer-provider fixtures and failures before
repairs. Required cases include the 422 mismatch, lost approval fields, false
cancellation success, UTF-8/SSE fragmentation, premature EOF, stale approvals,
cross-owner access, duplicate submissions and crash after committed effects.
Use a real ephemeral server plus deterministic provider for browser and packaged
VS Code smoke tests; fixtures alone cannot establish transport compatibility.
Live model/C5 evidence has separate gates and requires accepted released services.

Measure adapter overhead, first-token visibility, completion latency, memory,
queue wait and cancellation at matched concurrency/workloads before optimizing.
Preserve bounded queues/session caps and avoid locks across initialization or
shutdown. No required Redis/broker, new language, gRPC migration or new event
registry without a measured need. A one-process durable store does not establish
multiworker safety; distributed ownership/leases require their own acceptance.

## Drawbacks and Alternatives

| Alternative | Decision / risk |
| --- | --- |
| Redirect VS Code to the separate web server | Rejected as end state: it lacks the broader implemented IDE API and includes placeholders |
| Two servers kept in permanent lockstep | Rejected: duplicated auth, contracts and lifecycle owners caused the current drift |
| HTTP-to-HTTP bridge in one deployment | Avoid; share service code, keeping one client-to-service hop |
| Universal message object or giant client facade | Avoid; preserve cohesive typed capabilities behind common lifecycle and transport contracts |
| Rewrite everything as GraphQL, gRPC or MCP | Not justified by current measurements; raises migration cost without fixing ownership |
| Big-bang removal of old routes | Rejected; use explicit adapters and a supported-version window |

Risks are tenant/session leakage, duplicate external effects, false success,
compatibility breaks and unmanaged concurrency. Security and UX gates precede
performance tuning. Release, deployment and mixed-team acceptance remain separate
from a source merge. The tracker defines PR-sized milestones and durable evidence.

## Benefits

Users receive the same run identity, approval request and recorded outcome in the
browser, editor and automation client. Moving between those clients no longer
requires choosing between incompatible backend applications or losing session
continuity. A generated contract removes repeated manual synchronization of
payload fields, while real consumer-provider tests detect differences before a
release. Shared policy and lifecycle owners make alternative transports subject
to the same authorization and recovery rules. Bounded queues, admission and
explicit cancellation improve predictable resource use. Keeping embedded Python
access and a single network hop avoids introducing avoidable latency. Existing
session, event and recovery work remains useful rather than being replaced by a
new orchestration stack.

## Compatibility

The current Python framework and plugin-definition APIs remain supported. The
legacy HTTP and web entry points receive explicit compatibility adapters until
their documented deprecation window expires. Existing model identity, reasoning
effort and classification contracts retain their canonical definitions. Schema
versions and capability negotiation distinguish unsupported functionality from
network or authorization failure. Byte-compatible legacy defaults remain the
baseline unless a separately reviewed defect repair changes erroneous behavior.
New authorization modes and durable-run contracts are explicit configuration or
versioned surfaces. Session persistence format, deployment settings and generated
SDK package versions need migration tests, not only source-level type checks.

## Unresolved Questions

- Which existing durable store/transaction boundary will implement run admission
  and reliable dispatch? Resolve in VAS-07 with crash and duplicate-request tests.
- What session ownership/concurrent-turn rules apply to anonymous local mode?
  Resolve in VAS-05/06 without weakening hosted authenticated ownership.
- Which event classes are retained, for how long, and how are expired cursors
  reported? Resolve in VAS-10 with bounded-memory and snapshot-recovery evidence.
- Which external clients depend on legacy HTTP, GraphQL or protocol adapters?
  Resolve via consumer inventory before removal; absence of internal imports is
  insufficient evidence to remove a public interface.
- What absolute latency and resource SLOs should each workload meet? Record a
  measured baseline before accepting performance targets or changing languages.

These questions constrain their dependent increments; they do not authorize
silent fallback or prevent isolated bug fixes and existing framework recovery work.

## Implementation Plan

Use the linked tracker as the single status ledger rather than maintaining a
second checklist here. First land the isolated dependency/activation milestone
and agree the public contract. Reproduce current failures in existing test owners
before changing request or result handling. Then establish authenticated resource
ownership, bounded sessions and durable run admission, preserving existing owners.
Consolidate HTTP composition, generate the shared SDK, and implement event replay
and truthful terminal outcomes. Migrate editor and browser clients with real-server
and packaged-client smoke tests, followed by adapter deprecation and measured
performance acceptance. Each increment is independently reviewed and must pass
all applicable CI gates before squash merge into develop. Main promotion and
release are separate gates. The framework recovery/member-continuation path may
proceed independently toward released-foundation and C5 acceptance; it need not
wait for GraphQL cleanup or either UI migration. Record failed evidence and exact
commit/runtime identities at every material checkpoint.

## References

- [Canonical execution tracker](../docs/architecture/victor-agent-service-plan.md)
- [API audit and reproduced defects](../docs/development/victor-agent-service-audit.md)
- [FEP-0029 durable chat continuation](fep-0029-single-agent-durable-chat-continuation.md)
- [FEP-0037 classification](0037-v1-classify-endpoint.md)
- [FEP-0038 model identity and reasoning effort](0038-model-reasoning-effort.md)
- [Formation gap ledger](../docs/architecture/multiagent-formation-coverage-handoff.md)
- [OpenAPI](https://spec.openapis.org/oas/v3.1.2.html)
- [SSE framing and reconnect semantics](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [HTTP retry semantics](https://www.rfc-editor.org/rfc/rfc9110.html)
