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

## Summary and status

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

## Problem

`victor serve` and `web/server/main.py` expose different chat requests, streams,
session/token behavior and supported operations. VS Code's `streamChat()` matches
the latter, while its broad IDE API targets the former; the current stream body
returns HTTP 422 from the core router. Existing endpoint-presence tests do not
validate payload compatibility. The client also loses paused-run fields, accepts
premature EOF as completion, and overstates cancellation acknowledgement.

The web backend already has useful typed bounded session storage and v1 event
serialization; core already has broad routers and repaired HTTP/WS auth gates.
Preserve those investments. Neither implementation alone establishes safe,
durable, multi-principal agent-as-a-service behavior.

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

## Public contract direction

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

## Compatibility and migration

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

## Alternatives and risks

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
