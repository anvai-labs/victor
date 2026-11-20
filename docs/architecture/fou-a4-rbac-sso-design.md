# FOU-A4 design: RBAC/SSO wired into FastAPI middleware + per-user attribution

Status: Design (no implementation acceptance). Task ledger:
[victor-foundations-plan.md](victor-foundations-plan.md) FOU-A4 row.
Auth design of record for the principal/policy shapes:
[victor-agent-service-auth-policy.md](victor-agent-service-auth-policy.md) (VAS-05a).
Attribution contract: [FEP-0020](../feps/fep-0020-ai-gateway-usage-attribution.md).

Created: 2026-11-19. This document is the A4 pass a follow-up implementation
PR executes without re-litigating scope. All `file:line` references are
against `develop` at `b1d584955`.

## 1. Scope

FOU-A4 = "RBAC/SSO wired into FastAPI middleware; per-user attribution"
(foundations-plan row FOU-A4). Concretely:

1. A **verified principal** contract resolved at the FastAPI auth seam
   (today an opaque `client_id` string), with SSO (Kanidm OIDC) as an
   additional authentication method alongside API keys.
2. **Role/action policy enforcement** at the HTTP/WS route layer, per the
   VAS-05a resource/action policy shape — without displacing the existing
   tool-policy and approval owners.
3. **Per-user attribution**: the verified principal (subject + group) flows
   into the FEP-0020 usage/cost records for every request path.

Out of scope (owned elsewhere): multiuser session ownership/admission
(VAS-06), scoped WS subscription filtering (VAS-05c), browser/VS Code login
UX (VAS-05d), Sandhi-side delegation (VAS-05e/f), commercial SSO/SCIM
(AnvaiOps boundary per FEP-0020's open-core table).

## 2. Inventory — what exists

### 2.1 The live auth seam (FastAPI dependencies, not ASGI middleware)

`VictorFastAPIServer` (`victor/integrations/api/fastapi_server.py:423`):

- **Key map**: `api_keys: Dict[str, str]` (raw key → client_id label),
  constructed at `fastapi_server.py:432,462`; sourced from
  `VICTOR_API_KEYS` env (`victor/ui/commands/serve.py:193`
  `_api_keys_from_env`, consumed at `serve.py:235-246`).
- **Posture gate**: server refuses to start with neither keys nor explicit
  `allow_unauthenticated=True` (`fastapi_server.py:475-486`);
  OpenAPI docs are disabled when keys are configured
  (`fastapi_server.py:532`).
- **HTTP key check**: `_verify_api_key` (`fastapi_server.py:582-591`) —
  `Authorization: Bearer` → dict lookup → client_id, else 401.
- **HTTP dependency**: `_make_auth_dependency` (`fastapi_server.py:598-616`)
  — no-ops on keyless servers, exempts `AUTH_EXEMPT_PATHS`
  (`/health`, `/status`, `fastapi_server.py:596`), and records the resolved
  principal as `request.state.client_id` (VAS-05j seam).
- **WS dependency**: `_make_ws_auth_dependency`
  (`fastapi_server.py:618-637`) — Bearer header or `?api_key=` query
  parameter; `/ws` chat socket is exempt (message-level auth).
- **Router attachment**: `_include_with_auth`
  (`fastapi_server.py:639-686`) splits each router into fresh child routers
  by route class (HTTP / plain WS / guarded WS) so a mixed router gets the
  right guard without mutating shared router objects. All route modules
  (`victor/integrations/api/routes/`, 23 files) plus vertical entry-point
  routers (`_setup_router_plugins`, `fastapi_server.py:733-748`) attach
  through it. This is the **wiring point** for A4 — it is one choke point
  that already reaches every route.
- **WS chat auth**: message-level `auth` frame with constant-time key
  comparison sets `ws.state.authenticated` + `ws.state.client_id`
  (`fastapi_server.py:1044-1060`); unauthenticated turns are refused
  (`fastapi_server.py:1000-1006`).

### 2.2 Owner scoping (VAS-05j/05k, merged)

- `_owner_authorized` (`fastapi_server.py:1068`) — equality check of a
  stored `owner_client_id` against the caller's principal.
- `_broadcast_ws` / `_broadcast_agent_event`
  (`fastapi_server.py:1116,1149`) take an explicit `owner_client_id`;
  keyed servers deliver only to sockets whose `state.client_id` matches.
- Routes bind the caller principal at mutation/list time:
  `victor/integrations/api/routes/tool_routes.py:97-99,129-141`,
  `routes/workflow_routes.py:131,281-297`,
  `routes/agent_routes.py:79-158`; background agents carry
  `owner_client_id` end-to-end (`victor/agent/background_agent.py:114,258`).

This is **ownership equality**, not RBAC: there are no roles, actions, or
grants — only "the principal that created the resource may see/mutate it".

### 2.3 Per-user attribution (FEP-0020 Phase 1, shipped)

- **Context binding**: `auth_subject_id` / `auth_group_id` ContextVars
  (`victor/core/context.py:60-64`), getters
  (`context.py:181-187`), `bind_attribution(subject_id, group_id)`
  (`context.py:192-217`) — scopes identity to the request/task context and
  restores it on exit.
- **HTTP binding**: chat routes call `_verify_api_key` again at handler
  entry and bind `subject_id=client_id`
  (`routes/chat_routes.py:55-62,117,155,223`; same pattern in
  `routes/classify_routes.py:345`).
- **WS binding**: the `/ws` turn binds the authenticated client_id
  (`fastapi_server.py:1009-1018`).
- **Cost records**: `SessionCostTracker.record_request` carries optional
  `subject_id`/`group_id` (`victor/agent/session_cost_tracker.py:92-93,158-159`)
  and resolves `get_auth_subject_id() or self.subject_id` at record time
  (`session_cost_tracker.py:257-258,370-371`).
- **Usage gateway**: `victor/observability/sandhi_meter.py` maps each
  subject to a virtual key `victor:{subject}` and emits the FEP-0020
  neutral usage event per provider call (`sandhi_meter.py:85-92`);
  `UsageGatewaySettings` is default-off with operator-level
  `subject_id`/`group_id` fallbacks
  (`victor/config/groups/usage_gateway_config.py:27-38`); the reverse-proxy
  serve mode meters per virtual key (`victor/observability/gateway_proxy.py`).

**Gap inside the shipped join**: only `subject_id` is ever bound from the
server (`bind_attribution(group_id=...)` has zero callers in
`victor/integrations/`), and the "subject" is the API-key label — an
operator-chosen string in `VICTOR_API_KEYS`, not a verified identity.

### 2.4 Tool policy (exists; NOT API RBAC)

`victor/security/permissions.py`: `PermissionMode` (`permissions.py:48`),
`PermissionPolicy.authorize` (`permissions.py:343`) — tool-name/category
oriented with prompt/allow/deny modes. The VAS-05a design explicitly warns
these semantics "must not be reused as an automatic API superuser"
(auth-policy doc, Policy contract section). A4 does not touch this layer;
its decisions remain authoritative at tool dispatch.

### 2.5 Identity primitives (outbound-only today)

`victor/core/identity/` is a **provider-credential** layer, not inbound
authn: `TokenCredential`/`AccessToken` protocols
(`protocols.py:34-82`), Entra client-credential sources + factory
(`sources.py`, `factory.py:54-116`), and a delegated Entra OIDC sign-in
that resolves a *human* identity via Graph `/me` for HITL approver
attribution (`oidc.py` module docstring). The delegated flow is a useful
in-process precedent but is Entra-specific, Graph-validated (no JWT/JWKS),
and has no server middleware role.

### 2.6 Design-only contracts (VAS-05a; no implementation)

[victor-agent-service-auth-policy.md](victor-agent-service-auth-policy.md)
defines: verified `(issuer, subject)` principal with kind/auth-method/
credential/audience/expiry/scopes/group-provenance; the bounded
bindings/grants YAML shape with `default_effect: deny`, `deny_overrides`,
per-dimension grant matching, credential-scope ceiling; named roles
(`run_operator`, `run_observer`, `approver`, distinct administrators);
401-vs-403 semantics; reauth-on-reconnect leases. **Grep confirms zero
implementation**: no `VerifiedPrincipal` type, no Kanidm/OIDC server code,
no policy parser anywhere under `victor/`. VAS-05a–d are `TODO` in the
VAS tracker (`victor-agent-service-plan.md:3503-3509`).

### 2.7 Dormant seams (FEP-0041-class findings; do not build on blindly)

- `victor/integrations/api/middleware.py` — an **aiohttp** middleware stack
  (`APIMiddlewareStack`, `APIKeyAuthenticator`, rate limiter;
  `middleware.py:208-503`). The aiohttp server was removed
  (`victor/integrations/api/__init__.py:26`); the stack has **no callers**
  outside its own `__init__` re-exports. Its per-client rate limiter and
  key-hash ideas are salvageable but its shape is not the wiring point.
- `ServerSettings.server_session_secret` / `server_session_ttl_seconds` /
  `server_max_sessions` (`victor/config/groups/server_config.py:35-43`) —
  an HMAC session-token scheme that is **configured and tested at the
  settings layer only; never wired into the FastAPI server** (grep: zero
  consumers outside config/tests). `server_api_key` likewise has no
  consumer — the live key source is `VICTOR_API_KEYS` → the `api_keys`
  constructor argument.

### 2.8 Test anchors

`tests/unit/integrations/api/test_api_auth_boundary.py`,
`test_ws_auth_gate.py`, `test_fastapi_chat_attribution.py`,
`test_agent_routes_policy_snapshot.py`, `test_fastapi_server.py`. These are
the suites A4 extends; the WS-broadcast and owner-scope tests from
VAS-05h/05j pin behavior A4 must not regress.

## 3. Gap analysis — A4 needs vs what exists

| A4 need | Today | Gap |
| --- | --- | --- |
| Verified principal at the seam | Opaque `client_id` string from a dict lookup (`fastapi_server.py:582-591`) | No identity type: no issuer/subject/kind/method/expiry/groups. A4a introduces `VerifiedPrincipal` and derives `client_id` from it for compatibility. |
| SSO entry point | Bearer keys only (HTTP + WS); no OIDC anywhere in `victor/` | Kanidm OIDC token verification + browser code-flow login (VAS-05a selected Kanidm on 2026-10-08). A4c. |
| Role/action enforcement | Owner equality only (`_owner_authorized`), plus global key-or-nothing | No policy document, no route→action mapping, no deny/allow decisions. A4d, per the VAS-05a grant shape. |
| Per-user attribution into FEP-0020 events | Shipped, but subject = key label; `group_id` never bound from any caller path | Bind verified principal subject + verified group at the auth seam; thread group into `sandhi_meter` virtual keys and the neutral event. A4b. |
| Middleware wiring point | Per-route-class FastAPI dependencies via `_include_with_auth` — one choke point covering all routes incl. vertical routers | None structural: A4 extends the existing dependency closures rather than adding an ASGI middleware (dependencies already run per-route with the right HTTP/WS split; an ASGI middleware would re-create the WebSocket-scope problem `_include_with_auth` exists to avoid — see its docstring, `fastapi_server.py:642-659`). |
| Browser session semantics | Dormant `ServerSettings` HMAC session knobs, never wired | Either wire the dormant scheme or supersede it with the OIDC session design. Decision D6. |
| Rate limiting per principal | Dormant aiohttp limiter | Out of A4 scope; recorded for the dedup sweep (Decision D8). |

**Layering note**: the auth seam lives in the Client layer
(`victor/integrations/api/`); A4 keeps the principal/policy contracts there
(`victor/integrations/api/auth/`) so no `victor/framework/` public API
changes and no FEP is triggered. Framework/runtime code keeps reading only
the existing `bind_attribution` context vars — unchanged contract.

## 4. Phased implementation plan (one PR per phase)

Conventions per the foundations ledger: RED test first with retained
failed-run evidence, positive + negative test per behavior flip, pre-push
gates (black/ruff/affected tests/collect-only), adversarial review before
push (security work always qualifies). Every phase is behavior-neutral for
existing key-only deployments unless its flag is explicitly set.

### A4a — Principal resolution at the auth seam

**Goal**: every authenticated request carries a `VerifiedPrincipal`, not a
bare label. Key-only posture is byte-identical from the outside.

- New module `victor/integrations/api/auth/principal.py`:
  - `AuthMethod` enum: `api_key`, `oidc` (A4c fills the second value).
  - `PrincipalKind`: `human`, `service`.
  - `VerifiedPrincipal` frozen dataclass: `principal_id` (stable,
    immutable), `kind`, `auth_method`, `credential_id` (public key ID,
    never the secret), `issuer`/`subject` (OIDC only), `groups: tuple[str, ...]`
    (verified provenance only), `expires_at`, `scopes: frozenset[str]`.
  - `principal.client_id` compatibility property = today's label
    (key-map value / issuer-qualified subject), so
    `request.state.client_id` consumers (broadcast owner scoping, route
    owner checks) keep working untouched.
- Key-store upgrade: `VICTOR_API_KEYS` stays the compatibility source;
  add an optional keyed principal registry (`~/.victor/api_principals.yaml`
  or a `server.auth.principals` settings block — Decision D4) mapping
  key → `{principal_id, kind, subject_label, groups}`. Keys absent from the
  registry synthesize a `service` principal whose `principal_id` IS the
  legacy `client_id` — zero-migration for existing operators.
- Wiring: `_make_auth_dependency` and `_make_ws_auth_dependency` resolve a
  `VerifiedPrincipal` and store `request.state.principal` (plus the
  `client_id` shim); `_handle_ws_message`'s auth frame does the same on
  `ws.state`. `_verify_api_key` returns the principal (internal callers in
  chat/classify routes updated in the same PR).
- Rollout flag: none needed — the resolved object is additive; the
  negative test pins that keyless servers and key-only servers produce
  identical HTTP behavior (status codes, headers, broadcast delivery) with
  and without the registry file.

**Tests** (RED-first): unknown key 401 unchanged; exempt paths unchanged;
`request.state.principal.client_id == request.state.client_id` shim;
WS query-param and auth-frame paths both produce principals; concurrent
requests never observe each other's principal (context isolation);
registry-synthesized principal equals legacy label.

**Files**: `fastapi_server.py`, new `auth/principal.py` + `auth/__init__.py`,
`serve.py` (registry load), `routes/chat_routes.py`,
`routes/classify_routes.py` (internal `_verify_api_key` callers).

### A4b — Per-user attribution through FEP-0020 events

**Goal**: the neutral usage event and every `SessionCostTracker` row carry
the verified principal's subject and group.

- Centralize binding: bind `bind_attribution(subject_id=principal.subject_label,
  group_id=principal.primary_group)` inside the A4a auth dependency instead
  of the current per-route re-verification
  (`chat_routes.py:55-62` etc.). Starlette runs dependencies and the
  handler in the same request task, so the binding is request-scoped;
  a dedicated test proves no cross-request leakage under concurrency, and
  the per-route binds become dedup removals (behavior identical).
  - `subject_label` semantics per Decision D7: key-only mode keeps the
    legacy client_id label; OIDC mode uses the immutable Victor principal
    ID derived from `(issuer, sub)` (never email/username).
- Group plumbing: `sandhi_meter.SandhiMeter._ensure_key`
  (`sandhi_meter.py:85-92`) already accepts `group_id` — pass the bound
  group through `SessionCostTracker.record_request`
  (`session_cost_tracker.py:363` mirror call) so virtual keys and the
  neutral event gain verified groups.
- Keyless posture: `bind_attribution()` with nothing stays a no-op;
  `UsageGatewaySettings.subject_id`/`group_id` operator fallbacks
  (`usage_gateway_config.py:33-37`) remain the only attribution — pinned
  by a byte-identical negative test.
- Rollout flag: none (attribution is already active for keyed servers;
  this only upgrades the values). The observable-diff surface is
  usage-event contents, covered by snapshot updates.

**Tests** (RED-first): keyed HTTP chat request produces cost rows + gateway
events with the registry principal's subject/group (positive); keyless
server produces `None` attribution identical to pre-A4 snapshots
(negative); WS turn attribution uses the authenticated frame's principal;
two concurrent keyed requests attribute to their own principals only;
`sandhi_meter` virtual key names include the group when bound.

**Files**: `fastapi_server.py` (dependency binding), `routes/chat_routes.py`,
`routes/classify_routes.py` (dedup), `agent/session_cost_tracker.py`
(group passthrough if missing), `observability/sandhi_meter.py`,
`tests/unit/integrations/api/test_fastapi_chat_attribution.py` (extend).

### A4c — SSO/OIDC entry (Kanidm)

**Goal**: hosted deployments accept Kanidm OIDC access tokens as a first
authentication method producing the same `VerifiedPrincipal`.

- New module `victor/integrations/api/auth/oidc.py`:
  - Discovery: fetch and **pin** the exact issuer + audience from
    `https://<verified-idp>/oauth2/openid/<victor-client>/.well-known/openid-configuration`
    (Kanidm client-specific issuers; never reuse another product's
    registration — auth-policy doc, Kanidm co-design section). Startup
    validation fails closed: misconfigured IdP refuses to serve hosted
    mode rather than falling back to unauthenticated.
  - Token validation: ONE strategy per trusted issuer, configured —
    RFC 7662 introspection (recommended default; authoritative
    revocation/activity) or RFC 9068 JWT profile with pinned
    algorithms/JWKS/issuer/audience/expiry (Kanidm ES256 required in the
    supported set). Reject ID-token substitution, wrong audience, unknown
    algorithms; never fetch caller-supplied issuer/`jku`/`x5u`; bounded
    JWKS/metadata cache lifetimes (auth-policy doc, credential section).
  - Groups: only from verified configured claims, UserInfo `sub` matched
    against the validated token, bounded string arrays; missing/malformed
    groups grant nothing.
- Dependency: `authlib` behind an optional extra `victor[sso]` (Decision
  D2); absent extra + OIDC mode configured = loud startup failure, never
  silent downgrade.
- Modes: `server.auth.mode: api_key | oidc | oidc_and_api_key` — default
  `api_key` (today's behavior). Per the auth-policy credential rules:
  distinct credential namespaces (OAuth access tokens in `Authorization:
  Bearer`, keys in the existing scheme), reject ambiguous multiple
  credentials, **no fallback between methods** — an invalid OIDC token
  never retries as an API key.
- Endpoints: `POST /auth/login` (authorization-code + S256 PKCE, state,
  nonce, exact redirect URIs) and `GET /auth/userinfo` for the browser
  surface, issuing a server-side session (Decision D6) — the API/SDK path
  accepts only access tokens. WS auth frame extended to accept an OIDC
  bearer token alongside the key form.
- Rollout flag: `server.auth.mode` default `api_key` — existing key-only
  users see nothing.

**Tests** (RED-first, all against local fixture JWKS/introspection
doubles — no live IdP in unit CI): valid token → principal with verified
groups; wrong issuer/audience/alg → 401 fail-closed; ID-token substitution
rejected; expired/revoked (introspection `active:false`) rejected; group
claim malformed → no groups, not an error; `oidc_and_api_key` both-methods
ambiguity rejected; key-only server ignores OIDC config entirely
(compatibility negative); discovery failure refuses startup in `oidc`
mode; missing `victor[sso]` extra fails loudly.

**Files**: new `auth/oidc.py`, `auth/__init__.py`, `fastapi_server.py`
(mode resolution + dependency multiplexing + `/auth/*` routes),
`victor/config/groups/server_config.py` (auth mode block), `pyproject.toml`
(extra), `tests/unit/integrations/api/test_oidc_auth.py` (new).

### A4d — Role mapping + policy enforcement

**Goal**: a strict versioned policy document maps verified principals
(via roles) to allowed actions on resources; enforcement at the route
layer with 401/403 semantics; default-off.

- New module `victor/integrations/api/auth/policy.py`:
  - Strict parser for the VAS-05a shape (`version`, `revision`,
    `default_effect: deny`, `combining: deny_overrides`, `bindings`,
    `grants`) — unknown fields rejected, no permissive default.
  - `PolicyDecision` = allow/deny + stable reason code + policy revision +
    matched grant IDs. Policy errors deny.
  - Grant matching is conjunctive across all dimensions (actions × tenant ×
    workspaces × definitions × ownership × authn_methods); no cross-grant
    union; credential scopes are a ceiling.
- Route→action registry: a static table in
  `auth/policy.py` mapping each mounted route (method + path template) to
  an enumerated action (`chat.invoke`, `session.create`, `run.create`,
  `run.events.read`, `artifact.read`, `tool.approve`, `config.admin`, ...).
  Vertical entry-point routers default to a conservative
  `<prefix>.access` action so plugins cannot silently escape the policy.
- Enforcement: `_make_auth_dependency` gains an authorization stage —
  resolve (principal, action, resource context), evaluate, deny ⇒ 403
  (404 concealment for individual object lookups per the auth-policy
  decision contract); owner-equality checks (`_owner_authorized`) remain
  as the ownership dimension's implementation, not a parallel system.
  WS guards authorize at connect and re-check on reconnect frames
  (bounded lease; full per-subscription filtering stays VAS-05c/VAS-06).
- Tool policy and approval owners remain authoritative — API authorization
  never bypasses `PermissionPolicy` or HITL approval (explicit negative
  test).
- Rollout flag: `server.auth.policy_path: Optional[str] = None` — no
  policy file = exactly today's posture (keys gate everything). Setting a
  policy file opts the deployment in; the operator-owned `revision` string
  is audited on every decision.

**Tests** (RED-first): policy file present + authorized request allows;
cross-owner, cross-workspace, cross-definition, cross-action denials for
both auth methods; deny-overrides wins over matching allows; action
unions across grants rejected; malformed/unknown-field policy refused at
startup; no policy file ⇒ byte-identical legacy behavior (negative);
401 vs 403 vs concealed-404 contract; member/narrowed delegation cannot
widen the initiating principal; tool approval still requires the exact
approval owner with policy allow (no bypass).

**Files**: new `auth/policy.py`, `auth/actions.py` (route→action table),
`fastapi_server.py`, `victor/config/groups/server_config.py`,
`tests/unit/integrations/api/test_policy_enforcement.py` (new).

### Sequencing and dependencies

A4a → A4b → A4c → A4d strictly (each phase consumes the previous
principal object). A4a+A4b are Victor-local and carry no new dependencies;
A4c adds the `sso` extra; A4d is config-gated. Ledger rows: record as
FOU-A4a..A4d in the foundations tracker with the standard RED/GREEN
evidence discipline.

## 5. Decision register (owner calls required)

| # | Decision | Recommendation (default) | Notes |
| --- | --- | --- | --- |
| D1 | Auth provider for SSO | Kanidm OIDC — already selected 2026-10-08 (auth-policy doc header); confirm the deployed DS3 registration/issuer/audience before A4c merges | Entra delegated flow (`core/identity/oidc.py`) stays HITL-approver-only |
| D2 | Token validation strategy + library | RFC 7662 introspection default; RFC 9068 JWT profile as config; `authlib` in optional `victor[sso]` extra | Depends on verified Kanidm deployment capabilities (introspection method, ES256); probe before pinning |
| D3 | Role model | Named role bundles per VAS-05a (`run_operator`, `run_observer`, `approver`, distinct admins); no hierarchy, no ADMIN shortcut reuse from tool policy | Policy contract section of the auth-policy doc |
| D4 | Principal/key registry storage | Versioned YAML file (`server.auth.principals` path or `~/.victor/api_principals.yaml`), operator-managed immutable revision, not DB rows | Matches the policy document's revision discipline; avoids a settings-DB schema commitment |
| D5 | Attribution subject identity under OIDC | Immutable Victor principal ID derived from verified `(issuer, sub)`; email/username display-only | Auth-policy: caller names/emails are not keys |
| D6 | Browser session semantics | Server-side session (Secure/HttpOnly cookie, CSRF + Origin checks) issued by `/auth/login`; decide persistence home (project DB vs signed stateless cookie) — recommend DB-backed for revocation | The dormant `ServerSettings` HMAC knobs are the natural seam to wire or formally retire; owner chooses wire-vs-retire |
| D7 | `client_id` compatibility lifetime | Keep `client_id` as the key-only attribution label indefinitely; OIDC principals get issuer-qualified IDs; no rename of existing fields in A4 window | Protects dashboards/sandhi virtual-key continuity |
| D8 | Dormant `victor/integrations/api/middleware.py` aiohttp stack | Do not port into A4; schedule deletion in a duplication-sweep PR (its rate limiter idea may be re-homed later as a FastAPI dependency) | Zero callers today; keeping it invites the dormancy trap |
| D9 | WS subscription-granular policy | Defer per-subscription scoping to VAS-05c/VAS-06; A4d enforces at connect + reconnect only | Scoped subscription records are an existing open VAS item |
| D10 | Default posture flips | None in A4: mode stays `api_key`, policy off, attribution values change only for deployments that adopt the registry/OIDC | Key-only users are behavior-neutral by construction |

Decisions D1/D2/D6 block A4c only; A4a/A4b are executable immediately
after this design lands; A4d needs D3/D4 confirmed before merge.

## 6. Risks and non-goals

- **Do not weaken the existing gates**: the `_include_with_auth` child-router
  split, the keyless no-op short-circuit, and the owner-scoped broadcast
  admission are load-bearing (documented adversarial-review findings);
  A4 extends them in place and keeps their negative tests green.
- **No second user directory, no expression language, no second tool
  dispatcher** (auth-policy doc constraints). OPA/Cedar stays optional
  future work behind the same decision contract.
- **Attribution ≠ authorization**: FEP-0020 events carry identities for
  metering; A4d policy decisions are the only authorization authority, and
  an attribution join failure must never fail the request path (best-effort
  metering rule from FEP-0020 Phase 2 is preserved).
- Sandhi-side delegation (initiator/actor/provider-credential separation)
  remains VAS-05e/f; A4 only guarantees the Victor-side principal is real.

## 7. References

- [FEP-0020 — AI usage gateway](../feps/fep-0020-ai-gateway-usage-attribution.md)
- [Agent-service auth policy (VAS-05a design)](victor-agent-service-auth-policy.md)
- [Victor Foundations Plan — FOU-A4 row](victor-foundations-plan.md)
- [Victor agent-service plan — VAS-05 rows](victor-agent-service-plan.md)
- Kanidm OAuth2 integration and custom claims docs (see auth-policy
  Standards section)
