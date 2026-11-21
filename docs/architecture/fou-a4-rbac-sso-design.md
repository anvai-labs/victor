# FOU-A4 design: RBAC/SSO wired into FastAPI middleware + per-user attribution

Status: Design (no implementation acceptance). Task ledger:
[victor-foundations-plan.md](victor-foundations-plan.md) FOU-A4 row.
Auth design of record for the principal/policy shapes:
[victor-agent-service-auth-policy.md](victor-agent-service-auth-policy.md) (VAS-05a).
Attribution contract: [FEP-0020](../feps/fep-0020-ai-gateway-usage-attribution.md).

Created: 2026-11-19. Updated: 2026-11-21 (owner-directed evolution: Sentinelpass-backed
credentials, TLS, passkey — §5). This document is the A4 pass a follow-up
implementation PR executes without re-litigating scope. `file:line` references are
against Victor `develop` at `3af58a89f` and the local Sentinelpass checkout
(`~/code/sentinelpass`, Apache-2.0) at `52079eb`.

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
  add an optional keyed principal registry — **Decision D4 blessed
  (2026-11-20, modified recommendation)**: versioned YAML, operator-managed,
  default `/etc/victor/api_principals.yaml`, override
  `server.auth_principals_path`; the resolved path must live OUTSIDE every
  sandbox-writable surface (enumerated by
  `victor.tools.sandbox.writable_surface_roots`: workspace root, Victor
  state root `~/.victor`, temp dirs, operator allowed mounts) and the
  server refuses to start with a typed
  `PrincipalsRegistryPathError` otherwise — the agent must not be able to
  rewrite its own principals (the exact Codex finding for
  `~/.victor/api_principals.yaml`). Schema: top-level monotonic integer
  `revision`; per entry `key` (unique), `principal_id` (unique, becomes
  `client_id`), `kind: api_key` (only value in the A4a window; A4c adds
  `oidc`), `subject_label` (FEP-0020 subject), `groups` (unique strings),
  optional `expiry` (expired entries fail closed with the same 401 as
  unknown keys). Strict parsing: unknown fields rejected. Loading is
  read-only; activation is process restart or the explicit
  `reload_principals_registry()`; operators publish updates via atomic
  rename (write temp, `os.replace`) so multi-worker deployments converge
  on the new revision. Keys absent from the registry synthesize a
  `service` principal whose `principal_id` IS the legacy `client_id` —
  zero-migration for existing operators.
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

A4a → A4b (merged, #1993) → the A4c family → A4d (each phase consumes the
same `VerifiedPrincipal` object). The A4c family now has two independent
tracks: the owner-directed §5 phases **A4c-1 → A4c-2 → A4c-3**
(Sentinelpass credentials → TLS → passkey; A4c-3 hard-depends on A4c-2
because browser WebAuthn needs a secure context), and the Kanidm OIDC
phase (§A4c above, unchanged) which interleaves once D1/D2/D6 are decided.
A4d remains gated on D3 and enforces per-principal regardless of auth
method, so passkey/OIDC principals need no policy-model change. Ledger
rows: record as FOU-A4a..A4d (plus A4c-1/-2/-3) in the foundations tracker
with the standard RED/GREEN evidence discipline.

## 4.1 Implementation status (2026-11-20; updated 2026-11-21)

- **A4a + A4b MERGED** into `develop` as PR **#1993** (commit `f0d85b2f6`):
  principal shape as designed, resolved inside the existing dependencies
  (no ASGI middleware); the `client_id` shim preserved; keyless/exempt
  behavior byte-identical (pinned); attribution centralized in the
  dependency with the `/ws` turn binding upgraded to principal
  subject+group; FEP-0020 best-effort rule pinned by tests; dormancy guard
  via orchestrator- and provider-observed identity (streaming span
  included).
- **D4 registry implemented** per the (then) blessed decision above
  (`auth/principals_registry.py` + `server.auth_principals_path` +
  startup invariant over `writable_surface_roots`). The 2026-11-21
  owner-directed evolution (§5) revises D4: the YAML registry becomes the
  FALLBACK once A4c-1 lands; until then it stays the primary, unchanged.
- Reconciliation with #1981: the single-trusted-operator posture claim
  (`claim_shared_state`) now consumes the dependency-set
  `request.state.client_id` shim instead of re-calling `_verify_api_key`
  per route; posture semantics unchanged (enforced by the
  `test_multi_principal_refusal` suite).
- A4c (D1/D2/D6), A4c-1/-2/-3 (D4-revised/D11/D12), and A4d (D3) remain
  pending their decisions / implementation.

## 5. Sentinelpass-backed credentials, TLS, and passkey authentication (owner-directed evolution)

**Owner directive (2026-11-21)**: use [Sentinelpass](https://github.com/anvai-labs/sentinelpass)
— the operator's local-first, zero-knowledge password manager — as the
credential/principal store for Victor's server auth (API keys AND
TLS certificates/private keys), add comprehensive TLS 1.2/1.3 for the
server endpoint, and add passkey (WebAuthn/FIDO2) authentication with
Victor as the relying party/callback where feasible, so users can
authenticate to Victor's API with a passkey. This section is the design of
record for those capabilities; phases **A4c-1/-2/-3** below slot into the
A4 sequencing (§4) without disturbing the merged A4a/A4b behavior
(#1993) or A4d's gating. This is docs-only: no behavior changes without
each phase's explicit flags.

### 5.1 What Sentinelpass actually provides (verified against `~/code/sentinelpass` at `52079eb`)

- **Surfaces**: CLI (`sentinelpass`), background daemon, desktop UI,
  browser extension, sync relay (`README.md:15`). Apache-2.0, Rust.
- **Secrets broker for local tools** (`README.md:65-84`): tools
  authenticate with a per-client token and may only touch exactly the
  `client × domain × field` scopes granted. Surfaces: `secret allow`,
  `secret get`, `secret exec` (env injection into a child process),
  `secret list`, `secret audit`, `secret token mint/rotate/revoke`
  (revocation fail-closed).
- **Grant shape** — `ExternalSecretGrant`
  (`sentinelpass-core/src/external_secret_access.rs:26-35`): `client_id`,
  `domain`, `field: ExternalSecretField`, `expires_at`, `allow_write`.
  Scope matching is **exact triple equality — no wildcards**
  (`matches_scope`, `external_secret_access.rs:42-44`). `client_id` is
  normalized to lowercase ASCII alphanumerics plus `-`/`_`
  (`external_secret_access.rs:478-490`); `domain` is a host/service key
  where `/` and whitespace are rejected (`external_secret_access.rs:492-500`).
  Victor therefore uses `client_id = "victor"` and service domains such
  as `victor-auth` / `victor-tls`.
- **Field types are narrow**: `ExternalSecretField` has exactly three
  variants — `Username | Password | Title`
  (`sentinelpass-protocol/src/message.rs:9-13`). There is **no
  certificate/private-key field type**; PEM material rides the
  `password` field (typed upstream proposal recorded, §5.7).
  `CredentialType` is `Password | ApiKey | PasskeyReference`
  (`sentinelpass-core/src/vault/mod.rs:54-59`).
- **Client tokens**: `spt_`-prefixed, shown once at mint, stored only as
  SHA-256 (`external_secret_access.rs:345-366`); rotation kills the old
  token immediately (`:369-378`); **revocation is fail-closed** — a
  revoked client is denied even with its old token and never degrades to
  token-less legacy (`:57-68`, `:382-389`); constant-time verification
  (`:414-429`). The allowlist file itself is born-0600, refuses to load
  or write through symlinks, and fails closed on loose permissions
  (`:87-168`).
- **Daemon + IPC**: every secret fetch is allowlist-scoped and
  token-enforced under a client id (`get_secret_from_daemon`,
  `sentinelpass-cli/src/commands/secret.rs:70-105`); the CLI can request
  biometric unlock when the daemon is locked (`secret.rs:41-68`).
  **Exact-entry retrieval**: `VaultOp::ServiceGetSecret` serves "one field
  of one EXACT entry under a service grant (client token + exact entry_id
  + field). Never step-up"
  (`sentinelpass-protocol/src/service.rs:400-408`); grants are minted via
  `sentinelpass service grant` with optional expiry and optional SP-3
  executable pins (`main.rs:747-772`) — an SP-3/ADR-016 grant may pin
  SHA-256 digests of approved client binaries resolved from the kernel's
  `/proc/<pid>/exe`, denying when evidence is unavailable
  (`sentinelpass-core/src/daemon/exe_policy.rs`).
- **Vault model**: zero-knowledge at rest — Argon2id (t=3, m=256MB, p=4;
  `SECURITY_ARCHITECTURE.md:83`), AES-256-GCM, DEK-wrapped entries;
  biometric unlock stores OS-protected DEK material, never the master
  password (`SECURITY_ARCHITECTURE.md:76`); the Windows Hello-bound wrap
  requires a Hello-gated TPM signing operation to recover the DEK
  (`sentinelpass-core/src/biometric_hello.rs`). One daemon per vault
  (`README.md:206`).
- **Audit**: per-request daemon audit with purpose labels (`--purpose`,
  `secret.rs`), queryable with `sentinelpass secret audit --client-id
  victor [--failures-only]`.
- **Passkeys today are metadata references only**: `sentinelpass passkey
  add/list/get/delete` store `PasskeyReferenceMetadata` (RP ID, account
  label, platform, credential-id hint, sync source) and state "This is
  metadata only. Authentication remains with the platform
  authenticator." (`sentinelpass-cli/src/commands/passkey.rs:103-105`,
  `:195-198`). Actual passkey **private-key custody is an explicitly
  DEFERRED upstream roadmap item** (`ROADMAP.md:151`, guarded by "Passkey
  reference records remain metadata-only until a separate authenticator
  or credential-provider architecture is accepted and reviewed",
  `ROADMAP.md:153`). The `custody/` module (KC-01/02) exists but exposes
  no production enrollment constructor
  (`sentinelpass-core/src/custody/mod.rs:3-5`).

### 5.2 A4c-1 — Sentinelpass as the principal store (revises Decision D4)

**Decision (D4, revised 2026-11-21)**: the Sentinelpass vault becomes the
PRIMARY store of Victor's API-key credentials once `server.auth_credential_source:
sentinelpass` is set; the #1993 YAML registry
(`victor/integrations/api/auth/principals_registry.py`) becomes the
FALLBACK. Default source stays `yaml` until an operator opts in.

**Provisioning shape.**

- Each Victor principal is one vault entry (`CredentialType::api_key`):
  `title` = `principal_id`, service domain `victor-auth`, the raw API key
  in the `password` field.
- Authorization is one Sentinelpass **service grant** per entry:
  `client_id=victor`, exact `entry_id`, `fields=["password"]`, optional
  `expires_at`, and — recommended for the server deployment — an
  `--exe-sha256` pin of the server binary so the grant is bound to the
  actual `victor` executable, not merely to the client token.
- The YAML registry's role changes from secret store to **non-secret
  digest index**: per entry `key_digest` (the `sha256:<16hex>` form
  already used by `VerifiedPrincipal.credential_id`,
  `victor/integrations/api/auth/principal.py:129-135`), `principal_id`,
  `kind`, `subject_label`, `groups`, `expiry`,
  `sentinelpass_entry_id`. Strict parsing, uniqueness checks, and the
  placement invariant are kept — see below for why the invariant
  survives.

**Startup invariant: partially moot, and that is the security upgrade.**
The original D4 invariant (registry must live outside every
sandbox-writable surface, typed `PrincipalsRegistryPathError` otherwise)
existed because a file-based registry holding RAW keys inside
agent-writable state would let the agent rewrite its own principals. With
Sentinelpass-primary the secret material is no longer in any file: the
vault is zero-knowledge (Argon2id/AES-256-GCM), unlocked only by the
operator's master password or a biometric gesture, lives in the
Sentinelpass config dir rather than `~/.victor`, and grants are
operator-minted and revocable — the agent can neither read nor mutate
granted credentials. **The invariant therefore becomes moot for secret
material but is retained for the digest index and the YAML fallback**:
the index still carries authorization metadata (`groups`, `expiry`,
principal identity), so an agent-writable index would still let the agent
grant itself groups. Placement enforcement stays unconditional on the
resolved index/fallback path.

**Request-time resolution (the ordering problem, honestly).** A presented
bearer key must map to an entry without Victor storing raw keys. The
chain:

1. digest the presented key (SHA-256) → look up the digest in the index
   (unindexed digest = 401 immediately; no vault call for unknown keys);
2. fetch the entry's `password` from the vault through the service
   grant (exact-entry lookup);
3. constant-time compare the fetched key against the presented key;
4. build the `VerifiedPrincipal` (`auth_method=api_key`,
   `principal_id`/`subject_label`/`groups`/`expires_at` from the index)
   and drop the fetched secret from memory. The secret is never logged,
   cached to disk, or emitted in errors.

One vault round trip per authenticated request; unknown-key requests cost
nothing. Recorded upstream enhancement (§5.7): a digest-only
`ServiceVerifySecret` op would keep the secret vault-side entirely;
fetch-and-compare is the honest minimum until then. Caching is **off by
default** (revocation latency ≈ zero); `server.auth_sentinelpass_cache_ttl_seconds`
(default `0`) lets high-QPS deployments opt into a TTL-bounded window
explicitly.

**Lookup transport: CLI subprocess, not direct IPC.** The
`sentinelpass` CLI is the documented public surface for exactly this use
("least-privilege secrets broker for developer tools (AI agents, proxies,
scripts)", `README.md:65-67`). The daemon IPC is a Rust-native serde
envelope with its own enrollment/token machinery; a second external
protocol client would be an unreviewed trust-boundary contribution, so
Victor shells out via argv (no shell), with a bounded timeout, the client
token taken from `SENTINELPASS_CLIENT_TOKEN` or a 0600 token file
(`server.auth_sentinelpass_token_file`), and stderr captured but never
emitted above DEBUG (logging-level policy). Direct daemon IPC is recorded
as future work for when/whether Sentinelpass declares its wire protocol
stable for third-party clients.

**Typed degradation matrix (fail-closed discipline).**

| Condition | Behavior |
| --- | --- |
| Daemon not running / vault locked | Use the **startup-loaded** YAML fallback snapshot if one is configured (structured WARNING; per-request cost is one bounded failed lookup). No fallback configured → 401. No biometric prompt exists for a headless server; operators run the daemon unlocked or accept degraded operation explicitly. |
| Grant expired / client token revoked / allowlist deny | **FAIL CLOSED — never degrade.** These are operator decisions (`revoke_client_token` is fail-closed upstream, `external_secret_access.rs:380-389`); silently falling back to YAML after a revocation would defeat the revocation. |
| Digest indexed but entry/grant missing | 401 + audit-visible typed error (`SentinelpassDenied` vs `SentinelpassUnavailable` are distinct types so the matrix is enforceable). |
| `yaml` source selected | Today's byte-identical behavior (pinned by negative test). |

**Tests** (RED-first): granted key resolves to the correct principal
including groups; locked-vault condition falls back only when a fallback
snapshot exists; revoked-token denial does NOT fall back (negative);
expired grant = 401; digest mismatch = 401; absent fallback + daemon down
= 401; `yaml` mode byte-identical; secret absent from logs/metrics/errors
(redaction sweep); subprocess invoked without a shell and with timeout.

**Files**: new `auth/sentinelpass.py` (CLI wrapper, typed errors, index→
principal resolver), `auth/principals_registry.py` (index schema: `key`
→ `key_digest` + `sentinelpass_entry_id`, backward-compatible parsing of
full-key entries for fallback mode), `fastapi_server.py` (source
selection in `_verify_api_key`, `fastapi_server.py:857`),
`victor/config/groups/server_config.py` (`auth_credential_source`,
`auth_sentinelpass_*` settings),
`tests/unit/integrations/api/test_sentinelpass_credentials.py` (new).

### 5.3 A4c-2 — TLS 1.2/1.3 with vault-held certificates

- **Cert material in the vault**: the leaf certificate (public PEM) and
  the private key (PKCS#8 PEM) are two vault entries under service domain
  `victor-tls`, served to the server at startup through the same
  authorized-lookup machinery as A4c-1 (separate grants; the private-key
  grant SHOULD pin the server binary via `--exe-sha256`). Honest
  constraint: there is no certificate/key field type upstream
  (`ExternalSecretField` = `Username|Password|Title`,
  `sentinelpass-protocol/src/message.rs:9-13`), so PEM text rides the
  `password` field; entry encryption (AES-256-GCM) protects it
  regardless of the type label. A `CredentialType::Certificate` upstream
  proposal is recorded (§5.7).
- **Alternative launch shape**: `sentinelpass exec --client-id victor
  --env VICTOR_TLS_KEY=victor-tls-key -- victor serve ...` injects
  values as child-process environment (`README.md:78-80`) — documented as
  an operator alternative, not the default: the in-process startup lookup
  is testable without wrapping the process tree, and env-planted secrets
  would be inherited by every tool subprocess Victor spawns.
- **Fallbacks**: filesystem paths (today's posture: `server.tls.cert_path`/
  `key_path`) and OS keyring (via `python-keyring`, optional). Source
  precedence is explicit per config — no silent probing order.
- **Termination choice (Decision D12)**: uvicorn native TLS. Both
  `uvicorn.Config` construction sites (`victor/ui/commands/serve.py:250`,
  `serve.py:416`) and the server-managed run path
  (`fastapi_server.py:1850-1859`) receive a Victor-constructed
  `ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)` rather than uvicorn's ad-hoc
  `ssl_*` kwargs, so the policy below is enforced in one auditable
  builder. rustls termination would require replacing the ASGI server
  stack (uvicorn is OpenSSL-backed through Python `ssl`; it has no
  rustls option) — rejected for A4c-2 as a larger trust-boundary change
  than a policy-pinned SSLContext; Granian (rustls) is recorded as a
  future alternative if OpenSSL CVE posture ever drives the decision.
- **Policy**: minimum **TLS 1.2** (`minimum_version = TLSv1_2`);
  **TLS 1.3 preferred** and negotiated automatically where both peers
  support it (TLS 1.3 suites are OpenSSL-managed, not operator-tunable);
  TLS 1.2 cipher string pinned to forward-secret AEAD suites only:
  `ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256`.
  HSTS (`Strict-Transport-Security: max-age=63072000; includeSubDomains`)
  is set by a small HTTP dependency whenever TLS is enabled. OCSP
  stapling deferred (uvicorn limitation, recorded).
- **Config**: `server.tls.enabled` (default `false`),
  `server.tls.cert_source: sentinelpass|file|keyring`,
  `server.tls.min_tls_version` (validated to the {`1.2`, `1.3`} set),
  `server.tls.hsts_enabled` (default `true` when TLS is on).
- **Rotation**: restart-bounded in v1. uvicorn materializes the ssl
  context at config time and does not support hot context swap; this
  matches A4's registry activation semantics (restart or explicit
  reload). Runbook: rotate the vault entries, restart. Expiry monitoring
  is operator-side (follow-up recorded).
- **Attribution unchanged**: TLS is transport; the FEP-0020 join remains
  verified-principal → usage events (§A4b;
  `fastapi_server.py:1012,1486`). mTLS client certificates as an
  authentication method are explicitly out of A4c-2 scope and recorded —
  they would enter as a new `AuthMethod` consuming the same
  `VerifiedPrincipal` shape.
- **Tests**: SSLContext assertions (minimum version, TLS 1.2 set
  restricted to the pinned AEAD suites, TLS 1.3 left to OpenSSL); cert
  source matrix (vault fetch / file / keyring) with failure modes typed;
  HSTS present only when TLS is on; default-off byte-identical startup.

**Files**: new `victor/integrations/api/tls.py` (context builder + cert
source resolution, reusing the A4c-1 lookup client), `serve.py` (both
uvicorn sites), `server_config.py` (`server.tls` block),
`fastapi_server.py` (HSTS dependency), `tests/unit/integrations/api/test_tls_context.py`
(new).

### 5.4 A4c-3 — Passkey (WebAuthn/FIDO2) authentication, Victor as RP

**Feasibility verdict (honest).**

- Victor-as-RP with **platform authenticators** (iCloud Keychain, Windows
  Hello, Google Password Manager, roaming security keys): **feasible
  now**. WebAuthn ceremonies run between the user agent and Victor's
  server; Victor never handles passkey private keys.
- **Sentinelpass-held passkeys** (vault-side private-key custody): **not
  feasible today**. Private-key custody is an explicitly deferred
  upstream roadmap item (`ROADMAP.md:151`, guarded at `:153`), passkey
  entries are metadata-only references (`passkey.rs:103-105`), and the
  custody module exposes no production enrollment constructor
  (`custody/mod.rs:3-5`). This spec reserves the seam for it
  (`PasskeyCredentialStore`, below) and uses platform authenticators
  meanwhile. Naming trap: Sentinelpass's `biometric_hello`/Touch ID work
  (`biometric_hello.rs`) is **vault unlock**, not WebAuthn — do not
  conflate the two.
- **"Victor as the callback/RP"**: feasible and is exactly the v1 design
  — Victor hosts `/auth/passkey/*`, acts as relying party for the
  registration (attestation) and authentication (assertion) ceremonies,
  and issues a session after a verified assertion.

**Library evaluation.**

- `python-fido2` (Yubico) — **selected**. Actively maintained (v2.2.1,
  released 2026-06-29); provides a full WebAuthn RP implementation
  (`fido2.server.Fido2Server`) with registration/assertion verification,
  strict RP-ID/origin enforcement, attestation-policy handling, and
  first-class credential-id (non-resident) flows. Rationale: continuous
  security maintenance from the FIDO reference vendor and fail-closed
  verification defaults that match Victor's posture; also covers the
  exact v1 scope (server-side credential-id registration, no
  resident-key discovery).
- `py_webauthn` (duo-labs) — simpler, higher-level RP API, but the weaker
  maintenance signal of the two in 2026; positioned for prototyping.
  Rejected for v1; revisit if `python-fido2`'s API friction dominates
  real cost.

Behind optional extra `victor[passkey]` (mirroring the D2 `victor[sso]`
discipline): absent extra + passkey configured = loud startup failure,
never silent downgrade.

**Design.**

- New `AuthMethod.PASSKEY = "passkey"` (enum extension in
  `auth/principal.py`; `VerifiedPrincipal` is unchanged —
  `principal_id`, `groups`, `expires_at` all apply).
- **Credential storage is server-side public-key material, NOT the
  vault**: a `PasskeyCredentialStore` protocol backed by the project DB
  (consistent with the D6 leanings) maps `credential_id → principal_id`,
  public key, sign count, user-verified flag, transports,
  created/last-used. The RP registry must be queryable at assertion time
  without the operator unlocking a vault; isolating it behind the
  protocol keeps a future Sentinelpass-custody authenticator free to
  implement the assertion side when upstream custody ships.
- **Endpoints** (browser surface only; the API/SDK path stays API-key /
  OIDC): `POST /auth/passkey/register/start|finish` (attestation
  ceremony) and `POST /auth/passkey/assert/start|finish` (assertion
  ceremony). Registration requires an EXISTING authenticated principal
  (session or API key) or a one-time operator enrollment token —
  anonymous self-registration is refused. On a verified assertion the
  principal resolves from `credential_id`, and a session is issued in
  the D6 server-side session shape (Secure/HttpOnly cookie, CSRF + Origin
  checks); the WS auth frame gains a session-token form.
- **Secure-context requirement**: browsers restrict WebAuthn to HTTPS
  (or localhost), so passkey mode **hard-requires A4c-2 TLS** on any
  non-loopback bind — loud startup refusal otherwise.
- **v1 scope honestly bounded**: NO resident-key/discoverable-credential
  discovery flows; single-use challenges with bounded ceremony TTL and
  sign-count/clone detection enforced; `user_verification="required"`
  (the passkey is the primary factor, not a second factor); attestation
  conveyance = "none" with format enforcement optional-off (platform
  authenticators attest anonymously; metadata enforcement is recorded as
  future hardening); programmatic (non-browser) assertion out of scope.
- **Origin/RP-ID pinning**: RP ID = the deployed hostname; allowed
  origins = an explicit exact-match config list; any mismatch rejects
  the ceremony (D11).
- **Route wiring note**: `/auth/passkey/assert/*` must be reachable
  pre-authentication by design; it is added to the explicit
  `AUTH_EXEMPT_PATHS` enumeration (`fastapi_server.py:596` semantics),
  never a blanket dependency bypass, and `/auth/passkey/register/*`
  stays guarded.
- **Tests** (RED-first, fixture authenticator doubles / `python-fido2`
  ceremony vectors): register→assert round trip issues a session;
  wrong origin / wrong RP-ID / replayed challenge / missing-UV assertion
  rejected; anonymous registration refused; passkey principal flows into
  attribution and owner scoping unchanged; disabled mode byte-identical;
  missing `victor[passkey]` extra = loud failure; TLS-off on non-loopback
  = loud refusal.

**Files**: new `auth/passkey.py` (RP config + ceremony handlers over
`Fido2Server`) and the `PasskeyCredentialStore` protocol + project-DB
implementation, `auth/principal.py` (enum value), `fastapi_server.py`
(route registration through `_include_with_auth` + exempt-path entry),
`server_config.py` (`server.auth.passkey` block),
`tests/unit/integrations/api/test_passkey_auth.py` (new).

### 5.5 Phase summary (A4c family)

| Phase | Goal | Default-off flag | Depends on | Decision |
| --- | --- | --- | --- | --- |
| A4c-1 | Sentinelpass API-key lookup (vault primary, YAML fallback) | `server.auth_credential_source: yaml` | none | D4 revised |
| A4c-2 | TLS 1.2/1.3 + vault-held certs | `server.tls.enabled: false` | A4c-1 lookup machinery | D12 |
| A4c-3 | Passkey WebAuthn RP | `server.auth.passkey.enabled: false` | A4c-2 (secure context) | D11 |
| A4c (§4, unchanged) | Kanidm OIDC SSO | `server.auth.mode: api_key` | D1/D2/D6 | D1/D2/D6 |
| A4d (§4, unchanged) | Roles + policy enforcement | `server.auth.policy_path: None` | D3 | unchanged |

### 5.6 Security posture notes (honest limits)

- **What the vault does and does not protect**: it removes secret
  material from at-rest files and makes credentials operator-unlockable
  and revocable; it does not protect secrets once fetched into the Victor
  server process. An attacker with code execution inside the server
  process can still read process memory — the boundary is against at-rest
  exfiltration and agent-rewritable config, which is strictly stronger
  than the YAML status quo (keys at rest forever vs. in memory only for
  the comparison window). Victor's existing log/metric redaction
  discipline applies unchanged.
- **Availability dependency**: server auth now optionally depends on the
  operator's Sentinelpass daemon being up and unlocked. The YAML fallback
  plus the typed degradation matrix is the availability answer; the
  fail-closed denial rules are the security answer.
- **No new authorization authority**: Sentinelpass grants authorize
  *credential lookup only*; every authorization decision remains with
  A4d policy + owner scoping (attribution ≠ authorization rule, §6/§7).

### 5.7 Upstream (Sentinelpass) items recorded, not required

- `CredentialType::Certificate` / a dedicated PEM field type (cleaner
  TLS entry typing than password-field PEM).
- A digest-only `ServiceVerifySecret {entry_id, sha256}` op so the
  auth path never fetches the secret.
- A stable third-party client contract for the daemon IPC protocol
  (would replace the CLI subprocess transport).
- Passkey private-key custody and synchronization (`ROADMAP.md:151`) —
  enables the future Sentinelpass authenticator behind
  `PasskeyCredentialStore`.

## 6. Decision register (owner calls required)

| # | Decision | Recommendation (default) | Notes |
| --- | --- | --- | --- |
| D1 | Auth provider for SSO | Kanidm OIDC — already selected 2026-10-08 (auth-policy doc header); confirm the deployed DS3 registration/issuer/audience before A4c merges | Entra delegated flow (`core/identity/oidc.py`) stays HITL-approver-only |
| D2 | Token validation strategy + library | RFC 7662 introspection default; RFC 9068 JWT profile as config; `authlib` in optional `victor[sso]` extra | Depends on verified Kanidm deployment capabilities (introspection method, ES256); probe before pinning |
| D3 | Role model | Named role bundles per VAS-05a (`run_operator`, `run_observer`, `approver`, distinct admins); no hierarchy, no ADMIN shortcut reuse from tool policy | Policy contract section of the auth-policy doc |
| D4 | Principal/key registry storage | **REVISED 2026-11-21 (owner-directed; supersedes the raw-key portion of the 2026-11-20 blessing, which is retained as the fallback)**: Sentinelpass vault is the PRIMARY credential store (`server.auth_credential_source: sentinelpass`) — one `CredentialType::api_key` vault entry per principal, one exact-entry service grant per entry (`client_id=victor`, `entry_id`, `fields=["password"]`, optional expiry + `--exe-sha256` binary pin); the #1993 YAML registry becomes a **non-secret digest index** (`key_digest`/`principal_id`/`kind`/`subject_label`/`groups`/`expiry`/`sentinelpass_entry_id`) and the FALLBACK when the vault is unavailable; default source stays `yaml` (byte-identical) until an operator opts in. The startup placement invariant is RETAINED for the index/fallback file (it carries groups/expiry authorization metadata — versioned YAML at `/etc/victor/api_principals.yaml` default, `server.auth_principals_path` override, refusal inside any sandbox-writable surface per `victor/tools/sandbox/backends.py:91` `writable_surface_roots`) and is MOOT for secret material: the vault is zero-knowledge (Argon2id/AES-256-GCM), operator-unlocked, never agent-writable — strictly stronger than any file store. Typed degradation: infra-unavailable → startup-loaded YAML snapshot + WARNING (401 if none); grant/token deny → **fail closed, never degrade after revocation** | Acceptance (§5.2): positive grant→principal test incl. groups; locked-vault fallback honored only when configured; revoked-token no-degrade negative; digest mismatch 401; `yaml` mode byte-identical; secrets absent from logs/errors. Full design: §5.2; upstream follow-ups §5.7 |
| D5 | Attribution subject identity under OIDC | Immutable Victor principal ID derived from verified `(issuer, sub)`; email/username display-only | Auth-policy: caller names/emails are not keys |
| D6 | Browser session semantics | Server-side session (Secure/HttpOnly cookie, CSRF + Origin checks) issued by `/auth/login`; decide persistence home (project DB vs signed stateless cookie) — recommend DB-backed for revocation | The dormant `ServerSettings` HMAC knobs are the natural seam to wire or formally retire; owner chooses wire-vs-retire |
| D7 | `client_id` compatibility lifetime | Keep `client_id` as the key-only attribution label indefinitely; OIDC principals get issuer-qualified IDs; no rename of existing fields in A4 window | Protects dashboards/sandhi virtual-key continuity |
| D8 | Dormant `victor/integrations/api/middleware.py` aiohttp stack | Do not port into A4; schedule deletion in a duplication-sweep PR (its rate limiter idea may be re-homed later as a FastAPI dependency) | Zero callers today; keeping it invites the dormancy trap |
| D9 | WS subscription-granular policy | Defer per-subscription scoping to VAS-05c/VAS-06; A4d enforces at connect + reconnect only | Scoped subscription records are an existing open VAS item |
| D10 | Default posture flips | None in A4: mode stays `api_key`, policy off, credential source stays `yaml`, TLS off, passkey off; attribution values change only for deployments that adopt the registry/OIDC/vault/passkey | Key-only users are behavior-neutral by construction (extended to the §5 phases: every phase ships behind a default-off flag) |
| D11 | Passkey (WebAuthn) RP policy | Optional extra `victor[passkey]`; **`python-fido2` (Yubico) selected** over `py_webauthn` (v2.2.1, 2026-06-29, full `Fido2Server` RP + strict origin/RP-ID enforcement vs. weaker 2026 maintenance signal and prototyping-oriented API); `server.auth.passkey.enabled: false` default; RP ID + exact origin allowlist pinned at config; `user_verification="required"`; attestation conveyance "none"; **no resident-key discovery flows in v1** — credential-id registration stored server-side (project DB) behind the `PasskeyCredentialStore` protocol (reserved for future Sentinelpass custody, an upstream roadmap item); registration requires an existing authenticated principal or a one-time operator enrollment token; ceremonies require TLS (A4c-2) on non-loopback binds | Blocks A4c-3 only. Acceptance (§5.4): fixture-authenticator register→assert round trip issuing a session; wrong origin/RP-ID/replayed-challenge/missing-UV rejections; anonymous registration refusal; attribution + owner-scoping unchanged for passkey principals; disabled mode byte-identical; missing extra = loud failure; TLS-off non-loopback = loud refusal |
| D12 | TLS termination | **uvicorn + Python `ssl` (OpenSSL) with a Victor-constructed `SSLContext`** built in one auditable place (`victor/integrations/api/tls.py`), not uvicorn's ad-hoc `ssl_*` kwargs; **TLS 1.2 minimum**, TLS 1.3 preferred (OpenSSL-managed suites), TLS 1.2 pinned to the six-suite ECDHE+AEAD list (§5.3); HSTS on whenever TLS is on; cert sources: **Sentinelpass vault (primary), filesystem, OS keyring** (explicit precedence, no silent probing); rotation restart-bounded (uvicorn materializes the context at config time); rustls rejected for A4c-2 (would replace the ASGI server stack — Granian recorded as the future alternative); mTLS-as-authentication out of scope, recorded | Blocks A4c-2. Acceptance (§5.3): SSLContext assertions (min version + TLS 1.2 cipher set), vault/file/keyring source matrix with typed failures, HSTS-only-with-TLS, default-off byte-identical startup |

Decisions D1/D2/D6 block the Kanidm OIDC phase (§4 A4c) only; D4-revised
gates A4c-1, D12 gates A4c-2, D11 gates A4c-3; A4a/A4b are merged
(#1993); A4d needs D3 confirmed before merge.

## 7. Risks and non-goals

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
- **§5 evolution limits** (restated from §5.6): the vault protects at-rest
  material and revocability, not the server process's memory; the
  Sentinelpass daemon becomes an optional availability dependency with a
  typed (never security-weakening) degradation matrix; passkey custody
  upstream does not exist yet, so A4c-3 ships on platform authenticators
  only; and none of the §5 machinery creates a second authorization
  authority — grants authorize lookup, A4d authorizes actions.

## 8. References

- [FEP-0020 — AI usage gateway](../feps/fep-0020-ai-gateway-usage-attribution.md)
- [Agent-service auth policy (VAS-05a design)](victor-agent-service-auth-policy.md)
- [Victor Foundations Plan — FOU-A4 row](victor-foundations-plan.md)
- [Victor agent-service plan — VAS-05 rows](victor-agent-service-plan.md)
- Kanidm OAuth2 integration and custom claims docs (see auth-policy
  Standards section)
- Sentinelpass (local checkout `~/code/sentinelpass` at `52079eb`,
  Apache-2.0): `README.md` (secrets broker, `:65-84`),
  `SECURITY_ARCHITECTURE.md` (vault model),
  `sentinelpass-core/src/external_secret_access.rs`
  (`ExternalSecretGrant`/`ExternalSecretAllowlist`, client tokens),
  `sentinelpass-cli/src/commands/{secret,passkey}.rs`,
  `sentinelpass-protocol/src/{message,service}.rs`
  (`ExternalSecretField`, `VaultOp::ServiceGetSecret`),
  `sentinelpass-core/src/daemon/exe_policy.rs` (SP-3 binary pins),
  `sentinelpass-core/src/{biometric_hello,custody/mod}.rs`, `ROADMAP.md:151`
- WebAuthn server libraries: `python-fido2`
  (Yubico, selected — release notes
  <https://developers.yubico.com/python-fido2/Release_Notes.html>),
  `py_webauthn` (duo-labs, evaluated and deferred); W3C WebAuthn L2 spec
  for RP-ID/origin and UV semantics
