# Agent-service authentication and authorization policy

**Design for VAS-05; not implemented or deployment acceptance.** FEP-0039 is in
Review, not Accepted. The [canonical tracker](victor-agent-service-plan.md) owns delivery status.
The user selected Kanidm OIDC and scoped API keys on 2026-10-08. Authenticate both
into one principal contract, then apply one resource/action policy before dispatch.

## Customer experience and trust boundaries

Hosted browser/IDE users sign in with Kanidm. Automation may use renewable OAuth
access tokens or explicitly enabled, scoped service API keys. A successful login
alone grants no run, workspace, approval or administrative access. Embedded/local
use remains available under an explicit local profile; hosted startup never
falls back to unauthenticated mode because IdP configuration or discovery fails.
The existing legacy key-only server is not silently reconfigured by this proposal.

| Boundary | Authority / invariant |
| --- | --- |
| Accounts, MFA, login, membership | Existing Kanidm deployment; no Victor password or group directory |
| Human/workload identity | Verified `(issuer, subject)`; names/emails and caller headers are not keys |
| API-key identity | Durable key record bound to a service principal; a key ID is a credential, not a person |
| Victor roles and access grants | Versioned server policy with explicit subject/group bindings |
| Resource ownership | Persisted tenant/workspace/owner on sessions, runs, events, artifacts and approvals |
| Tool/action execution | Existing tool policy and exact approval/action owners; API permission never bypasses these |
| Provider credentials | Existing provider/vault configuration; do not forward a Victor caller token to Sandhi or InferFlux |

One agent, a team and a workflow use the same authenticated run boundary. Member
credentials/permissions may only narrow the initiating principal's effective
permissions and the approved definition's bounds. A supervisor does not acquire
administrator rights by coordinating other members. An agent cannot approve its
own sensitive action by presenting the run's key.

## Kanidm co-design and discovery

Read-only local evidence inspected on 2026-10-08:

- Sandhi `1a7e6ed3f35c001892fcac28333e74c1717108f8`,
  `docs/operator/oidc-sso.md` and `identity-groups.md`: dedicated client,
  issuer/subject identity, subject/group bindings, explicit credential grants,
  introspection and subject-matched UserInfo. Its identity document names
  `https://id.anvaiops.com` and warns that older `sso.singh.local` deployment
  configuration is not an alias. These are repository records, not a fresh probe.
- InferFlux `c5c2475c1d4d40aea383be4d9095e61d467cb850`,
  `docs/CONFIG_REFERENCE.md` §8: explicit OIDC/API-key scope checks, but discovery,
  ES256, TLS and Kanidm deployment acceptance are still separately called out.
  Do not assume equal authentication maturity or copy unverified deployment claims.

Register Victor independently. Kanidm has **client-specific issuers**; discover
`https://<verified-idp>/oauth2/openid/<victor-client>/.well-known/openid-configuration`
and pin the exact returned issuer and intended resource audience. Do not reuse
Sandhi's/InferFlux's client secret, client registration or token audience. Use separate confidential browser and public VS Code registrations. Verify
each with actual token probes and a reviewed finite issuer/audience mapping before
fixing deployment values; a passing probe never justifies sharing registrations.
Kanidm client issuers differ, and arbitrary extra audiences must not be accepted
to make login work. Cross-client resource continuity needs an explicit verified
account-link mapping from each `(issuer, sub)` to one immutable Victor principal;
do not merge identities by email, username or an assumed issuer alias. VAS-05a
must define and test linking/revocation before claiming browser-to-IDE ownership
continuity. Unlinked identities remain distinct.

Browser login uses authorization code + S256 PKCE, state and nonce, exact redirect
URIs, and an HTTPS server-side session (Secure/HttpOnly cookies, no tokens in web
storage). Writes require CSRF protection and exact Origin validation. VS Code uses
its supported public-client flow, PKCE and SecretStorage; no embedded client secret.
API authentication uses OAuth **access tokens**, never ID tokens or browser cookies
copied into an SDK. Token introspection/authentication methods must match verified
Kanidm discovery; do not assume client credentials grant, introspection `none`,
RS256-only signing or token exchange support from another deployment/version.

Configure one validated access-token strategy per trusted issuer: RFC 7662
introspection for authoritative activity/revocation, or a separately accepted
JWT access-token profile with pinned algorithms/JWKS/issuer/audience/expiry/nbf
and an explicit revocation/freshness bound. Kanidm ES256 support is required.
Reject ID/access-token substitution and wrong-audience tokens. Bound token size,
HTTP timeouts, metadata/JWKS cache lifetimes and key-refresh work; never fetch a
caller-supplied issuer, `jku` or `x5u`. Trust the deployed CA without disabling TLS
hostname verification. Fail closed when required validation is unavailable.

Groups come only from verified configured claims. If Kanidm publishes the custom
group claim in UserInfo, first validate the access token, then require UserInfo
`sub` to match it. Group values must be bounded arrays of strings. Missing groups
grant nothing; malformed/conflicting identity data is rejected. Record membership
freshness and maximum revocation delay; short-lived tokens alone do not imply
immediate group-removal enforcement. Recheck sensitive actions after approval.

## Credential selection and lifecycle

Hosted configuration permits `oidc`, `api_key`, or `oidc_and_api_key`; OIDC is the
recommended/default hosted profile, with service keys explicitly enabled. Local
unauthenticated access is a separate loopback-only profile, not a failure path.
Validate mode and trusted endpoints before accepting requests.

Use explicit credential namespaces: OAuth access tokens in `Authorization: Bearer`,
new Victor keys in a distinct scheme/header agreed in VAS-05a. Preserve legacy
Bearer keys only in an explicit key-only compatibility profile. Reject ambiguous
multiple credentials. A token's unverified claims must not select an arbitrary
issuer or choose a weaker authenticator. Invalid/expired OIDC never retries as an
API key, anonymous principal or another issuer. A configured API-key request is
its own supported path, not an automatic outage fallback.

Keys need cryptographic randomness, a public lookup ID plus secret, hash/verifier
storage, constant-time verification, expiry, durable revocation and rotation.
Reveal a newly minted secret once; audit only its public ID. Bind key scopes,
allowed workspaces/definitions and principal to server records. Key creation
requires explicit authority and cannot exceed the issuer's grants. Group-derived
keys require bounded delegation freshness; long-lived keys need an explicit
service-principal grant, not a permanent copy of a user's current groups. Keys
cannot mint keys, change policy, manage credentials or approve runs by default.

## Policy contract (proposed shape)

Use bounded RBAC grants plus resource attributes and ownership relationships.
Do not introduce a general expression language, dynamic Python imports, a second
user directory or a second tool dispatcher. Existing tool RBAC is category/action
oriented; its ADMIN shortcut and empty-category semantics do not establish API
resource/tenant isolation and must not be reused as an automatic API superuser.

The following is **illustrative**, not currently accepted configuration. Exact
field names belong in the reviewed strict schema (unknown fields rejected).

```yaml
version: 1
revision: "operator-managed-immutable-revision"
default_effect: deny
combining: deny_overrides
bindings:
  - issuer: "https://<verified-idp>/oauth2/openid/<victor-client>"
    group: "victor_developers" # verified configured claim, not request input
    roles: [run_operator]
  - service_principal: "victor-ci" # local durable principal, not an OIDC alias
    roles: [ci_runner]
grants:
  - id: own-project-runs
    role: run_operator
    actions: [agent.invoke, session.create, session.read, run.create, run.read,
              run.events.read, run.cancel, artifact.read]
    resources:
      tenant: "engineering"
      workspaces: ["victor-project"]
      agent_definitions: ["coding-assistant", "coding-team"]
      ownership: self
    authn_methods: [oidc]
  - id: ci-project-runs
    role: ci_runner
    actions: [agent.invoke, session.create, session.read, run.create, run.read,
              run.events.read, artifact.read]
    resources:
      tenant: "engineering"
      workspaces: ["victor-project"]
      agent_definitions: ["test-reviewer"]
      ownership: self
    authn_methods: [oidc, api_key]
```

A grant matches **all** its dimensions together. Never union actions from one
role with workspaces from another to create an unintended cross-product. Multiple
matching allow grants can authorize an operation, but an applicable deny wins.
Token/key scopes are a ceiling: the requested action must be within credential
scope **and** a complete server grant **and** resource ownership/share rules.
Resolve create targets from approved definitions and canonical workspace IDs;
callers cannot establish ownership by including another principal in JSON.

| Contract | Required fields / behavior |
| --- | --- |
| Verified principal | Immutable identity, principal kind, auth method, credential ID, audience, expiry, scopes and verified group provenance |
| Authorization request | Principal, enumerated action, resource kind/ID, persisted tenant/workspace/owner, relevant resource/policy versions |
| Decision | `allow` or `deny`, stable reason code, policy revision, matched grant IDs and decision ID; policy errors deny |
| Action approval | Existing exact action/payload/version/expiry binding, verified approver identity and fresh authority; not a substitute for authorization |
| Audit event | Decision/request/run/member/action correlation, public identity IDs, result and policy revision; redact tokens, secrets and private payloads |

No resource/action/owner mapping means deny. Filter lists/events/artifacts before
returning them; protect individual object lookups and alternate transports too.
Reauthorize stream reconnect and membership/credential expiry according to a
bounded lease; an open socket is not unlimited authorization. Missing credentials
produce 401; authenticated policy denial produces 403 (or consistently concealed
404 for object lookups). Never return an empty-success result for denial.

Roles are named bundles, not a hierarchy that grants everything above them.
Keep `run_operator`, scoped `run_observer`, designated `approver`, configuration
administrator and credential administrator distinct. Administrative policy edits
are versioned/audited and separately authorized; admin access does not implicitly
read every private conversation or approve a sensitive action. Bind approvers to
the relevant tenant/workspace, prohibit self-approval where required, and preserve
exact approval checks at execution. OPA/Cedar integration is optional future work
behind the same decision contract, justified by an actual deployment need.

## TDD and deployment acceptance

VAS-05a defines strict policy/principal contracts; VAS-05b implements Kanidm and
key authentication; VAS-05c applies policy to every transport/resource; VAS-05d
validates UI login, deployment and operations. Tests must include:

- Valid human/workload token and scoped key allow their authorized run; cross-owner,
  workspace, tenant, definition and action requests deny for both methods.
- Wrong issuer/audience/algorithm, ID-token substitution, expiry, unknown key,
  malformed group data, replayed login state and credential ambiguity fail closed.
- Kanidm key rotation, membership removal, revoked service key, IdP outage and
  policy reload respect documented cache/lease bounds; no alternate-auth fallback.
- Role grants cannot combine across resource dimensions; explicit deny wins;
  policy parser rejects unknown/malformed fields and preserves no permissive default.
- Resume/reconnect and delayed approval recheck identity, current authority and
  exact payload; member delegation never widens authority or bypasses tool policy.
- Browser/VS Code real login and HTTP/SSE/WS tests use the same policy decisions;
  negative UI controls are backed by server denial, not hidden buttons alone.
- Provision on the existing DS3 Kanidm only after verifying registrations, deployed
  version, discovery, CA, redirects, scopes and actual claims. Preserve existing
  apps/recovery access; do not transfer credentials or reset the realm. Record
  source/binary/config identity separately from the reviewed source implementation.

## Standards

The choices above apply OAuth security requirements and resource authorization
guidance to Victor; the particular role/grant schema is a project design proposal.

- [OAuth Security BCP / RFC 9700](https://www.rfc-editor.org/rfc/rfc9700.html)
- [JWT access-token profile / RFC 9068](https://www.rfc-editor.org/rfc/rfc9068.html)
- [OWASP authorization guidance](https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html)
- [Kanidm OAuth2 discovery and integration](https://kanidm.github.io/kanidm/stable/integrations/oauth2.html)
- [Kanidm custom group claims](https://kanidm.github.io/kanidm/stable/integrations/oauth2/custom_claims.html)

## Workflow UI → Victor → Sandhi → provider co-design

The scoped customer acceptance is: sign in with Kanidm, visually compose an
approved workflow, choose a supported formation, save/reload the same definition,
run it, review any exact approval, and see the verified result and correlated
Sandhi accounting. A green login page or graph layout change is insufficient.

Source review of `../sandhi` at the SHA above found:

| Boundary | Current source evidence | Required work / acceptance |
| --- | --- | --- |
| Inbound Victor caller | `victor/integrations/api/routes/chat_routes.py` binds local subject correlation | Verified principal and owner-scoped session/run state before multiuser service exposure |
| Outbound Victor credential | `victor/providers/openai_provider.py` and `victor/core/identity/gateway.py` select a configured provider/workload credential | Immutable per-call credential binding; never mutate a shared provider key/token file for another user |
| Sandhi identity | `crates/sandhi-proxy/src/auth.rs` validates its issuer/audience and subject-matched group evidence | A Victor-audience token is not a Sandhi token; wrong-audience requests must deny |
| Sandhi grant | `auth.rs` resolves a complete upstream/model/budget grant; `lib.rs` rejects forged subject headers | UI selects a permitted grant reference; server derives identity and enforces the grant |
| Provider mapping | `crates/sandhi-proxy/src/operator.rs` and `lib.rs` resolve upstream vault credentials | Provider receives its own API key; no user bearer, subject claim or browser cookie is forwarded upstream |
| Delegation | `/auth/keys` requires an already authenticated Sandhi identity; tokens-mode `/auth/token` is separate from OIDC | Neither endpoint establishes cross-service on-behalf-of token exchange |

Track three different values: **initiator** (human/service requesting the Victor
run), **actor** (identity authenticating the Sandhi call) and **provider credential
reference** (vault-owned upstream authority). They may differ. Workload-authenticated
Sandhi calls must be labelled as workload usage, with Victor's immutable initiator
mapping joined by run/member/request correlation. This is an acceptable intermediate
milestone, not the final requested human-scoped Sandhi authorization verdict.

For end-to-end human-scoped Sandhi policy, implement an explicit delegation path:
verify whether the deployed Kanidm version supports the needed token exchange and
claims, or design a narrowly scoped broker/delegation contract with Sandhi. It must
bind initiator, actor, target audience, allowed grant/model/resource and expiry;
attenuate authority and enforce revocation. A service-account token exchange is not
proof of human on-behalf-of support. Do not reuse client registrations, forward the
wrong bearer token, alias issuer-qualified subjects, forge attribution headers or
fallback to a shared privileged key. Record this prerequisite as VAS-05e/f.

Existing static token-file credentials remain suitable only for the explicitly
configured identity. Correct multiuser handling requires isolated principal/run
ownership first. Preserve in-flight admitted credentials during rotation and ensure
concurrent users cannot exchange cached handles, grants, sessions or accounting.
Direct-provider mode remains explicit and authenticated, and cannot claim Sandhi
metering. Hosted end-to-end acceptance uses the gateway path requested here.

## Headed AgentBrowser acceptance and authoring scope

Use the installed AgentBrowser CLI/MCP, not a second bespoke Playwright runner.
Record CLI/server/browser versions, headed session ID, exact application source,
routes, sanitized snapshots/screenshots and independent run/receipt/accounting
proof. Reuse current semantic element references; never invent selectors or call
raw evaluation to bypass the public action boundary. Human MFA stays interactive;
credentials/cookies/tokens are not exported into reports or source.

Discovery at this checkpoint:

- Victor's `workflow_visualizer.html` provides graph layout, zoom, monitoring and
  node detail controls. Source inventory found no workflow/formation authoring
  canvas. Displaying or dragging graph layout nodes does not create a definition.
- AgentBrowser CLI and live service report 1.15.2. Its discovered action catalog
  includes click/fill/select/press but no drag/drop primitive. A pointer drag demo
  therefore needs an accepted AgentBrowser action capability; an accessible
  keyboard move/add equivalent is separately labelled, not claimed as pointer drag.
- AgentBrowser's documented current egress policy closes page WebSockets and
  buffers SSE. Do not disable that security policy to manufacture streaming success.
  Polling recorded run state can establish a UI outcome; real streaming requires
  supported transport acceptance and remains a separate gate.

Build authoring as another adapter to the existing workflow compiler, approved
node definitions and `UnifiedTeamCoordinator`. A team is an existing StateGraph
node, not a parallel graph/runtime abstraction. Derive the formation palette from
the single registry/enum; distinguish all twelve canonical formations and three
ensemble policies from visual layout choices. Unsupported capabilities are disabled
with an explanation. Canonical terminology and supervisor semantics remain unchanged.

Required authoring cases: sequential/pipeline, parallel fan-out/join, conditional
routing, bounded reflection/loops and approval pauses. Validate graph structure,
references, typed ports, entry/terminal nodes and loop budgets before save/execute;
server validation is authoritative. Persist a versioned canonical definition and
stable hash separately from canvas coordinates, retain optimistic concurrency and
round-trip import/export. Do not accept arbitrary Python/code or credentials from
a dragged node. Include keyboard authoring, focus/order, undo/redo and clear errors.

The final headed demo matrix must cover:

1. Two isolated users log in through their authorized Kanidm roles. Unauthorized
   workspace/definition/approval actions are denied by the server for both users.
2. Drag approved nodes, connect them, choose a registered formation/policy, save,
   reload and verify the exact canonical definition/hash. Keyboard equivalence is
   tested separately. No fake topology or silent unsupported-node substitution.
3. Submit one authorized run, observe progress, approve the exact action where
   required, disconnect/reconnect, and verify no duplicated member/action effects.
4. Join user/actor/run/member/request identities through Victor and Sandhi, verify
   upstream grant/model restriction, actual deliverables, task pytest, distinct
   member sessions and usage conservation. Shared-key attribution is not human
   delegation acceptance.
5. Exercise wrong audience, revocation/expiry, policy denial, model unavailability,
   deadline, cancellation and lost response; preserve partial/unknown outcomes.
6. Run ZAI reference tasks first, then explicitly simpler local-model cohorts via
   released InferFlux/Sandhi. Preserve the existing six-Qwen/one-ZAI C5 gate and
   previously failed evidence; a UI demo does not replace that harness verdict.

## Embedded Sandhi versus gateway deployment

Sandhi is both a reusable in-process runtime/metering library and an external
routing service. Do not force an HTTP gateway into every embedded Victor call or
mistake installing the Python binding for deploying gateway identity/vault policy.
The pinned Python binding README exposes virtual keys/budgets/metering and typed
persistent provider handles; gateway OIDC middleware and vault mapping live in
`crates/sandhi-proxy`. Shared Rust code alone does not prove equal exposed features.

| Shape | Preferred use | Enforcement and proof |
| --- | --- | --- |
| In-process Sandhi | Embedded/local Victor; explicitly chosen isolated service deployments | Victor authenticates/authorizes its caller; available typed Sandhi library interfaces enforce their documented contracts. Explicitly establish credential storage, identity admission, durable settlement and policy parity before claiming gateway-equivalent controls. No loopback HTTP solely for metering. |
| External Sandhi | Hosted multiuser workflow service and the requested end-to-end gateway demo | Centralized independent OIDC/grants, provider vault, budgets and usage. Victor uses an authorized Sandhi credential/delegation; gateway translates to provider-specific authentication. One extra hop and service dependency are measured trade-offs. |

Use one explicit deployment/provider route choice and the same typed provider
request/result/accounting contracts. Do not add duplicate registries or a new
provider protocol. Never switch from gateway to direct on outage or policy denial.
If a selected embedded profile cannot satisfy a required control, fail startup or
reject that feature explicitly rather than silently presenting a weaker profile.
An authenticated in-process caller is a trusted typed principal supplied by Victor,
not a self-asserted user header or arbitrary string from a workflow node.

VAS-05e evaluates exposed library capability against external service policy;
VAS-05f implements the selected binding without shared mutable user credentials.
Acceptance compares both supported shapes on identical deterministic workloads,
including denied grants, cross-user state, timeouts and settlement. Report latency
and operational costs after measurement; prioritize security/UX parity over an
unmeasured performance claim. The live headed gateway demo uses external Sandhi;
embedded acceptance remains a separately labelled matrix row.

## Product-family theme and commercial UI boundary

Generic authentication, authorization, reusable UI components and standalone
agent-service behavior belong in Victor OSS. AnvaiOps owns commercial product
composition, managed operations and entitlements/billing. The shared theme must
not introduce private runtime dependencies or move basic security behind a
commercial gate; the [technology decision](victor-agent-service-technology.md)
defines the repository ownership and separate release boundaries.

Read-only source baselines fetched on 2026-10-08:

- AnvaiOps `origin/develop` `6f1cb95b8c1307a3271a636a4b67f9cc9c0d4dd2`,
  `apps/control-plane/src/tokens.css` (declared family token source) and `styles.css`.
- Sandesha Client `origin/develop` `15f1e445effa7e230121bd22a3d6ffbf4d0b75e4`,
  `app/globals.css` and `tailwind.config.ts`. Its shared-OIDC reuse document also
  records a dual-consumer extraction candidate, with wider adoption still HOLD.
  Reconcile that work before implementing a duplicate OIDC verifier. A prototype
  or another product's security waiver is not approved runtime dependency evidence.

Both local roots contain other sessions' extensive dirty work/feature branches;
fetching updated remote references did not authorize merging/rebasing those trees.
Implementation must use a reviewed immutable source baseline, not copy whatever
happens to be present in a sibling's working directory.

| Semantic role | AnvaiOps baseline | Sandesha alignment / Victor rule |
| --- | --- | --- |
| Brand chrome | Ink `#0f172a`, secondary `#1e293b` | Reuse family navigation/header hierarchy |
| Primary action/focus | Teal `#0f766e`, strong `#115e59` | Same teal identity; Sandesha dark primary `#2dd4bf` |
| Light surfaces | Background `#f6f7fb`, surface `#ffffff`, secondary `#f0f2f8` | Semantic tokens, not arbitrary per-panel colors |
| Typography | Inter with system fallbacks, monospace for code/IDs | Bundle licensed assets or use safe system fallback; no remote-font dependency for usable UI |
| Shape | 6/8/12px radii; restrained shadows | Consistent controls, tables, cards and workflow inspector |
| Status | Distinct success/warning/error semantic tokens | Keep status meaning, text/icon labels and accessible contrast in both themes |

Introduce a reviewed token adapter/asset boundary with provenance and a drift
check. Keep CSS/framework adapters thin (AnvaiOps static/TS, Sandesha Tailwind,
Victor web/Svelte/IDE); do not import an entire app, copy private business logic,
or make an OSS runtime depend on a private checkout. A shared package requires
license/publication review and immutable versions; local sibling paths are not
release dependencies. VS Code embedded views must respect editor high-contrast
and theme tokens while using the same information hierarchy and semantic roles.

VAS-14e owns Victor theme/component alignment; VAS-21 owns the Sandhi adapter
and cross-product acceptance. Sandhi is explicitly part of this same commercial
family; retain its standalone OSS dashboard and existing auth/admin boundaries.
Apply the semantic tokens to its current CSS/JS before considering a framework
migration. The [technology decision](victor-agent-service-technology.md) defines
the language/runtime boundary. Required component states include: navigation, sign-in/account/role indicator,
empty/error/loading/denied states, typography, focus, spacing, light/dark/high-contrast,
reduced motion and responsive editor layout. Use AgentBrowser headed snapshots
at desktop/narrow widths with role/state variants. A visually cohesive service
still needs separate product authorizations; a family SSO session never grants
cross-product access by itself. Existing OSS APIs remain usable without a
commercial control-plane deployment.

### Headed baseline evidence (not acceptance)

AgentBrowser service/CLI 1.15.2 launched dedicated headed Chromium 154.0.8037.98
at 1440×1000, isolated session `ses_1791463512549_u2pd7w1rg`. The current production
visualizer route was served on an ephemeral loopback port with deterministic
fixture graph data and application lifecycle disabled. No real users, OIDC login,
provider requests or workflow execution occurred. Snapshot and screenshot show
layout/zoom/refresh controls and a purple gradient inconsistent with family tokens.
The canvas failed with `cytoscape is not defined`; this records an unavailable
external script in this smoke environment, not a verified universal root cause.
Bundle/pin graph assets and exercise a functional canvas before UI acceptance.
Artifacts are preserved in the declared local archive; no drag/SSO pass is claimed.
