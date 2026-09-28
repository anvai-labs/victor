# OpenAI subscription gateway: Victor / sandhi co-design

Status: implemented on the subscription-gateway branches; release promotion remains gated.

## Ownership and request path

```
Victor client A -- virtual key A --+
                                  +--> sandhi --> subscription Responses backend
Victor client B -- virtual key B --+       ^
                                          |
Codex login owner --> access-only lease publisher --> OS credential vault
```

A gateway configuration is a complete authentication mode in Victor. It takes
precedence over `auth_mode=oauth`, explicit upstream keys, environment keys and
local credential resolution. Victor neither reads nor refreshes an OpenAI grant
in this mode. It sends ordinary Chat Completions with its virtual key to the
configured HTTPS or literal/localhost loopback HTTP gateway. There is no direct
upstream fallback. Invalid/missing gateway credentials fail before discovery.

Sandhi's `openai` vault entry with `scheme=oauth` contains a strict JSON envelope:
`access_token`, `account_id`, `expires_at` (Unix seconds). Unknown fields,
including refresh tokens, are rejected. The upstream is fixed to
`https://chatgpt.com/backend-api/codex`; an arbitrary override is rejected before
vault mutation. The handle declares Responses family explicitly and uses the
ChatGPT/Codex constrained codec: SSE even for aggregated completions,
`store=false`, system/developer instructions, and supported parameter mapping.
Raw forwarding is disabled so native Responses requests cannot bypass these
constraints or expiry checks. A caller cannot replace the vaulted account header.

## Login, rotation, revocation and expiry

The existing Codex process remains the sole writer/refresh owner for its grant.
`scripts/subscription_lease.py` reads an owner-only regular cache file without
following symlinks. The operator pins the expected account. The publisher copies
only the access token to sandhi's OS vault and limits the lease to five minutes
or the token's own expiry, whichever is earlier. The cache is never rewritten.
JWT expiry decoding is scheduling, not signature verification; OpenAI validates
the bearer. Keyring-only Codex installations need a separately supported export
or credential broker; do not silently downgrade keyring storage to a plaintext
refresh-token copy.

Renew the lease every minute. Every typed dispatch checks expiry with a 30-second
margin, including streaming and observed-attempt paths. The lease remains fenced
after gateway restart; expired vault entries do not yield an active handle.
In-flight requests may finish after expiry. Login disappearance/account change,
publisher failure or expired access token stops *new* dispatch at the deadline.
Run `codex login` to renew the login when needed; the publisher never performs
an undocumented refresh flow. An independent gateway-owned OAuth grant could be
added later, with its own single writer and persisted rotation protocol. Sharing
one refresh token between Victor, Codex and sandhi is rejected as a design.

Revoking a client virtual key stops future admission for that client. Revoking
the gateway vault entry stops local route dispatch, but does not revoke the
OpenAI grant. To revoke upstream access, use the account's session controls and
stop the publisher; otherwise it would republish. Admin credentials never go to
Victor. A missing/invalid token is not grounds to try a different account/provider.

## Per-client authority and accounting

Provision one virtual key per client, bound to an immutable subject, explicit
model allowlist, rate limit, expiry and blocking daily token budget. Use a shared
owner group for attribution. In OIDC mode, all keys for the same owner/grant
share the configured budget scope and rate bucket; rotation does not reset them. The gateway
rejects forged subject/group headers. All clients still share the account's real
subscription quota and limits; virtual keys do not create new entitlement,
independent subscriptions, or authorization to share/resell account access.

Store key material only in private client files or a credential broker. Usage
records contain attributed neutral token units and provider-reported usage;
subscription requests are not priced as ordinary Platform API-key requests.
Errors must not expose tokens, cache contents or upstream response bodies.
Subscription upstream transport retries are disabled in this route; callers
must distinguish pre-dispatch failure from ambiguous inference completion.
Agent/tool side effects require their own approval and idempotency ledger.

## Provisioning and client configuration

Build the matching gateway branch. Keep `SANDHI_AUTH_MODE=oidc` and the existing
`SANDHI_OIDC_CONFIG`, private `SANDHI_STORE`, `SANDHI_VAULT_BACKEND=keyring`, and a
unique vault label. Keep admin/dashboard private. `SANDHI_ADMIN_TOKEN` and the
script's static-key provisioning are compatibility-mode features; they do not
replace OIDC administration in the configured deployment. `subscription_lease.py --help`
documents publisher and scoped client provisioning. Admin JSON contains
`admin_token`; credential outputs use exclusive creation and never print keys.
A failed key-output write requires inventory reconciliation, not blind minting.

Victor profile provider kwargs:

```yaml
provider: openai
model: <model explicitly allowed on the virtual key>
# gateway is passed in provider kwargs; a profile can use env resolution below
```

Set `SANDHI_GATEWAY_URL` and `SANDHI_GATEWAY_VIRTUAL_KEY_OPENAI` only for that
client process, or pass `gateway={url, virtual_key}` to `SandhiOpenAIProvider`.
Do not set `OPENAI_API_KEY` to the subscription access token. Do not let global
routing settings silently move message-hub's private triage to a cloud provider.

## TDD and acceptance gates

Red tests first demonstrated local OAuth discovery in gateway mode, acceptance
of malformed/expired OAuth secrets, wrong protocol selection, and mutable
account headers. Green tests cover those boundaries, complete/stream expiry,
access-only publication, filesystem restrictions, account mismatch, and two
independent virtual-key clients through a mocked subscription SSE upstream.
The end-to-end test checks model denial, forged attribution, gateway-only bearer
replacement, fixed account identity, mandatory codec fields and per-client usage.
Existing provider/proxy suites and Victor direct OAuth tests remain required.

Live acceptance uses synthetic prompts only: both clients must succeed with
separate attributed records; denied models/spoofing must never reach upstream.
Test publisher/gateway restart and expiry; then configure supervision and record
binary/source provenance. Do not claim a live result from mocked conformance.

## Evidence and limits

[Official Codex authentication documentation](https://developers.openai.com/codex/auth/)
distinguishes ChatGPT login from separately billed Platform API keys and describes
cached credentials as sensitive. This route reuses the existing supported-in-code
subscription Responses adapter; it is not a claim that ChatGPT login is an
ordinary `api.openai.com` API key, or a provider guarantee for arbitrary workloads.
Account entitlements and backend behavior remain external acceptance gates.


## Identity-first follow-up

Migrate explicitly to the verified current `id.anvaiops.com` authority after
checking the restored client and current subject UUIDs. Group policy lives in sandhi; membership
lives in Kanidm. OIDC clients may exchange a verified access token for an attenuated
120-second virtual key. Owner/grant rate and budget scopes are shared across keys,
while each client key remains auditable. Renewal belongs to the caller credential
broker; a static environment key alone cannot support unattended renewal. Do not
deploy the earlier token-mode provisioning pilot as a substitute for this boundary.
See sandhi docs/operator/identity-groups.md for acceptance and IdP freshness limits.


## Renewable OIDC and long-lived virtual keys

OpenAI provider profile configuration:

```yaml
providers:
  openai:
    gateway:
      url: http://127.0.0.1:18789
      grant: subscription
      oidc:
        token_file: /absolute/private/path/access.json
        issuer: https://id.anvaiops.com/oauth2/openid/sandhi
        audience: sandhi
        subject: <verified user UUID>
```

The loopback URL is for a client on the gateway host. Remote clients need an
HTTPS endpoint whose certificate chain the installed transport trusts; the token
file's issuer field does not configure TLS trust for inference. Never disable
certificate verification to accommodate a private CA.

The operator-owned OIDC broker atomically replaces this owner-only regular file
(0600, at most 64 KiB; symlinks are rejected). Its exact JSON contract is:

```json
{
  "access_token": "<OIDC access token>",
  "token_type": "Bearer",
  "expires_on": 1790500000,
  "issuer": "https://id.anvaiops.com/oauth2/openid/sandhi",
  "audience": "sandhi",
  "subject": "<verified user UUID>"
}
```

`expires_on` is the real Unix expiry, not the example value. The file must have
more than 30 seconds and at most one hour remaining. The reader does not log in or
refresh; broker supervision and secure atomic publication remain operator duties.
It pins broker metadata; Sandhi independently authenticates the access token with
the IdP. No refresh token is permitted in this file.

For embedded clients, pass `gateway={url, credential, audience, grant}`, where
`credential` implements the existing async `TokenCredential` protocol. A caching
credential can coalesce renewal; no new identity protocol is introduced. Each
request resolves a valid credential with a bounded acquisition timeout before
dispatch. A failure is redacted and terminal for that attempt; there is no
upstream-key fallback. Transport adapters do not automatically replay. The typed transport replaces cached
handles on token rotation. In-flight calls retain their admitted credential.

Choose exactly one of `virtual_key`, `oidc`, or an injected `credential`.
Explicit OIDC suppresses environment-key fallback. `grant` sends only the named
grant selector; it cannot assign identity. Member-agent provider overrides retain
the same credential configuration. Renewable gateway credentials currently cover
the OpenAI path; the registry rejects this option for other provider families.

Alternatively, keep `gateway={url, virtual_key}` for an expiring, long-lived user
key issued by Sandhi's `/auth/keys` with explicit `ttl_seconds`. Sandhi must enable
`max_subject_key_ttl_seconds` and bind that user directly to the named grant.
Group-only authority cannot mint a long-lived key. The gateway checks revocation,
expiry and current subject policy on every call, but IdP account/group changes do
not automatically revoke this independent service grant. Prefer renewable OIDC
for directory-driven access. Use a different public key ID for each client;
budgets and rate limits remain shared by owner/grant.

Candidate validation: 161 affected Victor tests passed; the subsequent focused
gateway suite passed 19 tests including two new public chat/stream expiry-denial
tests. Coverage includes renewable-token
rotation, fail-closed acquisition, malformed/private-file rejection, configuration
propagation, member agents, existing OAuth and transport regression tests. The
live brokers now exchange and validate tokens successfully against
`https://id.anvaiops.com/oauth2/openid/sandhi`, with verified `sandhi_agents` and
`sandhi_viewers` claims for separate workload subjects. DS3 administrator recovery
and owner-only credential-file reconciliation are complete. The existing public
Sandhi client/PKCE registration was preserved; missing workload accounts were
recreated with new UUIDs and explicit memberships. Old subject UUIDs are not aliases.

The repaired broker is staged alongside the old one on DS3. Broker bootstrap
credentials expire after 30 days; emitted OIDC access tokens last 15 minutes.
Renewal supervision, profile cutover, browser callback and live subscription
inference remain acceptance gates. Existing gateway processes have not been
replaced by this candidate.

## Pluggable policy evaluation at the gateway

Sandhi's docs/operator/policy-evaluation.md documents the implemented, opt-in
candidate. Strict JSON policies select verified identities/groups/roles and
upstream/model facts. Local regex, numeric threshold and lexical Jaccard checks
run before budget reservation/provider dispatch. Block dominates quarantine,
which dominates forward; audit findings are independent. Mandatory metadata
receipts contain no prompt or detected secret. Quarantine holds dispatch without
retaining content or providing automatic replay. Lexical similarity is not an
embedding model or comprehensive DLP.

Victor now preserves policy_blocked, policy_quarantined and policy_unavailable
as terminal ProviderPolicyError outcomes with an opaque receipt ID. It suppresses
raw diagnostic bodies and exception chaining. Gateway mode disables SDK/FFI
retries and provider failover, including stream and circuit-open paths. A gateway
used in a fallback chain also cannot fall through to another provider after
failure. Direct-provider behavior is preserved. Clients send grant selectors,
not identity/group authority or executable policy. Local previews remain advisory;
Sandhi owns dispatch authorization.

TDD covers these outcomes and malformed error payloads. The affected Victor
suite passed 141 tests, full collection found 33,643 tests, and scoped Black,
Ruff and mypy passed. An actual rebuilt Python binding passed the isolated live
Kanidm → Victor → Sandhi → synthetic-provider smoke: allowed inference, terminal
block/quarantine, receipt preservation and stream denial with no retry/fallback.
This is not evidence of live OpenAI subscription inference or production cutover.

The MVP supports text-only OpenAI Chat Completions ingress, including translation
to subscription Responses. It inspects system/developer/user/assistant prefill,
tool results and schemas/decoded arguments. Unsupported fields/modalities fail
closed. Allowed transparent same-family requests retain bytes; subscription
translation keeps its existing contract. There is no output inspection/redaction.
Only gateway-owned upstream credentials make this an enforceable deployment
boundary; a generic external client may ignore error retry hints.

Sandhi's docs/td/TD-0005-declarative-policy-engine.md retains later-phase targets.
Its docs/operator/python-ml-evaluators.md documents warm, supervised Python NLP/ML
workers with pinned models and killable deadlines, now implemented as a candidate.
A backend-neutral score contract also admits native adapters. The optional
embedded ONNX CPU/ASCII TF-IDF profile is now implemented, with Python parity,
in-flight cancellation and live Victor/gateway acceptance. Authenticated remote
services now have a multi-model Flask/Gunicorn template and Sandhi replica adapter.
One listener/server hosts named evaluator URLs and bounded supervised model children.
Sandhi owns replica selection, TLS/mTLS service authentication, model/code provenance,
passive failure exclusion and the total deadline; Victor continues using its existing
Sandhi URL and identity. User virtual keys/OIDC tokens never reach evaluator servers.
Local block/quarantine/incomplete checks prevent remote text egress, and no failed
request is replayed to another replica. HAProxy/platform ingress is optional, while
HA for Sandhi itself and a shared budget/rate ledger remain separate requirements.
A local Python worker still uses private IPC and needs no Flask listener. MLflow owns offline experiment
tracking and controlled model promotion, with optional bounded asynchronous
metadata export. Neither model downloads nor tracking/registry lookups belong
on the synchronous admission path. Offline MLflow parameter/metric export is implemented and tested against local
SQLite; automatic registry promotion and production telemetry remain future work.
The working Python client binding is a different component. Horizontal evaluator
scaling does not supply a shared gateway budget/rate ledger.

Before routing the real message-hub classify workload through this path, verify
its provider configuration, latency and fallback behavior, supervised credential
renewal, browser callback and real subscription inference. The existing direct
tunnel remains an explicitly unaudited route until that cutover is accepted.


## Consolidation security review (2026-09-27)

An independent adversarial pass found that provider-level terminal policy handling
did not cover outer recovery code. The candidate now rejects explicit virtual-key
or OIDC configuration without a gateway URL and preserves the original policy
exception and receipt through subagent retry, recovery services, response completion,
summary helpers, the buffered loop, and real StateGraph ACT/node/runtime execution.
One shared gateway-boundary check recognizes ManagedProvider and ResilientProvider.
Regression tests reproduced the failures before fixes and exercise both actual
StateGraph run/stream execution and production provider wrappers.

Scope: policy denials are terminal, and gateway provider failures cannot choose a
direct upstream fallback. This is **not a universal exactly-once guarantee** for
the agent runtime. A buffered non-policy transport failure can still become an
empty loop result and trigger a separately admitted completion request. That
request passes through the same gateway policy and budget gates, but may duplicate
inference work; operators should reconcile ambiguous provider outcomes. The remote
evaluator's no-replay contract remains separate and strict.

See `docs/development/GATEWAY-SECURITY-REVIEW-2026-09-27.md` for final review evidence
and deployment limits. This change does not activate a production gateway route.
