# OpenAI subscription gateway: Victor / sandhi co-design

## Coordinated execution-contract migration

Victor **0.29.0** is prepared against `sandhi-gateway==0.17.0`. These source
versions do not imply that packages have been published or a service upgraded.
The user authorized a coordinated migration because Victor is the sole client;
there is no requirement to retain a permissive Sandhi legacy mode.

Both direct FFI and gateway handles set `max_retries=0`. Victor's logical-call
owner decides whether another attempt is authorized. A binding failure never
falls back to another transport. `output_projection_error` and `invalid_response`
are terminal even when their HTTP status is 502. Iterator EOF without a validated
finish is a terminal error; it cannot manufacture a successful final result.
The resilience owner's primary and fallback invocation deadlines follow that
same rule. Their cancellation cannot prove that the backend did not accept the
request, so they surface a terminal timeout with an unknown dispatch outcome.
An explicit transient provider rejection can still use the logical-call retry
policy and keeps the same call identity across authorized attempts.

Sandhi's four admitted typed streaming source families share source validation
and actual token observation across embedded and accounted calls. The paired candidate
adds the same qualified buffered boundary and explicit gateway-peer contract; the
mandatory gateway accounting-default migration remains separate. Portable/native delivery
are separate contracts: meaningful native state must survive compatible native
delivery or be refused explicitly. Disjoint fresh/read/write/output token counts
are evidence; cache policy and provider price schedules remain separate concerns.
An embedded observation is not a gateway ledger receipt.

Qualify the exact Sandhi candidate wheel with this Victor source, including tools,
cache splits, interrupted streams and refusal/no-replay paths. Keep Contracts and
CodeGraph on their independent release lines. Publish the qualified Sandhi artifacts
before making the matching Victor release installable from registries. Source
review, public-hosted CI, release and deployment are distinct gates.

The root, API and CPU-embeddings deployment locks carry the same exact Sandhi
pin. Until 0.17.0 is published, registry-only CI/install jobs cannot resolve that
pin. Local paired qualification installs the reviewed candidate wheel explicitly;
it is not evidence of a published dependency. Keep the consumer preparation
unmerged until the dependency's publication/qualification gate is met.

### Explicit endpoint usage conventions

OpenAI-compatible policy providers accept the constructor option
`endpoint_profile`; other Sandhi-backed providers can carry it in their extra
configuration. It participates in persistent handle identity and is never a model
parameter. Per-call overrides fail before credential refresh or transport. Direct
Z.AI Chat declares `openai_terminal_chunk_usage`; explicitly qualified custom
endpoints can select `standard`. Compatible Messages sources can explicitly select
`anthropic_implicit_cache`. Sandhi validates each profile against the actual wire
family, independently of the provider label or key contents.

Candidate evidence: 1,932 provider and adjacent resilience tests passed with five
skips against the candidate wheel; full collection found 40,170 tests. Black/Ruff
and typing for the three changed provider modules passed. This qualifies the
implemented constructor/streaming scope, not ordinary buffered qualification or
the complete packaging matrix.

The profile describes a direct HTTP source. Gateway mode never forwards it. Instead,
Victor constructs an explicit `peer_role="sandhi_gateway_v1"` handle with one client
dialect, zero hidden retries and only the virtual credential. The gateway declares
its actual emitted usage convention in bounded response metadata; the Sandhi peer
validates that convention before decoding. Backend endpoint profiles, credentials,
account headers and cache capabilities remain server owned.

### Independent gateway client dialect

`gateway.protocol` accepts `chat_completions`, `anthropic_messages`, `responses`, or
`gemini_generate_content`. It overrides the provider's direct protocol hint. With no
explicit override, Victor normalizes a subscription/Codex hint to ordinary
`responses`; otherwise the provider family supplies the client default. Local
Ollama/Cohere providers use portable Chat Completions ingress. The provider label
and requested model alias do not select or authorize a backend.

```yaml
gateway:
  url: https://gateway.example.test
  virtual_key: <private virtual credential>
  protocol: responses
```

The transport derives `/v1`, root Messages, or `/v1beta` from that dialect. It never
loads or refreshes local upstream OAuth/API credentials in gateway mode. Supported
OpenAI gateway OIDC credential refresh remains a separate gateway authentication
path. Direct-provider auth and endpoint profiles keep their existing behavior.

```mermaid
flowchart LR
  V[Victor: canonical messages/tools + cache intent] --> C[Explicit client dialect]
  C --> G[Sandhi: virtual-key admission + selected backend]
  G --> P[One cache compiler using declared capabilities]
  P --> U[Upstream inference: one physical attempt]
  U --> L[Source disjoint usage + durable receipt]
  L --> W[Client wire projection + convention metadata]
  W --> T[Victor: observed units + qualified pricing]
```

Pass `cache_intent={"mode": "auto_stable_prefix", "lifetime": "5m"}` as a call
option to request placement. Victor forwards it once and generates no automatic
Anthropic markers or compiled plan. Omitting intent leaves the server's declared
default active. The server refuses unsupported lifetimes/capabilities rather than
inventing support; a stable-prefix intent requires a system/tool prefix. Native
options that cannot survive the selected client dialect fail explicitly before
dispatch. Common canonical messages, tools, thinking and reasoning remain typed.

### Token units and monetary qualification

The selected client dialect determines Victor's compatibility `prompt_tokens`
shape: Messages is fresh-only; other client views include cache reads. These
compatibility prompt totals exclude cache writes. Bounded `sandhi_usage.typed_usage`
retains disjoint fresh/read/write/output units, optional zero versus absence,
reasoning inclusion and cache-read availability. `usage_basis` identifies a direct
source or a `gateway_client_projection`. Neither client view is the server's
frozen source receipt. In particular Gemini cannot carry a separate cache-write
category, and reasoning can be folded into inclusive output during translation.

The actual session metrics owner prices known direct sources using typed fresh
input, cache read/write and reasoning folded once. A gateway label/model alias
supplies no trusted source rate context. Gateway monetary cost is therefore
`unavailable`: request/JSON/CSV/task reports preserve units and qualification,
export null/blank monetary values and display “cost unavailable”. Mixed sessions
retain only a clearly named known estimate subtotal; unknown calls do not become
priced zero. Configured direct rates remain estimates, not gateway frozen charges.
Framework TaskResult metadata preserves unavailable pricing explicitly. Older
benchmark/evaluation and conversation exports retain numeric compatibility fields
where zero means unavailable; those sums are known subtotals and cannot establish
a complete gateway monetary total. Their schema migration is a separate follow-on.

Synthetic candidate coverage includes 32 real two-hop calls (four provider labels
× four client dialects × buffered/stream), one source POST and one server marker
each, unsupported TTL refusal before source dispatch, and durable source receipts.
Missing-usage terminal responses are also tested through Victor's default
resilience: one source POST, no successful final event, no retry/fallback and no
settlement receipt. This is local paired qualification, not vendor cache-hit,
subscription entitlement or published-wheel evidence.

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
in this mode. It sends the selected ordinary client dialect with its virtual key to the
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
The prepared consumer also omits `Idempotency-Key` on gateway calls, filtering
caller-provided variants case-insensitively. Run/session/step attribution remains;
local logical-call IDs are correlation, not replay or settlement authority.
The explicit peer follow-on adds counts/cache qualification; mandatory accounting
remains a separate server migration. Direct
provider custom headers retain their separate behavior.
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

## Missing upstream recovery

`502: no upstream registered for this key` means Sandhi resolved the client
credential, but could not find its bound upstream in the live provider registry.
It occurs before model dispatch. A passing Victor configuration check or a valid
Codex login does not prove that the gateway has activated that upstream. Victor
reports this as a terminal configuration error with recovery instructions.

The gateway operator should follow this procedure:

1. Inspect the existing client key's `upstream_ref` through authenticated
   `GET /admin/keys/virtual`, and inspect gateway startup logs for
   `credential not activated`. Use the exact provider and label from the binding;
   creating another label does not repair the existing key. For example,
   `openai:codex` requires provider `openai` and label `codex`.
2. For a Codex subscription route, run `codex login status` on the login-owner
   machine. If signed out, run `codex login`. Follow the
   [official authentication instructions](https://developers.openai.com/codex/auth/).
   Verify the publisher's interpreter still exists and its account check succeeds.
   The publisher should publish only bounded access-token leases; Codex retains
   refresh ownership.
3. Provision the lease into the **active** gateway credential backend. With
   `SANDHI_VAULT_BACKEND=credstore`, inspect `SANDHI_CREDENTIALS_DIR` and the service's
   credential declarations. Provision the exact credential as a regular file
   readable by the service, using the deployment's credential delivery mechanism.
   Writing to the old keyring or SentinelPass backend will not update credstore.
   Sandhi's credstore backend is read-only and skips symlink candidates.
4. Ensure the publisher can refresh both the credential visible to the running
   service and the cached gateway provider handle. A systemd `LoadCredentialEncrypted`
   snapshot alone is not a subscription-lease rotation mechanism: updating the
   source encrypted file does not update the running service's credential snapshot.
   A one-time five-minute lease or repeated gateway restart is not a durable repair.
   If the deployment has no supported rotation path, keep the route unavailable
   until one is configured.
5. After delivery, register the reference using the gateway's authenticated admin
   interface: `POST /admin/keys/reference`. For the example binding above, send:

   ```json
   {"provider":"openai","label":"codex","scheme":"oauth"}
   ```

   This body contains no credential. The operation reads the configured backend,
   validates the lease, and installs the live provider handle. Confirm success;
   `vault_missing` means delivery is still incomplete, while an invalid/expired
   credential error requires fixing the lease before retrying. A non-subscription
   upstream must use its own authentication scheme and endpoint configuration.
6. Retry a small synthetic request with the existing client key and selected model.
   Verify publication continues across a renewal interval and after gateway restart
   before treating recovery as complete. Preserve the client's model allowlist,
   attribution, and budget.

Do not change the selected provider as recovery from this setup failure. A client
key and a Codex login have separate roles: the client key authorizes the gateway
request, and the published access lease enables the gateway's upstream dispatch.

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


## Profile reasoning effort (2026-09-28)

Victor profiles accept a first-class `reasoning_effort` default:

```yaml
profiles:
  default:
    provider: openai
    model: gpt-6-sol
    reasoning_effort: medium
  sol-high:
    provider: openai
    model: gpt-6-sol
    reasoning_effort: high
```

The profile value reaches the orchestrator and both buffered and streaming
requests when the provider reports reasoning support. An explicit per-request
value takes precedence; explicit `None` suppresses the profile default. Leaving
it unset preserves the upstream default. GPT-6 models are recognized by the
OpenAI capability check. The profile schema accepts a cross-provider vocabulary;
the selected upstream model remains authoritative for the supported subset.

[OpenAI documents GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)
with `none`, `low`, `medium` (default), `high`, `xhigh`, and `max`. Values such as
`minimal` or `ultra` used by other models are not claimed to work with Sol.
Reasoning effort controls model deliberation, not tool permissions or policy.

Sandhi already preserves the typed request's `reasoning_effort` and maps it to
`reasoning.effort` for the subscription Responses backend. No gateway-wide
effort default or override is introduced: Victor selects the default, Sandhi
forwards it, and admission still enforces identity, model grants and budgets.
This route uses Responses upstream, including when Victor supplies tools.

TDD reproduced dropped profile values, omitted streaming defaults, missing
GPT-6 capability recognition and unvalidated labels. Regression coverage also
checks unsupported providers and per-request overrides (including `None`).
