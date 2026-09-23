# OAuth for gateway validation

Both `formation_gateway_matrix.py` and `multiagent_gateway_live.py` accept
`--auth-profile /absolute/private/profile.json`. Without it, their existing
virtual-key configuration and reports are unchanged. This opt-in harness adapter
does not change the application's authentication defaults. Sandhi defaults to
OIDC; token compatibility requires an explicit deployment choice.

The profile fixes the HTTPS gateway, trusted CA, local accounting database,
provider/model/grant bindings, and credential broker commands. Use a private
operator-owned file outside the repository:

```json
{
  "schema_version": 1,
  "gateway_url": "https://gateway.example",
  "ca_file": "/absolute/public-ca.pem",
  "database": "/absolute/private/usage.db",
  "credentials": {
    "member": {
      "command": ["/absolute/member-access-token-broker"],
      "issuer": "https://issuer.example/oauth2/openid/sandhi",
      "audience": "sandhi",
      "subject": "member-service-account-uuid"
    },
    "accounting": {
      "command": ["/absolute/accounting-access-token-broker"],
      "issuer": "https://issuer.example/oauth2/openid/sandhi",
      "audience": "sandhi",
      "subject": "accounting-service-account-uuid"
    }
  },
  "accounting": "accounting",
  "routes": {
    "zai": {"credential": "member", "grant": "cloud", "models": ["glm-5.3"]},
    "inferflux": {
      "credential": "member", "grant": "local", "models": ["qwen3-coder-30b"]
    }
  }
}
```

The trusted broker prints exactly one JSON object with `access_token`,
`token_type` (`Bearer`), `expires_on` (Unix seconds), `issuer`, `audience`, and
`subject`. Each freshly acquired credential must match its configured identity
and expire in more than 150 seconds and at most one hour. Bootstrap credentials
stay with the broker. An SSH broker must use a fixed remote command; never
interpolate a member's model, prompt, headers, or metadata into shell arguments.

The observer keeps access tokens in memory, reuses Victor's canonical credential
cache, and renews before a request with a 150-second margin. Acquisition, including
cache-lock waiting, has a 20-second deadline and a bounded cleanup period. Broker
stdout is bounded to 64 KiB; stderr is discarded. POSIX process groups are required.
Cancellation closes inherited pipes and signals only a still-running broker group;
detached children after leader exit are not contained. A broker must not daemonize.

Members receive random per-provider loopback capabilities. Only exact buffered
chat-completion routes and allowlisted models can use them. The adapter strips
incoming credentials, injects the fixed grant, verifies TLS, and refuses redirects.
It never replays an HTTP request after a 401/403 or falls back to another route.
These capabilities are not an OS sandbox against unrestricted same-user tools.

Accounting uses a different issuer/audience/subject identity and only fixed
dashboard, version, run and diagnostics endpoints. On Sandhi 0.9.0, give it a
viewer binding with explicit `allow_diagnostics`, no inference grants and no
administrative writes. The member identity receives inference grants only.
These read permissions are deployment-wide; they do not provide tenant isolation.

Keep the existing 120-second gateway deadline, correlation, usage conservation,
dashboard, deliverable and independent pytest assertions. Offline broker tests do
not establish actual-member SSO, token renewal during a live team, origin identity,
cache execution, cancellation or C5 acceptance. Record those separately in G49 and
InferFlux #184. Direct InferFlux access remains an explicitly selected authenticated
route and bypasses Sandhi metering and budgets; authentication failures never select it.
