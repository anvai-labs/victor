---
fep: "0004"
title: "Provider OAuth Authentication (Subscription-Based Access)"
type: Standards Track
status: Draft
created: 2026-03-05
modified: 2026-03-05
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/TBD
---

# FEP-0004: Provider OAuth Authentication

## Summary

Add OAuth 2.0 PKCE authentication to Victor's provider layer so users can authenticate
with their existing LLM subscriptions (ChatGPT Plus/Pro, Qwen Coding Plan, etc.) instead
of managing API keys. Reuses Victor's existing `SSOAuthenticator` infrastructure.

## Motivation

Users with ChatGPT Plus/Pro/Enterprise or Qwen Coding subscriptions currently cannot use
those subscriptions through Victor — they must obtain separate API keys with pay-per-token
billing. OAuth lets them reuse flat-rate subscriptions they already pay for.

### Provider OAuth Landscape (March 2026)

```
Provider          Subscription OAuth    Status          Notes
─────────────────────────────────────────────────────────────────────
OpenAI Codex      Yes                   Supported       Public client ID, PKCE
Qwen (Alibaba)    Yes                   Supported       qwen.ai OAuth, 60 req/min free
Google Gemini     Restricted            Partially       Actively blocking unauthorized 3P OAuth
Anthropic Claude  No                    Banned          Banned 3P OAuth (Jan 2026), ToS updated Feb 2026
xAI Grok          No                    API-key only    X Premium chat != API access, pay-per-token API
DeepSeek          No                    API-key only    No subscription tier, free credits on signup
Mistral (Le Chat) No (for API)          API-key only    OAuth for MCP tool connections TO Le Chat only
Kimi K2.5         No                    API-key only    Open-source, API-key or self-host
Z.AI (GLM)        No                    API-key only    GLM Coding Plan uses API keys, not OAuth
```

### Goals

1. OpenAI Codex OAuth via existing `SSOAuthenticator` PKCE flow
2. Qwen OAuth as second provider
3. Extensible pattern for future providers
4. Token persistence + auto-refresh
5. Zero-config upgrade path (existing API key users unaffected)

### Non-Goals

- Anthropic Claude OAuth (explicitly banned for 3P tools, Jan 2026)
- Google Gemini OAuth (actively restricting/banning 3P access)
- xAI Grok OAuth (no subscription-based API access)
- DeepSeek OAuth (no subscription tier exists)
- Mistral Le Chat OAuth (OAuth is for tool connections TO Le Chat, not API access)
- Multi-user / commercial token sharing
- Device code flow (headless environments) — future FEP

## Proposed Change

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User's Browser                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  auth.openai.com/oauth/authorize                      │  │
│  │  OR qwen.ai/oauth/authorize                           │  │
│  │                                                       │  │
│  │  User signs in with existing subscription account     │  │
│  └──────────────────────┬────────────────────────────────┘  │
└─────────────────────────┼───────────────────────────────────┘
                          │ redirect with ?code=...
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              localhost:8400/callback                         │
│              (SSOAuthenticator local server)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ auth code
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SSOAuthenticator._exchange_code()               │
│                                                             │
│  POST auth.openai.com/oauth/token                           │
│    grant_type=authorization_code                            │
│    code=<auth_code>                                         │
│    code_verifier=<pkce_verifier>                            │
│    client_id=app_EMoamEEZ73f0CkXaXp7hrann                  │
│                                                             │
│  Response: { access_token, refresh_token, expires_in }      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              OAuthTokenManager                               │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Persist to   │  │ Auto-refresh │  │ Inject into       │  │
│  │ ~/.victor/   │  │ on expiry    │  │ Provider client   │  │
│  │ oauth.yaml   │  │ (5min grace) │  │ as Bearer token   │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ access_token
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenAIProvider / QwenProvider                    │
│                                                             │
│  AsyncOpenAI(api_key=access_token, base_url=...)            │
│                                                             │
│  chat() ──► _ensure_valid_token() ──► API call              │
│  stream() ─► _ensure_valid_token() ──► API call             │
└─────────────────────────────────────────────────────────────┘
```

### Token Lifecycle

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  No Token │───►│  Login   │───►│  Active  │───►│ Expiring │
│  (first   │    │  (browser│    │  (use as │    │  (<5min  │
│   run)    │    │   PKCE)  │    │  Bearer) │    │   left)  │
└──────────┘    └──────────┘    └────┬─────┘    └────┬─────┘
                     ▲               │               │
                     │               │  on each      │ auto-refresh
                     │               │  API call     │ via refresh_token
                     │               ▼               ▼
                ┌────┴─────┐    ┌──────────┐    ┌──────────┐
                │  Refresh │◄───│  Check   │◄───│ Refreshed│
                │  Failed  │    │  Expiry  │    │  (new    │
                │(re-login)│    │          │    │  token)  │
                └──────────┘    └──────────┘    └──────────┘
```

### File Changes

```
Modified:
  victor/workflows/services/credentials.py   +30 lines  (SSOProvider enum, SSOConfig factories)
  victor/providers/openai_provider.py         +45 lines  (OAuth auth mode)
  victor/config/provider_config_registry.py   +10 lines  (auth_mode config)

New:
  victor/providers/oauth_manager.py           ~120 lines (token persistence + refresh)
  tests/unit/providers/test_oauth_manager.py  ~200 lines (TDD tests)
  tests/unit/providers/test_openai_oauth.py   ~150 lines (OpenAI OAuth tests)
```

### API Changes

```python
# OpenAI — API key (unchanged)
provider = OpenAIProvider(api_key="sk-...")

# OpenAI — OAuth (Codex subscription)
provider = OpenAIProvider(auth_mode="oauth")

# Qwen — API key
provider = QwenProvider(api_key="sk-...")

# Qwen — OAuth (Coding Plan subscription)
provider = QwenProvider(auth_mode="oauth")

# Z.AI — Standard API
provider = ZAIProvider(api_key="zai-key")

# Z.AI — Coding Plan (dedicated endpoint)
provider = ZAIProvider(api_key="zai-key", coding_plan=True)

# Z.AI — Named endpoint
provider = ZAIProvider(api_key="zai-key", endpoint="coding")  # or "standard", "china", "anthropic"
```

### New Classes

```python
@dataclass
class OAuthProviderConfig:
    """OAuth endpoints for an LLM provider."""
    provider_name: str           # "openai", "qwen"
    sso_provider: SSOProvider    # enum value
    issuer_url: str              # "https://auth.openai.com"
    client_id: str               # public client ID
    scopes: List[str]            # ["openid", "profile", "email", "offline_access"]
    token_endpoint: str          # "/oauth/token" (override for non-standard)
    redirect_port: int = 8400

# Registry
OAUTH_PROVIDERS = {
    "openai": OAuthProviderConfig(
        provider_name="openai",
        sso_provider=SSOProvider.OPENAI_CODEX,
        issuer_url="https://auth.openai.com",
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        scopes=["openid", "profile", "email", "offline_access"],
        token_endpoint="/oauth/token",
    ),
    "qwen": OAuthProviderConfig(
        provider_name="qwen",
        sso_provider=SSOProvider.QWEN,
        issuer_url="https://qwen.ai",
        client_id="<qwen-public-client-id>",  # TBD
        scopes=["openid", "profile", "email", "offline_access"],
        token_endpoint="/oauth/token",
    ),
}

class OAuthTokenManager:
    """Manages OAuth token lifecycle for LLM providers."""

    def __init__(self, provider: str, storage_dir: Path = None): ...
    async def get_valid_token(self) -> str: ...
    async def login(self) -> SSOTokens: ...
    async def refresh(self) -> SSOTokens: ...
    def _load_cached(self) -> Optional[SSOTokens]: ...
    def _save(self, tokens: SSOTokens) -> None: ...
    def clear(self) -> None: ...
```

### Configuration

```yaml
# ~/.victor/config.yaml
providers:
  openai:
    auth_mode: oauth           # triggers OAuth flow

  qwen:
    auth_mode: oauth

# Environment variable override
# VICTOR_OPENAI_AUTH_MODE=oauth
```

### Token Storage

```yaml
# ~/.victor/oauth_tokens.yaml (0600 permissions)
openai:
  access_token: "eyJ..."
  refresh_token: "eyJ..."
  expires_at: "2026-03-05T14:30:00Z"
  scopes: ["openid", "profile", "email", "offline_access"]
qwen:
  access_token: "..."
  refresh_token: "..."
  expires_at: "..."
```

## Benefits

- **Zero API key management** — sign in once with existing subscription
- **Flat-rate pricing** — use subscription instead of per-token billing
- **Existing infrastructure** — `SSOAuthenticator` handles 80% of the work
- **Non-breaking** — `auth_mode` defaults to `"api_key"`, zero impact on existing users

## Drawbacks and Alternatives

### Drawbacks

- **Provider policy risk** — providers may restrict 3P OAuth (Anthropic already has)
- **Personal use only** — subscription OAuth not for commercial/multi-user
- **Browser required** — headless environments need device code flow (future FEP)

### Alternatives

1. **Reverse proxy approach** — intercept/transform SDK requests (like opencode-codex-auth)
   - Rejected: too fragile, couples to backend API format changes

2. **Separate CLI tool** — standalone `victor auth login`
   - Partially adopted: CLI integration is Phase 2, core is library-first

## Implementation Plan

### Phase 1: Core (DONE)

- [x] FEP document
- [x] `OAuthTokenManager` with persistence + refresh
- [x] `SSOProvider.OPENAI_CODEX` / `QWEN` enums + factory methods
- [x] `OpenAIProvider` `auth_mode="oauth"` support
- [x] `QwenProvider` with OAuth + API-key dual auth (OpenAI-compatible)
- [x] `ZAIProvider` Coding Plan support (`coding_plan=True`, `endpoint="coding"`)
- [x] `ZAIConfig`, `ZAICodingPlanConfig`, `QwenConfig` in provider_config_registry
- [x] 112 unit tests passing (TDD)

### Phase 2: CLI + Config

- [ ] `victor auth login --provider openai` command
- [ ] `victor auth login --provider qwen` command
- [ ] `victor auth status` / `victor auth logout`

### Phase 3: Additional Providers

- [ ] Qwen OAuth integration
- [ ] Provider OAuth registry for extensibility

## Compatibility

- **Breaking change**: No
- **Migration required**: No
- **Minimum Python**: 3.10
- **New dependencies**: None (aiohttp already a dependency)

## Migration Path

Existing API-key configurations remain the default and require no changes. Users
who opt into subscription OAuth set `auth_mode: oauth` for one provider and run
that provider's login flow once CLI support is available. Token files are written
with user-only permissions, and reverting to `auth_mode: api_key` immediately
restores the existing authentication path without converting stored credentials.

## Unresolved Questions

- Which providers will formally permit third-party clients to use subscription
  OAuth tokens, and what scopes will they support?
- Should headless environments use device authorization, an out-of-band browser
  callback, or remain API-key-only?
- What token revocation and reauthentication behavior should the CLI expose when
  a provider changes scopes or invalidates a refresh token?
- Should OAuth token storage support an operating-system keychain backend before
  the feature advances beyond draft status?

## References

- [OpenAI Codex Auth Docs](https://developers.openai.com/codex/auth/)
- [opencode-openai-codex-auth](https://github.com/numman-ali/opencode-openai-codex-auth)
- [Cline OpenAI Codex OAuth](https://cline.bot/blog/introducing-openai-codex-oauth)
- [Qwen Code Auth Docs](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/)
- [Alibaba Qwen Coding Plan](https://alternativeto.net/news/2026/2/alibaba-cloud-model-studio-introduces-qwen-coding-plan-subscription/)
- [Anthropic Bans 3P OAuth](https://winbuzzer.com/2026/02/19/anthropic-bans-claude-subscription-oauth-in-third-party-apps-xcxwbn/)
- [Google Restricts 3P OAuth](https://www.trendingtopics.eu/google-blocks-paying-ai-subscribers-using-third-party-openclaw-tool/)

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
