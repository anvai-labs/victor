# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SSO / OAuth 2.0 / OIDC login for Victor CLI and subscription providers.

Moved from ``victor.workflows.services.credentials`` (layering round 2), which
re-exports these names unchanged so existing imports keep working.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SSOProvider(Enum):
    """Supported SSO providers."""

    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    ONELOGIN = "onelogin"
    PING = "ping"
    KEYCLOAK = "keycloak"
    GENERIC_OIDC = "oidc"
    GENERIC_SAML = "saml"
    # LLM provider subscription OAuth (FEP-0004)
    OPENAI_CODEX = "openai_codex"
    QWEN = "qwen"
    GOOGLE_GEMINI = "google_gemini"
    GITHUB_COPILOT = "github_copilot"


@dataclass
class SSOConfig:
    """SSO / OIDC configuration.

    Supports enterprise identity providers for Victor CLI authentication.

    Attributes:
        provider: SSO provider type
        issuer_url: OIDC issuer URL
        client_id: OAuth client ID
        client_secret: OAuth client secret (optional for PKCE)
        redirect_uri: OAuth redirect URI
        scopes: OAuth scopes to request
        audience: API audience (for JWT validation)
        organization_id: Okta organization ID
        use_pkce: Use PKCE flow (recommended)
    """

    provider: SSOProvider
    issuer_url: str
    client_id: str
    client_secret: Optional[str] = None
    redirect_uri: str = "http://localhost:8400/callback"
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    audience: Optional[str] = None
    organization_id: Optional[str] = None  # Okta org ID
    use_pkce: bool = True
    # Config-driven overrides (eliminate per-provider if/elif in SSOAuthenticator)
    authorize_path: Optional[str] = None  # e.g. "/o/oauth2/v2/auth" for Google
    token_url: Optional[str] = None  # Full token URL when host differs from issuer
    extra_auth_params: Optional[Dict[str, str]] = None  # e.g. {"access_type": "offline"}
    # Device code flow (for GitHub Copilot, headless environments)
    use_device_flow: bool = False  # Use device code flow instead of browser redirect
    device_code_url: Optional[str] = None  # URL for device code requests

    @classmethod
    def for_okta(
        cls,
        domain: str,
        client_id: str,
        client_secret: Optional[str] = None,
    ) -> "SSOConfig":
        """Create Okta SSO configuration.

        Args:
            domain: Okta domain (e.g., 'mycompany.okta.com')
            client_id: OAuth client ID
            client_secret: OAuth client secret (optional for PKCE)
        """
        return cls(
            provider=SSOProvider.OKTA,
            issuer_url=f"https://{domain}",
            client_id=client_id,
            client_secret=client_secret,
            scopes=["openid", "profile", "email", "groups"],
        )

    @classmethod
    def for_azure_ad(
        cls,
        tenant_id: str,
        client_id: str,
        client_secret: Optional[str] = None,
    ) -> "SSOConfig":
        """Create Azure AD SSO configuration."""
        return cls(
            provider=SSOProvider.AZURE_AD,
            issuer_url=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            client_id=client_id,
            client_secret=client_secret,
        )

    @classmethod
    def for_google(cls, client_id: str, client_secret: str) -> "SSOConfig":
        """Create Google SSO configuration."""
        return cls(
            provider=SSOProvider.GOOGLE,
            issuer_url="https://accounts.google.com",
            client_id=client_id,
            client_secret=client_secret,
        )

    @classmethod
    def for_openai_codex(cls) -> "SSOConfig":
        """Create OpenAI Codex subscription OAuth configuration.

        Uses OpenAI's public OAuth client for ChatGPT Plus/Pro/Enterprise
        subscription-based access. See FEP-0004.

        Scopes and endpoints sourced from openai/codex (codex-rs/login/src/server.rs).
        """
        return cls(
            provider=SSOProvider.OPENAI_CODEX,
            issuer_url="https://auth.openai.com",
            client_id="app_EMoamEEZ73f0CkXaXp7hrann",
            scopes=[
                "openid",
                "profile",
                "email",
                "offline_access",
                "api.connectors.read",
                "api.connectors.invoke",
            ],
            redirect_uri="http://localhost:1455/auth/callback",
            use_pkce=True,
        )

    @classmethod
    def for_qwen(cls) -> "SSOConfig":
        """Create Qwen (Alibaba) subscription OAuth configuration.

        Uses Qwen's OAuth for Coding Plan subscription access. See FEP-0004.
        """
        return cls(
            provider=SSOProvider.QWEN,
            issuer_url="https://chat.qwen.ai",
            client_id="qwen-code",
            scopes=["openid", "profile", "email", "offline_access"],
            use_pkce=True,
        )

    @classmethod
    def for_google_gemini(cls) -> "SSOConfig":
        """Create Google Gemini subscription OAuth configuration.

        Uses Google's public OAuth client (same as Gemini CLI).
        Google AI Pro/Ultra subscribers can use this to access Gemini models
        via their subscription instead of per-token API billing.

        Client ID and secret are public (standard for Google installed apps).
        Source: gemini-cli/packages/core/src/code_assist/oauth2.ts
        """
        return cls(
            provider=SSOProvider.GOOGLE_GEMINI,
            issuer_url="https://accounts.google.com",
            client_id=(
                "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j" ".apps.googleusercontent.com"
            ),
            client_secret="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",
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ],
            authorize_path="/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            extra_auth_params={"access_type": "offline"},
            redirect_uri="http://127.0.0.1:8401/oauth2callback",
            use_pkce=True,
        )

    @classmethod
    def for_github_copilot(cls) -> "SSOConfig":
        """Create GitHub Copilot subscription OAuth configuration.

        Uses OAuth device code flow (no browser redirect needed).
        Requires a GitHub Copilot subscription (Individual, Business, or Enterprise).
        Third-party access is officially supported via Copilot SDK (April 2026).

        Source: github/copilot-sdk/docs/setup/github-oauth.md
        """
        return cls(
            provider=SSOProvider.GITHUB_COPILOT,
            issuer_url="https://github.com",
            client_id="",  # Must be configured per-app via GitHub OAuth App
            scopes=["copilot"],
            use_pkce=False,
            use_device_flow=True,
            device_code_url="https://github.com/login/device/code",
            token_url="https://github.com/login/oauth/access_token",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "issuer_url": self.issuer_url,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "scopes": self.scopes,
            "audience": self.audience,
            "organization_id": self.organization_id,
            "use_pkce": self.use_pkce,
            "authorize_path": self.authorize_path,
            "token_url": self.token_url,
            "extra_auth_params": self.extra_auth_params,
            "use_device_flow": self.use_device_flow,
            "device_code_url": self.device_code_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSOConfig":
        return cls(
            provider=SSOProvider(data["provider"]),
            issuer_url=data["issuer_url"],
            client_id=data["client_id"],
            client_secret=data.get("client_secret"),
            redirect_uri=data.get("redirect_uri", "http://localhost:8400/callback"),
            scopes=data.get("scopes", ["openid", "profile", "email"]),
            audience=data.get("audience"),
            organization_id=data.get("organization_id"),
            use_pkce=data.get("use_pkce", True),
            authorize_path=data.get("authorize_path"),
            token_url=data.get("token_url"),
            extra_auth_params=data.get("extra_auth_params"),
            use_device_flow=data.get("use_device_flow", False),
            device_code_url=data.get("device_code_url"),
        )


@dataclass
class SSOTokens:
    """OAuth tokens from SSO authentication."""

    access_token: str
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    scopes: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": self.scopes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSOTokens":
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            id_token=data.get("id_token"),
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scopes=data.get("scopes", []),
        )


class SSOAuthenticator:
    """SSO authentication handler.

    Supports OAuth 2.0 / OIDC flows with enterprise IdPs.

    Example:
        config = SSOConfig.for_okta("mycompany.okta.com", "client_id")
        auth = SSOAuthenticator(config)

        # Start browser-based login
        tokens = await auth.login()

        # Refresh tokens
        new_tokens = await auth.refresh(tokens.refresh_token)
    """

    def __init__(self, config: SSOConfig):
        self.config = config
        self._server: Optional[Any] = None

    async def login(self, timeout: int = 120) -> SSOTokens:
        """Perform SSO login via browser or device code flow.

        Uses device code flow if config.use_device_flow is True,
        otherwise uses browser-based PKCE flow with local callback server.

        Args:
            timeout: Maximum time to wait for login

        Returns:
            SSOTokens with access/refresh tokens
        """
        # Device code flow (GitHub Copilot, headless environments)
        if self.config.use_device_flow:
            return await self._device_code_login(timeout)

        import secrets
        import urllib.parse
        import webbrowser

        # Generate state and PKCE verifier
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)

        if self.config.use_pkce:
            import hashlib

            code_challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
                .decode()
                .rstrip("=")
            )
            code_challenge_method = "S256"
        else:
            code_challenge = None
            code_challenge_method = None

        # Build authorization URL
        auth_params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
        }

        if code_challenge:
            auth_params["code_challenge"] = code_challenge
            auth_params["code_challenge_method"] = code_challenge_method

        if self.config.audience:
            auth_params["audience"] = self.config.audience

        # OpenAI Codex-specific params (from openai/codex server.rs)
        if self.config.provider == SSOProvider.OPENAI_CODEX:
            auth_params["codex_cli_simplified_flow"] = "true"
            auth_params["originator"] = "victor"

        # Merge provider-specific extra params (e.g. access_type=offline for Google)
        if self.config.extra_auth_params:
            auth_params.update(self.config.extra_auth_params)

        # Config-driven authorization path (new providers just set authorize_path)
        if self.config.authorize_path:
            authorize_path = self.config.authorize_path
        elif self.config.provider == SSOProvider.OPENAI_CODEX:
            authorize_path = "/oauth/authorize"
        elif self.config.provider == SSOProvider.OKTA:
            authorize_path = "/oauth2/v1/authorize"
        else:
            authorize_path = "/authorize"

        auth_url = f"{self.config.issuer_url}{authorize_path}?{urllib.parse.urlencode(auth_params)}"

        # Start local callback server
        callback_result: Dict[str, Any] = {}
        callback_event = asyncio.Event()

        async def handle_callback(request):
            from aiohttp import web

            code = request.query.get("code")
            returned_state = request.query.get("state")
            error = request.query.get("error")

            if error:
                callback_result["error"] = error
            elif returned_state != state:
                callback_result["error"] = "State mismatch"
            else:
                callback_result["code"] = code

            callback_event.set()

            return web.Response(
                text="<html><body><h1>Login successful!</h1>"
                "<p>You can close this window.</p></body></html>",
                content_type="text/html",
            )

        # Start server
        from aiohttp import web

        app = web.Application()
        parsed_redirect = urllib.parse.urlparse(self.config.redirect_uri)
        callback_path = parsed_redirect.path or "/callback"
        app.router.add_get(callback_path, handle_callback)

        port = parsed_redirect.port or 8400

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", port)
        await site.start()

        try:
            # Open browser
            logger.info("Opening browser for SSO login...")
            webbrowser.open(auth_url)

            # Wait for callback
            try:
                await asyncio.wait_for(callback_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError("SSO login timed out")

            if "error" in callback_result:
                raise ValueError(f"SSO error: {callback_result['error']}")

            # Exchange code for tokens
            return await self._exchange_code(
                callback_result["code"],
                code_verifier if self.config.use_pkce else None,
            )

        finally:
            await runner.cleanup()

    @staticmethod
    def _get_ssl_context():
        """Create SSL context using certifi certs (fixes macOS venv SSL errors)."""
        import ssl

        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return None  # Fall back to system default

    async def _device_code_login(self, timeout: int = 120) -> SSOTokens:
        """Perform OAuth device code flow (RFC 8628).

        Used by GitHub Copilot and other providers that support headless auth.
        User visits a URL, enters a code, and the CLI polls for completion.

        Args:
            timeout: Maximum time to wait for user to complete auth

        Returns:
            SSOTokens with access token
        """
        import aiohttp

        device_code_url = self.config.device_code_url
        if not device_code_url:
            device_code_url = f"{self.config.issuer_url}/login/device/code"

        # Resolve token URL
        if self.config.token_url:
            token_url = self.config.token_url
        else:
            token_url = f"{self.config.issuer_url}/login/oauth/access_token"

        # Step 1: Request device code
        ssl_ctx = self._get_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None

        headers = {"Accept": "application/json"}
        data = {
            "client_id": self.config.client_id,
            "scope": " ".join(self.config.scopes),
        }

        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(device_code_url, data=data, headers=headers) as response:
                if response.status != 200:
                    error = await response.text()
                    raise ValueError(f"Device code request failed: {error}")
                result = await response.json()

        device_code = result["device_code"]
        user_code = result["user_code"]
        verification_uri = result.get("verification_uri", result.get("verification_url", ""))
        interval = result.get("interval", 5)
        expires_in = result.get("expires_in", timeout)

        # Step 2: Show user the code and URL
        import webbrowser

        print(f"\n  Visit: {verification_uri}")
        print(f"  Enter code: {user_code}\n")

        try:
            webbrowser.open(verification_uri)
        except Exception:
            pass  # Browser open is best-effort

        # Step 3: Poll for token
        poll_data = {
            "client_id": self.config.client_id,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        if self.config.client_secret:
            poll_data["client_secret"] = self.config.client_secret

        import time

        deadline = time.time() + min(timeout, expires_in)
        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None

        async with aiohttp.ClientSession(connector=connector) as session:
            while time.time() < deadline:
                await asyncio.sleep(interval)

                async with session.post(token_url, data=poll_data, headers=headers) as response:
                    token_result = await response.json()

                error = token_result.get("error")
                if error == "authorization_pending":
                    continue
                elif error == "slow_down":
                    interval = min(interval + 5, 30)
                    continue
                elif error:
                    raise ValueError(
                        f"Device code auth failed: {error} - "
                        f"{token_result.get('error_description', '')}"
                    )

                # Success
                access_token = token_result["access_token"]
                expires_in_secs = token_result.get("expires_in", 28800)

                return SSOTokens(
                    access_token=access_token,
                    refresh_token=token_result.get("refresh_token"),
                    token_type=token_result.get("token_type", "Bearer"),
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in_secs),
                    scopes=self.config.scopes,
                )

        raise TimeoutError("Device code authentication timed out")

    async def _exchange_code(
        self,
        code: str,
        code_verifier: Optional[str],
    ) -> SSOTokens:
        """Exchange authorization code for tokens."""
        import aiohttp

        # Config-driven token URL (new providers just set token_url)
        if self.config.token_url:
            token_url = self.config.token_url
        elif self.config.provider == SSOProvider.OKTA:
            token_url = f"{self.config.issuer_url}/oauth2/v1/token"
        else:
            token_url = f"{self.config.issuer_url}/oauth/token"

        data = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "code": code,
            "redirect_uri": self.config.redirect_uri,
        }

        if code_verifier:
            data["code_verifier"] = code_verifier

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        ssl_ctx = self._get_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error = await response.text()
                    raise ValueError(f"Token exchange failed: {error}")

                result = await response.json()

                expires_at = None
                if "expires_in" in result:
                    expires_at = datetime.now(timezone.utc) + timedelta(
                        seconds=result["expires_in"]
                    )

                return SSOTokens(
                    access_token=result["access_token"],
                    refresh_token=result.get("refresh_token"),
                    id_token=result.get("id_token"),
                    token_type=result.get("token_type", "Bearer"),
                    expires_at=expires_at,
                    scopes=result.get("scope", "").split(),
                )

    async def refresh(self, refresh_token: str) -> SSOTokens:
        """Refresh access token using refresh token."""
        import aiohttp

        # Config-driven token URL
        if self.config.token_url:
            token_url = self.config.token_url
        elif self.config.provider == SSOProvider.OKTA:
            token_url = f"{self.config.issuer_url}/oauth2/v1/token"
        else:
            token_url = f"{self.config.issuer_url}/oauth/token"

        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "refresh_token": refresh_token,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        ssl_ctx = self._get_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error = await response.text()
                    raise ValueError(f"Token refresh failed: {error}")

                result = await response.json()

                expires_at = None
                if "expires_in" in result:
                    expires_at = datetime.now(timezone.utc) + timedelta(
                        seconds=result["expires_in"]
                    )

                return SSOTokens(
                    access_token=result["access_token"],
                    refresh_token=result.get("refresh_token", refresh_token),
                    id_token=result.get("id_token"),
                    token_type=result.get("token_type", "Bearer"),
                    expires_at=expires_at,
                )

    async def logout(self, id_token: Optional[str] = None) -> None:
        """Logout from SSO provider."""
        import webbrowser

        logout_url = f"{self.config.issuer_url}/logout"

        params = {"client_id": self.config.client_id}
        if id_token:
            params["id_token_hint"] = id_token

        if self.config.provider == SSOProvider.OKTA:
            logout_url = f"{self.config.issuer_url}/oauth2/v1/logout"

        import urllib.parse

        full_url = f"{logout_url}?{urllib.parse.urlencode(params)}"
        webbrowser.open(full_url)
