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

"""Entra (Azure AD) delegated / SSO sign-in for capturing *user* identity.

Where :mod:`victor.core.identity.sources` mints *app-only* tokens (the service
acting as itself), this module handles the **delegated** OIDC authorization-code
flow: a human signs in and we learn *who they are*. It is what lets a HITL
approval record the approver's identity ("approved by …").

Identity is resolved by calling Microsoft Graph ``/me`` with the delegated
access token rather than by validating the id_token JWT locally — Graph
validates the token, so no JWKS/JWT dependency is needed (consistent with the
rest of this package's raw-HTTP approach).
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional

from victor.core.identity.sources import ENTRA_AUTHORITY

_GRAPH_ME_URL = "https://graph.microsoft.com/v1.0/me"
# Minimal delegated scopes: OIDC identity + read the signed-in user's profile.
_DEFAULT_SCOPES = ("openid", "profile", "email", "User.Read")


@dataclass(frozen=True)
class UserIdentity:
    """The authenticated approver's identity, from Microsoft Graph ``/me``."""

    subject: Optional[str]  # stable directory object id (oid)
    name: Optional[str]  # display name
    email: Optional[str]  # mail or userPrincipalName

    def label(self) -> str:
        """Human-readable identifier for display/audit (name <email>)."""
        if self.name and self.email:
            return f"{self.name} <{self.email}>"
        return self.email or self.name or self.subject or "unknown"


@dataclass
class OidcConfig:
    """Configuration for the delegated authorization-code flow."""

    tenant_id: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: List[str] = field(default_factory=lambda: list(_DEFAULT_SCOPES))
    authority: str = ENTRA_AUTHORITY

    @classmethod
    def from_env(cls) -> Optional["OidcConfig"]:
        """Build from env, or return None when delegated SSO is not configured.

        Reuses the app's Entra registration (``AZURE_*`` / ``TEAMS_*`` aliases)
        and requires a redirect URI (``VICTOR_HITL_SSO_REDIRECT_URI``) pointing
        at the ``/hitl/auth/callback`` endpoint. SSO is opt-in: with no redirect
        URI configured this returns None and callers fall back to anonymous
        signed-link decisions.
        """
        tenant = os.getenv("AZURE_TENANT_ID") or os.getenv("TEAMS_TENANT_ID")
        client = os.getenv("AZURE_CLIENT_ID") or os.getenv("TEAMS_CLIENT_ID")
        secret = os.getenv("AZURE_CLIENT_SECRET") or os.getenv("TEAMS_CLIENT_SECRET")
        redirect_uri = os.getenv("VICTOR_HITL_SSO_REDIRECT_URI")
        if not (tenant and client and secret and redirect_uri):
            return None
        return cls(
            tenant_id=tenant,
            client_id=client,
            client_secret=secret,
            redirect_uri=redirect_uri,
        )

    def _endpoint(self, kind: str) -> str:
        return f"{self.authority.rstrip('/')}/{self.tenant_id}/oauth2/v2.0/{kind}"


def build_authorize_url(config: OidcConfig, *, state: str, nonce: Optional[str] = None) -> str:
    """Build the Entra authorization-code URL to redirect the approver to."""
    params = {
        "client_id": config.client_id,
        "response_type": "code",
        "redirect_uri": config.redirect_uri,
        "response_mode": "query",
        "scope": " ".join(config.scopes),
        "state": state,
    }
    if nonce:
        params["nonce"] = nonce
    return f"{config._endpoint('authorize')}?{urllib.parse.urlencode(params)}"


async def exchange_code(config: OidcConfig, code: str) -> str:
    """Exchange an authorization ``code`` for a delegated access token."""
    import aiohttp

    form = {
        "grant_type": "authorization_code",
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "code": code,
        "redirect_uri": config.redirect_uri,
        "scope": " ".join(config.scopes),
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(config._endpoint("token"), data=form) as resp:
            payload = await resp.json()
            if resp.status != 200 or "access_token" not in payload:
                detail = payload.get("error_description") or payload.get("error") or payload
                raise RuntimeError(f"OIDC code exchange failed ({resp.status}): {detail}")
            return str(payload["access_token"])


async def fetch_user_identity(access_token: str) -> UserIdentity:
    """Resolve the signed-in user's identity via Microsoft Graph ``/me``."""
    import aiohttp

    headers = {"Authorization": f"Bearer {access_token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(_GRAPH_ME_URL, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"Graph /me failed ({resp.status}): {data}")
    return UserIdentity(
        subject=data.get("id"),
        name=data.get("displayName"),
        email=data.get("mail") or data.get("userPrincipalName"),
    )


async def resolve_identity_from_code(config: OidcConfig, code: str) -> UserIdentity:
    """Full delegated round-trip: code -> access token -> ``/me`` identity."""
    access_token = await exchange_code(config, code)
    return await fetch_user_identity(access_token)
