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

"""Concrete :class:`TokenCredential` strategies.

Each class implements exactly one credential *flow* (Single Responsibility) and
is interchangeable behind the protocol (Liskov). They are composed — not
subclassed — by callers/factories, and wrapped with
:class:`~victor.core.identity.cache.CachingTokenCredential` for caching.

Flows provided:
    * :class:`ClientSecretCredential`     — Entra client-credentials, secret.
    * :class:`ClientAssertionCredential`  — Entra client-credentials, certificate
      (signed JWT assertion; secret-less, the Entra best practice).
    * :class:`ManagedIdentityCredential`  — Azure IMDS, no secret stored.
    * :class:`StaticTokenCredential`      — wraps an already-acquired token
      (e.g. injected via tool context) for backward compatibility.
    * :class:`ChainedTokenCredential`     — first-success fallback, à la
      ``DefaultAzureCredential``.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Union

from victor.core.identity.protocols import (
    AccessToken,
    CredentialUnavailableError,
    TokenCredential,
)

logger = logging.getLogger(__name__)

# Entra (Azure AD) public-cloud authority. Sovereign clouds override via the
# ``authority`` argument (e.g. https://login.microsoftonline.us).
ENTRA_AUTHORITY = "https://login.microsoftonline.com"
GRAPH_DEFAULT_SCOPE = "https://graph.microsoft.com/.default"
_CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"


def _token_endpoint(authority: str, tenant_id: str) -> str:
    return f"{authority.rstrip('/')}/{tenant_id}/oauth2/v2.0/token"


async def _post_token(url: str, form: dict) -> AccessToken:
    """POST a client-credentials request and parse the token response.

    ``aiohttp`` is imported lazily (it is an optional dependency, and importing
    at call time keeps this module import-light).
    """
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form) as resp:
            payload = await resp.json()
            if resp.status != 200 or "access_token" not in payload:
                detail = payload.get("error_description") or payload.get("error") or payload
                raise RuntimeError(f"Entra token request failed ({resp.status}): {detail}")
            expires_in = int(payload.get("expires_in", 3600))
            return AccessToken(
                token=str(payload["access_token"]),
                expires_on=time.time() + expires_in,
            )


class ClientSecretCredential:
    """Entra client-credentials flow using a client secret.

    ``tenant_id`` only scopes the token endpoint URL — it is **not** a
    credential. The ``client_secret`` is what authenticates the app.
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        *,
        authority: str = ENTRA_AUTHORITY,
    ) -> None:
        if not (tenant_id and client_id and client_secret):
            raise ValueError(
                "ClientSecretCredential requires tenant_id, client_id and client_secret"
            )
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._authority = authority

    async def get_token(self, *scopes: str) -> AccessToken:
        scope = " ".join(scopes) or GRAPH_DEFAULT_SCOPE
        form = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": scope,
        }
        return await _post_token(_token_endpoint(self._authority, self._tenant_id), form)


class ClientAssertionCredential:
    """Entra client-credentials flow using a signed JWT client assertion.

    This is the secret-less, certificate-based path (the Entra best practice).
    The signed assertion is supplied by the caller — either as a string or a
    zero-arg callable that returns a freshly-signed assertion — so this module
    needs no JWT/crypto dependency and stays agnostic to how the certificate is
    held (file, Key Vault, HSM, ...).
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        assertion: Union[str, Callable[[], str]],
        *,
        authority: str = ENTRA_AUTHORITY,
    ) -> None:
        if not (tenant_id and client_id and assertion):
            raise ValueError(
                "ClientAssertionCredential requires tenant_id, client_id and assertion"
            )
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._assertion = assertion
        self._authority = authority

    def _resolve_assertion(self) -> str:
        return self._assertion() if callable(self._assertion) else self._assertion

    async def get_token(self, *scopes: str) -> AccessToken:
        scope = " ".join(scopes) or GRAPH_DEFAULT_SCOPE
        form = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_assertion_type": _CLIENT_ASSERTION_TYPE,
            "client_assertion": self._resolve_assertion(),
            "scope": scope,
        }
        return await _post_token(_token_endpoint(self._authority, self._tenant_id), form)


class ManagedIdentityCredential:
    """Azure Managed Identity via the instance metadata endpoint (IMDS).

    No secret is stored; the platform issues tokens to the assigned identity.
    Raises :class:`CredentialUnavailableError` when not running on Azure so a
    chain can fall through to the next credential.
    """

    _IMDS_URL = "http://169.254.169.254/metadata/identity/oauth2/token"
    _API_VERSION = "2018-02-01"

    def __init__(self, *, client_id: Optional[str] = None) -> None:
        self._client_id = client_id  # for user-assigned identities

    async def get_token(self, *scopes: str) -> AccessToken:
        import aiohttp

        # IMDS takes a single resource, not a .default scope.
        resource = (scopes[0] if scopes else GRAPH_DEFAULT_SCOPE).removesuffix("/.default")
        params = {"api-version": self._API_VERSION, "resource": resource}
        if self._client_id:
            params["client_id"] = self._client_id
        try:
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self._IMDS_URL, params=params, headers={"Metadata": "true"}
                ) as resp:
                    if resp.status != 200:
                        raise CredentialUnavailableError(
                            f"IMDS returned {resp.status}; managed identity unavailable"
                        )
                    payload = await resp.json()
        except (aiohttp.ClientError, OSError) as exc:  # not on Azure / no IMDS route
            raise CredentialUnavailableError(f"IMDS unreachable: {exc}") from exc

        expires_on = payload.get("expires_on")
        return AccessToken(
            token=str(payload["access_token"]),
            expires_on=float(expires_on) if expires_on else time.time() + 3600,
        )


class StaticTokenCredential:
    """Wrap an already-acquired token as a :class:`TokenCredential`.

    Backward-compatibility shim for callers that receive a token out-of-band
    (e.g. injected into a tool's execution context). Treated as non-expiring
    unless an ``expires_on`` is given.
    """

    def __init__(self, token: str, *, expires_on: Optional[float] = None) -> None:
        if not token:
            raise CredentialUnavailableError("No static token supplied")
        # Far-future default so the cache never tries to refresh a token we
        # cannot re-mint.
        self._access_token = AccessToken(token=token, expires_on=expires_on or 1e18)

    async def get_token(self, *scopes: str) -> AccessToken:
        return self._access_token


class ChainedTokenCredential:
    """Try each credential in order, returning the first token (DefaultAzureCredential-style).

    A credential that raises :class:`CredentialUnavailableError` is skipped; any
    other error propagates (it indicates a real misconfiguration, not absence).
    """

    def __init__(self, *credentials: TokenCredential) -> None:
        if not credentials:
            raise ValueError("ChainedTokenCredential requires at least one credential")
        self._credentials: List[TokenCredential] = list(credentials)

    async def get_token(self, *scopes: str) -> AccessToken:
        unavailable: List[str] = []
        for cred in self._credentials:
            try:
                return await cred.get_token(*scopes)
            except CredentialUnavailableError as exc:
                unavailable.append(f"{type(cred).__name__}: {exc}")
                logger.debug("Skipping unavailable credential %s: %s", type(cred).__name__, exc)
        raise CredentialUnavailableError(
            "No credential in the chain could provide a token: " + "; ".join(unavailable)
        )
