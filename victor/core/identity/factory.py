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

"""Composition root for :class:`TokenCredential`s.

Callers depend on the :class:`TokenCredential` protocol; these factories are the
*one* place that knows how to assemble a concrete credential from config/env and
wrap it with caching. Keeping assembly here (and out of the consumers) preserves
Dependency Inversion — a transport never names a concrete flow.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Union

from victor.core.identity.cache import CachingTokenCredential
from victor.core.identity.protocols import TokenCredential
from victor.core.identity.sources import (
    ChainedTokenCredential,
    ClientAssertionCredential,
    ClientSecretCredential,
    ENTRA_AUTHORITY,
    ManagedIdentityCredential,
)

# Canonical env vars (preferred) and their accepted aliases, in priority order.
# AZURE_* is canonical (matches AzureCredentials + env_filtering); TEAMS_* is a
# backward-compatible alias for the previously-documented Teams variables.
_TENANT_ENV = ("AZURE_TENANT_ID", "TEAMS_TENANT_ID")
_CLIENT_ENV = ("AZURE_CLIENT_ID", "TEAMS_CLIENT_ID")
_SECRET_ENV = ("AZURE_CLIENT_SECRET", "TEAMS_CLIENT_SECRET")


def _first_env(names: tuple) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def build_entra_credential(
    *,
    tenant_id: str,
    client_id: str,
    client_secret: Optional[str] = None,
    client_assertion: Optional[Union[str, Callable[[], str]]] = None,
    authority: str = ENTRA_AUTHORITY,
    cache: bool = True,
) -> TokenCredential:
    """Build an Entra app-only credential from explicit values.

    Supply exactly one of ``client_secret`` (secret-based) or
    ``client_assertion`` (certificate-based). The result is wrapped with
    :class:`CachingTokenCredential` unless ``cache=False``.
    """
    if bool(client_secret) == bool(client_assertion):
        raise ValueError("Provide exactly one of client_secret or client_assertion (certificate)")
    base: TokenCredential
    if client_assertion is not None:
        base = ClientAssertionCredential(
            tenant_id, client_id, client_assertion, authority=authority
        )
    else:
        assert client_secret is not None  # narrowed by the xor check above
        base = ClientSecretCredential(tenant_id, client_id, client_secret, authority=authority)
    return CachingTokenCredential(base) if cache else base


def graph_credential_from_env(*, cache: bool = True) -> Optional[TokenCredential]:
    """Build an Entra app-only credential from environment variables.

    Reads ``AZURE_TENANT_ID``/``AZURE_CLIENT_ID``/``AZURE_CLIENT_SECRET`` (with
    ``TEAMS_*`` accepted as aliases). Returns ``None`` when the trio is
    incomplete so callers can fall back (e.g. to a webhook) rather than crash.
    """
    tenant_id = _first_env(_TENANT_ENV)
    client_id = _first_env(_CLIENT_ENV)
    client_secret = _first_env(_SECRET_ENV)
    if not (tenant_id and client_id and client_secret):
        return None
    return build_entra_credential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        cache=cache,
    )


def default_graph_credential(*, include_managed_identity: bool = True) -> Optional[TokenCredential]:
    """A ``DefaultAzureCredential``-style chain for Microsoft Graph.

    Order: env client-credentials, then (optionally) managed identity. Returns
    ``None`` if nothing is configured. The whole chain is cached once.
    """
    chain = []
    env_cred = graph_credential_from_env(cache=False)
    if env_cred is not None:
        chain.append(env_cred)
    if include_managed_identity:
        chain.append(ManagedIdentityCredential())
    if not chain:
        return None
    return CachingTokenCredential(ChainedTokenCredential(*chain))
