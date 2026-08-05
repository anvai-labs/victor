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

"""Provider-agnostic identity / token-credential layer.

Consumers depend on the :class:`TokenCredential` protocol and receive a concrete
credential via dependency injection (built by the factories here). The shape
mirrors ``azure-identity`` so it stays compatible with the cloud-native standard.

Example::

    from victor.core.identity import graph_credential_from_env, GRAPH_DEFAULT_SCOPE

    credential = graph_credential_from_env()          # None if unconfigured
    token = (await credential.get_token(GRAPH_DEFAULT_SCOPE)).token
"""

from __future__ import annotations

from victor.core.identity.cache import CachingTokenCredential
from victor.core.identity.factory import (
    build_entra_credential,
    default_graph_credential,
    graph_credential_from_env,
)
from victor.core.identity.oidc import (
    OidcConfig,
    UserIdentity,
    build_authorize_url,
    exchange_code,
    fetch_user_identity,
    resolve_identity_from_code,
)
from victor.core.identity.protocols import (
    AccessToken,
    CredentialUnavailableError,
    TokenCredential,
)
from victor.core.identity.sources import (
    ChainedTokenCredential,
    ClientAssertionCredential,
    ClientSecretCredential,
    ENTRA_AUTHORITY,
    GRAPH_DEFAULT_SCOPE,
    ManagedIdentityCredential,
    StaticTokenCredential,
)

__all__ = [
    # Protocol + value types
    "AccessToken",
    "TokenCredential",
    "CredentialUnavailableError",
    # Strategies
    "ClientSecretCredential",
    "ClientAssertionCredential",
    "ManagedIdentityCredential",
    "StaticTokenCredential",
    "ChainedTokenCredential",
    # Caching
    "CachingTokenCredential",
    # Factories
    "build_entra_credential",
    "graph_credential_from_env",
    "default_graph_credential",
    # Delegated / SSO (user identity)
    "OidcConfig",
    "UserIdentity",
    "build_authorize_url",
    "exchange_code",
    "fetch_user_identity",
    "resolve_identity_from_code",
    # Constants
    "ENTRA_AUTHORITY",
    "GRAPH_DEFAULT_SCOPE",
]
