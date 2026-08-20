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

"""Provider-agnostic token-credential abstraction.

This is the *only* type that token consumers (transports, tools, providers)
should depend on. Concrete credential flows (client-credentials, managed
identity, a pre-supplied token, ...) live in :mod:`victor.core.identity.sources`
and are interchangeable behind :class:`TokenCredential` (Liskov). The shape
deliberately mirrors ``azure.core.credentials.AccessToken`` /
``azure.core.credentials_async.AsyncTokenCredential`` so this layer stays
compatible with the cloud-native standard and could delegate to
``azure-identity`` later without changing any caller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AccessToken:
    """A bearer token and its absolute expiry.

    Attributes:
        token: The bearer access token string.
        expires_on: Absolute expiry as epoch seconds (UTC), matching the
            ``azure.core.credentials.AccessToken.expires_on`` convention.
    """

    token: str
    expires_on: float

    @property
    def expired(self) -> bool:
        """True once the token is at/after its expiry."""
        return time.time() >= self.expires_on

    def expires_within(self, seconds: float) -> bool:
        """True when the token expires within ``seconds`` (refresh-ahead check)."""
        return time.time() >= (self.expires_on - seconds)


@runtime_checkable
class TokenCredential(Protocol):
    """A source of bearer tokens, scoped to one or more resources.

    Implementations exchange *some* credential (a secret, a certificate
    assertion, a managed identity, a pre-supplied token, ...) for an
    :class:`AccessToken`. Consumers depend on this protocol only — never on a
    concrete flow — so new auth methods are added without touching them
    (Open/Closed), and tests inject a fake instead of mocking HTTP.
    """

    async def get_token(self, *scopes: str) -> AccessToken:
        """Return a (possibly cached) token valid for ``scopes``.

        Args:
            *scopes: Resource scopes, e.g. ``"https://graph.microsoft.com/.default"``.

        Returns:
            A valid :class:`AccessToken`.

        Raises:
            CredentialUnavailableError: This credential cannot produce a token
                (missing config, not running in the required environment, ...),
                signalling a :class:`ChainedTokenCredential` to try the next one.
        """
        ...


class CredentialUnavailableError(Exception):
    """Raised when a credential cannot produce a token in the current context.

    A chain treats this as "skip me, try the next credential" rather than a hard
    failure, mirroring ``azure.identity.CredentialUnavailableError``.
    """
