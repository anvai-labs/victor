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

"""Caching decorator for any :class:`TokenCredential`.

Caching is a cross-cutting concern, so it is a *decorator* (Open/Closed): it
wraps any credential without that credential — or its consumers — knowing about
it. Tokens are cached per scope-set and refreshed ahead of expiry; concurrent
refreshers are coalesced under a lock so a token stampede mints only once.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Tuple

from victor.core.identity.protocols import AccessToken, TokenCredential


class CachingTokenCredential:
    """Wrap a :class:`TokenCredential`, caching tokens with refresh-ahead.

    Args:
        inner: The credential to delegate to on a cache miss.
        refresh_before: Seconds before expiry at which a cached token is
            considered stale and re-minted (default 60s).
    """

    def __init__(self, inner: TokenCredential, *, refresh_before: float = 60.0) -> None:
        self._inner = inner
        self._refresh_before = refresh_before
        self._cache: Dict[Tuple[str, ...], AccessToken] = {}
        self._lock = asyncio.Lock()

    async def get_token(self, *scopes: str) -> AccessToken:
        key = tuple(sorted(scopes))
        cached = self._cache.get(key)
        if cached is not None and not cached.expires_within(self._refresh_before):
            return cached

        async with self._lock:
            # Re-check under the lock: another coroutine may have refreshed while
            # we awaited it (avoids a stampede of token requests).
            cached = self._cache.get(key)
            if cached is not None and not cached.expires_within(self._refresh_before):
                return cached
            token = await self._inner.get_token(*scopes)
            self._cache[key] = token
            return token

    def clear(self) -> None:
        """Drop all cached tokens (e.g. after a credential rotation)."""
        self._cache.clear()
