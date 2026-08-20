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

"""Signed callback tokens for HITL approve/reject links.

An approval message (Slack/Teams/...) embeds a browser-clickable link such as
``/hitl/respond/{request_id}?action=approve&token=...``. Without integrity the
``request_id`` alone is a forgeable authorization decision — anyone who learns
the URL could approve. These helpers bind ``(request_id, action, expiry)`` with
an HMAC so a link cannot be forged, cannot be edited to flip approve↔reject, and
expires. Replay of an *already-decided* request is rejected separately by the
store (a request can only be responded once), giving effective single use.

Signing is keyed by ``VICTOR_HITL_SIGNING_SECRET``. When unset, signing is
disabled (links are unsigned and accepted) so local/dev setups keep working;
production deployments should always set the secret.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Optional

_SECRET_ENV = "VICTOR_HITL_SIGNING_SECRET"
_DEFAULT_TTL_SECONDS = 24 * 60 * 60  # links are valid for 24h by default


def get_signing_secret() -> Optional[str]:
    """Return the configured callback-signing secret, or ``None`` if unset."""
    secret = os.getenv(_SECRET_ENV)
    return secret or None


def _digest(request_id: str, action: str, expires_at: int, secret: str) -> str:
    message = f"{request_id}:{action}:{expires_at}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def sign_action(
    request_id: str,
    action: str,
    *,
    secret: str,
    ttl: int = _DEFAULT_TTL_SECONDS,
    now: Optional[float] = None,
) -> str:
    """Return a signed token of the form ``"{expires_at}.{hexdigest}"``.

    The token binds the request id, the action and an absolute expiry, so it is
    valid only for that exact action on that exact request, until it expires.
    """
    expires_at = int((now if now is not None else time.time()) + ttl)
    return f"{expires_at}.{_digest(request_id, action, expires_at, secret)}"


def verify_action(
    request_id: str,
    action: str,
    token: Optional[str],
    *,
    secret: str,
    now: Optional[float] = None,
) -> bool:
    """Constant-time verify a token for ``(request_id, action)`` and check expiry."""
    if not token:
        return False
    try:
        expires_str, signature = token.split(".", 1)
        expires_at = int(expires_str)
    except (ValueError, AttributeError):
        return False
    if (now if now is not None else time.time()) > expires_at:
        return False
    expected = _digest(request_id, action, expires_at, secret)
    return hmac.compare_digest(signature, expected)
