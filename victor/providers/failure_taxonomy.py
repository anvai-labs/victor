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

"""Shared provider-error classification vocabulary (co-design review item 18).

Single source of truth for the exception-name / message-token tables used to
classify provider failures. Before this module existed, the hard-rate-limit
token list was duplicated verbatim between ``BaseProvider._looks_like_hard_rate_limit``
(base.py) and ``ProviderRetryStrategy._is_hard_rate_limit`` (resilience.py) —
a change to one copy without the other would silently desync classification
behavior between the two call paths. Consumers should import this module and
reference its attributes (``failure_taxonomy.HARD_RATE_LIMIT_TOKENS``, not
``from ... import HARD_RATE_LIMIT_TOKENS``) so a single edit here changes
behavior everywhere at once.
"""

from __future__ import annotations

# Exception class names (matched against type(exc).__name__ and its MRO)
# that indicate a transient transport/connection failure.
CONNECTION_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {
        "APIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TransportError",
        "WriteError",
    }
)

# Message substrings (case-insensitive) that indicate a connection failure.
CONNECTION_TOKENS: tuple[str, ...] = (
    "bad record mac",
    "broken pipe",
    "connection aborted",
    "connection error",
    "connection refused",
    "connection reset",
    "remote protocol error",
    "server disconnected",
    "ssl",
    "tls",
)

# Exception class names (matched against type(exc).__name__ and its MRO)
# that indicate a timeout failure.
TIMEOUT_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {
        "APITimeoutError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "TimeoutError",
        "WriteTimeout",
    }
)

# Message substrings (case-insensitive) that indicate a timeout failure.
TIMEOUT_TOKENS: tuple[str, ...] = ("timeout", "timed out")

# Extended exception-name patterns for resilience.py's general retryable
# check. A superset of CONNECTION_EXCEPTION_NAMES (adds "ProtocolError"):
# kept as its own table rather than reused verbatim because it answers a
# different question ("should this be retried at all?") than
# CONNECTION_EXCEPTION_NAMES ("does this look like a connection failure?").
RETRYABLE_EXCEPTION_NAMES: tuple[str, ...] = (
    "APIConnectionError",
    "RemoteProtocolError",
    "ProtocolError",
    "TransportError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "ConnectError",
    "WriteError",
)

# Message substrings (case-insensitive) indicating a 429 is a hard quota/
# billing exhaustion rather than a transient rate limit worth retrying.
HARD_RATE_LIMIT_TOKENS: tuple[str, ...] = (
    "billing",
    "credit balance",
    "credits exhausted",
    "current quota",
    "hard limit",
    "insufficient balance",
    "insufficient credits",
    "insufficient quota",
    "payment required",
    "quota exceeded",
    "quota exhausted",
    "resource exhausted",
)


def looks_like_connection_error(text: str) -> bool:
    """Case-insensitive substring check against CONNECTION_TOKENS."""
    lowered = text.lower()
    return any(token in lowered for token in CONNECTION_TOKENS)


def looks_like_timeout_error(text: str) -> bool:
    """Case-insensitive substring check against TIMEOUT_TOKENS."""
    lowered = text.lower()
    return any(token in lowered for token in TIMEOUT_TOKENS)
