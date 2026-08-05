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

"""Context-local request correlation helpers for observability.

This keeps a per-request correlation ID in a ContextVar so concurrent API
requests can safely emit tool events onto the shared observability bus
without stomping on each other's identifiers.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_REQUEST_CORRELATION_ID: ContextVar[Optional[str]] = ContextVar(
    "victor_request_correlation_id",
    default=None,
)


def get_request_correlation_id() -> Optional[str]:
    """Return the current request correlation ID, if one is active."""
    return _REQUEST_CORRELATION_ID.get()


@contextmanager
def request_correlation_id(correlation_id: str) -> Iterator[str]:
    """Bind a correlation ID to the current async context."""
    token = _REQUEST_CORRELATION_ID.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _REQUEST_CORRELATION_ID.reset(token)
