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

"""Deprecated coordinator-path re-export for legacy chat runtime protocols.

The canonical host for these protocol definitions is now
`victor.agent.services.protocols.chat_runtime`. This module remains only as a
warning shim for callers that still import coordinator-era protocol names.
"""

from __future__ import annotations

import warnings
from typing import Any

from victor.agent.services.protocols import chat_runtime as _chat_runtime

__all__ = list(_chat_runtime.__all__)

_DEPRECATED_MESSAGES = {
    name: (
        f"victor.agent.coordinators.chat_protocols.{name} is deprecated compatibility "
        "surface. Prefer victor.agent.services.protocols.chat_runtime or the "
        "service-first protocol package."
    )
    for name in __all__
}


def __getattr__(name: str) -> Any:
    """Resolve legacy protocol exports lazily with deprecation warnings."""
    if name in _DEPRECATED_MESSAGES:
        warnings.warn(
            _DEPRECATED_MESSAGES[name],
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_chat_runtime, name)
    raise AttributeError(
        f"module 'victor.agent.coordinators.chat_protocols' has no attribute {name!r}"
    )


def __dir__() -> list[str]:
    """Return the lazily exported legacy protocol surface."""
    return sorted(list(globals().keys()) + __all__)
