# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Client-facing long-term memory seam (FEP-0048, FOU-A2d).

The client layer (TUI slash commands, HTTP API routes) may not import
``victor.agent.*`` directly (boundary guard, ``test_architectural_boundaries``),
so the shared consolidator handle that the confirm surfaces drive is reached
through this framework submodule — the same lazy pattern as
``victor.framework.rl.coordinator``. The implementation lives in
``victor.agent.services.memory_consolidation``; these are thin delegates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.services.memory_consolidation import MemoryConsolidator
    from victor.storage.memory.preference_episodic import PreferenceEpisodicStore

__all__ = [
    "get_shared_memory_consolidator",
    "get_shared_memory_store",
    "reset_shared_memory",
]


def get_shared_memory_store() -> "PreferenceEpisodicStore":
    """The process-wide preference/episodic store (lazy; injectable)."""
    from victor.agent.services.memory_consolidation import (
        get_shared_memory_store as _get,
    )

    return _get()


def get_shared_memory_consolidator() -> "MemoryConsolidator":
    """The process-wide consolidator the confirm surfaces drive."""
    from victor.agent.services.memory_consolidation import (
        get_shared_memory_consolidator as _get,
    )

    return _get()


def reset_shared_memory() -> None:
    """Drop both shared handles (tests re-point storage between cases)."""
    from victor.agent.services.memory_consolidation import (
        reset_shared_memory as _reset,
    )

    _reset()
