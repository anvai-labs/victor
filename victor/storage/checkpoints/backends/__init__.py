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

"""Checkpoint storage backends.

SQLiteCheckpointBackend requires aiosqlite (optional dependency).
MemoryCheckpointBackend works without any additional dependencies.
"""

from typing import TYPE_CHECKING

# MemoryCheckpointBackend has no external dependencies - import eagerly
from victor.storage.checkpoints.backends.memory_backend import MemoryCheckpointBackend

# SQLiteCheckpointBackend requires aiosqlite - lazy import
if TYPE_CHECKING:
    from victor.storage.checkpoints.backends.sqlite_backend import (
        SQLiteCheckpointBackend,
    )

__all__ = ["SQLiteCheckpointBackend", "MemoryCheckpointBackend"]


def __getattr__(name: str):
    """Lazy import for backends with optional dependencies."""
    if name == "SQLiteCheckpointBackend":
        from victor.storage.checkpoints.backends.sqlite_backend import (
            SQLiteCheckpointBackend,
        )

        return SQLiteCheckpointBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
