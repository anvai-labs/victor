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

"""Core types for cross-vertical use.

This module re-exports types from `victor.core.vertical_types` for backward
compatibility. New code should import directly from `vertical_types`.

Note:
    For backward compatibility, these types are also re-exported from
    `victor.core.verticals.protocols`.
"""

from __future__ import annotations

# Re-export from vertical_types for backward compatibility
from victor.core.vertical_types import TaskTypeHint

__all__ = [
    "TaskTypeHint",
]
