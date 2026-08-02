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

"""Service adapters for protocol compatibility.

This package provides adapters that bridge existing components to service
protocols. Unlike deprecated coordinator shims, these adapters wrap
canonical implementations without fallback paths.

Deprecated adapter shims (ToolServiceAdapter, SessionServiceAdapter) were
removed in the service-first architecture migration. Use the canonical
services directly:
- ToolService for tool operations
- SessionService for session lifecycle
- ContextServiceAdapter for conversation context
"""

from __future__ import annotations

from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

__all__ = [
    "ContextServiceAdapter",
]
