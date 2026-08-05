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

"""Debug Adapter Protocol (DAP) integration for Victor.

This module provides a unified debugging interface following the
Debug Adapter Protocol (DAP) specification, enabling Victor to:
- Set breakpoints and step through code
- Inspect variables and stack frames
- Evaluate expressions in debug context
- Support multiple language debuggers via adapters

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Debug Manager                             │
    │  (Orchestrates debug sessions, manages adapter lifecycle)   │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                Debug Adapter Registry                        │
    │  (Discovers and instantiates language-specific adapters)    │
    └─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │  Python   │       │ Node.js   │       │   Rust    │
    │  Adapter  │       │  Adapter  │       │  Adapter  │
    │ (debugpy) │       │  (vscode) │       │  (lldb)   │
    └───────────┘       └───────────┘       └───────────┘
"""

from victor.observability.debug.protocol import (
    DebugSession,
    DebugState,
    Breakpoint,
    StackFrame,
    Variable,
    Scope,
    Thread,
    StopReason,
)
from victor.observability.debug.adapter import (
    DebugAdapter,
    DebugAdapterCapabilities,
)
from victor.observability.debug.manager import DebugManager
from victor.observability.debug.registry import DebugAdapterRegistry

__all__ = [
    # Protocol types
    "DebugSession",
    "DebugState",
    "Breakpoint",
    "StackFrame",
    "Variable",
    "Scope",
    "Thread",
    "StopReason",
    # Adapter interface
    "DebugAdapter",
    "DebugAdapterCapabilities",
    # Manager and registry
    "DebugManager",
    "DebugAdapterRegistry",
]
