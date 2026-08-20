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

"""Debug adapter implementations for various languages.

Available adapters:
- PythonDebugAdapter: Uses debugpy for Python debugging
- NodeDebugAdapter: Uses VS Code debug adapter for JavaScript/TypeScript
- LLDBDebugAdapter: Uses LLDB for Rust/C/C++
- DelveDebugAdapter: Uses Delve for Go
"""

# Lazy imports to avoid loading unused adapters
__all__ = [
    "PythonDebugAdapter",
]


def __getattr__(name: str):
    """Lazy import adapters."""
    if name == "PythonDebugAdapter":
        from victor.observability.debug.adapters.python import PythonDebugAdapter

        return PythonDebugAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
