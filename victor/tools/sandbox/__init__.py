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

"""OS-level sandboxing for subprocess-spawning tools.

See :mod:`victor.tools.sandbox.backends`. Activated via ``settings.sandbox``
(``sandbox_enabled``, off by default); resolution is fail-open.
"""

from victor.tools.sandbox.backends import (
    BubblewrapSandbox,
    NoneSandbox,
    SandboxBackend,
    SeatbeltSandbox,
    resolve_sandbox_backend,
)

__all__ = [
    "SandboxBackend",
    "NoneSandbox",
    "BubblewrapSandbox",
    "SeatbeltSandbox",
    "resolve_sandbox_backend",
]
