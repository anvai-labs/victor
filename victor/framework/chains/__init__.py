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

"""Framework-level chain registry for versioned tool chains (deprecated).

Deprecated (backlog F-019); removable in a future victor-ai release, no earlier
than the one shipping victor-contracts 0.14.0. Nothing in the runtime reads this registry: verticals
deliver chains through ``get_chains()`` / ``get_composed_chains()`` and the
host registers them in :mod:`victor.framework.chain_registry`.

Example:
    from victor.framework.chains import ChainRegistry

    # Register a chain
    ChainRegistry.register_chain(
        name="safe_edit_chain",
        version="1.0.0",
        chain=safe_edit_chain,
        category="editing",
        description="Safe edit with verification"
    )

    # Retrieve a chain
    chain = ChainRegistry.get_chain("safe_edit_chain")
"""

import warnings

from .registry import ChainMetadata, ChainRegistry, get_chain_registry

warnings.warn(
    "victor.framework.chains (the versioned chain registry) is deprecated and will be "
    "removed in a future victor-ai release (no earlier than the one shipping "
    "victor-contracts 0.14.0): nothing reads it at runtime. Use "
    "victor.framework.chain_registry, which the host fills from each vertical's "
    "get_chains() / get_composed_chains() hook.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ChainMetadata",
    "ChainRegistry",
    "get_chain_registry",
]
