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

"""Framework-level chain registry for versioned tool chains.

This package provides a central registry for LCEL-composed tool chains
that can be shared across all verticals.

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

from .registry import ChainMetadata, ChainRegistry, get_chain_registry

__all__ = [
    "ChainMetadata",
    "ChainRegistry",
    "get_chain_registry",
]
