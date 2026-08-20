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

"""Rust AST indexer wrapper.

Provides a protocol-compliant wrapper around the Rust AST indexer functions.
The wrapper delegates to victor_native functions while maintaining the
AstIndexerProtocol interface and observability hooks.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import victor_native

from victor.native.observability import InstrumentedAccelerator


class RustAstIndexer(InstrumentedAccelerator):
    """Rust implementation of AstIndexerProtocol.

    Wraps the high-performance Rust functions for stdlib detection
    and identifier extraction with protocol-compliant interface.

    Performance characteristics:
    - is_stdlib_module: 5-10x faster than Python (static HashSet)
    - extract_identifiers: 3-5x faster (compiled regex)
    """

    def __init__(self) -> None:
        super().__init__(backend="rust")
        self._version = victor_native.__version__

    def get_version(self) -> Optional[str]:
        return self._version

    def is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module name is a standard library module.

        Delegates to Rust implementation with O(1) HashSet lookup.

        Args:
            module_name: Full module name (e.g., "os.path", "typing")

        Returns:
            True if the module is in the stdlib or common third-party set
        """
        with self._timed_call("stdlib_check"):
            return victor_native.is_stdlib_module(module_name)

    def batch_is_stdlib_modules(self, module_names: List[str]) -> List[bool]:
        """Check multiple module names for stdlib membership.

        Delegates to Rust implementation for batch processing.

        Args:
            module_names: List of module names to check

        Returns:
            List of booleans, one per module name
        """
        with self._timed_call("batch_stdlib_check"):
            return victor_native.batch_is_stdlib_modules(module_names)

    def extract_identifiers(self, source: str) -> List[str]:
        """Extract all unique identifier references from source code.

        Delegates to Rust implementation with compiled regex.

        Args:
            source: Source code text

        Returns:
            List of unique identifiers found
        """
        with self._timed_call("extract_identifiers"):
            return victor_native.extract_identifiers(source)

    def extract_identifiers_with_positions(self, source: str) -> List[Tuple[str, int, int]]:
        """Extract identifiers with their positions.

        Delegates to Rust implementation.

        Args:
            source: Source code text

        Returns:
            List of (identifier, start_offset, end_offset) tuples
        """
        with self._timed_call("extract_identifiers_with_positions"):
            return victor_native.extract_identifiers_with_positions(source)

    def filter_stdlib_imports(self, imports: List[str]) -> Tuple[List[str], List[str]]:
        """Partition imports into stdlib and non-stdlib.

        Delegates to Rust implementation.

        Args:
            imports: List of import module names

        Returns:
            Tuple of (stdlib_imports, non_stdlib_imports)
        """
        with self._timed_call("filter_stdlib_imports"):
            return victor_native.filter_stdlib_imports(imports)
