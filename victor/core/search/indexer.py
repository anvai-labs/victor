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

"""Base codebase index implementation for core.

Provides the generic file-walking and indexing logic that can be
used for any directory-based agentic task (Coding, Research, RAG).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol

logger = logging.getLogger(__name__)

# Global cache for codebase indexes to avoid re-initializing/re-building
_INDEX_CACHE: Dict[str, Any] = {}


class BaseCodebaseIndex:
    """Generic index for codebase-like directory structures.

    This class handles:
    - Recursive file walking
    - Filtering via ignore patterns
    - Change detection (staleness)
    - Coordination with chunkers and vector stores
    """

    def __init__(
        self,
        root_path: Path,
        vector_store: Optional[Any] = None,
        chunking_strategy: Optional[str] = "text",
        **kwargs: Any,
    ):
        self.root_path = root_path
        self.vector_store = vector_store
        self.chunking_strategy = chunking_strategy
        self.metadata: Dict[str, Any] = kwargs

    async def index(self, force: bool = False) -> Dict[str, Any]:
        """Perform full indexing of the root path.

        Subclasses or specific implementations should override this
        to add domain-specific parsing (e.g., AST extraction in coding).
        """
        logger.info(f"Indexing {self.root_path} with strategy={self.chunking_strategy}")
        # Base implementation just walks and chunks as text
        files_processed = 0
        for path in self.root_path.rglob("*"):
            if path.is_file() and not self._should_ignore(path):
                # chunk and add to store
                files_processed += 1

        return {"files_processed": files_processed}

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored.

        Uses the capability registry to find the IgnorePatternsProtocol.
        """
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import IgnorePatternsProtocol

        registry = CapabilityRegistry.get_instance()
        ignorer = registry.get(IgnorePatternsProtocol)
        if ignorer:
            return ignorer.should_ignore_path(str(path))

        # Fallback to simple dot-file ignore
        return any(part.startswith(".") for part in path.parts)


class CodebaseIndexFactory(CodebaseIndexFactoryProtocol):
    """Default factory for codebase indexers."""

    def create(self, root_path: str, **kwargs: Any) -> BaseCodebaseIndex:
        """Create a new BaseCodebaseIndex instance."""
        return BaseCodebaseIndex(Path(root_path), **kwargs)


class EnhancedCodebaseIndexFactory(CodebaseIndexFactoryProtocol):
    """Factory that delegates to an external codebase indexing provider.

    External vertical packages (e.g., victor-coding) register their
    CodebaseIndex factory as a capability via CapabilityRegistry at
    bootstrap time. This factory discovers and delegates to that provider.

    NOTE: this is a CALLER-SIDE convenience — never register an
    EnhancedCodebaseIndexFactory as the provider itself. It delegates to
    ``registry.get(CodebaseIndexFactoryProtocol)`` and skips itself, so
    registering it would self-reference and raise ImportError. The vertical
    that owns a CodebaseIndex must register its real factory.
    """

    def create(self, root_path: str, **kwargs: Any) -> Any:
        """Create a CodebaseIndex from an installed codebase indexing provider.

        Discovery order:
        1. CapabilityRegistry (if a different factory was registered)
        2. Direct import of victor_coding.codebase.indexer.CodebaseIndex
        3. ImportError with install instructions
        """
        # 1. Check registry for a different (non-self) factory
        from victor.core.capability_registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        factory = registry.get(CodebaseIndexFactoryProtocol)
        if factory is not None and factory is not self:
            return factory.create(root_path, **kwargs)

        # No fallback — external verticals must register via CapabilityRegistry.
        # Direct imports of victor_coding violate the core→external boundary.
        raise ImportError(
            "CodebaseIndex requires a codebase indexing provider "
            "(e.g., pip install victor-coding). The provider registers "
            "its factory via CapabilityRegistry at bootstrap."
        )


def detect_enhanced_index_factory() -> Optional[CodebaseIndexFactoryProtocol]:
    """Detect whether a codebase indexing provider is registered.

    Returns the registered factory directly, or None. This is a PURE DETECTOR —
    it must NEVER return a delegating shell for re-registration as a provider
    (doing so creates a self-reference loop: the shell's create() would find
    only itself in the registry and raise ImportError). A vertical that owns a
    CodebaseIndex should register its real factory directly.
    """
    from victor.core.capability_registry import CapabilityRegistry

    registry = CapabilityRegistry.get_instance()
    return registry.get(CodebaseIndexFactoryProtocol)
