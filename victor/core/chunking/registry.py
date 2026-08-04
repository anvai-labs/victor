# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Chunking strategy registry with auto-detection.

Provides a unified interface for document chunking with automatic
strategy selection based on document type.
"""

import logging
from typing import Dict, List, Optional, Type

from victor.core.chunking.base import Chunk, ChunkingConfig, ChunkingStrategy
from victor.core.chunking.detector import detect_document_type

logger = logging.getLogger(__name__)


class ChunkingRegistry:
    """Registry for chunking strategies with auto-detection.

    Maintains a registry of chunking strategies and selects the
    appropriate one based on document type.

    Example:
        registry = ChunkingRegistry()
        chunks = registry.chunk(content, source="doc.html")

        # Or with explicit type
        chunks = registry.chunk(content, doc_type="json")
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize registry with default strategies.

        Args:
            config: Chunking configuration (shared by all strategies)
        """
        self.config = config or ChunkingConfig()
        self._strategies: Dict[str, ChunkingStrategy] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default chunking strategies and discover external ones."""
        from victor.core.chunking.strategies.code import CodeChunkingStrategy
        from victor.core.chunking.strategies.html import HTMLChunkingStrategy
        from victor.core.chunking.strategies.json import JSONChunkingStrategy
        from victor.core.chunking.strategies.markdown import MarkdownChunkingStrategy
        from victor.core.chunking.strategies.text import TextChunkingStrategy

        # Register builtin strategies
        self.register(TextChunkingStrategy(self.config))
        self.register(HTMLChunkingStrategy(self.config))
        self.register(JSONChunkingStrategy(self.config))
        self.register(MarkdownChunkingStrategy(self.config))
        self.register(CodeChunkingStrategy(self.config))

        # Discover external strategies via plugin system
        try:
            from victor.core.plugins.registry import PluginRegistry

            plugin_registry = PluginRegistry.get_instance()
            for plugin in plugin_registry.list_plugins():
                # Plugins can register strategies during their register() call
                # which would call chunking_registry.register(strategy)
                pass
        except ImportError:
            pass

        # Also support direct entry points for strategies
        self._discover_external_strategies()

    def _discover_external_strategies(self) -> None:
        """Discover strategies via 'victor.chunking_strategies' entry point."""
        import logging

        try:
            from victor.framework.entry_point_registry import get_entry_point_registry

            registry = get_entry_point_registry()
            group_obj = registry.get_group("victor.chunking_strategies")

            if group_obj:
                for ep_name, (ep, loaded) in group_obj.entry_points.items():
                    try:
                        # Load entry point if not already loaded
                        strategy_cls = loaded if loaded is not False else ep.load()
                        if isinstance(strategy_cls, type):
                            strategy = strategy_cls(self.config)
                        else:
                            strategy = strategy_cls

                        if isinstance(strategy, ChunkingStrategy):
                            self.register(strategy)
                            logger.debug(f"Discovered external strategy: {strategy.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load chunking strategy '{ep_name}': {e}")
        except Exception:
            pass

    def register(self, strategy: ChunkingStrategy) -> None:
        """Register a chunking strategy.

        Args:
            strategy: Strategy instance to register
        """
        for doc_type in strategy.supported_types:
            self._strategies[doc_type.lower()] = strategy
            logger.debug(f"Registered {strategy.name} strategy for type: {doc_type}")

    def get_strategy(self, doc_type: str) -> ChunkingStrategy:
        """Get strategy for a document type.

        Args:
            doc_type: Document type (e.g., "html", "json", "markdown")

        Returns:
            Appropriate chunking strategy (defaults to text)
        """
        strategy = self._strategies.get(doc_type.lower())
        if strategy:
            return strategy

        # Default to text strategy
        return self._strategies.get("text")

    def chunk(
        self,
        content: str,
        source: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Chunk]:
        """Chunk content with auto-detected or specified strategy.

        Args:
            content: Document content to chunk
            source: Source URL or file path (for type detection)
            doc_type: Explicit document type (overrides detection)

        Returns:
            List of Chunk objects
        """
        # Detect or use provided type
        if doc_type:
            detected_type = doc_type
        else:
            detected_type = detect_document_type(source, content)

        logger.info(f"Chunking with type={detected_type}, source={source}")

        # Get and apply strategy
        strategy = self.get_strategy(detected_type)
        if strategy.name == "code":
            from victor.core.chunking.strategies.code import CodeChunkingStrategy

            if isinstance(strategy, CodeChunkingStrategy):
                return strategy.chunk(
                    content,
                    source=source,
                    language=None if detected_type == "code" else detected_type,
                )
        return strategy.chunk(content)

    @property
    def available_strategies(self) -> List[str]:
        """List of available strategy names."""
        return list(set(s.name for s in self._strategies.values()))

    @property
    def supported_types(self) -> List[str]:
        """List of all supported document types."""
        return list(self._strategies.keys())


# Global registry instance (lazy initialization)
_default_registry: Optional[ChunkingRegistry] = None


def get_chunking_registry(config: Optional[ChunkingConfig] = None) -> ChunkingRegistry:
    """Get the default chunking registry.

    Args:
        config: Optional config (only used on first call)

    Returns:
        ChunkingRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ChunkingRegistry(config)
    return _default_registry


def chunk_document(
    content: str,
    source: Optional[str] = None,
    doc_type: Optional[str] = None,
    config: Optional[ChunkingConfig] = None,
) -> List[Chunk]:
    """Convenience function to chunk a document.

    Args:
        content: Document content to chunk
        source: Source URL or file path (for type detection)
        doc_type: Explicit document type (overrides detection)
        config: Chunking configuration

    Returns:
        List of Chunk objects

    Example:
        from victor.core.chunking import chunk_document

        # Auto-detect from source
        chunks = chunk_document(html_content, source="https://example.com/page.html")

        # Explicit type
        chunks = chunk_document(json_content, doc_type="json")
    """
    if config:
        registry = ChunkingRegistry(config)
    else:
        registry = get_chunking_registry()

    return registry.chunk(content, source=source, doc_type=doc_type)
