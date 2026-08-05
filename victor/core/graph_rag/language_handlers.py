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

"""Language-specific edge detection handlers for Graph RAG indexing.

This module establishes the SEAM between core graph indexing and
language-specific edge detection logic.

ARCHITECTURAL PRINCIPLE:
- Core indexing pipeline (indexing.py) should be language-agnostic
- Each language implements edge detection via LanguageEdgeHandler protocol
- Handlers are discovered via registry pattern (similar to victor_coding)

MIGRATION PATH (2026-04-29):
- Current: CALLS edge detection is hardcoded in indexing.py
- Target: Each language in victor-coding provides its own edge handler
- This module provides the protocol and registry for that migration

Future sessions should:
1. Implement handlers for additional languages (JS, TS, Go, Rust, etc.)
2. Move handler implementations to victor-coding package
3. Remove hardcoded edge detection from indexing.py once all languages migrate
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


# Non-CALLS edge types the analysis provider currently emits via
# extract_edges. Kept in sync with TreeSitterAnalysisProvider._edges_from_parsed
# (CALLS + INHERITS + IMPLEMENTS + COMPOSITION). When victor-coding adds a new
# structural edge type, extend this set so it stops being silently dropped.
_RELATIONSHIP_EDGE_TYPES = frozenset({"INHERITS", "IMPLEMENTS", "COMPOSITION"})

# Language aliases the TreeSitterAnalysisProtocol provider does not resolve on
# its own. Applied only as a fallback when the provider rejects the literal
# language, keeping edge-handler resolution alias-tolerant (this coverage was
# previously supplied by the now-removed victor_coding registry fallback).
_LANGUAGE_ALIASES = {
    "ts": "typescript",
    "tsx": "typescript",
    "jsx": "javascript",
    "golang": "go",
    "py": "python",
    "js": "javascript",
    "rs": "rust",
    "kt": "kotlin",
    "cs": "csharp",
    "rb": "ruby",
    "c++": "cpp",
}


@dataclass
class CallEdge:
    """A detected function call edge.

    Attributes:
        caller_name: Name of the calling function/symbol
        callee_name: Name of the called function/symbol
        caller_line: Line number where call occurs (for debugging)
        receiver_type: Inferred static type of the receiver for method calls
            (e.g. ``obj.method()`` → ``Foo``). ``None`` when the language
            plugin or analysis provider could not determine it.
        is_method_call: True when the call is dot-dispatch
            (``obj.method()``) regardless of whether ``receiver_type`` is
            known. Indexing uses this to drop name-only fallback for
            method calls that would otherwise bind to unrelated methods
            with the same leaf name.
    """

    caller_name: str
    callee_name: str
    caller_line: Optional[int] = None
    receiver_type: Optional[str] = None
    is_method_call: bool = False


@dataclass
class RelationshipEdge:
    """A detected non-CALLS structural relationship between two named symbols.

    Covers INHERITS/IMPLEMENTS/COMPOSITION edges produced by the analysis
    provider's ``extract_edges``. Source and target are unresolved leaf names
    (e.g. ``"Child"``, ``"Parent"``) — the indexing pipeline buffers these and
    resolves them to node_ids against a project-wide class/interface index
    after all nodes have been persisted.
    """

    source_name: str
    target_name: str
    edge_type: str  # INHERITS, IMPLEMENTS, COMPOSITION
    line_number: Optional[int] = None


@dataclass
class EdgeDetectionResult:
    """Result from edge detection for a file.

    Attributes:
        calls: List of CALLS edges detected
        relationships: List of non-CALLS structural edges
            (INHERITS/IMPLEMENTS/COMPOSITION) detected by the underlying
            provider. Optional so legacy handlers that only know about CALLS
            keep working.
        metadata: Additional metadata (language-specific)
    """

    calls: List[CallEdge]
    relationships: List[RelationshipEdge] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.relationships is None:
            self.relationships = []


class LanguageEdgeHandler(Protocol):
    """Protocol for language-specific edge detection.

    Each language implements this protocol to provide:
    - CALLS edge detection (function/method calls)
    - REFERENCES edge detection (variable/type references)
    - Other language-specific relationships

    Implementation pattern:
        1. Create handler class in handlers/ subdirectory
        2. Register via register_edge_handler()
        3. Core indexing pipeline discovers and uses handler by language name
    """

    def get_supported_languages(self) -> List[str]:
        """Get list of language identifiers this handler supports.

        Returns:
            List of language names (e.g., ["python", "py"])
        """
        ...

    async def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in parsed source code.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        ...

    def can_handle(self, language: str) -> bool:
        """Check if this handler can process the given language.

        Args:
            language: Language identifier

        Returns:
            True if handler supports this language
        """
        return language.lower() in [lang.lower() for lang in self.get_supported_languages()]


class EdgeHandlerRegistry:
    """Registry for language edge handlers.

    Provides:
    - Handler registration by language name
    - Handler discovery via language identifier
    - Fallback to generic handler when no specific handler exists

    Pattern follows victor_coding.languages.registry for consistency.
    """

    _handlers: Dict[str, LanguageEdgeHandler] = {}
    _fallback_handler: Optional[LanguageEdgeHandler] = None

    @classmethod
    def register(
        cls,
        handler: LanguageEdgeHandler,
        languages: List[str],
    ) -> None:
        """Register an edge handler for one or more languages.

        Args:
            handler: Edge handler instance
            languages: List of language identifiers this handler supports

        Example:
            registry.register(PythonEdgeHandler(), ["python", "py"])
        """
        for lang in languages:
            lang_lower = lang.lower()
            cls._handlers[lang_lower] = handler
            logger.debug(f"Registered edge handler for language: {lang_lower}")

    @classmethod
    def get_handler(cls, language: str) -> Optional[LanguageEdgeHandler]:
        """Get edge handler for a specific language.

        Args:
            language: Language identifier

        Returns:
            Handler instance or None if not registered
        """
        return cls._handlers.get(language.lower())

    @classmethod
    def get_or_fallback(cls, language: str) -> Optional[LanguageEdgeHandler]:
        """Get handler or fallback to generic handler.

        Args:
            language: Language identifier

        Returns:
            Specific handler, fallback handler, or None
        """
        handler = cls.get_handler(language)
        if handler is not None:
            return handler
        return cls._fallback_handler

    @classmethod
    def set_fallback(cls, handler: LanguageEdgeHandler) -> None:
        """Set the fallback handler for unsupported languages.

        Args:
            handler: Generic edge handler
        """
        cls._fallback_handler = handler
        logger.debug("Registered fallback edge handler")

    @classmethod
    def supported_languages(cls) -> List[str]:
        """Get list of all supported languages.

        Returns:
            List of language identifiers
        """
        return list(set(cls._handlers.keys()))


def register_edge_handler(
    handler: LanguageEdgeHandler,
    languages: List[str],
) -> None:
    """Register an edge handler.

    Convenience function for handler registration.

    Args:
        handler: Edge handler instance
        languages: List of language identifiers

    Example:
        register_edge_handler(PythonEdgeHandler(), ["python", "py"])
    """
    EdgeHandlerRegistry.register(handler, languages)


def get_edge_handler(language: str) -> Optional[LanguageEdgeHandler]:
    """Get edge handler for a language.

    Resolves through the enhanced ``TreeSitterAnalysisProtocol`` provider via
    the capability registry — the canonical, vertical-agnostic path. Accessing
    the registry lazily runs capability bootstrap (``ensure_bootstrapped``),
    which triggers plugin discovery, so any installed language provider (e.g.
    ``victor-coding``) registers itself without core importing it by name.

    Args:
        language: Language identifier

    Returns:
        Handler instance or None if no provider handles this language.
    """
    return _get_analysis_provider_handler(language)


def _get_analysis_provider_handler(language: str) -> Optional[LanguageEdgeHandler]:
    """Return an analysis-provider-backed handler when one is registered.

    Returns ``None`` when only the null stub is registered (no enhanced
    provider), or when the provider does not support ``language``. This
    is the preferred path: it does not import ``victor_coding`` directly
    and works for any language the provider's plugins declare.
    """
    try:
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import TreeSitterAnalysisProtocol

        registry = CapabilityRegistry.get_instance()
        if not registry.is_enhanced(TreeSitterAnalysisProtocol):
            return None
        provider = registry.get(TreeSitterAnalysisProtocol)
    except Exception:
        return None

    if provider is None:
        return None

    resolved = language
    if not provider.supports_language(resolved):
        alias = _LANGUAGE_ALIASES.get(language)
        if alias is None or not provider.supports_language(alias):
            return None
        resolved = alias
    return _AnalysisProviderEdgeHandler(provider, resolved)


class _AnalysisProviderEdgeHandler:
    """Adapter: ``TreeSitterAnalysisProtocol`` → ``LanguageEdgeHandler``.

    Wraps the analysis provider so that core indexing code can ask for
    CALLS edges through the existing ``handler.detect_calls_edges(tree,
    source_code, file_path)`` interface. The pre-parsed ``tree`` is
    intentionally ignored — the provider does its own parsing through
    :class:`TreeSitterService` (which shares the per-thread Parser cache),
    so the cost is one extra parse per file at most while the indexer is
    still building edges separately from the symbol pass. Once edge
    extraction joins symbol extraction in a single provider call, the
    duplicate parse goes away.
    """

    def __init__(self, provider: Any, language: str) -> None:
        self._provider = provider
        self._language = language

    def get_supported_languages(self) -> List[str]:
        return [self._language]

    def can_handle(self, language: str) -> bool:
        return language.lower() == self._language.lower()

    async def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        import asyncio

        edges = await asyncio.to_thread(
            self._provider.extract_edges,
            source_code.encode("utf-8"),
            self._language,
            file_path=str(file_path),
        )
        calls: List[CallEdge] = []
        relationships: List[RelationshipEdge] = []
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                continue
            edge_type = edge.get("edge_type")
            if edge_type == "CALLS":
                calls.append(
                    CallEdge(
                        caller_name=source,
                        callee_name=target,
                        caller_line=edge.get("line_number"),
                        receiver_type=edge.get("receiver_type"),
                        is_method_call=bool(edge.get("is_method_call", False)),
                    )
                )
            elif edge_type in _RELATIONSHIP_EDGE_TYPES:
                relationships.append(
                    RelationshipEdge(
                        source_name=source,
                        target_name=target,
                        edge_type=edge_type,
                        line_number=edge.get("line_number"),
                    )
                )
        return EdgeDetectionResult(calls=calls, relationships=relationships)


class _VictorCodingPluginAdapter:
    """Adapter to use victor_coding LanguagePlugin as LanguageEdgeHandler.

    This bridges the gap between:
    - victor_coding LanguagePlugin protocol (detect_calls_edges synchronous)
    - victor-ai LanguageEdgeHandler protocol (detect_calls_edges async)

    KNOWN GAP (2026-05): This fallback path only surfaces CALLS edges. The
    enhanced ``_AnalysisProviderEdgeHandler`` above also emits INHERITS /
    IMPLEMENTS / COMPOSITION because the TreeSitterAnalysisProvider runs the
    plugin's ``inheritance`` / ``implements`` / ``composition`` queries
    itself; LanguagePlugin doesn't expose an equivalent ``detect_*_edges``
    method, so we can't reach those queries through this adapter without
    either widening the plugin protocol or duplicating the TSA query
    machinery here. Since the TSA provider is the registered default
    everywhere except null-stub environments, this gap is documented rather
    than worked around — when only the stub is registered, the same
    edge-type coverage as before this refactor is preserved.
    """

    def __init__(self, plugin: Any):
        """Initialize adapter with a victor_coding language plugin.

        Args:
            plugin: LanguagePlugin instance from victor_coding
        """
        self._plugin = plugin
        self._supported_languages = [plugin.config.name] + plugin.config.aliases

    def get_supported_languages(self) -> List[str]:
        """Get list of language identifiers this handler supports."""
        return self._supported_languages

    async def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges using victor_coding plugin.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        # victor_coding plugins use synchronous detect_calls_edges
        # Run in thread pool to avoid blocking
        import asyncio

        return await asyncio.to_thread(
            self._plugin.detect_calls_edges,
            tree,
            source_code,
            file_path,
        )

    def can_handle(self, language: str) -> bool:
        """Check if this handler can process the given language."""
        return language.lower() in [lang.lower() for lang in self._supported_languages]


__all__ = [
    "CallEdge",
    "EdgeDetectionResult",
    "LanguageEdgeHandler",
    "EdgeHandlerRegistry",
    "register_edge_handler",
    "get_edge_handler",
    "_AnalysisProviderEdgeHandler",  # Exported for testing
    "_VictorCodingPluginAdapter",  # Exported for testing (legacy fallback)
]
