# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Compatibility bridge for structural codebase indexing.

This module upgrades ``victor-coding``'s symbol-oriented embedding API to the
main repo's structural chunking and vector-store stack without editing the
external plugin package. The bridge is intentionally narrow:

- ``code_search_tool`` can opt the codebase index into the bridge
- the bridge implements the ``victor-coding`` embedding provider contract
- the actual storage/search backend remains the normal main-repo provider

The result is a reusable, loosely coupled indexing path:
- no duplication of vector-store logic
- no hard dependency on sibling repo internals at import time
- explicit persistence manifesting so storage schema changes rebuild safely
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

from victor.core.verticals.import_resolver import vertical_module_candidates
from victor.core.utils.capability_loader import load_tree_sitter_get_parser
from victor.storage.vector_stores import EmbeddingConfig as InternalEmbeddingConfig
from victor.storage.vector_stores.base import (
    BaseEmbeddingProvider as InternalEmbeddingProvider,
)
from victor.storage.vector_stores.code_chunking import (
    CodeChunkingContext,
    TreeSitterParseContext,
    create_code_chunker,
)
from victor.storage.vector_stores.registry import (
    EmbeddingRegistry as InternalEmbeddingRegistry,
)

logger = logging.getLogger(__name__)

STRUCTURAL_CODEBASE_VECTOR_STORE = "victor_structural_bridge"
CODEBASE_INDEX_MANIFEST_NAME = "code_search_index_manifest.json"
CODEBASE_INDEX_SCHEMA_VERSION = 2
DEFAULT_CODEBASE_CHUNKING_STRATEGY = "tree_sitter_structural"
DEFAULT_CODEBASE_CHUNK_SIZE = 500
DEFAULT_CODEBASE_CHUNK_OVERLAP = 50

_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".r": "r",
}

_STRUCTURAL_PROVIDER_CLASS: Optional[type[Any]] = None


@dataclass(frozen=True)
class _BridgeSymbol:
    """Minimal symbol representation consumed by the chunking strategies."""

    name: str
    symbol_type: str
    line_start: int
    line_end: int
    parent_symbol: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        if self.parent_symbol and not self.name.startswith(f"{self.parent_symbol}."):
            return f"{self.parent_symbol}.{self.name}"
        return self.name


@dataclass
class _PendingFileBatch:
    """Accumulated symbol documents for a single file."""

    file_path: str
    documents: list[dict[str, Any]]
    replace_existing: bool = False


def _normalized_extra_config(config: Mapping[str, Any]) -> dict[str, Any]:
    extra = config.get("extra_config", {})
    if isinstance(extra, dict):
        return dict(extra)
    return {}


def _normalized_chunking_strategy(extra_config: Mapping[str, Any]) -> str:
    return (
        str(
            extra_config.get("code_chunking_strategy")
            or extra_config.get("chunking_strategy")
            or DEFAULT_CODEBASE_CHUNKING_STRATEGY
        ).strip()
        or DEFAULT_CODEBASE_CHUNKING_STRATEGY
    )


def _normalized_language_overrides(extra_config: Mapping[str, Any]) -> dict[str, str]:
    raw_overrides = extra_config.get("language_overrides", {})
    if not isinstance(raw_overrides, Mapping):
        return {}

    normalized: dict[str, str] = {}
    for suffix, language in raw_overrides.items():
        if not isinstance(suffix, str) or not isinstance(language, str):
            continue
        cleaned_suffix = suffix.strip().lower()
        cleaned_language = language.strip()
        if not cleaned_suffix or not cleaned_language:
            continue
        if not cleaned_suffix.startswith("."):
            cleaned_suffix = f".{cleaned_suffix}"
        normalized[cleaned_suffix] = cleaned_language
    return normalized


def build_codebase_index_manifest(
    embedding_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact persistence fingerprint for a codebase embedding config."""

    extra_config = _normalized_extra_config(embedding_config)
    vector_store = str(embedding_config.get("vector_store", "lancedb"))
    upstream_vector_store = str(extra_config.get("upstream_vector_store") or vector_store)
    bridge_enabled = vector_store == STRUCTURAL_CODEBASE_VECTOR_STORE or bool(
        extra_config.get("structural_indexing_enabled", False)
    )

    return {
        "schema_version": CODEBASE_INDEX_SCHEMA_VERSION,
        "index_backend": "structural_bridge" if bridge_enabled else "legacy_provider",
        "vector_store": upstream_vector_store,
        "embedding_provider": str(
            embedding_config.get("embedding_model_type", "sentence-transformers")
        ),
        "embedding_model": str(
            embedding_config.get("embedding_model_name", "BAAI/bge-small-en-v1.5")
        ),
        "distance_metric": str(embedding_config.get("distance_metric", "cosine")),
        "dimension": int(extra_config.get("dimension", 384)),
        "batch_size": int(extra_config.get("batch_size", 32)),
        "chunking_strategy": _normalized_chunking_strategy(extra_config),
        "chunk_size": int(extra_config.get("chunk_size", DEFAULT_CODEBASE_CHUNK_SIZE)),
        "chunk_overlap": int(extra_config.get("chunk_overlap", DEFAULT_CODEBASE_CHUNK_OVERLAP)),
        "language_overrides": _normalized_language_overrides(extra_config),
    }


def read_codebase_index_manifest(persist_directory: Path) -> Optional[dict[str, Any]]:
    """Read the persisted code_search manifest if it exists."""

    manifest_path = persist_directory / CODEBASE_INDEX_MANIFEST_NAME
    if not manifest_path.is_file():
        return None

    try:
        payload = json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError) as exc:
        logger.warning("Failed to read code_search manifest from %s: %s", manifest_path, exc)
        return None
    if isinstance(payload, dict):
        return payload
    return None


def write_codebase_index_manifest(persist_directory: Path, manifest: Mapping[str, Any]) -> None:
    """Persist the code_search manifest atomically."""

    persist_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = persist_directory / CODEBASE_INDEX_MANIFEST_NAME
    temp_path = manifest_path.with_suffix(".tmp")
    temp_path.write_text(json_dumps(dict(manifest), sort_keys=True, indent=2), encoding="utf-8")
    temp_path.replace(manifest_path)


def has_compatible_codebase_index_manifest(
    persist_directory: Path,
    manifest: Mapping[str, Any],
) -> bool:
    """Return True when the persisted fingerprint exactly matches the requested one.

    Logs detailed information about WHY the manifest doesn't match to help
    diagnose unnecessary rebuilds.
    """

    persisted = read_codebase_index_manifest(persist_directory)
    if persisted is None:
        logger.debug(
            "[code_search] Manifest validation: No persisted manifest found at %s",
            persist_directory,
        )
        return False

    # Fast path: exact match
    if persisted == dict(manifest):
        return True

    # Detailed mismatch logging for debugging
    logger.debug(
        "[code_search] Manifest mismatch details for %s:\n"
        "  Persisted hash: %s\n"
        "  Expected hash:  %s\n"
        "  Keys in persisted but not expected: %s\n"
        "  Keys in expected but not persisted: %s\n"
        "  Common keys with different values: %s",
        persist_directory,
        persisted.get("hash", "NO_HASH")[:12],
        dict(manifest).get("hash", "NO_HASH")[:12],
        set(persisted.keys()) - set(manifest.keys()),
        set(manifest.keys()) - set(persisted.keys()),
        _get_mismatched_keys(persisted, dict(manifest)),
    )

    return False


def _get_mismatched_keys(persisted: Dict[str, Any], expected: Dict[str, Any]) -> Set[str]:
    """Identify which keys have different values between two manifests.

    Returns a set of key names that differ, useful for debugging manifest
    validation failures.
    """
    common_keys = set(persisted.keys()) & set(expected.keys())
    mismatched = set()
    for key in common_keys:
        if persisted[key] != expected[key]:
            mismatched.add(key)
    return mismatched


def has_persisted_codebase_index_data(persist_directory: Path) -> bool:
    """Return True when the persist directory contains real index data."""

    if not persist_directory.exists():
        return False
    for child in persist_directory.iterdir():
        if child.name == CODEBASE_INDEX_MANIFEST_NAME:
            continue
        return True
    return False


def enable_structural_codebase_embeddings(
    embedding_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Rewrite a codebase embedding config to use the structural bridge when explicitly requested.

    The bridge is only activated if:
    1. structural_indexing_enabled is True in extra_config, AND
    2. victor-coding is installed, AND
    3. The user has NOT explicitly opted out via VICTOR_DISABLE_STRUCTURAL_BRIDGE env var

    By default, victor-coding uses its own providers (lancedb, proximadb) directly
    without needing this bridge. The bridge is only useful for advanced use cases where
    victor's structural chunking capabilities are needed.
    """

    import os

    config = dict(embedding_config)
    extra_config = _normalized_extra_config(config)

    # Check if structural indexing is explicitly disabled
    if not bool(extra_config.get("structural_indexing_enabled", False)):
        config["extra_config"] = extra_config
        return config

    # Check if the bridge is explicitly disabled via environment variable
    # This allows users to opt out of the bridge even if structural_indexing_enabled is True
    if os.environ.get("VICTOR_DISABLE_STRUCTURAL_BRIDGE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.debug(
            "[structural_bridge] Bridge disabled via VICTOR_DISABLE_STRUCTURAL_BRIDGE env var, "
            "using direct vector store: %s",
            config.get("vector_store", "lancedb"),
        )
        config["extra_config"] = extra_config
        return config

    vector_store = str(config.get("vector_store", "lancedb"))
    if vector_store == STRUCTURAL_CODEBASE_VECTOR_STORE:
        config["extra_config"] = extra_config
        return config

    if not register_structural_codebase_embedding_provider():
        logger.debug(
            "[structural_bridge] Bridge registration failed, using direct vector store: %s",
            vector_store,
        )
        config["extra_config"] = extra_config
        return config

    extra_config.setdefault("upstream_vector_store", vector_store)
    config["vector_store"] = STRUCTURAL_CODEBASE_VECTOR_STORE
    config["extra_config"] = extra_config
    logger.info(
        "[structural_bridge] Using structural bridge with upstream vector store: %s",
        vector_store,
    )
    return config


def _import_coding_codebase_module(module_suffix: str) -> Optional[Any]:
    """Import a coding-vertical codebase module without hard-coding package names.

    The structural bridge lives in core runtime code, so it must not couple
    itself directly to the external ``victor_coding`` package path. Resolve the
    module through the shared vertical import resolver instead.
    """
    for candidate in vertical_module_candidates("coding", module_suffix):
        try:
            return importlib.import_module(candidate)
        except ImportError:
            continue
    return None


def register_structural_codebase_embedding_provider() -> bool:
    """Register the structural bridge into victor-coding's embedding registry."""

    registry_module = _import_coding_codebase_module("codebase.embeddings.registry")
    if registry_module is None:
        return False

    registry = getattr(registry_module, "EmbeddingRegistry", None)
    if registry is None:
        return False

    if registry.is_registered(STRUCTURAL_CODEBASE_VECTOR_STORE):
        return True

    provider_class = get_structural_codebase_embedding_provider_class()
    if provider_class is None:
        return False

    registry.register(STRUCTURAL_CODEBASE_VECTOR_STORE, provider_class)
    return True


def get_structural_codebase_embedding_provider_class() -> Optional[type[Any]]:
    """Return the lazily constructed victor-coding bridge provider class."""

    global _STRUCTURAL_PROVIDER_CLASS
    if _STRUCTURAL_PROVIDER_CLASS is not None:
        return _STRUCTURAL_PROVIDER_CLASS

    base_module = _import_coding_codebase_module("codebase.embeddings.base")
    if base_module is None:
        return None

    base_provider_class = getattr(base_module, "BaseEmbeddingProvider", None)
    if base_provider_class is None:
        return None

    class StructuralCodebaseEmbeddingBridge(base_provider_class):  # type: ignore[misc, valid-type]
        """Bridge victor-coding's provider API to Victor's structural chunking stack."""

        def __init__(self, config: Any):
            super().__init__(config)
            self._delegate: Optional[InternalEmbeddingProvider] = None
            self._extra_config = dict(getattr(config, "extra_config", {}) or {})
            self._upstream_vector_store = str(
                self._extra_config.get("upstream_vector_store", "lancedb")
            )
            self._workspace_root = _infer_workspace_root(
                explicit_root=self._extra_config.get("workspace_root"),
                persist_directory=getattr(config, "persist_directory", None),
            )
            self._chunking_strategy = _normalized_chunking_strategy(self._extra_config)
            self._language_overrides = _normalized_language_overrides(self._extra_config)
            self._chunk_size = int(
                self._extra_config.get("chunk_size", DEFAULT_CODEBASE_CHUNK_SIZE)
            )
            self._chunk_overlap = int(
                self._extra_config.get("chunk_overlap", DEFAULT_CODEBASE_CHUNK_OVERLAP)
            )
            self._chunker = create_code_chunker(
                self._chunking_strategy,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            self._pending_file: Optional[_PendingFileBatch] = None
            self._full_rebuild_mode = False
            self._files_flushed_since_clear: set[str] = set()
            self._lock = asyncio.Lock()

        async def initialize(self) -> None:
            if self._initialized:
                return

            internal_config = InternalEmbeddingConfig(
                vector_store=self._upstream_vector_store,
                persist_directory=getattr(self.config, "persist_directory", None),
                distance_metric=getattr(self.config, "distance_metric", "cosine"),
                embedding_model_type=getattr(
                    self.config, "embedding_model_type", "sentence-transformers"
                ),
                embedding_model_name=getattr(
                    self.config, "embedding_model_name", "BAAI/bge-small-en-v1.5"
                ),
                embedding_api_key=getattr(self.config, "embedding_api_key", None),
                extra_config=self._build_delegate_extra_config(),
            )
            self._delegate = InternalEmbeddingRegistry.create(internal_config)
            await self._delegate.initialize()
            self._initialized = True

        async def embed_text(self, text: str) -> list[float]:
            await self.initialize()
            return await self._delegate.embed_text(text)  # type: ignore[union-attr]

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            await self.initialize()
            return await self._delegate.embed_batch(texts)  # type: ignore[union-attr]

        async def index_document(
            self,
            doc_id: str,
            content: str,
            metadata: Optional[dict[str, Any]] = None,
        ) -> None:
            await self.initialize()
            async with self._lock:
                self._full_rebuild_mode = False
                await self._append_document(
                    {
                        "id": doc_id,
                        "content": content,
                        "metadata": dict(metadata or {}),
                    },
                    replace_existing=True,
                )

        async def index_documents(self, documents: list[dict[str, Any]]) -> None:
            await self.initialize()
            async with self._lock:
                for document in documents:
                    await self._append_document(
                        {
                            "id": document["id"],
                            "content": document["content"],
                            "metadata": dict(document.get("metadata", {})),
                        },
                        replace_existing=not self._full_rebuild_mode,
                    )

        async def search_similar(
            self,
            query: str,
            limit: int = 10,
            filter_metadata: Optional[dict[str, Any]] = None,
        ) -> list[Any]:
            await self.initialize()
            async with self._lock:
                await self._flush_pending_file()
                results = await self._delegate.search_similar(  # type: ignore[union-attr]
                    query=query,
                    limit=limit,
                    filter_metadata=filter_metadata,
                )
            for result in results:
                if isinstance(getattr(result, "metadata", None), dict):
                    result.metadata.setdefault("chunking_strategy", self._chunking_strategy)
                    result.metadata.setdefault("vector_store", self._upstream_vector_store)
            return results

        async def delete_document(self, doc_id: str) -> None:
            await self.initialize()
            async with self._lock:
                await self._flush_pending_file()
                file_path = _extract_file_path_from_doc_id(doc_id)
                if file_path:
                    await self._delegate.delete_by_file(file_path)  # type: ignore[union-attr]
                else:
                    await self._delegate.delete_document(doc_id)  # type: ignore[union-attr]

        async def delete_by_file(self, file_path: str) -> int:
            await self.initialize()
            async with self._lock:
                await self._flush_pending_file()
                return await self._delegate.delete_by_file(file_path)  # type: ignore[union-attr]

        async def clear_index(self) -> None:
            await self.initialize()
            async with self._lock:
                self._pending_file = None
                self._files_flushed_since_clear.clear()
                self._full_rebuild_mode = True
                await self._delegate.clear_index()  # type: ignore[union-attr]

        async def get_stats(self) -> dict[str, Any]:
            await self.initialize()
            async with self._lock:
                await self._flush_pending_file()
                stats = dict(await self._delegate.get_stats())  # type: ignore[union-attr]
            stats.update(
                {
                    "provider": STRUCTURAL_CODEBASE_VECTOR_STORE,
                    "upstream_vector_store": self._upstream_vector_store,
                    "chunking_strategy": self._chunking_strategy,
                    "chunk_size": self._chunk_size,
                    "chunk_overlap": self._chunk_overlap,
                    "workspace_root": str(self._workspace_root),
                }
            )
            return stats

        async def close(self) -> None:
            if not self._initialized:
                return
            async with self._lock:
                await self._flush_pending_file()
                await self._delegate.close()  # type: ignore[union-attr]
                self._delegate = None
                self._initialized = False

        def _build_delegate_extra_config(self) -> dict[str, Any]:
            extra_config = dict(self._extra_config)
            extra_config.pop("upstream_vector_store", None)
            extra_config.pop("workspace_root", None)
            extra_config.pop("structural_indexing_enabled", None)
            extra_config.setdefault("chunking_strategy", self._chunking_strategy)
            extra_config.setdefault("code_chunking_strategy", self._chunking_strategy)
            extra_config.setdefault("chunk_size", self._chunk_size)
            extra_config.setdefault("chunk_overlap", self._chunk_overlap)
            return extra_config

        async def _append_document(
            self,
            document: dict[str, Any],
            *,
            replace_existing: bool,
        ) -> None:
            metadata = document.get("metadata", {})
            file_path = metadata.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                await self._flush_pending_file()
                await self._delegate.index_documents([document])  # type: ignore[union-attr]
                return

            if self._pending_file and self._pending_file.file_path != file_path:
                await self._flush_pending_file()

            if self._pending_file is None:
                self._pending_file = _PendingFileBatch(
                    file_path=file_path,
                    documents=[],
                    replace_existing=replace_existing
                    or file_path in self._files_flushed_since_clear,
                )

            self._pending_file.documents.append(document)
            self._pending_file.replace_existing = (
                self._pending_file.replace_existing or replace_existing
            )

        async def _flush_pending_file(self) -> None:
            pending = self._pending_file
            if pending is None:
                return

            self._pending_file = None
            file_path = pending.file_path

            if pending.replace_existing:
                await self._delegate.delete_by_file(file_path)  # type: ignore[union-attr]

            chunk_documents = self._build_chunk_documents(file_path, pending.documents)
            if chunk_documents:
                await self._delegate.index_documents(chunk_documents)  # type: ignore[union-attr]
            elif pending.documents:
                await self._delegate.index_documents(pending.documents)  # type: ignore[union-attr]

            self._files_flushed_since_clear.add(file_path)

        def _build_chunk_documents(
            self,
            file_path: str,
            documents: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            full_path = self._workspace_root / file_path
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                logger.debug("Falling back to symbol documents for missing file: %s", full_path)
                return []

            if not content.strip():
                return []

            symbols = _collect_bridge_symbols(documents)
            language = _resolve_language(
                full_path,
                documents,
                language_overrides=self._language_overrides,
            )
            parse_context = _build_tree_sitter_parse_context(
                file_path=str(full_path),
                content=content,
                language=language,
                chunking_strategy=self._chunking_strategy,
            )
            chunks = self._chunker.chunk(
                file_path=file_path,
                content=content,
                context=CodeChunkingContext(symbols=symbols, parse_context=parse_context),
            )

            chunk_documents: list[dict[str, Any]] = []
            for index, chunk in enumerate(chunks):
                symbol_name = chunk.symbol_name
                unified_id = (
                    f"symbol:{file_path}:{symbol_name}" if symbol_name else f"file:{file_path}"
                )
                chunk_id = _make_chunk_id(
                    file_path=file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    symbol_name=symbol_name,
                    chunk_type=chunk.chunk_type,
                    ordinal=index,
                )
                chunk_documents.append(
                    {
                        "id": chunk_id,
                        "content": chunk.content,
                        "metadata": {
                            "file_path": file_path,
                            "symbol_name": symbol_name,
                            "qualified_name": symbol_name,
                            "symbol_type": chunk.chunk_type,
                            "line_number": chunk.start_line,
                            "end_line": chunk.end_line,
                            "parent_symbol": chunk.parent_symbol,
                            "chunk_id": chunk_id,
                            "chunk_type": chunk.chunk_type,
                            "chunking_strategy": self._chunking_strategy,
                            "language": language,
                            "unified_id": unified_id,
                            "vector_store": self._upstream_vector_store,
                            "source": "structural_codebase_bridge",
                        },
                    }
                )
            return chunk_documents

    StructuralCodebaseEmbeddingBridge.__name__ = "StructuralCodebaseEmbeddingBridge"
    _STRUCTURAL_PROVIDER_CLASS = StructuralCodebaseEmbeddingBridge
    return _STRUCTURAL_PROVIDER_CLASS


def _infer_workspace_root(explicit_root: Any, persist_directory: Any) -> Path:
    if isinstance(explicit_root, str) and explicit_root:
        return Path(explicit_root).expanduser().resolve()

    if isinstance(persist_directory, str) and persist_directory:
        persist_path = Path(persist_directory).expanduser().resolve()
        if persist_path.name == "embeddings" and persist_path.parent.name == ".victor":
            return persist_path.parent.parent
        if persist_path.parent.name == ".victor":
            return persist_path.parent.parent
    return Path.cwd().resolve()


def _collect_bridge_symbols(documents: list[dict[str, Any]]) -> list[_BridgeSymbol]:
    symbols: list[_BridgeSymbol] = []
    seen: set[tuple[str, str, int, int, Optional[str]]] = set()
    for document in documents:
        metadata = document.get("metadata", {})
        symbol_name = metadata.get("symbol_name")
        if not isinstance(symbol_name, str) or not symbol_name:
            continue
        symbol_type = str(metadata.get("symbol_type") or "symbol")
        start_line = int(metadata.get("line_number") or 1)
        end_line = int(metadata.get("end_line") or start_line)
        if end_line < start_line:
            end_line = start_line
        parent_symbol = metadata.get("parent_symbol")
        if parent_symbol is None and "." in symbol_name:
            parent_symbol = symbol_name.rsplit(".", 1)[0]
        key = (symbol_name, symbol_type, start_line, end_line, parent_symbol)
        if key in seen:
            continue
        seen.add(key)
        symbols.append(
            _BridgeSymbol(
                name=symbol_name,
                symbol_type=symbol_type,
                line_start=start_line,
                line_end=end_line,
                parent_symbol=parent_symbol if isinstance(parent_symbol, str) else None,
            )
        )
    return sorted(symbols, key=lambda item: (item.line_start, item.line_end, item.name))


def _resolve_language(
    full_path: Path,
    documents: list[dict[str, Any]],
    *,
    language_overrides: Optional[Mapping[str, str]] = None,
) -> str:
    for document in documents:
        metadata = document.get("metadata", {})
        language = metadata.get("language")
        if isinstance(language, str) and language:
            return language

    if language_overrides:
        override = language_overrides.get(full_path.suffix.lower())
        if isinstance(override, str) and override:
            return override

    try:
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import LanguageRegistryProtocol

        registry = CapabilityRegistry.get_instance()
        language_registry = registry.get(LanguageRegistryProtocol)
        if language_registry is not None and hasattr(language_registry, "detect_language"):
            detected = language_registry.detect_language(full_path)
            if isinstance(detected, str) and detected:
                return detected
    except Exception:
        pass

    try:
        chunker_module = _import_coding_codebase_module("codebase.chunker")
        if chunker_module is None:
            raise ImportError("coding chunker module unavailable")
        detect_language = getattr(chunker_module, "detect_language", None)
        if callable(detect_language):
            detected = detect_language(str(full_path))
            if isinstance(detected, str) and detected:
                return detected
    except Exception:
        pass

    return _LANGUAGE_BY_SUFFIX.get(full_path.suffix.lower(), "text")


def _build_tree_sitter_parse_context(
    *,
    file_path: str,
    content: str,
    language: str,
    chunking_strategy: str,
) -> Optional[TreeSitterParseContext]:
    if chunking_strategy not in {"tree_sitter_structural", "ast_structural", "cast"}:
        return None

    # Prefer the enhanced TreeSitterAnalysisProtocol provider so chunking
    # reuses the same parse that the rest of the framework runs on. This
    # avoids re-parsing the file in every consumer (symbol extraction,
    # edge extraction, chunking) and lets root code stop importing
    # victor_coding directly.
    chunk_ctx = _build_chunk_context_via_provider(
        file_path=file_path, content=content, language=language
    )
    if chunk_ctx is not None:
        return chunk_ctx

    try:
        get_parser = load_tree_sitter_get_parser()
        parser = get_parser(language)
        tree = parser.parse(content.encode("utf-8"))
    except Exception as exc:
        logger.debug("Tree-sitter parse unavailable for %s (%s): %s", file_path, language, exc)
        return None

    try:
        return TreeSitterParseContext.from_content(content, tree.root_node)
    except Exception as exc:
        logger.debug("Failed to build parse context for %s: %s", file_path, exc)
        return None


def _build_chunk_context_via_provider(
    *, file_path: str, content: str, language: str
) -> Optional[TreeSitterParseContext]:
    """Try to build a parse context from the TreeSitterAnalysisProtocol provider."""
    try:
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import TreeSitterAnalysisProtocol

        registry = CapabilityRegistry.get_instance()
        if not registry.is_enhanced(TreeSitterAnalysisProtocol):
            return None
        provider = registry.get(TreeSitterAnalysisProtocol)
        if provider is None or not provider.supports_language(language):
            return None
        view = provider.build_chunk_context(content, language, file_path=file_path)
    except Exception as exc:
        logger.debug(
            "Analysis provider chunk context failed for %s (%s): %s",
            file_path,
            language,
            exc,
        )
        return None
    if view is None or getattr(view, "root_node", None) is None:
        return None
    try:
        return TreeSitterParseContext.from_content(content, view.root_node)
    except Exception as exc:
        logger.debug("Failed to wrap provider chunk context for %s: %s", file_path, exc)
        return None


def _make_chunk_id(
    *,
    file_path: str,
    start_line: int,
    end_line: int,
    symbol_name: Optional[str],
    chunk_type: str,
    ordinal: int,
) -> str:
    payload = f"{file_path}:{start_line}:{end_line}:{symbol_name or ''}:{chunk_type}:{ordinal}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"chunk:{file_path}:{digest}"


def _extract_file_path_from_doc_id(doc_id: str) -> Optional[str]:
    if doc_id.startswith("symbol:"):
        remainder = doc_id[len("symbol:") :]
        parts = remainder.rsplit(":", 1)
        if len(parts) == 2 and parts[0]:
            return parts[0]
    if doc_id.startswith("file:"):
        remainder = doc_id[len("file:") :]
        if remainder:
            return remainder
    if doc_id.startswith("chunk:"):
        remainder = doc_id[len("chunk:") :]
        parts = remainder.rsplit(":", 1)
        if len(parts) == 2 and parts[0]:
            return parts[0]
    return None


__all__ = [
    "CODEBASE_INDEX_MANIFEST_NAME",
    "CODEBASE_INDEX_SCHEMA_VERSION",
    "DEFAULT_CODEBASE_CHUNKING_STRATEGY",
    "DEFAULT_CODEBASE_CHUNK_SIZE",
    "DEFAULT_CODEBASE_CHUNK_OVERLAP",
    "STRUCTURAL_CODEBASE_VECTOR_STORE",
    "build_codebase_index_manifest",
    "enable_structural_codebase_embeddings",
    "get_structural_codebase_embedding_provider_class",
    "has_compatible_codebase_index_manifest",
    "has_persisted_codebase_index_data",
    "read_codebase_index_manifest",
    "register_structural_codebase_embedding_provider",
    "write_codebase_index_manifest",
]


# Auto-register the structural bridge provider on module import
# This ensures victor-coding's registry knows about the bridge without requiring
# explicit registration calls by users of the framework.
_try_auto_registration: bool = True


def _auto_register_bridge_on_import() -> None:
    """Attempt to register the structural bridge on module import.

    This is called at module import time to ensure the bridge is available
    for victor-coding's embedding registry. Failures are silently ignored
    since the bridge is an optional enhancement.
    """
    global _try_auto_registration
    if not _try_auto_registration:
        return

    try:
        if register_structural_codebase_embedding_provider():
            logger.debug(
                "[structural_bridge] Auto-registered %s provider",
                STRUCTURAL_CODEBASE_VECTOR_STORE,
            )
    except Exception:
        # Silently ignore registration failures - the bridge is optional
        _try_auto_registration = False


_auto_register_bridge_on_import()
