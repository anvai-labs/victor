import asyncio
import fnmatch
import importlib
import importlib.util
import logging
import os
import re
import warnings
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from victor_contracts.enrichment_runtime import CODE_PATTERNS
from victor_contracts.search_runtime import (
    CODEBASE_INDEX_MANIFEST_NAME,
    DEFAULT_CODEBASE_CHUNKING_STRATEGY,
    DEFAULT_CODEBASE_CHUNK_OVERLAP,
    DEFAULT_CODEBASE_CHUNK_SIZE,
    STRUCTURAL_CODEBASE_VECTOR_STORE,
    build_codebase_index_manifest,
    enable_structural_codebase_embeddings,
    enrich_code_search_results,
    extract_skeleton,
    has_compatible_codebase_index_manifest,
    has_persisted_codebase_index_data,
    rerank_code_search_results,
    write_codebase_index_manifest,
)
from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.common import EXCLUDE_DIRS, DEFAULT_CODE_EXTENSIONS, latest_mtime
from victor.tools.decorators import tool
from victor.tools.formatters import format_search_results

if TYPE_CHECKING:
    from victor.tools.cache_manager import CacheNamespace

    # File watching types
    from victor_contracts.indexing_runtime import FileChangeEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntegrityProbeOutcome:
    """Outcome of validating a persisted semantic index before reuse."""

    rebuilt: bool = False
    stale: bool = False


def _coerce_integrity_probe_outcome(value: Any) -> IntegrityProbeOutcome:
    """Normalize legacy boolean probe results into the explicit outcome contract."""

    if isinstance(value, IntegrityProbeOutcome):
        return value
    if isinstance(value, bool):
        return IntegrityProbeOutcome(rebuilt=value, stale=False)
    return IntegrityProbeOutcome()


@dataclass
class SearchFilters:
    """Filters for code search operations.

    Consolidates filter parameters into a single object.
    Reduces parameter count from 5 to 1 for code_search.
    """

    file_pattern: Optional[str] = None  # Search by filename pattern
    symbol: Optional[str] = None  # Search by symbol name
    language: Optional[str] = None  # Filter by programming language
    test_only: Optional[bool] = None  # Only search test files
    extensions: Optional[List[str]] = None  # Filter by file extensions


# Legacy cache for semantic indexes (use _get_index_cache() for DI support)
_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
_warned_graph_writer_compatibility = False


def clear_index_cache() -> None:
    """Clear all cached indexes. Call between benchmark tasks for isolation."""
    _INDEX_CACHE.clear()


def mark_index_cache_stale_for_path(
    file_path: str,
    *,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> int:
    """Mark cached indexes stale when a file changes before watchers catch up.

    This is used by write/edit flows to proactively invalidate any in-process
    semantic indexes that include the modified file. It closes the freshness
    gap between a successful edit and the asynchronous file-watcher update.

    Args:
        file_path: Modified file path
        exec_ctx: Optional execution context used to resolve DI-backed caches

    Returns:
        Number of cached index roots marked stale
    """

    try:
        changed_path = Path(file_path).expanduser().resolve()
    except Exception:
        changed_path = Path(file_path)

    stale_count = 0
    for root_key, cache_entry in _get_index_cache(exec_ctx).items():
        if not isinstance(cache_entry, dict):
            continue
        try:
            root_path = Path(root_key).expanduser().resolve()
        except Exception:
            continue

        try:
            changed_path.relative_to(root_path)
        except ValueError:
            continue

        cache_entry["stale"] = True
        cache_entry["stale_reason"] = "modified_file"
        cache_entry["last_invalidated_path"] = str(changed_path)
        cache_entry["invalidated_at"] = time.time()
        stale_count += 1

    if stale_count:
        logger.info(
            "[code_search] Marked %d cached index root(s) stale after file change: %s",
            stale_count,
            changed_path,
        )

    return stale_count


def _get_instance_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read explicitly assigned instance attributes without triggering mock fallbacks."""

    try:
        instance_dict = vars(obj)
    except TypeError:
        instance_dict = getattr(obj, "__dict__", None)

    if isinstance(instance_dict, dict) and name in instance_dict:
        return instance_dict[name]
    return default


def _get_settings_group_value(
    settings: Any, group_name: str, field_name: str, default: Any = None
) -> Any:
    """Resolve a setting from its nested group, with flat fallback for test doubles."""

    group = getattr(settings, group_name, None)
    if group is not None and hasattr(group, field_name):
        return getattr(group, field_name)
    return getattr(settings, field_name, default)


def _get_search_setting(settings: Any, field_name: str, default: Any = None) -> Any:
    """Resolve a code-search setting from the canonical search group."""

    return _get_settings_group_value(settings, "search", field_name, default)


def _resolve_graph_writer_mode(settings: Any) -> str:
    """Resolve graph writer mode with explicit compatibility-mode warning."""

    global _warned_graph_writer_compatibility

    graph_writer_mode = _get_search_setting(settings, "codebase_graph_writer_mode", "")
    if not graph_writer_mode:
        graph_writer_mode = os.environ.get("VICTOR_CODEBASE_GRAPH_WRITER_MODE", "off")

    mode = str(graph_writer_mode).strip().lower()
    if mode in {"", "off", "false", "0"}:
        return "off"
    if mode == "compatibility":
        if not _warned_graph_writer_compatibility:
            warnings.warn(
                "codebase_graph_writer_mode='compatibility' is deprecated compatibility "
                "support. Victor-coding graph writes should be treated as explicit, "
                "temporary compatibility behavior and are not the primary graph source.",
                UserWarning,
                stacklevel=2,
            )
            _warned_graph_writer_compatibility = True
        return "compatibility"
    if mode in {"on", "true", "1"}:
        warnings.warn(
            "codebase_graph_writer_mode only supports 'off' (default) and "
            "'compatibility' (deprecated). Treating value as 'off'.",
            UserWarning,
            stacklevel=2,
        )
        return "off"
    warnings.warn(
        "Unknown codebase_graph_writer_mode=%r; supported values are 'off' and "
        "'compatibility' (deprecated). Treating value as 'off'." % mode,
        UserWarning,
        stacklevel=2,
    )
    return "off"


def _get_embedding_setting(settings: Any, field_name: str, default: Any = None) -> Any:
    """Resolve an embedding setting from the canonical embedding group."""

    return _get_settings_group_value(settings, "embedding", field_name, default)


def _resolve_codebase_embedding_model(settings: Any) -> Any:
    """Resolve the codebase embedding model with nested-group fallback semantics."""

    codebase_model = _get_search_setting(settings, "codebase_embedding_model", None)
    if codebase_model not in (None, ""):
        return codebase_model
    return _get_embedding_setting(settings, "unified_embedding_model", "BAAI/bge-small-en-v1.5")


def _get_index_build_failure_cache(exec_ctx: Optional[Any] = None) -> Any:
    """Resolve the index-build failure cache without triggering mock fallback attrs."""

    cache_manager = None
    if exec_ctx and isinstance(exec_ctx, dict):
        cache_manager = exec_ctx.get("cache_manager")
    elif exec_ctx is not None:
        from victor.tools.context import ToolExecutionContext

        if isinstance(exec_ctx, ToolExecutionContext):
            cache_manager = exec_ctx.cache_manager

    if cache_manager:
        return cache_manager.get_namespace("index_build_failures")

    failure_cache = _get_instance_attr(_get_or_build_index, "_failure_cache")
    if failure_cache is None:
        failure_cache = {}
        _get_or_build_index._failure_cache = failure_cache
    return failure_cache


async def _probe_index_integrity(
    index: Any, timeout: Optional[float] = None
) -> IntegrityProbeOutcome:
    """Validate persistent index integrity with a lightweight check.

    Returns an explicit outcome so callers can distinguish a healthy persisted
    index from a successfully rebuilt one and from a still-stale failed rebuild.

    ENHANCEMENT: Distinguishes between init-time races and actual corruption.

    Args:
        index: The codebase index to probe
        timeout: Probe timeout in seconds (default: from VICTOR_TIMEOUT_INDEX_PROBE env var or 15.0)
    """
    import os

    if timeout is None:
        # Use centralized timeout config with env override
        timeout = float(
            os.environ.get(
                "VICTOR_TIMEOUT_INDEX_PROBE",
                os.environ.get("VICTOR_INDEX_PROBE_TIMEOUT", "15.0"),
            )
        )

    probe_task: Optional[asyncio.Task[Any]] = None
    try:
        # NEW: Check if index is currently being built
        if _get_instance_attr(index, "_is_indexing", False) is True:
            logger.debug("Index currently being built, skipping integrity check")
            return IntegrityProbeOutcome()  # Don't trigger rebuild during init

        # Quick check: does the vector store have data?
        store = _get_instance_attr(index, "_vector_store", None)
        if store is None:
            store = _get_instance_attr(index, "vector_store", None)
        if store:
            table = getattr(store, "_table", None)
            if table is not None:
                row_count = table.count_rows() if hasattr(table, "count_rows") else -1
                if isinstance(row_count, (int, float)) and row_count > 0:
                    logger.info("Persistent index healthy: %d rows in vector store", row_count)
                    return IntegrityProbeOutcome()  # Healthy

        # Fallback: try a semantic search with timeout
        probe_task = asyncio.get_running_loop().create_task(
            index.semantic_search(query="test", max_results=1)
        )
        done, pending = await asyncio.wait({probe_task}, timeout=timeout)
        if pending:
            probe_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "Persistent index probe timed out after %.2fs; rebuilding inline",
                timeout,
            )
            if hasattr(index, "_is_indexed"):
                index._is_indexed = False
            rebuilt = await _background_index_rebuild(index)
            return IntegrityProbeOutcome(rebuilt=rebuilt, stale=not rebuilt)
        await probe_task
        return IntegrityProbeOutcome()  # Healthy — no rebuild needed
    except asyncio.CancelledError:
        if probe_task is not None and not probe_task.done():
            probe_task.cancel()
            await asyncio.gather(probe_task, return_exceptions=True)
        raise
    except Exception as e:
        # ENHANCEMENT: Better error classification
        error_msg = str(e) if str(e) else f"{type(e).__name__}"

        # Don't rebuild for common init-time / transient errors
        transient_keywords = [
            "locked",
            "timeout",
            "timed out",
            "not ready",
            "unavailable",
            "initializing",
            "building",
            "in progress",
            "not have attribute",
            "has no attribute",  # missing attr = code bug, not corruption
        ]

        if any(kw in error_msg.lower() for kw in transient_keywords):
            logger.debug("Transient init issue detected (%s), skipping rebuild", error_msg)
            return IntegrityProbeOutcome()  # Don't rebuild for transient issues

        # Log with full error details for debugging
        logger.warning("Persistent index corrupt (%s); rebuilding inline", error_msg)

        if hasattr(index, "_is_indexed"):
            index._is_indexed = False

        # Rebuild inline while the startup path already holds the index lock.
        rebuilt = await _background_index_rebuild(index)
        return IntegrityProbeOutcome(rebuilt=rebuilt, stale=not rebuilt)


def _build_codebase_embedding_extra_config(
    settings: Any, root: Optional[Path] = None
) -> Dict[str, Any]:
    """Build provider-specific embedding config from settings.

    Existing search settings define dimension and batch size but the code_search
    path historically failed to forward them to the backend provider. This
    helper keeps the translation in one place and allows future provider-
    specific extensions without widening the factory call sites.
    """

    extra_config: Dict[str, Any] = {
        "dimension": _get_search_setting(settings, "codebase_dimension", 384),
        "batch_size": _get_search_setting(settings, "codebase_batch_size", 32),
        "structural_indexing_enabled": _get_search_setting(
            settings, "codebase_structural_indexing_enabled", False
        ),
        "code_chunking_strategy": _get_search_setting(
            settings, "codebase_chunking_strategy", DEFAULT_CODEBASE_CHUNKING_STRATEGY
        ),
        "chunk_size": _get_search_setting(
            settings, "codebase_chunk_size", DEFAULT_CODEBASE_CHUNK_SIZE
        ),
        "chunk_overlap": _get_search_setting(
            settings, "codebase_chunk_overlap", DEFAULT_CODEBASE_CHUNK_OVERLAP
        ),
    }

    custom_extra = _get_search_setting(settings, "codebase_embedding_extra_config", None)
    if isinstance(custom_extra, dict):
        extra_config.update(custom_extra)

    if root is not None:
        extra_config.setdefault("workspace_root", str(root))

    return extra_config


def _build_codebase_embedding_config(settings: Any, root: Path) -> Dict[str, Any]:
    """Build the embedding configuration passed to the codebase index factory."""

    from victor.config.settings import get_project_paths

    default_persist_dir = str(get_project_paths(root).embeddings_dir)
    config = {
        "vector_store": _get_search_setting(settings, "codebase_vector_store", "lancedb"),
        "embedding_model_type": _get_search_setting(
            settings, "codebase_embedding_provider", "sentence-transformers"
        ),
        "embedding_model_name": _resolve_codebase_embedding_model(settings),
        "persist_directory": _get_search_setting(settings, "codebase_persist_directory", None)
        or default_persist_dir,
        "extra_config": _build_codebase_embedding_extra_config(settings, root),
    }
    return enable_structural_codebase_embeddings(config)


def _collect_code_search_backend_metadata(index: Any, settings: Any) -> Dict[str, Any]:
    """Collect compact backend metadata for code_search responses."""

    metadata: Dict[str, Any] = {}
    provider = getattr(index, "embedding_provider", None)
    config = getattr(provider, "config", None)

    def _assign_if_present(key: str, value: Any) -> None:
        if value not in (None, "", {}):
            metadata[key] = value

    if config is not None:
        extra_config = getattr(config, "extra_config", None)
        raw_vector_store = getattr(config, "vector_store", None)
        if raw_vector_store == STRUCTURAL_CODEBASE_VECTOR_STORE and isinstance(extra_config, dict):
            _assign_if_present("vector_store", extra_config.get("upstream_vector_store"))
            metadata["indexing_backend"] = "structural_bridge"
            metadata["structural_indexing_enabled"] = True
        else:
            _assign_if_present("vector_store", raw_vector_store)
        _assign_if_present("embedding_provider", getattr(config, "embedding_model_type", None))
        _assign_if_present("embedding_model", getattr(config, "embedding_model_name", None))
        if isinstance(extra_config, dict):
            _assign_if_present("embedding_dimension", extra_config.get("dimension"))
            _assign_if_present("embedding_batch_size", extra_config.get("batch_size"))
            _assign_if_present(
                "chunking_strategy",
                extra_config.get("code_chunking_strategy") or extra_config.get("chunking_strategy"),
            )

    # Fall back to settings when the backend does not expose explicit config.
    metadata.setdefault(
        "vector_store",
        _get_search_setting(settings, "codebase_vector_store", "lancedb"),
    )
    metadata.setdefault(
        "embedding_provider",
        _get_search_setting(settings, "codebase_embedding_provider", "sentence-transformers"),
    )
    metadata.setdefault(
        "embedding_model",
        _resolve_codebase_embedding_model(settings),
    )
    metadata.setdefault(
        "embedding_dimension", _get_search_setting(settings, "codebase_dimension", 384)
    )
    metadata.setdefault(
        "embedding_batch_size", _get_search_setting(settings, "codebase_batch_size", 32)
    )
    metadata.setdefault(
        "chunking_strategy",
        _get_search_setting(
            settings, "codebase_chunking_strategy", DEFAULT_CODEBASE_CHUNKING_STRATEGY
        ),
    )
    return metadata


async def _finalize_index_storage(index: Any) -> None:
    """Flush provider-level buffered writes before marking an index reusable."""

    provider = getattr(index, "embedding_provider", None)
    config = getattr(provider, "config", None)
    if getattr(config, "vector_store", None) != STRUCTURAL_CODEBASE_VECTOR_STORE:
        return

    get_stats = getattr(provider, "get_stats", None)
    if not callable(get_stats):
        return

    try:
        await get_stats()
    except Exception as exc:
        logger.debug("Failed to finalize structural code_search provider writes: %s", exc)


def _cleanup_nested_victor_dirs(root: Path) -> None:
    """Remove .victor directories in unexpected locations.

    Addresses the issue where nested .victor directories are created
    when Victor is run from subdirectories. Only the top-level
    {project_root}/.victor should exist.

    Args:
        root: Project root directory
    """
    import shutil

    # List of known nested .victor paths to clean up
    nested_patterns = [
        root / "victor" / ".victor",  # victor/.victor
        root / ".victor" / "embeddings" / ".victor",  # .victor/embeddings/.victor
        root / "victor" / ".victor" / "embeddings",  # victor/.victor/embeddings
    ]

    for nested_path in nested_patterns:
        if nested_path.exists():
            try:
                size = sum(f.stat().st_size for f in nested_path.rglob("*") if f.is_file())
                logger.info(
                    "[code_search] Removing nested .victor directory: %s (%.1f MB)",
                    nested_path.relative_to(root),
                    size / (1024 * 1024),
                )
                shutil.rmtree(nested_path)
            except (OSError, PermissionError) as exc:
                logger.warning("[code_search] Failed to remove %s: %s", nested_path, exc)


def _load_codebase_index_factory_via_importlib() -> Optional[Any]:
    """Load a CodebaseIndex factory via runtime import fallback paths."""

    class _ImportedCodebaseIndexFactory:
        def __init__(self, index_cls: Any) -> None:
            self._index_cls = index_cls

        def create(self, root_path: str, **kwargs: Any) -> Any:
            return self._index_cls(root_path=root_path, **kwargs)

    for module_path in (
        "victor_coding.codebase.indexer",
        "victor.verticals.contrib.coding.codebase.indexer",
    ):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue

        index_cls = getattr(module, "CodebaseIndex", None)
        if index_cls is None:
            continue
        return _ImportedCodebaseIndexFactory(index_cls)

    return None


async def _background_index_rebuild(index: Any, rebuild_timeout: float = 120.0) -> bool:
    """Rebuild a corrupt index in the background without blocking callers."""
    import time

    start_time = time.time()
    try:
        index_path = getattr(index, "root", "unknown")
        logger.info(
            "Background index rebuild started (timeout=%ds, index=%s)",
            rebuild_timeout,
            index_path,
        )
        await asyncio.wait_for(index.index_codebase(), timeout=rebuild_timeout)
        await _finalize_index_storage(index)
        elapsed = time.time() - start_time
        logger.info(
            "Background index rebuild completed successfully in %.2fs (index=%s)",
            elapsed,
            index_path,
        )
        return True
    except asyncio.TimeoutError:
        logger.warning(
            "Background index rebuild timed out after %ds; index remains stale (index=%s)",
            rebuild_timeout,
            getattr(index, "root", "unknown"),
        )
        return False
    except Exception as err:
        error_msg = str(err) if str(err) else f"{type(err).__name__}"
        logger.warning(
            "Background index rebuild failed after %.2fs: %s (index=%s)",
            time.time() - start_time,
            error_msg,
            getattr(index, "root", "unknown"),
        )
        return False


def _filename_search_timeout() -> float:
    """Timeout (seconds) for the filename ``find`` subprocess (env-overridable).

    Walking a large source tree with ``find`` is O(files); the previous hardcoded
    15s fired on big repos. Override via ``VICTOR_TIMEOUT_FILENAME_SEARCH``.
    """
    return float(
        os.environ.get(
            "VICTOR_TIMEOUT_FILENAME_SEARCH",
            os.environ.get("VICTOR_FILENAME_SEARCH_TIMEOUT", "45.0"),
        )
    )


# Single-flight registry for background index rebuilds, keyed by project root.
# Without this, every stale code_search call spawns a fresh fire-and-forget
# rebuild; concurrent rebuilds on a large repo thrash the same embeddings store
# and cascade into repeated probe timeouts.
_INFLIGHT_INDEX_REBUILDS: Dict[str, "asyncio.Task[Any]"] = {}


def _spawn_index_rebuild_once(
    root_key: str, coro_factory: Callable[[], Any]
) -> "asyncio.Task[Any]":
    """Spawn a background rebuild for ``root_key`` unless one is already running.

    Returns the in-flight task (existing or newly created). ``coro_factory`` is
    only invoked when a new task is actually spawned, so no orphan coroutine is
    created when a rebuild is deduplicated.
    """
    existing = _INFLIGHT_INDEX_REBUILDS.get(root_key)
    if existing is not None and not existing.done():
        logger.info(
            "[code_search] Index rebuild already in-flight for %s; skipping duplicate",
            root_key,
        )
        return existing

    task = asyncio.create_task(coro_factory())
    _INFLIGHT_INDEX_REBUILDS[root_key] = task

    def _clear(done_task: "asyncio.Task[Any]", key: str = root_key) -> None:
        if _INFLIGHT_INDEX_REBUILDS.get(key) is done_task:
            _INFLIGHT_INDEX_REBUILDS.pop(key, None)

    task.add_done_callback(_clear)
    return task


def _get_index_cache(exec_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the index cache, preferring DI-injected cache if available.

    Args:
        exec_ctx: Execution context that may contain cache_manager

    Returns:
        Cache dict-like object for index storage
    """
    # Try to get from execution context (DI pattern)
    if exec_ctx is not None:
        # Check for ToolExecutionContext with cache_manager
        from victor.tools.context import ToolExecutionContext

        if isinstance(exec_ctx, ToolExecutionContext):
            if exec_ctx.cache_manager is not None:
                # Return the namespace's internal dict-like interface
                return exec_ctx.index_cache
        elif isinstance(exec_ctx, dict):
            # Legacy dict context - check for cache_manager
            cache_manager = exec_ctx.get("cache_manager")
            if cache_manager is not None:
                return cache_manager.index_cache

    # Fallback to global cache for backward compatibility
    return _INDEX_CACHE


def _cache_entry_matches_manifest(
    cache_entry: Optional[Dict[str, Any]],
    index_manifest: Dict[str, Any],
) -> bool:
    """Return True when an in-memory cache entry matches the requested index manifest."""

    if not isinstance(cache_entry, dict):
        return False
    cached_manifest = cache_entry.get("index_manifest")
    return isinstance(cached_manifest, dict) and cached_manifest == index_manifest


def _build_index_failure_key(root: Path, index_manifest: Dict[str, Any]) -> str:
    """Build a manifest-scoped failure cache key for code_search indexes."""

    import hashlib
    import json

    root_hash = hashlib.md5(str(root).encode()).hexdigest()
    manifest_hash = hashlib.md5(
        json.dumps(index_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{root_hash}:{manifest_hash}_build_failure"


def _build_provider_availability_failure_key(root: Path, settings: Any) -> str:
    """Build a repo-scoped failure key for missing semantic index providers.

    Missing provider availability is not specific to a single subdirectory index.
    Reuse the project root when the requested search root is inside the current
    workspace so repeated subdirectory searches do not re-run the same failed
    bootstrap path.
    """

    try:
        from victor.config.settings import get_project_paths

        project_root = Path(get_project_paths(root).project_root).resolve()
        try:
            root.resolve().relative_to(project_root)
            scope_root = project_root
        except ValueError:
            scope_root = root.resolve()
    except Exception:
        scope_root = root.resolve()

    config = _build_codebase_embedding_config(settings, scope_root)
    manifest = build_codebase_index_manifest(config)
    return _build_index_failure_key(scope_root, manifest).replace(
        "_build_failure", "_provider_unavailable"
    )


def _get_active_failure_error(failure_entry: Any) -> Optional[str]:
    """Return the cached error text when a failure entry is still active."""

    if failure_entry is None:
        return None

    if hasattr(failure_entry, "is_expired"):
        if failure_entry.is_expired():
            return None
        value = getattr(failure_entry, "value", {})
        if isinstance(value, dict):
            return str(value.get("error", "Unknown error"))
        return str(value)

    if isinstance(failure_entry, dict):
        return str(failure_entry.get("error", "Unknown error"))

    return str(failure_entry)


def _raise_if_cached_index_build_failure(failure_cache: Any, key: str) -> None:
    """Raise an ImportError when a matching cached failure is still active."""

    if failure_cache is None:
        return

    failure_entry = _failure_cache_get(failure_cache, key)
    error_text = _get_active_failure_error(failure_entry)
    if error_text:
        logger.info("[code_search] Skipping index build due to recent failure: %s", error_text)
        raise ImportError(
            f"Semantic index build recently failed: {error_text}\n"
            f"Use reindex=True to retry, or mode='literal' for keyword search."
        )


def _cache_index_build_failure(
    failure_cache: Any,
    key: str,
    error_msg: str,
    *,
    ttl: int = 3600,
) -> None:
    """Persist a semantic index failure in either cache backend."""

    if failure_cache is None:
        return

    from victor.tools.cache_manager import GenericCacheEntry

    failure_entry = GenericCacheEntry(
        value={"error": error_msg, "timestamp": time.time()},
        ttl=ttl,
    )
    _failure_cache_set(failure_cache, key, failure_entry)


def _is_recent_cached_failure_error(error_msg: str) -> bool:
    """Return True when an error message came from cached failure suppression."""

    return error_msg.startswith("Semantic index build recently failed:")


def _is_missing_semantic_provider_error(error_msg: str) -> bool:
    """Return True when semantic search is unavailable due to missing provider wiring."""

    lowered = error_msg.lower()
    return (
        "requires a codebase indexing provider" in lowered
        or "requires victor-coding package for semantic search" in lowered
        or "codebaseindex factory not registered" in lowered
        or "unknown embedding provider" in lowered
    )


def _extract_exception_cause(exc: Exception) -> str:
    """Extract the root cause from an exception chain for better error reporting.

    Walks the exception chain (__cause__ and __context__) to find the
    original error that triggered the failure.
    """
    current = exc
    visited = []
    max_depth = 5  # Prevent infinite loops

    while current and len(visited) < max_depth:
        if current in visited:
            break
        visited.append(current)

        # Format this exception
        exc_str = str(current).strip() if str(current).strip() else repr(current)
        exc_type = type(current).__name__

        # Check if there's a deeper cause
        if current.__cause__:
            current = current.__cause__
        elif current.__context__ and not isinstance(current, asyncio.TimeoutError):
            current = current.__context__
        else:
            # This is the root cause
            return f"{exc_type}: {exc_str[:500]}"

    # Fallback: return the original exception
    return f"{type(exc).__name__}: {str(exc)[:500] if str(exc) else repr(exc)[:500]}"


def _calculate_index_build_timeout(root_path: Path) -> float:
    """Calculate dynamic timeout for index building based on codebase size.

    Counts indexable source files across all supported languages (not just
    Python) as a proxy for embedding time. Formula:
    - Base timeout: 60 seconds
    - +5 seconds per source file (max 300s for files)
    - +1 second per 1000 estimated symbols (max 300s for symbols)
    - Cap at 600 seconds (10 minutes) for very large codebases

    Counting only ``.py`` files (the previous behavior) gave large Rust/Go/TS
    repositories the 60s base timeout, which always fired before the index
    finished and triggered repeated background rebuilds.

    Args:
        root_path: Path to the codebase root

    Returns:
        Timeout in seconds (float)
    """
    import os

    base_timeout = 60.0
    max_timeout = float(
        os.environ.get(
            "VICTOR_TIMEOUT_INDEX_BUILD_MAX",
            os.environ.get("VICTOR_INDEX_BUILD_TIMEOUT", "600.0"),
        )
    )

    try:
        # Count indexable source files (all supported languages) as a size proxy.
        source_file_count = 0
        total_size_bytes = 0
        source_extensions = tuple(DEFAULT_CODE_EXTENSIONS)
        # ``.victor`` can hold large caches (e.g. swe_bench_cache); ``env`` is a
        # common virtualenv dir. Exclude both alongside the shared EXCLUDE_DIRS.
        excluded_dirs = set(EXCLUDE_DIRS) | {".victor", "env"}

        for root, dirs, files in os.walk(root_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                if file.endswith(source_extensions):
                    source_file_count += 1
                    try:
                        total_size_bytes += os.path.getsize(os.path.join(root, file))
                    except (OSError, IOError):
                        pass

        # Estimate symbols: roughly 20 symbols per file on average.
        estimated_symbols = source_file_count * 20

        # Calculate timeout
        file_timeout = min(source_file_count * 5.0, 300.0)  # Max 5 minutes from files
        symbol_timeout = min(estimated_symbols / 1000.0 * 10.0, 300.0)  # Max 5 minutes from symbols
        size_timeout = min(total_size_bytes / (1024 * 1024) * 30.0, 200.0)  # Max 200s from size

        calculated_timeout = base_timeout + file_timeout + symbol_timeout + size_timeout
        final_timeout = min(calculated_timeout, max_timeout)

        logger.debug(
            "[code_search] Timeout calculation for %s: %d files, %d MB, ~%d symbols → %.1fs timeout",
            root_path,
            source_file_count,
            total_size_bytes // (1024 * 1024),
            estimated_symbols,
            final_timeout,
        )

        return final_timeout

    except Exception as exc:
        logger.warning(
            "[code_search] Failed to calculate dynamic timeout for %s: %s, using default 180s",
            root_path,
            exc,
        )
        return 180.0  # Fallback to previous default


def _failure_cache_get(failure_cache: Any, key: str) -> Any:
    """Get a failure entry from either a namespace cache or a plain dict."""

    getter = getattr(failure_cache, "get", None)
    if callable(getter):
        return getter(key)
    return None


def _failure_cache_set(failure_cache: Any, key: str, value: Any) -> None:
    """Set a failure entry on either a namespace cache or a plain dict."""

    setter = getattr(failure_cache, "set", None)
    if callable(setter):
        setter(key, value)
        return
    if isinstance(failure_cache, dict):
        failure_cache[key] = value


def _failure_cache_delete(failure_cache: Any, key: str) -> None:
    """Delete a failure entry from either a namespace cache or a plain dict."""

    deleter = getattr(failure_cache, "delete", None)
    if callable(deleter):
        deleter(key)
        return
    if isinstance(failure_cache, dict):
        failure_cache.pop(key, None)


def _is_cached_index_stale(root: Path, exec_ctx: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when the current cached semantic index for a root is marked stale."""

    cache_entry = _get_index_cache(exec_ctx).get(str(root), {})
    return bool(cache_entry.get("stale", False))


def _classify_semantic_fallback(exc: BaseException, *, scope: str) -> str:
    """Return a precise fallback reason for semantic code-search failures."""

    if isinstance(exc, asyncio.TimeoutError):
        return f"{scope}_timeout"
    if _is_missing_semantic_provider_error(str(exc)):
        return f"{scope}_provider_unavailable"
    return f"{scope}_error"


def _decorate_literal_fallback_result(
    result: Dict[str, Any],
    *,
    fallback: str,
    filters_applied: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach consistent fallback context to literal-search results."""

    if mode_override:
        result["mode"] = mode_override
    result["fallback"] = fallback
    merged_metadata = result.get("metadata")
    if not isinstance(merged_metadata, dict):
        merged_metadata = {}
    if filters_applied:
        merged_metadata["filters_applied"] = filters_applied
    if metadata:
        merged_metadata.update(metadata)
    if isinstance(fallback, str) and fallback.startswith("semantic_"):
        # Surface a prominent, model-visible note (not just nested metadata) so the agent
        # stops re-issuing natural-language semantic queries against a literal-only engine.
        # In the failing session the model looped, unaware semantic search was unavailable.
        if result.get("count") == 0 and result.get("mode") == "literal":
            merged_metadata["fallback_guidance"] = (
                "Semantic search was unavailable and literal fallback found no matches. "
                "Retry with concrete identifiers, symbols, or short tokens using mode='literal'."
            )
            result["note"] = (
                "Semantic code search is unavailable in this session (literal keyword search "
                "only) and found no matches. Retry with exact identifiers, symbols, file names, "
                "or short literal tokens — do not re-issue natural-language semantic queries. "
                "Install victor-coding to enable semantic search."
            )
        else:
            result.setdefault(
                "note",
                "Semantic code search is unavailable in this session; results are from literal "
                "keyword search. Prefer exact identifiers, symbols, or file names over prose.",
            )
    if merged_metadata:
        result["metadata"] = merged_metadata
    return result


def _requested_mode_literal_fallback_metadata(mode: str) -> Dict[str, Any]:
    """Describe early literal fallbacks for specialized search modes."""

    if mode in {"bugs", "localize", "impact"}:
        return {"requested_mode": mode, "fallback_mode": "literal"}
    return {}


def _early_semantic_fallback_metadata(
    mode: str,
    *,
    literal_escalation_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Preserve the original caller intent across early semantic fallbacks."""

    if literal_escalation_metadata:
        return dict(literal_escalation_metadata)
    return _requested_mode_literal_fallback_metadata(mode)


def _literal_fallback_mode_override(
    requested_mode: str,
    *,
    literal_escalation_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Preserve explicit direct-route modes when semantic fallback returns to literal search."""

    if literal_escalation_metadata and requested_mode in {"literal", "text"}:
        return requested_mode
    return None


def _normalize_search_filters(filters: Optional[Any]) -> Optional[SearchFilters]:
    """Trim user-provided filter text without mutating the original filter object."""

    if filters is None:
        return None

    if isinstance(filters, dict):
        filters = SearchFilters(
            file_pattern=filters.get("file_pattern"),
            symbol=filters.get("symbol"),
            language=filters.get("language"),
            test_only=filters.get("test_only"),
            extensions=filters.get("extensions"),
        )
    elif isinstance(filters, str):
        shorthand = filters.strip()
        if not shorthand:
            return None
        lowered = shorthand.lower()
        if lowered in {"test", "tests", "test_only"}:
            filters = SearchFilters(test_only=True)
        elif lowered in {"python", "javascript", "typescript", "go", "rust", "java"}:
            filters = SearchFilters(language=lowered)
        elif any(ch in shorthand for ch in "*?[]/\\") or "." in shorthand:
            filters = SearchFilters(file_pattern=shorthand)
        else:
            logger.warning(
                "Ignoring unsupported string filters=%r for code_search; expected SearchFilters or dict",
                filters,
            )
            return None
    elif not isinstance(filters, SearchFilters):
        logger.warning(
            "Ignoring unsupported filters type %s for code_search",
            type(filters).__name__,
        )
        return None

    def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
        if not isinstance(value, str):
            return value
        normalized_value = value.strip()
        return normalized_value or None

    normalized_file_pattern = _normalize_optional_text(filters.file_pattern)
    normalized_symbol = _normalize_optional_text(filters.symbol)
    normalized_language = _normalize_optional_text(filters.language)
    normalized_extensions = None
    if filters.extensions is not None:
        normalized_extensions = []
        for ext in filters.extensions:
            if isinstance(ext, str):
                normalized_ext = ext.strip()
                if normalized_ext:
                    normalized_extensions.append(normalized_ext)
            elif ext:
                normalized_extensions.append(ext)
        if not normalized_extensions:
            normalized_extensions = None

    if (
        normalized_file_pattern == filters.file_pattern
        and normalized_symbol == filters.symbol
        and normalized_language == filters.language
        and normalized_extensions == filters.extensions
    ):
        return filters

    return SearchFilters(
        file_pattern=normalized_file_pattern,
        symbol=normalized_symbol,
        language=normalized_language,
        test_only=filters.test_only,
        extensions=normalized_extensions,
    )


def _resolve_code_search_root(path: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve a search root consistently across semantic and literal modes."""

    search_root = path
    if not search_root or search_root == ".":
        try:
            from victor.config.settings import get_project_paths

            search_root = str(get_project_paths().project_root)
        except Exception:
            search_root = "."

    root_path = Path(search_root).resolve()
    if root_path.is_file():
        root_path = root_path.parent
        logger.debug("Path %r is a file, using parent: %s", search_root, root_path)
    elif not root_path.exists():
        parent_path = root_path.parent
        if parent_path.exists() and parent_path.is_dir():
            root_path = parent_path
            logger.debug("Path %r not found, using parent: %s", search_root, root_path)
        else:
            return (
                None,
                f"Search root '{search_root}' not found. Please provide a valid directory path.",
            )

    return root_path, None


def _glob_patterns_for_extensions(exts: Optional[List[str]]) -> List[str]:
    """Normalize extension filters into shell globs like ``*.py``."""

    source_exts = exts or sorted(DEFAULT_CODE_EXTENSIONS)
    patterns: List[str] = []
    seen: Set[str] = set()
    for ext in source_exts:
        if not ext:
            continue
        normalized_ext = ext if ext.startswith(".") else f".{ext}"
        pattern = f"*{normalized_ext}"
        if pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)
    return patterns


def _normalize_code_search_path(path_text: str) -> str:
    """Normalize path text for file-pattern matching."""

    normalized = path_text.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _matches_literal_file_pattern(file_path: str, file_pattern: str, *, search_root: str) -> bool:
    """Return whether a literal-search hit matches the caller's file pattern."""

    normalized_pattern = _normalize_code_search_path(file_pattern.strip())
    if not normalized_pattern:
        return True

    try:
        relative_path = Path(file_path).resolve().relative_to(Path(search_root).resolve())
        candidate_path = _normalize_code_search_path(str(relative_path))
    except Exception:
        candidate_path = _normalize_code_search_path(file_path)

    basename = os.path.basename(candidate_path)
    has_glob = any(token in normalized_pattern for token in "*?[]")

    if has_glob:
        if "/" in normalized_pattern:
            return fnmatch.fnmatch(candidate_path, normalized_pattern)
        return fnmatch.fnmatch(basename, normalized_pattern)

    if "/" in normalized_pattern:
        return candidate_path == normalized_pattern or candidate_path.endswith(
            f"/{normalized_pattern}"
        )

    return basename == normalized_pattern


def _is_test_file_path(file_path: str) -> bool:
    """Return whether a path looks like a test file."""

    normalized_path = _normalize_code_search_path(file_path).lower()
    return any(pattern in normalized_path for pattern in _TEST_FILE_PATH_PATTERNS)


def _file_pattern_has_glob(file_pattern: str) -> bool:
    """Return whether a file pattern uses glob syntax rather than exact matching."""

    normalized_pattern = _normalize_code_search_path(file_pattern.strip())
    return any(token in normalized_pattern for token in "*?[]")


def _file_pattern_uses_backend_exact_filter(file_pattern: str) -> bool:
    """Return whether a file pattern is safe to send as an exact backend file_path filter."""

    normalized_pattern = _normalize_code_search_path(file_pattern.strip())
    return (
        bool(normalized_pattern)
        and "/" in normalized_pattern
        and not _file_pattern_has_glob(normalized_pattern)
    )


def _filter_search_results_by_file_pattern(
    results: List[Any],
    file_pattern: str,
    *,
    search_root: Path,
) -> List[Any]:
    """Filter result objects by file pattern using the shared path matcher."""

    filtered_results: List[Any] = []
    for result in results:
        result_dict = _normalize_result_dict(result)
        file_path = result_dict.get("file_path") or result_dict.get("path")
        if not isinstance(file_path, str):
            continue
        if _matches_literal_file_pattern(file_path, file_pattern, search_root=str(search_root)):
            filtered_results.append(result)
    return filtered_results


def _filter_search_results_by_extensions(
    results: List[Any],
    exts: List[str],
) -> List[Any]:
    """Filter result objects by normalized file extension."""

    normalized_exts = _normalize_extensions(exts)
    filtered_results: List[Any] = []
    for result in results:
        result_dict = _normalize_result_dict(result)
        file_path = result_dict.get("file_path") or result_dict.get("path")
        if not isinstance(file_path, str):
            continue
        if Path(file_path).suffix in normalized_exts:
            filtered_results.append(result)
    return filtered_results


def _canonicalize_language_name(language: str) -> str:
    """Normalize a language name or alias for result filtering."""

    normalized = language.strip().lower()
    return _LANGUAGE_ALIASES.get(normalized, normalized)


def _result_matches_language_filter(result: Any, language: str) -> bool:
    """Return whether a search result matches the requested language."""

    canonical_language = _canonicalize_language_name(language)
    result_dict = _normalize_result_dict(result)
    metadata = result_dict.get("metadata")
    if isinstance(metadata, dict):
        metadata_language = metadata.get("language")
        if isinstance(metadata_language, str):
            if _canonicalize_language_name(metadata_language) == canonical_language:
                return True

    file_path = result_dict.get("file_path") or result_dict.get("path")
    if not isinstance(file_path, str):
        return False

    suffix = Path(file_path).suffix.lower()
    known_extensions = _LANGUAGE_EXTENSION_MAP.get(canonical_language)
    if known_extensions:
        return suffix in known_extensions

    return suffix == f".{canonical_language}"


def _filter_search_results_by_language(
    results: List[Any],
    language: str,
) -> List[Any]:
    """Filter result objects by normalized language."""

    return [result for result in results if _result_matches_language_filter(result, language)]


def _filter_search_results_by_test_only(
    results: List[Any],
    test_only: bool,
) -> List[Any]:
    """Filter result objects by test-file status."""

    filtered_results: List[Any] = []
    for result in results:
        result_dict = _normalize_result_dict(result)
        metadata = result_dict.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("is_test_file"), bool):
            is_test_file = metadata["is_test_file"]
        else:
            file_path = result_dict.get("file_path") or result_dict.get("path")
            if not isinstance(file_path, str):
                continue
            is_test_file = _is_test_file_path(file_path)
        if is_test_file == test_only:
            filtered_results.append(result)
    return filtered_results


def _result_matches_symbol_filter(result: Any, symbol: str) -> bool:
    """Return whether a search result matches the requested symbol filter."""

    target_symbol = symbol.strip()
    if not target_symbol:
        return True

    result_dict = _normalize_result_dict(result)
    metadata = result_dict.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    candidate_names = (
        result_dict.get("qualified_name"),
        result_dict.get("symbol_name"),
        result_dict.get("name"),
        metadata_dict.get("qualified_name"),
        metadata_dict.get("symbol_name"),
        metadata_dict.get("name"),
    )
    for candidate in candidate_names:
        if not isinstance(candidate, str):
            continue
        normalized_candidate = candidate.strip()
        if (
            normalized_candidate == target_symbol
            or normalized_candidate.endswith(f".{target_symbol}")
            or normalized_candidate.endswith(f":{target_symbol}")
        ):
            return True
        if ("." in target_symbol or ":" in target_symbol) and normalized_candidate.endswith(
            target_symbol
        ):
            return True

    symbol_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(target_symbol)}(?![A-Za-z0-9_])")
    for text in (result_dict.get("content"), result_dict.get("snippet")):
        if isinstance(text, str) and symbol_pattern.search(text):
            return True

    return False


def _filter_search_results_by_symbol(
    results: List[Any],
    symbol: str,
) -> List[Any]:
    """Filter result objects by symbol name."""

    return [result for result in results if _result_matches_symbol_filter(result, symbol)]


def _language_filter_extensions(language: str) -> Optional[List[str]]:
    """Return known file extensions for a language filter."""

    known_extensions = _LANGUAGE_EXTENSION_MAP.get(_canonicalize_language_name(language))
    if not known_extensions:
        return None
    return sorted(known_extensions)


def _build_literal_search_extensions(
    filters: Optional["SearchFilters"],
) -> Optional[List[str]]:
    """Build literal-search extension filters from explicit extensions or language hints."""

    if not filters:
        return None
    if filters.extensions:
        return list(filters.extensions)
    if filters.language:
        return _language_filter_extensions(filters.language)
    return None


def _needs_literal_postfilter_overfetch(filters: Optional["SearchFilters"]) -> bool:
    """Return whether literal results are narrowed by shared post-filters after retrieval."""

    return bool(filters and (filters.symbol or filters.language or filters.test_only is not None))


def _literal_search_fetch_limit(requested_limit: int, filters: Optional["SearchFilters"]) -> int:
    """Return the candidate count to fetch before applying literal post-filters."""

    if requested_limit <= 0:
        return requested_limit
    if _needs_literal_postfilter_overfetch(filters):
        return requested_limit * 2
    return requested_limit


def _apply_literal_result_filters(
    result: Dict[str, Any],
    filters: Optional["SearchFilters"],
    *,
    max_results: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply shared post-filters to literal-search results."""

    if not filters or not isinstance(result.get("results"), list):
        return dict(result)

    filtered_result = dict(result)
    filtered_hits: List[Any] = list(result["results"])
    if filters.extensions:
        filtered_hits = _filter_search_results_by_extensions(filtered_hits, filters.extensions)
    if filters.symbol:
        filtered_hits = _filter_search_results_by_symbol(filtered_hits, filters.symbol)
    if filters.language:
        filtered_hits = _filter_search_results_by_language(filtered_hits, filters.language)
    if filters.test_only is not None:
        filtered_hits = _filter_search_results_by_test_only(filtered_hits, filters.test_only)
    if max_results is not None:
        filtered_hits = filtered_hits[:max_results]
    filtered_result["results"] = filtered_hits
    filtered_result["count"] = len(filtered_hits)
    return filtered_result


def _build_literal_search_kwargs(
    *,
    allow_filename_autodetect: bool,
    filename_only: Optional[bool] = None,
    file_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Build stable kwargs for literal-search call sites without emitting empty filters."""

    kwargs: Dict[str, Any] = {"allow_filename_autodetect": allow_filename_autodetect}
    if filename_only is not None:
        kwargs["filename_only"] = filename_only
    if file_pattern is not None:
        kwargs["file_pattern"] = file_pattern
    return kwargs


_EXTRA_FILENAME_QUERY_EXTENSIONS = {
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".r",
    ".sql",
}
_FILENAME_QUERY_EXTENSIONS = tuple(
    sorted(set(DEFAULT_CODE_EXTENSIONS) | _EXTRA_FILENAME_QUERY_EXTENSIONS)
)
_EXTRA_LANGUAGE_EXTENSION_MAP: Dict[str, Set[str]] = {
    "shell": {".sh", ".bash", ".zsh", ".fish"},
    "markdown": {".md"},
    "yaml": {".yaml", ".yml"},
    "json": {".json"},
    "toml": {".toml"},
    "html": {".html"},
    "css": {".css", ".scss"},
    "csharp": {".cs", ".csx", ".csproj", ".sln", ".xaml"},
    "vb": {".vb", ".vbproj"},
    "fsharp": {".fs", ".fsx", ".fsproj"},
}
_LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "mts": "typescript",
    "cts": "typescript",
    "golang": "go",
    "rs": "rust",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hh": "cpp",
    "hpp": "cpp",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
    "md": "markdown",
    "yml": "yaml",
}
_TEST_FILE_PATH_PATTERNS = ("test_", "_test.", ".test.", "/test/", "/tests/")


def _build_language_extension_map() -> Dict[str, Set[str]]:
    """Build a canonical language-to-extension map from shared code patterns."""

    mapping: Dict[str, Set[str]] = {}
    for language, patterns in CODE_PATTERNS.items():
        extensions = {
            pattern[1:].lower()
            for pattern in patterns
            if pattern.startswith("*.")
            and "/" not in pattern
            and not any(token in pattern[2:] for token in "*?[]")
        }
        if extensions:
            mapping[language] = extensions
    for language, extensions in _EXTRA_LANGUAGE_EXTENSION_MAP.items():
        mapping.setdefault(language, set()).update(ext.lower() for ext in extensions)
    return mapping


_LANGUAGE_EXTENSION_MAP = _build_language_extension_map()


def _looks_like_filename_query(query: str) -> bool:
    """Return True when a query is more likely a filename than semantic text."""

    query = query.strip()
    query_lower = query.lower()
    return (
        any(query_lower.endswith(ext) for ext in _FILENAME_QUERY_EXTENSIONS)
        and " " not in query
        and len(query) < 100
    )


# Directories that indicate non-core code (lower importance)
NON_CORE_DIRS = {
    "test",
    "tests",
    "testing",
    "spec",
    "specs",
    "demo",
    "demos",
    "example",
    "examples",
    "sample",
    "samples",
    "client",
    "clients",
    "sdk",
    "benchmark",
    "benchmarks",
    "bench",
    "doc",
    "docs",
    "documentation",
    "script",
    "scripts",
    "tool",
    "tools",
    "util",
    "utils",
    "mock",
    "mocks",
    "fixture",
    "fixtures",
    "stub",
    "stubs",
}

# Directories that indicate core code (higher importance)
CORE_DIRS = {
    "src",
    "lib",
    "core",
    "pkg",
    "internal",
    "main",
    "engine",
    "engines",
    "service",
    "services",
    "storage",
    "index",
    "compute",
    "network",
    "api",
}

GRAPH_FOLLOW_UP_SYMBOL_TYPES = {"function", "method"}
ENTRYPOINT_SYMBOL_NAMES = {"main", "run", "start", "serve", "cli", "bootstrap"}


def _calculate_importance_score(file_path: str, symbol_type: Optional[str] = None) -> float:
    """Calculate importance score for a search result.

    Higher scores = more architecturally important.

    Scoring factors:
    - Core directories: +0.3
    - Non-core directories: -0.4
    - Class/struct definitions: +0.2
    - Test files: -0.3
    """
    score = 1.0
    path_lower = file_path.lower()
    parts = set(path_lower.replace("\\", "/").split("/"))

    # Check for core directories
    if parts & CORE_DIRS:
        score += 0.3

    # Check for non-core directories
    if parts & NON_CORE_DIRS:
        score -= 0.4

    # Explicit test file patterns
    if _is_test_file_path(path_lower):
        score -= 0.3

    # Symbol type bonus
    if symbol_type in ("class", "struct", "trait", "interface", "impl"):
        score += 0.2

    # Depth penalty - deeper nesting = less core
    depth = len([p for p in parts if p and p not in {".", ".."}])
    if depth > 5:
        score -= 0.1

    return max(0.1, score)  # Floor at 0.1


def _latest_mtime(root: Path) -> float:
    """Find latest modification time under root, respecting EXCLUDE_DIRS."""
    return latest_mtime(root)


async def _subscribe_to_file_watcher(
    root: Path,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> bool:
    """Subscribe to file watcher for automatic index invalidation.

    Args:
        root: Root path of codebase
        exec_ctx: Execution context
    """
    from victor_contracts.indexing_runtime import FileWatcherRegistry

    try:
        # Get or create file watcher for this path
        watcher_registry = FileWatcherRegistry.get_instance()
        file_watcher = await watcher_registry.get_watcher(root)

        # Subscribe to file changes (synchronous method)
        file_watcher.subscribe(lambda e: _schedule_file_change_refresh(e, root, exec_ctx))

        logger.info(f"[code_search] Subscribed to file watcher for {root}")
        return True
    except Exception as e:
        logger.error(f"[code_search] Failed to subscribe to file watcher: {e}")
        return False


def _schedule_file_change_refresh(
    event: "FileChangeEvent",
    root: Path,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> asyncio.Task[Any]:
    """Schedule watcher-driven cache refresh work with exception containment."""

    async def _runner() -> None:
        try:
            await _on_file_change(event, root, exec_ctx)
        except Exception:
            logger.exception("[code_search] File change refresh failed for %s", root)

    return asyncio.create_task(_runner())


async def _ensure_file_watcher_subscription(
    cache_entry: Dict[str, Any],
    root: Path,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> bool:
    """Subscribe a cached index to file watching exactly once per root.

    Concurrent cache-hit callers can reach the watcher subscription path before
    the per-root index lock is acquired. Keep the subscription single-flight so
    duplicate callbacks do not stack up on the same watcher.
    """

    if cache_entry.get("watcher_subscribed", False):
        return True

    pending_task = cache_entry.get("watcher_subscription_task")
    if isinstance(pending_task, asyncio.Task) and pending_task.done():
        try:
            subscribed = bool(pending_task.result())
        except asyncio.CancelledError:
            subscribed = False
        except Exception:
            subscribed = False
        if cache_entry.get("watcher_subscription_task") is pending_task:
            cache_entry.pop("watcher_subscription_task", None)
        cache_entry["watcher_subscribed"] = subscribed
        if subscribed:
            return True
        pending_task = None

    if not isinstance(pending_task, asyncio.Task):
        pending_task = asyncio.create_task(_subscribe_to_file_watcher(root, exec_ctx))
        cache_entry["watcher_subscription_task"] = pending_task

    try:
        subscribed = bool(await asyncio.shield(pending_task))
    except asyncio.CancelledError:
        if cache_entry.get("watcher_subscription_task") is pending_task and pending_task.done():
            cache_entry.pop("watcher_subscription_task", None)
        raise
    finally:
        if cache_entry.get("watcher_subscription_task") is pending_task and pending_task.done():
            cache_entry.pop("watcher_subscription_task", None)

    cache_entry["watcher_subscribed"] = subscribed
    return subscribed


async def _refresh_cached_index_incrementally(
    cache_entry: Dict[str, Any],
    root: Path,
    *,
    exec_ctx: Optional[Dict[str, Any]] = None,
    ensure_watcher: bool = False,
) -> bool:
    """Incrementally refresh a cached index entry in-place.

    Returns True when the cached index supports incremental refresh and was
    updated successfully. Returns False when the index cannot be refreshed
    incrementally, leaving the caller to mark or treat the cache as stale.
    """

    index = cache_entry.get("index")
    if not hasattr(index, "incremental_reindex"):
        logger.warning(
            "[code_search] Index doesn't support incremental updates, marking stale: %s",
            root,
        )
        cache_entry["stale"] = True
        return False

    await index.incremental_reindex()
    await _finalize_index_storage(index)
    cache_entry["latest_mtime"] = _latest_mtime(root)
    cache_entry["indexed_at"] = time.time()
    cache_entry["stale"] = False

    if ensure_watcher:
        await _ensure_file_watcher_subscription(cache_entry, root, exec_ctx)

    return True


async def _on_file_change(
    event: "FileChangeEvent",
    root: Path,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> None:
    """Handle file change event for codebase index auto-update.

    Invalidates cache and triggers incremental update if index exists.

    Args:
        event: File change event from FileWatcherService
        root: Root path of codebase
        exec_ctx: Execution context for cache access
    """
    from victor_contracts.indexing_runtime import FileChangeType
    from victor_contracts.indexing_runtime import IndexLockRegistry

    index_cache = _get_index_cache(exec_ctx)
    index_key = str(root)
    lock_registry = IndexLockRegistry.get_instance()
    # A watcher refresh writes the incremental update, so it needs the exclusive
    # (writer) lock — a shared lock would allow concurrent writers to corrupt it.
    path_lock = await lock_registry.acquire_lock(root)

    async with path_lock:
        cache_entry = index_cache.get(index_key)

        if not cache_entry:
            return  # No index exists, nothing to update

        if event.change_type in (
            FileChangeType.DELETED,
            FileChangeType.MODIFIED,
            FileChangeType.CREATED,
            FileChangeType.RENAMED,
        ):
            logger.info(
                "[code_search] File change detected (%s), triggering incremental update: %s",
                event.change_type.value,
                event.path,
            )

            try:
                refreshed = await _refresh_cached_index_incrementally(cache_entry, root)
                if refreshed:
                    logger.info(f"[code_search] Incremental update complete for {root}")
            except Exception as e:
                cache_entry["stale"] = True
                logger.error(f"[code_search] Incremental update failed: {e}")


def _normalize_extensions(exts: Optional[List[str]]) -> Set[str]:
    """Normalize extensions to ensure they have leading dots.

    Handles:
    - None -> returns DEFAULT_CODE_EXTENSIONS
    - String 'py' -> {'.py'}
    - List ['py', 'js'] -> {'.py', '.js'}
    - Already dotted ['.py'] -> {'.py'}
    """
    if exts is None:
        return DEFAULT_CODE_EXTENSIONS

    # Handle string input (convert to list)
    if isinstance(exts, str):
        exts = [exts]

    # Normalize each extension to have leading dot
    normalized = set()
    for ext in exts:
        ext = ext.strip()
        if not ext.startswith("."):
            ext = "." + ext
        normalized.add(ext)

    return normalized


def _gather_files(root: str, exts: Optional[List[str]], max_files: int) -> List[str]:
    """Gather files from directory tree."""
    ext_set: Set[str] = _normalize_extensions(exts)
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if os.path.splitext(fname)[1] in ext_set:
                files.append(os.path.join(dirpath, fname))
                if len(files) >= max_files:
                    return files
    return files


def _keyword_score(text: str, query: str) -> int:
    q = query.lower().split()
    t = text.lower()
    return sum(t.count(word) for word in q)


def _normalize_result_dict(result: Any) -> Dict[str, Any]:
    """Normalize search results from models or dict-like providers."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return dict(result)
    return dict(result)


_SEARCH_RESULT_METADATA_DROP_KEYS = {
    "vector",
    "embedding",
    "embeddings",
    "content",
    "file_path",
    "path",
    "score",
    "similarity",
    "combined_score",
    "semantic_score",
    "keyword_score",
}


def _sanitize_search_metadata(value: Any) -> Any:
    """Remove bulky or duplicate search metadata before tool return.

    Search hits only need compact grounding metadata for follow-up reads.
    Raw embeddings and duplicate payload fields add large token cost without
    helping the LLM act on the result.
    """
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _SEARCH_RESULT_METADATA_DROP_KEYS:
                continue
            sanitized[key] = _sanitize_search_metadata(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_search_metadata(item) for item in value]
    return value


async def _graph_search(
    query: str,
    root_path: Path,
    k: int,
    max_hops: int,
    edge_types: Optional[List[str]],
    filters: Optional[SearchFilters],
    settings: Any,
) -> Dict[str, Any]:
    """Execute graph-based multi-hop search.

    Args:
        query: Search query
        root_path: Root directory to search
        k: Maximum results
        max_hops: Maximum traversal depth
        edge_types: Edge types to traverse
        filters: Search filters
        settings: Settings object

    Returns:
        Dictionary with results, metadata
    """
    import time
    from victor_contracts.graph_rag_runtime import MultiHopRetriever, RetrievalConfig
    from victor.storage.graph import create_graph_store

    start_time = time.time()

    # Default edge types for graph traversal
    default_edge_types = ["CALLS", "REFERENCES", "IMPORTS", "INHERITS", "CONTAINS"]
    if edge_types is None:
        edge_types = default_edge_types

    # Create graph store
    try:
        graph_store = create_graph_store("sqlite", root_path)
        await graph_store.initialize()

        # Check if graph has data
        stats = await graph_store.stats()
        node_count = stats.get("node_count", stats.get("nodes", 0))

        if node_count == 0:
            logger.info("Graph store is empty for graph search")
            await graph_store.close()
            return {"results": [], "error": "graph_empty"}

    except Exception as e:
        logger.warning(f"Failed to initialize graph store: {e}")
        return {"results": [], "error": str(e)}

    # Create retrieval configuration
    config = RetrievalConfig(
        seed_count=k,
        max_hops=min(max_hops, 3),  # Cap at 3 hops
        top_k=k,
        edge_types=edge_types,
    )

    # Create retriever
    retriever = MultiHopRetriever(graph_store, config)

    # Execute retrieval
    try:
        result = await retriever.retrieve(query, config)

        # Convert graph nodes to search results format
        results = []
        for node in result.nodes:
            # Apply filters if provided
            if filters:
                if filters.language and node.lang != filters.language:
                    continue
                if filters.symbol and node.name != filters.symbol:
                    continue

            result_item = {
                "file_path": node.file,
                "path": node.file,
                "line_number": node.line,
                "end_line": node.end_line,
                "symbol_name": node.name,
                "symbol_type": node.type,
                "content": node.docstring or f"# {node.type}: {node.name}",
                "score": result.scores.get(node.node_id, 0.5),
                "similarity": result.scores.get(node.node_id, 0.5),
                "combined_score": result.scores.get(node.node_id, 0.5),
                "metadata": {
                    "node_id": node.node_id,
                    "search_mode": "graph",
                    "hop_distance": result.hop_distances.get(node.node_id, 0),
                    "signature": node.signature,
                    "language": node.lang,
                },
            }
            results.append(result_item)

        execution_time_ms = (time.time() - start_time) * 1000

        await graph_store.close()

        return {
            "results": results,
            "edge_types_used": edge_types,
            "nodes_traversed": len(result.nodes),
            "execution_time_ms": execution_time_ms,
        }

    except Exception as e:
        logger.warning(f"Graph retrieval failed: {e}")
        await graph_store.close()
        return {"results": [], "error": str(e)}


def _prepare_ranked_results(
    results: List[Any],
    search_mode: str,
    max_content_chars: int = 500,
) -> List[Dict[str, Any]]:
    """Normalize, truncate, and importance-rank search results."""
    ranked_results: List[Dict[str, Any]] = []
    for result in results:
        result_dict = _normalize_result_dict(result)
        result_dict.setdefault("search_mode", search_mode)
        metadata = result_dict.get("metadata")
        if isinstance(metadata, dict):
            result_dict["metadata"] = _sanitize_search_metadata(metadata)
            if not result_dict.get("qualified_name"):
                qualified_name = metadata.get("qualified_name")
                if isinstance(qualified_name, str) and qualified_name.strip():
                    result_dict["qualified_name"] = qualified_name.strip()

        content = result_dict.get("content", "")
        if isinstance(content, str) and len(content) > max_content_chars:
            result_dict["content"] = (
                content[:max_content_chars] + f"... [truncated, {len(content)} chars total]"
            )
            result_dict["content_truncated"] = True

        file_path = result_dict.get("file_path", result_dict.get("path", ""))
        symbol_type = result_dict.get("symbol_type")
        importance = _calculate_importance_score(file_path, symbol_type)
        result_dict["importance_score"] = round(importance, 2)

        raw_score = result_dict.get("score", result_dict.get("similarity"))
        if raw_score is None:
            raw_score = result_dict.get("combined_score", 0.5)
        try:
            numeric_score = float(raw_score)
        except (TypeError, ValueError):
            numeric_score = 0.5
        result_dict["combined_score"] = round(numeric_score * importance, 3)

        ranked_results.append(result_dict)

    ranked_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    return ranked_results


def _extract_graph_follow_up_symbol(result: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract a symbol candidate suitable for graph follow-up suggestions."""
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    symbol_type = (
        result.get("symbol_type")
        or result.get("type")
        or metadata_dict.get("symbol_type")
        or metadata_dict.get("type")
    )
    symbol_name = (
        result.get("qualified_name")
        or metadata_dict.get("qualified_name")
        or result.get("name")
        or result.get("symbol_name")
        or metadata_dict.get("name")
        or metadata_dict.get("symbol_name")
    )

    if not isinstance(symbol_type, str) or not isinstance(symbol_name, str):
        return None

    normalized_type = symbol_type.strip().lower()
    normalized_name = symbol_name.strip()
    if normalized_type not in GRAPH_FOLLOW_UP_SYMBOL_TYPES or not normalized_name:
        return None

    return {"name": normalized_name, "symbol_type": normalized_type}


def _build_graph_follow_up_suggestions(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build graph-tool follow-up suggestions from ranked search results.

    Optimized to avoid duplication:
    - Combines multiple modes into single suggestion using pipe separator
    - Truncates long node definitions to save tokens
    - Shows unique variations only (no duplicate node values)
    """
    for result in results:
        symbol = _extract_graph_follow_up_symbol(result)
        if symbol is None:
            continue

        symbol_name = symbol["name"]
        symbol_name_lower = symbol_name.lower()
        suggestions: List[Dict[str, Any]] = []

        # Truncate long definitions to save tokens (keep first 100 chars)
        if len(symbol_name) > 100:
            display_name = symbol_name[:97] + "..."
        else:
            display_name = symbol_name

        if symbol_name_lower in ENTRYPOINT_SYMBOL_NAMES:
            # Entry point gets separate trace suggestion
            suggestions.append(
                {
                    "tool": "graph",
                    "command": f'graph(mode="trace", node="{display_name}", depth=3)',
                    "arguments": {"mode": "trace", "node": symbol_name, "depth": 3},
                    "reason": f"Trace execution starting from {display_name}.",
                }
            )

        # Combine callers and callees into single suggestion with pipe separator
        # Instead of duplicating the entire node definition
        combined_modes = "callers|callees"
        depth = 2
        suggestions.append(
            {
                "tool": "graph",
                "command": f'graph(mode="{combined_modes}", node="{display_name}", depth={depth})',
                "arguments": {
                    "mode": combined_modes,
                    "node": symbol_name,
                    "depth": depth,
                },
                "reason": f"Explore call relationships for {display_name} (who calls it and what it calls).",
            }
        )

        # Add impact analysis suggestion for all symbols
        suggestions.append(
            {
                "tool": "graph",
                "command": f'graph(mode="impact", node="{display_name}")',
                "arguments": {
                    "mode": "impact",
                    "node": symbol_name,
                },
                "reason": f"Analyze data and control flow impact of changing {display_name}.",
            }
        )

        return suggestions

    return []


def _build_search_response(
    *,
    results: List[Dict[str, Any]],
    mode: str,
    rebuilt: bool,
    root_path: Path,
    exec_ctx: Optional[Dict[str, Any]],
    filters_applied: List[str],
    ranking_note: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    follow_up_suggestions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a consistent search tool response envelope."""
    cache_entry = _get_index_cache(exec_ctx).get(str(root_path), {})
    hint = (
        "Use read with offset/limit based on line_number/end_line for precise reads. "
        "Example: read(path, offset=line_number-1, limit=end_line-line_number+5)"
    )
    if follow_up_suggestions:
        hint += f" Graph follow-up: {follow_up_suggestions[0]['command']}"

    metadata = {
        "rebuilt": rebuilt,
        "root": str(root_path),
        "indexed_at": cache_entry.get("indexed_at"),
        "filters_applied": filters_applied if filters_applied else None,
        "chunking_strategy": "provider_default",
        "importance_weighted": True,
        "available_filters": [
            "file_path",
            "symbol_type",
            "visibility",
            "language",
            "is_test_file",
            "has_docstring",
        ],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if follow_up_suggestions:
        metadata["follow_up_suggestions"] = follow_up_suggestions

    preview_results: List[Dict[str, Any]] = []
    for result in results:
        content = result.get("content", "")
        snippet = result.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            if isinstance(content, str):
                snippet = " ".join(
                    line.strip() for line in content.splitlines()[:2] if line.strip()
                )
            else:
                snippet = ""

        preview_results.append(
            {
                "path": result.get("file_path", result.get("path", "unknown")),
                "line": result.get("line_number", result.get("line", "?")),
                "score": result.get("score", result.get("combined_score", 0)),
                "snippet": snippet,
            }
        )

    formatted_result = format_search_results(
        {
            "results": preview_results,
            "mode": mode,
        }
    )

    return {
        "success": True,
        "results": results,
        "count": len(results),
        "mode": mode,
        "hint": hint,
        "ranking_note": ranking_note,
        "metadata": metadata,
        "formatted_results": formatted_result.content,
        "contains_markup": formatted_result.contains_markup,
    }


async def _get_or_build_index(
    root: Path,
    settings: Any,
    force_reindex: bool = False,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, bool]:
    """Return cached CodebaseIndex or build/update one. Returns (index, rebuilt?).

    Uses intelligent caching:
    1. In-memory cache for same session (DI-aware)
    2. Persistent disk storage in {root}/.victor/embeddings/
    3. Incremental updates for changed files only (not full rebuild)
    4. Per-path locking to prevent concurrent indexing (NEW)

    Args:
        root: Root path for the codebase
        settings: Application settings
        force_reindex: Force full re-index
        exec_ctx: Execution context for DI-based cache access
    """
    # Clean up nested .victor directories on first access
    _cleanup_nested_victor_dirs(root)

    from victor_contracts.capability_runtime import (
        CodebaseIndexFactoryProtocol,
        get_capability_provider,
        is_capability_enhanced,
    )
    from victor_contracts.indexing_runtime import IndexLockRegistry

    embedding_config = _build_codebase_embedding_config(settings, root)
    index_manifest = build_codebase_index_manifest(embedding_config)

    # Get or create failure cache for index build failures before attempting
    # provider bootstrap so missing-provider failures can short-circuit.
    failure_cache = _get_index_build_failure_cache(exec_ctx)
    failure_key = _build_index_failure_key(root, index_manifest)
    provider_failure_key = _build_provider_availability_failure_key(root, settings)
    if not force_reindex:
        _raise_if_cached_index_build_failure(failure_cache, provider_failure_key)
        _raise_if_cached_index_build_failure(failure_cache, failure_key)

    _index_factory = get_capability_provider(CodebaseIndexFactoryProtocol)

    if _index_factory is None or not is_capability_enhanced(CodebaseIndexFactoryProtocol):
        if importlib.util.find_spec("victor_coding") is not None:
            error_msg = (
                "CodebaseIndex factory not registered. The victor-coding package "
                "is installed but failed to register its provider. Try:\n"
                "  1. Restart your Python session\n"
                "  2. Reinstall: pip install --force-reinstall victor-coding\n"
                "  3. Check for errors during package import\n\n"
                "For literal/keyword search only, use mode='literal' in code_search()."
            )
        else:
            error_msg = (
                "CodebaseIndex requires victor-coding package for semantic search. "
                "To enable semantic code search:\n"
                "  pip install victor-coding\n\n"
                "For literal/keyword search only, use mode='literal' in code_search()."
            )
        _cache_index_build_failure(failure_cache, provider_failure_key, error_msg)
        raise ImportError(error_msg)

    # Get cache using DI-aware accessor
    index_cache = _get_index_cache(exec_ctx)

    cache_entry = index_cache.get(str(root))
    cached_index = cache_entry["index"] if cache_entry else None
    last_mtime = cache_entry["latest_mtime"] if cache_entry else 0.0

    latest = _latest_mtime(root)

    # Check if we have a valid cached index
    if cached_index and not force_reindex:
        # Check if cache is marked as stale from file watcher
        if cache_entry.get("stale", False):
            logger.info(f"[code_search] Cache marked stale for {root}, will rebuild")
            # Fall through to rebuild below
        elif not _cache_entry_matches_manifest(cache_entry, index_manifest):
            logger.info("[code_search] Cache manifest mismatch for %s, rebuilding", root)
            # Fall through to rebuild below
        elif latest <= last_mtime:
            # No files changed, use cache directly
            # Subscribe to file watcher for auto-invalidation (only once per index)
            await _ensure_file_watcher_subscription(cache_entry, root, exec_ctx)

            return cached_index, False
        else:
            logger.info("[code_search] Cache changed for %s, refreshing under lock", root)

    # Acquire an EXCLUSIVE lock for the build/refresh — the body below writes to
    # the index, so a shared lock would let two processes rebuild concurrently.
    # Use a bounded timeout: if a background refresh (possibly in another process)
    # holds the lock, serve the stale cache instead of blocking for the full
    # timeout. This is what stops a long graph-watch refresh from starving
    # code_search into a hard timeout.
    #
    # We enter/exit the lock manually (rather than `async with`) so a TimeoutError
    # raised on *acquisition* can fall back to the stale cache, while the lock is
    # still released before the post-lock cleanup runs.
    lock_registry = IndexLockRegistry.get_instance()
    path_lock = await lock_registry.acquire_lock(root, timeout_seconds=10.0)
    _lock_entered = False
    try:
        await path_lock.__aenter__()
        _lock_entered = True
        # Double-check cache inside lock (another task may have built it while we waited)
        cache_entry = index_cache.get(str(root))
        cached_index = cache_entry["index"] if cache_entry else None
        last_mtime = cache_entry["latest_mtime"] if cache_entry else 0.0

        latest = _latest_mtime(root)

        # Check if we have a valid cached index (double-check)
        if cached_index and not force_reindex:
            if cache_entry.get("stale", False):
                logger.info(
                    "[code_search] Cache marked stale for %s (inside lock), rebuilding",
                    root,
                )
            elif not _cache_entry_matches_manifest(cache_entry, index_manifest):
                logger.info(
                    "[code_search] Cache manifest mismatch for %s (inside lock), rebuilding",
                    root,
                )
            elif latest <= last_mtime:
                # Another task built it while we waited for lock
                logger.info(f"[code_search] Cache hit for {root} (inside lock)")

                # Subscribe to file watcher if not already subscribed
                await _ensure_file_watcher_subscription(cache_entry, root, exec_ctx)

                return cached_index, False
            else:
                refreshed = await _refresh_cached_index_incrementally(
                    cache_entry,
                    root,
                    exec_ctx=exec_ctx,
                    ensure_watcher=True,
                )
                if refreshed:
                    logger.info(
                        f"[code_search] Incremental refresh complete for {root} (inside lock)"
                    )
                    return cached_index, False

        # Build index with exclusive access to this path
        logger.info(f"[code_search] Building index for {root} (exclusive lock acquired)")

        graph_store_name = getattr(settings, "codebase_graph_store", "sqlite")
        graph_path = getattr(settings, "codebase_graph_path", None)
        graph_writer_mode = _resolve_graph_writer_mode(settings)

        # Create new index - it will load from disk if available
        index = _index_factory.create(
            root_path=str(root),
            use_embeddings=True,
            embedding_config=embedding_config,
            graph_store_name=graph_store_name,
            graph_path=Path(graph_path) if graph_path else None,
            graph_writer_mode=str(graph_writer_mode),
        )

        logger.info(f"[code_search] Index creation complete for {root}")

        # Only do full index if forced or no persistent data exists
        persist_path = Path(embedding_config["persist_directory"])
        has_persistent_data = has_persisted_codebase_index_data(persist_path)
        has_compatible_manifest = has_compatible_codebase_index_manifest(
            persist_path, index_manifest
        )
        if has_persistent_data and not has_compatible_manifest:
            logger.info(
                "Persistent code_search index manifest mismatch for %s; rebuilding (%s)",
                persist_path,
                CODEBASE_INDEX_MANIFEST_NAME,
            )

        if force_reindex or not has_persistent_data or not has_compatible_manifest:
            # First time or forced - full index
            # Check if we have stale persistent data we can use immediately
            has_usable_stale_data = has_persistent_data and not has_compatible_manifest

            if has_usable_stale_data and not force_reindex:
                # Mark current index as usable with stale data
                if hasattr(index, "_is_indexed"):
                    index._is_indexed = True
                logger.info(
                    "[code_search] Returning stale index immediately (manifest mismatch), rebuilding in background"
                )

                # Write manifest FIRST to prevent rebuild loop on next access
                try:
                    write_codebase_index_manifest(persist_path, index_manifest)
                    logger.info(
                        "[code_search] Wrote manifest for stale index, background rebuild in progress"
                    )
                except OSError as exc:
                    logger.warning(
                        "[code_search] Failed to write manifest before stale return: %s",
                        exc,
                    )

                # Spawn background rebuild task (fire and forget)
                async def _rebuildInBackground():
                    try:
                        logger.info("[code_search] Background rebuild started for %s", root)
                        await index.index_codebase()
                        await _finalize_index_storage(index)
                        # Write manifest after successful rebuild
                        try:
                            write_codebase_index_manifest(persist_path, index_manifest)
                            logger.info(
                                "[code_search] Wrote manifest after background rebuild for %s",
                                root,
                            )
                        except OSError as exc:
                            logger.warning(
                                "[code_search] Failed to write manifest after rebuild: %s",
                                exc,
                            )
                        # Update cache with fresh index after rebuild
                        current_cache = index_cache.get(str(root))
                        if current_cache:
                            current_cache["stale"] = False
                            current_cache["latest_mtime"] = _latest_mtime(root)
                            current_cache["index_manifest"] = index_manifest
                        logger.info("[code_search] Background rebuild complete for %s", root)
                    except Exception as exc:
                        error_str = str(exc).lower()
                        # Detect embedding-specific errors
                        if "end_line" in error_str and "int64" in error_str:
                            logger.error(
                                "[code_search] Background rebuild failed due to embedding schema error: %s. "
                                "This may be caused by None values in metadata. "
                                "Consider using mode='literal' for text search instead.",
                                exc,
                            )
                        elif "embed" in error_str or "lance" in error_str:
                            logger.warning(
                                "[code_search] Background rebuild failed due to embedding error: %s. "
                                "Graph data may still be available for structural queries.",
                                exc,
                            )
                        else:
                            logger.warning(
                                "[code_search] Background rebuild failed for %s: %s",
                                root,
                                exc,
                            )

                # Spawn background task (single-flight per root to avoid concurrent
                # rebuilds thrashing the same embeddings store).
                _spawn_index_rebuild_once(str(root), _rebuildInBackground)

                # Cache stale index for immediate use
                index_cache[str(root)] = {
                    "index": index,
                    "latest_mtime": latest,
                    "indexed_at": time.time(),
                    "index_manifest": index_manifest,
                    "watcher_subscribed": False,
                    "stale": True,  # Mark as stale since rebuild is in progress
                }

                # Return immediately with stale data
                return index, False  # Not rebuilt from caller's perspective

            # No stale data available, must block for initial build
            try:
                await index.index_codebase()
            except Exception as exc:
                error_str = str(exc).lower()
                # Detect embedding-specific errors and provide helpful guidance
                if "end_line" in error_str and "int64" in error_str:
                    logger.error(
                        "[code_search] Index build failed due to embedding schema error: %s. "
                        "This indicates a data type issue in the embedding metadata. "
                        "Graph data may still be available - try using graph(mode='stats') or "
                        "code_search(mode='literal', ...) for text-based search.",
                        exc,
                    )
                    raise RuntimeError(
                        f"Embedding index build failed due to schema error: {exc}. "
                        "Try using code_search(mode='literal', ...) for text search or "
                        "graph(mode='stats', reindex=True) to rebuild the graph."
                    ) from exc
                elif "embed" in error_str or "lance" in error_str:
                    logger.warning(
                        "[code_search] Index build failed due to embedding error: %s. "
                        "Graph data may still be available for structural queries.",
                        exc,
                    )
                    raise RuntimeError(
                        f"Embedding index build failed: {exc}. "
                        "Graph data may still be available - try graph(mode='stats') or "
                        "code_search(mode='literal', ...) for text search."
                    ) from exc
                else:
                    raise
            await _finalize_index_storage(index)
            rebuilt = True
            cache_stale = False
        else:
            # Persistent embeddings exist on disk (LanceDB tables).
            # Mark as indexed so semantic_search() works directly against
            # the persisted data without triggering a full rebuild.
            if hasattr(index, "_is_indexed"):
                index._is_indexed = True
            logger.info("Using persistent embeddings from %s (skip full rebuild)", persist_path)
            # Validate integrity — corrupt LanceDB data will fail silently
            probe_outcome = _coerce_integrity_probe_outcome(await _probe_index_integrity(index))
            rebuilt = probe_outcome.rebuilt
            cache_stale = probe_outcome.stale

        try:
            write_codebase_index_manifest(persist_path, index_manifest)
        except OSError as exc:
            logger.warning(
                "Failed to write code_search index manifest to %s: %s",
                persist_path,
                exc,
            )

        index_cache[str(root)] = {
            "index": index,
            "latest_mtime": latest,
            "indexed_at": time.time(),
            "index_manifest": index_manifest,
            "watcher_subscribed": False,  # Will be subscribed on next access
            "stale": cache_stale,
        }

        # Mark lock as used
        lock_registry.mark_lock_used(root)

        logger.info(f"[code_search] Index build complete for {root} (releasing lock)")

    except TimeoutError:
        if _lock_entered:
            raise
        # Lock busy (a background refresh holds it) — serve the stale cache
        # rather than block, otherwise fall back to a read-only on-disk load.
        if cached_index:
            logger.info(
                "[code_search] Lock busy (background refresh active), serving cached index for %s",
                root,
            )
            return cached_index, False
        logger.info("[code_search] Lock busy, attempting read-only index load for %s", root)
        try:
            index = _index_factory.create(
                root_path=str(root),
                use_embeddings=True,
                embedding_config=embedding_config,
                graph_store_name=getattr(settings, "codebase_graph_store", "sqlite"),
                graph_path=(
                    Path(settings.codebase_graph_path)
                    if getattr(settings, "codebase_graph_path", None)
                    else None
                ),
                graph_writer_mode="off",  # Force read-only
            )
            return index, False
        except Exception as e:
            logger.warning("[code_search] Read-only index load failed: %s", e)
            # If the indexing provider is simply absent (victor-coding not installed), surface
            # that as-is — NOT as a timeout. Masking it as TimeoutError made the caller log a
            # bogus "build timed out after 600s", classify it as semantic_index_timeout instead
            # of provider_unavailable, and skip caching the failure — so parallel/subsequent
            # code_search calls kept re-racing the lock. Re-raising the provider error lets the
            # caller classify + cache it and fall back to literal search in <1s.
            if _is_missing_semantic_provider_error(str(e)):
                raise
            raise TimeoutError(f"Index lock for {root} busy and no usable cache found.") from e
    finally:
        if _lock_entered:
            await path_lock.__aexit__(None, None, None)

    # Clear failure cache on successful build
    try:
        failure_cache = _get_index_build_failure_cache(exec_ctx)

        cleared_any = False
        for key in (failure_key, provider_failure_key):
            if failure_cache is not None and _failure_cache_get(failure_cache, key):
                _failure_cache_delete(failure_cache, key)
                cleared_any = True
        if cleared_any:
            logger.info("[code_search] Cleared index build failure cache after successful build")
    except Exception as cache_err:
        logger.debug(f"[code_search] Failed to clear index build failure cache: {cache_err}")

    return index, rebuilt


async def _literal_search(
    query: str,
    path: str = ".",
    k: int = 5,
    exts: Optional[List[str]] = None,
    *,
    filename_only: bool = False,
    allow_filename_autodetect: bool = True,
    file_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Literal/keyword search using ripgrep (rg) or grep subprocess.

    Uses native search tools instead of Python file-by-file scanning for
    speed and correctness — no file count limits, handles binary detection,
    and searches the full directory tree efficiently.

    Args:
        query: Search terms
        path: Directory to search
        k: Max results
        exts: File extensions ([".py", ".js"])
        filename_only: Restrict filename-like queries to filename matching only.
        allow_filename_autodetect: Whether filename-like queries may use filename search.
        file_pattern: Optional path pattern filter for narrowing content hits.
    """
    import shutil
    import subprocess

    try:
        # Resolve empty or '.' to the framework's project root
        search_path = path
        if not search_path or search_path == ".":
            try:
                from victor.config.settings import get_project_paths

                search_path = str(get_project_paths().project_root)
            except Exception:
                search_path = "."  # Fallback to CWD

        # Normalize dotted notation: "Class.method" → search for "def method"
        # in files containing "class Class". This handles the common pattern
        # where models search for "SQLCompiler.get_order_by" but the source has
        # "class SQLCompiler:" and "def get_order_by(self):" on separate lines.
        search_query = query
        dotted_class = None
        should_preserve_filename_query = filename_only or (
            allow_filename_autodetect and _looks_like_filename_query(query)
        )
        if (
            not should_preserve_filename_query
            and "." in query
            and not query.startswith(".")
            and " " not in query
        ):
            parts = query.rsplit(".", 1)
            if len(parts) == 2 and parts[0][0].isupper():
                dotted_class, method = parts
                search_query = f"def {method}"
                logger.info(
                    "Dotted notation detected: %s → searching for '%s' in files with 'class %s'",
                    query,
                    search_query,
                    dotted_class,
                )

        logger.info(
            f"Literal search: query={query!r}, path={path!r}, "
            f"resolved={search_path!r}, exts={exts}, file_pattern={file_pattern!r}"
        )

        # Filename detection: explicit filename mode should always use filename matching,
        # while other literal searches only do so for filename-like queries.
        should_use_filename_search = filename_only or (
            allow_filename_autodetect and _looks_like_filename_query(search_query)
        )

        if should_use_filename_search:
            import platform

            system = platform.system()
            normalized_filename_query = search_query.replace("\\", "/").lstrip("./")
            query_has_path_components = "/" in normalized_filename_query
            query_has_glob = any(token in normalized_filename_query for token in "*?[]")
            query_suffix = normalized_filename_query.lower()
            filename_filter_pattern = file_pattern
            if filename_filter_pattern is None and _looks_like_filename_query(search_query):
                filename_filter_pattern = normalized_filename_query
            logger.info(
                f"Filename query detected: using {'dir' if system == 'Windows' else 'find'} "
                f"for {search_query!r} on {system}"
            )

            if system == "Windows":
                # Windows: use dir /s /b for recursive file search
                if query_has_path_components:
                    find_cmd = ["cmd", "/c", "dir", "/s", "/b"]
                else:
                    find_cmd = ["cmd", "/c", "dir", "/s", "/b", f"*{search_query}*"]
                find_cwd = search_path
            else:
                # Linux / macOS: use find
                find_cmd = ["find", search_path, "-type", "f"]
                if not query_has_path_components:
                    find_cmd.extend(["-name", f"*{search_query}*"])
                find_cwd = None

            try:
                # Use asyncio subprocess to avoid blocking event loop
                proc = await asyncio.create_subprocess_exec(
                    *find_cmd,
                    cwd=find_cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                filename_timeout = _filename_search_timeout()
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=filename_timeout
                )
                find_result = type(
                    "obj",
                    (object,),
                    {"stdout": stdout.decode("utf-8", errors="ignore")},
                )
                found_files = [f.strip() for f in find_result.stdout.splitlines() if f.strip()]
                if query_has_path_components and not query_has_glob:
                    found_files = [
                        fpath
                        for fpath in found_files
                        if query_suffix in fpath.replace("\\", "/").lower()
                    ]
                if filename_filter_pattern:
                    found_files = [
                        fpath
                        for fpath in found_files
                        if _matches_literal_file_pattern(
                            fpath,
                            filename_filter_pattern,
                            search_root=search_path,
                        )
                    ]
                if found_files:
                    results = []
                    for fpath in found_files[:k]:
                        normalized_path = fpath.replace("\\", "/").lower()
                        results.append(
                            {
                                "path": fpath,
                                "score": (10 if normalized_path.endswith(query_suffix) else 5),
                                "snippet": f"[File found: {fpath}]",
                            }
                        )
                    logger.info(
                        f"Filename search: found {len(found_files)} files matching "
                        f"{search_query!r} in {search_path}"
                    )
                    return {
                        "success": True,
                        "results": results,
                        "count": len(results),
                        "mode": "filename",
                    }
            except asyncio.TimeoutError:
                logger.debug("Filename search timed out after %.0fs", filename_timeout)
                if filename_only:
                    return {
                        "success": False,
                        "error": (
                            f"Filename search timed out after {filename_timeout:.0f}s "
                            "(raise VICTOR_TIMEOUT_FILENAME_SEARCH for very large trees)"
                        ),
                        "results": [],
                        "count": 0,
                        "mode": "filename",
                    }
            except (subprocess.TimeoutExpired, Exception) as e:
                logger.debug(f"Filename search failed, falling back to content: {e}")
                if filename_only:
                    return {
                        "success": False,
                        "error": str(e),
                        "results": [],
                        "count": 0,
                        "mode": "filename",
                    }
            if filename_only:
                return {
                    "success": True,
                    "results": [],
                    "count": 0,
                    "mode": "filename",
                }
            # Fall through to content search if find returns nothing

        # Build command: prefer ripgrep, fall back to grep
        use_rg = shutil.which("rg") is not None
        if use_rg:
            cmd = [
                "rg",
                "--no-heading",
                "--line-number",
                "--color=never",
                "--max-count=5",  # max matches per file
                "--max-columns=200",
            ]
            if exts:
                for pattern in _glob_patterns_for_extensions(exts):
                    cmd.extend(["--glob", pattern])
            else:
                # Default: code files only
                cmd.extend(
                    [
                        "--type=py",
                        "--type=js",
                        "--type=ts",
                        "--type=go",
                        "--type=java",
                        "--type=c",
                        "--type=cpp",
                        "--type=rust",
                        "--type=yaml",
                    ]
                )
            cmd.extend(["--", search_query, search_path])
        else:
            cmd = ["grep", "-rn"]
            for pattern in _glob_patterns_for_extensions(exts):
                cmd.append(f"--include={pattern}")
            cmd.extend(["--", search_query, search_path])

        # Use asyncio subprocess to avoid blocking event loop
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        result = type("obj", (object,), {"stdout": stdout.decode("utf-8", errors="ignore")})

        # Parse results: group by file, take top k files
        file_matches: Dict[str, List[str]] = {}
        for line in result.stdout.splitlines():
            # Format: path:line_number:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                fpath = parts[0]
                if fpath not in file_matches:
                    file_matches[fpath] = []
                file_matches[fpath].append(line)

        # For dotted notation (Class.method), filter to files containing the class
        if dotted_class and file_matches:
            filtered = {}
            for fpath, matches in file_matches.items():
                try:
                    content = Path(fpath).read_text(errors="ignore")
                    if f"class {dotted_class}" in content:
                        filtered[fpath] = matches
                except Exception:
                    pass
            if filtered:
                file_matches = filtered
                logger.info(
                    "Dotted notation filter: %d files contain 'class %s'",
                    len(filtered),
                    dotted_class,
                )

        if file_pattern and file_matches:
            file_matches = {
                fpath: matches
                for fpath, matches in file_matches.items()
                if _matches_literal_file_pattern(fpath, file_pattern, search_root=search_path)
            }

        # Sort by number of matches (most matches = most relevant)
        ranked = sorted(file_matches.items(), key=lambda x: len(x[1]), reverse=True)
        top = ranked[:k]

        results = []
        for fpath, matches in top:
            snippet = "\n".join(matches[:5])  # first 5 matching lines
            results.append(
                {
                    "path": fpath,
                    "score": len(matches),
                    "snippet": snippet,
                }
            )

        logger.info(
            f"Literal search: found {len(file_matches)} files matching "
            f"{query!r} in {search_path} ({'rg' if use_rg else 'grep'})"
        )
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "mode": "literal",
        }
    except asyncio.TimeoutError:
        return {"success": False, "error": "Search timed out after 30s"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Search timed out after 30s"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@tool(
    category="search",
    priority=Priority.CRITICAL,  # Always available for code exploration
    access_mode=AccessMode.READONLY,  # Only reads files for search
    danger_level=DangerLevel.SAFE,  # No side effects
    # Registry-driven metadata for tool selection and loop detection
    signature_params=[
        "query",
        "path",
        "mode",
    ],  # Different queries/paths = exploration not loop
    stages=[
        "initial",
        "planning",
        "reading",
        "analysis",
    ],  # Relevant for exploration stages
    task_types=["search", "analysis"],  # Classification-aware selection
    execution_category="read_only",  # Safe for parallel execution
    keywords=[
        "search",
        "semantic",
        "embedding",
        "concept",
        "pattern",
        "similar",
        "bug",
        "bugs",
        "regression",
        "crash",
        "failure",
        "find",
        "find file",
        "locate file",
        "grep",
        "literal",
        "code search",
        "filename",
        "graph",
        "graph search",
        "multi-hop",
        "dependencies",
        "relationships",
        "traverse",
    ],
    mandatory_keywords=[
        "search code",
        "find in code",
        "search for",
        "code search",
        # Analysis-related from MANDATORY_TOOL_KEYWORDS
        "analyze",
        "analyze codebase",
        "codebase analysis",
        "understand",
        "explore codebase",
        "architecture",
        "search",
    ],  # Force inclusion
    aliases=["search"],  # Backward compatibility alias
    timeout=300.0,  # Increased to 5 minutes for large codebases (was 60s)
)
async def code_search(
    query: str,
    path: str = ".",
    k: int = 10,
    mode: str = "semantic",
    reindex: bool = False,
    filters: Optional[SearchFilters] = None,
    max_hops: int = 2,
    graph_edge_types: Optional[List[str]] = None,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find code by CONCEPT, TEXT, FILENAME, or GRAPH when you DON'T know exact location.

    Use this tool for exploration when you need to discover where relevant code
    lives. Returns file snippets ranked by relevance.

    Modes:
    - semantic: Vector search using embeddings (default)
    - text: Literal text search
    - filename: Search by filename pattern
    - graph: Multi-hop graph traversal for code relationships (experimental)

    Graph mode parameters:
    - max_hops: Maximum traversal depth (default: 2, range: 1-3)
    - graph_edge_types: Edge types to traverse (default: CALLS, REFERENCES, IMPORTS, INHERITS)

    Args:
        query: Search query (code concept, text, or filename pattern)
        path: Root directory to search (default: ".")
        k: Maximum number of results to return (default: 10)
        mode: Search mode - 'semantic', 'text', 'filename', or 'graph' (default: 'semantic')
        reindex: Force rebuild of search index (default: False)
        filters: SearchFilters object with file_pattern, symbol, language, test_only, extensions
        max_hops: Maximum hops for graph traversal (default: 2)
        graph_edge_types: Specific edge types to traverse (default: CALLS, REFERENCES, IMPORTS, INHERITS)

    Returns:
        Dictionary with search results including matches, scores, file paths

    Example:
        # Simple semantic search
        code_search(query="dataclass validation", path=".")

        # Search with filters
        filters = SearchFilters(
            language="python",
            test_only=True,
            extensions=[".py"]
        )
        code_search(query="pytest fixtures", path=".", filters=filters)

        # Filename search (auto-detected from query)
        code_search(query="*.py", path=".")

    DIFFERS FROM:
    - symbol(): Gets FULL CODE of a known symbol. Use when you know file + name.
    - refs(): Finds USAGE locations of a known symbol. Use for "where is X called".
    - graph(): Shows RELATIONSHIPS between symbols. Use for dependencies/impact.

    Legacy modes:
    - "literal": Keyword matching (like grep). Best for exact text/identifiers.
    - "bugs": Similar bug search with graph context when supported by the provider.
    - "localize": File-level issue localization using semantic seeds plus graph expansion.
    - "impact": Change-impact / blast-radius analysis using graph expansion when available.
    """
    query = query.strip()
    query_from_file_pattern = False
    filters = _normalize_search_filters(filters)
    if not query and filters and filters.file_pattern:
        query = filters.file_pattern
        query_from_file_pattern = True
    if not query:
        return {
            "success": False,
            "error": "Search query cannot be empty.",
        }

    requested_mode = mode
    literal_escalation_metadata: Dict[str, Any] = {}
    mode_fallback_to_semantic = False
    allow_filename_autodetect = requested_mode not in {"literal", "text"}
    literal_file_pattern = filters.file_pattern if filters and filters.file_pattern else None

    # Auto-detect filename search mode
    if mode == "semantic":
        if query_from_file_pattern and filters and filters.file_pattern and not filters.symbol:
            # If only file_pattern is provided, treat as filename search.
            mode = "filename"
        elif _looks_like_filename_query(query) and not (filters and filters.symbol):
            mode = "filename"

    root_path, root_error = _resolve_code_search_root(path)
    if root_error:
        return {
            "success": False,
            "error": root_error,
        }
    assert root_path is not None
    resolved_search_root = str(root_path)

    filename_query = query
    if mode == "filename" and filters and filters.file_pattern:
        filename_query = filters.file_pattern

    # Route to literal / filename search for non-semantic modes
    if mode in ("literal", "text", "filename"):
        literal_fetch_limit = _literal_search_fetch_limit(k, filters)
        exts = _build_literal_search_extensions(filters)
        result = await _literal_search(
            filename_query,
            resolved_search_root,
            literal_fetch_limit,
            exts,
            **_build_literal_search_kwargs(
                filename_only=(mode == "filename"),
                allow_filename_autodetect=allow_filename_autodetect,
                file_pattern=literal_file_pattern,
            ),
        )
        result = _apply_literal_result_filters(result, filters, max_results=k)
        result["mode"] = mode
        if result.get("count", 0) > 0:
            return result
        if mode == "filename":
            return result
        # Auto-escalate: literal returned 0 results, try semantic if available
        settings = _exec_ctx.get("settings") if _exec_ctx else None
        disable_embeddings = (_exec_ctx or {}).get("disable_embeddings", False)
        if settings and not disable_embeddings:
            logger.info(
                "Literal search returned 0 results for '%s', auto-escalating to semantic",
                query,
            )
            literal_escalation_metadata = {
                "requested_mode": requested_mode,
                "fallback_mode": "semantic",
                "fallback_reason": "literal_no_results",
            }
            mode_fallback_to_semantic = True
            mode = "semantic"  # Fall through to semantic path below
        else:
            return result
    exts: Optional[List[str]] = _build_literal_search_extensions(filters)
    try:
        # Check if path is a subdirectory to provide smart hints
        # This helps avoid duplicate embeddings while supporting cross-repo analysis
        try:
            from victor.config.settings import get_project_paths

            project_root = Path(get_project_paths().project_root).resolve()

            # Check if root_path is within the same git repository as project root
            # If they're in different repos, no warning needed (cross-repo analysis)
            def get_git_repo_root(path: Path) -> Optional[Path]:
                """Find the git repository root for a given path."""
                current = path
                while current != current.parent:  # Stop at filesystem root
                    git_dir = current / ".git"
                    if git_dir.exists():
                        return current
                    current = current.parent
                return None

            root_repo = get_git_repo_root(root_path)
            project_repo = get_git_repo_root(project_root)

            # Only warn if they're in the same repo and root_path is a subdirectory
            if root_repo and project_repo and root_repo == project_repo:
                try:
                    root_path.relative_to(project_root)
                    is_subdirectory = True
                except ValueError:
                    is_subdirectory = False

                if is_subdirectory and root_path != project_root:
                    # Path is a subdirectory within the same repo
                    rel_path = root_path.relative_to(project_root)
                    logger.info(
                        f"[code_search] Searching in subdirectory '{rel_path}' (same repo as project root). "
                        f"Consider using path='.' for full repository coverage if broader context is needed."
                    )
                    # Proceed with subdirectory search - user may have specific intent
            elif root_repo and project_repo and root_repo != project_repo:
                # Different repos - cross-repo analysis, no warning needed
                logger.debug(
                    f"[code_search] Cross-repo analysis: searching in separate repo at {root_repo}"
                )
        except Exception:
            # If we can't determine repo boundaries, proceed with provided path
            pass

        settings = _exec_ctx.get("settings") if _exec_ctx else None
        if settings is None:
            return {
                "success": False,
                "error": "Settings not available in tool context.",
            }

        # Check if embeddings are disabled for this agent (workflow-level service mode)
        disable_embeddings = _exec_ctx.get("disable_embeddings", False) if _exec_ctx else False
        fallback_search_path = str(root_path)
        if disable_embeddings:
            logger.info("Embeddings disabled for this agent, falling back to literal search")
            literal_fetch_limit = _literal_search_fetch_limit(k, filters)
            exts = _build_literal_search_extensions(filters)
            result = await _literal_search(
                query,
                fallback_search_path,
                literal_fetch_limit,
                exts,
                **_build_literal_search_kwargs(
                    allow_filename_autodetect=allow_filename_autodetect,
                    file_pattern=literal_file_pattern,
                ),
            )
            result = _apply_literal_result_filters(result, filters, max_results=k)
            return _decorate_literal_fallback_result(
                result,
                fallback="semantic_disabled",
                metadata=_early_semantic_fallback_metadata(
                    mode,
                    literal_escalation_metadata=literal_escalation_metadata,
                ),
                mode_override=_literal_fallback_mode_override(
                    requested_mode,
                    literal_escalation_metadata=literal_escalation_metadata,
                ),
            )

        # Build metadata filter from optional parameters
        filter_metadata: Optional[Dict[str, Any]] = None
        filters_applied = []
        manual_file_pattern_filter: Optional[str] = None
        manual_extension_filter: Optional[List[str]] = None
        manual_symbol_filter: Optional[str] = None
        manual_language_filter: Optional[str] = None
        manual_test_only_filter: Optional[bool] = None
        if mode_fallback_to_semantic:
            filters_applied.append("mode_fallback=semantic")

        if filters and any(
            [
                filters.file_pattern,
                filters.symbol,
                filters.language,
                filters.test_only is not None,
                filters.extensions,
            ]
        ):
            filter_metadata = {}
            if filters.file_pattern:
                filters_applied.append(f"file={filters.file_pattern}")
                if _file_pattern_uses_backend_exact_filter(filters.file_pattern):
                    filter_metadata["file_path"] = filters.file_pattern
                else:
                    manual_file_pattern_filter = filters.file_pattern
            if filters.symbol:
                filter_metadata["symbol_name"] = filters.symbol
                manual_symbol_filter = filters.symbol
                filters_applied.append(f"symbol={filters.symbol}")
            if filters.language:
                manual_language_filter = filters.language
                filters_applied.append(f"lang={filters.language}")
            if filters.test_only is not None:
                manual_test_only_filter = filters.test_only
                filters_applied.append(f"test={filters.test_only}")
            if filters.extensions:
                manual_extension_filter = filters.extensions
                filters_applied.append(f"ext={','.join(filters.extensions)}")

        try:
            # Calculate dynamic timeout based on codebase size
            build_timeout = _calculate_index_build_timeout(root_path)
            index, rebuilt = await asyncio.wait_for(
                _get_or_build_index(root_path, settings, force_reindex=reindex, exec_ctx=_exec_ctx),
                timeout=build_timeout,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            error_msg = str(exc) if str(exc) else repr(exc)
            error_type = type(exc).__name__
            is_cached_failure = _is_recent_cached_failure_error(error_msg)
            is_missing_provider = _is_missing_semantic_provider_error(error_msg)
            is_timeout = isinstance(exc, asyncio.TimeoutError)

            if is_timeout:
                logger.warning(
                    "[code_search] Semantic index build timed out after %.1fs (TimeoutError: %s), "
                    "falling back to literal search. The embedding process is still running in the background. "
                    "Consider increasing VICTOR_INDEX_BUILD_TIMEOUT environment variable for very large codebases.",
                    build_timeout,
                    error_msg,
                )
            elif is_cached_failure:
                logger.info(
                    "Semantic index build skipped due to cached recent failure (%s), "
                    "falling back to literal search",
                    exc,
                )
            elif is_missing_provider:
                logger.info(
                    "Semantic index provider unavailable (%s), falling back to literal search",
                    exc,
                )
            else:
                logger.warning(
                    "[code_search] Semantic index build failed (type=%s, error=%s), falling back to literal search. "
                    "Root cause: %s",
                    error_type,
                    error_msg[:200] if len(error_msg) > 200 else error_msg,
                    _extract_exception_cause(exc),
                )

            # Cache the failure for 1 hour to prevent repeated attempts
            if not is_cached_failure:
                try:
                    failure_key = _build_index_failure_key(
                        root_path,
                        build_codebase_index_manifest(
                            _build_codebase_embedding_config(settings, root_path)
                        ),
                    )

                    # Get failure cache (same logic as in _get_or_build_index)
                    failure_cache = _get_index_build_failure_cache(_exec_ctx)
                    _cache_index_build_failure(failure_cache, failure_key, error_msg)
                    if is_missing_provider:
                        provider_failure_key = _build_provider_availability_failure_key(
                            root_path, settings
                        )
                        _cache_index_build_failure(
                            failure_cache,
                            provider_failure_key,
                            error_msg,
                        )
                    logger.info("[code_search] Cached index build failure for 1 hour")
                except Exception as cache_err:
                    logger.debug(f"[code_search] Failed to cache index build failure: {cache_err}")

            literal_fetch_limit = _literal_search_fetch_limit(k, filters)
            exts = _build_literal_search_extensions(filters)
            result = await _literal_search(
                query,
                fallback_search_path,
                literal_fetch_limit,
                exts,
                **_build_literal_search_kwargs(
                    allow_filename_autodetect=allow_filename_autodetect,
                    file_pattern=literal_file_pattern,
                ),
            )
            result = _apply_literal_result_filters(result, filters, max_results=k)
            return _decorate_literal_fallback_result(
                result,
                fallback=_classify_semantic_fallback(exc, scope="semantic_index"),
                filters_applied=filters_applied,
                metadata=_early_semantic_fallback_metadata(
                    mode,
                    literal_escalation_metadata=literal_escalation_metadata,
                ),
                mode_override=_literal_fallback_mode_override(
                    requested_mode,
                    literal_escalation_metadata=literal_escalation_metadata,
                ),
            )

        if _is_cached_index_stale(root_path, _exec_ctx):
            logger.warning(
                "Semantic index for %s remains stale after integrity recovery; falling back to literal search",
                root_path,
            )
            literal_fetch_limit = _literal_search_fetch_limit(k, filters)
            exts = _build_literal_search_extensions(filters)
            result = await _literal_search(
                query,
                fallback_search_path,
                literal_fetch_limit,
                exts,
                **_build_literal_search_kwargs(
                    allow_filename_autodetect=allow_filename_autodetect,
                    file_pattern=literal_file_pattern,
                ),
            )
            result = _apply_literal_result_filters(result, filters, max_results=k)
            return _decorate_literal_fallback_result(
                result,
                fallback="semantic_index_stale",
                filters_applied=filters_applied,
                metadata=_early_semantic_fallback_metadata(
                    mode,
                    literal_escalation_metadata=literal_escalation_metadata,
                ),
                mode_override=_literal_fallback_mode_override(
                    requested_mode,
                    literal_escalation_metadata=literal_escalation_metadata,
                ),
            )

        backend_metadata = _collect_code_search_backend_metadata(index, settings)

        if mode == "bugs":
            ignored_filters = []
            if filters:
                ignored_filters = [
                    name
                    for name, value in (
                        ("file", filters.file_pattern),
                        ("symbol", filters.symbol),
                        ("test", filters.test_only),
                    )
                    if value is not None
                ]
            try:
                bug_results = await index.find_similar_bugs(
                    bug_description=query,
                    language=filters.language if filters else None,
                    top_k=k,
                    include_graph_context=True,
                    context_limit=min(max(1, k), 3),
                )
                ranked_results = _prepare_ranked_results(
                    bug_results,
                    search_mode="bug_similarity",
                )
                ranked_results, enrichment_metadata = enrich_code_search_results(
                    ranked_results,
                    root_path=root_path,
                )
                extra_metadata = {
                    **backend_metadata,
                    "provider_capability": "find_similar_bugs",
                }
                extra_metadata.update(enrichment_metadata)
                if ignored_filters:
                    extra_metadata["ignored_filters"] = ignored_filters
                follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)
                return _build_search_response(
                    results=ranked_results,
                    mode="bugs",
                    rebuilt=rebuilt,
                    root_path=root_path,
                    exec_ctx=_exec_ctx,
                    filters_applied=filters_applied,
                    ranking_note="Results ranked by combined_score (bug_similarity × importance). Graph context included when available.",
                    extra_metadata=extra_metadata,
                    follow_up_suggestions=follow_up_suggestions,
                )
            except NotImplementedError as exc:
                logger.info(
                    "Bug similarity mode is unsupported by %s; falling back to semantic search",
                    type(index).__name__,
                )
                filters_applied.append("mode_fallback=semantic")
                fallback_metadata = {
                    "requested_mode": "bugs",
                    "fallback_mode": "semantic",
                    "fallback_reason": str(exc),
                }
            else:
                fallback_metadata = {}
        elif mode == "localize":
            ignored_filters = []
            if filters:
                ignored_filters = [
                    name
                    for name, value in (
                        ("file", filters.file_pattern),
                        ("symbol", filters.symbol),
                        ("test", filters.test_only),
                    )
                    if value is not None
                ]
            try:
                localize_issue = getattr(index, "localize_issue", None)
                if localize_issue is None:
                    raise NotImplementedError("localize_issue unsupported")

                localization_results = await localize_issue(
                    issue_description=query,
                    language=filters.language if filters else None,
                    top_k=k,
                    include_graph_context=True,
                    context_limit=min(max(1, k), 3),
                )
                ranked_results = _prepare_ranked_results(
                    localization_results,
                    search_mode="issue_localization",
                )
                ranked_results, enrichment_metadata = enrich_code_search_results(
                    ranked_results,
                    root_path=root_path,
                )
                extra_metadata = {
                    **backend_metadata,
                    "provider_capability": "localize_issue",
                }
                extra_metadata.update(enrichment_metadata)
                if ignored_filters:
                    extra_metadata["ignored_filters"] = ignored_filters
                follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)
                return _build_search_response(
                    results=ranked_results,
                    mode="localize",
                    rebuilt=rebuilt,
                    root_path=root_path,
                    exec_ctx=_exec_ctx,
                    filters_applied=filters_applied,
                    ranking_note="Results ranked by combined_score (issue_localization × importance). Graph-expanded file candidates included when available.",
                    extra_metadata=extra_metadata,
                    follow_up_suggestions=follow_up_suggestions,
                )
            except NotImplementedError as exc:
                logger.info(
                    "Issue localization mode is unsupported by %s; falling back to semantic search",
                    type(index).__name__,
                )
                filters_applied.append("mode_fallback=semantic")
                fallback_metadata = {
                    "requested_mode": "localize",
                    "fallback_mode": "semantic",
                    "fallback_reason": str(exc),
                }
        elif mode == "impact":
            ignored_filters = []
            if filters:
                ignored_filters = [
                    name
                    for name, value in (
                        ("file", filters.file_pattern),
                        ("symbol", filters.symbol),
                        ("test", filters.test_only),
                    )
                    if value is not None
                ]
            try:
                analyze_change_impact = getattr(index, "analyze_change_impact", None)
                if analyze_change_impact is None:
                    raise NotImplementedError("analyze_change_impact unsupported")

                impact_results = await analyze_change_impact(
                    change_description=query,
                    language=filters.language if filters else None,
                    top_k=k,
                    include_graph_context=True,
                    context_limit=min(max(1, k), 3),
                )
                ranked_results = _prepare_ranked_results(
                    impact_results,
                    search_mode="change_impact",
                )
                ranked_results, enrichment_metadata = enrich_code_search_results(
                    ranked_results,
                    root_path=root_path,
                )
                extra_metadata = {
                    **backend_metadata,
                    "provider_capability": "analyze_change_impact",
                }
                extra_metadata.update(enrichment_metadata)
                if ignored_filters:
                    extra_metadata["ignored_filters"] = ignored_filters
                follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)
                return _build_search_response(
                    results=ranked_results,
                    mode="impact",
                    rebuilt=rebuilt,
                    root_path=root_path,
                    exec_ctx=_exec_ctx,
                    filters_applied=filters_applied,
                    ranking_note="Results ranked by combined_score (change_impact × importance). Graph-expanded blast-radius candidates included when available.",
                    extra_metadata=extra_metadata,
                    follow_up_suggestions=follow_up_suggestions,
                )
            except NotImplementedError as exc:
                logger.info(
                    "Impact mode is unsupported by %s; falling back to semantic search",
                    type(index).__name__,
                )
                filters_applied.append("mode_fallback=semantic")
                fallback_metadata = {
                    "requested_mode": "impact",
                    "fallback_mode": "semantic",
                    "fallback_reason": str(exc),
                }
        elif mode == "graph":
            # Graph-based multi-hop search using Graph RAG pipeline
            from victor_contracts.feature_flag_runtime import (
                get_feature_flag_manager,
                FeatureFlag,
            )

            flag_manager = get_feature_flag_manager()
            graph_enabled = flag_manager.is_enabled(
                FeatureFlag.USE_GRAPH_RAG
            ) and flag_manager.is_enabled(FeatureFlag.USE_MULTI_HOP_RETRIEVAL)

            if not graph_enabled:
                logger.info("Graph search features not enabled, falling back to semantic search")
                filters_applied.append("mode_fallback=semantic")
                fallback_metadata = {
                    "requested_mode": "graph",
                    "fallback_mode": "semantic",
                    "fallback_reason": "graph_features_disabled",
                }
            else:
                try:
                    graph_results = await _graph_search(
                        query=query,
                        root_path=root_path,
                        k=k,
                        max_hops=max_hops,
                        edge_types=graph_edge_types,
                        filters=filters,
                        settings=settings,
                    )

                    if graph_results.get("results"):
                        ranked_results = _prepare_ranked_results(
                            graph_results["results"],
                            search_mode="graph",
                        )
                        ranked_results, enrichment_metadata = enrich_code_search_results(
                            ranked_results,
                            root_path=root_path,
                        )

                        extra_metadata = {
                            **backend_metadata,
                            "graph_search": {
                                "max_hops": max_hops,
                                "edge_types": graph_edge_types
                                or graph_results.get("edge_types_used", []),
                                "nodes_traversed": graph_results.get("nodes_traversed", 0),
                                "execution_time_ms": graph_results.get("execution_time_ms", 0),
                            },
                        }
                        extra_metadata.update(enrichment_metadata)

                        follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)
                        return _build_search_response(
                            results=ranked_results,
                            mode="graph",
                            rebuilt=rebuilt,
                            root_path=root_path,
                            exec_ctx=_exec_ctx,
                            filters_applied=filters_applied,
                            ranking_note=f"Results ranked by graph traversal (max {max_hops} hops). Includes related code through CALLS, REFERENCES, and other relationships.",
                            extra_metadata=extra_metadata,
                            follow_up_suggestions=follow_up_suggestions,
                        )
                    else:
                        # No graph results, fall back to semantic
                        logger.info(
                            "Graph search returned no results, falling back to semantic search"
                        )
                        filters_applied.append("mode_fallback=semantic")
                        fallback_metadata = {
                            "requested_mode": "graph",
                            "fallback_mode": "semantic",
                            "fallback_reason": "graph_no_results",
                        }

                except Exception as exc:
                    logger.warning("Graph search failed (%s), falling back to semantic search", exc)
                    filters_applied.append("mode_fallback=semantic")
                    fallback_metadata = {
                        "requested_mode": "graph",
                        "fallback_mode": "semantic",
                        "fallback_reason": f"graph_error: {exc}",
                    }
        else:
            fallback_metadata = {
                **dict(backend_metadata),
                **literal_escalation_metadata,
            }

        # Get semantic search configuration from settings
        # Default threshold lowered from 0.5 to 0.25 for better recall on technical queries
        similarity_threshold = getattr(settings, "semantic_similarity_threshold", 0.25)
        expand_query = getattr(settings, "semantic_query_expansion_enabled", True)
        enable_hybrid = getattr(settings, "enable_hybrid_search", False)

        # Strip filter fields not in the index schema to avoid LanceDB errors.
        # The index has: file_path, symbol_name, symbol_type, line_number, end_line.
        # Fields like "language", "is_test_file" may not exist in all index backends.
        _INDEX_FILTER_FIELDS = {
            "file_path",
            "symbol_name",
            "symbol_type",
            "line_number",
            "end_line",
        }
        safe_filter = None
        if filter_metadata:
            safe_filter = {k: v for k, v in filter_metadata.items() if k in _INDEX_FILTER_FIELDS}
            dropped = set(filter_metadata) - set(safe_filter)
            if dropped:
                logger.debug("Dropped unsupported filter fields: %s", dropped)
            if not safe_filter:
                safe_filter = None

        # Perform semantic search with timeout and literal fallback
        try:
            semantic_max_results = k * 2 if enable_hybrid else k
            if (
                manual_file_pattern_filter
                or manual_extension_filter
                or manual_symbol_filter
                or manual_language_filter
                or manual_test_only_filter is not None
            ):
                semantic_max_results = max(semantic_max_results, k * 4)
            results = await asyncio.wait_for(
                index.semantic_search(
                    query=query,
                    max_results=semantic_max_results,
                    filter_metadata=safe_filter,
                    similarity_threshold=similarity_threshold,
                    expand_query=expand_query,
                ),
                timeout=15.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Semantic search failed (%s), falling back to literal search", exc)
            literal_fetch_limit = _literal_search_fetch_limit(k, filters)
            result = await _literal_search(
                query,
                fallback_search_path,
                literal_fetch_limit,
                exts,
                **_build_literal_search_kwargs(
                    allow_filename_autodetect=allow_filename_autodetect,
                    file_pattern=literal_file_pattern,
                ),
            )
            result = _apply_literal_result_filters(result, filters, max_results=k)
            return _decorate_literal_fallback_result(
                result,
                fallback=_classify_semantic_fallback(exc, scope="semantic_search"),
                filters_applied=filters_applied,
                metadata={**backend_metadata, **fallback_metadata},
                mode_override=_literal_fallback_mode_override(
                    requested_mode,
                    literal_escalation_metadata=literal_escalation_metadata,
                ),
            )

        if manual_file_pattern_filter:
            results = _filter_search_results_by_file_pattern(
                results,
                manual_file_pattern_filter,
                search_root=root_path,
            )
        if manual_extension_filter:
            results = _filter_search_results_by_extensions(results, manual_extension_filter)
        if manual_symbol_filter:
            results = _filter_search_results_by_symbol(results, manual_symbol_filter)
        if manual_language_filter:
            results = _filter_search_results_by_language(results, manual_language_filter)
        if manual_test_only_filter is not None:
            results = _filter_search_results_by_test_only(results, manual_test_only_filter)

        # Record outcome for RL threshold learning if enabled
        if _get_search_setting(settings, "enable_semantic_threshold_rl_learning", False):
            try:
                from victor_contracts.rl_runtime import get_rl_coordinator
                from victor_contracts.rl_runtime import RLOutcome

                coordinator = get_rl_coordinator()

                # Get embedding model from settings
                embedding_model = _resolve_codebase_embedding_model(settings)

                # Get task type from execution context (default to "search")
                task_type = _exec_ctx.get("task_type", "search") if _exec_ctx else "search"

                # Create outcome with semantic search metadata
                outcome = RLOutcome(
                    provider=embedding_model,  # Using embedding model as "provider"
                    model="code_search",  # Tool name as "model"
                    task_type=task_type,
                    success=(len(results) > 0),
                    quality_score=0.5,  # Neutral score, would need user feedback for better estimate
                    metadata={
                        "embedding_model": embedding_model,
                        "tool_name": "code_search",
                        "query": query,
                        "results_count": len(results),
                        "threshold_used": similarity_threshold,
                        "false_negatives": (len(results) == 0),
                        "false_positives": False,  # Would need user feedback
                    },
                    vertical="coding",
                )

                # Record outcome
                coordinator.record_outcome("semantic_threshold", outcome, "coding")

                # Check if we have a learned threshold recommendation
                recommendation = coordinator.get_recommendation(
                    "semantic_threshold",
                    embedding_model,  # provider param
                    "code_search",  # model param (tool name)
                    task_type,
                )

                if recommendation and recommendation.value is not None:
                    logger.debug(
                        f"RL: Learned threshold {recommendation.value:.2f} "
                        f"(current: {similarity_threshold:.2f}, "
                        f"confidence={recommendation.confidence:.2f}) for "
                        f"{embedding_model}:{task_type}:code_search"
                    )

            except Exception as e:
                logger.debug(f"Failed to record threshold learning outcome: {e}")

        response_mode = "semantic"

        # Optionally combine with keyword search using hybrid RRF
        if enable_hybrid:
            try:
                from victor_contracts.search_runtime import create_hybrid_search_engine

                # Get keyword search results
                keyword_fetch_limit = _literal_search_fetch_limit(k * 2, filters)
                keyword_results = await _literal_search(
                    query,
                    str(root_path),
                    keyword_fetch_limit,
                    exts=exts,
                    **_build_literal_search_kwargs(
                        allow_filename_autodetect=allow_filename_autodetect,
                        file_pattern=literal_file_pattern,
                    ),
                )
                keyword_results = _apply_literal_result_filters(
                    keyword_results, filters, max_results=k * 2
                )

                if keyword_results.get("success"):
                    # Convert semantic results to dict format for hybrid engine
                    semantic_dicts = [
                        {
                            "file_path": r.get("file_path", ""),
                            "content": r.get("content", ""),
                            "score": r.get("score", 0.5),
                            "line_number": r.get("line_number", 0),
                            "metadata": r.get("metadata", {}),
                        }
                        for r in results
                    ]

                    # Create hybrid search engine with configured weights
                    semantic_weight = getattr(settings, "hybrid_search_semantic_weight", 0.6)
                    keyword_weight = getattr(settings, "hybrid_search_keyword_weight", 0.4)
                    engine = create_hybrid_search_engine(semantic_weight, keyword_weight)

                    # Combine results using RRF
                    hybrid_results = engine.combine_results(
                        semantic_dicts,
                        keyword_results.get("results", []),
                        max_results=k,
                    )

                    # Convert back to dict format
                    results = [
                        {
                            "file_path": hr.file_path,
                            "content": hr.content,
                            "score": hr.combined_score,
                            "semantic_score": hr.semantic_score,
                            "keyword_score": hr.keyword_score,
                            "line_number": hr.line_number,
                            "metadata": hr.metadata,
                            "search_mode": "hybrid",
                        }
                        for hr in hybrid_results
                    ]
                    response_mode = "hybrid"

                    logger.info(
                        f"Hybrid search combined semantic + keyword → {len(results)} results"
                    )

            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to semantic: {e}")
                # Fall back to semantic-only results (already have them)

        ranked_results = _prepare_ranked_results(results, search_mode="semantic")
        ranked_results, enrichment_metadata = enrich_code_search_results(
            ranked_results,
            root_path=root_path,
        )
        ranked_results, utility_metadata = rerank_code_search_results(
            ranked_results,
            query=query,
        )
        fallback_metadata.update(enrichment_metadata)
        fallback_metadata.update(backend_metadata)
        fallback_metadata["retrieval_utility"] = utility_metadata
        follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)

        return _build_search_response(
            results=ranked_results,
            mode=response_mode,
            rebuilt=rebuilt,
            root_path=root_path,
            exec_ctx=_exec_ctx,
            filters_applied=filters_applied,
            ranking_note="Results ranked by utility-adjusted score (semantic/hybrid relevance × importance, then bounded file-diversity and duplicate-snippet adjustments).",
            extra_metadata=fallback_metadata,
            follow_up_suggestions=follow_up_suggestions,
        )
    except ImportError as exc:
        return {
            "success": False,
            "error": f"Semantic search dependencies missing: {exc}. Install lancedb/chromadb + sentence-transformers.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
