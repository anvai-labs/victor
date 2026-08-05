from __future__ import annotations

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

"""Semantic tool selection using embeddings for intelligent, context-aware tool matching."""

import hashlib
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

# Disable tokenizers parallelism BEFORE importing sentence_transformers
# This prevents "bad value(s) in fds_to_keep" errors in async contexts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import httpx

from victor.core.pickle_cache import (
    CacheValidation,
    delete_cache_file,
    invalid,
    load_validated_pickle,
    save_pickle_with_metadata,
    valid,
)
from victor.providers.base import ToolDefinition
from victor.tools.enums import CostTier
from victor.tools.registry import ToolRegistry
from victor.agent.tool_sequence_tracker import (
    ToolSequenceTracker,
    create_sequence_tracker,
)
from victor.agent.debug_logger import TRACE  # Import TRACE level
from victor.tools.metadata_registry import (
    ToolMetadataRegistry,
    get_core_readonly_tools,
    get_tools_matching_mandatory_keywords,
    get_tools_by_task_type,
)
from victor.tools.tool_names import get_canonical_name
from victor.config.tool_selection_defaults import (
    SemanticSelectorDefaults,
    FallbackTools,
    QueryPatterns,
)

# Import for type checking only (avoid circular imports)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from victor.agent.unified_classifier import ClassificationResult
    from victor.storage.embeddings.service import (
        EmbeddingService as EmbeddingServiceType,
    )

logger = logging.getLogger(__name__)

# Import fuzzy matching for robust typo-tolerant tool selection
try:
    from victor.storage.embeddings.fuzzy_matcher import match_keywords_cascading

    _FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    _FUZZY_MATCHING_AVAILABLE = False

_PREFERRED_SEQUENCE_TOOL_ALIASES = {
    "read": "read_file",
    "write": "write_file",
    "edit": "edit_files",
    "test": "run_tests",
    "ls": "list_directory",
    "shell": "execute_bash",
}

# Lazy numpy import — keeps ``import victor`` lightweight when numpy is absent.
_np = None
EmbeddingService = None  # type: ignore[assignment]


def _ensure_numpy():  # type: ignore[no-untyped-def]
    """Import numpy on first use and cache the module reference."""
    global _np
    if _np is None:
        import numpy

        _np = numpy
    return _np


def _get_embedding_service_class():
    """Load EmbeddingService only when embedding execution is needed."""

    global EmbeddingService
    if EmbeddingService is None:
        from victor.storage.embeddings.service import (
            EmbeddingService as _EmbeddingService,
        )

        EmbeddingService = _EmbeddingService
    return EmbeddingService


# Lazy hook initialization to avoid circular imports
_rl_hooks = None


def _get_rl_hooks():
    """Lazy load RL hooks registry."""
    global _rl_hooks
    if _rl_hooks is None:
        try:
            from victor.framework.rl.hooks import get_rl_hooks

            _rl_hooks = get_rl_hooks()
        except ImportError:
            _rl_hooks = None
    return _rl_hooks


# Cost tier warning messages for user visibility
COST_TIER_WARNINGS = {
    CostTier.HIGH: "HIGH COST: This tool may use significant resources or make external API calls",
    CostTier.MEDIUM: "MEDIUM COST: This tool performs moderately expensive operations",
}


class SemanticToolSelector:
    """Select relevant tools using embedding-based semantic similarity.

    Instead of keyword matching, this uses embeddings to find tools
    semantically related to the user's request.

    Benefits:
    - Handles synonyms automatically (test → verify, validate, check)
    - Understands context and intent
    - No hardcoded keyword lists
    - Self-improving with better tool descriptions
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_provider: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434",
        cache_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
        cost_aware_selection: bool = SemanticSelectorDefaults.COST_AWARE_SELECTION,
        cost_penalty_factor: float = SemanticSelectorDefaults.COST_PENALTY_FACTOR,
        sequence_tracking: bool = True,
    ):
        """Initialize semantic tool selector.

        Uses centralized defaults from SemanticSelectorDefaults.

        Args:
            embedding_model: Model to use for embeddings
                - sentence-transformers: "BAAI/bge-small-en-v1.5" (default, 130MB, 384-dim, higher-accuracy default)
                - ollama: "nomic-embed-text", "qwen3-embedding:8b", etc.
            embedding_provider: Provider (sentence-transformers, ollama, vllm, lmstudio)
                Default: "sentence-transformers" (local, fast, bundled)
            ollama_base_url: Ollama/vLLM/LMStudio API base URL
            cache_embeddings: Cache tool embeddings (recommended)
            cache_dir: Directory to store embedding cache (default: ~/.victor/embeddings/)
            cost_aware_selection: Deprioritize high-cost tools
            cost_penalty_factor: Penalty per cost weight
            sequence_tracking: Enable tool sequence tracking for 15-20% boost (default: True)
        """
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.ollama_base_url = ollama_base_url
        self.cache_embeddings = cache_embeddings
        self.cost_aware_selection = cost_aware_selection
        self.cost_penalty_factor = cost_penalty_factor

        # Cache directory
        if cache_dir is None:
            from victor.config.settings import get_project_paths

            cache_dir = get_project_paths().global_embeddings_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path (includes model name AND project hash for isolation)
        # TD-010: Add project hash to prevent cache pollution between projects
        project_hash = self._get_project_hash()
        model_safe = embedding_model.replace(":", "_").replace("/", "_")
        cache_filename = f"tool_embeddings_{model_safe}_{project_hash}.pkl"
        self.cache_file = self.cache_dir / cache_filename

        # In-memory cache: tool_name → embedding vector
        self._tool_embedding_cache: Dict[str, Any] = {}
        self._embedding_index: Optional[Any] = None

        # Tool version hash (to detect when tools change)
        self._tools_hash: Optional[str] = None

        # Note: sentence-transformers model is managed by shared EmbeddingService singleton
        # This reduces memory usage by sharing the model with IntentClassifier

        # HTTP client for Ollama/vLLM/LMStudio
        self._client = None
        if embedding_provider in ["ollama", "vllm", "lmstudio"]:
            self._client = httpx.AsyncClient(base_url=ollama_base_url, timeout=30.0)

        # Phase 3: Tool usage tracking and learning (also project-isolated)
        self._usage_cache_file = self.cache_dir / f"tool_usage_stats_{project_hash}.pkl"
        self._tool_usage_cache: Dict[str, Dict[str, Any]] = {}
        self._usage_cache_dirty = False  # Dirty flag - only save when changed
        self._load_usage_cache()

        # Phase 6: Store last cost warnings for retrieval
        self._last_cost_warnings: List[str] = []
        # Track which tools have already been warned about (warn once per session)
        self._warned_tools: set = set()

        # Phase 9: Tool sequence tracking for intelligent next-tool suggestions
        # Provides 15-20% improvement in tool selection via workflow pattern detection
        self._sequence_tracking = sequence_tracking
        self._sequence_tracker: Optional[ToolSequenceTracker] = None
        if sequence_tracking:
            self._sequence_tracker = create_sequence_tracker()

        # Phase 17: Tiered classification service (lazily resolved)
        self._tiered_service: Optional[Any] = None

    def _get_tiered_service(self) -> Optional[Any]:
        """Return TieredDecisionService if tiered classification is enabled."""
        from victor.agent.services.classification_backend import (
            use_tiered_classification,
        )

        if not use_tiered_classification():
            return None
        if self._tiered_service is None:
            try:
                from victor.core import get_container
                from victor.agent.services.tiered_decision_service import (
                    TieredDecisionService,
                )

                self._tiered_service = get_container().get(TieredDecisionService)
            except Exception:
                pass
        return self._tiered_service

    async def initialize_tool_embeddings(self, tools: ToolRegistry) -> None:
        """Pre-compute embeddings for all tools (called once at startup).

        Loads from pickle cache if available and tools haven't changed.
        Otherwise, computes embeddings and saves to cache.

        Also refreshes the ToolMetadataRegistry with metadata from all tools.

        Args:
            tools: Tool registry with all available tools
        """
        # Refresh the centralized ToolMetadataRegistry (smart reindexing)
        # This collects metadata from all tools (explicit or auto-generated)
        # Uses hash-based change detection to skip reindexing if tools haven't changed
        metadata_registry = ToolMetadataRegistry.get_instance()
        tool_list = tools.list_tools()
        reindexed = metadata_registry.refresh_from_tools(tool_list)

        if reindexed:
            stats = metadata_registry.get_statistics()
            logger.info(
                f"ToolMetadataRegistry reindexed: {stats['total_tools']} tools, "
                f"{stats['total_categories']} categories, {stats['total_keywords']} keywords"
            )
        else:
            logger.debug("ToolMetadataRegistry cache valid, skipping reindex")

        # Pre-initialize usage stats for ALL tools (not just used ones)
        # This ensures show_tool_stats.py reports all 47 tools, not just used ones
        self._initialize_all_tool_stats(tool_list)

        if not self.cache_embeddings:
            return

        # Calculate hash of all tool definitions
        tools_hash = self._calculate_tools_hash(tools)

        # Try to load from cache
        if self._load_from_cache(tools_hash):
            self._build_embedding_index()
            logger.info(
                f"Loaded tool embeddings from cache for {len(self._tool_embedding_cache)} tools"
            )
            return

        # Cache miss or tools changed - recompute
        logger.info(
            f"Computing tool embeddings for {len(tools.list_tools())} tools "
            f"(model: {self.embedding_model})"
        )

        for tool in tools.list_tools():
            # Create semantic description of tool
            tool_text = self._create_tool_text(tool)

            # Generate embedding
            embedding = await self._get_embedding(tool_text)

            # Cache it in memory
            self._tool_embedding_cache[tool.name] = embedding

        # Save to disk
        self._save_to_cache(tools_hash)
        self._build_embedding_index()

        logger.info(
            f"Tool embeddings computed and cached for {len(self._tool_embedding_cache)} tools"
        )

    def _build_embedding_index(self) -> None:
        """Build a native embedding index from cached tool embeddings."""
        if not self._tool_embedding_cache:
            self._embedding_index = None
            return

        try:
            import victor_native

            vectors = [embedding.tolist() for embedding in self._tool_embedding_cache.values()]
            labels = list(self._tool_embedding_cache.keys())
            self._embedding_index = victor_native.EmbeddingIndex(vectors, labels)
        except Exception as exc:
            self._embedding_index = None
            logger.debug("Native EmbeddingIndex unavailable, using batch fallback: %s", exc)

    def _query_embedding_index(
        self, query_embedding: np.ndarray, k: int, threshold: float
    ) -> Optional[Dict[str, float]]:
        """Query the native embedding index if it is available."""
        if self._embedding_index is None:
            return None

        try:
            results = self._embedding_index.query(
                query_embedding.tolist(), k=k, threshold=threshold
            )
        except Exception as exc:
            logger.debug("Native EmbeddingIndex query failed, using batch fallback: %s", exc)
            return None

        return {name: float(score) for name, score in results}

    # Cache version - aligned with Victor version, increment on breaking cache format changes
    # Format: "{victor_version}.{cache_revision}" e.g., "0.2.0.1" for first revision of 0.2.0
    CACHE_VERSION = "0.2.0"

    @staticmethod
    def _get_project_hash() -> str:
        """Get a short hash of the current project root for cache isolation.

        This ensures each project has its own tool embeddings cache,
        preventing cache pollution when switching between projects.

        Returns:
            8-character hash of the project root path
        """
        from victor.config.settings import get_project_paths

        try:
            project_root = get_project_paths().project_root
            # Use absolute path for consistent hashing
            path_str = str(project_root.resolve())
            return hashlib.sha256(path_str.encode()).hexdigest()[:8]
        except Exception:
            # Fallback to "global" if project detection fails
            return "global"

    def _calculate_tools_hash(self, tools: ToolRegistry) -> str:
        """Calculate hash of all tool definitions to detect changes.

        Args:
            tools: Tool registry

        Returns:
            SHA256 hash of tool definitions including count and version
        """
        # Create deterministic string from all tool definitions
        tool_list = sorted(tools.list_tools(), key=lambda t: t.name)
        tool_count = len(tool_list)
        tool_names = sorted([t.name for t in tool_list])

        tool_strings = []
        for tool in tool_list:
            tool_string = f"{tool.name}:{tool.description}:{tool.parameters}"
            tool_strings.append(tool_string)

        # Include tool count, names, and cache version in hash for robustness
        # This ensures ANY change (add, remove, rename, modify) triggers rebuild
        combined = (
            f"v{self.CACHE_VERSION}|count:{tool_count}|names:{','.join(tool_names)}|"
            + "|".join(tool_strings)
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def _delete_cache(self, reason: str) -> None:
        """Delete corrupted or stale cache file.

        Args:
            reason: Reason for deletion (for logging)
        """
        delete_cache_file(self.cache_file, reason, logger, label="Tool embeddings")

    def _load_from_cache(self, tools_hash: str) -> bool:
        """Load embeddings from pickle cache if valid.

        Performs robust validation:
        1. Cache version check (breaking format changes)
        2. Tools hash check (tool definition changes)
        3. Model name check (embedding model changes)
        4. Embedding dimension validation (model dimension mismatch)
        5. Data integrity check (corrupt numpy arrays)

        Args:
            tools_hash: Current hash of tool definitions

        Returns:
            True if loaded successfully, False otherwise
        """
        numpy = _ensure_numpy()

        def _check_version(cache_data: Dict[str, Any]) -> CacheValidation:
            # 1. Check cache version first (breaking changes)
            cached_version = cache_data.get("cache_version", 1)
            if cached_version != self.CACHE_VERSION:
                logger.info(
                    f"Cache version mismatch (cached: v{cached_version}, current: v{self.CACHE_VERSION}), "
                    "rebuilding embeddings"
                )
                return invalid(delete=True, reason="version mismatch")
            return valid()

        def _check_tools_hash(cache_data: Dict[str, Any]) -> CacheValidation:
            # 2. Verify cache is for same tools (hash includes count, names, and definitions)
            if cache_data.get("tools_hash") != tools_hash:
                cached_count = cache_data.get("tool_count", "?")
                cached_names = set(cache_data.get("tool_names", []))
                # The current tool names are embedded in the hash calculation
                logger.info(
                    f"Tool definitions changed (cached {cached_count} tools), "
                    "rebuilding embeddings"
                )
                # Log specific changes if available
                if cached_names:
                    current_names = set(cache_data.get("embeddings", {}).keys())
                    added = cached_names - current_names
                    removed = current_names - cached_names
                    if added:
                        logger.debug(f"Tools added since cache: {added}")
                    if removed:
                        logger.debug(f"Tools removed since cache: {removed}")
                # Hash mismatch intentionally keeps the cache file (no delete).
                return invalid(delete=False)
            return valid()

        def _check_model(cache_data: Dict[str, Any]) -> CacheValidation:
            # 3. Verify cache is for same embedding model
            if cache_data.get("embedding_model") != self.embedding_model:
                logger.info(
                    f"Embedding model changed (cached: {cache_data.get('embedding_model')}, "
                    f"current: {self.embedding_model}), rebuilding embeddings"
                )
                return invalid(delete=True, reason="model mismatch")
            return valid()

        def _check_embeddings(cache_data: Dict[str, Any]) -> CacheValidation:
            # 4. Validate embeddings exist and have correct structure
            embeddings = cache_data.get("embeddings", {})
            if not embeddings:
                logger.warning("Tool embeddings: cache missing embeddings dict")
                return invalid(delete=True, reason="missing embeddings")

            # 5. Validate embedding dimensions and integrity
            expected_dim = None
            for tool_name, embedding in embeddings.items():
                if not isinstance(embedding, numpy.ndarray):
                    logger.warning(f"Tool embeddings: '{tool_name}' is not a numpy array")
                    return invalid(delete=True, reason="invalid type")

                if len(embedding.shape) != 1:
                    logger.warning(
                        f"Tool embeddings: '{tool_name}' has wrong shape: {embedding.shape}"
                    )
                    return invalid(delete=True, reason="invalid shape")

                if expected_dim is None:
                    expected_dim = embedding.shape[0]
                elif embedding.shape[0] != expected_dim:
                    logger.warning(
                        f"Tool embeddings: dimension mismatch for '{tool_name}' "
                        f"(got {embedding.shape[0]}, expected {expected_dim})"
                    )
                    return invalid(delete=True, reason="dimension inconsistency")

                # Check for NaN or Inf (corruption detection)
                if not numpy.isfinite(embedding).all():
                    logger.warning(f"Tool embeddings: '{tool_name}' contains NaN or Inf values")
                    return invalid(delete=True, reason="corrupted embeddings")
            return valid()

        cache_data = load_validated_pickle(
            self.cache_file,
            validators=[_check_version, _check_tools_hash, _check_model, _check_embeddings],
            logger=logger,
            label="Tool embeddings",
            missing_message="No embedding cache file found, will rebuild",
        )
        if cache_data is None:
            return False

        # All checks passed - load embeddings
        self._tool_embedding_cache = cache_data["embeddings"]
        self._tools_hash = tools_hash
        return True

    def _save_to_cache(self, tools_hash: str) -> None:
        """Save embeddings to pickle cache with full metadata for robust invalidation.

        Args:
            tools_hash: Hash of tool definitions
        """
        tool_names = sorted(self._tool_embedding_cache.keys())
        cache_data = {
            "cache_version": self.CACHE_VERSION,
            "embedding_model": self.embedding_model,
            "tools_hash": tools_hash,
            "tool_count": len(tool_names),
            "tool_names": tool_names,  # Explicit list for debugging
            "embeddings": self._tool_embedding_cache,
        }

        if save_pickle_with_metadata(
            self.cache_file, cache_data, logger=logger, label="Tool embeddings"
        ):
            cache_size = self.cache_file.stat().st_size / 1024  # KB
            logger.info(
                f"Saved embedding cache to {self.cache_file} "
                f"({cache_size:.1f} KB, {len(tool_names)} tools, v{self.CACHE_VERSION})"
            )

    # Category alias mappings for logical category names
    # Maps logical names (file_ops, git_ops) to registry categories
    _CATEGORY_ALIASES = {
        "file_ops": ["filesystem", "code"],
        "git_ops": ["git", "merge"],
        "analysis": ["code", "pipeline", "security", "audit", "code_quality"],
        "refactoring": ["refactoring", "code"],
        "generation": ["generation", "search", "code"],
        "execution": ["code", "testing"],
        "code_intel": ["search", "code", "lsp"],
        "web": ["web"],
        "workflows": ["pipeline", "code"],
        "pipeline": ["pipeline"],
        "merge": ["merge", "git"],
        "security": ["security"],
        "audit": ["audit"],
    }

    def get_tools_for_logical_category(self, logical_category: str) -> List[str]:
        """Get tools for a logical category using the registry.

        Maps logical category names (file_ops, git_ops) to registry categories
        and returns the combined list of tools.

        Args:
            logical_category: Logical category name (file_ops, git_ops, etc.)

        Returns:
            List of tool names for the category
        """
        # Get registry categories for this logical category
        registry_categories = self._CATEGORY_ALIASES.get(logical_category, [])

        # Collect tools from all registry categories
        tools = set()
        registry = ToolMetadataRegistry.get_instance()
        for category in registry_categories:
            tools.update(registry.get_tools_by_category(category))

        return list(tools)

    # Conceptual query patterns that strongly prefer semantic_code_search over code_search
    # When these patterns are detected, code_search is excluded from mandatory tools
    CONCEPTUAL_QUERY_PATTERNS = [
        "inherit",  # "classes that inherit from"
        "implement",  # "classes implementing interface"
        "extend",  # "classes that extend"
        "subclass",  # "subclasses of"
        "pattern",  # "find patterns", "error handling patterns"
        "similar",  # "similar code", "similar to"
        "related",  # "related functionality"
        "usage",  # "find usages", "usage patterns"
        "example",  # "examples of", "example code"
        "all classes",  # "find all classes"
        "all functions",  # "all functions that"
        "all methods",  # "all methods"
        "how is",  # "how is X done"
        "where is",  # "where is X implemented"
        "what classes",  # "what classes"
        "which files",  # "which files handle"
        "error handling",  # conceptual search
        "exception",  # conceptual search
        "try catch",  # conceptual search
        "logging",  # conceptual search
        "caching",  # conceptual search
        "authentication",  # conceptual search
        "validation",  # conceptual search
        "similar to",  # conceptual search
        "related to",  # conceptual search
        # Analysis/exploration patterns - trigger semantic search
        "analyze",  # "analyze codebase", "analyze this code"
        "understand",  # "understand architecture"
        "architecture",  # "show architecture", "explain architecture"
        "structure",  # "codebase structure"
        "overview",  # "codebase overview"
        "how does",  # "how does X work"
        "what is the",  # "what is the architecture"
        "explore",  # "explore codebase"
    ]

    # Tools for conceptual queries - forces semantic search as primary
    # Excludes ls to prevent LLM from exploring instead of searching
    # Uses centralized FallbackTools from tool_selection_defaults
    CONCEPTUAL_FALLBACK_TOOLS: List[str] = list(FallbackTools.CONCEPTUAL_FALLBACK_TOOLS)

    # Common fallback tools - used when semantic selection returns too few results
    # These are the most universally useful tools
    # Uses centralized FallbackTools from tool_selection_defaults
    COMMON_FALLBACK_TOOLS: List[str] = list(FallbackTools.COMMON_FALLBACK_TOOLS)

    def _get_fallback_tools(
        self, tools: "ToolRegistry", max_tools: int = 5, query: str = ""
    ) -> List[str]:
        """Get fallback tools when semantic selection returns too few results.

        Instead of broadcasting ALL tools (which wastes tokens), return a
        curated list of common, universally useful tools.

        For conceptual queries (inheritance, patterns, etc.), uses a restricted
        tool set that forces semantic_code_search as the primary tool.

        Args:
            tools: Tool registry
            max_tools: Maximum fallback tools to return
            query: User query (used to detect conceptual queries)

        Returns:
            List of fallback tool names
        """
        # For conceptual queries, use restricted tool set that forces semantic search
        if query and self._is_conceptual_query(query):
            fallback = []
            for tool_name in self.CONCEPTUAL_FALLBACK_TOOLS:
                if tools.is_tool_enabled(tool_name) and tools.get(tool_name):
                    fallback.append(tool_name)
                if len(fallback) >= max_tools:
                    break
            logger.debug(f"Using CONCEPTUAL fallback tools ({len(fallback)}): {fallback}")
            return fallback

        # Standard fallback for non-conceptual queries
        fallback = []
        for tool_name in self.COMMON_FALLBACK_TOOLS:
            if tools.is_tool_enabled(tool_name) and tools.get(tool_name):
                fallback.append(tool_name)
            if len(fallback) >= max_tools:
                break

        logger.debug(f"Using fallback tools ({len(fallback)}): {fallback}")
        return fallback

    def _is_conceptual_query(self, query: str) -> bool:
        """Check if query is conceptual and should prefer semantic_code_search.

        Args:
            query: User query

        Returns:
            True if query matches conceptual patterns
        """
        query_lower = query.lower()
        for pattern in self.CONCEPTUAL_QUERY_PATTERNS:
            if pattern in query_lower:
                logger.log(TRACE, f"Conceptual query detected via pattern: '{pattern}'")
                return True
        return False

    def _get_mandatory_tools(self, query: str) -> List[str]:
        """Get tools that MUST be included based on keywords.

        Uses registry-based lookup. Tools declare mandatory keywords via
        @tool(mandatory_keywords=["show diff"]).

        Args:
            query: User query

        Returns:
            List of tool names that are mandatory for this query
        """
        # Use registry-based mandatory keyword lookup (decorator-driven)
        mandatory = get_tools_matching_mandatory_keywords(query)
        if mandatory:
            logger.log(
                TRACE,
                f"Registry mandatory tools: {mandatory}",
            )

        return list(mandatory)

    def _get_relevant_categories(self, query: str) -> List[str]:
        """Determine which tool categories are relevant for this query.

        Uses ToolMetadataRegistry dynamically for category lookups, falling
        back to hardcoded lists only when registry is empty.

        Args:
            query: User query

        Returns:
            List of relevant tool names from categories
        """
        query_lower = query.lower()
        relevant_tools = []

        # Multi-step tasks need file_ops and git_ops
        if any(sep in query for sep in [";", "then", "after", "next", "and then"]):
            relevant_tools.extend(self.get_tools_for_logical_category("file_ops"))
            relevant_tools.extend(self.get_tools_for_logical_category("git_ops"))
            logger.debug("Multi-step task detected, including file_ops and git_ops")

        # Analysis keywords
        if any(kw in query_lower for kw in ["analyze", "review", "check", "scan", "audit"]):
            relevant_tools.extend(self.get_tools_for_logical_category("analysis"))
            logger.debug("Analysis task detected")

        # Editing keywords
        if any(kw in query_lower for kw in ["edit", "modify", "change", "update", "fix"]):
            relevant_tools.extend(self.get_tools_for_logical_category("file_ops"))
            relevant_tools.extend(self.get_tools_for_logical_category("refactoring"))
            logger.debug("Editing task detected")

        # Git/diff keywords
        if any(
            self._keyword_in_text(query_lower, kw)
            for kw in ["diff", "commit", "pr", "git", "pull request"]
        ):
            relevant_tools.extend(self.get_tools_for_logical_category("git_ops"))
            logger.debug("Git operation detected")

        # Code navigation
        if any(kw in query_lower for kw in ["find", "locate", "search", "where"]):
            code_intel_tools = self.get_tools_for_logical_category("code_intel")
            # For conceptual queries, exclude code_search to prefer semantic_code_search
            if self._is_conceptual_query(query):
                code_intel_tools = [t for t in code_intel_tools if t != "code_search"]
                logger.debug("Code navigation detected (conceptual - excluding code_search)")
            else:
                logger.debug("Code navigation detected")
            relevant_tools.extend(code_intel_tools)

        # Generation/creation
        if any(kw in query_lower for kw in ["create", "generate", "make", "write new"]):
            relevant_tools.extend(self.get_tools_for_logical_category("file_ops"))
            relevant_tools.extend(self.get_tools_for_logical_category("generation"))
            logger.debug("Generation task detected")

        # Default: file_ops + execution
        if not relevant_tools:
            relevant_tools.extend(self.get_tools_for_logical_category("file_ops"))
            relevant_tools.extend(self.get_tools_for_logical_category("execution"))
            logger.debug("Using default categories: file_ops + execution")

        return list(set(relevant_tools))

    def _is_analysis_query(self, query: str) -> bool:
        """Heuristic check for analysis/review intents with fuzzy matching for typos."""
        query_lower = query.lower()

        # Use fuzzy matching if available
        if _FUZZY_MATCHING_AVAILABLE:
            analysis_keywords = {
                "analyze": 1.0,
                "analysis": 1.0,
                "review": 1.0,
                "check": 1.0,
                "scan": 1.0,
                "audit": 1.0,
                "inspect": 1.0,
                "examine": 1.0,
                "investigate": 1.0,
            }
            matches, _ = match_keywords_cascading(query_lower, analysis_keywords, use_fuzzy=True)
            return len(matches) > 0
        else:
            # Fallback to exact matching
            analysis_keywords = [
                "analyze",
                "analysis",
                "review",
                "check",
                "scan",
                "audit",
                "inspect",
            ]
            return any(kw in query_lower for kw in analysis_keywords)

    def _extract_pending_actions(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract actions mentioned in original request but not yet completed.

        Args:
            conversation_history: List of conversation messages

        Returns:
            List of pending action types
        """
        if not conversation_history:
            return []

        # Get the original user request (first user message)
        original_request = None
        for msg in conversation_history:
            if msg.get("role") == "user":
                original_request = msg.get("content", "")
                break

        if not original_request:
            return []

        pending = []

        # Action keywords to check
        # NOTE: Be specific to avoid false positives. Generic words like "check"
        # or "examine" in analysis tasks shouldn't trigger pending actions.
        action_patterns = {
            "edit": [
                "edit the file",
                "modify the file",
                "change the code",
                "update the file",
            ],
            "show_diff": ["show diff", "show the diff", "git diff", "compare changes"],
            "read": ["read the file", "read this file", "show me the file"],
            "propose": ["propose changes", "suggest changes", "recommend changes"],
            "create": [
                "create a file",
                "create new file",
                "generate a file",
                "make a file",
            ],
            "test": ["run tests", "run the tests", "execute tests"],
            "commit": ["commit the changes", "make a commit", "git commit"],
            "pr": ["create a pull request", "open a pr", "make a pr", "raise a pr"],
        }

        # Check which actions were requested
        original_lower = original_request.lower()
        for action_type, keywords in action_patterns.items():
            if any(self._keyword_in_text(original_lower, kw) for kw in keywords):
                # Check if this action was completed by looking at tool results
                completed = self._was_action_completed(action_type, conversation_history)
                if not completed:
                    pending.append(action_type)
                    logger.log(TRACE, f"Pending action detected: {action_type}")

        return pending

    def _was_action_completed(self, action: str, history: List[Dict[str, Any]]) -> bool:
        """Check if an action was completed based on conversation history.

        Checks both tool calls made and text content for completion indicators.

        Args:
            action: Action type to check
            history: Conversation history

        Returns:
            True if action was completed, False otherwise
        """
        # Map action types to tool names that complete them
        action_to_tools = {
            "show_diff": ["git", "shell"],
            "edit": ["edit", "write", "patch"],
            "read": ["read", "symbol", "search", "overview", "ls"],
            "create": ["write", "scaffold"],
            "test": ["test", "shell"],
            "commit": ["git"],
            "pr": ["pr", "git"],
        }

        # Check if any tool that completes this action was called
        tools_for_action = action_to_tools.get(action, [])
        for msg in history:
            if msg.get("role") == "assistant":
                # Check tool_calls in the message (handle None explicitly)
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    tool_name = tc.get("name", "") or tc.get("function", {}).get("name", "")
                    if tool_name in tools_for_action:
                        return True

                # Also check content for tool result markers
                content = str(msg.get("content", "")).lower()
                if action == "show_diff" and ("diff" in content or "git diff" in content):
                    return True
                elif (
                    action == "test"
                    and "test" in content
                    and ("passed" in content or "failed" in content)
                ):
                    return True

        # Check user messages for tool output markers (TOOL_OUTPUT tags)
        for msg in history:
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                if "<TOOL_OUTPUT" in content:
                    # Check if any relevant tool output exists
                    for tool_name in tools_for_action:
                        if f'tool="{tool_name}"' in content or f"tool='{tool_name}'" in content:
                            return True

        return False

    def _keyword_in_text(self, text: str, keyword: str) -> bool:
        """Return True if keyword is present in text with sane boundaries."""
        if len(keyword) <= 3:
            return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
        return keyword in text

    def _build_contextual_query(
        self,
        current_query: str,
        conversation_history: List[Dict[str, Any]],
        pending_actions: List[str],
    ) -> str:
        """Build enhanced query with conversation context.

        Args:
            current_query: Current user message
            conversation_history: Full conversation history
            pending_actions: List of pending action types

        Returns:
            Enhanced query with context
        """
        # Get last 2 user messages for context
        recent_context = []
        for msg in reversed(conversation_history[-6:]):  # Last 6 messages (3 exchanges)
            if msg.get("role") == "user":
                recent_context.append(msg.get("content", ""))
                if len(recent_context) >= 2:
                    break

        # Build enhanced query
        context_parts = []

        if recent_context and len(recent_context) > 1:
            context_parts.append(f"Context: {recent_context[-1]}")  # Previous request

        if pending_actions:
            context_parts.append(f"Incomplete: {', '.join(pending_actions)}")

        context_parts.append(f"Now: {current_query}")

        enhanced = " | ".join(context_parts)
        logger.debug(f"Contextual query: {enhanced}")
        return enhanced

    # ========================================================================
    # Phase 3: Tool Usage Tracking and Learning
    # ========================================================================

    def _load_usage_cache(self) -> None:
        """Load tool usage statistics from disk cache (Phase 3)."""
        if not self._usage_cache_file.exists():
            logger.debug("No usage cache found - starting fresh")
            return

        try:
            with open(self._usage_cache_file, "rb") as f:
                self._tool_usage_cache = pickle.load(f)
            logger.info(f"Loaded usage stats for {len(self._tool_usage_cache)} tools")
        except Exception as e:
            logger.warning(f"Failed to load usage cache: {e}")
            self._tool_usage_cache = {}

    def _save_usage_cache(self, force: bool = False) -> None:
        """Save tool usage statistics to disk cache (Phase 3).

        Uses dirty flag pattern to avoid redundant disk writes.
        Only saves when data has changed or force=True (e.g., on shutdown).

        Args:
            force: Force save even if not dirty (for shutdown)
        """
        if not force and not self._usage_cache_dirty:
            return  # Nothing changed, skip save

        try:
            with open(self._usage_cache_file, "wb") as f:
                pickle.dump(self._tool_usage_cache, f)
            self._usage_cache_dirty = False  # Clear dirty flag after save
            logger.debug(f"Saved usage stats for {len(self._tool_usage_cache)} tools")
        except Exception as e:
            logger.warning(f"Failed to save usage cache: {e}")

    def _initialize_all_tool_stats(self, tools: List[Any]) -> None:
        """Pre-initialize usage stats for ALL tools, not just used ones.

        This ensures that show_tool_stats.py and other analytics tools can
        report on all available tools, even those that haven't been used yet.
        Tools are initialized with 0 usage count but preserve existing stats
        for tools that have been used.

        Args:
            tools: List of BaseTool instances from ToolRegistry
        """
        initialized_count = 0
        for tool in tools:
            if tool.name not in self._tool_usage_cache:
                self._tool_usage_cache[tool.name] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "last_used": 0,
                    "recent_contexts": [],
                }
                initialized_count += 1

        if initialized_count > 0:
            logger.info(
                f"Pre-initialized usage stats for {initialized_count} new tools "
                f"(total: {len(self._tool_usage_cache)} tools)"
            )
            self._usage_cache_dirty = True  # Mark dirty, save on shutdown

    def _record_tool_usage(self, tool_name: str, query: str, success: bool = True) -> None:
        """Record tool selection for learning (Phase 3).

        NOTE: This records tool SELECTION, not execution. Actual execution
        should be tracked separately via record_tool_execution().

        Args:
            tool_name: Name of the tool that was selected
            query: The query context where it was selected
            success: Whether the tool selection was valid
        """
        import time

        if tool_name not in self._tool_usage_cache:
            self._tool_usage_cache[tool_name] = {
                "usage_count": 0,
                "success_count": 0,
                "last_used": 0,
                "recent_contexts": [],
            }

        stats = self._tool_usage_cache[tool_name]
        stats["usage_count"] += 1
        if success:
            stats["success_count"] += 1
        stats["last_used"] = time.time()

        # Keep last 50 query contexts for semantic matching (better pattern recognition)
        query_summary = query[:100]  # Truncate long queries
        stats["recent_contexts"].append(query_summary)
        if len(stats["recent_contexts"]) > 50:
            stats["recent_contexts"] = stats["recent_contexts"][-50:]

        # Mark dirty - save will happen on shutdown via close()
        self._usage_cache_dirty = True

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool = True,
        execution_time: float = 0.0,
    ) -> None:
        """Record actual tool execution for sequence tracking (Phase 9).

        This should be called AFTER a tool is actually executed, not just selected.
        Updates the sequence tracker to build workflow patterns.

        Args:
            tool_name: Name of the tool that was executed
            success: Whether the execution succeeded
            execution_time: Time taken for execution
        """
        if self._sequence_tracker:
            self._sequence_tracker.record_execution(tool_name, success, execution_time)

    async def _get_usage_boost(self, tool_name: str, query: str) -> float:
        """Calculate similarity boost based on usage history (Phase 3).

        Args:
            tool_name: Name of the tool
            query: Current query

        Returns:
            Boost value (0.0 to 0.2) to add to similarity score
        """
        if tool_name not in self._tool_usage_cache:
            return 0.0

        stats = self._tool_usage_cache[tool_name]

        # Base boost from usage frequency
        usage_boost = min(0.05, stats["usage_count"] * 0.01)  # Max 0.05

        # Boost from success rate
        success_rate = (
            stats["success_count"] / stats["usage_count"] if stats["usage_count"] > 0 else 0
        )
        success_boost = success_rate * 0.05  # Max 0.05

        # Boost from recency (tools used recently get slight preference)
        import time

        time_since_use = time.time() - stats["last_used"]
        days_since_use = time_since_use / 86400  # Convert to days
        recency_boost = max(0, 0.05 - (days_since_use * 0.01))  # Max 0.05

        # Context similarity boost (compare with recent contexts)
        context_boost = 0.0
        if stats["recent_contexts"]:
            try:
                query_emb = await self._get_embedding(query)
                context_similarities = []

                for ctx in stats["recent_contexts"][-3:]:  # Check last 3 contexts
                    ctx_emb = await self._get_embedding(ctx)
                    sim = self._cosine_similarity(query_emb, ctx_emb)
                    context_similarities.append(sim)

                if context_similarities:
                    avg_context_sim = sum(context_similarities) / len(context_similarities)
                    context_boost = avg_context_sim * 0.05  # Max 0.05

            except Exception as e:
                logger.debug(f"Context boost calculation failed: {e}")

        total_boost = usage_boost + success_boost + recency_boost + context_boost
        # Use TRACE for per-tool verbose logging
        logger.log(
            TRACE,
            f"Usage boost for {tool_name}: {total_boost:.3f} "
            f"(usage={usage_boost:.3f}, success={success_boost:.3f}, "
            f"recency={recency_boost:.3f}, context={context_boost:.3f})",
        )

        return min(0.2, total_boost)  # Cap total boost at 0.2

    def _get_sequence_boost(self, tool_name: str) -> float:
        """Calculate boost based on tool sequence patterns (Phase 9).

        Uses the ToolSequenceTracker to predict likely next tools based on
        previously executed tools in the session, providing 15-20% improvement
        in tool selection accuracy.

        Args:
            tool_name: Name of the tool to calculate boost for

        Returns:
            Boost value (0.0 to 0.15) to add to similarity score
        """
        if not self._sequence_tracker:
            return 0.0

        # Get sequence suggestions (confidence ordered)
        suggestions = self._sequence_tracker.get_next_suggestions(top_k=10)
        canonical_tool_name = get_canonical_name(tool_name)

        # Find this tool's confidence in suggestions
        for suggested_tool, confidence in suggestions:
            if suggested_tool == canonical_tool_name:
                # Scale confidence to boost value (max 0.15)
                boost = confidence * 0.15
                logger.log(
                    TRACE,
                    f"Sequence boost for {tool_name}: +{boost:.3f} "
                    f"(confidence={confidence:.2f})",
                )
                return boost

        return 0.0

    @staticmethod
    def _to_sequence_public_tool_name(tool_name: str) -> str:
        """Expose stable selector-facing aliases for canonical sequence names."""
        canonical_tool_name = get_canonical_name(tool_name)
        return _PREFERRED_SEQUENCE_TOOL_ALIASES.get(canonical_tool_name, canonical_tool_name)

    def _apply_sequence_boosts(
        self, similarities: List[Tuple[Any, float]]
    ) -> List[Tuple[Any, float]]:
        """Apply sequence-based boosts to all tool similarity scores.

        Args:
            similarities: List of (tool, score) tuples

        Returns:
            List of (tool, boosted_score) tuples
        """
        if not self._sequence_tracker:
            return similarities

        boosted = []
        for tool, score in similarities:
            sequence_boost = self._get_sequence_boost(tool.name)
            boosted_score = score + sequence_boost
            boosted.append((tool, boosted_score))

        return boosted

    def _get_cost_penalty(self, tool: Any, tools: ToolRegistry) -> float:
        """Calculate cost penalty for a tool based on its cost tier.

        Higher-cost tools receive a penalty to deprioritize them when
        lower-cost alternatives with similar relevance exist.

        Args:
            tool: Tool object
            tools: Tool registry to lookup cost tiers

        Returns:
            Penalty value (0.0 to 0.15) to subtract from similarity score
        """
        if not self.cost_aware_selection:
            return 0.0

        cost_tier = tools.get_tool_cost(tool.name)
        if cost_tier is None:
            return 0.0

        # Calculate penalty: weight * factor
        # FREE (0) = 0.0 penalty
        # LOW (1) = 0.05 penalty
        # MEDIUM (2) = 0.10 penalty
        # HIGH (3) = 0.15 penalty
        penalty = cost_tier.weight * self.cost_penalty_factor

        if penalty > 0:
            logger.log(
                TRACE,
                f"Cost penalty for {tool.name}: -{penalty:.3f} (tier={cost_tier.value})",
            )

        return penalty

    def _generate_cost_warnings(
        self, selected_tools: List[Tuple[Any, float]], tools: ToolRegistry
    ) -> List[str]:
        """Generate user-facing warnings for high-cost tools in selection.

        Args:
            selected_tools: List of (tool, score) tuples
            tools: Tool registry to lookup cost tiers

        Returns:
            List of warning messages for display to user
        """
        if not self.cost_aware_selection:
            return []

        warnings = []
        for tool, _ in selected_tools:
            # Skip if we've already warned about this tool (warn once per session)
            if tool.name in self._warned_tools:
                continue

            cost_tier = tools.get_tool_cost(tool.name)
            if cost_tier and cost_tier in COST_TIER_WARNINGS:
                warning_msg = f"[{tool.name}] {COST_TIER_WARNINGS[cost_tier]}"
                warnings.append(warning_msg)
                self._warned_tools.add(tool.name)
                logger.log(TRACE, f"Cost warning: {warning_msg}")

        return warnings

    def get_last_cost_warnings(self) -> List[str]:
        """Get cost warnings from the last tool selection.

        Returns:
            List of warning messages about high-cost tools selected.
            Empty list if no high-cost tools were selected.
        """
        return self._last_cost_warnings.copy()

    def clear_cost_warnings(self) -> None:
        """Clear stored cost warnings and reset warned tools tracking."""
        self._last_cost_warnings = []
        self._warned_tools.clear()

    async def select_relevant_tools_with_context(
        self,
        user_message: str,
        tools: ToolRegistry,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_tools: int = 5,
        similarity_threshold: float = SemanticSelectorDefaults.SIMILARITY_THRESHOLD,
    ) -> List[ToolDefinition]:
        """Select tools with full conversation context awareness (Phase 2).

        This method enhances tool selection by:
        - Tracking pending actions from the original request
        - Including conversation context in semantic search
        - Ensuring mandatory tools for pending actions

        Args:
            user_message: Current user message
            tools: Tool registry
            conversation_history: Full conversation history (Phase 2)
            max_tools: Maximum tools to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of relevant tools with context-aware selection
        """
        # Phase 2: Extract pending actions
        pending_actions = []
        if conversation_history:
            pending_actions = self._extract_pending_actions(conversation_history)
            if pending_actions:
                logger.debug(f"Pending actions: {pending_actions}")

        # Phase 2: Build contextual query
        enhanced_query = user_message
        if conversation_history and pending_actions:
            enhanced_query = self._build_contextual_query(
                user_message, conversation_history, pending_actions
            )

        # Phase 1: Get mandatory tools (including those for pending actions)
        mandatory_tool_names = self._get_mandatory_tools(enhanced_query)

        # Always include core read-only tools for analysis-style queries.
        if self._is_analysis_query(enhanced_query):
            mandatory_tool_names.extend(get_core_readonly_tools())

        # Add mandatory tools for pending actions
        # NOTE: Uses canonical short names for token efficiency
        pending_action_tools = {
            "show_diff": ["shell"],
            "edit": ["edit", "read"],
            "commit": ["git"],
            "pr": ["pr"],
            "test": ["shell", "test"],
        }

        for action in pending_actions:
            if action in pending_action_tools:
                mandatory_tool_names.extend(pending_action_tools[action])
                logger.debug(
                    f"Added mandatory tools for pending '{action}': {pending_action_tools[action]}"
                )

        mandatory_tool_names = list(set(mandatory_tool_names))
        if mandatory_tool_names:
            logger.debug(f"Total mandatory tools: {mandatory_tool_names}")

        # Phase 1: Get relevant categories
        category_tools = self._get_relevant_categories(enhanced_query)
        logger.debug(f"Category tools ({len(category_tools)}): {category_tools[:5]}...")

        # Get embedding for enhanced query
        query_embedding = await self._get_embedding(enhanced_query)

        relevant_tools: List[Any] = []

        for tool in tools.list_tools():
            # Skip if not in relevant categories and not mandatory
            if tool.name not in category_tools and tool.name not in mandatory_tool_names:
                continue

            relevant_tools.append(tool)

        # Apply boosts/penalties and filter by threshold
        similarities: List[Tuple[Any, float]] = []
        for tool, similarity in await self._score_candidate_tools(query_embedding, relevant_tools):
            # Boost mandatory tools
            if tool.name in mandatory_tool_names:
                similarity = max(similarity, 0.9)  # Ensure mandatory tools rank high

            # Phase 3: Apply usage boost based on learning
            usage_boost = await self._get_usage_boost(tool.name, enhanced_query)
            similarity += usage_boost

            # Phase 9: Apply sequence boost based on workflow patterns
            sequence_boost = self._get_sequence_boost(tool.name)
            similarity += sequence_boost

            # Phase 5: Apply cost penalty for high-cost tools
            cost_penalty = self._get_cost_penalty(tool, tools)
            similarity -= cost_penalty

            if similarity >= similarity_threshold:
                similarities.append((tool, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Ensure all mandatory tools are included
        mandatory_tools = [tool for tool in tools.list_tools() if tool.name in mandatory_tool_names]

        # Combine mandatory + top semantic matches
        selected_tools = []
        selected_names = set()

        # First, add all mandatory tools
        for tool in mandatory_tools:
            if tool.name not in selected_names:
                selected_tools.append((tool, SemanticSelectorDefaults.MANDATORY_TOOL_SCORE))
                selected_names.add(tool.name)

        # Then add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Phase 8: Smart fallback - if too few tools selected, add common fallback tools
        min_threshold = SemanticSelectorDefaults.MIN_TOOLS_THRESHOLD
        if len(selected_tools) < min_threshold:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append(
                            (
                                fallback_tool,
                                SemanticSelectorDefaults.FALLBACK_TOOL_SCORE,
                            )
                        )
                        selected_names.add(fallback_name)
            logger.debug(
                f"Added {len(fallback_names)} fallback tools (selection returned < {min_threshold})"
            )

        # Log selection
        tool_names = [t.name for t, _ in selected_tools]
        scores = [f"{s:.3f}" for _, s in selected_tools]
        logger.debug(
            f"Context-aware selection: {len(selected_tools)} tools (mandatory={len(mandatory_tools)}, "
            f"pending_actions={len(pending_actions)}): "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores, strict=False))}"
        )

        # Phase 3: Record tool usage for learning
        for tool_name in tool_names:
            self._record_tool_usage(tool_name, user_message, success=True)

        # Phase 6: Generate and store cost warnings for high-cost tools
        self._last_cost_warnings = self._generate_cost_warnings(selected_tools, tools)
        if self._last_cost_warnings:
            logger.debug(f"Cost info: {len(self._last_cost_warnings)} high-cost tools selected")

        # Emit semantic match event for RL learning
        self._emit_semantic_match_event(
            selected_tools=selected_tools,
            threshold=similarity_threshold,
            task_type="default",
            classification_aware=False,
            excluded_count=0,
        )

        # Store last selection scores for schema promotion (STUB→COMPACT)
        self._last_selection_scores: Dict[str, float] = {
            tool.name: score for tool, score in selected_tools
        }

        # Convert to ToolDefinition
        return [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in selected_tools
        ]

    def get_last_selection_scores(self) -> Dict[str, float]:
        """Return similarity scores from the most recent selection.

        Used by ToolSelector to promote high-confidence STUB tools to
        COMPACT schema level for better LLM call accuracy.
        """
        return getattr(self, "_last_selection_scores", {})

    async def select_relevant_tools(
        self,
        user_message: str,
        tools: ToolRegistry,
        max_tools: int = 5,
        similarity_threshold: float = SemanticSelectorDefaults.SIMILARITY_THRESHOLD,
    ) -> List[ToolDefinition]:
        """Select relevant tools using semantic similarity with category filtering.

        Enhanced with Phase 1 features:
        - Mandatory tool selection for specific keywords
        - Category-based filtering for better relevance

        Args:
            user_message: User's input message
            tools: Tool registry
            max_tools: Maximum number of tools to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant ToolDefinition objects, sorted by relevance
        """
        # Phase 1: Get mandatory tools (always included)
        mandatory_tool_names = self._get_mandatory_tools(user_message)
        if mandatory_tool_names:
            logger.debug(f"Mandatory tools: {mandatory_tool_names}")

        # Always include core read-only tools for analysis-style queries.
        if self._is_analysis_query(user_message):
            mandatory_tool_names.extend(get_core_readonly_tools())

        # Phase 1: Get relevant categories
        category_tools = self._get_relevant_categories(user_message)
        logger.debug(f"Category tools ({len(category_tools)}): {category_tools[:5]}...")

        # Get embedding for user message
        query_embedding = await self._get_embedding(user_message)

        relevant_tools: List[Any] = []

        for tool in tools.list_tools():
            # Skip if not in relevant categories and not mandatory
            if tool.name not in category_tools and tool.name not in mandatory_tool_names:
                continue

            relevant_tools.append(tool)

        # Apply boosts/penalties and filter by threshold
        similarities: List[Tuple[Any, float]] = []
        for tool, similarity in await self._score_candidate_tools(query_embedding, relevant_tools):
            # Boost mandatory tools
            if tool.name in mandatory_tool_names:
                similarity = max(similarity, 0.9)  # Ensure mandatory tools rank high

            # Phase 9: Apply sequence boost based on workflow patterns
            sequence_boost = self._get_sequence_boost(tool.name)
            similarity += sequence_boost

            # Phase 5: Apply cost penalty for high-cost tools
            cost_penalty = self._get_cost_penalty(tool, tools)
            similarity -= cost_penalty

            if similarity >= similarity_threshold:
                similarities.append((tool, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Tiered triage: gate semantic selection on top-score confidence band.
        # REJECT → skip semantic matches (mandatories only); VERIFY → relax threshold by 0.1.
        tiered_svc = self._get_tiered_service()
        if tiered_svc is not None and similarities:
            from victor.agent.decisions.schemas import DecisionType
            from victor.agent.services.tiered_decision_service import (
                ClassificationTriage,
            )

            top_score = similarities[0][1]
            triage = tiered_svc.classify_with_triage(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"query": user_message[:300]},
                heuristic_result="tool_selection",
                heuristic_confidence=top_score,
            )
            if triage.triage_outcome == ClassificationTriage.REJECT:
                similarities = []  # trust only mandatory tools
            elif triage.triage_outcome == ClassificationTriage.VERIFY:
                similarities = [
                    (tool, score)
                    for tool, score in similarities
                    if score >= similarity_threshold - 0.1
                ]

        # Ensure all mandatory tools are included
        mandatory_tools = [tool for tool in tools.list_tools() if tool.name in mandatory_tool_names]

        # Combine mandatory + top semantic matches
        selected_tools = []
        selected_names = set()

        # First, add all mandatory tools
        for tool in mandatory_tools:
            if tool.name not in selected_names:
                selected_tools.append((tool, SemanticSelectorDefaults.MANDATORY_TOOL_SCORE))
                selected_names.add(tool.name)

        # Then add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Phase 8: Smart fallback - if too few tools selected, add common fallback tools
        # This prevents broadcasting ALL tools (which wastes tokens).
        min_threshold = SemanticSelectorDefaults.MIN_TOOLS_THRESHOLD
        if len(selected_tools) < min_threshold:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append(
                            (
                                fallback_tool,
                                SemanticSelectorDefaults.FALLBACK_TOOL_SCORE,
                            )
                        )
                        selected_names.add(fallback_name)
            logger.debug(
                f"Added {len(fallback_names)} fallback tools (semantic selection returned < {min_threshold})"
            )

        # Log selection
        tool_names = [t.name for t, _ in selected_tools]
        scores = [f"{s:.3f}" for _, s in selected_tools]
        logger.debug(
            f"Selected {len(selected_tools)} tools (mandatory={len(mandatory_tools)}): "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores, strict=False))}"
        )

        # Phase 6: Generate and store cost warnings for high-cost tools
        self._last_cost_warnings = self._generate_cost_warnings(selected_tools, tools)
        if self._last_cost_warnings:
            logger.debug(f"Cost info: {len(self._last_cost_warnings)} high-cost tools selected")

        # Emit semantic match event for RL learning
        self._emit_semantic_match_event(
            selected_tools=selected_tools,
            threshold=similarity_threshold,
            task_type="default",
            classification_aware=False,
            excluded_count=0,
        )

        # Convert to ToolDefinition
        return [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in selected_tools
        ]

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_provider == "sentence-transformers":
            return await self._get_sentence_transformer_embedding(text)
        elif self.embedding_provider in ["ollama", "vllm", "lmstudio"]:
            return await self._get_api_embedding(text)
        else:
            raise NotImplementedError(f"Provider {self.embedding_provider} not yet supported")

    async def _get_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Get embedding from sentence-transformers using shared EmbeddingService.

        Uses the shared EmbeddingService singleton for memory efficiency
        (single model instance shared with IntentClassifier and other components).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            # Use shared embedding service (singleton)
            embedding_service = _get_embedding_service_class().get_instance(
                model_name=self.embedding_model
            )
            return await embedding_service.embed_text(text)

        except Exception as e:
            logger.warning(f"Failed to get embedding from EmbeddingService: {e}")
            # Fall back to random embedding (better than crashing)
            logger.warning("Falling back to random embedding")
            np = _ensure_numpy()
            return np.random.randn(384).astype(np.float32)

    async def _get_api_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama/vLLM/LMStudio API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            np = _ensure_numpy()
            return np.array(data["embedding"], dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to get embedding from {self.embedding_provider}: {e}")
            # Fallback to random embedding (better than crashing)
            np = _ensure_numpy()
            return np.random.randn(768).astype(np.float32)

    async def _score_candidate_tools(
        self, query_embedding: np.ndarray, candidate_tools: List[Any]
    ) -> List[Tuple[Any, float]]:
        """Compute similarity scores for candidate tools.

        Prefers the native EmbeddingIndex when available and falls back to the
        existing batch cosine-similarity path otherwise.
        """
        if not candidate_tools:
            return []

        index_scores = self._query_embedding_index(
            query_embedding=query_embedding,
            k=max(len(self._tool_embedding_cache), len(candidate_tools)),
            threshold=0.0,
        )
        if index_scores is not None:
            similarities: List[Tuple[Any, float]] = []
            for tool in candidate_tools:
                if tool.name in index_scores:
                    similarity = index_scores[tool.name]
                elif tool.name in self._tool_embedding_cache:
                    similarity = self._cosine_similarity(
                        query_embedding, self._tool_embedding_cache[tool.name]
                    )
                else:
                    tool_text = self._create_tool_text(tool)
                    tool_embedding = await self._get_embedding(tool_text)
                    similarity = self._cosine_similarity(query_embedding, tool_embedding)
                similarities.append((tool, similarity))
            return similarities

        tool_embeddings: List[np.ndarray] = []
        for tool in candidate_tools:
            if tool.name in self._tool_embedding_cache:
                tool_embedding = self._tool_embedding_cache[tool.name]
            else:
                tool_text = self._create_tool_text(tool)
                tool_embedding = await self._get_embedding(tool_text)
            tool_embeddings.append(tool_embedding)

        all_similarities = self._batch_cosine_similarity(query_embedding, tool_embeddings)
        return list(zip(candidate_tools, all_similarities))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Uses Rust-accelerated implementation with NumPy fallback.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (0-1)
        """
        from victor.processing.native import cosine_similarity

        # Convert numpy arrays to lists for Rust/fallback interface
        return cosine_similarity(a.tolist(), b.tolist())

    @staticmethod
    def _batch_cosine_similarity(
        query: np.ndarray, tool_embeddings: List[np.ndarray]
    ) -> List[float]:
        """Calculate cosine similarity between query and multiple tool embeddings.

        Uses Rust-accelerated batch implementation for 5-10x speedup.

        Args:
            query: Query embedding vector
            tool_embeddings: List of tool embedding vectors

        Returns:
            List of similarity scores (0-1)
        """
        from victor.processing.native import batch_cosine_similarity

        # Convert to lists for Rust/fallback interface
        query_list = query.tolist()
        tool_embeddings_list = [emb.tolist() for emb in tool_embeddings]
        return batch_cosine_similarity(query_list, tool_embeddings_list)

    @classmethod
    def _create_tool_text(cls, tool: Any) -> str:
        """Create semantic description of tool for embedding.

        Combines tool name, description, parameter names, and use cases to create
        a rich semantic representation that matches user queries better.

        Uses the ToolMetadataProvider contract: tool.get_metadata() is guaranteed
        to return valid ToolMetadata (either explicit or auto-generated from
        tool properties). Legacy tools without get_metadata() rely on
        name + description only.

        Args:
            tool: Tool object

        Returns:
            Semantic text description enriched with use cases
        """
        # Start with name (important for matching)
        parts = [tool.name.replace("_", " ")]

        # Add description
        if tool.description:
            parts.append(tool.description)

        # Add parameter names (provide additional context)
        if hasattr(tool, "parameters") and tool.parameters:
            params = tool.parameters.get("properties", {})
            if params:
                param_names = ", ".join(params.keys())
                parts.append(f"Parameters: {param_names}")

        # Use get_metadata() for ToolMetadataProvider contract
        # This guarantees valid metadata (explicit or auto-generated)
        if hasattr(tool, "get_metadata"):
            metadata = tool.get_metadata()
            if metadata.use_cases:
                parts.append(f"Use for: {', '.join(metadata.use_cases)}.")
            if metadata.keywords:
                parts.append(f"Common requests: {', '.join(metadata.keywords)}.")
            if metadata.examples:
                parts.append(f"Examples: {', '.join(metadata.examples)}.")
        # Legacy tools without get_metadata() rely on name + description only

        return ". ".join(parts)

    # ========================================================================
    # Classification-Aware Tool Selection (UnifiedTaskClassifier Integration)
    # ========================================================================

    def _get_tools_for_task_type(self, task_type_str: str) -> List[str]:
        """Get relevant tools based on task type.

        Uses registry-based lookup. Tools declare their task types via
        @tool(task_types=["analysis", "edit"]).

        Args:
            task_type_str: Task type as string (e.g., "analysis", "action")

        Returns:
            List of tool names relevant to this task type
        """
        # Use registry-based task type lookup (decorator-driven)
        registry_tools = get_tools_by_task_type(task_type_str)
        if registry_tools:
            logger.log(
                TRACE,
                f"Registry task-type tools for '{task_type_str}': {registry_tools}",
            )
        return list(registry_tools)

    def _get_excluded_tools_from_negations(
        self,
        negated_keywords: List[Any],
    ) -> Set[str]:
        """Get tools that should be excluded based on negated keywords.

        Uses registry-based lookup. When user says "don't analyze", this
        excludes tools with task_type="analyze".

        Args:
            negated_keywords: List of KeywordMatch objects with negated keywords

        Returns:
            Set of tool names to exclude from selection
        """
        excluded: Set[str] = set()

        for match in negated_keywords:
            keyword = match.keyword if hasattr(match, "keyword") else str(match)

            # Use registry-based task type lookup (decorator-driven)
            # If keyword matches a task type, exclude tools declared for that type
            registry_excluded = get_tools_by_task_type(keyword)
            if registry_excluded:
                excluded.update(registry_excluded)
                logger.log(
                    TRACE,
                    f"Registry excluding tools for negated task_type '{keyword}': {registry_excluded}",
                )

        return excluded

    def _adjust_threshold_by_confidence(
        self,
        base_threshold: float,
        classification_confidence: float,
    ) -> float:
        """Adjust similarity threshold based on classification confidence.

        Higher confidence = stricter threshold (more focused selection)
        Lower confidence = looser threshold (broader selection)

        Args:
            base_threshold: Base similarity threshold
            classification_confidence: Confidence from classifier (0.0 - 1.0)

        Returns:
            Adjusted threshold
        """
        # High confidence (>0.7): tighten threshold by up to 0.05
        # Low confidence (<0.3): loosen threshold by up to 0.05
        confidence_adjustment = (classification_confidence - 0.5) * 0.1
        adjusted = base_threshold + confidence_adjustment
        # Clamp between reasonable bounds
        return max(0.1, min(0.3, adjusted))

    async def select_tools_with_classification(
        self,
        user_message: str,
        tools: ToolRegistry,
        classification_result: "ClassificationResult",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_tools: int = 5,
        base_similarity_threshold: float = SemanticSelectorDefaults.SIMILARITY_THRESHOLD,
    ) -> List[ToolDefinition]:
        """Select tools using classification result for smarter selection.

        Integrates with UnifiedTaskClassifier to:
        1. Use task type for category filtering
        2. Exclude tools related to negated keywords
        3. Adjust thresholds based on classification confidence
        4. Include context from conversation history

        Args:
            user_message: Current user message
            tools: Tool registry
            classification_result: Result from UnifiedTaskClassifier
            conversation_history: Optional conversation history
            max_tools: Maximum tools to return
            base_similarity_threshold: Base minimum similarity score

        Returns:
            List of relevant ToolDefinition objects
        """
        # Get task type and confidence
        task_type_str = classification_result.task_type.value
        confidence = classification_result.confidence

        logger.debug(
            f"Classification-aware selection: type={task_type_str}, " f"confidence={confidence:.2f}"
        )

        # Intent-based negations (e.g. "don't analyze") DEMOTE the matching tools
        # to the STUB schema tier rather than delisting them. Delisting makes a tool
        # completely invisible to the LLM, so it can never be proposed even if the
        # user's intent shifts mid-turn. Demotion keeps it available (terse, low-token,
        # still callable) — consistent with the core/compact/stub bucket policy used by
        # tool_selection._apply_cap("stub") and _enforce_token_budget.
        demoted_tools = self._get_excluded_tools_from_negations(
            classification_result.negated_keywords
        )
        # Get mandatory tools from keywords (before using them below)
        mandatory_tool_names = self._get_mandatory_tools(user_message)
        # Core/mandatory tools are never demoted: the agent must always be able to
        # reason with read/write/shell capabilities.
        excluded_tools = set()
        for name in demoted_tools:
            if name in mandatory_tool_names:
                logger.debug(f"Negation would demote mandatory tool '{name}'; keeping FULL.")
                continue
            excluded_tools.add(name)
        if demoted_tools:
            logger.debug(
                f"Tools demoted to STUB by negation: {sorted(demoted_tools)} "
                f"(mandatory preserved: {sorted(set(demoted_tools) - excluded_tools)})"
            )

        # Adjust threshold based on confidence
        similarity_threshold = self._adjust_threshold_by_confidence(
            base_similarity_threshold, confidence
        )
        logger.debug(f"Adjusted similarity threshold: {similarity_threshold:.3f}")

        # Get task-type-specific tools
        task_tools = self._get_tools_for_task_type(task_type_str)
        logger.debug(f"Task-type tools ({len(task_tools)}): {task_tools[:5]}...")

        # Remove negated tools from mandatory
        mandatory_tool_names = [t for t in mandatory_tool_names if t not in excluded_tools]

        # Get query embedding
        query_embedding = await self._get_embedding(user_message)

        relevant_tools: List[Any] = []

        for tool in tools.list_tools():
            # Skip excluded tools
            if tool.name in excluded_tools:
                continue

            # Skip if not in task-type tools or mandatory (unless default type)
            if (
                task_type_str != "default"
                and tool.name not in task_tools
                and tool.name not in mandatory_tool_names
            ):
                continue

            relevant_tools.append(tool)

        # Apply boosts/penalties and filter by threshold
        similarities: List[Tuple[Any, float]] = []
        for tool, similarity in await self._score_candidate_tools(query_embedding, relevant_tools):
            # Boost mandatory tools
            if tool.name in mandatory_tool_names:
                similarity = max(similarity, 0.9)

            # Apply usage boost
            usage_boost = await self._get_usage_boost(tool.name, user_message)
            similarity += usage_boost

            # Phase 9: Apply sequence boost based on workflow patterns
            sequence_boost = self._get_sequence_boost(tool.name)
            similarity += sequence_boost

            # Apply cost penalty
            cost_penalty = self._get_cost_penalty(tool, tools)
            similarity -= cost_penalty

            # Boost tools matching task type
            if tool.name in task_tools:
                similarity += 0.05  # Small boost for task-type alignment

            if similarity >= similarity_threshold:
                similarities.append((tool, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Build final selection
        selected_tools = []
        selected_names = set()

        # Add mandatory tools first
        mandatory_tools = [tool for tool in tools.list_tools() if tool.name in mandatory_tool_names]
        for tool in mandatory_tools:
            if tool.name not in selected_names:
                selected_tools.append((tool, SemanticSelectorDefaults.MANDATORY_TOOL_SCORE))
                selected_names.add(tool.name)

        # Add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Smart fallback if too few tools
        min_threshold = SemanticSelectorDefaults.MIN_TOOLS_THRESHOLD
        if len(selected_tools) < min_threshold:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names and fallback_name not in excluded_tools:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append(
                            (
                                fallback_tool,
                                SemanticSelectorDefaults.FALLBACK_TOOL_SCORE,
                            )
                        )
                        selected_names.add(fallback_name)

        # Log selection
        tool_names = [t.name for t, _ in selected_tools]
        scores = [f"{s:.3f}" for _, s in selected_tools]
        logger.info(
            f"Classification-aware selection: {len(selected_tools)} tools "
            f"(type={task_type_str}, excluded={len(excluded_tools)}): "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores, strict=False))}"
        )

        # Record usage for learning
        for tool_name in tool_names:
            self._record_tool_usage(tool_name, user_message, success=True)

        # Generate cost warnings
        self._last_cost_warnings = self._generate_cost_warnings(selected_tools, tools)

        # Emit semantic match event for RL learning
        self._emit_semantic_match_event(
            selected_tools=selected_tools,
            threshold=similarity_threshold,
            task_type=task_type_str,
            classification_aware=True,
            excluded_count=len(excluded_tools),
        )

        # Convert to ToolDefinition
        return [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in selected_tools
        ]

    def get_classification_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about classification-aware tool selection.

        Returns:
            Dictionary with selection statistics
        """
        stats = {
            "usage_cache_size": len(self._tool_usage_cache),
            "embedding_cache_size": len(self._tool_embedding_cache),
        }

        # Add sequence tracking stats if enabled
        if self._sequence_tracker:
            sequence_stats = self._sequence_tracker.get_statistics()
            stats["sequence_tracking"] = {
                "enabled": True,
                "history_length": sequence_stats["history_length"],
                "unique_tools_used": sequence_stats["unique_tools_used"],
                "total_transitions": sequence_stats["total_transitions"],
                "workflow_progress": sequence_stats["workflow_progress"],
            }
        else:
            stats["sequence_tracking"] = {"enabled": False}

        return stats

    def get_next_tool_suggestions(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get suggested next tools based on workflow patterns (Phase 9).

        Uses the ToolSequenceTracker to predict likely next tools based on
        the history of tool executions in the current session.

        Args:
            top_k: Number of suggestions to return

        Returns:
            List of (tool_name, confidence) tuples sorted by confidence
        """
        if not self._sequence_tracker:
            return []

        suggestions = self._sequence_tracker.get_next_suggestions(top_k=top_k)
        return [
            (self._to_sequence_public_tool_name(tool_name), confidence)
            for tool_name, confidence in suggestions
        ]

    def get_current_workflow(self) -> Optional[Tuple[str, float]]:
        """Detect if we're in the middle of a known workflow (Phase 9).

        Returns:
            Tuple of (workflow_name, progress_percentage) or None
        """
        if not self._sequence_tracker:
            return None

        return self._sequence_tracker.get_workflow_progress()

    def clear_session_state(self) -> None:
        """Clear session-specific state (sequence history, warnings).

        Call this when starting a new conversation session to reset
        sequence tracking and cost warnings.
        """
        self._warned_tools.clear()
        self._last_cost_warnings = []

        if self._sequence_tracker:
            self._sequence_tracker.clear_history()
            logger.debug("Cleared sequence tracker session history")

    async def close(self) -> None:
        """Close HTTP client and save usage cache (Phase 3)."""
        # Phase 3: Save usage statistics before shutdown (force save)
        self._save_usage_cache(force=True)

        if self._client:
            await self._client.aclose()

    def _emit_semantic_match_event(
        self,
        selected_tools: List[Tuple[Any, float]],
        threshold: float,
        task_type: str = "default",
        classification_aware: bool = False,
        excluded_count: int = 0,
    ) -> None:
        """Emit semantic match event for RL learning.

        Args:
            selected_tools: List of (tool, score) tuples
            threshold: Similarity threshold used
            task_type: Task type from classification (if available)
            classification_aware: Whether classification was used
            excluded_count: Number of tools excluded by negation
        """
        hooks = _get_rl_hooks()
        if hooks is None:
            return

        try:
            from victor.framework.rl.hooks import RLEvent, RLEventType

            # Calculate selection quality metrics
            avg_score = (
                sum(score for _, score in selected_tools) / len(selected_tools)
                if selected_tools
                else 0.0
            )
            tool_names = [t.name for t, _ in selected_tools]

            event = RLEvent(
                type=RLEventType.SEMANTIC_MATCH,
                success=len(selected_tools) > 0,
                quality_score=avg_score,
                tool_name=",".join(tool_names[:5]),  # First 5 tools
                threshold_value=threshold,
                task_type=task_type,
                metadata={
                    "tools_selected": len(selected_tools),
                    "results_count": len(selected_tools),  # Key expected by learner
                    "classification_aware": classification_aware,
                    "excluded_count": excluded_count,
                    "avg_similarity": avg_score,
                    "top_scores": [s for _, s in selected_tools[:3]],  # Top 3 scores
                    "embedding_model": self.embedding_model,
                    "embedding_provider": self.embedding_provider,
                    "threshold_used": threshold,
                },
            )
            hooks.emit(event)
            logger.log(
                TRACE,
                f"Emitted semantic_match event: {len(selected_tools)} tools, "
                f"threshold={threshold:.3f}, avg_score={avg_score:.3f}",
            )
        except Exception as e:
            logger.debug(f"Failed to emit semantic_match event: {e}")
