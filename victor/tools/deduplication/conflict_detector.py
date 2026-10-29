"""Conflict detection strategies for tool deduplication.

This module provides various strategies for detecting tool conflicts:
- Exact name matching
- Semantic similarity (embedding-based)
- Capability overlap (keyword + parameter schema)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    """Type of conflict detected between tools."""

    EXACT_NAME = "exact_name"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CAPABILITY_OVERLAP = "capability_overlap"


class ConflictResult(BaseModel):
    """Result of conflict detection between two tools."""

    is_conflict: bool
    conflict_type: Optional[ConflictType] = None
    confidence: float = 0.0
    reason: str = ""


class ConflictDetector:
    """Detects conflicts between tools using multiple strategies.

    Usage:
        detector = ConflictDetector(threshold=0.85)
        result = detector.are_tools_conflicting(tool1, tool2)
    """

    def __init__(
        self,
        semantic_threshold: float = 0.85,
        enable_semantic: bool = False,
        enable_capability: bool = True,
    ) -> None:
        """Initialize conflict detector.

        Args:
            semantic_threshold: Threshold for semantic similarity (0.0-1.0)
            enable_semantic: Enable embedding-based semantic detection
            enable_capability: Enable capability overlap detection
        """
        self._semantic_threshold = semantic_threshold
        self._enable_semantic = enable_semantic
        self._enable_capability = enable_capability

        # Keyword sets for capability detection
        self._search_keywords = {
            "search",
            "find",
            "lookup",
            "query",
            "grep",
            "locate",
        }
        self._fetch_keywords = {"fetch", "get", "request", "http", "curl", "download"}
        self._shell_keywords = {"shell", "terminal", "command", "exec", "run", "bash"}
        self._file_keywords = {"file", "read", "write", "edit", "modify", "create"}
        self._web_keywords = {"web", "internet", "online", "url", "http", "https"}

    def are_tools_conflicting(self, tool1: Any, tool2: Any) -> ConflictResult:
        """Check if two tools are conflicting using all enabled strategies.

        Args:
            tool1: First tool
            tool2: Second tool

        Returns:
            ConflictResult with conflict status and details
        """
        # Exact name matching (always enabled)
        if self._exact_name_match(tool1, tool2):
            return ConflictResult(
                is_conflict=True,
                conflict_type=ConflictType.EXACT_NAME,
                confidence=1.0,
                reason="Exact name match after normalization",
            )

        # Semantic similarity (if enabled)
        if self._enable_semantic:
            semantic_result = self._semantic_similarity(tool1, tool2)
            if semantic_result.is_conflict:
                return semantic_result

        # Capability overlap (if enabled)
        if self._enable_capability:
            capability_result = self._capability_overlap(tool1, tool2)
            if capability_result.is_conflict:
                return capability_result

        return ConflictResult(is_conflict=False, confidence=0.0, reason="No conflict detected")

    def _exact_name_match(self, tool1: Any, tool2: Any) -> bool:
        """Check if tools have exact same normalized name."""
        name1 = self._normalize_tool_name(tool1)
        name2 = self._normalize_tool_name(tool2)
        return name1 == name2

    def _semantic_similarity(self, tool1: Any, tool2: Any) -> ConflictResult:
        """Check semantic similarity using embeddings (if available)."""
        try:
            # Try to use sentence-transformers if available
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("BAAI/bge-small-en-v1.5")

            desc1 = self._get_tool_description(tool1)
            desc2 = self._get_tool_description(tool2)

            # Encode descriptions
            emb1 = model.encode(desc1)
            emb2 = model.encode(desc2)

            # Compute cosine similarity
            import numpy as np

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            if similarity >= self._semantic_threshold:
                return ConflictResult(
                    is_conflict=True,
                    conflict_type=ConflictType.SEMANTIC_SIMILARITY,
                    confidence=similarity,
                    reason=f"Semantic similarity {similarity:.2f} >= {self._semantic_threshold}",
                )

        except ImportError:
            logger.debug("sentence-transformers not available, skipping semantic detection")
        except Exception as e:
            logger.warning(f"Semantic similarity detection failed: {e}")

        return ConflictResult(
            is_conflict=False,
            confidence=0.0,
            reason="Semantic similarity below threshold",
        )

    def _capability_overlap(self, tool1: Any, tool2: Any) -> ConflictResult:
        """Check capability overlap based on keywords and parameters."""
        name1 = self._get_tool_name(tool1).lower()
        name2 = self._get_tool_name(tool2).lower()
        desc1 = self._get_tool_description(tool1).lower()
        desc2 = self._get_tool_description(tool2).lower()

        # Combine name and description for analysis
        text1 = f"{name1} {desc1}"
        text2 = f"{name2} {desc2}"

        # Check for overlapping capability keywords
        for keyword_set, capability_name in [
            (self._search_keywords, "search"),
            (self._fetch_keywords, "fetch"),
            (self._shell_keywords, "shell"),
            (self._file_keywords, "file"),
            (self._web_keywords, "web"),
        ]:
            overlap1 = any(keyword in text1 for keyword in keyword_set)
            overlap2 = any(keyword in text2 for keyword in keyword_set)

            if overlap1 and overlap2:
                # Check parameter similarity
                param_similarity = self._parameter_similarity(tool1, tool2)

                if param_similarity > 0.5:
                    return ConflictResult(
                        is_conflict=True,
                        conflict_type=ConflictType.CAPABILITY_OVERLAP,
                        confidence=param_similarity,
                        reason=f"Both tools appear to handle {capability_name} with similar parameters",
                    )

        return ConflictResult(
            is_conflict=False,
            confidence=0.0,
            reason="No significant capability overlap",
        )

    def _parameter_similarity(self, tool1: Any, tool2: Any) -> float:
        """Calculate parameter similarity between two tools (0.0-1.0)."""
        try:
            params1 = self._get_tool_parameters(tool1)
            params2 = self._get_tool_parameters(tool2)

            if not params1 or not params2:
                return 0.0

            # Get parameter names
            names1 = set(params1.get("properties", {}).keys())
            names2 = set(params2.get("properties", {}).keys())

            if not names1 or not names2:
                return 0.0

            # Jaccard similarity
            intersection = len(names1 & names2)
            union = len(names1 | names2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _get_tool_name(self, tool: Any) -> str:
        """Extract tool name."""
        if hasattr(tool, "name"):
            return tool.name
        elif hasattr(tool, "__name__"):
            return tool.__name__
        return str(tool)

    def _get_tool_description(self, tool: Any) -> str:
        """Extract tool description."""
        if hasattr(tool, "description"):
            return tool.description or ""
        return ""

    def _get_tool_parameters(self, tool: Any) -> Dict[str, Any]:
        """Extract tool parameters schema."""
        if hasattr(tool, "parameters"):
            return tool.parameters or {}
        elif hasattr(tool, "args_schema"):
            return tool.args_schema or {}
        return {}

    def _normalize_tool_name(self, tool: Any) -> str:
        """Normalize tool name for comparison."""
        name = self._get_tool_name(tool).lower()
        # Remove source prefixes
        for prefix in ["lgc_", "langchain_", "mcp_", "plg_", "plugin_"]:
            if name.startswith(prefix):
                name = name[len(prefix) :]
        # Normalize separators
        name = name.replace("_", " ").replace("-", " ")
        return " ".join(name.split())
