"""Tool selection and deduplication configuration.

This module contains settings for:
- Semantic tool selection
- Tool call deduplication
- Tool selection fallback
"""

from pydantic import BaseModel, field_validator


class ToolSelectionSettings(BaseModel):
    """Tool selection and deduplication settings.

    Controls how tools are selected, deduplicated, and managed
    during agent execution.
    """

    # ==========================================================================
    # Tool Selection Strategy
    # ==========================================================================
    # Enable semantic tool selection using embeddings instead of keyword matching.
    # Provides better tool relevance for complex queries.
    use_semantic_tool_selection: bool = True  # Use embeddings instead of keywords (DEFAULT)

    # Defer embedding model load until first semantic query to reduce startup time.
    preload_embeddings: bool = False

    # ==========================================================================
    # Tool Call Deduplication
    # ==========================================================================
    # Enable deduplication tracker to prevent redundant tool calls with identical parameters.
    # Tracks recent tool calls and skips duplicates within a time window.
    enable_tool_deduplication: bool = True

    # Number of recent tool calls to track for deduplication detection.
    # Larger window = better duplicate detection but more memory usage.
    tool_deduplication_window_size: int = 20

    # ==========================================================================
    # Tool Selection Fallback
    # ==========================================================================
    # Cap tool list when stage pruning removes everything.
    # Ensures agent always has some tools available even after aggressive filtering.
    fallback_max_tools: int = 8

    @field_validator("tool_deduplication_window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        """Validate deduplication window size is positive."""
        if v < 1:
            raise ValueError("tool_deduplication_window_size must be >= 1")
        if v > 100:
            raise ValueError("tool_deduplication_window_size must be <= 100 (unreasonably large)")
        return v

    @field_validator("fallback_max_tools")
    @classmethod
    def validate_fallback_max_tools(cls, v: int) -> int:
        """Validate fallback max tools is positive."""
        if v < 1:
            raise ValueError("fallback_max_tools must be >= 1")
        return v
