"""Fuzzy matching configuration for robust classification.

This module contains settings for:
- Fuzzy matching enabled/disabled
- Minimum similarity ratio thresholds
- Per-classifier overrides
- Performance optimization settings
"""

from typing import Optional
from pydantic import BaseModel, field_validator


class FuzzyMatchingSettings(BaseModel):
    """Fuzzy matching settings for robust classification.

    Controls Levenshtein distance-based fuzzy matching to handle typos
    and spelling variations in classification systems.

    Benefits:
    - Improved robustness against typos
    - Better user experience (fewer "I don't understand" responses)
    - Enhanced safety (correct state machine transitions)
    - Reduced token overhead (less clarification needed)
    - Edge-driven optimizations (micro-decisions work with typos)
    """

    # ==========================================================================
    # Core Fuzzy Matching Configuration
    # ==========================================================================
    # Master switch for fuzzy matching across all classification systems
    enabled: bool = True

    # Minimum similarity ratio for fuzzy matches [0.5-0.95]
    # Levenshtein ratio: 1.0 = identical, 0.0 = completely different
    # Default 0.75 balances:
    # - False positives (too low) vs
    # - False negatives (too high)
    min_similarity_ratio: float = 0.75

    # Use LRU caching for repeated queries
    # Improves performance by ~10x for repeated queries
    use_caching: bool = True

    # Cache size for fuzzy matching results
    # Default 1024 provides good memory/performance tradeoff
    cache_size: int = 1024

    # ==========================================================================
    # Per-Classifier Overrides
    # ==========================================================================
    # Enable/disable fuzzy matching for specific classifiers
    # Useful for A/B testing or debugging

    # Task classification (TaskTypeClassifier, UnifiedTaskClassifier)
    task_classifier_enabled: bool = True

    # Intent classification (IntentClassifier)
    intent_classifier_enabled: bool = True

    # Tool selection (SemanticToolSelector)
    tool_selector_enabled: bool = True

    # Edge model decisions (FEP-0001)
    edge_model_enabled: bool = True

    # ==========================================================================
    # Advanced Configuration
    # ==========================================================================
    # Adaptive edit distance thresholds
    # Thresholds based on word length:
    # - 1-3 chars: 0 tolerance (too short, typos change meaning)
    # - 4-6 chars: 1 edit (e.g., "analyze" → "analize")
    # - 7-9 chars: 2 edits (e.g., "structure" → "structre")
    # - 10+ chars: max(2, length//4) (~25% tolerance)
    use_adaptive_thresholds: bool = True

    # Custom minimum similarity ratio per classifier
    # Overrides min_similarity_ratio for specific classifiers
    task_classifier_min_similarity: Optional[float] = None  # Use default
    intent_classifier_min_similarity: Optional[float] = None  # Use default
    tool_selector_min_similarity: Optional[float] = None  # Use default
    edge_model_min_similarity: Optional[float] = None  # Use default

    # ==========================================================================
    # Fallback Behavior
    # ==========================================================================
    # Fallback to exact matching if fuzzy matching fails
    fallback_to_exact: bool = True

    # Log fuzzy match statistics for debugging
    log_fuzzy_matches: bool = False

    @field_validator("min_similarity_ratio")
    @classmethod
    def validate_min_similarity_ratio(cls, v: float) -> float:
        """Validate minimum similarity ratio is in valid range.

        Args:
            v: Minimum similarity ratio

        Returns:
            Validated similarity ratio

        Raises:
            ValueError: If ratio is out of valid range
        """
        if not 0.5 <= v <= 0.95:
            raise ValueError(f"min_similarity_ratio must be between 0.5 and 0.95, got {v}")
        return v

    @field_validator("cache_size")
    @classmethod
    def validate_cache_size(cls, v: int) -> int:
        """Validate cache size is positive.

        Args:
            v: Cache size

        Returns:
            Validated cache size

        Raises:
            ValueError: If cache size is not positive
        """
        if v < 0:
            raise ValueError(f"cache_size must be non-negative, got {v}")
        return v

    def get_effective_similarity_ratio(self, classifier_type: str) -> float:
        """Get effective similarity ratio for a classifier.

        Args:
            classifier_type: Type of classifier
                ("task", "intent", "tool", "edge")

        Returns:
            Effective similarity ratio for this classifier
        """
        # Check for classifier-specific override
        if classifier_type == "task" and self.task_classifier_min_similarity is not None:
            return self.task_classifier_min_similarity
        elif classifier_type == "intent" and self.intent_classifier_min_similarity is not None:
            return self.intent_classifier_min_similarity
        elif classifier_type == "tool" and self.tool_selector_min_similarity is not None:
            return self.tool_selector_min_similarity
        elif classifier_type == "edge" and self.edge_model_min_similarity is not None:
            return self.edge_model_min_similarity

        # Use default
        return self.min_similarity_ratio

    def is_classifier_enabled(self, classifier_type: str) -> bool:
        """Check if fuzzy matching is enabled for a classifier.

        Args:
            classifier_type: Type of classifier
                ("task", "intent", "tool", "edge")

        Returns:
            True if fuzzy matching is enabled for this classifier
        """
        # Master switch
        if not self.enabled:
            return False

        # Per-classifier switch
        if classifier_type == "task":
            return self.task_classifier_enabled
        elif classifier_type == "intent":
            return self.intent_classifier_enabled
        elif classifier_type == "tool":
            return self.tool_selector_enabled
        elif classifier_type == "edge":
            return self.edge_model_enabled

        return False
