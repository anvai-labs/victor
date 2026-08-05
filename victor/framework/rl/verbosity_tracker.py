# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Verbosity tracking for automated CONCISE_MODE_GUIDANCE evolution.

This module provides automated detection of verbose agent responses,
generating implicit feedback signals for PrefPO optimization without
requiring explicit user feedback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from victor.config.prompt_optimization_settings import VerbositySettings

logger = logging.getLogger(__name__)


class VerbosityTracker:
    """Track response verbosity to generate implicit feedback for PrefPO.

    Analyzes agent responses to detect verbosity patterns that should
    drive CONCISE_MODE_GUIDANCE evolution. Uses configurable thresholds
    for character count, line count, and verbosity ratio.
    """

    VERBOSITY_FAILURE_CATEGORY = "verbosity"

    def __init__(self, settings: Optional[VerbositySettings] = None):
        """Initialize the verbosity tracker.

        Args:
            settings: Verbosity detection settings. Uses defaults if None.
        """
        self._settings = settings or VerbositySettings()

    def is_too_verbose(
        self,
        response: str,
        max_chars: Optional[int] = None,
        max_lines: Optional[int] = None,
    ) -> bool:
        """Detect if response exceeds verbosity thresholds.

        Args:
            response: The agent response text to analyze.
            max_chars: Override max character threshold.
            max_lines: Override max line threshold.

        Returns:
            True if response is too verbose, False otherwise.
        """
        if not self._settings.enabled:
            return False

        max_chars = max_chars or self._settings.max_response_chars
        max_lines = max_lines or self._settings.max_response_lines

        char_count = len(response)
        line_count = len([line for line in response.strip().splitlines() if line.strip()])

        return char_count > max_chars or line_count > max_lines

    def get_verbosity_metrics(self, response: str) -> Dict[str, Any]:
        """Extract verbosity metrics from a response.

        Args:
            response: The agent response text to analyze.

        Returns:
            Dictionary with verbosity metrics.
        """
        lines = response.strip().splitlines()
        non_empty_lines = [line for line in lines if line.strip()]

        return {
            "char_count": len(response),
            "line_count": len(lines),
            "non_empty_line_count": len(non_empty_lines),
            "avg_line_length": sum(len(line) for line in non_empty_lines)
            / max(len(non_empty_lines), 1),
            "verbosity_ratio_chars": len(response) / self._settings.max_response_chars,
            "verbosity_ratio_lines": len(non_empty_lines) / self._settings.max_response_lines,
        }

    def generate_implicit_feedback(
        self,
        response: str,
        success: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Generate implicit feedback signal for PrefPO.

        Creates a feedback signal when response exceeds verbosity thresholds.
        This can be stored in tool_failures as a 'verbosity' category.

        Args:
            response: The agent response text to analyze.
            success: Whether the task completed successfully.

        Returns:
            Feedback dict if verbose, None otherwise.
        """
        if not self._settings.enabled or not success:
            return None

        if not self.is_too_verbose(response):
            return None

        metrics = self.get_verbosity_metrics(response)

        # Calculate feedback score based on verbosity severity
        verbosity_ratio = max(
            metrics["verbosity_ratio_chars"],
            metrics["verbosity_ratio_lines"],
        )
        severity = min((verbosity_ratio - 1.0) / 2.0, 1.0)  # Normalize to 0-1
        feedback_score = 0.3 + (severity * 0.4)  # 0.3 to 0.7 range

        return {
            "category": self.VERBOSITY_FAILURE_CATEGORY,
            "score": feedback_score,
            "signal": "response_too_long",
            "metrics": metrics,
            "weight": self._settings.auto_feedback_weight,
        }

    def inject_verbosity_failure(
        self,
        trace: Any,
        response: str,
    ) -> Any:
        """Inject verbosity failure into a trace for PrefPO processing.

        Modifies the trace's tool_failures dict to include verbosity as a
        failure category, allowing PrefPO to detect it during optimization.

        Args:
            trace: The trace object to modify (must have tool_failures dict).
            response: The agent response text to analyze.

        Returns:
            The modified trace (or original if no verbosity issue).
        """
        feedback = self.generate_implicit_feedback(response)

        if not feedback:
            return trace

        # Add to tool_failures for PrefPO to detect
        tool_failures = getattr(trace, "tool_failures", None)
        if tool_failures is None:
            logger.debug("Verbosity: trace has no tool_failures dict")
            return trace

        # Count as a failure with the feedback score
        tool_failures[self.VERBOSITY_FAILURE_CATEGORY] = str(feedback["score"])

        logger.debug(
            "Verbosity: injected failure (score=%.2f, chars=%d, lines=%d)",
            feedback["score"],
            feedback["metrics"]["char_count"],
            feedback["metrics"]["non_empty_line_count"],
        )

        return trace

    @staticmethod
    def get_verbosity_hint() -> str:
        """Get the failure hint for verbosity issues.

        This hint will be used by PrefPO when generating guidance items.
        """
        return (
            "Keep responses concise and direct. Avoid unnecessary preamble, "
            "summaries, or explanations. Skip 'I'll' and 'Let me' phrases. "
            "For code: show the code with minimal commentary. For questions: "
            "answer directly, then stop. Maximum 3 sentences for simple queries."
        )
