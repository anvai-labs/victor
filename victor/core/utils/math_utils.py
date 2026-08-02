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

"""Mathematical utility functions for evaluation and scoring.

Provides common numeric operations used throughout the codebase,
particularly in evaluation, feedback, and scoring modules.
"""

from __future__ import annotations

from typing import Any


def clamp(value: float, lower: float, upper: float) -> float:
    """Bound a numeric score into a closed interval.

    Args:
        value: The value to clamp
        lower: Lower bound (inclusive)
        upper: Upper bound (inclusive)

    Returns:
        The clamped value, guaranteed to be in [lower, upper]

    Examples:
        >>> clamp(5, 0, 10)
        5
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(15, 0, 10)
        10
    """
    return max(lower, min(upper, value))


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float, with default fallback.

    Args:
        value: Value to coerce
        default: Default value if coercion fails

    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int, with default fallback.

    Args:
        value: Value to coerce
        default: Default value if coercion fails

    Returns:
        Int value or default
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values.

    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0-1)

    Returns:
        Interpolated value

    Examples:
        >>> lerp(0, 10, 0.5)
        5.0
        >>> lerp(0, 10, 0.75)
        7.5
    """
    return a + (b - a) * t


def normalize_score(score: float, min_val: float, max_val: float) -> float:
    """Normalize a score to the 0-1 range.

    Args:
        score: Value to normalize
        min_val: Minimum possible value
        max_val: Maximum possible value

    Returns:
        Normalized score in [0, 1]
    """
    if max_val == min_val:
        return 0.5
    return clamp((score - min_val) / (max_val - min_val), 0.0, 1.0)
