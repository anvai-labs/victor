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

"""TOON (Token-Optimized Object Notation) production support.

This module provides production-ready TOON format support for structured
data serialization, achieving 30-60% token reduction over JSON.

Features:
- Automatic TOON format selection based on data type
- Configurable thresholds for when to use TOON
- Validation and fallback to JSON on errors
- Provider-specific TOON preferences

Usage:
    from victor.agent.toon_production import ToonFormatter, ToonConfig

    # Enable TOON for production
    config = ToonConfig(
        enabled=True,
        min_structured_items=5,  # Use TOON for lists/dicts with 5+ items
        token_savings_threshold=0.20,  # Use TOON if saves >20% tokens
    )

    formatter = ToonFormatter(config)
    result = formatter.format([{"name": "Alice", "age": 30}, ...])
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ToonConfig:
    """Configuration for TOON formatter in production.

    Attributes:
        enabled: Whether TOON formatting is enabled
        min_structured_items: Minimum items for TOON on structured data (default 5)
        token_savings_threshold: Minimum token savings to use TOON (0.0-1.0)
        max_nesting_depth: Maximum nesting depth before falling back to JSON (default 3)
        fallback_on_error: Whether to fall back to JSON on TOON errors (default True)
        validate_output: Whether to validate TOON output can be parsed back (default False)
        provider_whitelist: List of providers allowed to use TOON (empty = all)
    """

    enabled: bool = False
    min_structured_items: int = 5
    token_savings_threshold: float = 0.20  # 20% savings threshold
    max_nesting_depth: int = 3
    fallback_on_error: bool = True
    validate_output: bool = False
    provider_whitelist: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.token_savings_threshold <= 1.0:
            raise ValueError("token_savings_threshold must be between 0.0 and 1.0")

        if self.min_structured_items < 2:
            raise ValueError("min_structured_items must be at least 2")

        if self.max_nesting_depth < 1:
            raise ValueError("max_nesting_depth must be at least 1")

    def should_use_toon(
        self,
        data: Any,
        provider_name: Optional[str] = None,
    ) -> bool:
        """Determine if TOON should be used for this data.

        Args:
            data: Data to format
            provider_name: Optional provider name for whitelist check

        Returns:
            True if TOON should be used
        """
        if not self.enabled:
            return False

        # Check provider whitelist
        if self.provider_whitelist and provider_name:
            if provider_name not in self.provider_whitelist:
                return False

        # Only use TOON for structured data
        if not isinstance(data, (list, dict)):
            return False

        # Check minimum size threshold
        item_count = len(data) if isinstance(data, (list, dict)) else 0
        if item_count < self.min_structured_items:
            return False

        # Check nesting depth
        depth = self._calculate_nesting_depth(data)
        if depth > self.max_nesting_depth:
            return False

        return True

    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of data structure.

        Args:
            data: Data to analyze
            current_depth: Current depth (for recursion)

        Returns:
            Maximum nesting depth
        """
        if not isinstance(data, (list, dict)):
            return current_depth

        if current_depth >= self.max_nesting_depth:
            return current_depth

        max_child_depth = current_depth

        if isinstance(data, list):
            for item in data[:10]:  # Limit check to first 10 items
                child_depth = self._calculate_nesting_depth(item, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        elif isinstance(data, dict):
            for value in list(data.values())[:10]:  # Limit check to first 10 values
                child_depth = self._calculate_nesting_depth(value, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth


class ToonFormatter:
    """Production-ready TOON formatter with validation and fallback.

    This formatter provides safe TOON serialization with automatic
    fallback to JSON on errors or when TOON doesn't provide sufficient
    token savings.

    Usage:
        config = ToonConfig(enabled=True, min_structured_items=5)
        formatter = ToonFormatter(config)

        # Format data
        result = formatter.format(
            data=[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
            provider_name="openai",
        )
    """

    def __init__(self, config: Optional[ToonConfig] = None):
        """Initialize TOON formatter.

        Args:
            config: TOON configuration (uses defaults if None)
        """
        self.config = config or ToonConfig()

    def format(
        self,
        data: Any,
        provider_name: Optional[str] = None,
    ) -> str:
        """Format data using TOON or JSON based on configuration.

        Args:
            data: Data to format
            provider_name: Optional provider name for whitelist check

        Returns:
            Formatted string (TOON or JSON)
        """
        # Check if TOON should be used
        if not self.config.should_use_toon(data, provider_name):
            return self._format_json(data)

        # Try TOON formatting
        try:
            toon_result = self._format_toon(data)

            # Calculate token savings
            json_result = self._format_json(data)
            json_tokens = self._estimate_tokens(json_result)
            toon_tokens = self._estimate_tokens(toon_result)

            savings = (json_tokens - toon_tokens) / json_tokens if json_tokens > 0 else 0

            # Only use TOON if savings meet threshold
            if savings >= self.config.token_savings_threshold:
                logger.debug(
                    f"Using TOON format: {savings:.1%} token savings "
                    f"({toon_tokens} vs {json_tokens} tokens)"
                )

                # Validate output if requested
                if self.config.validate_output:
                    if not self._validate_toon(toon_result):
                        logger.warning("TOON validation failed, falling back to JSON")
                        return json_result

                return toon_result
            else:
                logger.debug(
                    f"TOON savings ({savings:.1%}) below threshold "
                    f"({self.config.token_savings_threshold:.1%}), using JSON"
                )
                return json_result

        except Exception as e:
            if self.config.fallback_on_error:
                logger.warning(f"TOON formatting failed: {e}, falling back to JSON")
                return self._format_json(data)
            raise

    def _format_toon(self, data: Any) -> str:
        """Format data as TOON (compact notation).

        Args:
            data: Data to format

        Returns:
            TOON-formatted string
        """
        return self._toon_serialize(data)

    def _toon_serialize(self, data: Any, indent: int = 0) -> str:
        """Serialize data to TOON format.

        TOON rules:
        - Lists: Use | delimiter between items
        - Dicts: Use : delimiter, one key-value per line
        - Strings: Quoted if contain special characters
        - Numbers: Unquoted
        - Indentation: 2 spaces per nesting level
        """
        if isinstance(data, str):
            # Quote if contains special characters
            if any(c in data for c in ["|", ":", "\n", '"']):
                return f'"{data}"'
            return data

        if isinstance(data, (int, float, bool)) or data is None:
            return str(data)

        if isinstance(data, list):
            if not data:
                return "[]"

            # Simple list: single line with | delimiter
            if all(not isinstance(item, (list, dict)) for item in data):
                items = [self._toon_serialize(item, indent) for item in data]
                return " | ".join(items)

            # Complex list: multi-line
            lines = []
            for item in data:
                serialized = self._toon_serialize(item, indent + 1)
                lines.append("  " * indent + serialized)
            return "\n".join(lines)

        if isinstance(data, dict):
            if not data:
                return "{}"

            lines = []
            for key, value in data.items():
                key_str = str(key)
                value_str = self._toon_serialize(value, indent + 1)
                lines.append("  " * indent + f"{key_str}: {value_str}")
            return "\n".join(lines)

        return str(data)

    def _format_json(self, data: Any) -> str:
        """Format data as JSON (fallback).

        Args:
            data: Data to format

        Returns:
            JSON-formatted string
        """
        try:
            return json_dumps(data, ensure_ascii=False, default=str)
        except Exception:
            return str(data)

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars).

        Args:
            content: Content string

        Returns:
            Estimated token count
        """
        return len(content) // 4

    def _validate_toon(self, toon_output: str) -> bool:
        """Validate TOON output can be reasonably parsed.

        This is a basic validation - doesn't fully parse TOON but checks
        for common issues like unbalanced delimiters.

        Args:
            toon_output: TOON-formatted string

        Returns:
            True if output appears valid
        """
        # Basic checks
        if not toon_output or not toon_output.strip():
            return False

        # Check for unbalanced quotes
        quote_count = toon_output.count('"')
        if quote_count % 2 != 0:
            return False

        # Check for obvious formatting issues
        lines = toon_output.split("\n")
        for line in lines:
            # Each line should have proper structure
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                    return False

        return True


def create_toon_formatter(
    enabled: bool = False,
    min_structured_items: int = 5,
    token_savings_threshold: float = 0.20,
    provider_whitelist: Optional[List[str]] = None,
) -> ToonFormatter:
    """Factory function to create a configured ToonFormatter.

    Args:
        enabled: Whether TOON is enabled
        min_structured_items: Minimum items for TOON
        token_savings_threshold: Minimum token savings threshold (0.0-1.0)
        provider_whitelist: List of providers allowed to use TOON

    Returns:
        Configured ToonFormatter instance

    Example:
        # Enable TOON for cloud providers only
        formatter = create_toon_formatter(
            enabled=True,
            provider_whitelist=["openai", "anthropic", "xai"],
        )
    """
    config = ToonConfig(
        enabled=enabled,
        min_structured_items=min_structured_items,
        token_savings_threshold=token_savings_threshold,
        provider_whitelist=provider_whitelist or [],
    )

    return ToonFormatter(config)
