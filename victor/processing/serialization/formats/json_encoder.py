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

"""JSON format encoders for serialization.

Provides standard and minified JSON encoding with token efficiency metrics.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from victor.processing.serialization.formats.base import FormatEncoder, EncodingResult
from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    DataCharacteristics,
)

logger = logging.getLogger(__name__)


class JSONEncoder(FormatEncoder):
    """Standard JSON encoder with formatting.

    Baseline encoder with universal LLM compatibility.
    Uses 2-space indentation for readability.
    """

    format_id = SerializationFormat.JSON
    format_name = "JSON"
    format_description = "Standard JSON with 2-space indentation for readability."

    # Capabilities
    supports_nested = True
    supports_arrays = True
    supports_special_chars = True
    max_nesting_depth = 100

    # Token efficiency (baseline)
    base_efficiency = 1.0

    def can_encode(self, data: Any, characteristics: DataCharacteristics) -> bool:
        """JSON can encode any Python data structure."""
        try:
            # Quick check - can it be JSON serialized?
            json.dumps(data, default=str)
            return True
        except (TypeError, ValueError):
            return False

    def encode(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> EncodingResult:
        """Encode data to formatted JSON.

        Args:
            data: Data to encode
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            EncodingResult with JSON content
        """
        try:
            content = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            return EncodingResult(
                content=content,
                success=True,
                metadata={
                    "encoder": "json",
                    "char_count": len(content),
                    "estimated_tokens": self.estimate_tokens(content),
                },
            )
        except Exception as e:
            logger.warning(f"JSON encoding failed: {e}")
            return EncodingResult(
                content="",
                success=False,
                error=str(e),
            )

    def suitability_score(
        self,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> float:
        """JSON is always suitable but never optimal for token efficiency.

        Returns:
            Base suitability score of 0.3 (other formats should beat this)
        """
        # JSON is the baseline fallback
        # It's always compatible but rarely the most efficient
        return 0.3

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens for JSON content.

        JSON has high syntax overhead:
        - Quotes around keys and strings
        - Braces, brackets, colons, commas
        - Whitespace in formatted output

        Args:
            content: JSON string

        Returns:
            Estimated token count
        """
        # JSON has ~25% overhead from syntax
        # ~3.5 chars per token on average
        return int(len(content) / 3.5)

    def validate_output(self, content: str) -> bool:
        """Validate JSON is well-formed.

        Args:
            content: JSON string to validate

        Returns:
            True if valid JSON
        """
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False


class MinifiedJSONEncoder(FormatEncoder):
    """Minified JSON encoder without whitespace.

    Removes all unnecessary whitespace for 10-20% token savings.
    Ideal when readability is not needed.
    """

    format_id = SerializationFormat.JSON_MINIFIED
    format_name = "Minified JSON"
    format_description = "Compact JSON without whitespace for token efficiency."

    # Capabilities
    supports_nested = True
    supports_arrays = True
    supports_special_chars = True
    max_nesting_depth = 100

    # Token efficiency (10-20% better than JSON)
    base_efficiency = 0.85

    def can_encode(self, data: Any, characteristics: DataCharacteristics) -> bool:
        """Minified JSON can encode any Python data structure."""
        try:
            json.dumps(data, default=str)
            return True
        except (TypeError, ValueError):
            return False

    def encode(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> EncodingResult:
        """Encode data to minified JSON.

        Args:
            data: Data to encode
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            EncodingResult with minified JSON content
        """
        try:
            content = json.dumps(
                data,
                separators=(",", ":"),  # Minimal separators
                ensure_ascii=False,
                default=str,
            )
            return EncodingResult(
                content=content,
                success=True,
                metadata={
                    "encoder": "json_minified",
                    "char_count": len(content),
                    "estimated_tokens": self.estimate_tokens(content),
                },
            )
        except Exception as e:
            logger.warning(f"Minified JSON encoding failed: {e}")
            return EncodingResult(
                content="",
                success=False,
                error=str(e),
            )

    def suitability_score(
        self,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> float:
        """Score minified JSON based on data characteristics.

        Better than JSON for all data types, but tabular formats
        are more efficient for uniform arrays.

        Returns:
            Suitability score (0.4-0.5)
        """
        base_score = 0.45

        # Slight bonus for simple structures
        if characteristics.is_flat():
            base_score += 0.05

        # Penalty for deeply nested (where whitespace helps readability)
        if characteristics.nesting_depth > 3:
            base_score -= 0.05

        return min(0.6, max(0.3, base_score))

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens for minified JSON.

        Minified JSON is more token-dense.

        Args:
            content: Minified JSON string

        Returns:
            Estimated token count
        """
        # Denser than formatted JSON: ~3.8 chars per token
        return int(len(content) / 3.8)

    def validate_output(self, content: str) -> bool:
        """Validate minified JSON is well-formed.

        Args:
            content: JSON string to validate

        Returns:
            True if valid JSON
        """
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False
