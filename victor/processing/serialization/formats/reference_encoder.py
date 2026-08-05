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

"""Reference encoding for highly repetitive data.

Reference encoding achieves 40-70% token savings when data contains
many repeated values by:
- Creating a lookup table of common values
- Replacing repeated values with short reference keys
- Particularly effective for status fields, categories, etc.

Format:
    refs={
      $a: "repeated_value_1",
      $b: "repeated_value_2"
    }
    data=<encoded_data_with_$refs>

Example input:
    [{"status": "active", "type": "user"}, {"status": "active", "type": "admin"}]

Example output:
    refs={$a:"active",$b:"user",$c:"admin"}
    data=[{status:$a,type:$b},{status:$a,type:$c}]
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Tuple

from victor.processing.serialization.formats.base import FormatEncoder, EncodingResult
from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    DataCharacteristics,
)

logger = logging.getLogger(__name__)


class ReferenceEncoder(FormatEncoder):
    """Reference encoder for data with repeated values.

    Creates a lookup table of frequently occurring values and
    replaces them with short reference keys ($a, $b, etc.).
    """

    format_id = SerializationFormat.REFERENCE_ENCODED
    format_name = "Reference Encoded"
    format_description = (
        "Reference encoding: Common values defined in refs={} block, "
        "referenced by $key in data. Optimal for repetitive values."
    )

    # Capabilities
    supports_nested = True
    supports_arrays = True
    supports_special_chars = True
    max_nesting_depth = 100

    # Token efficiency depends on repetition
    base_efficiency = 0.6

    # Minimum occurrences to create a reference
    MIN_OCCURRENCES = 2

    # Minimum string length to consider for reference
    MIN_STRING_LENGTH = 4

    def can_encode(self, data: Any, characteristics: DataCharacteristics) -> bool:
        """Check if reference encoding is beneficial.

        Reference encoding only helps when there are repeated values.

        Args:
            data: Data to check
            characteristics: Data analysis results

        Returns:
            True if reference encoding would save tokens
        """
        # Need high repetition to benefit
        if not characteristics.has_high_repetition():
            return False

        # Need some repeated values
        if not characteristics.common_values:
            return False

        return True

    def encode(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> EncodingResult:
        """Encode data with reference replacement.

        Args:
            data: Data to encode
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            EncodingResult with reference-encoded content
        """
        try:
            # Build reference table from common values
            ref_table, reverse_table = self._build_reference_table(characteristics.common_values)

            if not ref_table:
                # No references worth creating, fall back to minified JSON
                content = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
                return EncodingResult(
                    content=content,
                    success=True,
                    metadata={"encoder": "reference_encoded", "refs_count": 0},
                )

            # Replace values with references in data
            encoded_data = self._replace_with_refs(data, reverse_table)

            # Build refs block
            refs_items = [f"${k}:{json.dumps(v)}" for k, v in ref_table.items()]
            refs_block = "refs={" + ",".join(refs_items) + "}"

            # Encode data with references
            data_json = json.dumps(encoded_data, separators=(",", ":"), ensure_ascii=False)
            data_block = f"data={data_json}"

            content = f"{refs_block}\n{data_block}"

            return EncodingResult(
                content=content,
                success=True,
                metadata={
                    "encoder": "reference_encoded",
                    "refs_count": len(ref_table),
                    "char_count": len(content),
                    "estimated_tokens": self.estimate_tokens(content),
                },
                reference_table=ref_table,
            )

        except Exception as e:
            logger.warning(f"Reference encoding failed: {e}")
            return EncodingResult(
                content="",
                success=False,
                error=str(e),
            )

    def _build_reference_table(
        self,
        common_values: Dict[str, int],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build reference table from common values.

        Args:
            common_values: Map of value -> occurrence count

        Returns:
            Tuple of (ref_key -> value, value -> ref_key)
        """
        ref_table: Dict[str, str] = {}
        reverse_table: Dict[str, str] = {}

        # Filter values worth referencing
        candidates = [
            (value, count)
            for value, count in common_values.items()
            if count >= self.MIN_OCCURRENCES and len(value) >= self.MIN_STRING_LENGTH
        ]

        # Sort by savings potential (count * length)
        candidates.sort(key=lambda x: x[1] * len(x[0]), reverse=True)

        # Generate reference keys (a, b, c, ... aa, ab, ...)
        ref_key = 0
        for value, _count in candidates:
            key = self._generate_ref_key(ref_key)
            ref_table[key] = value
            reverse_table[value] = f"${key}"
            ref_key += 1

        return ref_table, reverse_table

    def _generate_ref_key(self, index: int) -> str:
        """Generate a short reference key from index.

        0 -> 'a', 1 -> 'b', ..., 25 -> 'z', 26 -> 'aa', ...

        Args:
            index: Reference index

        Returns:
            Short reference key
        """
        result = ""
        while True:
            result = chr(ord("a") + (index % 26)) + result
            index = index // 26 - 1
            if index < 0:
                break
        return result

    def _replace_with_refs(
        self,
        data: Any,
        reverse_table: Dict[str, str],
    ) -> Any:
        """Recursively replace values with references.

        Args:
            data: Data to process
            reverse_table: Map of value -> reference key

        Returns:
            Data with values replaced by references
        """
        if isinstance(data, str):
            return reverse_table.get(data, data)
        elif isinstance(data, dict):
            return {k: self._replace_with_refs(v, reverse_table) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_with_refs(item, reverse_table) for item in data]
        else:
            return data

    def suitability_score(
        self,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> float:
        """Calculate reference encoding suitability.

        Best when:
        - High value repetition
        - Long repeated strings
        - Many occurrences of same values

        Args:
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            Suitability score (0.0-1.0)
        """
        if not config.enable_reference_encoding:
            return 0.0

        if characteristics.value_repetition_ratio < config.min_repetition_for_references:
            return 0.0

        base_score = 0.3

        # Scale with repetition ratio
        base_score += characteristics.value_repetition_ratio * 0.4

        # Bonus for many repeated values
        if len(characteristics.common_values) >= 10:
            base_score += 0.15
        elif len(characteristics.common_values) >= 5:
            base_score += 0.10

        return min(0.85, max(0.0, base_score))

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens for reference-encoded content.

        Reference encoding is efficient when refs are short:
        - ~4 chars per token

        Args:
            content: Reference-encoded string

        Returns:
            Estimated token count
        """
        return int(len(content) / 4)

    def validate_output(self, content: str) -> bool:
        """Validate reference-encoded format.

        Checks:
        - Has refs={} block
        - Has data= block

        Args:
            content: Reference-encoded string

        Returns:
            True if valid format
        """
        if not content:
            return False

        return "refs={" in content and "data=" in content
