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

"""Data structure analyzer for optimal serialization format selection.

Analyzes data to determine:
- Structure type (uniform array, nested object, flat, etc.)
- Size metrics (elements, nesting depth, unique keys)
- Content characteristics (special chars, nulls)
- Repetition patterns (for reference encoding)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from victor.processing.serialization.strategy import (
    DataCharacteristics,
    DataStructureType,
)

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Analyzes data structures for optimal serialization.

    Provides comprehensive analysis of data to enable intelligent
    format selection by the adaptive serializer.
    """

    # Characters that require escaping in most formats
    SPECIAL_CHARS = set(',"\n\r\t\\|{}[]')

    # Minimum sample size for uniformity analysis
    UNIFORMITY_SAMPLE_SIZE = 100

    def analyze(self, data: Any) -> DataCharacteristics:
        """Analyze data and return characteristics.

        Args:
            data: Any Python data structure

        Returns:
            DataCharacteristics with analysis results
        """
        characteristics = DataCharacteristics()

        if data is None:
            characteristics.structure_type = DataStructureType.EMPTY
            return characteristics

        if isinstance(data, str):
            return self._analyze_string(data)
        elif isinstance(data, (int, float, bool)):
            return self._analyze_primitive(data)
        elif isinstance(data, list):
            return self._analyze_list(data)
        elif isinstance(data, dict):
            return self._analyze_dict(data)
        else:
            # Unknown type - try JSON serialization to estimate size
            characteristics.structure_type = DataStructureType.MIXED
            try:
                json_str = json.dumps(data, default=str)
                characteristics.estimated_json_chars = len(json_str)
                characteristics.estimated_json_tokens = len(json_str) // 4
            except (TypeError, ValueError):
                pass
            return characteristics

    def _analyze_string(self, data: str) -> DataCharacteristics:
        """Analyze a string value.

        Args:
            data: String to analyze

        Returns:
            DataCharacteristics for string
        """
        chars = DataCharacteristics()
        chars.structure_type = DataStructureType.TEXT
        chars.total_elements = 1
        chars.has_special_chars = any(c in self.SPECIAL_CHARS for c in data)
        chars.estimated_json_chars = len(json.dumps(data))
        chars.estimated_json_tokens = chars.estimated_json_chars // 4
        return chars

    def _analyze_primitive(self, data: Any) -> DataCharacteristics:
        """Analyze a primitive value (int, float, bool).

        Args:
            data: Primitive to analyze

        Returns:
            DataCharacteristics for primitive
        """
        chars = DataCharacteristics()
        chars.structure_type = DataStructureType.TEXT
        chars.total_elements = 1
        json_str = json.dumps(data)
        chars.estimated_json_chars = len(json_str)
        chars.estimated_json_tokens = max(1, len(json_str) // 4)
        return chars

    def _analyze_list(self, data: List[Any]) -> DataCharacteristics:
        """Analyze a list/array.

        Args:
            data: List to analyze

        Returns:
            DataCharacteristics for list
        """
        chars = DataCharacteristics()

        if not data:
            chars.structure_type = DataStructureType.EMPTY
            chars.estimated_json_chars = 2  # []
            chars.estimated_json_tokens = 1
            return chars

        chars.array_length = len(data)
        chars.total_elements = len(data)

        # Check if all items are dicts (potential uniform array)
        all_dicts = all(isinstance(item, dict) for item in data)
        all_primitives = all(isinstance(item, (str, int, float, bool, type(None))) for item in data)

        if all_primitives:
            chars.structure_type = DataStructureType.PRIMITIVE_ARRAY
            chars.array_uniformity = 1.0
            chars = self._add_value_analysis(chars, data)

        elif all_dicts:
            chars = self._analyze_uniform_array(chars, data)

        else:
            chars.structure_type = DataStructureType.MIXED
            chars = self._analyze_nested(chars, data)

        # Estimate JSON size
        try:
            json_str = json.dumps(data, default=str)
            chars.estimated_json_chars = len(json_str)
            chars.estimated_json_tokens = len(json_str) // 4
        except (TypeError, ValueError):
            chars.estimated_json_chars = chars.total_elements * 20
            chars.estimated_json_tokens = chars.total_elements * 5

        return chars

    def _analyze_uniform_array(
        self,
        chars: DataCharacteristics,
        data: List[Dict[str, Any]],
    ) -> DataCharacteristics:
        """Analyze array of dicts for uniformity.

        Args:
            chars: Characteristics to update
            data: List of dicts

        Returns:
            Updated characteristics
        """
        # Sample for efficiency with large arrays
        sample = data[: self.UNIFORMITY_SAMPLE_SIZE]

        # Collect all keys
        all_keys: List[Set[str]] = []
        for item in sample:
            all_keys.append(set(item.keys()))

        if not all_keys:
            chars.structure_type = DataStructureType.EMPTY
            return chars

        # Find common keys (present in all items)
        common_keys = set.intersection(*all_keys) if all_keys else set()

        # Find all unique keys
        unique_keys = set.union(*all_keys) if all_keys else set()

        chars.unique_keys = len(unique_keys)
        chars.total_keys = sum(len(keys) for keys in all_keys)

        # Calculate uniformity (how similar the key sets are)
        if unique_keys:
            # Jaccard-like uniformity score
            uniformity_sum = sum(len(common_keys) / len(keys) if keys else 0 for keys in all_keys)
            chars.array_uniformity = uniformity_sum / len(all_keys)
        else:
            chars.array_uniformity = 1.0

        # Determine structure type
        if chars.array_uniformity >= 0.9:
            chars.structure_type = DataStructureType.UNIFORM_ARRAY
            chars.field_names = sorted(common_keys)
        else:
            chars.structure_type = DataStructureType.MIXED

        # Check for nested structures
        for item in sample:
            for value in item.values():
                if isinstance(value, dict):
                    chars.has_nested_objects = True
                elif isinstance(value, list):
                    chars.has_nested_arrays = True
                elif value is None:
                    chars.has_null_values = True
                elif isinstance(value, str) and any(c in self.SPECIAL_CHARS for c in value):
                    chars.has_special_chars = True

        # Analyze value repetition
        chars = self._add_value_analysis(chars, data)

        # Analyze field types
        chars.field_types = self._analyze_field_types(sample, common_keys)

        # Calculate nesting depth
        chars.nesting_depth = self._calculate_nesting_depth(sample[0])

        return chars

    def _analyze_dict(self, data: Dict[str, Any]) -> DataCharacteristics:
        """Analyze a dictionary/object.

        Args:
            data: Dict to analyze

        Returns:
            DataCharacteristics for dict
        """
        chars = DataCharacteristics()

        if not data:
            chars.structure_type = DataStructureType.EMPTY
            chars.estimated_json_chars = 2  # {}
            chars.estimated_json_tokens = 1
            return chars

        chars.unique_keys = len(data)
        chars.total_keys = len(data)
        chars.total_elements = len(data)
        chars.field_names = list(data.keys())

        # Check for nested structures
        has_nested = False
        for value in data.values():
            if isinstance(value, dict):
                chars.has_nested_objects = True
                has_nested = True
            elif isinstance(value, list):
                chars.has_nested_arrays = True
                has_nested = True
            elif value is None:
                chars.has_null_values = True
            elif isinstance(value, str) and any(c in self.SPECIAL_CHARS for c in value):
                chars.has_special_chars = True

        if has_nested:
            chars.structure_type = DataStructureType.NESTED_OBJECT
        else:
            chars.structure_type = DataStructureType.FLAT_OBJECT

        chars.nesting_depth = self._calculate_nesting_depth(data)

        # Estimate JSON size
        try:
            json_str = json.dumps(data, default=str)
            chars.estimated_json_chars = len(json_str)
            chars.estimated_json_tokens = len(json_str) // 4
        except (TypeError, ValueError):
            chars.estimated_json_chars = chars.total_elements * 20
            chars.estimated_json_tokens = chars.total_elements * 5

        return chars

    def _analyze_nested(
        self,
        chars: DataCharacteristics,
        data: Any,
    ) -> DataCharacteristics:
        """Recursively analyze nested structure.

        Args:
            chars: Characteristics to update
            data: Data to analyze

        Returns:
            Updated characteristics
        """
        chars.nesting_depth = self._calculate_nesting_depth(data)

        # Recursively check for nested structures
        def check_nested(obj: Any) -> None:
            if isinstance(obj, dict):
                chars.has_nested_objects = True
                for v in obj.values():
                    check_nested(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_nested(item)

        check_nested(data)
        return chars

    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth.

        Args:
            data: Data to analyze
            current_depth: Current depth in recursion

        Returns:
            Maximum nesting depth
        """
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_nesting_depth(v, current_depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_nesting_depth(item, current_depth + 1) for item in data)
        else:
            return current_depth

    def _add_value_analysis(
        self,
        chars: DataCharacteristics,
        data: List[Any],
    ) -> DataCharacteristics:
        """Analyze value repetition patterns.

        Args:
            chars: Characteristics to update
            data: List of items

        Returns:
            Updated characteristics with repetition metrics
        """
        # Collect all string values
        all_values: List[str] = []

        def collect_values(obj: Any) -> None:
            if isinstance(obj, str):
                all_values.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect_values(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect_values(item)

        for item in data:
            collect_values(item)

        if not all_values:
            return chars

        # Count value occurrences
        value_counts = Counter(all_values)

        # Find repeated values (occur more than once)
        repeated = {v: c for v, c in value_counts.items() if c > 1}

        if repeated:
            total_repeats = sum(repeated.values())
            chars.value_repetition_ratio = total_repeats / len(all_values)
            # Store top 20 most common values
            chars.common_values = dict(value_counts.most_common(20))
        else:
            chars.value_repetition_ratio = 0.0

        return chars

    def _analyze_field_types(
        self,
        sample: List[Dict[str, Any]],
        fields: Set[str],
    ) -> Dict[str, str]:
        """Determine the predominant type for each field.

        Args:
            sample: Sample of dicts to analyze
            fields: Field names to analyze

        Returns:
            Map of field name -> type name
        """
        field_types: Dict[str, str] = {}

        for field in fields:
            types: Counter = Counter()
            for item in sample:
                if field in item:
                    value = item[field]
                    if value is None:
                        types["null"] += 1
                    elif isinstance(value, bool):
                        types["bool"] += 1
                    elif isinstance(value, int):
                        types["int"] += 1
                    elif isinstance(value, float):
                        types["float"] += 1
                    elif isinstance(value, str):
                        types["string"] += 1
                    elif isinstance(value, list):
                        types["array"] += 1
                    elif isinstance(value, dict):
                        types["object"] += 1
                    else:
                        types["unknown"] += 1

            if types:
                field_types[field] = types.most_common(1)[0][0]

        return field_types


# Global analyzer instance
_analyzer: Optional[DataAnalyzer] = None


def get_data_analyzer() -> DataAnalyzer:
    """Get the global data analyzer instance.

    Returns:
        DataAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = DataAnalyzer()
    return _analyzer
