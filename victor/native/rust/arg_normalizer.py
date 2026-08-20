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

"""Rust argument normalizer wrapper.

Provides a protocol-compliant wrapper around the Rust argument normalizer
functions. The wrapper delegates to victor_native functions while maintaining
the ArgumentNormalizerProtocol interface and observability hooks.

Performance characteristics:
- normalize_json: 5-10x faster with streaming parser
- coerce_type: 3-5x faster with direct parsing
- repair_quotes: 2-3x faster with single-pass state machine
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

import victor_native

from victor.native.observability import InstrumentedAccelerator
from victor.native.protocols import CoercedType, CoercedValue

# Mapping from Rust type names to CoercedType enum
_TYPE_MAP = {
    "string": CoercedType.STRING,
    "int": CoercedType.INT,
    "float": CoercedType.FLOAT,
    "bool": CoercedType.BOOL,
    "null": CoercedType.NULL,
    "list": CoercedType.LIST,
    "dict": CoercedType.DICT,
}


class RustArgumentNormalizer(InstrumentedAccelerator):
    """Rust implementation of ArgumentNormalizerProtocol.

    Wraps the high-performance Rust functions for JSON repair and type
    coercion with protocol-compliant interface.

    Performance characteristics:
    - normalize_json: 5-10x faster (serde_json + streaming repair)
    - coerce_type: 3-5x faster (direct parsing, no exceptions)
    - repair_quotes: 2-3x faster (single-pass state machine)
    """

    def __init__(self) -> None:
        super().__init__(backend="rust")
        self._version = victor_native.__version__

    def get_version(self) -> Optional[str]:
        return self._version

    def normalize_json(self, value: str) -> Tuple[str, bool]:
        """Normalize a potentially malformed JSON string.

        Delegates to Rust implementation with serde_json validation.

        Args:
            value: Potentially malformed JSON string

        Returns:
            Tuple of (normalized JSON string, success boolean)
        """
        with self._timed_call("json_repair"):
            return victor_native.normalize_json_string(value)

    def coerce_type(self, value: str) -> CoercedValue:
        """Coerce a string value to its likely type.

        Delegates to Rust implementation for fast type detection.

        Args:
            value: String value to coerce

        Returns:
            CoercedValue with detected type
        """
        with self._timed_call("type_coercion"):
            # Rust returns (type_name: str, value_str: str, confidence: float)
            type_name, value_str, confidence = victor_native.coerce_string_type(value)

            # Convert type name to enum
            coerced_type = _TYPE_MAP.get(type_name, CoercedType.STRING)

            # Convert value string to actual Python value
            actual_value = self._convert_value(value_str, coerced_type)

            return CoercedValue(
                value=actual_value,
                original=value,
                coerced_type=coerced_type,
                confidence=confidence,
            )

    def repair_quotes(self, value: str) -> str:
        """Repair mismatched or incorrect quotes in JSON.

        Delegates to Rust implementation with single-pass state machine.

        Args:
            value: String with potential quote issues

        Returns:
            String with repaired quotes
        """
        with self._timed_call("quote_repair"):
            return victor_native.repair_quotes(value)

    def _convert_value(self, value_str: str, coerced_type: CoercedType) -> Any:
        """Convert a value string to its Python type.

        Args:
            value_str: String representation of the value
            coerced_type: Target type

        Returns:
            Python value of the appropriate type
        """
        if coerced_type == CoercedType.NULL:
            return None
        elif coerced_type == CoercedType.BOOL:
            return value_str.lower() == "true"
        elif coerced_type == CoercedType.INT:
            return int(value_str)
        elif coerced_type == CoercedType.FLOAT:
            return float(value_str)
        elif coerced_type in (CoercedType.LIST, CoercedType.DICT):
            return json.loads(value_str)
        else:
            return value_str
