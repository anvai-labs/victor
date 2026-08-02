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

"""State serialization for conversation checkpoints.

Handles serialization/deserialization of conversation state for checkpoint
persistence. Supports compression for large states and includes integrity
checksums.
"""

import gzip
import hashlib
import logging

from victor.core.json_utils import json_dumps, json_loads
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

# Compression threshold in bytes
COMPRESSION_THRESHOLD = 10 * 1024  # 10KB


@dataclass
class SerializedState:
    """Container for serialized state with metadata.

    Attributes:
        data: Serialized state data (JSON-compatible)
        compressed: Whether data is compressed
        checksum: SHA-256 checksum of original data
        serializer_version: Version of serializer for compatibility
    """

    data: Union[bytes, Dict[str, Any]]
    compressed: bool
    checksum: str
    serializer_version: int = 1


class StateSerializer:
    """Serializes conversation state for checkpoint storage.

    Handles:
    - JSON serialization of complex types (datetime, sets, etc.)
    - Optional compression for large states
    - Integrity checksums
    - Forward compatibility via versioning
    """

    VERSION = 1

    def __init__(
        self,
        compress: bool = True,
        compression_threshold: int = COMPRESSION_THRESHOLD,
    ):
        """Initialize the serializer.

        Args:
            compress: Whether to compress large states
            compression_threshold: Size threshold for compression in bytes
        """
        self.compress = compress
        self.compression_threshold = compression_threshold

    def serialize(self, state: Dict[str, Any]) -> SerializedState:
        """Serialize conversation state.

        Args:
            state: Dictionary containing conversation state

        Returns:
            SerializedState with data, compression flag, and checksum
        """
        # Convert to JSON-safe format
        json_safe = self._to_json_safe(state)

        # Serialize to JSON string
        json_str = json_dumps(json_safe, sort_keys=True, ensure_ascii=False)
        json_bytes = json_str.encode("utf-8")

        # Calculate checksum of uncompressed data
        checksum = hashlib.sha256(json_bytes).hexdigest()

        # Optionally compress
        compressed = False
        if self.compress and len(json_bytes) > self.compression_threshold:
            compressed_bytes = gzip.compress(json_bytes)
            # Only use compression if it actually reduces size
            if len(compressed_bytes) < len(json_bytes):
                json_bytes = compressed_bytes
                compressed = True
                logger.debug(
                    f"Compressed state from {len(json_str)} to {len(compressed_bytes)} bytes"
                )

        return SerializedState(
            data=json_bytes if compressed else json_safe,
            compressed=compressed,
            checksum=checksum,
            serializer_version=self.VERSION,
        )

    def deserialize(self, serialized: SerializedState) -> Dict[str, Any]:
        """Deserialize conversation state.

        Args:
            serialized: SerializedState from serialize()

        Returns:
            Restored conversation state dictionary

        Raises:
            ValueError: If checksum validation fails
        """
        if serialized.compressed:
            if not isinstance(serialized.data, bytes):
                raise ValueError("Compressed data must be bytes")

            # Decompress
            json_bytes = gzip.decompress(serialized.data)
            json_str = json_bytes.decode("utf-8")

            # Verify checksum
            checksum = hashlib.sha256(json_bytes).hexdigest()
            if checksum != serialized.checksum:
                raise ValueError(
                    f"Checksum mismatch: expected {serialized.checksum}, got {checksum}"
                )

            json_safe = json_loads(json_str)
        else:
            if isinstance(serialized.data, bytes):
                json_bytes = serialized.data
                json_str = json_bytes.decode("utf-8")

                # Verify checksum
                checksum = hashlib.sha256(json_bytes).hexdigest()
                if checksum != serialized.checksum:
                    raise ValueError(
                        f"Checksum mismatch: expected {serialized.checksum}, got {checksum}"
                    )

                json_safe = json_loads(json_str)
            else:
                json_safe = serialized.data

                # Verify checksum
                json_str = json_dumps(json_safe, sort_keys=True, ensure_ascii=False)
                checksum = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
                if checksum != serialized.checksum:
                    raise ValueError(
                        f"Checksum mismatch: expected {serialized.checksum}, got {checksum}"
                    )

        # Restore Python types
        return self._from_json_safe(json_safe)

    def _to_json_safe(self, obj: Any) -> Any:
        """Convert Python objects to JSON-safe format.

        Handles:
        - datetime -> ISO format string with marker
        - set -> list with marker
        - bytes -> base64 string with marker
        - dataclass -> dict
        - other types -> str representation
        """
        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}

        if isinstance(obj, set):
            return {
                "__type__": "set",
                "value": [self._to_json_safe(item) for item in sorted(obj, key=str)],
            }

        if isinstance(obj, bytes):
            import base64

            return {"__type__": "bytes", "value": base64.b64encode(obj).decode("ascii")}

        if isinstance(obj, dict):
            return {k: self._to_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._to_json_safe(item) for item in obj]

        # Handle dataclasses
        if hasattr(obj, "__dataclass_fields__"):
            return self._to_json_safe(asdict(obj))

        # Fallback: convert to string
        logger.warning(f"Converting unknown type {type(obj)} to string")
        return {"__type__": "str_repr", "value": str(obj)}

    def _from_json_safe(self, obj: Any) -> Any:
        """Restore Python objects from JSON-safe format."""
        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, dict):
            # Check for type markers
            if "__type__" in obj:
                type_name = obj["__type__"]
                value = obj["value"]

                if type_name == "datetime":
                    return datetime.fromisoformat(value)

                if type_name == "set":
                    return set(self._from_json_safe(item) for item in value)

                if type_name == "bytes":
                    import base64

                    return base64.b64decode(value)

                if type_name == "str_repr":
                    return value  # Keep as string

            # Regular dict
            return {k: self._from_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._from_json_safe(item) for item in obj]

        return obj


def serialize_conversation_state(
    state: Dict[str, Any],
    compress: bool = True,
) -> Dict[str, Any]:
    """Convenience function to serialize conversation state.

    Args:
        state: Conversation state dictionary
        compress: Whether to compress large states

    Returns:
        Dictionary suitable for storage
    """
    serializer = StateSerializer(compress=compress)
    result = serializer.serialize(state)

    if result.compressed:
        import base64

        return {
            "data": base64.b64encode(result.data).decode("ascii"),
            "compressed": True,
            "checksum": result.checksum,
            "version": result.serializer_version,
        }
    else:
        return {
            "data": result.data,
            "compressed": False,
            "checksum": result.checksum,
            "version": result.serializer_version,
        }


def deserialize_conversation_state(stored: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to deserialize conversation state.

    Args:
        stored: Dictionary from storage

    Returns:
        Restored conversation state
    """
    serializer = StateSerializer()

    if stored.get("compressed"):
        import base64

        data = base64.b64decode(stored["data"])
    else:
        data = stored["data"]

    serialized = SerializedState(
        data=data,
        compressed=stored.get("compressed", False),
        checksum=stored.get("checksum", ""),
        serializer_version=stored.get("version", 1),
    )

    return serializer.deserialize(serialized)
