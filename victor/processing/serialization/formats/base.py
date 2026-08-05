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

"""Base classes for format encoders with plugin architecture.

Provides extensible framework for adding new serialization formats.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    DataCharacteristics,
)

logger = logging.getLogger(__name__)


@dataclass
class EncodingResult:
    """Result from a format encoder.

    Attributes:
        content: The encoded string
        success: Whether encoding succeeded
        error: Error message if failed
        metadata: Additional encoder-specific metadata
        reference_table: For reference encoding, the lookup table
    """

    content: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reference_table: Optional[Dict[str, str]] = None


class FormatEncoder(ABC):
    """Abstract base class for format encoders.

    To create a custom encoder:

    1. Inherit from FormatEncoder
    2. Set class attributes: format_id, format_name, format_description
    3. Implement: can_encode(), encode(), suitability_score()
    4. Optionally override: estimate_tokens(), get_format_hint()
    5. Register with FormatRegistry

    Example:
        class MyEncoder(FormatEncoder):
            format_id = SerializationFormat.CUSTOM
            format_name = "My Custom Format"
            format_description = "A custom format for special data."

            def can_encode(self, data, characteristics):
                return isinstance(data, list)

            def encode(self, data, characteristics, config):
                return EncodingResult(content=str(data))

            def suitability_score(self, characteristics):
                return 0.5
    """

    # Class attributes - override in subclasses
    format_id: SerializationFormat = SerializationFormat.JSON
    format_name: str = "Unknown Format"
    format_description: str = "No description available."

    # Encoder capabilities
    supports_nested: bool = True
    supports_arrays: bool = True
    supports_special_chars: bool = True
    max_nesting_depth: int = 100

    # Token efficiency baseline (1.0 = same as JSON)
    base_efficiency: float = 1.0

    @abstractmethod
    def can_encode(self, data: Any, characteristics: DataCharacteristics) -> bool:
        """Check if this encoder can handle the given data.

        Args:
            data: The data to encode
            characteristics: Analysis of data structure

        Returns:
            True if encoder can handle this data
        """
        pass

    @abstractmethod
    def encode(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> EncodingResult:
        """Encode data to string format.

        Args:
            data: The data to encode
            characteristics: Analysis of data structure
            config: Serialization configuration

        Returns:
            EncodingResult with encoded content
        """
        pass

    @abstractmethod
    def suitability_score(
        self,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> float:
        """Calculate how suitable this format is for the data.

        Score from 0.0 (not suitable) to 1.0 (ideal).
        Used by adaptive serializer for format selection.

        Args:
            characteristics: Analysis of data structure
            config: Serialization configuration

        Returns:
            Suitability score (0.0-1.0)
        """
        pass

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count for encoded content.

        Default implementation uses ~4 chars per token heuristic.
        Override for format-specific estimation.

        Args:
            content: Encoded content string

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English text
        # JSON overhead adds ~20% more tokens
        return len(content) // 4

    def get_format_hint(self, config: SerializationConfig) -> Optional[str]:
        """Generate format hint for LLM.

        Args:
            config: Serialization configuration

        Returns:
            Format hint string or None if hints disabled
        """
        if not config.include_format_hint:
            return None

        return config.format_hint_template.format(
            format_name=self.format_name,
            format_description=self.format_description,
        )

    def validate_output(self, content: str) -> bool:
        """Validate encoded output is well-formed.

        Override for format-specific validation.

        Args:
            content: Encoded content to validate

        Returns:
            True if valid
        """
        return bool(content)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(format={self.format_id.value})>"


class FormatRegistry:
    """Registry for format encoders with plugin support.

    Allows dynamic registration of new formats without modifying core code.
    Supports priority ordering and capability-based selection.

    Usage:
        # Get global registry
        registry = get_format_registry()

        # Register custom encoder
        registry.register(MyCustomEncoder())

        # Get encoder by format
        encoder = registry.get_encoder(SerializationFormat.TOON)

        # Find best encoder for data
        encoder = registry.select_best_encoder(characteristics, config)
    """

    def __init__(self):
        """Initialize empty registry."""
        self._encoders: Dict[SerializationFormat, FormatEncoder] = {}
        self._priority_order: List[SerializationFormat] = []
        self._hooks: Dict[str, List[Callable]] = {
            "pre_encode": [],
            "post_encode": [],
            "on_register": [],
        }

    def register(
        self,
        encoder: FormatEncoder,
        priority: Optional[int] = None,
    ) -> None:
        """Register a format encoder.

        Args:
            encoder: The encoder instance to register
            priority: Optional priority (lower = higher priority)
        """
        format_id = encoder.format_id

        if format_id in self._encoders:
            logger.debug(f"Replacing existing encoder for {format_id.value}")

        self._encoders[format_id] = encoder

        # Manage priority order
        if format_id in self._priority_order:
            self._priority_order.remove(format_id)

        if priority is not None and priority < len(self._priority_order):
            self._priority_order.insert(priority, format_id)
        else:
            self._priority_order.append(format_id)

        # Invoke registration hooks
        for hook in self._hooks["on_register"]:
            try:
                hook(encoder)
            except Exception as e:
                logger.warning(f"Registration hook failed: {e}")

        logger.debug(f"Registered encoder: {encoder}")

    def unregister(self, format_id: SerializationFormat) -> bool:
        """Unregister a format encoder.

        Args:
            format_id: Format to unregister

        Returns:
            True if encoder was removed
        """
        if format_id in self._encoders:
            del self._encoders[format_id]
            if format_id in self._priority_order:
                self._priority_order.remove(format_id)
            logger.debug(f"Unregistered encoder: {format_id.value}")
            return True
        return False

    def get_encoder(self, format_id: SerializationFormat) -> Optional[FormatEncoder]:
        """Get encoder by format ID.

        Args:
            format_id: Format to look up

        Returns:
            FormatEncoder or None if not found
        """
        return self._encoders.get(format_id)

    def get_all_encoders(self) -> Dict[SerializationFormat, FormatEncoder]:
        """Get all registered encoders.

        Returns:
            Dictionary of format -> encoder
        """
        return self._encoders.copy()

    def list_formats(self) -> List[SerializationFormat]:
        """List all registered formats in priority order.

        Returns:
            List of format IDs
        """
        return self._priority_order.copy()

    def select_best_encoder(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
    ) -> Optional[FormatEncoder]:
        """Select the best encoder for given data.

        Considers:
        - Data characteristics
        - Config preferences and restrictions
        - Encoder suitability scores
        - Priority order

        Args:
            data: Data to encode
            characteristics: Data analysis results
            config: Serialization configuration

        Returns:
            Best matching encoder or None
        """
        # If preferred format is set and encoder exists, use it
        if config.preferred_format:
            encoder = self.get_encoder(config.preferred_format)
            if encoder and encoder.can_encode(data, characteristics):
                return encoder

        # Score all eligible encoders
        candidates: List[tuple[float, FormatEncoder]] = []

        for format_id in self._priority_order:
            # Skip disabled formats
            if format_id in config.disabled_formats:
                continue

            # Skip if not in allowed list (if list is specified)
            if config.allowed_formats and format_id not in config.allowed_formats:
                continue

            encoder = self._encoders.get(format_id)
            if not encoder:
                continue

            # Check if encoder can handle data
            if not encoder.can_encode(data, characteristics):
                continue

            # Calculate suitability score
            score = encoder.suitability_score(characteristics, config)

            # Apply minimum savings threshold
            if format_id != SerializationFormat.JSON:
                estimated_savings = characteristics.estimated_savings(format_id)
                if estimated_savings < config.min_savings_threshold:
                    score *= 0.5  # Penalize if savings below threshold

            candidates.append((score, encoder))

        if not candidates:
            # Fall back to default
            return self.get_encoder(config.fallback_format)

        # Sort by score (descending) and return best
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def add_hook(self, hook_type: str, callback: Callable) -> None:
        """Add a hook callback.

        Hook types:
        - pre_encode: Called before encoding (data, characteristics, config)
        - post_encode: Called after encoding (result, encoder)
        - on_register: Called when encoder is registered (encoder)

        Args:
            hook_type: Type of hook
            callback: Callback function
        """
        if hook_type in self._hooks:
            self._hooks[hook_type].append(callback)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")

    def remove_hook(self, hook_type: str, callback: Callable) -> bool:
        """Remove a hook callback.

        Args:
            hook_type: Type of hook
            callback: Callback to remove

        Returns:
            True if callback was removed
        """
        if hook_type in self._hooks and callback in self._hooks[hook_type]:
            self._hooks[hook_type].remove(callback)
            return True
        return False

    def invoke_hooks(self, hook_type: str, *args, **kwargs) -> None:
        """Invoke all hooks of a given type.

        Args:
            hook_type: Type of hooks to invoke
            *args, **kwargs: Arguments to pass to hooks
        """
        for hook in self._hooks.get(hook_type, []):
            try:
                hook(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {hook_type} failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry stats
        """
        return {
            "registered_formats": len(self._encoders),
            "formats": [f.value for f in self._priority_order],
            "hooks": {k: len(v) for k, v in self._hooks.items()},
        }


# Global registry instance
_registry: Optional[FormatRegistry] = None


def get_format_registry() -> FormatRegistry:
    """Get the global format registry.

    Returns:
        Global FormatRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = FormatRegistry()
    return _registry


def reset_format_registry() -> None:
    """Reset the global format registry (mainly for testing)."""
    global _registry
    _registry = None
