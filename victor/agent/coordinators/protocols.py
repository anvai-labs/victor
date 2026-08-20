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

"""Coordinator protocols for Dependency Inversion Principle (DIP) compliance.

This module defines protocols for coordinator dependencies, enabling the
Dependency Inversion Principle (DIP) by depending on abstractions rather
than concrete implementations.

Design Pattern: Protocol (Interface Segregation Principle)
- CacheProvider: Protocol for caching operations
- EventEmitter: Protocol for event emission
- ConfigProvider: Protocol for configuration access

Phase 2: Fix SOLID Violations via Coordinator Extraction
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    TypeVar,
)
from abc import ABC, abstractmethod
from enum import Enum

from victor.agent.services.protocols.chat_runtime import ExecutionMode

T = TypeVar("T")


class CacheEventType(Enum):
    """Types of cache events."""

    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    INVALIDATION = "invalidation"
    CLEAR = "clear"


@runtime_checkable
class CacheProvider(Protocol):
    """Protocol for cache providers.

    Defines the interface for caching operations, enabling coordinators
    to work with any cache implementation that satisfies this protocol.

    This promotes DIP by allowing high-level modules to depend on this
    abstraction rather than concrete cache implementations.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if key was found and deleted, False otherwise
        """
        ...

    def clear(self) -> None:
        """Clear all values from the cache."""
        ...

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from the cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values (only found keys included)
        """
        ...

    def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set multiple values in the cache.

        Args:
            items: Dictionary of key-value pairs
            ttl: Optional time-to-live in seconds
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, etc.)
        """
        ...


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for event emission.

    Defines the interface for event emission, enabling coordinators
    to emit events without depending on concrete event systems.

    This promotes DIP by allowing coordinators to emit events through
    this abstraction rather than depending on a specific event bus implementation.
    """

    def emit(self, event_name: str, data: Any = None, **kwargs: Any) -> None:
        """Emit an event.

        Args:
            event_name: Name of the event
            data: Event data
            **kwargs: Additional event parameters
        """
        ...

    async def emit_async(self, event_name: str, data: Any = None, **kwargs: Any) -> None:
        """Emit an event asynchronously.

        Args:
            event_name: Name of the event
            data: Event data
            **kwargs: Additional event parameters
        """
        ...

    def on(self, event_name: str, callback: Callable[..., Any]) -> Callable[..., Any]:
        """Register a callback for an event.

        Args:
            event_name: Name of the event
            callback: Callback function

        Returns:
            The callback function (for decorator usage)
        """
        ...

    def off(self, event_name: str, callback: Optional[Callable[..., Any]] = None) -> None:
        """Unregister a callback for an event.

        Args:
            event_name: Name of the event
            callback: Callback function to remove (None = remove all)
        """
        ...

    def once(self, event_name: str, callback: Callable[..., Any]) -> Callable[..., Any]:
        """Register a one-time callback for an event.

        Args:
            event_name: Name of the event
            callback: Callback function

        Returns:
            The callback function (for decorator usage)
        """
        ...

    def emit_to_stream(
        self, event_name: str, data: Any = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Emit events to a stream.

        Args:
            event_name: Name of the event
            data: Event data
            **kwargs: Additional event parameters

        Yields:
            Event data
        """
        ...


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration access.

    Defines the interface for configuration access, enabling coordinators
    to access configuration without depending on concrete config implementations.

    This promotes DIP by allowing coordinators to access configuration through
    this abstraction rather than depending on a specific configuration system.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        ...

    def get_int(self, key: str, default: int = 0) -> int:
        """Get a configuration value as an integer.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value as integer
        """
        ...

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a configuration value as a float.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value as float
        """
        ...

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a configuration value as a boolean.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value as boolean
        """
        ...

    def get_str(self, key: str, default: str = "") -> str:
        """Get a configuration value as a string.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value as string
        """
        ...

    def get_list(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        """Get a configuration value as a list.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value as list of strings
        """
        ...

    def get_dict(self, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a configuration value as a dictionary.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value as dictionary
        """
        ...

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        ...

    def has(self, key: str) -> bool:
        """Check if a configuration key exists.

        Args:
            key: Configuration key

        Returns:
            True if key exists, False otherwise
        """
        ...

    def reload(self) -> None:
        """Reload configuration from source."""
        ...

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Dictionary of all configuration key-value pairs
        """
        ...


class NoOpCacheProvider:
    """No-op cache provider for testing and fallback.

    Implements CacheProvider protocol without actual caching.
    Useful for testing and as a fallback when no cache is available.
    """

    def __init__(self) -> None:
        """Initialize the no-op cache provider."""
        self._storage: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value (always returns default)."""
        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value (no-op)."""
        pass

    def delete(self, key: str) -> bool:
        """Delete a value (always returns False)."""
        return False

    def clear(self) -> None:
        """Clear all values (no-op)."""
        pass

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values (always returns empty dict)."""
        return {}

    def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set multiple values (no-op)."""
        pass

    def exists(self, key: str) -> bool:
        """Check if key exists (always returns False)."""
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "type": "noop",
        }


class NoOpEventEmitter:
    """No-op event emitter for testing and fallback.

    Implements EventEmitter protocol without actual event emission.
    Useful for testing and as a fallback when no event system is available.
    """

    def emit(self, event_name: str, data: Any = None, **kwargs: Any) -> None:
        """Emit an event (no-op)."""
        pass

    async def emit_async(self, event_name: str, data: Any = None, **kwargs: Any) -> None:
        """Emit an event asynchronously (no-op)."""
        pass

    def on(self, event_name: str, callback: Callable[..., Any]) -> Callable[..., Any]:
        """Register a callback (returns callback unchanged)."""
        return callback

    def off(self, event_name: str, callback: Optional[Callable[..., Any]] = None) -> None:
        """Unregister a callback (no-op)."""
        pass

    def once(self, event_name: str, callback: Callable[..., Any]) -> Callable[..., Any]:
        """Register a one-time callback (returns callback unchanged)."""
        return callback

    async def emit_to_stream(
        self, event_name: str, data: Any = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Emit events to a stream (yields nothing)."""
        return
        yield  # pragma: no cover


class DictConfigProvider:
    """Dictionary-based config provider for testing and simple use cases.

    Implements ConfigProvider protocol using a dictionary backend.
    Useful for testing and simple configuration scenarios.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the dictionary config provider.

        Args:
            config: Initial configuration dictionary
        """
        self._config = config or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with dot notation support."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value if value is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get a configuration value as an integer."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a configuration value as a float."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a configuration value as a boolean."""
        value = self.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def get_str(self, key: str, default: str = "") -> str:
        """Get a configuration value as a string."""
        value = self.get(key)
        if value is None:
            return default
        return str(value)

    def get_list(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        """Get a configuration value as a list."""
        if default is None:
            default = []
        value = self.get(key)
        if value is None:
            return default
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
        return [str(value)]

    def get_dict(self, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a configuration value as a dictionary."""
        if default is None:
            default = {}
        value = self.get(key)
        if value is None:
            return default
        if isinstance(value, dict):
            return value
        return {}

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def has(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return self.get(key) is not None

    def reload(self) -> None:
        """Reload configuration (no-op for dict provider)."""
        pass

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()


__all__ = [
    "CacheProvider",
    "CacheEventType",
    "EventEmitter",
    "ConfigProvider",
    "NoOpCacheProvider",
    "NoOpEventEmitter",
    "DictConfigProvider",
    # Phase 3: Enhanced protocol-based injection
    "ExecutionProvider",
    "ToolExecutor",
    "MessageStore",
    "StateManager",
    "ExecutionMode",
]


# =============================================================================
# Phase 3: Enhanced Protocol-Based Injection
# =============================================================================


@runtime_checkable
class ExecutionProvider(Protocol):
    """[LEGACY PROTOCOL] Interface for execution operations.

    Superseded by ChatServiceProtocol for high-level operations.
    Used by TurnExecutor for executing model turns.
    """

    async def execute_turn(
        self,
        messages: List[Any],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a single model turn.

        Args:
            messages: Conversation history
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse from the model
        """
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """[LEGACY PROTOCOL] Interface for tool execution.

    Superseded by ToolServiceProtocol for high-level operations.
    Used by coordinators for executing tools.
    """

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool with arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Dict with tool execution result:
            - success: bool
            - result: Any (on success)
            - error: Optional[str] (on failure)
        """
        ...

    async def execute_tool_calls(
        self,
        tool_calls: List[Any],
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool call objects

        Returns:
            List of tool execution results
        """
        ...


@runtime_checkable
class MessageStore(Protocol):
    """[LEGACY PROTOCOL] Interface for message storage.

    Superseded by ContextServiceProtocol for high-level operations.
    Used by coordinators for conversation history.
    """

    @property
    def messages(self) -> List[Any]:
        """Get current message history."""
        ...

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add message to history.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
        """
        ...

    def get_messages(self, limit: Optional[int] = None) -> List[Any]:
        """Get messages from history.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        ...

    def clear_messages(self) -> None:
        """Clear message history."""
        ...

    @property
    def conversation(self) -> Any:
        """Get conversation manager."""
        ...


@runtime_checkable
class StateManager(Protocol):
    """[LEGACY PROTOCOL] Interface for state management.

    Superseded by ContextServiceProtocol / SessionServiceProtocol for
    high-level operations.
    Used by coordinators for state tracking.
    """

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value or default
        """
        ...

    def set_state(self, key: str, value: Any) -> None:
        """Set state value.

        Args:
            key: State key
            value: State value
        """
        ...

    def delete_state(self, key: str) -> bool:
        """Delete state value.

        Args:
            key: State key

        Returns:
            True if key was found and deleted, False otherwise
        """
        ...

    def get_all_state(self) -> Dict[str, Any]:
        """Get all state values.

        Returns:
            Dictionary of all state key-value pairs
        """
        ...

    def clear_state(self) -> None:
        """Clear all state values."""
        ...
