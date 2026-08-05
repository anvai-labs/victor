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

"""Debug Adapter Registry for discovering and managing adapters.

This module provides a registry for debug adapters using the
Factory Pattern, allowing dynamic discovery and instantiation
of language-specific debuggers.

Usage:
    registry = DebugAdapterRegistry()
    registry.register("python", PythonDebugAdapter)

    # Get adapter for a language
    adapter = registry.get_adapter("python")

    # Auto-discover adapters
    registry.discover_adapters()
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

from victor.observability.debug.adapter import (
    BaseDebugAdapter,
    DebugAdapter,
    DebugAdapterCapabilities,
)

logger = logging.getLogger(__name__)

# Type alias for adapter factory
AdapterFactory = Callable[[], DebugAdapter]


class DebugAdapterRegistry:
    """Registry for debug adapters.

    Manages registration and discovery of debug adapters for different
    languages. Supports both explicit registration and auto-discovery.

    Example:
        registry = DebugAdapterRegistry()

        # Register adapter class
        registry.register("python", PythonDebugAdapter)

        # Or register factory function
        registry.register("python", lambda: PythonDebugAdapter(config))

        # Get adapter
        adapter = registry.get_adapter("python")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._adapters: Dict[str, AdapterFactory] = {}
        self._instances: Dict[str, DebugAdapter] = {}
        self._language_aliases: Dict[str, str] = {
            # Common aliases
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "jsx": "javascript",
            "rs": "rust",
            "rb": "ruby",
            "go": "golang",
            "cs": "csharp",
            "c#": "csharp",
            "c++": "cpp",
            "cxx": "cpp",
        }

    def register(
        self,
        language: str,
        adapter: Union[Type[DebugAdapter], AdapterFactory],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a debug adapter for a language.

        Args:
            language: Primary language name (e.g., "python")
            adapter: Adapter class or factory function
            aliases: Additional language aliases (e.g., ["py", "python3"])
        """
        # Convert class to factory if needed
        if isinstance(adapter, type):
            factory: AdapterFactory = adapter  # type: ignore
        else:
            factory = adapter

        self._adapters[language.lower()] = factory
        logger.info(f"Registered debug adapter for: {language}")

        # Register aliases
        if aliases:
            for alias in aliases:
                self._language_aliases[alias.lower()] = language.lower()

    def unregister(self, language: str) -> None:
        """Unregister an adapter.

        Args:
            language: Language to unregister
        """
        language = self._resolve_language(language)
        if language in self._adapters:
            del self._adapters[language]
            # Clean up instance if exists
            if language in self._instances:
                del self._instances[language]
            logger.info(f"Unregistered debug adapter for: {language}")

    def get_adapter(self, language: str, create_new: bool = False) -> DebugAdapter:
        """Get a debug adapter for a language.

        Args:
            language: Language name or alias
            create_new: If True, create new instance instead of cached

        Returns:
            Debug adapter instance

        Raises:
            ValueError: If no adapter registered for language
        """
        language = self._resolve_language(language)

        if language not in self._adapters:
            available = ", ".join(sorted(self._adapters.keys()))
            raise ValueError(
                f"No debug adapter registered for '{language}'. " f"Available: {available}"
            )

        # Return cached instance if available and not forcing new
        if not create_new and language in self._instances:
            return self._instances[language]

        # Create new instance
        factory = self._adapters[language]
        adapter = factory()
        self._instances[language] = adapter
        return adapter

    def has_adapter(self, language: str) -> bool:
        """Check if an adapter is registered for a language.

        Args:
            language: Language name or alias

        Returns:
            True if adapter exists
        """
        language = self._resolve_language(language)
        return language in self._adapters

    def list_adapters(self) -> Dict[str, DebugAdapterCapabilities]:
        """List all registered adapters with their capabilities.

        Returns:
            Dict mapping language to capabilities
        """
        result = {}
        for language, _factory in self._adapters.items():
            try:
                adapter = self.get_adapter(language)
                result[language] = adapter.capabilities
            except Exception as e:
                logger.warning(f"Failed to get capabilities for {language}: {e}")
                result[language] = DebugAdapterCapabilities()
        return result

    def supported_languages(self) -> List[str]:
        """Get list of all supported languages.

        Returns:
            List of language names
        """
        return sorted(self._adapters.keys())

    def _resolve_language(self, language: str) -> str:
        """Resolve language alias to canonical name.

        Args:
            language: Language name or alias

        Returns:
            Canonical language name
        """
        language = language.lower()
        return self._language_aliases.get(language, language)

    def discover_adapters(self, paths: Optional[List[Path]] = None) -> int:
        """Auto-discover debug adapters.

        Searches for adapter modules in standard locations and
        plugin directories.

        Args:
            paths: Additional paths to search

        Returns:
            Number of adapters discovered
        """
        discovered = 0

        # Built-in adapters
        builtin_adapters = [
            ("python", "victor.debug.adapters.python", "PythonDebugAdapter"),
            ("javascript", "victor.debug.adapters.node", "NodeDebugAdapter"),
            ("typescript", "victor.debug.adapters.node", "NodeDebugAdapter"),
            ("rust", "victor.debug.adapters.lldb", "LLDBDebugAdapter"),
            ("cpp", "victor.debug.adapters.lldb", "LLDBDebugAdapter"),
            ("c", "victor.debug.adapters.lldb", "LLDBDebugAdapter"),
            ("go", "victor.debug.adapters.delve", "DelveDebugAdapter"),
        ]

        for language, module_name, class_name in builtin_adapters:
            try:
                module = importlib.import_module(module_name)
                adapter_class = getattr(module, class_name)
                self.register(language, adapter_class)
                discovered += 1
            except ImportError:
                logger.debug(f"Adapter not available: {module_name}")
            except AttributeError:
                logger.warning(f"Adapter class not found: {class_name} in {module_name}")

        # Plugin adapters from paths
        if paths:
            for path in paths:
                discovered += self._discover_from_path(path)

        logger.info(f"Discovered {discovered} debug adapters")
        return discovered

    def _discover_from_path(self, path: Path) -> int:
        """Discover adapters from a directory.

        Args:
            path: Directory to search

        Returns:
            Number discovered
        """
        discovered = 0

        if not path.exists() or not path.is_dir():
            return 0

        for py_file in path.glob("*_adapter.py"):
            try:
                # Import module
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find adapter classes
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, BaseDebugAdapter)
                            and obj is not BaseDebugAdapter
                        ):
                            # Register for each supported language
                            instance = obj()
                            for lang in instance.languages:
                                self.register(lang, obj)
                                discovered += 1

            except Exception as e:
                logger.warning(f"Failed to load adapter from {py_file}: {e}")

        return discovered


# Global registry instance
_global_registry: Optional[DebugAdapterRegistry] = None


def get_debug_registry() -> DebugAdapterRegistry:
    """Get the global debug adapter registry.

    Returns:
        Global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = DebugAdapterRegistry()
    return _global_registry
