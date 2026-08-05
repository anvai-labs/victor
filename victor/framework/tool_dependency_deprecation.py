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

"""Tool Dependency Deprecation Helper.

Provides infrastructure for deprecating tool dependency constants while
maintaining backward compatibility. This enables the gradual migration
from hard-coded Python dictionaries to YAML-based configuration.

Example:
    # In a vertical's tool_dependencies.py:
    from victor.framework.tool_dependency_deprecation import (
        create_vertical_deprecation_module,
    )

    # Create deprecated constant descriptors for the vertical
    _deprecated = create_vertical_deprecation_module(
        vertical_name="coding",
        yaml_path=Path(__file__).parent / "tool_dependencies.yaml",
        constant_prefix="CODING",
    )

    # Module-level __getattr__ for backward compatibility
    def __getattr__(name: str):
        if hasattr(_deprecated, name):
            return getattr(_deprecated, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DeprecatedConstantDescriptor(Generic[T]):
    """Generic descriptor for deprecated tool dependency constants.

    Provides lazy initialization with deprecation warnings and caching.
    When a deprecated constant is accessed, it:
    1. Emits a deprecation warning (once per constant)
    2. Loads the value using the provided loader function
    3. Caches the value for subsequent accesses

    Attributes:
        constant_name: Name of the deprecated constant
        deprecation_message: Warning message to emit
        loader: Function to load the actual value
        _cached_value: Cached value after first access
        _loaded: Whether value has been loaded

    Example:
        class _DeprecatedConstants:
            MY_CONSTANT = DeprecatedConstantDescriptor(
                constant_name="MY_CONSTANT",
                deprecation_message="Use new_method() instead",
                loader=lambda: load_from_yaml(),
            )

        _deprecated = _DeprecatedConstants()
        # Accessing triggers warning and caching
        value = _deprecated.MY_CONSTANT
    """

    def __init__(
        self,
        constant_name: str,
        deprecation_message: str,
        loader: Callable[[], T],
        *,
        stacklevel: int = 2,
    ) -> None:
        """Initialize the deprecated constant descriptor.

        Args:
            constant_name: Name of the constant (for logging/debugging)
            deprecation_message: Message to include in deprecation warning
            loader: Callable that returns the constant's value when invoked
            stacklevel: Stack level for warnings.warn (default: 2)
        """
        self.constant_name = constant_name
        self.deprecation_message = deprecation_message
        self.loader = loader
        self.stacklevel = stacklevel
        self._cached_value: Optional[T] = None
        self._loaded: bool = False
        self._warned: bool = False

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        """Get the constant value, emitting deprecation warning on first access.

        Args:
            obj: Instance accessing the descriptor (unused for class-level)
            objtype: Type of the class (unused)

        Returns:
            The loaded and cached constant value
        """
        if not self._warned:
            logger.warning(self.deprecation_message)
            self._warned = True

        if not self._loaded:
            try:
                self._cached_value = self.loader()
                self._loaded = True
                logger.debug(f"Loaded deprecated constant: {self.constant_name}")
            except Exception as e:
                logger.error(f"Failed to load deprecated constant {self.constant_name}: {e}")
                raise

        return self._cached_value  # type: ignore[return-value]

    def reset(self) -> None:
        """Reset the cached value and warning state.

        Useful for testing to ensure fresh loads.
        """
        self._cached_value = None
        self._loaded = False
        self._warned = False


class VerticalDeprecationModule:
    """Container for deprecated constant descriptors for a vertical.

    Created by create_vertical_deprecation_module() to provide a namespace
    for all deprecated constants of a specific vertical.

    Attributes:
        vertical_name: Name of the vertical (e.g., "coding", "devops")
        yaml_path: Path to the YAML configuration file
        constant_prefix: Prefix for constant names (e.g., "CODING")
        descriptors: Dict mapping constant names to their descriptors
    """

    def __init__(
        self,
        vertical_name: str,
        yaml_path: Path,
        constant_prefix: str,
    ) -> None:
        """Initialize the vertical deprecation module.

        Args:
            vertical_name: Name of the vertical
            yaml_path: Path to YAML configuration file
            constant_prefix: Prefix for constant names
        """
        self.vertical_name = vertical_name
        self.yaml_path = yaml_path
        self.constant_prefix = constant_prefix
        self._descriptors: Dict[str, DeprecatedConstantDescriptor] = {}
        self._config_cache: Optional[Any] = None

    def _get_config(self) -> Any:
        """Get or load the YAML configuration.

        Returns:
            Loaded ToolDependencyConfig
        """
        if self._config_cache is None:
            from victor.core.tool_dependency_loader import load_tool_dependency_yaml

            self._config_cache = load_tool_dependency_yaml(
                self.yaml_path,
                canonicalize=True,
            )
        return self._config_cache

    def _create_descriptor(
        self,
        suffix: str,
        extractor: str,
        provider_method: str,
    ) -> DeprecatedConstantDescriptor:
        """Create a descriptor for a specific constant.

        Args:
            suffix: Constant name suffix (e.g., "TOOL_DEPENDENCIES")
            extractor: Attribute name on ToolDependencyConfig
            provider_method: Recommended method name on provider class

        Returns:
            Configured DeprecatedConstantDescriptor
        """
        constant_name = f"{self.constant_prefix}_{suffix}"
        deprecation_message = (
            f"{constant_name} is deprecated. "
            f"Use create_vertical_tool_dependency_provider('{self.vertical_name}')"
            f".{provider_method}() instead."
        )

        def loader() -> Any:
            config = self._get_config()
            return getattr(config, extractor)

        return DeprecatedConstantDescriptor(
            constant_name=constant_name,
            deprecation_message=deprecation_message,
            loader=loader,
            stacklevel=3,  # Extra level for __getattr__
        )

    def add_descriptor(
        self,
        suffix: str,
        extractor: str,
        provider_method: str,
    ) -> None:
        """Add a deprecated constant descriptor.

        Args:
            suffix: Constant name suffix (e.g., "TOOL_DEPENDENCIES")
            extractor: Attribute name on ToolDependencyConfig
            provider_method: Recommended method name on provider class
        """
        constant_name = f"{self.constant_prefix}_{suffix}"
        self._descriptors[constant_name] = self._create_descriptor(
            suffix=suffix,
            extractor=extractor,
            provider_method=provider_method,
        )

    def __getattr__(self, name: str) -> Any:
        """Get a deprecated constant by name.

        Args:
            name: Constant name

        Returns:
            The constant value

        Raises:
            AttributeError: If constant not found
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if name in self._descriptors:
            return self._descriptors[name].__get__(self, type(self))

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def get_constant_names(self) -> Set[str]:
        """Get all registered constant names.

        Returns:
            Set of constant names
        """
        return set(self._descriptors.keys())

    def reset_all(self) -> None:
        """Reset all descriptors and config cache.

        Useful for testing to ensure fresh loads.
        """
        self._config_cache = None
        for descriptor in self._descriptors.values():
            descriptor.reset()


def create_vertical_deprecation_module(
    vertical_name: str,
    yaml_path: Path,
    constant_prefix: str,
    *,
    include_standard: bool = True,
    extra_mappings: Optional[Dict[str, tuple]] = None,
) -> VerticalDeprecationModule:
    """Factory to create deprecated constant descriptors for a vertical.

    Creates a VerticalDeprecationModule with pre-configured descriptors
    for standard tool dependency constants.

    Args:
        vertical_name: Name of the vertical (e.g., "coding", "devops")
        yaml_path: Path to the YAML configuration file
        constant_prefix: Prefix for constant names (e.g., "CODING")
        include_standard: If True, include standard tool dependency constants
        extra_mappings: Additional constant mappings as dict of
            suffix -> (extractor, provider_method)

    Returns:
        Configured VerticalDeprecationModule

    Example:
        _deprecated = create_vertical_deprecation_module(
            vertical_name="coding",
            yaml_path=Path(__file__).parent / "tool_dependencies.yaml",
            constant_prefix="CODING",
        )

        # Access triggers deprecation warning
        deps = _deprecated.CODING_TOOL_DEPENDENCIES
    """
    module = VerticalDeprecationModule(
        vertical_name=vertical_name,
        yaml_path=yaml_path,
        constant_prefix=constant_prefix,
    )

    if include_standard:
        # Standard tool dependency constant mappings
        standard_mappings = {
            "TOOL_DEPENDENCIES": ("dependencies", "get_dependencies"),
            "TOOL_TRANSITIONS": ("transitions", "get_tool_transitions"),
            "TOOL_CLUSTERS": ("clusters", "get_tool_clusters"),
            "TOOL_SEQUENCES": ("sequences", "get_tool_sequences"),
            "REQUIRED_TOOLS": ("required_tools", "get_required_tools"),
            "OPTIONAL_TOOLS": ("optional_tools", "get_optional_tools"),
        }

        for suffix, (extractor, provider_method) in standard_mappings.items():
            module.add_descriptor(
                suffix=suffix,
                extractor=extractor,
                provider_method=provider_method,
            )

    if extra_mappings:
        for suffix, (extractor, provider_method) in extra_mappings.items():
            module.add_descriptor(
                suffix=suffix,
                extractor=extractor,
                provider_method=provider_method,
            )

    logger.debug(
        f"Created deprecation module for '{vertical_name}' "
        f"with {len(module.get_constant_names())} constants"
    )

    return module


def create_module_getattr(
    deprecated_module: VerticalDeprecationModule,
) -> Callable[[str], Any]:
    """Create a __getattr__ function for module-level deprecated constant access.

    Returns a function suitable for use as a module's __getattr__ that
    provides access to deprecated constants with warnings.

    Args:
        deprecated_module: The VerticalDeprecationModule containing descriptors

    Returns:
        A __getattr__ function for the module

    Example:
        _deprecated = create_vertical_deprecation_module(...)

        # Use the returned function as the module's __getattr__
        __getattr__ = create_module_getattr(_deprecated)
    """
    constant_names = deprecated_module.get_constant_names()

    def module_getattr(name: str) -> Any:
        if name in constant_names:
            return getattr(deprecated_module, name)
        raise AttributeError(f"module has no attribute {name!r}")

    return module_getattr


__all__ = [
    "DeprecatedConstantDescriptor",
    "VerticalDeprecationModule",
    "create_vertical_deprecation_module",
    "create_module_getattr",
]
