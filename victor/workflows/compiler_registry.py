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

"""Thread-safe registry for third-party workflow compiler plugins.

Purpose:
    Allow third-party verticals to register custom workflow compilers
    without modifying Victor's core code.

Refactored in Phase 4 to use ItemRegistry base class for:
- Thread-safe singleton pattern
- Consistent registry interface
- Proper test isolation via reset_instance()

Example (Third-party package):
    from victor.workflows.plugins import WorkflowCompilerPlugin
    from victor.workflows.compiler_registry import register_compiler

    class SecurityCompilerPlugin(WorkflowCompilerPlugin):
        def compile(self, source, *, workflow_name=None, validate=True):
            # Security-specific compilation
            return self._compile_security_workflow(source)

    # Register the plugin
    register_compiler("security", SecurityCompilerPlugin)

    # Later, use the plugin
    from victor.workflows.compiler_registry import get_compiler
    compiler = get_compiler("security")
    result = compiler.compile("security_workflow.yaml")
"""

from __future__ import annotations

import logging
from typing import Any, List, Type

from victor.core.registry import ItemRegistry

logger = logging.getLogger(__name__)


class WorkflowCompilerRegistry(ItemRegistry["WorkflowCompilerRegistry"]):
    """Thread-safe singleton registry for workflow compiler plugins.

    Provides centralized registration and retrieval of third-party
    workflow compilers. Inherits thread-safe singleton pattern from
    ItemRegistry.

    Example:
        registry = WorkflowCompilerRegistry.get_instance()
        registry.register_compiler("security", SecurityCompilerPlugin)
        compiler = registry.get_compiler("security", enable_caching=True)
    """

    def register_compiler(self, name: str, compiler_class: Type[Any]) -> None:
        """Register a third-party workflow compiler plugin.

        Args:
            name: Unique name for the compiler (e.g., "security", "ml-pipeline")
            compiler_class: Plugin class implementing WorkflowCompilerPlugin protocol

        Example:
            registry.register_compiler("security", SecurityCompilerPlugin)
        """
        if self.contains(name):
            logger.warning(f"Compiler plugin '{name}' already registered, overwriting")

        self.register(name, compiler_class)
        logger.info(f"Registered compiler plugin: {name} -> {compiler_class.__name__}")

    def unregister_compiler(self, name: str) -> None:
        """Unregister a compiler plugin.

        Args:
            name: Name of the compiler to unregister

        Raises:
            KeyError: If name is not registered
        """
        if not self.contains(name):
            raise KeyError(f"No compiler plugin registered for: {name}")

        self.unregister(name)
        logger.info(f"Unregistered compiler plugin: {name}")

    def get_compiler(self, name: str, **options: Any) -> Any:
        """Get an instance of a registered compiler plugin.

        Args:
            name: Name of the registered compiler
            **options: Options to pass to the compiler constructor

        Returns:
            Compiler instance

        Raises:
            ValueError: If name is not registered

        Example:
            compiler = registry.get_compiler("security", enable_caching=True)
            compiled = compiler.compile("workflow.yaml")
        """
        compiler_class = self.get(name)
        if compiler_class is None:
            registered = ", ".join(self.list_names())
            raise ValueError(
                f"Unknown compiler plugin: '{name}'. "
                f"Registered compilers: {registered or '(none)'}"
            )

        return compiler_class(**options)

    def is_compiler_registered(self, name: str) -> bool:
        """Check if a compiler plugin is registered.

        Args:
            name: Name of the compiler to check

        Returns:
            True if registered, False otherwise
        """
        return self.contains(name)

    def list_compilers(self) -> List[str]:
        """List all registered compiler plugin names.

        Returns:
            List of registered compiler names
        """
        return self.list_names()


# =============================================================================
# Module-level convenience functions (backward compatibility)
# =============================================================================


def register_compiler(name: str, compiler_class: Type[Any]) -> None:
    """Register a third-party workflow compiler plugin.

    Args:
        name: Unique name for the compiler (e.g., "security", "ml-pipeline")
        compiler_class: Plugin class implementing WorkflowCompilerPlugin protocol

    Example:
        from victor.workflows.compiler_registry import register_compiler
        from victor.workflows.plugins import WorkflowCompilerPlugin

        class MyCustomPlugin(WorkflowCompilerPlugin):
            def compile(self, source, *, workflow_name=None, validate=True):
                # Custom compilation logic
                pass

        register_compiler("my_custom", MyCustomPlugin)
    """
    WorkflowCompilerRegistry.get_instance().register_compiler(name, compiler_class)


def unregister_compiler(name: str) -> None:
    """Unregister a compiler plugin.

    Args:
        name: Name of the compiler to unregister

    Raises:
        KeyError: If name is not registered
    """
    WorkflowCompilerRegistry.get_instance().unregister_compiler(name)


def get_compiler(name: str, **options: Any) -> Any:
    """Get an instance of a registered compiler plugin.

    Args:
        name: Name of the registered compiler
        **options: Options to pass to the compiler constructor

    Returns:
        Compiler instance

    Raises:
        ValueError: If name is not registered

    Example:
        from victor.workflows.compiler_registry import get_compiler

        compiler = get_compiler("security", enable_caching=True)
        compiled = compiler.compile("workflow.yaml")
    """
    return WorkflowCompilerRegistry.get_instance().get_compiler(name, **options)


def is_registered(name: str) -> bool:
    """Check if a compiler plugin is registered.

    Args:
        name: Name of the compiler to check

    Returns:
        True if registered, False otherwise
    """
    return WorkflowCompilerRegistry.get_instance().is_compiler_registered(name)


def list_compilers() -> List[str]:
    """List all registered compiler plugin names.

    Returns:
        List of registered compiler names
    """
    return WorkflowCompilerRegistry.get_instance().list_compilers()


__all__ = [
    # Class-based API
    "WorkflowCompilerRegistry",
    # Function-based API (backward compatibility)
    "register_compiler",
    "unregister_compiler",
    "get_compiler",
    "is_registered",
    "list_compilers",
]
