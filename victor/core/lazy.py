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

"""Lazy initialization utilities for avoiding circular imports.

This module provides reusable patterns for lazy initialization that are used
throughout the Victor codebase to prevent circular import errors.

Patterns Implemented:
1. LazyProperty - Descriptor for lazy-initialized instance properties
2. deferred_import - Safe import with fallback for optional dependencies
3. SingletonFactory - Thread-safe singleton creation with lazy instantiation
4. CircularImportInfo - Documentation of known circular import chains

Usage Examples:
    # LazyProperty
    class MyClass:
        @LazyProperty
        def heavy_dependency(self) -> HeavyClass:
            from victor.heavy import HeavyClass
            return HeavyClass()

    # deferred_import
    classifier = deferred_import(
        'victor.storage.embeddings.task_classifier',
        'TaskTypeClassifier',
        call_method='get_instance'
    )

    # SingletonFactory
    analyzer = SingletonFactory.get_or_create(TaskAnalyzer)

Known Circular Import Chains (documented for future developers):
    1. orchestrator ← code_correction_middleware ← evaluation ← agent_adapter
    2. task_analyzer ← embeddings.task_classifier ← agent modules
    3. intelligent_pipeline ← prompt_builder/mode_controller/etc.
    4. tool_calling/registry ← all adapters
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, overload

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LazyProperty(Generic[T]):
    """Descriptor for lazy-initialized properties with caching.

    This eliminates the common pattern of:
        @property
        def foo(self) -> Foo:
            if self._foo is None:
                self._foo = Foo()
            return self._foo

    Usage:
        class MyClass:
            @LazyProperty
            def expensive_thing(self) -> ExpensiveThing:
                return ExpensiveThing()

    The property is computed once on first access and cached thereafter.
    Thread-safe for the common case (single-threaded initialization).
    """

    def __init__(self, factory: Callable[..., T]) -> None:
        """Initialize with a factory function.

        Args:
            factory: Callable that returns the lazy-initialized value.
                     Receives `self` as first argument when called.
        """
        self._factory = factory
        self._attr_name: Optional[str] = None

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self._attr_name = f"_lazy_{name}"

    @overload
    def __get__(self, obj: None, objtype: Type[Any]) -> "LazyProperty[T]": ...

    @overload
    def __get__(self, obj: object, objtype: Type[Any]) -> T: ...

    def __get__(self, obj: Optional[object], objtype: Type[Any]) -> Any:
        """Get the lazy-initialized value, creating it on first access."""
        if obj is None:
            return self
        if self._attr_name is None:
            raise RuntimeError("LazyProperty used without __set_name__")

        # Check if already initialized
        cached = getattr(obj, self._attr_name, None)
        if cached is not None:
            return cached

        # Initialize and cache
        value = self._factory(obj)
        setattr(obj, self._attr_name, value)
        return value


def deferred_import(
    module_path: str,
    class_name: str,
    fallback: Optional[T] = None,
    call_method: Optional[str] = None,
    init_args: Optional[tuple] = None,
    init_kwargs: Optional[Dict[str, Any]] = None,
    logger_name: str = __name__,
) -> Optional[T]:
    """Safely import and instantiate a class with fallback on error.

    This is useful for breaking circular import chains by deferring imports
    to runtime instead of module load time.

    Args:
        module_path: Full module path (e.g., 'victor.storage.embeddings.task_classifier')
        class_name: Name of the class to import (e.g., 'TaskTypeClassifier')
        fallback: Value to return if import/instantiation fails
        call_method: Optional method to call on the class (e.g., 'get_instance')
        init_args: Positional arguments for instantiation (if no call_method)
        init_kwargs: Keyword arguments for instantiation (if no call_method)
        logger_name: Logger name for warning messages

    Returns:
        Instantiated object or fallback value

    Example:
        # Import and call get_instance()
        classifier = deferred_import(
            'victor.storage.embeddings.task_classifier',
            'TaskTypeClassifier',
            call_method='get_instance'
        )

        # Import and instantiate with args
        processor = deferred_import(
            'victor.processing',
            'DataProcessor',
            init_kwargs={'batch_size': 32}
        )
    """
    try:
        # Dynamic import
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)

        # Call factory method if specified
        if call_method:
            return getattr(cls, call_method)()

        # Otherwise instantiate
        args = init_args or ()
        kwargs = init_kwargs or {}
        return cls(*args, **kwargs)

    except (ImportError, AttributeError, TypeError) as e:
        log = logging.getLogger(logger_name)
        log.debug(f"Deferred import failed: {class_name} from {module_path}: {e}")
        return fallback


class SingletonFactory:
    """Thread-safe factory for lazy singleton initialization.

    Provides a centralized way to manage singleton instances across the
    codebase, with proper thread safety and cleanup for testing.

    Usage:
        # Get or create singleton
        analyzer = SingletonFactory.get_or_create(TaskAnalyzer)

        # With custom factory
        pipeline = SingletonFactory.get_or_create(
            IntelligentPipeline,
            factory=lambda: IntelligentPipeline.create(config)
        )

        # Clear for testing
        SingletonFactory.clear(TaskAnalyzer)
    """

    _instances: Dict[Type[Any], Any] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        service_type: Type[T],
        factory: Optional[Callable[[], T]] = None,
    ) -> T:
        """Get or create a singleton instance.

        Args:
            service_type: The class type to get/create
            factory: Optional factory function; if None, calls service_type()

        Returns:
            The singleton instance
        """
        if service_type not in cls._instances:
            with cls._lock:
                # Double-check inside lock
                if service_type not in cls._instances:
                    if factory is None:
                        cls._instances[service_type] = service_type()
                    else:
                        cls._instances[service_type] = factory()
        return cls._instances[service_type]

    @classmethod
    def has_instance(cls, service_type: Type[Any]) -> bool:
        """Check if a singleton instance exists."""
        return service_type in cls._instances

    @classmethod
    def clear(cls, service_type: Optional[Type[Any]] = None) -> None:
        """Clear singleton instance(s).

        Args:
            service_type: Specific type to clear, or None to clear all
        """
        with cls._lock:
            if service_type:
                cls._instances.pop(service_type, None)
            else:
                cls._instances.clear()

    @classmethod
    def set_instance(cls, service_type: Type[T], instance: T) -> None:
        """Set a singleton instance directly (useful for testing).

        Args:
            service_type: The class type
            instance: The instance to set
        """
        with cls._lock:
            cls._instances[service_type] = instance


@dataclass
class CircularImportInfo:
    """Documents a known circular import chain and its solution.

    This provides structured documentation for why certain lazy
    initialization patterns exist in the codebase.
    """

    module: str
    chain: List[str]
    reason: str
    solution_file: str
    solution_line: int
    solution_type: str  # "deferred_import", "lazy_property", "type_checking"
    fixed: bool = True


# Registry of known circular import chains (for documentation)
KNOWN_CIRCULAR_IMPORTS: Dict[str, CircularImportInfo] = {
    "orchestrator_evaluation": CircularImportInfo(
        module="orchestrator.py",
        chain=[
            "orchestrator",
            "code_correction_middleware",
            "evaluation.correction",
            "evaluation.__init__",
            "agent_adapter",
            "orchestrator",
        ],
        reason="Evaluation system needs orchestrator type hints; agent_adapter references orchestrator",
        solution_file="victor/agent/orchestrator.py",
        solution_line=620,
        solution_type="deferred_import",
    ),
    "task_analyzer_embeddings": CircularImportInfo(
        module="task_analyzer.py",
        chain=[
            "task_analyzer",
            "embeddings.task_classifier",
            "agent modules (potential)",
        ],
        reason="Embeddings classifiers might reference agent modules dynamically",
        solution_file="victor/agent/task_analyzer.py",
        solution_line=192,
        solution_type="deferred_import_in_property",
    ),
    "intelligent_pipeline_components": CircularImportInfo(
        module="runtime_intelligence_pipeline.py",
        chain=[
            "runtime_intelligence_pipeline",
            "prompt_builder",
            "mode_controller",
            "quality_scorer",
            "grounding_verifier",
            "resilient_executor",
        ],
        reason="Pipeline components might reference each other or orchestrator",
        solution_file="victor/agent/runtime_intelligence_pipeline.py",
        solution_line=196,
        solution_type="deferred_import_async",
    ),
    "tool_calling_registry": CircularImportInfo(
        module="tool_calling/registry.py",
        chain=[
            "registry",
            "adapters (6 classes)",
            "each adapter imports base/providers",
        ],
        reason="Registry needs all adapters but adapters need base classes",
        solution_file="victor/agent/tool_calling/registry.py",
        solution_line=81,
        solution_type="deferred_import",
    ),
    "orchestrator_integration": CircularImportInfo(
        module="orchestrator.py",
        chain=[
            "orchestrator",
            "orchestrator_integration",
            "runtime_intelligence_pipeline",
            "orchestrator (type hints)",
        ],
        reason="Integration bridge needs orchestrator types but is imported by orchestrator",
        solution_file="victor/agent/orchestrator.py",
        solution_line=58,
        solution_type="type_checking",
    ),
}


def get_circular_import_info(key: str) -> Optional[CircularImportInfo]:
    """Get documentation for a known circular import chain.

    Args:
        key: The chain identifier

    Returns:
        CircularImportInfo or None if not found
    """
    return KNOWN_CIRCULAR_IMPORTS.get(key)


def list_circular_imports() -> List[str]:
    """List all documented circular import chain keys."""
    return list(KNOWN_CIRCULAR_IMPORTS.keys())


__all__ = [
    "LazyProperty",
    "deferred_import",
    "SingletonFactory",
    "CircularImportInfo",
    "KNOWN_CIRCULAR_IMPORTS",
    "get_circular_import_info",
    "list_circular_imports",
]
