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

"""Analyzer registry for evaluation components.

Provides centralized access to evaluation analyzers with proper
singleton management and configuration support.

This module acts as a service locator to avoid circular dependencies
between harness.py and code_quality.py.
"""

import logging
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic analyzer type
T = TypeVar("T")


class AnalyzerRegistry:
    """Registry for evaluation analyzers.

    Provides singleton access to analyzers used during evaluation.
    Supports lazy initialization and configuration.

    Example:
        # Get default code quality analyzer
        analyzer = AnalyzerRegistry.get_code_quality_analyzer()

        # Register custom analyzer
        AnalyzerRegistry.register("custom", my_analyzer)

        # Get registered analyzer
        custom = AnalyzerRegistry.get("custom")
    """

    _instances: dict[str, Any] = {}
    _factories: dict[str, callable] = {}

    @classmethod
    def register(cls, name: str, instance: Any) -> None:
        """Register an analyzer instance.

        Args:
            name: Unique name for the analyzer
            instance: Analyzer instance
        """
        cls._instances[name] = instance
        logger.debug(f"Registered analyzer: {name}")

    @classmethod
    def register_factory(cls, name: str, factory: callable) -> None:
        """Register a factory function for lazy instantiation.

        Args:
            name: Unique name for the analyzer
            factory: Callable that creates the analyzer
        """
        cls._factories[name] = factory
        logger.debug(f"Registered analyzer factory: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Get an analyzer by name.

        Args:
            name: Analyzer name

        Returns:
            Analyzer instance or None
        """
        if name in cls._instances:
            return cls._instances[name]

        if name in cls._factories:
            instance = cls._factories[name]()
            cls._instances[name] = instance
            return instance

        return None

    @classmethod
    def get_code_quality_analyzer(cls) -> Any:
        """Get the code quality analyzer.

        Returns:
            CodeQualityAnalyzer instance
        """
        if "code_quality" not in cls._instances:
            from victor.evaluation.code_quality import CodeQualityAnalyzer

            cls._instances["code_quality"] = CodeQualityAnalyzer()
        return cls._instances["code_quality"]

    @classmethod
    def get_pass_at_k_evaluator(cls, **kwargs) -> Any:
        """Get the Pass@k evaluator.

        Args:
            **kwargs: Arguments passed to PassAtKEvaluator

        Returns:
            PassAtKEvaluator instance
        """
        if "pass_at_k" not in cls._instances:
            from victor.evaluation.pass_at_k import PassAtKEvaluator

            cls._instances["pass_at_k"] = PassAtKEvaluator(**kwargs)
        return cls._instances["pass_at_k"]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered analyzers.

        Useful for testing or reconfiguration.
        """
        cls._instances.clear()
        cls._factories.clear()

    @classmethod
    def configure_code_quality(
        cls,
        use_ruff: bool = True,
        use_radon: bool = True,
        style_guide: str = "pep8",
    ) -> None:
        """Configure and register the code quality analyzer.

        Args:
            use_ruff: Whether to use ruff for linting
            use_radon: Whether to use radon for complexity
            style_guide: Style guide to follow
        """
        from victor.evaluation.code_quality import CodeQualityAnalyzer

        analyzer = CodeQualityAnalyzer(
            use_ruff=use_ruff,
            use_radon=use_radon,
            style_guide=style_guide,
        )
        cls.register("code_quality", analyzer)

    @classmethod
    def configure_pass_at_k(
        cls,
        k_values: Optional[list[int]] = None,
        default_n_samples: int = 100,
        temperature: float = 0.8,
    ) -> None:
        """Configure and register the Pass@k evaluator.

        Args:
            k_values: k values to compute (default: [1, 5, 10, 100])
            default_n_samples: Default number of samples per task
            temperature: Sampling temperature for diversity
        """
        from victor.evaluation.pass_at_k import PassAtKEvaluator

        evaluator = PassAtKEvaluator(
            k_values=k_values,
            default_n_samples=default_n_samples,
            temperature=temperature,
        )
        cls.register("pass_at_k", evaluator)


# Convenience functions for common access patterns
def get_code_quality_analyzer():
    """Get the singleton code quality analyzer."""
    return AnalyzerRegistry.get_code_quality_analyzer()


def get_pass_at_k_evaluator(**kwargs):
    """Get the Pass@k evaluator."""
    return AnalyzerRegistry.get_pass_at_k_evaluator(**kwargs)
