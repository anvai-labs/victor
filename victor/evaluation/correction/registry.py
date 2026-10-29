# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Validator registry with plugin-based auto-discovery.

This module implements the Registry Pattern with automatic discovery
of validator plugins. New validators are found automatically without
modifying existing code (Open/Closed Principle).

Design Patterns:
- Registry Pattern: Central lookup for validators
- Plugin Pattern: Auto-discovery of validator implementations
- Null Object Pattern: GenericCodeValidator as fallback

Enterprise Integration Patterns:
- Service Locator: Central registry for service lookup
- Plugin Architecture: Extensible without modification
"""

import importlib
import importlib.metadata
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .base import BaseCodeValidator
    from .types import Language

logger = logging.getLogger(__name__)


class CodeValidatorRegistry:
    """Registry for language-specific validators with auto-discovery.

    This registry implements the Service Locator pattern with plugin
    auto-discovery. Validators are discovered from the validators/
    directory automatically.

    Features:
    - Automatic discovery of validators in validators/ package
    - Fallback to generic validator for unsupported languages
    - Thread-safe singleton pattern
    - Lazy loading of validators

    Usage:
        registry = CodeValidatorRegistry()
        registry.discover_validators()  # Auto-load from validators/

        validator = registry.get_validator(Language.PYTHON)
        result = validator.validate(code)

    To add a new language:
    1. Create validators/my_language_validator.py
    2. Implement class inheriting BaseCodeValidator
    3. Registry auto-discovers on next discover_validators() call

    External packages (verticals, future LSP-backed validators) can instead declare
    a ``victor.code_validators`` Python entry point pointing at a ``BaseCodeValidator``
    subclass; those are discovered and take precedence over built-ins (FEP-0024),
    with no edits to ``victor/``.
    """

    _instance: Optional["CodeValidatorRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "CodeValidatorRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize registry (only runs once due to singleton)."""
        if CodeValidatorRegistry._initialized:
            return

        from .types import Language

        self._validators: dict[Language, "BaseCodeValidator"] = {}
        self._fallback: Optional["BaseCodeValidator"] = None
        self._discovered: bool = False

        CodeValidatorRegistry._initialized = True
        logger.debug("CodeValidatorRegistry initialized")

    def register(self, validator: "BaseCodeValidator") -> None:
        """Manually register a validator.

        Args:
            validator: Validator instance to register
        """
        for lang in validator.supported_languages:
            self._validators[lang] = validator
            logger.debug(f"Registered validator for {lang.name}: {type(validator).__name__}")

    def unregister(self, language: "Language") -> None:
        """Remove a validator for a language.

        Args:
            language: Language to unregister
        """
        if language in self._validators:
            del self._validators[language]
            logger.debug(f"Unregistered validator for {language.name}")

    def get_validator(self, language: "Language") -> "BaseCodeValidator":
        """Get validator for language, with fallback to generic.

        Args:
            language: Target language

        Returns:
            Appropriate validator (never None)
        """
        # Ensure discovery has run
        if not self._discovered:
            self.discover_validators()

        validator = self._validators.get(language)
        if validator:
            return validator

        # Fallback to generic validator
        if self._fallback is None:
            self._fallback = self._create_fallback()

        return self._fallback

    def has_validator(self, language: "Language") -> bool:
        """Check if a specialized validator exists for language.

        Args:
            language: Language to check

        Returns:
            True if specialized validator registered
        """
        return language in self._validators

    def discover_validators(self) -> int:
        """Auto-discover and register validators.

        Two sources, in order (later registrations override earlier ones for a given
        language, so entry-point validators take precedence over built-ins):

        1. Built-in validators path-scanned from the ``validators/`` package.
        2. Entry-point validators declared under the ``victor.code_validators`` group
           (FEP-0024). This lets external verticals and future LSP-backed validators
           register without editing ``victor/``.

        Returns:
            Total number of validators discovered (built-in + entry-point).
        """
        builtin_count = self._discover_builtin_validators()
        entry_point_count = self._discover_entry_point_validators()
        self._discovered = True
        total = builtin_count + entry_point_count
        logger.debug(
            "Discovered %d validators (%d built-in, %d entry-point)",
            total,
            builtin_count,
            entry_point_count,
        )
        return total

    def _discover_builtin_validators(self) -> int:
        """Path-scan the local ``validators/`` package for built-in validators."""
        from .base import BaseCodeValidator

        validators_package = "victor.evaluation.correction.validators"
        validators_path = Path(__file__).parent / "validators"

        if not validators_path.exists():
            logger.warning(f"Validators directory not found: {validators_path}")
            return 0

        count = 0
        for _, module_name, _ in pkgutil.iter_modules([str(validators_path)]):
            if module_name.startswith("_"):
                continue

            try:
                module = importlib.import_module(f"{validators_package}.{module_name}")

                # Find validator classes in module
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue

                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseCodeValidator)
                        and attr is not BaseCodeValidator
                    ):
                        try:
                            validator = attr()
                            self.register(validator)
                            count += 1
                            logger.debug(
                                f"Discovered validator: {attr.__name__} "
                                f"for {[lang.name for lang in validator.supported_languages]}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to instantiate {attr.__name__}: {e}")

            except Exception as e:
                logger.warning(f"Failed to import validator module {module_name}: {e}")

        return count

    def _discover_entry_point_validators(self) -> int:
        """Load validators declared under the ``victor.code_validators`` entry-point group.

        Each entry point's target may be a ``BaseCodeValidator`` subclass (preferred,
        per FEP-0024 Q1), an already-instantiated validator, or a zero-arg callable
        returning one. Failures loading/instantiating a single entry point are logged
        and skipped (one bad registration must not break discovery).
        """

        try:
            entry_points = importlib.metadata.entry_points(group="victor.code_validators")
        except Exception as e:  # pragma: no cover - defensive: metadata API/stdlib edge
            logger.debug("Could not read victor.code_validators entry points: %s", e)
            return 0

        count = 0
        for ep in entry_points:
            try:
                loaded = ep.load()
            except Exception as e:
                logger.warning("Failed to load code validator entry point %s: %s", ep.name, e)
                continue

            validator = self._coerce_entry_point_validator(loaded, ep.name)
            if validator is None:
                logger.warning("Entry point %s did not yield a BaseCodeValidator instance", ep.name)
                continue

            self.register(validator)
            count += 1
            logger.debug(
                "Discovered entry-point validator: %s for %s",
                ep.name,
                [lang.name for lang in validator.supported_languages],
            )
        return count

    @staticmethod
    def _coerce_entry_point_validator(loaded: Any, name: str) -> Optional["BaseCodeValidator"]:
        """Coerce an entry-point target into a ``BaseCodeValidator`` instance.

        Accepts a class (instantiated), an instance (used directly), or a zero-arg
        callable returning one. Returns None (and logs) if it cannot be coerced.
        """
        from .base import BaseCodeValidator

        try:
            if isinstance(loaded, BaseCodeValidator):
                return loaded
            if isinstance(loaded, type) and issubclass(loaded, BaseCodeValidator):
                return loaded()
            if callable(loaded):
                candidate = loaded()
                if isinstance(candidate, BaseCodeValidator):
                    return candidate
        except Exception as e:
            logger.warning("Failed to instantiate entry-point validator %s: %s", name, e)
        return None

    def _create_fallback(self) -> "BaseCodeValidator":
        """Create the generic fallback validator."""
        from .validators.generic_validator import GenericCodeValidator

        return GenericCodeValidator()

    @property
    def registered_languages(self) -> "set[Language]":
        """Languages with registered validators."""
        return set(self._validators.keys())

    @property
    def all_validators(self) -> "list[BaseCodeValidator]":
        """All registered validators (unique instances)."""
        return list(set(self._validators.values()))

    def reset(self) -> None:
        """Reset registry state (mainly for testing)."""
        self._validators.clear()
        self._fallback = None
        self._discovered = False

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance (for testing only)."""
        cls._instance = None
        cls._initialized = False


def get_registry() -> CodeValidatorRegistry:
    """Get the global validator registry instance.

    Convenience function for accessing the singleton registry.

    Returns:
        The global CodeValidatorRegistry instance
    """
    return CodeValidatorRegistry()
