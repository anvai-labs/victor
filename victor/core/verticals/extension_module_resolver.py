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

"""Extension module resolution for verticals.

Extracts module resolution, availability checking, and attribute loading
from ``VerticalExtensionLoader`` into a focused, reusable component.

Responsibilities:
- Build ordered import-candidate lists for optional vertical extensions
- Probe whether a module path is importable without fully loading it
- Lazy-import a module and retrieve a named attribute
- Auto-generate extension class names from vertical class names

This component is consumed by ``VerticalExtensionLoader`` and should not
be used directly by vertical implementations.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any, List, Optional

from victor.core.verticals.import_resolver import vertical_runtime_module_candidates

logger = logging.getLogger(__name__)


class ExtensionModuleResolver:
    """Resolves and loads extension modules for verticals.

    Extracts module resolution, availability checking, and attribute loading
    from VerticalExtensionLoader into a focused, reusable component.
    """

    def __init__(self, pressure_monitor: Any) -> None:
        """Initialise the resolver.

        Args:
            pressure_monitor: An ``ExtensionLoaderPressureMonitor`` instance
                used for recording missing-module lookups.
        """
        self._pressure_monitor = pressure_monitor

    # ------------------------------------------------------------------
    # Candidate resolution
    # ------------------------------------------------------------------

    def resolve_candidates(self, vertical_name: str, suffix: str) -> List[str]:
        """Return possible module paths for an optional vertical extension.

        Delegates to :func:`vertical_runtime_module_candidates` from the
        import resolver, which builds an ordered list of candidates across
        external-package, legacy in-package, and contrib namespaces, with
        mixed-mode ``runtime/`` sub-package support for runtime-owned modules.

        Args:
            vertical_name: The logical name of the vertical (e.g. ``"coding"``).
            suffix: Module suffix within the vertical package
                (e.g. ``"safety"``, ``"middleware"``).

        Returns:
            Ordered list of fully-qualified module paths to probe.
            Empty list when either argument is falsy.
        """
        if not suffix:
            return []

        if not isinstance(vertical_name, str) or not vertical_name:
            return []

        return vertical_runtime_module_candidates(vertical_name, suffix)

    # ------------------------------------------------------------------
    # Availability checking
    # ------------------------------------------------------------------

    def is_available(
        self,
        module_path: str,
        vertical_display_name: str = "",
        caller_class_name: str = "",
    ) -> bool:
        """Return ``True`` when the extension module can be imported.

        Uses :func:`importlib.util.find_spec` to probe module availability
        without performing a full import.  On failure, records the miss via
        the pressure monitor so that the first occurrence is logged while
        subsequent identical misses are suppressed.

        Args:
            module_path: Fully-qualified dotted module path to probe.
            vertical_display_name: Human-readable vertical name for log
                messages (falls back to *caller_class_name* when empty).
            caller_class_name: The ``__name__`` of the calling class, used
                as both the cache-key prefix and the log fallback.

        Returns:
            ``True`` if ``find_spec`` locates the module, ``False`` otherwise.
        """
        if not module_path:
            return False

        try:
            spec = importlib.util.find_spec(module_path)
            if spec is not None:
                return True
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError) as e:
            logger.debug(
                "Optional extension module lookup failed for '%s': %s",
                module_path,
                e,
            )

        cache_key = f"{caller_class_name}:{module_path}"
        display = vertical_display_name or caller_class_name
        if self._pressure_monitor.record_missing_module(cache_key):
            logger.debug(
                "Optional extension module '%s' not found for vertical '%s'; "
                "capability package likely not installed.",
                module_path,
                display,
            )
        return False

    # ------------------------------------------------------------------
    # Attribute loading
    # ------------------------------------------------------------------

    def load_attribute(self, import_path: str, attribute_name: str) -> Any:
        """Import a module and return the named attribute from it.

        Performs a lazy import via :func:`__import__` so the module is only
        loaded when first accessed.

        Args:
            import_path: Fully-qualified dotted module path to import.
            attribute_name: Name of the attribute (typically a class) to
                retrieve from the imported module.

        Returns:
            The requested attribute from the module.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the attribute does not exist on the module.
        """
        module = __import__(import_path, fromlist=[attribute_name])
        return getattr(module, attribute_name)

    # ------------------------------------------------------------------
    # Class-name generation
    # ------------------------------------------------------------------

    def find_available_candidates(
        self,
        vertical_name: str,
        suffix: str,
        vertical_display_name: str = "",
        caller_class_name: str = "",
    ) -> list[str]:
        """Resolve candidates and filter to only available ones.

        Combines resolve_candidates + is_available into a single call,
        eliminating the repeated list-comprehension pattern:
            [p for p in cls._extension_module_candidates(suffix)
             if cls._extension_module_available(p)]

        Args:
            vertical_name: Logical vertical name.
            suffix: Module suffix (e.g. "safety", "middleware").
            vertical_display_name: For log messages.
            caller_class_name: For cache keys.

        Returns:
            List of available module paths (may be empty).
        """
        return [
            path
            for path in self.resolve_candidates(vertical_name, suffix)
            if self.is_available(path, vertical_display_name, caller_class_name)
        ]

    def try_load_from_candidates(
        self,
        candidate_paths: list[str],
        factory_attr: str,
    ) -> Any | None:
        """Try to import a factory function from candidate modules.

        Iterates through candidate paths, imports each module, and looks
        for a callable named *factory_attr*. Returns the first found
        callable or None.

        Args:
            candidate_paths: Ordered module paths to try.
            factory_attr: Name of the callable to look for in each module.

        Returns:
            The callable if found, None otherwise.

        Raises:
            ImportError: If the last candidate fails to import.
        """
        last_error: Exception | None = None
        for module_path in candidate_paths:
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                last_error = exc
                continue
            fn = getattr(module, factory_attr, None)
            if callable(fn):
                return fn
        if last_error:
            raise last_error
        return None

    def auto_generate_class_name(self, vertical_class_name: str, extension_key: str) -> str:
        """Auto-generate a class name from vertical class name and extension key.

        Strips the ``"Assistant"`` suffix from the vertical class name and
        converts the underscore-separated extension key to TitleCase.

        Examples::

            ("CodingAssistant", "safety_extension") -> "CodingSafetyExtension"
            ("DevOpsAssistant", "prompt_contributor") -> "DevOpsPromptContributor"
            ("RAGAssistant", "middleware")            -> "RAGMiddleware"

        Args:
            vertical_class_name: The ``__name__`` of the vertical class
                (e.g. ``"CodingAssistant"``).
            extension_key: Underscore-separated extension identifier
                (e.g. ``"safety_extension"``).

        Returns:
            The generated class name string.
        """
        # "CodingAssistant" -> "Coding"
        vertical_prefix = vertical_class_name.replace("Assistant", "")
        # "safety_extension" -> "SafetyExtension"
        extension_type = extension_key.replace("_", " ").title().replace(" ", "")
        return f"{vertical_prefix}{extension_type}"
