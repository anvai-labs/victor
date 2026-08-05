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

"""Dynamic capability definitions for the coding vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the coding vertical with custom functionality.

The module follows the CapabilityLoader's discovery patterns:
1. CAPABILITIES list for batch registration
2. @capability decorator for function-based capabilities
3. Capability classes for complex implementations

Example:
    # Register capabilities with loader
    from victor.framework import CapabilityLoader
    loader = CapabilityLoader()
    loader.load_from_module("victor_coding.capabilities")

    # Or use directly
    from victor_coding.capabilities import (
        get_coding_capabilities,
        CodingCapabilityProvider,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from victor_contracts.capabilities import (
    BaseCapabilityProvider,
    CapabilityEntry,
    CapabilityMetadata,
    CapabilityType,
    OrchestratorCapability,
    build_capability_loader,
    capability,
    load_capability_config,
    store_capability_config,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Config Helpers (P1: Framework CapabilityConfigService Migration)
# =============================================================================


_CODE_STYLE_DEFAULTS: Dict[str, Any] = {
    "formatter": "black",
    "linter": "ruff",
    "max_line_length": 100,
    "enforce_type_hints": True,
}
_TEST_REQUIREMENTS_DEFAULTS: Dict[str, Any] = {
    "min_coverage": 0.0,
    "required_patterns": [],
    "framework": "pytest",
    "run_on_edit": False,
}
_LSP_DEFAULT_LANGUAGES: List[str] = ["python", "typescript", "javascript", "rust", "go"]
_LSP_DEFAULTS: Dict[str, Any] = {
    "languages": list(_LSP_DEFAULT_LANGUAGES),
    "features": {"hover": True, "references": True, "symbols": True},
}
_REFACTOR_DEFAULTS: Dict[str, Any] = {
    "operations": {"rename": True, "extract": True, "inline": True},
    "require_tests": True,
}

# =============================================================================
# Capability Handlers
# =============================================================================


def configure_git_safety(
    orchestrator: Any,
    *,
    block_force_push: bool = True,
    block_main_push: bool = True,
    require_tests_before_commit: bool = False,
    allowed_branches: Optional[List[str]] = None,
) -> None:
    """Configure git safety rules for the orchestrator.

    This capability configures the orchestrator's git safety
    checks to prevent dangerous operations.

    Args:
        orchestrator: Target orchestrator
        block_force_push: Block git push --force
        block_main_push: Block direct push to main/master
        require_tests_before_commit: Require tests pass before commit
        allowed_branches: Whitelist of branches for push
    """
    from victor_coding.safety import CodingSafetyExtension

    safety = CodingSafetyExtension()

    # Configure patterns
    if block_force_push:
        safety.add_dangerous_pattern(r"git\s+push\s+.*--force")
        safety.add_dangerous_pattern(r"git\s+push\s+-f")

    if block_main_push:
        safety.add_dangerous_pattern(r"git\s+push\s+.*\b(main|master)\b")

    config = {
        "require_tests_before_commit": require_tests_before_commit,
        "allowed_branches": allowed_branches or [],
    }

    service_stored = store_capability_config(orchestrator, "git_safety_config", config)

    # Legacy compatibility path while runtime consumers migrate.
    if not service_stored and hasattr(orchestrator, "safety_config"):
        orchestrator.safety_config["git"] = {
            "require_tests_before_commit": require_tests_before_commit,
            "allowed_branches": allowed_branches or [],
        }

    logger.info("Configured git safety rules")


def configure_code_style(
    orchestrator: Any,
    *,
    formatter: str = "black",
    linter: str = "ruff",
    max_line_length: int = 100,
    enforce_type_hints: bool = True,
) -> None:
    """Configure code style preferences for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        formatter: Code formatter to use (black, autopep8, yapf)
        linter: Linter to use (ruff, flake8, pylint)
        max_line_length: Maximum line length
        enforce_type_hints: Whether to enforce type hints
    """
    store_capability_config(
        orchestrator,
        "code_style",
        {
            "formatter": formatter,
            "linter": linter,
            "max_line_length": max_line_length,
            "enforce_type_hints": enforce_type_hints,
        },
    )

    logger.info(f"Configured code style: formatter={formatter}, linter={linter}")


def get_code_style(orchestrator: Any) -> Dict[str, Any]:
    """Get current code style configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Code style configuration dict
    """
    return load_capability_config(orchestrator, "code_style", _CODE_STYLE_DEFAULTS)


def configure_test_requirements(
    orchestrator: Any,
    *,
    min_coverage: float = 0.0,
    required_test_patterns: Optional[List[str]] = None,
    test_framework: str = "pytest",
    run_tests_on_edit: bool = False,
) -> None:
    """Configure test requirements for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        min_coverage: Minimum code coverage percentage
        required_test_patterns: Patterns tests must match
        test_framework: Test framework to use
        run_tests_on_edit: Automatically run tests after edits
    """
    store_capability_config(
        orchestrator,
        "test_config",
        {
            "min_coverage": min_coverage,
            "required_patterns": required_test_patterns or [],
            "framework": test_framework,
            "run_on_edit": run_tests_on_edit,
        },
    )

    logger.info(f"Configured test requirements: framework={test_framework}")


def configure_language_server(
    orchestrator: Any,
    *,
    languages: Optional[List[str]] = None,
    enable_hover: bool = True,
    enable_references: bool = True,
    enable_symbols: bool = True,
) -> None:
    """Configure LSP settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        languages: Languages to enable LSP for
        enable_hover: Enable hover information
        enable_references: Enable find references
        enable_symbols: Enable document symbols
    """
    store_capability_config(
        orchestrator,
        "lsp_config",
        {
            "languages": languages or list(_LSP_DEFAULT_LANGUAGES),
            "features": {
                "hover": enable_hover,
                "references": enable_references,
                "symbols": enable_symbols,
            },
        },
    )

    logger.info(f"Configured LSP for languages: {languages or _LSP_DEFAULT_LANGUAGES}")


def configure_refactoring(
    orchestrator: Any,
    *,
    enable_rename: bool = True,
    enable_extract: bool = True,
    enable_inline: bool = True,
    require_tests: bool = True,
) -> None:
    """Configure refactoring capabilities.

    Args:
        orchestrator: Target orchestrator
        enable_rename: Enable rename refactoring
        enable_extract: Enable extract method/variable
        enable_inline: Enable inline refactoring
        require_tests: Require tests before refactoring
    """
    store_capability_config(
        orchestrator,
        "refactor_config",
        {
            "operations": {
                "rename": enable_rename,
                "extract": enable_extract,
                "inline": enable_inline,
            },
            "require_tests": require_tests,
        },
    )

    logger.info("Configured refactoring capabilities")


# =============================================================================
# Decorated Capability Functions
# =============================================================================


@capability(
    name="coding_git_safety",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Git safety rules for preventing dangerous operations",
)
def coding_git_safety(
    block_force_push: bool = True,
    block_main_push: bool = True,
    **kwargs: Any,
) -> Callable:
    """Git safety capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_git_safety(
            orchestrator,
            block_force_push=block_force_push,
            block_main_push=block_main_push,
            **kwargs,
        )

    return handler


@capability(
    name="coding_style",
    capability_type=CapabilityType.MODE,  # Use MODE for configuration
    version="1.0",
    description="Code style and formatting configuration",
    getter="get_code_style",
)
def coding_style(
    formatter: str = "black",
    linter: str = "ruff",
    **kwargs: Any,
) -> Callable:
    """Code style capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_code_style(
            orchestrator,
            formatter=formatter,
            linter=linter,
            **kwargs,
        )

    return handler


@capability(
    name="coding_tests",
    capability_type=CapabilityType.MODE,  # Use MODE for configuration
    version="1.0",
    description="Test configuration and requirements",
)
def coding_tests(
    min_coverage: float = 0.0,
    test_framework: str = "pytest",
    **kwargs: Any,
) -> Callable:
    """Test requirements capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_test_requirements(
            orchestrator,
            min_coverage=min_coverage,
            test_framework=test_framework,
            **kwargs,
        )

    return handler


@capability(
    name="coding_lsp",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Language server protocol configuration",
)
def coding_lsp(languages: Optional[List[str]] = None, **kwargs: Any) -> Callable:
    """LSP capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_language_server(orchestrator, languages=languages, **kwargs)

    return handler


# =============================================================================
# Capability Provider Class
# =============================================================================


class CodingCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Provider for coding-specific capabilities.

    This class provides a structured way to access and apply
    coding capabilities to an orchestrator. It inherits from
    BaseCapabilityProvider for consistent capability registration
    and discovery across all verticals.

    Example:
        provider = CodingCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_git_safety(orchestrator)
        provider.apply_code_style(orchestrator, formatter="black")

        # Use BaseCapabilityProvider interface
        cap = provider.get_capability("git_safety")
        if cap:
            cap(orchestrator)
    """

    def __init__(self):
        """Initialize the capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "git_safety": configure_git_safety,
            "code_style": configure_code_style,
            "test_requirements": configure_test_requirements,
            "language_server": configure_language_server,
            "refactoring": configure_refactoring,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "git_safety": CapabilityMetadata(
                name="git_safety",
                description="Git safety rules for preventing dangerous operations",
                version="1.0",
                tags=["safety", "git", "version-control"],
            ),
            "code_style": CapabilityMetadata(
                name="code_style",
                description="Code style and formatting configuration",
                version="1.0",
                tags=["style", "formatting", "linting"],
            ),
            "test_requirements": CapabilityMetadata(
                name="test_requirements",
                description="Test configuration and requirements",
                version="1.0",
                tags=["testing", "coverage", "quality"],
            ),
            "language_server": CapabilityMetadata(
                name="language_server",
                description="Language server protocol configuration",
                version="1.0",
                tags=["lsp", "ide", "code-intelligence"],
            ),
            "refactoring": CapabilityMetadata(
                name="refactoring",
                description="Refactoring tool configuration",
                version="1.0",
                dependencies=["language_server"],
                tags=["refactoring", "code-transformation"],
            ),
        }

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        """Return all registered capabilities.

        Returns:
            Dictionary mapping capability names to handler functions.
        """
        return self._capabilities.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata.
        """
        return self._metadata.copy()

    def apply_git_safety(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply git safety capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Git safety options
        """
        configure_git_safety(orchestrator, **kwargs)
        self._applied.add("git_safety")

    def apply_code_style(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply code style capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Code style options
        """
        configure_code_style(orchestrator, **kwargs)
        self._applied.add("code_style")

    def apply_test_requirements(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply test requirements capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Test options
        """
        configure_test_requirements(orchestrator, **kwargs)
        self._applied.add("test_requirements")

    def apply_language_server(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply language server capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: LSP options
        """
        configure_language_server(orchestrator, **kwargs)
        self._applied.add("language_server")

    def apply_refactoring(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply refactoring capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Refactoring options
        """
        configure_refactoring(orchestrator, **kwargs)
        self._applied.add("refactoring")

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all coding capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_git_safety(orchestrator)
        self.apply_code_style(orchestrator)
        self.apply_test_requirements(orchestrator)
        self.apply_language_server(orchestrator)
        self.apply_refactoring(orchestrator)

    def get_applied(self) -> Set[str]:
        """Get set of applied capability names.

        Returns:
            Set of applied capability names
        """
        return self._applied.copy()


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


CAPABILITIES: List[CapabilityEntry] = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_git_safety",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_git_safety",
            description="Git safety rules for preventing dangerous operations",
        ),
        handler=configure_git_safety,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_style",
            capability_type=CapabilityType.MODE,  # Use MODE for configuration
            version="1.0",
            setter="configure_code_style",
            getter="get_code_style",
            description="Code style and formatting configuration",
        ),
        handler=configure_code_style,
        getter_handler=get_code_style,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_tests",
            capability_type=CapabilityType.MODE,  # Use MODE for configuration
            version="1.0",
            setter="configure_test_requirements",
            description="Test configuration and requirements",
        ),
        handler=configure_test_requirements,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_lsp",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_language_server",
            description="Language server protocol configuration",
        ),
        handler=configure_language_server,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_refactoring",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_refactoring",
            description="Refactoring tool configuration",
        ),
        handler=configure_refactoring,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_coding_capabilities() -> List[CapabilityEntry]:
    """Get all coding capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_coding_capability_loader(
    loader_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """Create and populate a host capability loader for the coding vertical.

    Args:
        loader_factory: Host-side factory that returns a compatible capability loader.

    Returns:
        Host capability loader populated with coding capability entries.
    """
    if loader_factory is None:
        raise RuntimeError(
            "create_coding_capability_loader() requires a host loader_factory, such as "
            "victor.framework.CapabilityLoader."
        )

    return build_capability_loader(
        CAPABILITIES,
        loader_factory=loader_factory,
        source_module="victor_coding.capabilities",
    )


# =============================================================================
# SOLID: Centralized Config Storage
# =============================================================================


def get_capability_configs() -> Dict[str, Any]:
    """Get coding capability configurations for centralized storage.

    Returns default coding configuration for VerticalContext storage.
    This replaces direct orchestrator code_style/test_config assignment.

    Returns:
        Dict with default coding capability configurations
    """
    return {
        "code_style": {
            "formatter": "black",
            "linter": "ruff",
            "max_line_length": 100,
            "enforce_type_hints": True,
        },
        "test_config": {
            "min_coverage": 0.0,
            "required_patterns": [],
            "framework": "pytest",
            "run_on_edit": False,
        },
        "git_safety_config": {
            "block_force_push": True,
            "block_main_push": True,
            "require_tests_before_commit": False,
            "allowed_branches": [],
        },
        "lsp_config": {
            "languages": ["python", "typescript", "javascript", "rust", "go"],
            "features": {
                "hover": True,
                "references": True,
                "symbols": True,
            },
        },
        "refactor_config": {
            "operations": {
                "rename": True,
                "extract": True,
                "inline": True,
            },
            "require_tests": True,
        },
    }


__all__ = [
    # Handlers
    "configure_git_safety",
    "configure_code_style",
    "configure_test_requirements",
    "configure_language_server",
    "configure_refactoring",
    "get_code_style",
    # Provider class and base types
    "CodingCapabilityProvider",
    "CapabilityMetadata",  # Re-exported from framework for convenience
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_coding_capabilities",
    "create_coding_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]
