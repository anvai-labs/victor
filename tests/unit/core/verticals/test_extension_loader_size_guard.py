# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Guard tests preventing VerticalExtensionLoader from re-growing into a God Class.

After decomposing 19 type-specific factory methods into extension_handlers/,
these tests ensure new extension types are added as handlers — not inlined
back into the loader.
"""

import ast
from pathlib import Path

import pytest

EXTENSION_LOADER_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "victor"
    / "core"
    / "verticals"
    / "extension_loader.py"
)

HANDLERS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "victor"
    / "core"
    / "verticals"
    / "extension_handlers"
)


class TestExtensionLoaderSizeGuard:
    """Prevent VerticalExtensionLoader from growing back into a God Class."""

    def test_line_count_under_threshold(self):
        """extension_loader.py must stay under 1,700 LOC.

        Was 2,001 LOC before decomposition. Now ~1,581 LOC with thin
        delegation methods. This threshold prevents adding new logic
        without extracting to handlers.
        """
        lines = EXTENSION_LOADER_PATH.read_text().splitlines()
        assert len(lines) <= 1700, (
            f"extension_loader.py has {len(lines)} lines (max 1700). "
            f"Extract new extension types to extension_handlers/."
        )

    def test_delegation_methods_are_thin(self):
        """Each get_* delegation method must be <= 6 lines of body.

        This ensures factory logic stays in handlers, not in the loader.
        """
        tree = ast.parse(EXTENSION_LOADER_PATH.read_text())
        DELEGATION_METHODS = {
            "get_middleware",
            "get_safety_extension",
            "get_prompt_contributor",
            "get_mode_config_provider",
            "get_mode_config",
            "get_task_type_hints",
            "get_tool_dependency_provider",
            "get_tool_graph",
            "get_tiered_tool_config",
            "get_rl_config_provider",
            "get_rl_hooks",
            "get_team_spec_provider",
            "get_team_specs",
            "get_capability_provider",
            "get_service_provider",
            "get_composed_chains",
            "get_personas",
            "get_enrichment_strategy",
            "get_workflow_provider",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "VerticalExtensionLoader":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name in DELEGATION_METHODS:
                        body_lines = method.end_lineno - method.lineno
                        assert body_lines <= 6, (
                            f"{method.name}() has {body_lines} body lines (max 6). "
                            f"Move logic to its extension handler."
                        )


class TestExtensionHandlerRegistryCompleteness:
    """Verify all handlers are properly registered."""

    def test_all_expected_handlers_registered(self):
        """Every expected extension type must have a registered handler."""
        # Force-import all handler modules to trigger @register decorators
        from victor.core.verticals.extension_handlers import middleware  # noqa: F401
        from victor.core.verticals.extension_handlers import safety  # noqa: F401
        from victor.core.verticals.extension_handlers import prompt  # noqa: F401
        from victor.core.verticals.extension_handlers import mode_config  # noqa: F401
        from victor.core.verticals.extension_handlers import tool_deps  # noqa: F401
        from victor.core.verticals.extension_handlers import rl  # noqa: F401
        from victor.core.verticals.extension_handlers import team  # noqa: F401
        from victor.core.verticals.extension_handlers import service  # noqa: F401
        from victor.core.verticals.extension_handlers import workflow  # noqa: F401
        from victor.core.verticals.extension_handlers import enrichment  # noqa: F401
        from victor.core.verticals.extension_handlers import chains  # noqa: F401
        from victor.core.verticals.extension_handlers import personas  # noqa: F401
        from victor.core.verticals.extension_handlers.registry import (
            ExtensionHandlerRegistry,
        )

        handlers = ExtensionHandlerRegistry.all_handlers()
        EXPECTED = {
            "middleware",
            "safety",
            "prompt",
            "mode_config",
            "tool_deps",
            "rl_config",
            "rl_hooks",
            "team_spec",
            "capability",
            "service",
            "workflow",
            "enrichment",
            "chains",
            "personas",
        }
        missing = EXPECTED - set(handlers.keys())
        assert not missing, f"Missing handlers: {missing}"

    def test_handler_files_exist(self):
        """Each handler must have its own file in extension_handlers/."""
        expected_files = [
            "middleware.py",
            "safety.py",
            "prompt.py",
            "mode_config.py",
            "tool_deps.py",
            "rl.py",
            "team.py",
            "service.py",
            "workflow.py",
            "enrichment.py",
            "chains.py",
            "personas.py",
        ]
        for filename in expected_files:
            path = HANDLERS_DIR / filename
            assert path.exists(), f"Handler file missing: {path}"

    def test_base_and_registry_exist(self):
        """Infrastructure files must exist."""
        assert (HANDLERS_DIR / "base.py").exists()
        assert (HANDLERS_DIR / "registry.py").exists()
        assert (HANDLERS_DIR / "__init__.py").exists()
