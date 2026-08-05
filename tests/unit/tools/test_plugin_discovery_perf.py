# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for plugin discovery performance.

Ensures ToolRegistry uses UnifiedEntryPointRegistry instead of
redundant direct importlib.metadata.entry_points() calls.
"""

import ast
from pathlib import Path

REGISTRY_FILE = Path(__file__).parent.parent.parent.parent / "victor" / "tools" / "registry.py"


class TestNoDirectEntryPointScans:
    """ToolRegistry must use UnifiedEntryPointRegistry, not direct scans."""

    def test_no_importlib_entry_points_calls(self):
        """ToolRegistry must not call importlib.metadata.entry_points() directly.

        Direct calls bypass the UnifiedEntryPointRegistry cache, causing
        redundant package scans (~200-500ms each). All entry point access
        should go through get_entry_point_objects() or get_entry_point_values().
        """
        source = REGISTRY_FILE.read_text()
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            # Check for: importlib.metadata.entry_points()
            if isinstance(node, ast.Call):
                func = node.func
                # Match: entry_points() as attribute call
                if isinstance(func, ast.Attribute) and func.attr == "entry_points":
                    # Only flag if it's importlib.metadata.entry_points
                    if isinstance(func.value, ast.Attribute) and func.value.attr == "metadata":
                        violations.append(f"Line {node.lineno}: direct entry_points() call")
                    elif isinstance(func.value, ast.Name) and func.value.id == "metadata":
                        violations.append(f"Line {node.lineno}: direct entry_points() call")

        assert not violations, (
            "ToolRegistry must use UnifiedEntryPointRegistry for entry point discovery.\n"
            "Use get_entry_point_objects() instead of importlib.metadata.entry_points().\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )

    def test_uses_unified_registry(self):
        """ToolRegistry should import from entry_point_registry module."""
        source = REGISTRY_FILE.read_text()
        assert "get_entry_point_objects" in source, (
            "ToolRegistry should use get_entry_point_objects() from "
            "victor.framework.entry_point_registry"
        )

    def test_registry_importable(self):
        """Verify ToolRegistry imports cleanly after changes."""
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        assert hasattr(registry, "discover_plugins")
        assert hasattr(registry, "register_from_entry_points")
