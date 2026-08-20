# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for tree_sitter capability sharing pattern.

Verifies that tree-sitter support in victor core uses the Capability
Registry pattern (entry points + protocols) instead of direct imports
from victor-coding. This ensures any vertical can provide tree-sitter
by registering the same entry point.
"""

import ast
from pathlib import Path

CORE_DIR = Path(__file__).parent.parent.parent.parent / "victor"


class TestTreeSitterCapabilityPattern:
    """Verify tree-sitter is accessed via capability registry, not direct imports."""

    def test_no_direct_tree_sitter_manager_imports(self):
        """Core must not import tree_sitter_manager from victor_coding directly.

        tree_sitter_manager is an implementation detail in victor-coding.
        Core accesses it via CapabilityRegistry + entry point discovery.
        """
        violations = []

        for py_file in CORE_DIR.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                source = py_file.read_text()
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "tree_sitter_manager" in node.module and "victor_coding" in node.module:
                        rel = py_file.relative_to(CORE_DIR.parent)
                        violations.append(f"{rel}:{node.lineno}: from {node.module}")

        assert not violations, (
            "Core must not import tree_sitter_manager from victor_coding.\n"
            "Use CapabilityRegistry or capability_loader.py instead.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )

    def test_capability_loader_provides_tree_sitter(self):
        """capability_loader.py must expose load_tree_sitter_get_parser()."""
        from victor.core.utils.capability_loader import load_tree_sitter_get_parser

        assert callable(load_tree_sitter_get_parser)

    def test_tree_sitter_protocol_in_framework(self):
        """TreeSitterParserProtocol must be defined in framework, not in victor-coding."""
        from victor.framework.vertical_protocols import TreeSitterParserProtocol

        assert hasattr(TreeSitterParserProtocol, "get_parser")

    def test_capability_loader_uses_entry_points(self):
        """capability_loader.py must use entry point discovery for tree_sitter."""
        source = (CORE_DIR / "core" / "utils" / "capability_loader.py").read_text()

        assert (
            "entry_point" in source.lower() or "_try_entry_point" in source
        ), "capability_loader.py should use entry point discovery for tree_sitter"

    def test_no_victor_coding_in_core_tree_sitter_consumers(self):
        """Files that use tree_sitter in core must not import from victor_coding."""
        tree_sitter_consumers = [
            "tools/code_intelligence_tool.py",
            "tools/documentation_tool.py",
            "storage/memory/extractors/tree_sitter_extractor.py",
            "native/python/symbol_extractor.py",
        ]

        violations = []
        for rel_path in tree_sitter_consumers:
            full_path = CORE_DIR / rel_path
            if not full_path.exists():
                continue

            source = full_path.read_text()
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("victor_coding"):
                        violations.append(f"{rel_path}:{node.lineno}: from {node.module}")

        assert not violations, (
            "Core tree-sitter consumers must use CapabilityRegistry, not victor_coding.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )
