# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for decoupled capability_loader.py — no direct victor_coding imports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestNoCodingImports:
    """Verify coding_support.py has no direct from victor_coding imports."""

    def test_no_direct_victor_coding_imports(self):
        """The module must not contain 'from victor_coding' import statements."""
        source_file = (
            Path(__file__).parent.parent.parent.parent
            / "victor"
            / "core"
            / "utils"
            / "capability_loader.py"
        )
        source = source_file.read_text()
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("victor_coding"):
                    violations.append(f"Line {node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("victor_coding"):
                        violations.append(f"Line {node.lineno}: import {alias.name}")

        assert not violations, (
            "coding_support.py must not import from victor_coding directly.\n"
            "Use entry-point discovery instead.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )

    def test_load_functions_exist(self):
        """The public API must still be available."""
        from victor.core.utils.capability_loader import (
            load_codebase_analyzer_module,
            load_codebase_analyzer_attr,
            load_tree_sitter_get_parser,
            load_coding_analyze_app,
        )

        # Functions exist (may raise ImportError if no provider — that's OK)
        assert callable(load_codebase_analyzer_module)
        assert callable(load_tree_sitter_get_parser)
        assert callable(load_coding_analyze_app)
