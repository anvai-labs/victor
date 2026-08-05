# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests that Contracts Boundary architecture doc stays in sync with code."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DOC_PATH = REPO_ROOT / "docs" / "architecture" / "CONTRACTS_BOUNDARY.md"


class TestSDKBoundaryDoc:
    def test_doc_exists(self):
        assert DOC_PATH.exists(), f"Missing: {DOC_PATH}"

    def test_references_key_files(self):
        content = DOC_PATH.read_text()
        key_files = [
            "victor_contracts/verticals/manifest.py",
            "victor/core/verticals/capability_negotiator.py",
            "victor_contracts/testing",
            "victor_contracts/core/plugins.py",
        ]
        for f in key_files:
            assert f in content, f"Doc should reference {f}"

    def test_references_key_classes(self):
        content = DOC_PATH.read_text()
        key_classes = [
            "ExtensionManifest",
            "CapabilityNegotiator",
            "VerticalBase",
            "PluginContext",
            "VictorPlugin",
            "MockPluginContext",
        ]
        for cls in key_classes:
            assert cls in content, f"Doc should reference {cls}"

    def test_describes_entry_points(self):
        content = DOC_PATH.read_text()
        assert "victor.plugins" in content

    def test_referenced_paths_exist(self):
        """Parse backtick-quoted paths from doc and verify they exist."""
        content = DOC_PATH.read_text()
        # Match patterns like `victor_contracts/verticals/manifest.py` or `victor/core/verticals/`
        paths = re.findall(r"`((?:victor|tests|scripts)[^\s`]*\.py)`", content)
        missing = []
        for p in paths:
            # Try both victor-contracts and repo root
            candidates = [REPO_ROOT / p, REPO_ROOT / "victor-contracts" / p]
            if not any(c.exists() for c in candidates):
                missing.append(p)
        assert not missing, f"Doc references files that don't exist: {missing}"
