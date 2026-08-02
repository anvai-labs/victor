# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for audit_vertical_dependencies()."""

from __future__ import annotations


import pytest

from victor_contracts.verticals.manifest import ExtensionDependency, ExtensionManifest
from victor_contracts.verticals.validation import audit_vertical_dependencies


@pytest.fixture
def tmp_vertical(tmp_path):
    """Create a temporary vertical source directory."""

    def _make(files: dict):
        for name, content in files.items():
            filepath = tmp_path / name
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)
        return str(tmp_path)

    return _make


class TestAuditVerticalDependencies:
    def test_detects_undeclared_third_party_import(self, tmp_vertical):
        src = tmp_vertical(
            {
                "assistant.py": "import chromadb\nfrom pathlib import Path\n",
            }
        )
        manifest = ExtensionManifest(name="test", version="1.0.0")
        report = audit_vertical_dependencies(src, manifest)
        codes = [i.code for i in report.issues]
        assert "undeclared_dependency:chromadb" in codes

    def test_passes_when_deps_match(self, tmp_vertical):
        src = tmp_vertical(
            {
                "assistant.py": "import chromadb\n",
            }
        )
        manifest = ExtensionManifest(
            name="test",
            version="1.0.0",
            extension_dependencies=[
                ExtensionDependency(extension_name="chromadb"),
            ],
        )
        report = audit_vertical_dependencies(src, manifest)
        assert report.ok

    def test_ignores_stdlib_imports(self, tmp_vertical):
        src = tmp_vertical(
            {
                "assistant.py": "import os\nimport json\nfrom pathlib import Path\n",
            }
        )
        manifest = ExtensionManifest(name="test", version="1.0.0")
        report = audit_vertical_dependencies(src, manifest)
        assert report.ok

    def test_ignores_victor_contracts_imports(self, tmp_vertical):
        src = tmp_vertical(
            {
                "assistant.py": "from victor_contracts import VerticalBase\n",
            }
        )
        manifest = ExtensionManifest(name="test", version="1.0.0")
        report = audit_vertical_dependencies(src, manifest)
        assert report.ok
        assert not any(
            issue.code == "undeclared_dependency:victor_contracts" for issue in report.issues
        )

    def test_source_not_found(self):
        report = audit_vertical_dependencies("/nonexistent/path", None)
        assert any(i.code == "source_not_found" for i in report.issues)

    def test_undeclared_deps_are_warnings_not_errors(self, tmp_vertical):
        src = tmp_vertical(
            {
                "assistant.py": "import requests\n",
            }
        )
        manifest = ExtensionManifest(name="test", version="1.0.0")
        report = audit_vertical_dependencies(src, manifest)
        for issue in report.issues:
            if "undeclared" in issue.code:
                assert issue.level == "warning"

    def test_no_manifest_skips_dep_check(self, tmp_vertical):
        src = tmp_vertical(
            {
                "assistant.py": "import chromadb\n",
            }
        )
        report = audit_vertical_dependencies(src, manifest=None)
        # Without manifest, can't compare — no warnings
        assert report.ok
