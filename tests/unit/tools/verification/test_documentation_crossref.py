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

"""Tests for DocumentationCrossReference module."""

import pytest
from pathlib import Path

from victor.tools.verification import (
    ClaimIssue,
    DocumentationCrossReference,
)


@pytest.fixture
def project_with_tech_debt(tmp_path: Path) -> Path:
    """Create a temporary project with tech debt documentation."""
    docs_dir = tmp_path / "docs" / "10-quality"
    docs_dir.mkdir(parents=True, exist_ok=True)

    tech_debt_file = docs_dir / "TECHNICAL_DEBT.adoc"
    tech_debt_file.write_text("""
= Technical Debt

== TD-CROSS-LAYER: Cross-layer dependencies
[severity=high]
[status=open]
Storage module depends on Index module, creating cross-layer dependency.

== TD-GLOBAL-STATE: Global mutable state
[severity=medium]
Test utilities use global state for convenience.
""")

    return tmp_path


@pytest.fixture
def project_with_roadmap(tmp_path: Path) -> Path:
    """Create a temporary project with roadmap documentation."""
    docs_dir = tmp_path / "docs" / "_internal"
    docs_dir.mkdir(parents=True, exist_ok=True)

    roadmap_file = docs_dir / "roadmap.md"
    roadmap_file.write_text("""
# Roadmap

## [priority=p0] Fix cross-layer dependencies
[priority=p0]
Refactor storage to remove index dependency.

## [priority=p2] Improve test coverage
[priority=p2]
Add integration tests for core modules.
""")

    return tmp_path


class TestDocumentationCrossReference:
    """Tests for DocumentationCrossReference class."""

    def test_is_tracked_debt_match(self, project_with_tech_debt: Path):
        """Test matching issue with tracked debt."""
        crossref = DocumentationCrossReference(project_root=project_with_tech_debt)

        issue = ClaimIssue(
            issue_type="cross_layer_dependency",
            description="Storage depends on Index",
        )

        is_tracked = crossref.is_tracked_debt(issue)

        assert is_tracked

    def test_is_tracked_debt_no_match(self, project_with_tech_debt: Path):
        """Test issue not matching tracked debt."""
        crossref = DocumentationCrossReference(project_root=project_with_tech_debt)

        issue = ClaimIssue(
            issue_type="unrelated_issue",
            description="Some unrelated problem",
        )

        is_tracked = crossref.is_tracked_debt(issue)

        assert not is_tracked

    def test_check_roadmap_alignment(self, project_with_roadmap: Path):
        """Test checking roadmap alignment."""
        crossref = DocumentationCrossReference(project_root=project_with_roadmap)

        issue = ClaimIssue(
            issue_type="cross_layer_dependency",
            description="Fix cross-layer dependencies",
        )

        aligned = crossref.check_roadmap_alignment(issue)

        # Should find alignment with roadmap entry
        assert aligned

    def test_get_doc_references(self, project_with_tech_debt: Path):
        """Test getting documentation references."""
        crossref = DocumentationCrossReference(project_root=project_with_tech_debt)

        issue = ClaimIssue(
            issue_type="cross_layer_dependency",
            description="Storage depends on Index",
        )

        refs = crossref.get_doc_references(issue)

        assert len(refs) > 0
        assert any(
            "TD-CROSS-LAYER" in str(ref) or ref.reference_id == "TD-CROSS-LAYER" for ref in refs
        )

    def test_get_tech_debt_markers(self, project_with_tech_debt: Path):
        """Test getting tech debt markers."""
        crossref = DocumentationCrossReference(project_root=project_with_tech_debt)

        markers = crossref.get_tech_debt_markers()

        assert "TD-CROSS-LAYER" in markers
        assert "TD-GLOBAL-STATE" in markers

    def test_get_roadmap_priorities(self, project_with_roadmap: Path):
        """Test getting roadmap priorities."""
        crossref = DocumentationCrossReference(project_root=project_with_roadmap)

        priorities = crossref.get_roadmap_priorities()

        assert "p0" in priorities
        assert "p2" in priorities
        assert priorities["p0"] >= 1

    def test_reload_documentation(self, project_with_tech_debt: Path):
        """Test reloading documentation."""
        crossref = DocumentationCrossReference(project_root=project_with_tech_debt)

        # Modify documentation
        tech_debt_file = project_with_tech_debt / "docs" / "10-quality" / "TECHNICAL_DEBT.adoc"
        original_content = tech_debt_file.read_text()
        tech_debt_file.write_text(
            original_content + "\n\n== TD-NEW: New issue\nNew issue description.\n"
        )

        crossref.reload_documentation()

        markers = crossref.get_tech_debt_markers()
        assert "TD-NEW" in markers

    def test_without_documentation(self, tmp_path: Path):
        """Test crossref without documentation files."""
        crossref = DocumentationCrossReference(project_root=tmp_path)

        issue = ClaimIssue(issue_type="any", description="Any issue")

        assert not crossref.is_tracked_debt(issue)
        assert not crossref.check_roadmap_alignment(issue)
        assert len(crossref.get_doc_references(issue)) == 0
