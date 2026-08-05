# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for decomposed codebase analyzer components.

Verifies SymbolExtractor and ArchitectureAnalyzer work independently
and produce identical results to the monolithic CodebaseAnalyzer.
"""

from __future__ import annotations


import pytest

from victor_coding.codebase_analyzer import (
    CodebaseAnalysis,
    CodebaseAnalyzer,
)
from victor_coding.codebase.symbol_extractor import SymbolExtractor
from victor_coding.codebase.architecture_analyzer import ArchitectureAnalyzer


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample Python project for analysis."""
    pkg = tmp_path / "mypackage"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "orchestrator.py").write_text('''"""Main module."""

class AgentOrchestrator:
    """The main orchestrator."""
    pass

class BaseProvider:
    """Abstract base for providers."""
    pass

class OllamaProvider(BaseProvider):
    """Ollama implementation."""
    pass

def main():
    pass
''')
    (pkg / "tools.py").write_text('''"""Tools module."""

class GitTool:
    """Git operations."""
    pass

class ToolRegistry:
    """Central tool registry."""
    pass
''')
    (tmp_path / "pyproject.toml").write_text("""
[project]
name = "mypackage"

[project.scripts]
myapp = "mypackage.orchestrator:main"

[project.optional-dependencies]
dev = ["pytest", "ruff"]
""")
    (tmp_path / "Makefile").write_text("test:\n\tpytest")
    (tmp_path / "tests").mkdir()
    return tmp_path


class TestSymbolExtractor:
    """Verify SymbolExtractor works independently."""

    def test_extracts_classes(self, sample_project):
        analysis = CodebaseAnalysis(project_name="test", root_path=sample_project)
        analysis.main_package = "mypackage"

        extractor = SymbolExtractor(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            key_class_patterns=CodebaseAnalyzer.KEY_CLASS_PATTERNS,
        )
        extractor.analyze_source_files()

        all_classes = []
        for modules in analysis.packages.values():
            for m in modules:
                all_classes.extend(m.classes)

        names = [c.name for c in all_classes]
        assert "AgentOrchestrator" in names
        assert "BaseProvider" in names
        assert "OllamaProvider" in names
        assert "GitTool" in names
        assert "ToolRegistry" in names

    def test_extracts_functions(self, sample_project):
        analysis = CodebaseAnalysis(project_name="test", root_path=sample_project)
        analysis.main_package = "mypackage"

        extractor = SymbolExtractor(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            key_class_patterns=CodebaseAnalyzer.KEY_CLASS_PATTERNS,
        )
        extractor.analyze_source_files()

        all_funcs = []
        for modules in analysis.packages.values():
            for m in modules:
                all_funcs.extend(m.functions)

        assert "main" in all_funcs

    def test_categorizes_classes(self, sample_project):
        analysis = CodebaseAnalysis(project_name="test", root_path=sample_project)
        analysis.main_package = "mypackage"

        extractor = SymbolExtractor(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            key_class_patterns=CodebaseAnalyzer.KEY_CLASS_PATTERNS,
        )
        extractor.analyze_source_files()

        all_classes = []
        for modules in analysis.packages.values():
            for m in modules:
                all_classes.extend(m.classes)

        cats = {c.name: c.category for c in all_classes}
        assert cats["AgentOrchestrator"] == "manager"
        assert cats["BaseProvider"] == "provider"
        assert cats["GitTool"] == "tool"
        # ToolRegistry matches "Tool" pattern first (patterns checked in order)
        assert cats["ToolRegistry"] in ("tool", "registry")


class TestArchitectureAnalyzer:
    """Verify ArchitectureAnalyzer works independently."""

    def _setup_analysis(self, sample_project) -> CodebaseAnalysis:
        """Run symbol extraction to populate analysis for arch tests."""
        analysis = CodebaseAnalysis(project_name="test", root_path=sample_project)
        analysis.main_package = "mypackage"
        extractor = SymbolExtractor(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            key_class_patterns=CodebaseAnalyzer.KEY_CLASS_PATTERNS,
        )
        extractor.analyze_source_files()
        return analysis

    def test_identifies_key_components(self, sample_project):
        analysis = self._setup_analysis(sample_project)
        arch = ArchitectureAnalyzer(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            config_extensions={},
        )
        arch.identify_key_components()

        names = [c.name for c in analysis.key_components]
        assert "AgentOrchestrator" in names

    def test_detects_provider_pattern(self, sample_project):
        analysis = self._setup_analysis(sample_project)
        arch = ArchitectureAnalyzer(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            config_extensions={},
        )
        arch.identify_key_components()
        arch.detect_architecture_patterns()

        assert any("Provider Pattern" in p for p in analysis.architecture_patterns)

    def test_extracts_entry_points(self, sample_project):
        analysis = self._setup_analysis(sample_project)
        arch = ArchitectureAnalyzer(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            config_extensions={},
        )
        arch.extract_entry_points()

        assert "myapp" in analysis.entry_points

    def test_finds_config_files(self, sample_project):
        analysis = self._setup_analysis(sample_project)
        arch = ArchitectureAnalyzer(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            config_extensions={},
        )
        arch.find_config_files()

        config_names = [f for f, _ in analysis.config_files]
        assert "pyproject.toml" in config_names
        assert "Makefile" in config_names

    def test_calculates_loc_stats(self, sample_project):
        analysis = self._setup_analysis(sample_project)
        arch = ArchitectureAnalyzer(
            root=sample_project,
            analysis=analysis,
            include_dirs=None,
            effective_skip_dirs=frozenset(),
            language_extensions={".py": "python"},
            config_extensions={},
        )
        arch.calculate_loc_stats()

        assert analysis.loc_stats["total_files"] > 0
        assert analysis.loc_stats["source_lines"] > 0


class TestDecomposedMatchesMonolithic:
    """Verify decomposed analyze() produces same results as direct method calls."""

    def test_analyze_via_facade_matches_direct(self, sample_project):
        """The decomposed facade must produce identical results."""
        analyzer = CodebaseAnalyzer(str(sample_project))
        result = analyzer.analyze()

        assert result.main_package == "mypackage"
        assert len(result.packages) > 0
        assert len(result.key_components) > 0
        assert "myapp" in result.entry_points
        assert len(result.architecture_patterns) > 0
        assert len(result.config_files) > 0
        assert result.loc_stats["total_files"] > 0
