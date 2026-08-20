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

"""Tests for grounding verification."""

import tempfile
from pathlib import Path

import pytest

from victor.agent.grounding_verifier import (
    GroundingIssue,
    GroundingVerifier,
    IssueSeverity,
    IssueType,
    VerificationResult,
    VerifierConfig,
)


class TestGroundingIssue:
    """Tests for GroundingIssue dataclass."""

    def test_creation(self):
        """Test creating a grounding issue."""
        issue = GroundingIssue(
            issue_type=IssueType.FILE_NOT_FOUND,
            severity=IssueSeverity.HIGH,
            description="File 'nonexistent.py' does not exist",
            reference="nonexistent.py",
        )

        assert issue.issue_type == IssueType.FILE_NOT_FOUND
        assert issue.severity == IssueSeverity.HIGH
        assert "nonexistent.py" in issue.description


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_add_issue_reduces_confidence(self):
        """Adding issues should reduce confidence."""
        result = VerificationResult(is_grounded=True, confidence=1.0)

        result.add_issue(
            GroundingIssue(
                issue_type=IssueType.FILE_NOT_FOUND,
                severity=IssueSeverity.HIGH,
                description="Test",
                reference="test.py",
            )
        )

        assert result.confidence < 1.0
        assert len(result.issues) == 1

    def test_severity_affects_confidence_penalty(self):
        """Higher severity should reduce confidence more."""
        result_low = VerificationResult(is_grounded=True, confidence=1.0)
        result_high = VerificationResult(is_grounded=True, confidence=1.0)

        result_low.add_issue(
            GroundingIssue(
                issue_type=IssueType.PATH_INVALID,
                severity=IssueSeverity.LOW,
                description="Test",
                reference="test.py",
            )
        )

        result_high.add_issue(
            GroundingIssue(
                issue_type=IssueType.FILE_NOT_FOUND,
                severity=IssueSeverity.HIGH,
                description="Test",
                reference="test.py",
            )
        )

        assert result_low.confidence > result_high.confidence

    def test_generate_feedback_prompt_empty_issues(self):
        """Should return empty string when no issues."""
        result = VerificationResult(is_grounded=True, confidence=1.0)
        assert result.generate_feedback_prompt() == ""

    def test_generate_feedback_prompt_file_not_found(self):
        """Should generate feedback for file not found issues."""
        result = VerificationResult(is_grounded=False, confidence=0.5)
        result.add_issue(
            GroundingIssue(
                issue_type=IssueType.FILE_NOT_FOUND,
                severity=IssueSeverity.HIGH,
                description="File 'missing.py' not found",
                reference="missing.py",
            )
        )

        feedback = result.generate_feedback_prompt()
        assert "GROUNDING CORRECTION REQUIRED" in feedback
        assert "missing.py" in feedback
        assert "Use read or ls" in feedback

    def test_generate_feedback_prompt_symbol_not_found(self):
        """Should generate feedback for symbol not found issues."""
        result = VerificationResult(is_grounded=False, confidence=0.5)
        result.add_issue(
            GroundingIssue(
                issue_type=IssueType.SYMBOL_NOT_FOUND,
                severity=IssueSeverity.MEDIUM,
                description="Symbol 'process_data' not found",
                reference="process_data",
            )
        )

        feedback = result.generate_feedback_prompt()
        assert "GROUNDING CORRECTION REQUIRED" in feedback
        assert "process_data" in feedback
        assert "code_search" in feedback

    def test_generate_feedback_prompt_code_mismatch(self):
        """Should generate feedback for code mismatch issues."""
        result = VerificationResult(is_grounded=False, confidence=0.5)
        result.add_issue(
            GroundingIssue(
                issue_type=IssueType.CODE_MISMATCH,
                severity=IssueSeverity.HIGH,
                description="Code doesn't match file content",
                reference="main.py:15",
            )
        )

        feedback = result.generate_feedback_prompt()
        assert "GROUNDING CORRECTION REQUIRED" in feedback
        assert "main.py" in feedback
        assert "Quote code exactly" in feedback

    def test_generate_feedback_prompt_fabricated_content(self):
        """Should generate feedback for fabricated content issues."""
        result = VerificationResult(is_grounded=False, confidence=0.3)
        result.add_issue(
            GroundingIssue(
                issue_type=IssueType.FABRICATED_CONTENT,
                severity=IssueSeverity.CRITICAL,
                description="Content appears fabricated",
                reference="imaginary_class.py",
            )
        )

        feedback = result.generate_feedback_prompt()
        assert "GROUNDING CORRECTION REQUIRED" in feedback
        assert "Fabricated content" in feedback
        assert "tool output" in feedback

    def test_generate_feedback_prompt_includes_suggestion(self):
        """Should include issue suggestions in feedback."""
        result = VerificationResult(is_grounded=False, confidence=0.5)
        result.add_issue(
            GroundingIssue(
                issue_type=IssueType.FILE_NOT_FOUND,
                severity=IssueSeverity.HIGH,
                description="File not found",
                reference="config.py",
                suggestion="Did you mean 'settings.py'?",
            )
        )

        feedback = result.generate_feedback_prompt()
        assert "Did you mean 'settings.py'?" in feedback

    def test_generate_feedback_prompt_max_issues(self):
        """Should respect max_issues parameter."""
        result = VerificationResult(is_grounded=False, confidence=0.1)
        # Add 5 issues
        for i in range(5):
            result.add_issue(
                GroundingIssue(
                    issue_type=IssueType.FILE_NOT_FOUND,
                    severity=IssueSeverity.MEDIUM,
                    description=f"File {i} not found",
                    reference=f"file{i}.py",
                )
            )

        # Default max is 3
        feedback = result.generate_feedback_prompt(max_issues=2)
        # Should only include 2 file references (grouped together)
        assert feedback.count("file") < 5

    def test_generate_feedback_prompt_prioritizes_severity(self):
        """Should include critical issues first."""
        result = VerificationResult(is_grounded=False, confidence=0.5)
        # Add low severity first
        result.add_issue(
            GroundingIssue(
                issue_type=IssueType.PATH_INVALID,
                severity=IssueSeverity.LOW,
                description="Minor path issue",
                reference="low.py",
            )
        )
        # Add critical severity second
        result.add_issue(
            GroundingIssue(
                issue_type=IssueType.FABRICATED_CONTENT,
                severity=IssueSeverity.CRITICAL,
                description="Fabricated content",
                reference="critical.py",
            )
        )

        feedback = result.generate_feedback_prompt(max_issues=1)
        # Critical issue should be included
        assert "Fabricated content" in feedback


class TestGroundingVerifier:
    """Tests for GroundingVerifier."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create test files
            (root / "main.py").write_text('''
"""Main entry point."""

def main():
    """Main function."""
    print("Hello, world!")

class Application:
    """Main application class."""
    pass
''')

            (root / "utils.py").write_text('''
"""Utility functions."""

def helper():
    """Helper function."""
    return 42
''')

            (root / "src").mkdir()
            (root / "src" / "module.py").write_text('''
"""A module."""

class MyClass:
    pass
''')

            yield root

    @pytest.fixture
    def verifier(self, temp_project):
        """Create a verifier for the temp project."""
        return GroundingVerifier(project_root=str(temp_project))

    def test_extract_file_references(self, verifier):
        """Should extract file paths from response."""
        response = """
        Check the `main.py` file for the entry point.
        The utils.py module contains helper functions.
        Also look at src/module.py for additional classes.
        """

        paths = verifier.extract_file_references(response)

        assert "main.py" in paths
        assert "utils.py" in paths
        assert "src/module.py" in paths

    def test_extract_line_references_includes_rust_files(self, verifier):
        """Should extract exact line references across supported code extensions."""
        response = "See src/storage/engine.rs:38 and main.py lines 2-4."

        refs = verifier.extract_line_references(response)

        assert {
            "path": "src/storage/engine.rs",
            "start_line": 38,
            "end_line": 38,
        } in refs
        assert {"path": "main.py", "start_line": 2, "end_line": 4} in refs

    def test_extract_code_snippets(self, verifier):
        """Should extract code blocks from response."""
        response = """
        Here's the code:

        ```python
        def example():
            return 42
        ```

        And another:

        ```javascript
        function test() {}
        ```
        """

        snippets = verifier.extract_code_snippets(response)

        assert len(snippets) == 2
        assert "def example():" in snippets[0]["code"]
        assert "function test()" in snippets[1]["code"]

    def test_extract_symbols(self, verifier):
        """Should extract symbol references."""
        response = """
        The `class Application` is the main class.
        Use `def main` to start the application.
        The function helper does the work.
        """

        symbols = verifier.extract_symbols(response)

        assert "Application" in symbols
        assert "main" in symbols
        assert "helper" in symbols

    @pytest.mark.asyncio
    async def test_verify_ignores_explanatory_nouns_as_symbols(self, verifier):
        """Common prose nouns should not trigger symbol verification loops."""
        response = "The `information` and `findings` are enough to answer the question."

        result = await verifier.verify(response)

        assert not any(
            issue.issue_type == IssueType.SYMBOL_NOT_FOUND
            and issue.reference in {"information", "findings"}
            for issue in result.issues
        )

    @pytest.mark.asyncio
    async def test_verify_existing_file_paths(self, verifier):
        """Should verify existing file paths."""
        response = "Check the main.py file for details."

        result = await verifier.verify(response)

        assert "main.py" in result.verified_references
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_verify_flags_line_reference_outside_file_bounds(self, verifier):
        """Should flag exact file:line claims that exceed the file length."""
        response = "The implementation is in main.py:999."

        result = await verifier.verify(response)

        assert any(
            issue.issue_type == IssueType.UNVERIFIABLE and issue.reference == "main.py:999"
            for issue in result.issues
        )
        assert "main.py:999" in result.unverified_references

    @pytest.mark.asyncio
    async def test_verify_nonexistent_file_paths(self, verifier):
        """Should flag nonexistent file paths."""
        response = "Check the nonexistent.py file for details."

        result = await verifier.verify(response)

        assert any(issue.issue_type == IssueType.FILE_NOT_FOUND for issue in result.issues)
        assert result.confidence < 1.0

    @pytest.mark.asyncio
    async def test_verify_code_snippet_matches(self, verifier):
        """Should verify matching code snippets."""
        response = """
        The main function looks like:

        ```python
        def main():
            print("Hello, world!")
        ```
        """

        result = await verifier.verify(response, context={"files_read": ["main.py"]})

        # Should find partial match
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_verify_fabricated_code_snippet(self, temp_project):
        """Should flag likely fabricated code."""
        # Use skip_generated_code=False to actually verify the code snippet
        config = VerifierConfig(skip_generated_code=False)
        verifier = GroundingVerifier(project_root=str(temp_project), config=config)

        response = """
        Here's the implementation:

        ```python
        def example_function():
            # TODO: implement
            pass
        ```
        """

        result = await verifier.verify(response)

        # Should detect fabrication indicators
        assert any(
            issue.issue_type in (IssueType.FABRICATED_CONTENT, IssueType.UNVERIFIABLE)
            for issue in result.issues
        )

    @pytest.mark.asyncio
    async def test_grounding_threshold(self, verifier):
        """Should respect confidence threshold."""
        config = VerifierConfig(min_confidence=0.9)
        strict_verifier = GroundingVerifier(project_root=verifier.project_root, config=config)

        response = "Check nonexistent1.py, nonexistent2.py, and nonexistent3.py"

        result = await strict_verifier.verify(response)

        assert not result.is_grounded

    @pytest.mark.asyncio
    async def test_strict_mode_fails_on_any_issue(self, verifier):
        """Strict mode should fail on any issue."""
        config = VerifierConfig(strict_mode=True)
        strict_verifier = GroundingVerifier(project_root=verifier.project_root, config=config)

        response = "Check nonexistent.py"

        result = await strict_verifier.verify(response)

        assert not result.is_grounded

    @pytest.mark.asyncio
    async def test_verify_with_context_files(self, verifier):
        """Should check context files during verification."""
        response = "The main function is defined in main.py"

        result = await verifier.verify(response, context={"files_read": ["main.py"]})

        assert "main.py" in result.verified_references

    @pytest.mark.asyncio
    async def test_clear_cache(self, verifier):
        """Should clear file cache."""
        # Access a file to populate cache
        response = "Check main.py"
        await verifier.verify(response)

        # Clear cache
        verifier.clear_cache()

        # Cache should be empty
        assert len(verifier._file_cache) == 0
        assert verifier._existing_files is None

    @pytest.mark.asyncio
    async def test_empty_response(self, verifier):
        """Should handle empty response."""
        result = await verifier.verify("")

        assert result.is_grounded
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_path_with_incorrect_directory(self, verifier):
        """Should flag paths with wrong directory."""
        response = "Check wrong/path/main.py for the entry point."

        result = await verifier.verify(response)

        # Should find partial match but note path is wrong
        issues_with_suggestions = [i for i in result.issues if i.suggestion]
        assert len(issues_with_suggestions) > 0 or result.confidence < 1.0


class TestCodeMatching:
    """Tests for code matching logic."""

    @pytest.fixture
    def verifier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield GroundingVerifier(project_root=tmpdir)

    def test_exact_match(self, verifier):
        """Should match identical code."""
        content = "def foo():\n    return 42"
        snippet = "def foo():\n    return 42"

        assert verifier._code_matches(snippet, content)

    def test_whitespace_tolerance(self, verifier):
        """Should match despite whitespace differences."""
        content = "def foo():\n    return 42"
        snippet = "def foo():    return 42"  # Different whitespace

        assert verifier._code_matches(snippet, content)

    def test_partial_match(self, verifier):
        """Should match partial code."""
        content = """
def foo():
    x = 1
    y = 2
    return x + y

def bar():
    pass
"""
        snippet = """
def foo():
    x = 1
    y = 2
"""

        assert verifier._code_matches(snippet, content)

    def test_no_match(self, verifier):
        """Should not match completely different code."""
        content = "def foo(): return 42"
        snippet = "class Bar: pass"

        assert not verifier._code_matches(snippet, content)

    def test_looks_fabricated_with_todo(self, verifier):
        """Should detect fabrication with TODO."""
        code = """
def example():
    # TODO: implement
    pass
"""
        assert verifier._looks_fabricated(code)

    def test_looks_fabricated_with_generic_name(self, verifier):
        """Should detect fabrication with generic names."""
        code = """
def example_function():
    return do_something()
"""
        assert verifier._looks_fabricated(code)

    def test_not_fabricated_specific_code(self, verifier):
        """Should not flag specific, complete code."""
        code = """
def calculate_tax(amount: float, rate: float) -> float:
    return amount * rate
"""
        assert not verifier._looks_fabricated(code)
