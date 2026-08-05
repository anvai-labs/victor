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

"""Unit tests for FEP CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from victor.ui.commands.fep import fep_app

runner = CliRunner()


@pytest.fixture
def sample_fep_content():
    """Sample FEP content for testing."""
    return """---
fep: 1
title: "Test FEP"
type: Standards Track
status: Draft
created: 2025-01-09
modified: 2025-01-09
authors:
  - name: "Test Author"
    email: "test@example.com"
    github: "testauthor"
reviewers:
  - "reviewer1"
discussion: "https://github.com/test/discussion/1"
implementation: "https://github.com/test/pull/1"
---

## Summary

This is a test FEP for unit testing purposes. It provides a comprehensive summary of the proposed enhancement to the Victor framework. This section should be around two hundred words to meet the minimum requirements for validation. The purpose is to ensure that the validation logic correctly identifies when sections meet the minimum word count requirements. This is important for maintaining quality standards across all Framework Enhancement Proposals.

## Motivation

### Problem Statement

This is the problem statement. It describes the issue that this FEP aims to address.

### Goals

1. Goal 1: Test goal
2. Goal 2: Another test goal

### Non-Goals

- Out of scope item 1
- Out of scope item 2

## Proposed Change

### High-Level Design

This section describes the proposed change in detail. It includes architectural diagrams, flow charts, or high-level descriptions. The content should be comprehensive enough to understand the technical approach.

## Benefits

### For Framework Users

- Benefit 1: Improved user experience
- Benefit 2: Better performance

### For the Ecosystem

- Benefit 1: Enhanced extensibility

## Drawbacks and Alternatives

### Drawbacks

- Drawback 1: Increased complexity: Mitigation strategy

### Alternatives Considered

1. **Alternative 1: Status Quo**
   - Description: Keep current implementation
   - Pros: No changes needed
   - Cons: Problem persists
   - Why rejected: Does not solve the problem

## Unresolved Questions

- **Question 1**: Open question (Proposed answer: Initial thoughts)

## Implementation Plan

### Phase 1: Foundation (1 week)

- [ ] Task 1
- [ ] Task 2

**Deliverable**: Foundation implementation

### Phase 2: Integration (1 week)

- [ ] Task 1

**Deliverable**: Complete integration

### Testing Strategy

- Unit tests: Coverage > 80%
- Integration tests: Core scenarios

## Migration Path

This is not a breaking change, so no migration is needed.

## Compatibility

### Backward Compatibility

- **Breaking change**: No
- **Migration required**: No

### Version Compatibility

- Minimum Python version: No change (3.10+)

### Vertical Compatibility

- Built-in verticals: No impact
- External verticals: No impact

## References

- [Related FEP-0000](link)
- [GitHub Issue #1](link)
"""


@pytest.fixture
def invalid_fep_content():
    """Invalid FEP content for testing."""
    return """---
fep: 0
title: ""
type: InvalidType
status: InvalidStatus
created: 2025-01-09
modified: 2025-01-09
authors: []
---

## Summary

Too short summary.

## Missing Sections

This FEP is missing many required sections.
"""


class TestFEPValidator:
    """Tests for FEPValidator."""

    def test_parse_valid_fep_metadata(self, sample_fep_content):
        """Test parsing valid FEP metadata."""
        from victor.feps import parse_fep_metadata

        metadata = parse_fep_metadata(sample_fep_content)

        assert metadata.fep == 1
        assert metadata.title == "Test FEP"
        assert metadata.type.value == "Standards Track"
        assert metadata.status.value == "Draft"
        assert metadata.created == "2025-01-09"
        assert metadata.modified == "2025-01-09"
        assert len(metadata.authors) == 1
        assert metadata.authors[0]["name"] == "Test Author"
        assert len(metadata.reviewers) == 1
        assert metadata.discussion == "https://github.com/test/discussion/1"
        assert metadata.implementation == "https://github.com/test/pull/1"

    def test_parse_invalid_fep_metadata(self, invalid_fep_content):
        """Test parsing invalid FEP metadata."""
        from victor.feps import parse_fep_metadata

        with pytest.raises(ValueError, match="Invalid FEP type"):
            parse_fep_metadata(invalid_fep_content)

    def test_parse_missing_frontmatter(self):
        """Test parsing FEP without frontmatter."""
        from victor.feps import parse_fep_metadata

        content = "No frontmatter here"
        with pytest.raises(ValueError, match="Missing YAML frontmatter"):
            parse_fep_metadata(content)

    def test_validate_valid_fep(self, sample_fep_content):
        """Test validating a valid FEP."""
        from victor.feps import FEPValidator

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(sample_fep_content)
            f.flush()
            fep_path = Path(f.name)

        try:
            validator = FEPValidator()
            result = validator.validate_file(fep_path)

            assert result.is_valid
            assert result.metadata is not None
            assert result.metadata.fep == 1
            assert len(result.errors) == 0
            assert len(result.sections) > 0
        finally:
            fep_path.unlink()

    def test_validate_invalid_fep(self, invalid_fep_content):
        """Test validating an invalid FEP."""
        from victor.feps import FEPValidator

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(invalid_fep_content)
            f.flush()
            fep_path = Path(f.name)

        try:
            validator = FEPValidator()
            result = validator.validate_file(fep_path)

            assert not result.is_valid
            assert len(result.errors) > 0
        finally:
            fep_path.unlink()

    def test_validate_missing_sections(self):
        """Test validation of FEP with missing required sections."""
        from victor.feps import FEPValidator

        content = """---
fep: 2
title: "Test"
type: Standards Track
status: Draft
created: 2025-01-09
modified: 2025-01-09
authors:
  - name: "Test"
---

## Summary

A summary that is long enough to meet the minimum word count requirements for validation. This ensures the section passes the quality checks.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            fep_path = Path(f.name)

        try:
            validator = FEPValidator()
            result = validator.validate_file(fep_path)

            assert not result.is_valid
            assert any("Missing required section" in error.message for error in result.errors)
        finally:
            fep_path.unlink()

    def test_validate_short_sections(self):
        """Test validation of FEP with short sections."""
        from victor.feps import FEPValidator

        content = """---
fep: 3
title: "Test"
type: Standards Track
status: Draft
created: 2025-01-09
modified: 2025-01-09
authors:
  - name: "Test"
---

## Summary

Too short.

## Motivation

Also too short.

## Proposed Change

Not enough detail.

## Benefits

Minimal benefits.

## Drawbacks and Alternatives

Few drawbacks.

## Implementation Plan

Basic plan.

## Migration Path

No migration needed.

## Compatibility

Compatible.

## References

No references.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            fep_path = Path(f.name)

        try:
            validator = FEPValidator()
            result = validator.validate_file(fep_path)

            # Should have warnings for short sections
            assert len(result.warnings) > 0
            assert any("too short" in warning.message.lower() for warning in result.warnings)
        finally:
            fep_path.unlink()


class TestFEPListCommand:
    """Tests for `victor fep list` command."""

    def test_list_feps(self, sample_fep_content):
        """Test listing FEPs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)

            # Create sample FEP files
            (feps_dir / "fep-0001-test.md").write_text(sample_fep_content)
            (feps_dir / "fep-0002-another.md").write_text(
                sample_fep_content.replace("fep: 1", "fep: 2").replace("Test FEP", "Another FEP")
            )

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                result = runner.invoke(fep_app, ["list"])

                assert result.exit_code == 0
                assert "FEP-0001" in result.stdout
                assert "FEP-0002" in result.stdout
                assert "Test FEP" in result.stdout
                assert "Another FEP" in result.stdout

    def test_list_feps_with_status_filter(self, sample_fep_content):
        """Test listing FEPs with status filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)

            # Create FEP with different status
            content = sample_fep_content.replace("status: Draft", "status: Accepted")
            (feps_dir / "fep-0001-test.md").write_text(content)

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                result = runner.invoke(fep_app, ["list", "--status", "accepted"])

                assert result.exit_code == 0
                assert "FEP-0001" in result.stdout

    def test_list_empty_feps_directory(self):
        """Test listing FEPs with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                result = runner.invoke(fep_app, ["list"])

                assert result.exit_code == 0
                assert "No FEPs found" in result.stdout


class TestFEPViewCommand:
    """Tests for `victor fep view` command."""

    def test_view_existing_fep(self, sample_fep_content):
        """Test viewing an existing FEP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)
            (feps_dir / "fep-0001-test.md").write_text(sample_fep_content)

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                result = runner.invoke(fep_app, ["view", "1"])

                assert result.exit_code == 0
                assert "FEP-0001" in result.stdout
                assert "Test FEP" in result.stdout
                assert "Summary" in result.stdout

    def test_view_nonexistent_fep(self):
        """Test viewing a non-existent FEP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                result = runner.invoke(fep_app, ["view", "999"])

                assert result.exit_code == 1
                assert "FEP-0999 not found" in result.stdout


class TestFEPValidateCommand:
    """Tests for `victor fep validate` command."""

    def test_validate_valid_fep(self, sample_fep_content):
        """Test validating a valid FEP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)
            fep_path = feps_dir / "fep-0001-test.md"
            fep_path.write_text(sample_fep_content)

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                result = runner.invoke(fep_app, ["validate", str(fep_path)])

                assert result.exit_code == 0
                assert "FEP is valid" in result.stdout

    def test_validate_invalid_fep(self, invalid_fep_content):
        """Test validating an invalid FEP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)
            fep_path = feps_dir / "fep-0001-invalid.md"
            # Use a simpler invalid content
            invalid_content = """---
fep: 0
title: ""
type: InvalidType
status: Draft
created: 2025-01-09
modified: 2025-01-09
authors: []
---

## Summary

Summary here.
"""
            fep_path.write_text(invalid_content)

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                result = runner.invoke(fep_app, ["validate", str(fep_path)])

                assert result.exit_code == 1
                assert "validation failed" in result.stdout

    def test_validate_nonexistent_file(self):
        """Test validating a non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                result = runner.invoke(fep_app, ["validate", "nonexistent.md"])

                assert result.exit_code == 1
                assert "not found" in result.stdout


class TestFEPCreateCommand:
    """Tests for `victor fep create` command."""

    def test_create_fep_with_defaults(self):
        """Test creating a FEP with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)
            template_path = feps_dir / "fep-0000-template.md"
            template_content = """# FEP-{number}: {Title}

- **FEP**: {number}
- **Title**: {Brief, descriptive title}
- **Type**: Standards Track
- **Status**: Draft
- **Authors**: {Name} <{email}> (@{github})
- **Created**: {YYYY-MM-DD}
- **Modified**: {YYYY-MM-DD}

---

## Summary

Template summary.
"""
            template_path.write_text(template_content)

            output_path = Path(tmpdir) / "fep-XXXX-test.md"

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                with patch(
                    "victor.ui.commands.fep._get_template_path",
                    return_value=template_path,
                ):
                    result = runner.invoke(
                        fep_app,
                        [
                            "create",
                            "--title",
                            "Test Feature",
                            "--type",
                            "standards",
                            "--author",
                            "Test Author",
                            "--email",
                            "test@example.com",
                            "--github",
                            "testauthor",
                            "--output",
                            str(output_path),
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Created FEP" in result.stdout
                    assert output_path.exists()

                    # Check content was filled
                    content = output_path.read_text()
                    assert "Test Feature" in content
                    assert "Test Author" in content
                    assert "test@example.com" in content
                    assert "testauthor" in content

    def test_create_fep_default_output_lands_in_feps_dir(self, tmp_path, monkeypatch):
        """Default output (no --output) must land in feps/, not the cwd.

        Guards against the regression where the scaffold wrote
        fep-XXXX-<slug>.md to the current working directory, leaving scratch
        (e.g. fep-XXXX-test.md) at the repo root.
        """
        feps_dir = tmp_path / "feps"
        feps_dir.mkdir()
        template_path = feps_dir / "fep-0000-template.md"
        template_path.write_text("# FEP-{number}: {Title}\n\n## Summary\nTemplate.\n")

        # cwd is tmp_path, deliberately NOT feps/ — a cwd-relative default would
        # drop the file here.
        monkeypatch.chdir(tmp_path)

        with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
            with patch(
                "victor.ui.commands.fep._get_template_path",
                return_value=template_path,
            ):
                result = runner.invoke(
                    fep_app,
                    [
                        "create",
                        "--title",
                        "My Feature",
                        "--type",
                        "standards",
                        "--author",
                        "Test Author",
                        "--email",
                        "test@example.com",
                        "--github",
                        "testauthor",
                    ],
                )

        assert result.exit_code == 0, result.output
        expected = feps_dir / "fep-XXXX-my-feature.md"
        assert expected.exists(), "default output must land in feps/"
        # And it must NOT pollute the cwd.
        assert not (tmp_path / "fep-XXXX-my-feature.md").exists(), "must not write to cwd"

        content = expected.read_text()
        assert "My Feature" in content

    def test_create_fep_without_git(self):
        """Test creating a FEP when git is not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feps_dir = Path(tmpdir)
            template_path = feps_dir / "fep-0000-template.md"
            template_path.write_text("# Template")

            with patch("victor.ui.commands.fep._get_feps_dir", return_value=feps_dir):
                with patch(
                    "victor.ui.commands.fep._get_template_path",
                    return_value=template_path,
                ):
                    with patch("subprocess.run", side_effect=FileNotFoundError):
                        result = runner.invoke(
                            fep_app,
                            [
                                "create",
                                "--title",
                                "Test",
                                "--type",
                                "standards",
                                "--author",
                                "Test Author",
                            ],
                        )

                        # Should succeed with explicit author
                        assert result.exit_code == 0 or "Git not found" in result.stdout
