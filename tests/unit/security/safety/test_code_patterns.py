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

"""Tests for victor.security.safety.code_patterns module."""

import pytest

from victor.security.safety.code_patterns import (
    CodePatternCategory,
    CodePatternScanner,
    GIT_PATTERNS,
    REFACTORING_PATTERNS,
    PACKAGE_MANAGER_PATTERNS,
    BUILD_DEPLOY_PATTERNS,
    SENSITIVE_FILE_PATTERNS,
    ScanResult,
    scan_command,
    is_sensitive_file,
    get_all_patterns,
)


class TestPatternLists:
    """Tests for pattern list definitions."""

    def test_git_patterns_not_empty(self):
        """GIT_PATTERNS should have patterns."""
        assert len(GIT_PATTERNS) > 0

    def test_refactoring_patterns_not_empty(self):
        """REFACTORING_PATTERNS should have patterns."""
        assert len(REFACTORING_PATTERNS) > 0

    def test_package_manager_patterns_not_empty(self):
        """PACKAGE_MANAGER_PATTERNS should have patterns."""
        assert len(PACKAGE_MANAGER_PATTERNS) > 0

    def test_build_deploy_patterns_not_empty(self):
        """BUILD_DEPLOY_PATTERNS should have patterns."""
        assert len(BUILD_DEPLOY_PATTERNS) > 0

    def test_sensitive_file_patterns_not_empty(self):
        """SENSITIVE_FILE_PATTERNS should have patterns."""
        assert len(SENSITIVE_FILE_PATTERNS) > 0

    def test_all_patterns_have_required_fields(self):
        """All patterns should have required fields."""
        all_patterns = get_all_patterns()
        for pattern in all_patterns:
            assert pattern.pattern, "Pattern should not be empty"
            assert pattern.description, "Description should not be empty"
            assert pattern.risk_level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            assert pattern.category, "Category should not be empty"


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_empty_result(self):
        """Empty ScanResult should have correct defaults."""
        result = ScanResult()
        assert result.matches == []
        assert result.risk_summary == {}
        assert result.has_critical is False
        assert result.has_high is False

    def test_add_match_high(self):
        """Adding HIGH match should update has_high."""
        result = ScanResult()
        result.add_match(GIT_PATTERNS[0])  # First git pattern is HIGH
        assert result.has_high is True
        assert len(result.matches) == 1
        assert result.risk_summary.get("HIGH", 0) > 0

    def test_add_match_critical(self):
        """Adding CRITICAL match should update has_critical."""
        # Find a CRITICAL pattern
        critical_pattern = None
        for pattern in BUILD_DEPLOY_PATTERNS:
            if pattern.risk_level == "CRITICAL":
                critical_pattern = pattern
                break

        if critical_pattern:
            result = ScanResult()
            result.add_match(critical_pattern)
            assert result.has_critical is True


class TestCodePatternScanner:
    """Tests for CodePatternScanner class."""

    def test_scan_git_force_push(self):
        """Scanner should detect git force push."""
        scanner = CodePatternScanner()
        result = scanner.scan_command("git push --force origin main")
        assert result.has_high is True
        assert any("force push" in m.description.lower() for m in result.matches)

    def test_scan_git_reset_hard(self):
        """Scanner should detect git reset --hard."""
        scanner = CodePatternScanner()
        result = scanner.scan_command("git reset --hard HEAD~1")
        assert result.has_high is True
        assert any("uncommitted" in m.description.lower() for m in result.matches)

    def test_scan_git_clean(self):
        """Scanner should detect git clean -fd."""
        scanner = CodePatternScanner()
        result = scanner.scan_command("git clean -fd")
        assert result.has_high is True

    def test_scan_pip_uninstall(self):
        """Scanner should detect pip uninstall."""
        scanner = CodePatternScanner()
        result = scanner.scan_command("pip uninstall package-name")
        assert len(result.matches) > 0
        assert any("uninstall" in m.description.lower() for m in result.matches)

    def test_scan_npm_uninstall(self):
        """Scanner should detect npm uninstall."""
        scanner = CodePatternScanner()
        result = scanner.scan_command("npm uninstall lodash")
        assert len(result.matches) > 0

    def test_scan_terraform_destroy(self):
        """Scanner should detect terraform destroy."""
        scanner = CodePatternScanner()
        result = scanner.scan_command("terraform destroy")
        assert result.has_critical is True

    def test_scan_safe_command(self):
        """Scanner should return empty for safe commands."""
        scanner = CodePatternScanner()
        result = scanner.scan_command("git status")
        assert len(result.matches) == 0
        assert result.has_high is False
        assert result.has_critical is False

    def test_scan_multiple_commands(self):
        """Scanner should detect patterns in multiple commands."""
        scanner = CodePatternScanner()
        result = scanner.scan_commands(
            [
                "git status",
                "git push --force origin main",
                "npm install",
            ]
        )
        assert result.has_high is True
        assert len(result.matches) >= 1

    def test_is_sensitive_file_env(self):
        """Scanner should detect .env files."""
        scanner = CodePatternScanner()
        assert scanner.is_sensitive_file(".env") is True
        assert scanner.is_sensitive_file("config/.env") is True
        assert scanner.is_sensitive_file(".env.local") is True

    def test_is_sensitive_file_key(self):
        """Scanner should detect key/pem files."""
        scanner = CodePatternScanner()
        assert scanner.is_sensitive_file("private.key") is True
        assert scanner.is_sensitive_file("cert.pem") is True

    def test_is_sensitive_file_credentials(self):
        """Scanner should detect credentials files."""
        scanner = CodePatternScanner()
        assert scanner.is_sensitive_file("credentials.json") is True

    def test_is_not_sensitive_file(self):
        """Scanner should not flag regular files."""
        scanner = CodePatternScanner()
        assert scanner.is_sensitive_file("main.py") is False
        assert scanner.is_sensitive_file("package.json") is False

    def test_scan_file_path(self):
        """scan_file_path should return ScanResult."""
        scanner = CodePatternScanner()
        result = scanner.scan_file_path(".env")
        assert len(result.matches) > 0

    def test_get_patterns_by_category(self):
        """get_patterns_by_category should filter correctly."""
        scanner = CodePatternScanner()
        git_patterns = scanner.get_patterns_by_category(CodePatternCategory.GIT)
        assert len(git_patterns) > 0
        for pattern in git_patterns:
            assert pattern.category == "git"

    def test_get_patterns_by_risk(self):
        """get_patterns_by_risk should filter correctly."""
        scanner = CodePatternScanner()
        high_patterns = scanner.get_patterns_by_risk("HIGH")
        assert len(high_patterns) > 0
        for pattern in high_patterns:
            assert pattern.risk_level == "HIGH"

    def test_scanner_with_custom_patterns(self):
        """Scanner should include custom patterns."""
        from victor.security.safety.types import SafetyPattern

        custom = SafetyPattern(
            pattern=r"dangerous_custom_command",
            description="Custom dangerous pattern",
            risk_level="CRITICAL",
            category="custom",
        )
        scanner = CodePatternScanner(custom_patterns=[custom])
        result = scanner.scan_command("dangerous_custom_command --flag")
        assert result.has_critical is True

    def test_scanner_exclude_categories(self):
        """Scanner should exclude categories when configured."""
        scanner = CodePatternScanner(
            include_git=False,
            include_refactoring=True,
            include_packages=True,
            include_build=True,
        )
        result = scanner.scan_command("git push --force origin main")
        # Should not detect git patterns
        assert len(result.matches) == 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_scan_command_function(self):
        """scan_command convenience function should work."""
        matches = scan_command("git reset --hard")
        assert len(matches) > 0

    def test_is_sensitive_file_function(self):
        """is_sensitive_file convenience function should work."""
        assert is_sensitive_file(".env") is True
        assert is_sensitive_file("main.py") is False

    def test_get_all_patterns_function(self):
        """get_all_patterns should return combined list."""
        all_patterns = get_all_patterns()
        assert len(all_patterns) > 0
        # Should include patterns from all categories
        categories = {p.category for p in all_patterns}
        assert "git" in categories
