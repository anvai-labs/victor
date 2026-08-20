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

"""Code safety patterns for development operations.

This module provides consolidated safety patterns for:
- Git operations (force push, reset, clean, etc.)
- Package management (npm, pip, yarn, cargo)
- Refactoring operations
- Build and deployment commands

These patterns are used across verticals (coding, devops) to detect
potentially dangerous commands and file operations.

Example usage:
    from victor.security.safety.code_patterns import (
        CodePatternScanner,
        CodePatternCategory,
        scan_command,
        is_sensitive_file,
    )

    # Scan a command
    scanner = CodePatternScanner()
    patterns = scanner.scan_command("git push --force origin main")
    for pattern in patterns:
        print(f"[{pattern.risk_level}] {pattern.description}")

    # Check if a file is sensitive
    if is_sensitive_file(".env"):
        print("Sensitive file!")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from victor.core.severity import SeverityLevel

from victor.security.safety.types import SafetyPattern

# =============================================================================
# Enumerations
# =============================================================================


class CodePatternCategory(Enum):
    """Categories for code safety patterns."""

    GIT = "git"
    REFACTORING = "refactoring"
    PACKAGE_MANAGER = "package"
    BUILD_DEPLOY = "build"
    SECRETS = "secrets"
    FILESYSTEM = "filesystem"


class RiskLevel(Enum):
    """Risk levels for safety patterns."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def to_severity(self) -> "SeverityLevel":
        """Convert to unified SeverityLevel."""
        from victor.core.severity import SeverityLevel

        return SeverityLevel.from_risk_level(self)


# =============================================================================
# Pattern Definitions
# =============================================================================


# Git-specific dangerous patterns
GIT_PATTERNS: List[SafetyPattern] = [
    # CRITICAL risk - data loss
    SafetyPattern(
        pattern=r"git\s+reset\s+--hard",
        description="Discard all uncommitted changes",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+clean\s+-fd",
        description="Delete untracked files",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+push\s+.*--force",
        description="Force push (may lose commits)",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+push\s+-f",
        description="Force push (may lose commits)",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+push\s+--force-with-lease",
        description="Force push with lease (safer but still rewrites)",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+rebase\s+.*--force",
        description="Force rebase",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+branch\s+-D",
        description="Force delete branch",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+push\s+.*:\s*\S+",
        description="Delete remote branch",
        risk_level="HIGH",
        category="git",
    ),
    # MEDIUM risk - could cause issues
    SafetyPattern(
        pattern=r"git\s+checkout\s+--\s+",
        description="Discard changes to file",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+stash\s+drop",
        description="Discard stashed changes",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+stash\s+clear",
        description="Clear all stashes",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+rebase\s+-i",
        description="Interactive rebase (rewrites history)",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+commit\s+--amend",
        description="Amend last commit",
        risk_level="MEDIUM",
        category="git",
    ),
]


# Refactoring-specific patterns
REFACTORING_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"refactor.*--all",
        description="Refactor across entire codebase",
        risk_level="HIGH",
        category="refactoring",
    ),
    SafetyPattern(
        pattern=r"rename.*--recursive",
        description="Recursive symbol rename",
        risk_level="MEDIUM",
        category="refactoring",
    ),
    SafetyPattern(
        pattern=r"sed\s+-i.*-r?e",
        description="In-place file modification with regex",
        risk_level="MEDIUM",
        category="refactoring",
    ),
]


# Package manager patterns
PACKAGE_MANAGER_PATTERNS: List[SafetyPattern] = [
    # Python
    SafetyPattern(
        pattern=r"pip\s+uninstall",
        description="Uninstall Python package",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"pip\s+install\s+--force-reinstall",
        description="Force reinstall package",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"pip\s+install\s+--upgrade\s+--force-reinstall",
        description="Force upgrade package",
        risk_level="MEDIUM",
        category="package",
    ),
    # JavaScript
    SafetyPattern(
        pattern=r"npm\s+uninstall",
        description="Uninstall npm package",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"npm\s+cache\s+clean\s+--force",
        description="Force clean npm cache",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"yarn\s+remove",
        description="Remove yarn package",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"pnpm\s+remove",
        description="Remove pnpm package",
        risk_level="MEDIUM",
        category="package",
    ),
    # Rust
    SafetyPattern(
        pattern=r"cargo\s+remove",
        description="Remove Rust crate",
        risk_level="MEDIUM",
        category="package",
    ),
    # Ruby
    SafetyPattern(
        pattern=r"gem\s+uninstall",
        description="Uninstall Ruby gem",
        risk_level="MEDIUM",
        category="package",
    ),
    # Go
    SafetyPattern(
        pattern=r"go\s+clean\s+-cache",
        description="Clear Go module cache",
        risk_level="LOW",
        category="package",
    ),
]


# Build/deployment patterns
BUILD_DEPLOY_PATTERNS: List[SafetyPattern] = [
    # Docker
    SafetyPattern(
        pattern=r"docker\s+system\s+prune",
        description="Remove unused Docker data",
        risk_level="MEDIUM",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+image\s+prune\s+-a",
        description="Remove all unused images",
        risk_level="MEDIUM",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+container\s+prune",
        description="Remove stopped containers",
        risk_level="MEDIUM",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+volume\s+prune",
        description="Remove unused volumes",
        risk_level="HIGH",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+rm\s+-f",
        description="Force remove container",
        risk_level="MEDIUM",
        category="docker",
    ),
    # Kubernetes (basic patterns - more in infrastructure.py)
    SafetyPattern(
        pattern=r"kubectl\s+delete",
        description="Delete Kubernetes resources",
        risk_level="HIGH",
        category="kubernetes",
    ),
    # Infrastructure
    SafetyPattern(
        pattern=r"terraform\s+destroy",
        description="Destroy infrastructure",
        risk_level="CRITICAL",
        category="infrastructure",
    ),
]


# Sensitive file patterns
SENSITIVE_FILE_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"\.env$",
        description="Environment file with secrets",
        risk_level="HIGH",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"\.env\..*",
        description="Environment variant file",
        risk_level="HIGH",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"\.pem$|\.key$",
        description="Private key file",
        risk_level="CRITICAL",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"credentials\.json",
        description="Credentials file",
        risk_level="HIGH",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"\.git/",
        description="Git internal directory",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"id_rsa$|id_ed25519$|id_ecdsa$",
        description="SSH private key",
        risk_level="CRITICAL",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"\.ssh/",
        description="SSH configuration directory",
        risk_level="HIGH",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"\.aws/credentials",
        description="AWS credentials file",
        risk_level="CRITICAL",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"\.kube/config",
        description="Kubernetes config with credentials",
        risk_level="CRITICAL",
        category="secrets",
    ),
]


# =============================================================================
# CodePatternScanner
# =============================================================================


@dataclass
class SafetyScanResult:
    """Result from safety pattern scanning.

    Renamed from ScanResult to be semantically distinct:
    - SafetyScanResult (here): Safety pattern matching results
    - IaCScanResult (victor.iac.protocol): Infrastructure-as-Code scan results

    Attributes:
        matches: List of matched patterns
        risk_summary: Dict of risk level to count
        has_critical: Whether any CRITICAL patterns matched
        has_high: Whether any HIGH patterns matched
    """

    matches: List[SafetyPattern] = field(default_factory=list)
    risk_summary: Dict[str, int] = field(default_factory=dict)
    has_critical: bool = False
    has_high: bool = False

    def add_match(self, pattern: SafetyPattern) -> None:
        """Add a matched pattern."""
        self.matches.append(pattern)
        level = pattern.risk_level
        self.risk_summary[level] = self.risk_summary.get(level, 0) + 1
        if level == "CRITICAL":
            self.has_critical = True
        elif level == "HIGH":
            self.has_high = True


# Backward compatibility alias
ScanResult = SafetyScanResult


class CodePatternScanner:
    """Scanner for git and code safety patterns.

    Provides comprehensive scanning for dangerous operations in:
    - Git commands
    - Package manager commands
    - Refactoring operations
    - Build and deployment commands
    - Sensitive file paths

    Example:
        scanner = CodePatternScanner()

        # Scan a command
        result = scanner.scan_command("git push --force origin main")
        if result.has_high:
            print("Dangerous command detected!")
            for pattern in result.matches:
                print(f"  - {pattern.description}")

        # Check if file is sensitive
        if scanner.is_sensitive_file(".env"):
            print("This is a sensitive file")
    """

    def __init__(
        self,
        include_git: bool = True,
        include_refactoring: bool = True,
        include_packages: bool = True,
        include_build: bool = True,
        include_files: bool = True,
        custom_patterns: Optional[List[SafetyPattern]] = None,
    ):
        """Initialize the scanner.

        Args:
            include_git: Include git patterns
            include_refactoring: Include refactoring patterns
            include_packages: Include package manager patterns
            include_build: Include build/deploy patterns
            include_files: Include sensitive file patterns
            custom_patterns: Additional custom patterns to include
        """
        self._patterns: List[SafetyPattern] = []
        self._file_patterns: List[SafetyPattern] = []

        if include_git:
            self._patterns.extend(GIT_PATTERNS)
        if include_refactoring:
            self._patterns.extend(REFACTORING_PATTERNS)
        if include_packages:
            self._patterns.extend(PACKAGE_MANAGER_PATTERNS)
        if include_build:
            self._patterns.extend(BUILD_DEPLOY_PATTERNS)
        if include_files:
            self._file_patterns.extend(SENSITIVE_FILE_PATTERNS)

        if custom_patterns:
            self._patterns.extend(custom_patterns)

        # Compile patterns for efficiency
        self._compiled_patterns = [
            (re.compile(p.pattern, re.IGNORECASE), p) for p in self._patterns
        ]
        self._compiled_file_patterns = [
            (re.compile(p.pattern, re.IGNORECASE), p) for p in self._file_patterns
        ]

    def scan_command(self, command: str) -> ScanResult:
        """Scan a command for dangerous patterns.

        Args:
            command: Shell command to scan

        Returns:
            ScanResult with matched patterns
        """
        result = ScanResult()
        for regex, pattern in self._compiled_patterns:
            if regex.search(command):
                result.add_match(pattern)
        return result

    def scan_commands(self, commands: List[str]) -> ScanResult:
        """Scan multiple commands for dangerous patterns.

        Args:
            commands: List of shell commands

        Returns:
            Combined ScanResult
        """
        result = ScanResult()
        for command in commands:
            for regex, pattern in self._compiled_patterns:
                if regex.search(command):
                    result.add_match(pattern)
        return result

    def is_sensitive_file(self, path: str) -> bool:
        """Check if a file path is sensitive.

        Args:
            path: File path to check

        Returns:
            True if path matches sensitive patterns
        """
        for regex, _ in self._compiled_file_patterns:
            if regex.search(path):
                return True
        return False

    def scan_file_path(self, path: str) -> ScanResult:
        """Scan a file path for sensitive patterns.

        Args:
            path: File path to scan

        Returns:
            ScanResult with matched patterns
        """
        result = ScanResult()
        for regex, pattern in self._compiled_file_patterns:
            if regex.search(path):
                result.add_match(pattern)
        return result

    def get_patterns_by_category(self, category: CodePatternCategory) -> List[SafetyPattern]:
        """Get patterns for a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of patterns in category
        """
        return [p for p in self._patterns if p.category == category.value]

    def get_patterns_by_risk(self, risk_level: str) -> List[SafetyPattern]:
        """Get patterns by risk level.

        Args:
            risk_level: Risk level to filter by

        Returns:
            List of patterns at risk level
        """
        return [p for p in self._patterns if p.risk_level == risk_level]

    @property
    def all_patterns(self) -> List[SafetyPattern]:
        """Get all command patterns."""
        return self._patterns.copy()

    @property
    def file_patterns(self) -> List[SafetyPattern]:
        """Get all file patterns."""
        return self._file_patterns.copy()


# =============================================================================
# Convenience Functions
# =============================================================================


def scan_command(command: str) -> List[SafetyPattern]:
    """Scan a command for dangerous patterns.

    Convenience function for quick scanning.

    Args:
        command: Command to scan

    Returns:
        List of matched patterns
    """
    scanner = CodePatternScanner()
    return scanner.scan_command(command).matches


def is_sensitive_file(path: str) -> bool:
    """Check if a file path is sensitive.

    Convenience function for quick checking.

    Args:
        path: File path to check

    Returns:
        True if sensitive
    """
    scanner = CodePatternScanner()
    return scanner.is_sensitive_file(path)


def get_all_patterns() -> List[SafetyPattern]:
    """Get all code safety patterns.

    Returns:
        Combined list of all patterns
    """
    return (
        GIT_PATTERNS
        + REFACTORING_PATTERNS
        + PACKAGE_MANAGER_PATTERNS
        + BUILD_DEPLOY_PATTERNS
        + SENSITIVE_FILE_PATTERNS
    )


__all__ = [
    # Enums
    "CodePatternCategory",
    "RiskLevel",
    # Pattern lists
    "GIT_PATTERNS",
    "REFACTORING_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "BUILD_DEPLOY_PATTERNS",
    "SENSITIVE_FILE_PATTERNS",
    # Classes
    "ScanResult",
    "CodePatternScanner",
    # Functions
    "scan_command",
    "is_sensitive_file",
    "get_all_patterns",
]
