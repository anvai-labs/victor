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

"""Secret and credential detection utilities.

This module provides pattern-based detection of secrets, credentials,
and API keys in text content. Useful across all verticals for preventing
accidental exposure of sensitive information.

Example usage:
    from victor.security.safety.secrets import (
        SecretScanner,
        CREDENTIAL_PATTERNS,
        detect_secrets,
    )

    # Quick detection
    secrets = detect_secrets(code_content)
    for secret in secrets:
        print(f"Found {secret.secret_type} at position {secret.start}")

    # Full scanner with custom patterns
    scanner = SecretScanner()
    scanner.add_pattern("custom_token", r"my_token_[a-z0-9]{32}")
    results = scanner.scan(content)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Pattern, Tuple

if TYPE_CHECKING:
    from victor.core.severity import SeverityLevel

# Rust-accelerated secret scanning (with Python fallback)
_RUST_SECRETS_AVAILABLE = False
try:
    from victor.processing.native import (
        scan_secrets as rust_scan_secrets,
        has_secrets as rust_has_secrets,
        mask_secrets as rust_mask_secrets,
    )

    _RUST_SECRETS_AVAILABLE = True
except ImportError:
    pass


class SecretSeverity(Enum):
    """Severity levels for detected secrets."""

    CRITICAL = "critical"  # AWS keys, private keys
    HIGH = "high"  # API tokens, passwords
    MEDIUM = "medium"  # Generic secrets
    LOW = "low"  # Potentially sensitive

    def to_severity(self) -> "SeverityLevel":
        """Convert to unified SeverityLevel."""
        from victor.core.severity import SeverityLevel

        return SeverityLevel.from_secret_severity(self)


@dataclass
class SecretMatch:
    """A detected secret in content.

    Attributes:
        secret_type: Type of secret detected
        matched_text: The matched text (partially redacted)
        start: Start position in content
        end: End position in content
        line_number: Line number (1-indexed)
        severity: Severity level
        suggestion: Remediation suggestion
    """

    secret_type: str
    matched_text: str
    start: int
    end: int
    line_number: int
    severity: SecretSeverity
    suggestion: str = ""

    def __post_init__(self):
        # Redact the matched text for safety
        if len(self.matched_text) > 8:
            visible = min(4, len(self.matched_text) // 4)
            self.matched_text = self.matched_text[:visible] + "..." + self.matched_text[-visible:]


# =============================================================================
# Credential Patterns
# =============================================================================

# AWS Credentials
AWS_ACCESS_KEY_PATTERN = r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"
AWS_SECRET_KEY_PATTERN = r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"
AWS_SESSION_TOKEN_PATTERN = (
    r"(?i)aws[_-]?session[_-]?token\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{100,}['\"]?"
)

# GitHub Tokens
GITHUB_PAT_PATTERN = r"ghp_[0-9a-zA-Z]{36}"
GITHUB_OAUTH_PATTERN = r"gho_[0-9a-zA-Z]{36}"
GITHUB_APP_PATTERN = r"(?:ghu|ghs)_[0-9a-zA-Z]{36}"
GITHUB_REFRESH_PATTERN = r"ghr_[0-9a-zA-Z]{36}"

# Other Cloud Providers
GOOGLE_API_KEY_PATTERN = r"AIza[0-9A-Za-z\-_]{35}"
AZURE_SUBSCRIPTION_KEY_PATTERN = (
    r"(?i)(?:subscription[_-]?key|api[_-]?key)\s*[=:]\s*['\"]?[a-f0-9]{32}['\"]?"
)
SLACK_TOKEN_PATTERN = r"xox[baprs]-[0-9]{10,13}-[0-9a-zA-Z]{24}"
STRIPE_KEY_PATTERN = r"(?:sk|pk)_(?:live|test)_[0-9a-zA-Z]{24,}"

# Generic Secrets
GENERIC_PASSWORD_PATTERN = r"(?i)(?:password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}['\"]"
GENERIC_SECRET_PATTERN = r"(?i)(?:secret|token|api[_-]?key)\s*[=:]\s*['\"][^'\"]{8,}['\"]"
GENERIC_BEARER_PATTERN = r"(?i)bearer\s+[a-zA-Z0-9\-_\.]{20,}"

# Private Keys
PRIVATE_KEY_PATTERN = r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"
# This is the detection regex used by the secret scanner itself, not an
# actual private key. nosemgrep: generic.secrets.security.detected-pgp-private-key-block.detected-pgp-private-key-block
PGP_PRIVATE_PATTERN = r"-----BEGIN PGP PRIVATE KEY BLOCK-----"  # nosemgrep

# Database Connection Strings
POSTGRES_URI_PATTERN = r"postgres(?:ql)?://[^:]+:[^@]+@[^/]+"
MYSQL_URI_PATTERN = r"mysql://[^:]+:[^@]+@[^/]+"
MONGODB_URI_PATTERN = r"mongodb(?:\+srv)?://[^:]+:[^@]+@[^/]+"
REDIS_URI_PATTERN = r"redis://[^:]+:[^@]+@[^/]+"

# JWT Tokens (base64 encoded)
JWT_PATTERN = r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]+"


# Pattern registry with metadata
CREDENTIAL_PATTERNS: Dict[str, Tuple[str, SecretSeverity, str]] = {
    # AWS
    "aws_access_key": (
        AWS_ACCESS_KEY_PATTERN,
        SecretSeverity.CRITICAL,
        "Use AWS IAM roles or environment variables instead of hardcoded keys",
    ),
    "aws_secret_key": (
        AWS_SECRET_KEY_PATTERN,
        SecretSeverity.CRITICAL,
        "Use AWS Secrets Manager or environment variables",
    ),
    "aws_session_token": (
        AWS_SESSION_TOKEN_PATTERN,
        SecretSeverity.HIGH,
        "Session tokens should not be stored in code",
    ),
    # GitHub
    "github_pat": (
        GITHUB_PAT_PATTERN,
        SecretSeverity.CRITICAL,
        "Use GitHub Actions secrets or environment variables",
    ),
    "github_oauth": (
        GITHUB_OAUTH_PATTERN,
        SecretSeverity.HIGH,
        "OAuth tokens should be stored securely",
    ),
    "github_app": (
        GITHUB_APP_PATTERN,
        SecretSeverity.HIGH,
        "GitHub App tokens should use secure storage",
    ),
    # Cloud Providers
    "google_api_key": (
        GOOGLE_API_KEY_PATTERN,
        SecretSeverity.HIGH,
        "Use Google Cloud Secret Manager",
    ),
    "slack_token": (
        SLACK_TOKEN_PATTERN,
        SecretSeverity.HIGH,
        "Use Slack's secure token storage",
    ),
    "stripe_key": (
        STRIPE_KEY_PATTERN,
        SecretSeverity.CRITICAL,
        "Never expose Stripe keys; use environment variables",
    ),
    # Generic
    "generic_password": (
        GENERIC_PASSWORD_PATTERN,
        SecretSeverity.HIGH,
        "Use a secrets manager or environment variables",
    ),
    "generic_secret": (
        GENERIC_SECRET_PATTERN,
        SecretSeverity.MEDIUM,
        "Consider using a secrets manager",
    ),
    "bearer_token": (
        GENERIC_BEARER_PATTERN,
        SecretSeverity.MEDIUM,
        "Bearer tokens should not be hardcoded",
    ),
    # Private Keys
    "private_key": (
        PRIVATE_KEY_PATTERN,
        SecretSeverity.CRITICAL,
        "Private keys must never be committed to version control",
    ),
    "pgp_private_key": (
        PGP_PRIVATE_PATTERN,
        SecretSeverity.CRITICAL,
        "PGP private keys must be stored securely",
    ),
    # Database URIs
    "postgres_uri": (
        POSTGRES_URI_PATTERN,
        SecretSeverity.HIGH,
        "Use environment variables for database credentials",
    ),
    "mysql_uri": (
        MYSQL_URI_PATTERN,
        SecretSeverity.HIGH,
        "Use environment variables for database credentials",
    ),
    "mongodb_uri": (
        MONGODB_URI_PATTERN,
        SecretSeverity.HIGH,
        "Use environment variables for database credentials",
    ),
    "redis_uri": (
        REDIS_URI_PATTERN,
        SecretSeverity.MEDIUM,
        "Use environment variables for Redis credentials",
    ),
    # JWT
    "jwt_token": (
        JWT_PATTERN,
        SecretSeverity.MEDIUM,
        "JWT tokens should not be hardcoded; they may contain sensitive claims",
    ),
}


# =============================================================================
# Secret Scanner
# =============================================================================


class SecretScanner:
    """Scanner for detecting secrets and credentials in text content.

    Example:
        scanner = SecretScanner()
        results = scanner.scan(file_content)
        for match in results:
            print(f"Found {match.secret_type}: {match.suggestion}")
    """

    def __init__(
        self,
        patterns: Optional[Dict[str, Tuple[str, SecretSeverity, str]]] = None,
        include_low_severity: bool = False,
    ):
        """Initialize the scanner.

        Args:
            patterns: Custom patterns (defaults to CREDENTIAL_PATTERNS)
            include_low_severity: Whether to include LOW severity matches
        """
        self._patterns = patterns or CREDENTIAL_PATTERNS.copy()
        self._compiled: Dict[str, Pattern] = {}
        self._include_low = include_low_severity
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        for name, (pattern, _, _) in self._patterns.items():
            try:
                self._compiled[name] = re.compile(pattern)
            except re.error:
                # Skip invalid patterns
                pass

    def add_pattern(
        self,
        name: str,
        pattern: str,
        severity: SecretSeverity = SecretSeverity.MEDIUM,
        suggestion: str = "",
    ) -> None:
        """Add a custom pattern.

        Args:
            name: Pattern identifier
            pattern: Regex pattern string
            severity: Severity level
            suggestion: Remediation suggestion
        """
        self._patterns[name] = (pattern, severity, suggestion)
        try:
            self._compiled[name] = re.compile(pattern)
        except re.error:
            pass

    def remove_pattern(self, name: str) -> None:
        """Remove a pattern.

        Args:
            name: Pattern identifier to remove
        """
        self._patterns.pop(name, None)
        self._compiled.pop(name, None)

    def scan(self, content: str) -> List[SecretMatch]:
        """Scan content for secrets.

        Args:
            content: Text content to scan

        Returns:
            List of SecretMatch objects for detected secrets
        """
        matches = []
        lines = content.split("\n")
        line_offsets = self._compute_line_offsets(lines)

        for name, compiled in self._compiled.items():
            pattern_str, severity, suggestion = self._patterns[name]

            # Skip low severity if not requested
            if not self._include_low and severity == SecretSeverity.LOW:
                continue

            for match in compiled.finditer(content):
                line_num = self._get_line_number(match.start(), line_offsets)
                matches.append(
                    SecretMatch(
                        secret_type=name,
                        matched_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        line_number=line_num,
                        severity=severity,
                        suggestion=suggestion,
                    )
                )

        # Sort by position
        matches.sort(key=lambda m: m.start)
        return matches

    def scan_file(self, file_path: str) -> List[SecretMatch]:
        """Scan a file for secrets.

        Args:
            file_path: Path to file to scan

        Returns:
            List of SecretMatch objects
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.scan(content)
        except Exception:
            return []

    def _compute_line_offsets(self, lines: List[str]) -> List[int]:
        """Compute byte offsets for each line."""
        offsets = [0]
        offset = 0
        for line in lines:
            offset += len(line) + 1  # +1 for newline
            offsets.append(offset)
        return offsets

    def _get_line_number(self, position: int, offsets: List[int]) -> int:
        """Get 1-indexed line number for a position."""
        for i, offset in enumerate(offsets):
            if position < offset:
                return i
        return len(offsets)

    def get_summary(self, matches: List[SecretMatch]) -> Dict[str, int]:
        """Get summary of matches by severity.

        Args:
            matches: List of SecretMatch objects

        Returns:
            Dict mapping severity to count
        """
        summary = {s.value: 0 for s in SecretSeverity}
        for match in matches:
            summary[match.severity.value] += 1
        return summary


# =============================================================================
# Convenience Functions
# =============================================================================


def detect_secrets(content: str, include_low_severity: bool = False) -> List[SecretMatch]:
    """Quick function to detect secrets in content.

    Uses Rust-accelerated scanning when available (5-10x faster),
    falls back to Python regex otherwise.

    Args:
        content: Text content to scan
        include_low_severity: Include LOW severity matches

    Returns:
        List of SecretMatch objects
    """
    # Try Rust-accelerated scanning first (faster for large content)
    if _RUST_SECRETS_AVAILABLE and not include_low_severity:
        try:
            rust_matches = rust_scan_secrets(content)
            # Convert Rust SecretMatch to Python SecretMatch
            lines = content.split("\n")
            line_offsets = [0]
            offset = 0
            for line in lines:
                offset += len(line) + 1
                line_offsets.append(offset)

            result = []
            for rm in rust_matches:
                # Find line number
                line_num = 1
                for i, off in enumerate(line_offsets):
                    if rm.start < off:
                        line_num = i
                        break

                result.append(
                    SecretMatch(
                        secret_type=rm.secret_type,
                        matched_text=rm.matched_text,
                        start=rm.start,
                        end=rm.end,
                        line_number=line_num,
                        severity=SecretSeverity.HIGH,  # Rust patterns are high-value
                        suggestion="Remove or rotate this credential",
                    )
                )
            return result
        except Exception:
            pass  # Fall through to Python implementation

    # Fallback to Python regex scanning
    scanner = SecretScanner(include_low_severity=include_low_severity)
    return scanner.scan(content)


def has_secrets(content: str) -> bool:
    """Check if content contains any secrets.

    Uses Rust-accelerated detection when available.

    Args:
        content: Text content to check

    Returns:
        True if secrets are detected
    """
    # Use Rust has_secrets for speed if available
    if _RUST_SECRETS_AVAILABLE:
        try:
            return rust_has_secrets(content)
        except Exception:
            pass

    return len(detect_secrets(content)) > 0


def get_secret_types() -> List[str]:
    """Get list of all detectable secret types.

    Returns:
        List of secret type names
    """
    return list(CREDENTIAL_PATTERNS.keys())


def mask_secrets(content: str, replacement: str = "[REDACTED]") -> str:
    """Mask all detected secrets in content.

    Uses Rust-accelerated masking when available.

    Args:
        content: Text content with potential secrets
        replacement: Text to replace secrets with

    Returns:
        Content with secrets masked
    """
    # Try Rust-accelerated masking first
    if _RUST_SECRETS_AVAILABLE and replacement == "[REDACTED]":
        try:
            return rust_mask_secrets(content)
        except Exception:
            pass

    # Fallback to Python implementation
    matches = detect_secrets(content, include_low_severity=True)

    # Sort by position descending to replace from end first
    matches.sort(key=lambda m: m.start, reverse=True)

    result = content
    for match in matches:
        result = result[: match.start] + replacement + result[match.end :]

    return result


__all__ = [
    # Types
    "SecretSeverity",
    "SecretMatch",
    # Patterns
    "CREDENTIAL_PATTERNS",
    # Scanner
    "SecretScanner",
    # Functions
    "detect_secrets",
    "has_secrets",
    "get_secret_types",
    "mask_secrets",
]
