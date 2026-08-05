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

"""Personally Identifiable Information (PII) detection and anonymization.

This module provides pattern-based detection of PII in data, along with
anonymization suggestions. Useful across all verticals for GDPR/CCPA
compliance and data privacy.

Example usage:
    from victor.security.safety.pii import (
        PIIScanner,
        detect_pii_columns,
        get_anonymization_suggestion,
    )

    # Detect PII columns in a dataframe
    columns = ["first_name", "email", "age", "ssn"]
    pii_cols = detect_pii_columns(columns)
    for col, pii_type in pii_cols:
        print(f"{col}: {pii_type} - {get_anonymization_suggestion(pii_type)}")

    # Scan text content
    scanner = PIIScanner()
    matches = scanner.scan(text_content)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Pattern, Tuple

if TYPE_CHECKING:
    from victor.core.severity import SeverityLevel


class PIIType(Enum):
    """Types of personally identifiable information."""

    # Direct identifiers
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    SSN = "ssn"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"

    # Financial
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    TAX_ID = "tax_id"

    # Health
    MEDICAL_RECORD = "medical_record"
    HEALTH_INSURANCE = "health_insurance"

    # Personal
    DOB = "date_of_birth"
    AGE = "age"
    GENDER = "gender"
    RACE = "race"
    RELIGION = "religion"

    # Digital
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    USER_ID = "user_id"
    USERNAME = "username"

    # Biometric
    BIOMETRIC = "biometric"


class PIISeverity(Enum):
    """Sensitivity levels for PII types."""

    CRITICAL = "critical"  # SSN, passport, medical
    HIGH = "high"  # Financial, direct identifiers
    MEDIUM = "medium"  # Contact info, DOB
    LOW = "low"  # Demographics, preferences

    def to_severity(self) -> "SeverityLevel":
        """Convert to unified SeverityLevel."""
        from victor.core.severity import SeverityLevel

        return SeverityLevel.from_pii_severity(self)


@dataclass
class PIIMatch:
    """A detected PII element.

    Attributes:
        pii_type: Type of PII detected
        matched_text: The matched text (partially redacted)
        source: Source identifier (column name, field path, etc.)
        severity: Sensitivity level
        suggestion: Anonymization suggestion
    """

    pii_type: PIIType
    matched_text: str
    source: str
    severity: PIISeverity
    suggestion: str = ""

    def __post_init__(self):
        # Redact matched text for safety in logs
        if len(self.matched_text) > 8:
            visible = min(4, len(self.matched_text) // 4)
            self.matched_text = self.matched_text[:visible] + "***" + self.matched_text[-visible:]


# =============================================================================
# Column Name Patterns (for detecting PII columns in datasets)
# =============================================================================

PII_COLUMN_PATTERNS: Dict[PIIType, str] = {
    # Names
    PIIType.NAME: r"(?i)^(first|last|full|middle|maiden)?[_\s-]?name$",
    # Contact
    PIIType.EMAIL: r"(?i)e?mail(_address)?",
    PIIType.PHONE: r"(?i)(phone|mobile|cell|tel|fax)(_num(ber)?)?",
    PIIType.ADDRESS: r"(?i)(address|street|city|state|zip|postal|country)",
    # Government IDs
    PIIType.SSN: r"(?i)(ssn|social[_\s-]?security|sin|national[_\s-]?id)",
    PIIType.PASSPORT: r"(?i)passport(_num(ber)?)?",
    PIIType.DRIVERS_LICENSE: r"(?i)(driver'?s?[_\s-]?licen[sc]e|dl[_\s-]?num)",
    PIIType.TAX_ID: r"(?i)(tax[_\s-]?id|ein|tin|vat)",
    # Financial
    PIIType.CREDIT_CARD: r"(?i)(credit[_\s-]?card|cc[_\s-]?num|card[_\s-]?num)",
    PIIType.BANK_ACCOUNT: r"(?i)(bank[_\s-]?account|iban|routing|swift)",
    # Health
    PIIType.MEDICAL_RECORD: r"(?i)(medical|diagnosis|health[_\s-]?record|patient[_\s-]?id)",
    PIIType.HEALTH_INSURANCE: r"(?i)(insurance[_\s-]?id|policy[_\s-]?num|member[_\s-]?id)",
    # Personal
    PIIType.DOB: r"(?i)(dob|birth[_\s-]?date|date[_\s-]?of[_\s-]?birth|birthday)",
    PIIType.AGE: r"(?i)^age$",
    PIIType.GENDER: r"(?i)^(gender|sex)$",
    PIIType.RACE: r"(?i)^(race|ethnicity)$",
    PIIType.RELIGION: r"(?i)^religion$",
    # Digital
    PIIType.IP_ADDRESS: r"(?i)ip[_\s-]?addr(ess)?",
    PIIType.MAC_ADDRESS: r"(?i)mac[_\s-]?addr(ess)?",
    PIIType.USER_ID: r"(?i)(user[_\s-]?id|customer[_\s-]?id|account[_\s-]?id|member[_\s-]?id)",
    PIIType.USERNAME: r"(?i)(username|user[_\s-]?name|login|handle)",
}

# =============================================================================
# Content Patterns (for detecting PII in text content)
# =============================================================================

PII_CONTENT_PATTERNS: Dict[PIIType, Tuple[str, PIISeverity]] = {
    # SSN patterns
    PIIType.SSN: (
        r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
        PIISeverity.CRITICAL,
    ),
    # Credit card patterns (Luhn algorithm validation recommended)
    PIIType.CREDIT_CARD: (
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b",
        PIISeverity.CRITICAL,
    ),
    # Email
    PIIType.EMAIL: (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        PIISeverity.MEDIUM,
    ),
    # Phone (US format and international)
    PIIType.PHONE: (
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        PIISeverity.MEDIUM,
    ),
    # IP Address (IPv4)
    PIIType.IP_ADDRESS: (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        PIISeverity.LOW,
    ),
    # MAC Address
    PIIType.MAC_ADDRESS: (
        r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        PIISeverity.LOW,
    ),
    # Date of birth (common formats)
    PIIType.DOB: (
        r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b",
        PIISeverity.MEDIUM,
    ),
    # Passport (simplified, country-specific patterns vary)
    PIIType.PASSPORT: (
        r"\b[A-Z]{1,2}\d{6,9}\b",
        PIISeverity.CRITICAL,
    ),
}

# =============================================================================
# Severity Mapping
# =============================================================================

PII_SEVERITY: Dict[PIIType, PIISeverity] = {
    # Critical - Government/Medical/Financial IDs
    PIIType.SSN: PIISeverity.CRITICAL,
    PIIType.PASSPORT: PIISeverity.CRITICAL,
    PIIType.DRIVERS_LICENSE: PIISeverity.CRITICAL,
    PIIType.MEDICAL_RECORD: PIISeverity.CRITICAL,
    PIIType.CREDIT_CARD: PIISeverity.CRITICAL,
    PIIType.BANK_ACCOUNT: PIISeverity.CRITICAL,
    PIIType.BIOMETRIC: PIISeverity.CRITICAL,
    # High - Direct identifiers
    PIIType.NAME: PIISeverity.HIGH,
    PIIType.TAX_ID: PIISeverity.HIGH,
    PIIType.HEALTH_INSURANCE: PIISeverity.HIGH,
    # Medium - Contact info
    PIIType.EMAIL: PIISeverity.MEDIUM,
    PIIType.PHONE: PIISeverity.MEDIUM,
    PIIType.ADDRESS: PIISeverity.MEDIUM,
    PIIType.DOB: PIISeverity.MEDIUM,
    PIIType.USER_ID: PIISeverity.MEDIUM,
    PIIType.USERNAME: PIISeverity.MEDIUM,
    # Low - Demographics
    PIIType.AGE: PIISeverity.LOW,
    PIIType.GENDER: PIISeverity.LOW,
    PIIType.RACE: PIISeverity.LOW,
    PIIType.RELIGION: PIISeverity.LOW,
    PIIType.IP_ADDRESS: PIISeverity.LOW,
    PIIType.MAC_ADDRESS: PIISeverity.LOW,
}

# =============================================================================
# Anonymization Suggestions
# =============================================================================

ANONYMIZATION_SUGGESTIONS: Dict[PIIType, str] = {
    # Names
    PIIType.NAME: "Replace with fake names using Faker library, or hash with salt",
    # Contact
    PIIType.EMAIL: "Replace domain with example.com, hash local part, or remove entirely",
    PIIType.PHONE: "Remove or replace with random numbers preserving format",
    PIIType.ADDRESS: "Generalize to region/city level, use zip code prefix only",
    # Government IDs
    PIIType.SSN: "Remove entirely - never store in analysis datasets",
    PIIType.PASSPORT: "Remove entirely or replace with random identifier",
    PIIType.DRIVERS_LICENSE: "Remove entirely or hash with salt",
    PIIType.TAX_ID: "Remove or tokenize with secure mapping",
    # Financial
    PIIType.CREDIT_CARD: "Mask all but last 4 digits, or remove entirely",
    PIIType.BANK_ACCOUNT: "Remove entirely or tokenize",
    # Health
    PIIType.MEDICAL_RECORD: "Use de-identification per HIPAA Safe Harbor method",
    PIIType.HEALTH_INSURANCE: "Tokenize or remove",
    # Personal
    PIIType.DOB: "Convert to age ranges (e.g., 25-34) or year only",
    PIIType.AGE: "Convert to age ranges",
    PIIType.GENDER: "Consider if needed; use categories or remove",
    PIIType.RACE: "Aggregate to broader categories if required; otherwise remove",
    PIIType.RELIGION: "Remove unless essential; aggregate if kept",
    # Digital
    PIIType.IP_ADDRESS: "Mask last octet (e.g., 192.168.1.x) or hash",
    PIIType.MAC_ADDRESS: "Hash or remove",
    PIIType.USER_ID: "Hash with salt or replace with sequential IDs",
    PIIType.USERNAME: "Hash or replace with pseudonyms",
    # Biometric
    PIIType.BIOMETRIC: "Remove entirely - biometric data cannot be anonymized",
}


# =============================================================================
# PII Scanner
# =============================================================================


class PIIScanner:
    """Scanner for detecting PII in text content and data structures.

    Example:
        scanner = PIIScanner()

        # Scan columns
        columns = ["name", "email", "score"]
        pii_cols = scanner.detect_columns(columns)

        # Scan content
        matches = scanner.scan_content(text)
    """

    def __init__(self):
        """Initialize the scanner."""
        self._column_patterns: Dict[PIIType, Pattern] = {}
        self._content_patterns: Dict[PIIType, Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        for pii_type, pattern in PII_COLUMN_PATTERNS.items():
            try:
                self._column_patterns[pii_type] = re.compile(pattern)
            except re.error:
                pass

        for pii_type, (pattern, _) in PII_CONTENT_PATTERNS.items():
            try:
                self._content_patterns[pii_type] = re.compile(pattern)
            except re.error:
                pass

    def detect_columns(self, columns: List[str]) -> List[Tuple[str, PIIType]]:
        """Detect PII columns in a list of column names.

        Args:
            columns: List of column names

        Returns:
            List of (column_name, pii_type) tuples
        """
        detected = []
        for col in columns:
            for pii_type, pattern in self._column_patterns.items():
                if pattern.search(col):
                    detected.append((col, pii_type))
                    break  # One match per column
        return detected

    def scan_content(self, content: str) -> List[PIIMatch]:
        """Scan text content for PII.

        Args:
            content: Text content to scan

        Returns:
            List of PIIMatch objects
        """
        matches = []

        for pii_type, pattern in self._content_patterns.items():
            _, severity = PII_CONTENT_PATTERNS[pii_type]

            for match in pattern.finditer(content):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        matched_text=match.group(),
                        source="content",
                        severity=severity,
                        suggestion=get_anonymization_suggestion(pii_type),
                    )
                )

        # Sort by position
        matches.sort(key=lambda m: m.matched_text)
        return matches

    def get_summary(self, matches: List[PIIMatch]) -> Dict[str, int]:
        """Get summary of matches by severity.

        Args:
            matches: List of PIIMatch objects

        Returns:
            Dict mapping severity to count
        """
        summary = {s.value: 0 for s in PIISeverity}
        for match in matches:
            summary[match.severity.value] += 1
        return summary


# =============================================================================
# Convenience Functions
# =============================================================================


def detect_pii_columns(columns: List[str]) -> List[Tuple[str, PIIType]]:
    """Detect PII columns in a list of column names.

    Args:
        columns: List of column names to check

    Returns:
        List of (column_name, pii_type) tuples
    """
    scanner = PIIScanner()
    return scanner.detect_columns(columns)


def detect_pii_in_content(content: str) -> List[PIIMatch]:
    """Detect PII in text content.

    Args:
        content: Text content to scan

    Returns:
        List of PIIMatch objects
    """
    scanner = PIIScanner()
    return scanner.scan_content(content)


def get_anonymization_suggestion(pii_type: PIIType) -> str:
    """Get anonymization suggestion for a PII type.

    Args:
        pii_type: Type of PII

    Returns:
        Suggestion string for anonymization
    """
    return ANONYMIZATION_SUGGESTIONS.get(pii_type, "Consider removing or hashing")


def get_pii_severity(pii_type: PIIType) -> PIISeverity:
    """Get severity level for a PII type.

    Args:
        pii_type: Type of PII

    Returns:
        Severity level
    """
    return PII_SEVERITY.get(pii_type, PIISeverity.MEDIUM)


def has_pii(content: str) -> bool:
    """Check if content contains any PII.

    Args:
        content: Text content to check

    Returns:
        True if PII is detected
    """
    return len(detect_pii_in_content(content)) > 0


def get_pii_types() -> List[str]:
    """Get list of all PII types.

    Returns:
        List of PII type names
    """
    return [t.value for t in PIIType]


def get_safety_reminders() -> List[str]:
    """Get safety reminders for handling PII.

    Returns:
        List of reminder strings
    """
    return [
        "Check for PII before sharing results or publishing",
        "Use sample/synthetic data for development, not production data",
        "Anonymize sensitive columns before analysis or export",
        "Document data sources, permissions, and retention policies",
        "Never commit datasets with PII to version control",
        "Apply principle of least privilege - only access needed data",
        "Implement audit logging for PII access",
        "Ensure compliance with GDPR, CCPA, HIPAA as applicable",
    ]


__all__ = [
    # Types
    "PIIType",
    "PIISeverity",
    "PIIMatch",
    # Patterns
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "ANONYMIZATION_SUGGESTIONS",
    # Scanner
    "PIIScanner",
    # Functions
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "has_pii",
    "get_pii_types",
    "get_safety_reminders",
]
