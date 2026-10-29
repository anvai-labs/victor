"""SDK-owned PII detection helpers for external verticals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Pattern, Tuple


class PIIType(Enum):
    """Types of personally identifiable information."""

    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    SSN = "ssn"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    TAX_ID = "tax_id"
    MEDICAL_RECORD = "medical_record"
    HEALTH_INSURANCE = "health_insurance"
    DOB = "date_of_birth"
    AGE = "age"
    GENDER = "gender"
    RACE = "race"
    RELIGION = "religion"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    USER_ID = "user_id"
    USERNAME = "username"
    BIOMETRIC = "biometric"


class PIISeverity(Enum):
    """Sensitivity levels for PII types."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def to_severity(self) -> Any:
        """Convert to the host runtime's unified ``SeverityLevel``."""
        try:
            from victor.core.severity import SeverityLevel
        except ImportError as exc:  # pragma: no cover - only exercised outside host runtime
            raise RuntimeError(
                "PIISeverity.to_severity() requires the Victor runtime host package."
            ) from exc
        return SeverityLevel.from_pii_severity(self)


@dataclass
class PIIMatch:
    """A detected PII element with a redacted display value."""

    pii_type: PIIType
    matched_text: str
    source: str
    severity: PIISeverity
    suggestion: str = ""

    def __post_init__(self) -> None:
        if len(self.matched_text) > 8:
            visible = min(4, len(self.matched_text) // 4)
            self.matched_text = f"{self.matched_text[:visible]}***{self.matched_text[-visible:]}"


PII_COLUMN_PATTERNS: Dict[PIIType, str] = {
    PIIType.NAME: r"(?i)^(first|last|full|middle|maiden)?[_\s-]?name$",
    PIIType.EMAIL: r"(?i)e?mail(_address)?",
    PIIType.PHONE: r"(?i)(phone|mobile|cell|tel|fax)(_num(ber)?)?",
    PIIType.ADDRESS: r"(?i)(address|street|city|state|zip|postal|country)",
    PIIType.SSN: r"(?i)(ssn|social[_\s-]?security|sin|national[_\s-]?id)",
    PIIType.PASSPORT: r"(?i)passport(_num(ber)?)?",
    PIIType.DRIVERS_LICENSE: r"(?i)(driver'?s?[_\s-]?licen[sc]e|dl[_\s-]?num)",
    PIIType.TAX_ID: r"(?i)(tax[_\s-]?id|ein|tin|vat)",
    PIIType.CREDIT_CARD: r"(?i)(credit[_\s-]?card|cc[_\s-]?num|card[_\s-]?num)",
    PIIType.BANK_ACCOUNT: r"(?i)(bank[_\s-]?account|iban|routing|swift)",
    PIIType.MEDICAL_RECORD: r"(?i)(medical|diagnosis|health[_\s-]?record|patient[_\s-]?id)",
    PIIType.HEALTH_INSURANCE: r"(?i)(insurance[_\s-]?id|policy[_\s-]?num|member[_\s-]?id)",
    PIIType.DOB: r"(?i)(dob|birth[_\s-]?date|date[_\s-]?of[_\s-]?birth|birthday)",
    PIIType.AGE: r"(?i)^age$",
    PIIType.GENDER: r"(?i)^(gender|sex)$",
    PIIType.RACE: r"(?i)^(race|ethnicity)$",
    PIIType.RELIGION: r"(?i)^religion$",
    PIIType.IP_ADDRESS: r"(?i)ip[_\s-]?addr(ess)?",
    PIIType.MAC_ADDRESS: r"(?i)mac[_\s-]?addr(ess)?",
    PIIType.USER_ID: r"(?i)(user[_\s-]?id|customer[_\s-]?id|account[_\s-]?id|member[_\s-]?id)",
    PIIType.USERNAME: r"(?i)(username|user[_\s-]?name|login|handle)",
}

PII_CONTENT_PATTERNS: Dict[PIIType, Tuple[str, PIISeverity]] = {
    PIIType.SSN: (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", PIISeverity.CRITICAL),
    PIIType.CREDIT_CARD: (
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b",
        PIISeverity.CRITICAL,
    ),
    PIIType.EMAIL: (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        PIISeverity.MEDIUM,
    ),
    PIIType.PHONE: (
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        PIISeverity.MEDIUM,
    ),
    PIIType.IP_ADDRESS: (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        PIISeverity.LOW,
    ),
    PIIType.MAC_ADDRESS: (
        r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        PIISeverity.LOW,
    ),
    PIIType.DOB: (
        r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b",
        PIISeverity.MEDIUM,
    ),
    PIIType.PASSPORT: (r"\b[A-Z]{1,2}\d{6,9}\b", PIISeverity.CRITICAL),
}

PII_SEVERITY: Dict[PIIType, PIISeverity] = {
    PIIType.SSN: PIISeverity.CRITICAL,
    PIIType.PASSPORT: PIISeverity.CRITICAL,
    PIIType.DRIVERS_LICENSE: PIISeverity.CRITICAL,
    PIIType.MEDICAL_RECORD: PIISeverity.CRITICAL,
    PIIType.CREDIT_CARD: PIISeverity.CRITICAL,
    PIIType.BANK_ACCOUNT: PIISeverity.CRITICAL,
    PIIType.BIOMETRIC: PIISeverity.CRITICAL,
    PIIType.NAME: PIISeverity.HIGH,
    PIIType.TAX_ID: PIISeverity.HIGH,
    PIIType.HEALTH_INSURANCE: PIISeverity.HIGH,
    PIIType.EMAIL: PIISeverity.MEDIUM,
    PIIType.PHONE: PIISeverity.MEDIUM,
    PIIType.ADDRESS: PIISeverity.MEDIUM,
    PIIType.DOB: PIISeverity.MEDIUM,
    PIIType.USER_ID: PIISeverity.MEDIUM,
    PIIType.USERNAME: PIISeverity.MEDIUM,
    PIIType.AGE: PIISeverity.LOW,
    PIIType.GENDER: PIISeverity.LOW,
    PIIType.RACE: PIISeverity.LOW,
    PIIType.RELIGION: PIISeverity.LOW,
    PIIType.IP_ADDRESS: PIISeverity.LOW,
    PIIType.MAC_ADDRESS: PIISeverity.LOW,
}

ANONYMIZATION_SUGGESTIONS: Dict[PIIType, str] = {
    PIIType.NAME: "Replace with fake names using Faker library, or hash with salt",
    PIIType.EMAIL: "Replace domain with example.com, hash local part, or remove entirely",
    PIIType.PHONE: "Remove or replace with random numbers preserving format",
    PIIType.ADDRESS: "Generalize to region/city level, use zip code prefix only",
    PIIType.SSN: "Remove entirely - never store in analysis datasets",
    PIIType.PASSPORT: "Remove entirely or replace with random identifier",
    PIIType.DRIVERS_LICENSE: "Remove entirely or hash with salt",
    PIIType.TAX_ID: "Remove or tokenize with secure mapping",
    PIIType.CREDIT_CARD: "Mask all but last 4 digits, or remove entirely",
    PIIType.BANK_ACCOUNT: "Remove entirely or tokenize",
    PIIType.MEDICAL_RECORD: "Use de-identification per HIPAA Safe Harbor method",
    PIIType.HEALTH_INSURANCE: "Tokenize or remove",
    PIIType.DOB: "Convert to age ranges (e.g., 25-34) or year only",
    PIIType.AGE: "Convert to age ranges",
    PIIType.GENDER: "Consider if needed; use categories or remove",
    PIIType.RACE: "Aggregate to broader categories if required; otherwise remove",
    PIIType.RELIGION: "Remove unless essential; aggregate if kept",
    PIIType.IP_ADDRESS: "Mask last octet (e.g., 192.168.1.x) or hash",
    PIIType.MAC_ADDRESS: "Hash or remove",
    PIIType.USER_ID: "Hash with salt or replace with sequential IDs",
    PIIType.USERNAME: "Hash or replace with pseudonyms",
    PIIType.BIOMETRIC: "Remove entirely - biometric data cannot be anonymized",
}


class PIIScanner:
    """Pattern-based PII scanner for columns and text content."""

    def __init__(self) -> None:
        self._column_patterns: Dict[PIIType, Pattern[str]] = {
            pii_type: re.compile(pattern) for pii_type, pattern in PII_COLUMN_PATTERNS.items()
        }
        self._content_patterns: Dict[PIIType, Pattern[str]] = {
            pii_type: re.compile(pattern) for pii_type, (pattern, _) in PII_CONTENT_PATTERNS.items()
        }

    def detect_columns(self, columns: List[str]) -> List[Tuple[str, PIIType]]:
        """Detect likely PII columns by name."""
        matches = []
        for column in columns:
            for pii_type, pattern in self._column_patterns.items():
                if pattern.search(column):
                    matches.append((column, pii_type))
                    break
        return matches

    def scan_content(self, content: str) -> List[PIIMatch]:
        """Scan text content for PII."""
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
        matches.sort(key=lambda match: match.matched_text)
        return matches

    def get_summary(self, matches: List[PIIMatch]) -> Dict[str, int]:
        """Return match counts by severity."""
        summary = {severity.value: 0 for severity in PIISeverity}
        for match in matches:
            summary[match.severity.value] += 1
        return summary


def detect_pii_columns(columns: List[str]) -> List[Tuple[str, PIIType]]:
    """Detect PII columns in a list of column names."""
    return PIIScanner().detect_columns(columns)


def detect_pii_in_content(content: str) -> List[PIIMatch]:
    """Detect PII in text content."""
    return PIIScanner().scan_content(content)


def get_anonymization_suggestion(pii_type: PIIType) -> str:
    """Get an anonymization suggestion for a PII type."""
    return ANONYMIZATION_SUGGESTIONS.get(pii_type, "Consider removing or hashing")


def get_pii_severity(pii_type: PIIType) -> PIISeverity:
    """Get the default sensitivity level for a PII type."""
    return PII_SEVERITY.get(pii_type, PIISeverity.MEDIUM)


def has_pii(content: str) -> bool:
    """Return True when content contains supported PII patterns."""
    return bool(detect_pii_in_content(content))


def get_pii_types() -> List[str]:
    """Return all supported PII type values."""
    return [pii_type.value for pii_type in PIIType]


def get_safety_reminders() -> List[str]:
    """Return safety reminders for handling PII."""
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
    "ANONYMIZATION_SUGGESTIONS",
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "PIIMatch",
    "PIIScanner",
    "PIISeverity",
    "PIIType",
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "get_pii_types",
    "get_safety_reminders",
    "has_pii",
]
