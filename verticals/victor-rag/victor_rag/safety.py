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

"""RAG Safety Extension - Security patterns for RAG operations.

This module provides RAG-specific safety patterns for:
- Bulk deletion warnings
- Untrusted source ingestion detection
- PII detection in documents (delegates to framework)
- Secret detection (API keys, private keys)
- Data loss prevention

Example:
    from victor_rag.safety import RAGSafetyExtension

    safety = RAGSafetyExtension()
    patterns = safety.get_bash_patterns()

    # Check for dangerous operations
    for pattern in patterns:
        if pattern.matches(command):
            print(f"Warning: {pattern.description}")

    # Scan document content for PII (uses framework PIIScanner)
    pii_matches = safety.scan_for_pii(document_content)
"""

import re
from typing import Dict, List, Tuple

from victor_contracts.verticals.protocols import SafetyExtensionProtocol
from victor_contracts.verticals.protocols.promoted_types import (
    SafetyPatternData as SafetyPattern,
)
from victor_contracts.safety.framework import SafetyEnforcer, SafetyRule, SafetyLevel
from victor_contracts.safety import (
    PIISeverity,
    PII_CONTENT_PATTERNS,
    detect_pii_in_content,
)

# Risk levels
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


# RAG-specific dangerous patterns
RAG_DANGER_PATTERNS: List[SafetyPattern] = [
    # Bulk deletion patterns
    SafetyPattern(
        pattern=r"rag_delete\s+.*\*",
        description="Bulk deletion with wildcard - may delete all documents",
        risk_level=HIGH,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"rag_delete\s+--all",
        description="Delete all documents from knowledge base",
        risk_level=HIGH,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"DELETE\s+FROM\s+.*WHERE\s+1\s*=\s*1",
        description="SQL pattern that deletes all records",
        risk_level=HIGH,
        category="rag",
    ),
    # Untrusted source patterns
    SafetyPattern(
        pattern=r"rag_ingest.*http://(?!localhost)",
        description="Ingesting from non-HTTPS URL - potential untrusted source",
        risk_level=MEDIUM,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"rag_ingest.*(?:pastebin|gist\.github|hastebin)",
        description="Ingesting from paste service - potentially untrusted content",
        risk_level=MEDIUM,
        category="rag",
    ),
    # Large batch operations
    SafetyPattern(
        pattern=r"rag_ingest.*--batch\s+\d{4,}",
        description="Large batch ingestion (1000+ files) - verify this is intentional",
        risk_level=LOW,
        category="rag",
    ),
]

# RAG-specific secret detection patterns (not standard PII)
# Standard PII patterns are delegated to victor.security.safety.pii
SECRET_PATTERNS: Dict[str, Tuple[str, str, str]] = {
    "aws_key": (
        r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}",
        "AWS access key detected",
        HIGH,
    ),
    "api_key": (
        r"(?i)(?:api[_-]?key|apikey)['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{20,}",
        "API key pattern detected",
        HIGH,
    ),
    "private_key": (
        r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----",
        "Private key detected",
        HIGH,
    ),
    "github_token": (
        r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}",
        "GitHub token detected",
        HIGH,
    ),
    "slack_token": (
        r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}",
        "Slack token detected",
        HIGH,
    ),
}

# Document ingestion safety patterns
INGESTION_SAFETY_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"\.(?:exe|dll|bat|cmd|sh|ps1)$",
        description="Executable file type - should not be ingested as document",
        risk_level=MEDIUM,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"\.(?:zip|tar|gz|7z|rar)$",
        description="Archive file - consider extracting before ingestion",
        risk_level=LOW,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"(?:/etc/passwd|/etc/shadow|\.ssh/)",
        description="System file path - may contain sensitive data",
        risk_level=HIGH,
        category="rag",
    ),
]


class RAGSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for RAG tasks.

    Provides RAG-specific dangerous operation patterns including:
    - Bulk deletion warnings
    - Untrusted source detection
    - PII detection in documents
    - Data loss prevention

    Example:
        safety = RAGSafetyExtension()

        # Get all dangerous patterns
        patterns = safety.get_bash_patterns()

        # Scan document content for PII
        pii_matches = safety.scan_for_pii(document_content)
    """

    def __init__(
        self,
        include_pii_detection: bool = True,
        include_ingestion_safety: bool = True,
    ):
        """Initialize the safety extension.

        Args:
            include_pii_detection: Include PII detection patterns
            include_ingestion_safety: Include ingestion safety patterns
        """
        self._include_pii_detection = include_pii_detection
        self._include_ingestion_safety = include_ingestion_safety

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Return RAG-specific bash patterns.

        Returns:
            List of SafetyPattern for dangerous RAG commands.
        """
        patterns = list(RAG_DANGER_PATTERNS)
        if self._include_ingestion_safety:
            patterns.extend(INGESTION_SAFETY_PATTERNS)
        return patterns

    def get_danger_patterns(self) -> List[Tuple[str, str, str]]:
        """Return RAG-specific danger patterns (legacy format).

        Returns:
            List of (regex_pattern, description, risk_level) tuples.
        """
        return [(p.pattern, p.description, p.risk_level) for p in self.get_bash_patterns()]

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Return file operation patterns for RAG.

        Returns:
            List of safety patterns for dangerous file operations.
        """
        return INGESTION_SAFETY_PATTERNS if self._include_ingestion_safety else []

    def get_blocked_operations(self) -> List[str]:
        """Return operations that should be blocked in RAG context.

        Returns:
            List of blocked operation descriptions.
        """
        return [
            "bulk_delete_all_documents",
            "ingest_executable_files",
            "expose_pii_to_logs",
            "ingest_system_files",
        ]

    def get_pii_patterns(self) -> Dict[str, Tuple[str, str, str]]:
        """Return patterns for detecting PII and secrets in documents.

        Combines framework PII patterns with RAG-specific secret patterns.

        Returns:
            Dict of pii_type -> (regex_pattern, description, risk_level).
        """
        # Start with framework PII patterns (canonical source)
        patterns: Dict[str, Tuple[str, str, str]] = {}
        for pii_type, (pattern, severity) in PII_CONTENT_PATTERNS.items():
            risk = (
                HIGH
                if severity in (PIISeverity.CRITICAL, PIISeverity.HIGH)
                else (MEDIUM if severity == PIISeverity.MEDIUM else LOW)
            )
            patterns[pii_type.value] = (
                pattern,
                f"{pii_type.value.replace('_', ' ').title()} detected",
                risk,
            )

        # Add RAG-specific secret patterns
        patterns.update(SECRET_PATTERNS)
        return patterns

    def scan_for_pii(self, content: str) -> List[Dict]:
        """Scan content for PII and secrets.

        Delegates to framework PIIScanner for standard PII detection,
        then adds RAG-specific secret pattern scanning.

        Args:
            content: Text content to scan

        Returns:
            List of match dictionaries with type, severity, and location
        """
        if not self._include_pii_detection:
            return []

        matches = []
        lines = content.split("\n")

        # Delegate to framework PIIScanner for standard PII
        pii_matches = detect_pii_in_content(content)
        for pii_match in pii_matches:
            # Find line number for the match
            line_num = 1
            col = 0
            for i, line in enumerate(lines, 1):
                if pii_match.matched_text in line:
                    line_num = i
                    col = line.find(pii_match.matched_text)
                    break

            matches.append(
                {
                    "type": pii_match.pii_type.value,
                    "description": f"{pii_match.pii_type.value.replace('_', ' ').title()} detected",
                    "severity": (
                        HIGH
                        if pii_match.severity in (PIISeverity.CRITICAL, PIISeverity.HIGH)
                        else (MEDIUM if pii_match.severity == PIISeverity.MEDIUM else LOW)
                    ),
                    "line": line_num,
                    "column": col,
                    "masked_value": pii_match.matched_text,  # Already masked by PIIMatch
                }
            )

        # Add RAG-specific secret pattern scanning
        for secret_type, (pattern, description, risk_level) in SECRET_PATTERNS.items():
            compiled = re.compile(pattern)
            for line_num, line in enumerate(lines, 1):
                for match in compiled.finditer(line):
                    matches.append(
                        {
                            "type": secret_type,
                            "description": description,
                            "severity": risk_level,
                            "line": line_num,
                            "column": match.start(),
                            "masked_value": self._mask_value(match.group(), secret_type),
                        }
                    )

        return matches

    def _mask_value(self, value: str, pii_type: str) -> str:
        """Mask a detected PII value.

        Args:
            value: The detected value
            pii_type: Type of PII

        Returns:
            Masked version of the value
        """
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def validate_ingestion_source(self, source: str) -> List[str]:
        """Validate an ingestion source for safety issues.

        Args:
            source: File path or URL to validate

        Returns:
            List of warning messages (empty if safe)
        """
        warnings = []

        for pattern in INGESTION_SAFETY_PATTERNS:
            if re.search(pattern.pattern, source, re.IGNORECASE):
                warnings.append(f"{pattern.risk_level}: {pattern.description}")

        # Check for untrusted URLs
        for pattern in RAG_DANGER_PATTERNS:
            if "ingest" in pattern.pattern.lower():
                if re.search(pattern.pattern, f"rag_ingest {source}", re.IGNORECASE):
                    warnings.append(f"{pattern.risk_level}: {pattern.description}")

        return warnings

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for RAG output.

        Returns:
            List of safety reminder strings.
        """
        return [
            "Verify source authenticity before ingesting external content",
            "Review documents for PII before adding to knowledge base",
            "Use --dry-run for bulk operations when available",
            "Backup knowledge base before major deletions",
            "Consider content licensing and copyright restrictions",
        ]

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier
        """
        return "rag"

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dict mapping tool names to list of restricted argument patterns.
        """
        return {
            "rag_delete": [
                r"--all",
                r"\*",
                r"--force\s+--all",
            ],
            "rag_ingest": [
                r"/etc/",
                r"\.ssh/",
                r"credentials",
                r"secrets",
            ],
        }


__all__ = [
    "RAGSafetyExtension",
    "RAG_DANGER_PATTERNS",
    "INGESTION_SAFETY_PATTERNS",
    "SECRET_PATTERNS",
    "HIGH",
    "MEDIUM",
    "LOW",
    # New framework-based safety rules
    "create_rag_deletion_safety_rules",
    "create_rag_ingestion_safety_rules",
    "create_all_rag_safety_rules",
]


# =============================================================================
# Framework-Based Safety Rules (New)
# =============================================================================

"""Framework-based safety rules for RAG operations.

This section provides factory functions that register safety rules
with the framework-level SafetyEnforcer. This is the new recommended
approach for safety enforcement in RAG workflows.

Example:
    from victor_contracts.safety.framework import SafetyEnforcer, SafetyConfig, SafetyLevel
    from victor_rag.safety import create_all_rag_safety_rules

    enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
    create_all_rag_safety_rules(enforcer)

    # Check operations
    allowed, reason = enforcer.check_operation("rag_delete --all")
    if not allowed:
        print(f"Blocked: {reason}")
"""


def create_rag_deletion_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_bulk_delete: bool = True,
    block_delete_all: bool = True,
    protected_collections: list[str] | None = None,
) -> None:
    """Register RAG deletion-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_bulk_delete: Block bulk deletion operations with wildcards
        block_delete_all: Block deleting all documents
        protected_collections: List of protected collection names

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_rag_deletion_safety_rules(
            enforcer,
            block_bulk_delete=True,
            protected_collections=["production", "main"]
        )
    """
    if block_bulk_delete:
        enforcer.add_rule(
            SafetyRule(
                name="rag_block_bulk_delete",
                description="Block bulk deletion with wildcards (rag_delete *)",
                check_fn=lambda op: "rag_delete" in op
                and ("*" in op or "--all" in op or "WHERE 1=1" in op),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_delete_all:
        enforcer.add_rule(
            SafetyRule(
                name="rag_block_delete_all",
                description="Block deleting all documents from knowledge base",
                check_fn=lambda op: "rag_delete" in op and "--all" in op,
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )


def create_rag_ingestion_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_executable_files: bool = True,
    block_system_files: bool = True,
    require_https: bool = True,
    warn_large_batches: bool = True,
) -> None:
    """Register RAG ingestion-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_executable_files: Block ingestion of executable files (.exe, .dll, .sh)
        block_system_files: Block ingestion of system files (/etc/, ~/.ssh/)
        require_https: Require HTTPS URLs for ingestion
        warn_large_batches: Warn for large batch ingestion (1000+ files)

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_rag_ingestion_safety_rules(
            enforcer,
            block_executable_files=True,
            require_https=True
        )
    """
    if block_executable_files:
        enforcer.add_rule(
            SafetyRule(
                name="rag_block_executable_ingestion",
                description="Block ingestion of executable files",
                check_fn=lambda op: "rag_ingest" in op
                and any(
                    ext in op.lower() for ext in [".exe", ".dll", ".bat", ".cmd", ".sh", ".ps1"]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_system_files:
        enforcer.add_rule(
            SafetyRule(
                name="rag_block_system_files",
                description="Block ingestion of system files (/etc/, ~/.ssh/)",
                check_fn=lambda op: "rag_ingest" in op
                and any(path in op for path in ["/etc/", "/.ssh/", "~/.ssh/", "passwd", "shadow"]),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_https:
        enforcer.add_rule(
            SafetyRule(
                name="rag_require_https",
                description="Require HTTPS for remote ingestion (block http://)",
                check_fn=lambda op: "rag_ingest" in op
                and "http://" in op
                and "https://" not in op
                and "localhost" not in op,
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if warn_large_batches:
        enforcer.add_rule(
            SafetyRule(
                name="rag_warn_large_batches",
                description="Warn for large batch ingestion (1000+ files)",
                check_fn=lambda op: "rag_ingest" in op and "--batch" in op,
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_all_rag_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    protected_collections: list[str] | None = None,
) -> None:
    """Register all RAG safety rules at once.

    This is a convenience function that registers all RAG-specific
    safety rules with appropriate defaults.

    Args:
        enforcer: SafetyEnforcer to register rules with
        protected_collections: Protected collection names

    Example:
        from victor_contracts.safety.framework import SafetyEnforcer, SafetyConfig, SafetyLevel

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_rag_safety_rules(enforcer)

        # Now all operations are checked
        allowed, reason = enforcer.check_operation("rag_delete --all")
    """
    create_rag_deletion_safety_rules(enforcer, protected_collections=protected_collections)
    create_rag_ingestion_safety_rules(enforcer)
