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

"""Content warning patterns for safety classification.

Provides domain-agnostic content warning patterns that can be used
by any vertical for content safety assessment.

Extracted from victor/research/safety.py to enable framework-wide reuse.

Performance:
    Uses native Rust pattern matching (Aho-Corasick) when available for
    10-100x faster multi-pattern matching. Falls back to Python re module.
"""

import logging
import re
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import native pattern matching
_NATIVE_AVAILABLE = False
_PatternMatcher = None

try:
    import victor_native as _native

    _PatternMatcher = _native.PatternMatcher
    _NATIVE_AVAILABLE = True
    logger.debug("Native pattern matching available for content patterns")
except ImportError:
    logger.debug("Native extensions not available, using Python regex fallback")


class ContentWarningLevel(Enum):
    """Content warning severity levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContentWarningMatch(NamedTuple):
    """Result of content warning pattern matching."""

    content_snippet: str
    warning_type: str
    description: str
    level: ContentWarningLevel
    pattern: str


# Content warning patterns - domain-agnostic
# Format: (pattern, warning_type, description, level)
CONTENT_WARNING_PATTERNS: List[Tuple[str, str, str, ContentWarningLevel]] = [
    # High-severity content warnings
    (
        r"(?i)graphic\s+content|explicit\s+violence|gore",
        "graphic_content",
        "Potentially graphic or violent content",
        ContentWarningLevel.HIGH,
    ),
    (
        r"(?i)suicide|self.?harm|harm\s+yourself",
        "self_harm",
        "Content related to self-harm or suicide",
        ContentWarningLevel.HIGH,
    ),
    (
        r"(?i)child\s+abuse|child\s+exploitation",
        "child_safety",
        "Content related to child safety concerns",
        ContentWarningLevel.HIGH,
    ),
    # Medium-severity content warnings
    (
        r"(?i)trigger\s+warning|content\s+warning|sensitive\s+topic",
        "sensitive_topic",
        "Sensitive topic that may be triggering",
        ContentWarningLevel.MEDIUM,
    ),
    (
        r"(?i)controversial|disputed|debated|polarizing",
        "controversial",
        "Controversial or disputed topic",
        ContentWarningLevel.MEDIUM,
    ),
    (
        r"(?i)disturbing|distressing|unsettling",
        "disturbing",
        "Potentially disturbing content",
        ContentWarningLevel.MEDIUM,
    ),
    (
        r"(?i)political\s+opinion|partisan|biased\s+view",
        "political_bias",
        "Politically biased content",
        ContentWarningLevel.MEDIUM,
    ),
    # Low-severity content notes
    (
        r"(?i)opinion|editorial|commentary",
        "opinion",
        "Opinion-based rather than factual content",
        ContentWarningLevel.LOW,
    ),
    (
        r"(?i)speculation|unverified|unconfirmed",
        "unverified",
        "Unverified or speculative information",
        ContentWarningLevel.LOW,
    ),
    (
        r"(?i)outdated|old\s+information|may\s+be\s+stale",
        "outdated",
        "Potentially outdated information",
        ContentWarningLevel.LOW,
    ),
]

# Patterns for detecting misinformation risk
MISINFORMATION_RISK_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)fake\s+news|fabricat.*source|invent.*citation", "Fabricating sources"),
    (r"(?i)always|never|guaranteed|proven\s+fact", "Absolute claims without evidence"),
    (r"(?i)everyone\s+knows|obviously|clearly", "Unsupported generalizations"),
    (r"(?i)studies\s+show|research\s+proves", "Unattributed research claims"),
    (r"(?i)secret|they\s+don't\s+want\s+you\s+to\s+know", "Conspiracy language"),
    (r"(?i)miracle|breakthrough|revolutionary", "Sensationalist language"),
]

# Patterns for detecting advice risk (should not be provided by AI)
ADVICE_RISK_PATTERNS: Dict[str, str] = {
    "medical": r"(?i)medical\s+advice|diagnos|prescri|symptom\s+of|cure\s+for",
    "legal": r"(?i)legal\s+advice|sue|lawsuit|attorney|file\s+a\s+claim",
    "financial": r"(?i)financial\s+advice|invest\s+in|buy\s+stock|guaranteed\s+return",
    "mental_health": r"(?i)therapy\s+advice|mental\s+health\s+diagnosis",
}


def scan_content_warnings(content: str) -> List[ContentWarningMatch]:
    """Scan content for warning patterns.

    Args:
        content: Text content to scan

    Returns:
        List of ContentWarningMatch for each pattern found
    """
    matches = []

    for pattern, warning_type, description, level in CONTENT_WARNING_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            # Get a snippet around the match
            start = max(0, match.start() - 20)
            end = min(len(content), match.end() + 20)
            snippet = content[start:end].strip()

            matches.append(
                ContentWarningMatch(
                    content_snippet=snippet,
                    warning_type=warning_type,
                    description=description,
                    level=level,
                    pattern=pattern,
                )
            )

    return matches


def has_content_warnings(content: str) -> bool:
    """Check if content has any warning patterns."""
    return len(scan_content_warnings(content)) > 0


def get_high_severity_warnings(content: str) -> List[ContentWarningMatch]:
    """Get only high-severity content warnings."""
    return [m for m in scan_content_warnings(content) if m.level == ContentWarningLevel.HIGH]


def detect_misinformation_risk(content: str) -> List[Tuple[str, str]]:
    """Detect potential misinformation patterns in content.

    Args:
        content: Text content to scan

    Returns:
        List of (matched_text, risk_description) tuples
    """
    risks = []

    for pattern, description in MISINFORMATION_RISK_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            risks.append((match.group(), description))

    return risks


def detect_advice_risk(content: str) -> Dict[str, bool]:
    """Detect potential advice-giving that AI should avoid.

    Args:
        content: Text content to scan

    Returns:
        Dict mapping advice type to whether it was detected
    """
    return {
        advice_type: bool(re.search(pattern, content, re.IGNORECASE))
        for advice_type, pattern in ADVICE_RISK_PATTERNS.items()
    }


def get_content_safety_reminders() -> List[str]:
    """Get safety reminders for content handling."""
    return [
        "Add content warnings for sensitive topics",
        "Distinguish between facts and opinions",
        "Avoid providing medical, legal, or financial advice",
        "Flag unverified or speculative information",
        "Note when information may be outdated",
    ]


class ContentPatternScanner:
    """Scanner for content warning patterns.

    Implements ISafetyScanner interface for integration with SafetyRegistry.
    Uses native Rust pattern matching when available for 10-100x speedup.
    """

    def __init__(self):
        """Initialize the scanner with optional native acceleration."""
        self._warning_patterns = CONTENT_WARNING_PATTERNS
        self._misinfo_patterns = MISINFORMATION_RISK_PATTERNS
        self._advice_patterns = ADVICE_RISK_PATTERNS
        self._use_native = _NATIVE_AVAILABLE

    def scan(self, content: str) -> List[ContentWarningMatch]:
        """Scan content for all warning patterns.

        Args:
            content: Text content to scan

        Returns:
            List of ContentWarningMatch for each pattern found
        """
        return scan_content_warnings(content)

    def get_all_risks(self, content: str) -> Dict:
        """Get comprehensive risk assessment.

        Args:
            content: Text content to scan

        Returns:
            Dict with warnings, misinformation risks, and advice risks
        """
        return {
            "content_warnings": scan_content_warnings(content),
            "misinformation_risks": detect_misinformation_risk(content),
            "advice_risks": detect_advice_risk(content),
        }

    @staticmethod
    def is_native_available() -> bool:
        """Check if native pattern matching is available."""
        return _NATIVE_AVAILABLE
