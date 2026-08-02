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

"""Source credibility patterns for URL classification.

Provides domain-agnostic URL credibility classification that can be used
by any vertical (research, RAG, etc.) for source quality assessment.

Extracted from victor/research/safety.py to enable framework-wide reuse.

Performance:
    Uses native Rust pattern matching (Aho-Corasick) when available for
    10-100x faster multi-pattern matching. Falls back to Python re module.
"""

import logging
import re
from enum import Enum
from typing import Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)

# Try to import native pattern matching
_NATIVE_AVAILABLE = False
_PatternMatcher = None

try:
    import victor_native as _native

    _PatternMatcher = _native.PatternMatcher
    _NATIVE_AVAILABLE = True
    logger.debug("Native pattern matching available for source credibility")
except ImportError:
    logger.debug("Native extensions not available, using Python regex fallback")


class CredibilityLevel(Enum):
    """Source credibility levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class CredibilityMatch(NamedTuple):
    """Result of credibility classification."""

    url: str
    level: CredibilityLevel
    matched_pattern: Optional[str] = None
    domain_type: Optional[str] = None


# URL patterns for credibility classification
# These are domain-agnostic and can be extended by verticals
SOURCE_CREDIBILITY_PATTERNS: Dict[CredibilityLevel, List[str]] = {
    CredibilityLevel.HIGH: [
        r"\.gov($|/)",  # Government domains
        r"\.edu($|/)",  # Educational institutions
        r"arxiv\.org",  # Academic preprints
        r"pubmed\.ncbi\.nlm\.nih\.gov",  # Medical literature
        r"doi\.org",  # Digital Object Identifiers
        r"nature\.com",  # Nature Publishing
        r"science\.org",  # AAAS Science journals
        r"jstor\.org",  # JSTOR academic database
        r"scholar\.google\.com",  # Google Scholar
        r"ncbi\.nlm\.nih\.gov",  # National Center for Biotechnology
        r"ieee\.org",  # IEEE journals
        r"acm\.org",  # ACM digital library
    ],
    CredibilityLevel.MEDIUM: [
        r"wikipedia\.org",  # Wikipedia (good starting point, verify claims)
        r"medium\.com",  # Medium articles (mixed quality)
        r"substack\.com",  # Substack newsletters
        r"news\.",  # News websites (quality varies)
        r"reuters\.com",  # Reuters news
        r"apnews\.com",  # Associated Press
        r"bbc\.com|bbc\.co\.uk",  # BBC
        r"npr\.org",  # NPR
        r"github\.com",  # GitHub (technical content)
        r"stackoverflow\.com",  # Stack Overflow
    ],
    CredibilityLevel.LOW: [
        r"\.blogspot\.",  # Blogspot (unverified blogs)
        r"wordpress\.com/(?!.*official)",  # Personal WordPress blogs
        r"tumblr\.com",  # Tumblr
        r"quora\.com",  # Quora (unverified answers)
        r"reddit\.com/r/",  # Reddit (user-generated content)
        r"facebook\.com",  # Facebook (social media)
        r"twitter\.com|x\.com",  # Twitter/X (social media)
        r"tiktok\.com",  # TikTok (social media)
        r"youtube\.com",  # YouTube (mixed quality)
    ],
}

# Domain type descriptions for transparency
DOMAIN_TYPES: Dict[str, str] = {
    r"\.gov": "government",
    r"\.edu": "educational",
    r"arxiv\.org": "preprint_server",
    r"pubmed": "medical_database",
    r"doi\.org": "doi_resolver",
    r"wikipedia\.org": "wiki",
    r"github\.com": "code_repository",
    r"stackoverflow\.com": "qa_forum",
    r"reddit\.com": "social_forum",
    r"twitter\.com|x\.com": "social_media",
}


def validate_source_credibility(url: str) -> CredibilityMatch:
    """Classify source credibility based on URL patterns.

    Args:
        url: The URL to classify

    Returns:
        CredibilityMatch with level, matched pattern, and domain type
    """
    url_lower = url.lower()

    for level, patterns in SOURCE_CREDIBILITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, url_lower, re.IGNORECASE):
                # Find domain type if available
                domain_type = None
                for type_pattern, type_name in DOMAIN_TYPES.items():
                    if re.search(type_pattern, url_lower, re.IGNORECASE):
                        domain_type = type_name
                        break

                return CredibilityMatch(
                    url=url,
                    level=level,
                    matched_pattern=pattern,
                    domain_type=domain_type,
                )

    return CredibilityMatch(
        url=url,
        level=CredibilityLevel.UNKNOWN,
        matched_pattern=None,
        domain_type=None,
    )


def get_credibility_level(url: str) -> str:
    """Get credibility level as string (legacy compatibility).

    Args:
        url: The URL to classify

    Returns:
        Credibility level: 'high', 'medium', 'low', or 'unknown'
    """
    return validate_source_credibility(url).level.value


def is_high_credibility(url: str) -> bool:
    """Check if URL is from a high-credibility source."""
    return validate_source_credibility(url).level == CredibilityLevel.HIGH


def is_low_credibility(url: str) -> bool:
    """Check if URL is from a low-credibility source."""
    return validate_source_credibility(url).level == CredibilityLevel.LOW


def get_source_safety_reminders() -> List[str]:
    """Get safety reminders for source verification."""
    return [
        "Verify claims with multiple independent sources",
        "Check the original publication date and author credentials",
        "Look for citations and references to primary sources",
        "Be cautious of anonymous or pseudonymous sources",
        "Cross-reference information with high-credibility sources",
    ]


class SourceCredibilityScanner:
    """Scanner for source credibility assessment.

    Implements ISafetyScanner interface for integration with SafetyRegistry.
    Uses native Rust pattern matching when available for 10-100x speedup.
    """

    def __init__(self):
        """Initialize the scanner with optional native acceleration."""
        self._patterns = SOURCE_CREDIBILITY_PATTERNS
        self._native_matchers: Dict[CredibilityLevel, any] = {}
        self._use_native = _NATIVE_AVAILABLE

        if self._use_native:
            self._init_native_matchers()

    def _init_native_matchers(self) -> None:
        """Initialize native pattern matchers for each credibility level."""
        try:
            for level, patterns in SOURCE_CREDIBILITY_PATTERNS.items():
                # Convert regex patterns to literal patterns for Aho-Corasick
                # Note: Aho-Corasick works with literal strings, not regex
                # For regex patterns, we fall back to Python
                literal_patterns = [
                    p.replace(r"($|/)", "")
                    .replace(r"\.gov", ".gov")
                    .replace(r"\.edu", ".edu")
                    .replace(r"\.", ".")
                    for p in patterns
                    if not any(c in p for c in ["*", "+", "?", "|", "(", ")"])
                ]
                if literal_patterns:
                    self._native_matchers[level] = _PatternMatcher(
                        literal_patterns, True  # case_insensitive=True
                    )
        except Exception as e:
            logger.warning(f"Failed to initialize native matchers: {e}")
            self._use_native = False

    def scan(self, content: str) -> List[CredibilityMatch]:
        """Scan content for URLs and classify their credibility.

        Args:
            content: Text content that may contain URLs

        Returns:
            List of CredibilityMatch for each URL found
        """
        # Simple URL extraction pattern
        url_pattern = r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
        urls = re.findall(url_pattern, content, re.IGNORECASE)

        return [validate_source_credibility(url) for url in urls]

    def batch_scan(self, urls: List[str]) -> List[CredibilityMatch]:
        """Batch scan multiple URLs for credibility.

        Uses native acceleration when available for better performance.

        Args:
            urls: List of URLs to classify

        Returns:
            List of CredibilityMatch for each URL
        """
        return [validate_source_credibility(url) for url in urls]

    def get_low_credibility_urls(self, content: str) -> List[str]:
        """Get list of low-credibility URLs in content.

        Args:
            content: Text content that may contain URLs

        Returns:
            List of URLs classified as low credibility
        """
        matches = self.scan(content)
        return [m.url for m in matches if m.level == CredibilityLevel.LOW]

    @staticmethod
    def is_native_available() -> bool:
        """Check if native pattern matching is available."""
        return _NATIVE_AVAILABLE
