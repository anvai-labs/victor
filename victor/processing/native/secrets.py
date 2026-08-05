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

"""Secret detection and pattern matching with native acceleration."""

import re
from typing import Any, Dict, List, Optional

from victor.processing.native._base import _NATIVE_AVAILABLE, _native

# =============================================================================
# SECRET DETECTION
# =============================================================================


class SecretMatchFallback:
    """Fallback class for secret match results."""

    def __init__(
        self,
        secret_type: str,
        matched_text: str,
        severity: str,
        start: int,
        end: int,
        line_number: int,
    ):
        self.secret_type = secret_type
        self.matched_text = matched_text
        self.severity = severity
        self.start = start
        self.end = end
        self.line_number = line_number

    def __repr__(self) -> str:
        return f"SecretMatch(type='{self.secret_type}', severity='{self.severity}', line={self.line_number})"


# Secret patterns for fallback
_SECRET_PATTERNS = [
    ("aws_access_key", r"AKIA[0-9A-Z]{16}", "high"),
    ("github_token", r"gh[pousr]_[A-Za-z0-9_]{36,}", "high"),
    ("google_api_key", r"AIza[0-9A-Za-z_-]{35}", "high"),
    ("stripe_key", r"(?:sk|pk)_(?:live|test)_[a-zA-Z0-9]{24,}", "high"),
    ("jwt_token", r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "medium"),
    (
        "private_key",
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "critical",
    ),
]


def scan_secrets(text: str) -> List[Any]:
    """Scan text for secrets.

    Args:
        text: Text to scan

    Returns:
        List of SecretMatch objects
    """
    if _NATIVE_AVAILABLE:
        return _native.scan_secrets(text)

    # Pure Python fallback
    matches = []
    lines = text.split("\n")
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)

    for name, pattern, severity in _SECRET_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # Find line number
            line_num = 1
            for i, start in enumerate(line_starts):
                if start > m.start():
                    break
                line_num = i + 1
            matches.append(
                SecretMatchFallback(name, m.group(), severity, m.start(), m.end(), line_num)
            )

    return matches


def has_secrets(text: str) -> bool:
    """Check if text contains secrets.

    Args:
        text: Text to check

    Returns:
        True if secrets are found
    """
    if _NATIVE_AVAILABLE:
        return _native.has_secrets(text)

    # Pure Python fallback
    for _, pattern, _ in _SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def get_secret_types(text: str) -> List[str]:
    """Get types of secrets found in text.

    Args:
        text: Text to scan

    Returns:
        List of secret type names
    """
    if _NATIVE_AVAILABLE:
        return _native.get_secret_types(text)

    # Pure Python fallback
    types = []
    for name, pattern, _ in _SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            types.append(name)
    return types


def mask_secrets(text: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask secrets in text.

    Args:
        text: Text containing secrets
        mask_char: Character to use for masking
        visible_chars: Number of chars to keep visible at start/end

    Returns:
        Text with secrets masked
    """
    if _NATIVE_AVAILABLE:
        return _native.mask_secrets(text, mask_char, visible_chars)

    # Pure Python fallback
    result = text
    for _, pattern, _ in _SECRET_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            matched = m.group()
            if len(matched) > visible_chars * 2:
                mask_len = len(matched) - visible_chars * 2
                masked = matched[:visible_chars] + mask_char * mask_len + matched[-visible_chars:]
            else:
                masked = mask_char * len(matched)
            result = result.replace(matched, masked)
    return result


def list_secret_patterns() -> List[str]:
    """List available secret pattern names.

    Returns:
        List of pattern names
    """
    if _NATIVE_AVAILABLE:
        return _native.list_secret_patterns()

    # Pure Python fallback
    return [name for name, _, _ in _SECRET_PATTERNS]


def scan_secrets_summary(text: str) -> Dict[str, Any]:
    """Scan secrets and return summary.

    Args:
        text: Text to scan

    Returns:
        Summary dict with counts
    """
    if _NATIVE_AVAILABLE:
        return _native.scan_secrets_summary(text)

    # Pure Python fallback
    matches = scan_secrets(text)
    by_type: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}

    for m in matches:
        by_type[m.secret_type] = by_type.get(m.secret_type, 0) + 1
        by_severity[m.severity] = by_severity.get(m.severity, 0) + 1

    return {
        "total_matches": len(matches),
        "has_secrets": len(matches) > 0,
        "by_type": by_type,
        "by_severity": by_severity,
    }


# Re-export native class when available
if _NATIVE_AVAILABLE:
    SecretMatch = _native.SecretMatch
else:
    SecretMatch = SecretMatchFallback


# =============================================================================
# PATTERN MATCHING (Aho-Corasick)
# =============================================================================


class PatternMatchFallback:
    """Fallback class for pattern match results."""

    def __init__(self, pattern_idx: int, matched_text: str, start: int, end: int):
        self.pattern_idx = pattern_idx
        self.matched_text = matched_text
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"PatternMatch(pattern={self.pattern_idx}, text='{self.matched_text}', span=({self.start}, {self.end}))"


class PatternMatcherFallback:
    """Fallback pattern matcher using Python regex."""

    def __init__(self, patterns: List[str], case_insensitive: bool = True):
        self.patterns = patterns
        self.case_insensitive = case_insensitive
        flags = re.IGNORECASE if case_insensitive else 0
        self._compiled = [(i, re.compile(re.escape(p), flags)) for i, p in enumerate(patterns)]

    def find_all(self, text: str) -> List[PatternMatchFallback]:
        matches = []
        for idx, pattern in self._compiled:
            for m in pattern.finditer(text):
                matches.append(PatternMatchFallback(idx, m.group(), m.start(), m.end()))
        matches.sort(key=lambda x: x.start)
        return matches

    def contains_any(self, text: str) -> bool:
        for _, pattern in self._compiled:
            if pattern.search(text):
                return True
        return False

    def count_matches(self, text: str) -> int:
        return len(self.find_all(text))

    def matched_patterns(self, text: str) -> List[int]:
        seen = set()
        for m in self.find_all(text):
            seen.add(m.pattern_idx)
        return sorted(seen)

    def matched_pattern_strings(self, text: str) -> List[str]:
        return [self.patterns[i] for i in self.matched_patterns(text)]

    def count_by_pattern(self, text: str) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for m in self.find_all(text):
            counts[m.pattern_idx] = counts.get(m.pattern_idx, 0) + 1
        return counts

    def get_pattern(self, idx: int) -> Optional[str]:
        return self.patterns[idx] if 0 <= idx < len(self.patterns) else None

    def pattern_count(self) -> int:
        return len(self.patterns)

    def replace_all(self, text: str, replacement: str) -> str:
        result = text
        for _, pattern in self._compiled:
            result = pattern.sub(replacement, result)
        return result


def contains_any_pattern(text: str, patterns: List[str], case_insensitive: bool = True) -> bool:
    """Check if text contains any pattern.

    Args:
        text: Text to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        True if any pattern matches
    """
    if _NATIVE_AVAILABLE:
        return _native.contains_any_pattern(text, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return matcher.contains_any(text)


def find_all_patterns(text: str, patterns: List[str], case_insensitive: bool = True) -> List[Any]:
    """Find all pattern matches in text.

    Args:
        text: Text to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        List of PatternMatch objects
    """
    if _NATIVE_AVAILABLE:
        return _native.find_all_patterns(text, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return matcher.find_all(text)


def count_pattern_matches(text: str, patterns: List[str], case_insensitive: bool = True) -> int:
    """Count pattern matches in text.

    Args:
        text: Text to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        Total match count
    """
    if _NATIVE_AVAILABLE:
        return _native.count_pattern_matches(text, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return matcher.count_matches(text)


def get_matched_pattern_indices(
    text: str, patterns: List[str], case_insensitive: bool = True
) -> List[int]:
    """Get indices of matched patterns.

    Args:
        text: Text to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        List of matched pattern indices
    """
    if _NATIVE_AVAILABLE:
        return _native.get_matched_pattern_indices(text, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return matcher.matched_patterns(text)


def batch_contains_any(
    texts: List[str], patterns: List[str], case_insensitive: bool = True
) -> List[bool]:
    """Check multiple texts for pattern matches.

    Args:
        texts: List of texts to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        List of booleans, one per text
    """
    if _NATIVE_AVAILABLE:
        return _native.batch_contains_any(texts, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return [matcher.contains_any(text) for text in texts]


def weighted_pattern_score(
    text: str, patterns: List[str], weights: List[float], case_insensitive: bool = True
) -> float:
    """Calculate weighted score for pattern matches.

    Args:
        text: Text to search
        patterns: Patterns to match
        weights: Weight for each pattern
        case_insensitive: Whether to ignore case

    Returns:
        Sum of weights for matched patterns
    """
    if _NATIVE_AVAILABLE:
        return _native.weighted_pattern_score(text, patterns, weights, case_insensitive)

    if len(patterns) != len(weights):
        raise ValueError(
            f"Pattern count ({len(patterns)}) must match weight count ({len(weights)})"
        )

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    matched = matcher.matched_patterns(text)
    return sum(weights[i] for i in matched)


# Re-export native classes when available
if _NATIVE_AVAILABLE:
    PatternMatcher = _native.PatternMatcher
    PatternMatch = _native.PatternMatch
else:
    PatternMatcher = PatternMatcherFallback
    PatternMatch = PatternMatchFallback
