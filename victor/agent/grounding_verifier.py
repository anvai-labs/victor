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

"""Grounding verification for LLM response validation.

This module implements verification patterns to detect and prevent
hallucination in LLM responses, especially for code-related queries.

Design Pattern: Strategy + Chain of Responsibility
=================================================
Multiple verification strategies are applied in sequence, each
contributing to a confidence score. If confidence drops below
threshold, the response is flagged for review or rejection.

Key Verifiers:
- FileExistenceVerifier: Checks that referenced files actually exist
- CodeSnippetVerifier: Validates code snippets against actual file content
- SymbolVerifier: Confirms referenced functions/classes exist
- PathVerifier: Validates file paths mentioned in response

Usage:
    verifier = GroundingVerifier(project_root="/path/to/project")
    result = await verifier.verify(response, context)

    if not result.is_grounded:
        # Response contains potential hallucinations
        for issue in result.issues:
            print(f"  - {issue.description}")
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.protocols.provider_adapter import IProviderAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# GROUNDING STOPWORDS
# =============================================================================
# Common English words that should NOT be treated as code symbols during
# grounding verification. These words frequently appear in LLM explanations
# but are not actual code identifiers.
#
# Issue Reference: GAP-15, workflow-test-issues.md Issue #3
# =============================================================================

GROUNDING_STOPWORDS: frozenset = frozenset(
    {
        # Articles and determiners
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "some",
        "any",
        "each",
        "every",
        "all",
        "both",
        "few",
        "many",
        "much",
        "other",
        "another",
        # Pronouns
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "he",
        "him",
        "his",
        "she",
        "her",
        "who",
        "which",
        "what",
        # Common verbs (often in descriptions)
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "has",
        "have",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "get",
        "gets",
        "got",
        "getting",
        "make",
        "makes",
        "made",
        "making",
        "take",
        "takes",
        "took",
        "taking",
        "taken",
        "let",
        "lets",
        "letting",
        "put",
        "puts",
        "putting",
        "see",
        "sees",
        "saw",
        "seeing",
        "seen",
        "know",
        "knows",
        "knew",
        "knowing",
        "known",
        "want",
        "wants",
        "wanted",
        "wanting",
        "need",
        "needs",
        "needed",
        "needing",
        "use",
        "uses",
        "used",
        "using",
        "find",
        "finds",
        "found",
        "finding",
        "give",
        "gives",
        "gave",
        "giving",
        "given",
        "tell",
        "tells",
        "told",
        "telling",
        "call",
        "calls",
        "called",
        "calling",
        "try",
        "tries",
        "tried",
        "trying",
        "ask",
        "asks",
        "asked",
        "asking",
        "work",
        "works",
        "worked",
        "working",
        "seem",
        "seems",
        "seemed",
        "seeming",
        "feel",
        "feels",
        "felt",
        "feeling",
        "leave",
        "leaves",
        "left",
        "leaving",
        "keep",
        "keeps",
        "kept",
        "keeping",
        "begin",
        "begins",
        "began",
        "beginning",
        "begun",
        "show",
        "shows",
        "showed",
        "showing",
        "shown",
        # Common explanation nouns that are often capitalized or backticked in
        # prose but are not code symbols.
        "analysis",
        "codebase",
        "detail",
        "details",
        "evidence",
        "findings",
        "information",
        "results",
        "summary",
        "hear",
        "hears",
        "heard",
        "hearing",
        "run",
        "runs",
        "ran",
        "running",
        "move",
        "moves",
        "moved",
        "moving",
        "live",
        "lives",
        "lived",
        "living",
        "believe",
        "believes",
        "believed",
        "believing",
        "bring",
        "brings",
        "brought",
        "bringing",
        "happen",
        "happens",
        "happened",
        "happening",
        "write",
        "writes",
        "wrote",
        "writing",
        "written",
        "provide",
        "provides",
        "provided",
        "providing",
        "sit",
        "sits",
        "sat",
        "sitting",
        "stand",
        "stands",
        "stood",
        "standing",
        "lose",
        "loses",
        "lost",
        "losing",
        "pay",
        "pays",
        "paid",
        "paying",
        "meet",
        "meets",
        "met",
        "meeting",
        "include",
        "includes",
        "included",
        "including",
        "continue",
        "continues",
        "continued",
        "continuing",
        "set",
        "sets",
        "setting",
        "learn",
        "learns",
        "learned",
        "learning",
        "change",
        "changes",
        "changed",
        "changing",
        "lead",
        "leads",
        "led",
        "leading",
        "understand",
        "understands",
        "understood",
        "understanding",
        "watch",
        "watches",
        "watched",
        "watching",
        "follow",
        "follows",
        "followed",
        "following",
        "stop",
        "stops",
        "stopped",
        "stopping",
        "create",
        "creates",
        "created",
        "creating",
        "speak",
        "speaks",
        "spoke",
        "speaking",
        "spoken",
        "read",
        "reads",
        "reading",
        "allow",
        "allows",
        "allowed",
        "allowing",
        "add",
        "adds",
        "added",
        "adding",
        "spend",
        "spends",
        "spent",
        "spending",
        "grow",
        "grows",
        "grew",
        "growing",
        "grown",
        "open",
        "opens",
        "opened",
        "opening",
        "walk",
        "walks",
        "walked",
        "walking",
        "win",
        "wins",
        "won",
        "winning",
        "offer",
        "offers",
        "offered",
        "offering",
        "remember",
        "remembers",
        "remembered",
        "remembering",
        "consider",
        "considers",
        "considered",
        "considering",
        "appear",
        "appears",
        "appeared",
        "appearing",
        "buy",
        "buys",
        "bought",
        "buying",
        "wait",
        "waits",
        "waited",
        "waiting",
        "serve",
        "serves",
        "served",
        "serving",
        "die",
        "dies",
        "died",
        "dying",
        "send",
        "sends",
        "sent",
        "sending",
        "expect",
        "expects",
        "expected",
        "expecting",
        "build",
        "builds",
        "built",
        "building",
        "stay",
        "stays",
        "stayed",
        "staying",
        "fall",
        "falls",
        "fell",
        "falling",
        "fallen",
        "cut",
        "cuts",
        "cutting",
        "reach",
        "reaches",
        "reached",
        "reaching",
        "kill",
        "kills",
        "killed",
        "killing",
        "remain",
        "remains",
        "remained",
        "remaining",
        # Action words commonly used in tech documentation
        "handles",
        "manages",
        "supports",
        "enables",
        "returns",
        "contains",
        "removes",
        "updates",
        "deletes",
        "modifies",
        "processes",
        "executes",
        "implements",
        "defines",
        "declares",
        "initializes",
        "configures",
        "validates",
        "parses",
        "formats",
        "converts",
        "transforms",
        "maps",
        "filters",
        "reduces",
        "sorts",
        # Prepositions
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "against",
        "among",
        "throughout",
        "around",
        "within",
        "without",
        "toward",
        "towards",
        "upon",
        # Conjunctions
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "while",
        "although",
        "because",
        "unless",
        "since",
        "whether",
        "however",
        "therefore",
        "thus",
        "hence",
        "moreover",
        "furthermore",
        "nevertheless",
        # Common words in tech contexts
        "here",
        "there",
        "now",
        "also",
        "just",
        "only",
        "very",
        "more",
        "most",
        "so",
        "than",
        "too",
        "even",
        "still",
        "already",
        "yet",
        "again",
        "always",
        "never",
        "often",
        "sometimes",
        "usually",
        "really",
        "quite",
        "rather",
        "almost",
        "well",
        "back",
        "up",
        "down",
        "out",
        "away",
        "off",
        "first",
        "last",
        "next",
        "new",
        "old",
        "same",
        "different",
        "good",
        "bad",
        "great",
        "little",
        "big",
        "small",
        "long",
        "short",
        "high",
        "low",
        "right",
        "wrong",
        "able",
        "available",
        # Numbers and ordinals
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "second",
        "third",
        "fourth",
        "fifth",
        # Question words
        "how",
        "why",
        "whom",
        "whose",
        # Negations
        "no",
        "not",
        "none",
        "nothing",
        "nobody",
        "nowhere",
        "neither",
        "nor",
        # Tech explanation phrases (common in LLM output)
        "file",
        "code",
        "function",
        "method",
        "class",
        "module",
        "package",
        "variable",
        "value",
        "data",
        "type",
        "object",
        "instance",
        "parameter",
        "argument",
        "result",
        "output",
        "input",
        "error",
        "exception",
        "message",
        "name",
        "path",
        "directory",
        "folder",
        "line",
        "number",
        "string",
        "list",
        "dict",
        "array",
        "example",
        "case",
        "issue",
        "problem",
        "solution",
        "step",
        "process",
        "system",
        "service",
        "request",
        "response",
        "user",
        "default",
        "option",
        "config",
        "configuration",
    }
)


class IssueType(Enum):
    """Types of grounding issues."""

    FILE_NOT_FOUND = "file_not_found"
    SYMBOL_NOT_FOUND = "symbol_not_found"
    CODE_MISMATCH = "code_mismatch"
    PATH_INVALID = "path_invalid"
    FABRICATED_CONTENT = "fabricated_content"
    UNVERIFIABLE = "unverifiable"


class IssueSeverity(Enum):
    """Severity of grounding issues."""

    LOW = "low"  # Minor discrepancy, safe to proceed
    MEDIUM = "medium"  # Noticeable issue, user should verify
    HIGH = "high"  # Significant hallucination detected
    CRITICAL = "critical"  # Response is unreliable


@dataclass
class GroundingIssue:
    """A single grounding issue detected in the response.

    Attributes:
        issue_type: Type of issue
        severity: Severity level
        description: Human-readable description
        reference: What was referenced (file path, symbol, etc.)
        context: Additional context about the issue
        suggestion: Suggested correction or action
    """

    issue_type: IssueType
    severity: IssueSeverity
    description: str
    reference: str
    context: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class GroundingVerificationResult:
    """Result of agent-level grounding verification.

    Renamed from VerificationResult to be semantically distinct:
    - GroundingVerificationResult (here): Agent grounding verification with issues/references
    - ClaimVerificationResult (victor.protocols.grounding): Protocol-level claim verification

    Attributes:
        is_grounded: Whether response is sufficiently grounded
        confidence: Confidence score (0.0 to 1.0)
        issues: List of detected issues
        verified_references: References that were successfully verified
        unverified_references: References that couldn't be verified
        metadata: Additional verification metadata
    """

    is_grounded: bool
    confidence: float
    issues: List[GroundingIssue] = field(default_factory=list)
    verified_references: List[str] = field(default_factory=list)
    unverified_references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: GroundingIssue) -> None:
        """Add an issue and update confidence."""
        self.issues.append(issue)
        # Reduce confidence based on severity
        penalty = {
            IssueSeverity.LOW: 0.05,
            IssueSeverity.MEDIUM: 0.15,
            IssueSeverity.HIGH: 0.30,
            IssueSeverity.CRITICAL: 0.50,
        }
        self.confidence = max(0.0, self.confidence - penalty.get(issue.severity, 0.1))

    def generate_feedback_prompt(self, max_issues: int = 3) -> str:
        """Generate actionable feedback prompt from grounding issues.

        Creates a concise prompt that guides the model to correct specific
        grounding issues in its response, using suggestions when available.

        Args:
            max_issues: Maximum number of issues to include (default 3)

        Returns:
            Feedback prompt string, or empty string if no issues
        """
        if not self.issues:
            return ""

        # Sort by severity (critical first)
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
        }
        sorted_issues = sorted(self.issues, key=lambda i: severity_order.get(i.severity, 4))[
            :max_issues
        ]

        # Build feedback sections by issue type for clarity
        feedback_parts = []

        # Group corrections by type
        file_issues = [i for i in sorted_issues if i.issue_type == IssueType.FILE_NOT_FOUND]
        symbol_issues = [i for i in sorted_issues if i.issue_type == IssueType.SYMBOL_NOT_FOUND]
        code_issues = [i for i in sorted_issues if i.issue_type == IssueType.CODE_MISMATCH]
        fabricated = [i for i in sorted_issues if i.issue_type == IssueType.FABRICATED_CONTENT]
        other_issues = [
            i
            for i in sorted_issues
            if i.issue_type
            not in (
                IssueType.FILE_NOT_FOUND,
                IssueType.SYMBOL_NOT_FOUND,
                IssueType.CODE_MISMATCH,
                IssueType.FABRICATED_CONTENT,
            )
        ]

        if file_issues:
            files = [f"'{i.reference}'" for i in file_issues]
            feedback_parts.append(
                f"- File(s) not found: {', '.join(files)}. "
                "Use read or ls to verify paths before referencing."
            )

        if symbol_issues:
            symbols = [f"'{i.reference}'" for i in symbol_issues]
            feedback_parts.append(
                f"- Symbol(s) not found: {', '.join(symbols)}. "
                "Use code_search to find actual function/class names in the codebase."
            )

        if code_issues:
            refs = [i.reference for i in code_issues][:2]
            feedback_parts.append(
                f"- Code mismatch in: {', '.join(refs)}. "
                "Quote code exactly as it appears in tool output."
            )

        if fabricated:
            refs = [i.reference for i in fabricated][:2]
            feedback_parts.append(
                f"- Fabricated content detected: {', '.join(refs)}. "
                "Only reference content from actual tool output."
            )

        # Add specific suggestions if available
        for issue in sorted_issues:
            if issue.suggestion and issue.suggestion not in str(feedback_parts):
                feedback_parts.append(f"- {issue.suggestion}")

        # Add any other issues
        for issue in other_issues:
            if issue.issue_type not in (
                IssueType.FILE_NOT_FOUND,
                IssueType.SYMBOL_NOT_FOUND,
                IssueType.CODE_MISMATCH,
                IssueType.FABRICATED_CONTENT,
            ):
                feedback_parts.append(f"- {issue.description}")

        if not feedback_parts:
            return ""

        header = (
            "GROUNDING CORRECTION REQUIRED: Your previous response contained unverified references. "
            "Please correct the following issues:\n"
        )
        return header + "\n".join(feedback_parts)


# Backward compatibility alias
VerificationResult = GroundingVerificationResult


@dataclass
class VerifierConfig:
    """Configuration for grounding verification.

    Attributes:
        min_confidence: Minimum confidence to consider grounded
        verify_file_paths: Check file path existence
        verify_code_snippets: Check code snippet accuracy
        verify_symbols: Check function/class existence
        max_files_to_check: Limit file checks for performance
        ignore_patterns: Path patterns to ignore
        strict_mode: Fail on any issue (not just severe ones)
        skip_generated_code: Skip verification for code that appears to be newly generated
        generated_code_patterns: Patterns indicating generated code (test files, etc.)
    """

    min_confidence: float = 0.7
    verify_file_paths: bool = True
    verify_code_snippets: bool = True
    verify_symbols: bool = True
    max_files_to_check: int = 20
    ignore_patterns: List[str] = field(
        default_factory=lambda: [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
        ]
    )
    strict_mode: bool = False
    skip_generated_code: bool = True  # Skip verification for generated code
    generated_code_patterns: List[str] = field(
        default_factory=lambda: [
            r"test_\w+\.py",  # Test files being created
            r"tests?/",  # Test directories
            r"conftest\.py",  # Pytest fixtures
            r"_test\.py$",  # Alternative test naming
            r"spec\.py$",  # Spec files
        ]
    )


class GroundingVerifier:
    """Verifies LLM responses against actual project content.

    Detects hallucinations by checking referenced files, code snippets,
    and symbols against the actual codebase.
    """

    # Patterns for extracting references from responses
    FILE_PATH_PATTERN = re.compile(
        r"(?:^|\s|`|\")"  # Preceding context
        r"((?:[\w./-]+/)?[\w.-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|c|cpp|h|hpp|md|yaml|yml|json|toml))"  # Path
        r"(?:\s|`|\"|:|,|$)",  # Following context
        re.MULTILINE,
    )

    CODE_BLOCK_PATTERN = re.compile(
        r"```(?:\w+)?\n(.*?)```",
        re.DOTALL,
    )

    # Pattern for Python function/class definitions
    # Only match actual code definitions, not natural language phrases like "let me"
    # Require either code block context or CamelCase/snake_case naming conventions
    SYMBOL_PATTERN = re.compile(
        r"(?:class|def|function|const|var|type|interface)\s+([A-Z_][A-Za-z0-9_]*|[a-z][a-z0-9_]*[A-Z_][A-Za-z0-9_]*|[a-z_][a-z0-9_]+)",
        re.MULTILINE,
    )

    # Pattern for line references like "file.py:123" or "lines 10-20"
    LINE_REF_PATTERN = re.compile(
        r"([\w./-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|c|cpp|h|hpp|md|yaml|yml|json|toml))(?::(\d+)(?:-(\d+))?|,?\s*(?:line|lines?)\s*(\d+)(?:\s*-\s*(\d+))?)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        project_root: Optional[str] = None,
        config: Optional[VerifierConfig] = None,
        provider_adapter: Optional["IProviderAdapter"] = None,
        grounding_threshold_learner: Optional[Any] = None,
    ):
        """Initialize the grounding verifier.

        Args:
            project_root: Root directory of the project to verify against
            config: Verification configuration
            provider_adapter: Provider adapter for provider-specific grounding settings
            grounding_threshold_learner: Optional GroundingThresholdLearner for RL-based thresholds
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Use provider-specific grounding settings if available
        if provider_adapter and config is None:
            caps = provider_adapter.capabilities
            config = VerifierConfig(
                min_confidence=caps.grounding_strictness,
                strict_mode=caps.grounding_required,
            )
            logger.debug(
                f"GroundingVerifier using provider-specific settings: "
                f"min_confidence={caps.grounding_strictness}, "
                f"strict_mode={caps.grounding_required}"
            )

        self.config = config or VerifierConfig()
        self._file_cache: Dict[str, str] = {}
        self._existing_files: Optional[Set[str]] = None
        self._provider_adapter = provider_adapter
        self._grounding_threshold_learner = grounding_threshold_learner

        if grounding_threshold_learner:
            logger.info("RL: GroundingVerifier using unified GroundingThresholdLearner")

        logger.debug(f"GroundingVerifier initialized with project_root={self.project_root}")

    def _get_rl_threshold(self, provider: str, response_type: str) -> Optional[float]:
        """Get RL-recommended threshold for given context.

        Args:
            provider: Provider name
            response_type: Type of response (code_generation, explanation, etc.)

        Returns:
            Recommended threshold or None if learner unavailable
        """
        if not self._grounding_threshold_learner:
            return None

        try:
            rec = self._grounding_threshold_learner.get_recommendation(
                provider=provider,
                model="",
                task_type=response_type,
            )
            if rec and rec.confidence > 0.4:  # Only use if reasonably confident
                logger.debug(
                    f"RL: Using learned threshold {rec.value:.2f} for {provider}/{response_type} "
                    f"(confidence={rec.confidence:.2f})"
                )
                return float(rec.value) if isinstance(rec.value, (int, float)) else None
        except Exception as e:
            logger.debug(f"RL: Could not get threshold recommendation: {e}")

        return None

    def _record_verification_outcome(
        self,
        provider: str,
        model: str,
        response_type: str,
        threshold_used: float,
        actual_hallucination: bool,
        detected_hallucination: bool,
    ) -> None:
        """Record verification outcome to RL learner.

        Args:
            provider: Provider name
            model: Model name
            response_type: Type of response
            threshold_used: Threshold that was used
            actual_hallucination: Whether there was actually a hallucination
            detected_hallucination: Whether hallucination was detected
        """
        if not self._grounding_threshold_learner:
            return

        try:
            from victor.framework.rl.base import RLOutcome

            outcome = RLOutcome(
                provider=provider,
                model=model or "unknown",
                task_type=response_type,
                success=not actual_hallucination,  # Success if no actual hallucination
                quality_score=1.0 if not actual_hallucination else 0.0,
                metadata={
                    "response_type": response_type,
                    "threshold_used": threshold_used,
                    "actual_hallucination": actual_hallucination,
                    "detected_hallucination": detected_hallucination,
                },
            )
            self._grounding_threshold_learner.record_outcome(outcome)
            logger.debug(
                f"RL: Recorded grounding outcome for {provider}/{model}: "
                f"actual={actual_hallucination}, detected={detected_hallucination}"
            )
        except Exception as e:
            logger.debug(f"RL: Failed to record grounding outcome: {e}")

    def _get_existing_files(self) -> Set[str]:
        """Get set of existing files in project (cached)."""
        if self._existing_files is None:
            self._existing_files = set()
            try:
                for root, dirs, files in os.walk(self.project_root):
                    # Filter out ignored directories
                    dirs[:] = [
                        d
                        for d in dirs
                        if d not in self.config.ignore_patterns and not d.startswith(".")
                    ]

                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), self.project_root)
                        self._existing_files.add(rel_path)
                logger.debug(
                    f"GroundingVerifier scanned {len(self._existing_files)} files "
                    f"under {self.project_root}"
                )
            except Exception as e:
                logger.warning(f"Error scanning project files: {e}")
        return self._existing_files

    def _read_file_cached(self, path: str) -> Optional[str]:
        """Read file content with caching."""
        if path in self._file_cache:
            return self._file_cache[path]

        full_path = self.project_root / path
        if not full_path.exists():
            return None

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            self._file_cache[path] = content
            return content
        except Exception as e:
            logger.debug(f"Error reading {path}: {e}")
            return None

    def extract_file_references(self, response: str) -> List[str]:
        """Extract file path references from response.

        Args:
            response: LLM response text

        Returns:
            List of file paths mentioned in response
        """
        # Find explicit file paths
        matches = self.FILE_PATH_PATTERN.findall(response)

        # Also extract from line references
        line_refs = self.LINE_REF_PATTERN.findall(response)
        matches.extend([ref[0] for ref in line_refs])

        # Normalize and dedupe
        paths = set()
        for match in matches:
            # Clean up the path
            path = match.strip("`\"'")
            # Skip URLs
            if path.startswith("http"):
                continue
            # Skip obvious non-paths
            if path.startswith(".") and "/" not in path:
                continue
            paths.add(path)

        return list(paths)

    def extract_line_references(self, response: str) -> List[Dict[str, Any]]:
        """Extract file line references from response."""
        refs: List[Dict[str, Any]] = []
        for match in self.LINE_REF_PATTERN.findall(response):
            path, colon_start, colon_end, word_start, word_end = match
            start_text = colon_start or word_start
            end_text = colon_end or word_end or start_text
            try:
                start_line = int(start_text)
                end_line = int(end_text)
            except (TypeError, ValueError):
                continue
            refs.append(
                {
                    "path": path.strip("`\"'"),
                    "start_line": min(start_line, end_line),
                    "end_line": max(start_line, end_line),
                }
            )
        return refs

    def extract_code_snippets(self, response: str) -> List[Dict[str, Any]]:
        """Extract code snippets from response.

        Args:
            response: LLM response text

        Returns:
            List of code snippets with metadata
        """
        snippets = []
        for match in self.CODE_BLOCK_PATTERN.finditer(response):
            code = match.group(1).strip()
            if len(code) > 10:  # Skip tiny snippets
                snippets.append(
                    {
                        "code": code,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                    }
                )
        return snippets

    def extract_symbols(self, response: str) -> List[str]:
        """Extract symbol references (functions, classes) from response.

        Args:
            response: LLM response text

        Returns:
            List of symbol names
        """
        matches = self.SYMBOL_PATTERN.findall(response)
        return list(set(matches))

    async def verify_file_paths(self, paths: List[str], result: VerificationResult) -> None:
        """Verify file path references exist.

        Args:
            paths: List of file paths to verify
            result: Verification result to update
        """
        existing_files = self._get_existing_files()

        for path in paths[: self.config.max_files_to_check]:
            # Check exact match
            if path in existing_files:
                result.verified_references.append(path)
                continue

            # Check with project root prefix stripped
            clean_path = path.lstrip("./")
            if clean_path in existing_files:
                result.verified_references.append(path)
                continue

            # Check partial match (filename only or path suffix)
            filename = os.path.basename(path)
            # Match files ending with the exact filename (separated by /)
            partial_matches = [
                f
                for f in existing_files
                if f.endswith(filename) and (f == filename or f.endswith("/" + filename))
            ]

            if partial_matches:
                # File exists - if it's just a filename without path, count as verified
                # (model may have abbreviated the path which is fine)
                if "/" not in path:
                    # Just a filename - count as verified
                    result.verified_references.append(path)
                elif len(partial_matches) == 1 and partial_matches[0] == clean_path:
                    # Single match and path is correct - count as verified
                    result.verified_references.append(path)
                elif len(partial_matches) == 1 and partial_matches[0].endswith("/" + clean_path):
                    # Single match and given path is a valid suffix of actual path
                    # (model used relative path from subdirectory which is fine)
                    result.verified_references.append(path)
                elif len(partial_matches) == 1:
                    # Single match but wrong directory path - flag it but low severity
                    result.add_issue(
                        GroundingIssue(
                            issue_type=IssueType.PATH_INVALID,
                            severity=IssueSeverity.LOW,
                            description=f"Path '{path}' has incorrect directory",
                            reference=path,
                            suggestion=f"Correct path is: {partial_matches[0]}",
                        )
                    )
                    result.unverified_references.append(path)
                else:
                    # Multiple matches and a partial path was given - ambiguous
                    result.add_issue(
                        GroundingIssue(
                            issue_type=IssueType.PATH_INVALID,
                            severity=IssueSeverity.LOW,
                            description=f"Path '{path}' is ambiguous",
                            reference=path,
                            suggestion=f"Could be: {', '.join(partial_matches[:3])}",
                        )
                    )
                    result.unverified_references.append(path)
            else:
                # File doesn't exist at all
                result.add_issue(
                    GroundingIssue(
                        issue_type=IssueType.FILE_NOT_FOUND,
                        severity=IssueSeverity.HIGH,
                        description=f"File '{path}' does not exist in project",
                        reference=path,
                        suggestion="This may be a hallucinated file path",
                    )
                )
                result.unverified_references.append(path)

    def _resolve_existing_file_reference(self, path: str) -> Optional[str]:
        """Resolve a referenced path to a single existing project-relative file path."""
        existing_files = self._get_existing_files()
        clean_path = path.lstrip("./")
        if path in existing_files:
            return path
        if clean_path in existing_files:
            return clean_path

        filename = os.path.basename(path)
        partial_matches = [
            f
            for f in existing_files
            if f.endswith(filename) and (f == filename or f.endswith("/" + filename))
        ]
        if "/" not in path and len(partial_matches) == 1:
            return partial_matches[0]
        if len(partial_matches) == 1 and partial_matches[0].endswith("/" + clean_path):
            return partial_matches[0]
        return None

    async def verify_line_references(
        self,
        line_refs: List[Dict[str, Any]],
        result: VerificationResult,
    ) -> None:
        """Verify cited line numbers exist in the referenced files."""
        for ref in line_refs[: self.config.max_files_to_check]:
            path = ref["path"]
            resolved_path = self._resolve_existing_file_reference(path)
            if not resolved_path:
                continue

            content = self._read_file_cached(resolved_path)
            if content is None:
                continue

            line_count = len(content.splitlines())
            start_line = ref["start_line"]
            end_line = ref["end_line"]
            reference = (
                f"{path}:{start_line}"
                if start_line == end_line
                else f"{path}:{start_line}-{end_line}"
            )

            if start_line < 1 or end_line > line_count:
                result.add_issue(
                    GroundingIssue(
                        issue_type=IssueType.UNVERIFIABLE,
                        severity=IssueSeverity.MEDIUM,
                        description=(
                            f"Line reference '{reference}' is outside file bounds "
                            f"(file has {line_count} lines)"
                        ),
                        reference=reference,
                        suggestion=(
                            "Re-read the file with line-numbered output before citing exact lines."
                        ),
                    )
                )
                result.unverified_references.append(reference)
            else:
                result.verified_references.append(reference)

    def _is_generated_code_context(self, context: Optional[Dict[str, Any]]) -> bool:
        """Check if the context indicates code generation task.

        Args:
            context: Verification context

        Returns:
            True if this appears to be a code generation task
        """
        if not context:
            return False

        # Check for explicit code generation indicators
        task_type = context.get("task_type", "").lower()
        creation_task_types = (
            "code_generation",
            "create",
            "create_simple",
            "test",
            "testing",
            "implementation",
            "implement",
            "write",
            "add",
            "build",
        )
        if task_type in creation_task_types:
            return True

        # Check for creation-intent keywords in query
        query = context.get("query", "").lower()
        creation_keywords = [
            # Test creation
            "create test",
            "write test",
            "pytest",
            "test suite",
            "generate test",
            "add test",
            # Code creation
            "create a",
            "create the",
            "write a",
            "write the",
            "implement",
            "add a",
            "add the",
            "build a",
            "build the",
            "generate",
            "make a",
            "make the",
            # File operations
            "new file",
            "new function",
            "new class",
            "new module",
        ]
        if any(kw in query for kw in creation_keywords):
            return True

        # Check for creation intent in is_action_task flag
        if context.get("is_action_task", False):
            # Action tasks are often creation tasks
            return True

        return False

    def _has_creation_intent_in_response(self, response: str) -> bool:
        """Detect creation intent from response content.

        Looks for patterns indicating the model is creating new code,
        not referencing existing code.

        Args:
            response: Model response text

        Returns:
            True if response indicates creation intent
        """
        response_lower = response.lower()

        # Creation intent patterns
        creation_patterns = [
            r"(?:i'll|let me|i will|i can)\s+(?:create|write|implement|add|build)",
            r"here(?:'s| is) (?:the|a) (?:new|updated|complete)",
            r"creating (?:a |the )?(?:new |)(?:file|function|class|module)",
            r"adding (?:a |the )?(?:new |)(?:file|function|class|module)",
            r"implementing",
            r"writing (?:a |the )?(?:new |)",
            r"here(?:'s| is) (?:the |)(?:implementation|code)",
        ]

        for pattern in creation_patterns:
            if re.search(pattern, response_lower):
                return True

        return False

    def _is_generated_code_path(self, path: str) -> bool:
        """Check if a file path matches generated code patterns.

        Args:
            path: File path to check

        Returns:
            True if path matches a generated code pattern
        """
        for pattern in self.config.generated_code_patterns:
            if re.search(pattern, path):
                return True
        return False

    async def verify_code_snippets(
        self,
        snippets: List[Dict[str, Any]],
        file_paths: List[str],
        result: VerificationResult,
        is_code_generation: bool = False,
    ) -> None:
        """Verify code snippets match actual file content.

        Args:
            snippets: Code snippets to verify
            file_paths: Referenced file paths
            result: Verification result to update
            is_code_generation: True if this is a code generation task
        """
        for snippet in snippets:
            code = snippet["code"]

            # Skip verification for code generation tasks if configured
            if self.config.skip_generated_code and is_code_generation:
                logger.debug("[GroundingVerifier] Skipping verification for generated code snippet")
                result.metadata["skipped_generated_snippets"] = (
                    result.metadata.get("skipped_generated_snippets", 0) + 1
                )
                continue

            # Try to find this code in referenced files
            found = False
            for path in file_paths:
                content = self._read_file_cached(path)
                if content and self._code_matches(code, content):
                    found = True
                    result.verified_references.append(f"code_snippet@{path}")
                    break

            if not found:
                # Check if code looks like it's for a test file (generated)
                if self._looks_like_generated_test(code):
                    # Skip or use very low severity for test code
                    logger.debug(
                        "[GroundingVerifier] Code looks like generated test - skipping issue"
                    )
                    result.metadata["skipped_test_snippets"] = (
                        result.metadata.get("skipped_test_snippets", 0) + 1
                    )
                    continue

                # Check if it looks like fabricated content
                if self._looks_fabricated(code):
                    result.add_issue(
                        GroundingIssue(
                            issue_type=IssueType.FABRICATED_CONTENT,
                            severity=IssueSeverity.HIGH,
                            description="Code snippet may be fabricated",
                            reference=code[:100] + "...",
                            context="Code does not match any file in the project",
                        )
                    )
                else:
                    # Could be new code suggestion - lower severity
                    result.add_issue(
                        GroundingIssue(
                            issue_type=IssueType.UNVERIFIABLE,
                            severity=IssueSeverity.LOW,
                            description="Code snippet could not be verified",
                            reference=code[:100] + "...",
                            context="May be a new code suggestion",
                        )
                    )

    def _looks_like_generated_test(self, code: str) -> bool:
        """Check if code looks like a generated test file.

        Args:
            code: Code snippet to check

        Returns:
            True if code appears to be test code
        """
        # Test file indicators
        test_indicators = [
            r"^import pytest",
            r"^from pytest import",
            r"def test_\w+",
            r"class Test\w+",
            r"@pytest\.",
            r"@mock\.",
            r"@patch\(",
            r"assert\s+\w+",
            r"mock\.\w+",
            r"fixture",
        ]

        for pattern in test_indicators:
            if re.search(pattern, code, re.MULTILINE):
                return True

        return False

    def _code_matches(self, snippet: str, content: str) -> bool:
        """Check if a code snippet matches file content.

        Uses fuzzy matching to handle whitespace/formatting differences.

        Args:
            snippet: Code snippet to check
            content: File content to check against

        Returns:
            True if snippet appears to match content
        """
        # Normalize whitespace
        snippet_normalized = " ".join(snippet.split())
        content_normalized = " ".join(content.split())

        # Check for substantial substring match
        if snippet_normalized in content_normalized:
            return True

        # Check line-by-line with tolerance
        snippet_lines = [line.strip() for line in snippet.split("\n") if line.strip()]
        content_lines = [line.strip() for line in content.split("\n") if line.strip()]

        if not snippet_lines:
            return True  # Empty snippet matches anything

        # Find matching lines
        matches = 0
        for snippet_line in snippet_lines:
            if any(snippet_line in content_line for content_line in content_lines):
                matches += 1

        # Consider matched if >70% of lines match
        return matches / len(snippet_lines) >= 0.7

    def _looks_fabricated(self, code: str) -> bool:
        """Heuristic check if code looks fabricated.

        Args:
            code: Code snippet to check

        Returns:
            True if code has fabrication indicators
        """
        # Look for common fabrication indicators
        fabrication_patterns = [
            r"# TODO: implement",
            r"// TODO: implement",
            r"pass\s*$",  # Python pass statement at end
            r"\.\.\.",  # Ellipsis placeholder
            r"raise NotImplementedError",
            r"throw new Error\(['\"]Not implemented",
        ]

        for pattern in fabrication_patterns:
            if re.search(pattern, code):
                return True

        # Check for suspiciously generic names
        generic_names = [
            "example_function",
            "do_something",
            "process_data",
            "handle_request",
            "some_function",
            "my_function",
        ]
        code_lower = code.lower()
        for name in generic_names:
            if name in code_lower:
                return True

        return False

    async def verify_symbols(
        self,
        symbols: List[str],
        file_paths: List[str],
        result: VerificationResult,
    ) -> None:
        """Verify symbol references exist in codebase.

        Args:
            symbols: Symbol names to verify
            file_paths: Files to check for symbols
            result: Verification result to update
        """
        # Build symbol index from referenced files
        known_symbols: Set[str] = set()

        for path in file_paths:
            content = self._read_file_cached(path)
            if content:
                # Extract symbols from file
                file_symbols = self.SYMBOL_PATTERN.findall(content)
                known_symbols.update(file_symbols)

        # Verify each symbol, filtering out stopwords and common English words
        # that frequently appear in LLM explanations but are not code symbols
        for symbol in symbols:
            symbol_lower = symbol.lower()

            # Skip if in known symbols from files
            if symbol in known_symbols:
                result.verified_references.append(f"symbol:{symbol}")
                continue

            # GAP-15 FIX: Skip common English words, language keywords, and stopwords
            # These are NOT user-defined symbols and should not be flagged
            if symbol_lower in GROUNDING_STOPWORDS:
                continue

            # Skip very short symbols (likely not meaningful code identifiers)
            if len(symbol) <= 2:
                continue

            # Skip Python magic methods and dunder names
            if symbol.startswith("__") and symbol.endswith("__"):
                continue

            # Skip common built-in type names
            builtin_types = {
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "set",
                "tuple",
                "bytes",
                "None",
                "True",
                "False",
                "type",
            }
            if symbol in builtin_types:
                continue

            # If not filtered out, flag as potential issue
            result.add_issue(
                GroundingIssue(
                    issue_type=IssueType.SYMBOL_NOT_FOUND,
                    severity=IssueSeverity.MEDIUM,
                    description=f"Symbol '{symbol}' not found in referenced files",
                    reference=symbol,
                    suggestion="May be defined elsewhere or hallucinated",
                )
            )
            result.unverified_references.append(symbol)

    async def verify(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify a response for grounding.

        Args:
            response: LLM response to verify
            context: Optional context (e.g., which files were read)

        Returns:
            VerificationResult with confidence and issues
        """
        result = VerificationResult(
            is_grounded=True,
            confidence=1.0,
            metadata={"response_length": len(response)},
        )

        context = context or {}

        # Check if this is a code generation context (from context or response content)
        is_code_generation = self._is_generated_code_context(context)

        # Also check response for creation intent patterns
        if not is_code_generation:
            is_code_generation = self._has_creation_intent_in_response(response)
            if is_code_generation:
                logger.debug("[GroundingVerifier] Creation intent detected in response content")
                result.metadata["creation_intent_source"] = "response"

        result.metadata["is_code_generation"] = is_code_generation

        if is_code_generation and self.config.skip_generated_code:
            logger.debug(
                "[GroundingVerifier] Code generation context detected - using relaxed verification"
            )
            result.metadata["verification_mode"] = "relaxed"

        # Extract references
        file_paths = self.extract_file_references(response)
        line_refs = self.extract_line_references(response)
        code_snippets = self.extract_code_snippets(response)
        symbols = self.extract_symbols(response)

        result.metadata["file_refs"] = len(file_paths)
        result.metadata["line_refs"] = len(line_refs)
        result.metadata["code_snippets"] = len(code_snippets)
        result.metadata["symbols"] = len(symbols)

        # Add context files to check
        if "files_read" in context:
            file_paths.extend(context["files_read"])
        file_paths = list(set(file_paths))

        # Filter out generated code paths from verification
        if self.config.skip_generated_code:
            original_count = len(file_paths)
            file_paths = [p for p in file_paths if not self._is_generated_code_path(p)]
            if len(file_paths) < original_count:
                result.metadata["skipped_generated_paths"] = original_count - len(file_paths)

        # Run verifications
        if self.config.verify_file_paths and file_paths:
            await self.verify_file_paths(file_paths, result)

        if self.config.verify_file_paths and line_refs:
            await self.verify_line_references(line_refs, result)

        if self.config.verify_code_snippets and code_snippets:
            await self.verify_code_snippets(
                code_snippets, file_paths, result, is_code_generation=is_code_generation
            )

        if self.config.verify_symbols and symbols:
            # Skip symbol verification for code generation tasks
            if not is_code_generation:
                await self.verify_symbols(symbols, file_paths, result)
            else:
                result.metadata["skipped_symbol_verification"] = True
                logger.debug(
                    "[GroundingVerifier] Skipping symbol verification for code generation task"
                )

        # Get provider, model, and response type for RL
        provider = context.get("provider", "unknown")
        model = context.get("model", "unknown")
        response_type = (
            "code_generation" if is_code_generation else context.get("task_type", "general")
        )

        # Try RL-learned threshold, fall back to config
        threshold = self._get_rl_threshold(provider, response_type)
        if threshold is None:
            threshold = self.config.min_confidence
        result.metadata["threshold_used"] = threshold
        result.metadata["threshold_source"] = (
            "rl" if threshold != self.config.min_confidence else "config"
        )

        # Determine if grounded based on confidence
        result.is_grounded = result.confidence >= threshold

        # In strict mode, any issue fails
        if self.config.strict_mode and result.issues:
            result.is_grounded = False

        logger.debug(
            f"Grounding verification: confidence={result.confidence:.2f}, "
            f"threshold={threshold:.2f} ({result.metadata['threshold_source']}), "
            f"issues={len(result.issues)}, is_grounded={result.is_grounded}, "
            f"code_generation={is_code_generation}"
        )

        # Record outcome to RL learner for future threshold optimization
        # Note: actual_hallucination would need external feedback; for now, we estimate
        # based on issue severity (HIGH/CRITICAL issues suggest actual hallucination)
        actual_hallucination = any(
            issue.severity in (IssueSeverity.HIGH, IssueSeverity.CRITICAL)
            for issue in result.issues
        )
        self._record_verification_outcome(
            provider=provider,
            model=model,
            response_type=response_type,
            threshold_used=threshold,
            actual_hallucination=actual_hallucination,
            detected_hallucination=not result.is_grounded,
        )

        return result

    def clear_cache(self) -> None:
        """Clear file content cache."""
        self._file_cache.clear()
        self._existing_files = None
