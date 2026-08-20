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

"""Query complexity detection for routing to Bayesian orchestration."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set


class ComplexityLevel(str, Enum):
    """Query complexity levels."""

    SIMPLE = "simple"  # Fast path: single agent, simple majority
    MODERATE = "moderate"  # Middle path: limited Bayesian
    COMPLEX = "complex"  # Full Bayesian orchestration


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis."""

    level: ComplexityLevel
    confidence: float  # 0.0 to 1.0
    reasons: List[str]  # Human-readable explanations
    suggested_agents: Optional[int] = None  # Optimal number of agents
    needs_voi: bool = False  # Whether to use VoI-based selection


class QueryComplexityDetector:
    """Detects query complexity to determine orchestration strategy."""

    # Simple query patterns (fast path indicators)
    SIMPLE_PATTERNS = [
        r"^(what|how|where|when|who|why|is|are|do|does|can|could)\s+",  # Simple questions
        r"^(explain|describe|define|list)\s+",  # Simple explanations
        r"^(tell|show|give)\s+me\s+",  # Simple requests
    ]

    # Complex query patterns (Bayesian indicators)
    COMPLEX_PATTERNS = [
        r"(?i)(analyze|investigate|diagnose|debug|troubleshoot)",  # Deep analysis
        r"(?i)(compare|contrast|evaluate|assess)",  # Comparative analysis
        r"(?i)(design|architect|implement|optimize)",  # Complex tasks
        r"(?i)(multi|multiple|several|various)\s+(agent|model|approach)",  # Multi-agent
        r"(?i)(uncertain|unclear|ambiguous|conflicting)",  # Ambiguity
        r"(?i)(should|would|could|might)\s+i\s+",  # Decision support
        r"(?i)(best|better|optimal|recommend|suggest)",  # Recommendations
    ]

    # Tool requirement indicators
    TOOL_KEYWORDS = {
        "file": {"file", "read", "write", "edit", "code", "directory"},
        "git": {"git", "commit", "branch", "merge", "push", "pull"},
        "search": {"search", "find", "locate", "grep"},
        "test": {"test", "assert", "verify", "check"},
        "build": {"build", "compile", "make", "cmake"},
        "database": {"database", "query", "sql", "migration"},
        "deploy": {"deploy", "release", "ship", "production"},
    }

    # Ambiguity indicators
    AMBIGUITY_KEYWORDS = {
        "maybe",
        "perhaps",
        "possibly",
        "might",
        "could",
        "uncertain",
        "unclear",
        "depends",
        "context",
        "not sure",
        "don't know",
        "unsure",
    }

    def __init__(
        self,
        simple_threshold: float = 0.3,
        complex_threshold: float = 0.7,
        min_length_for_complex: int = 100,
    ):
        """Initialize complexity detector.

        Args:
            simple_threshold: Score below which queries are simple
            complex_threshold: Score above which queries are complex
            min_length_for_complex: Minimum character length for complex classification
        """
        self.simple_threshold = simple_threshold
        self.complex_threshold = complex_threshold
        self.min_length_for_complex = min_length_for_complex

    def analyze(self, query: str) -> ComplexityAnalysis:
        """Analyze query complexity.

        Args:
            query: User query string

        Returns:
            ComplexityAnalysis with level and reasons
        """
        reasons = []
        complexity_score = 0.0

        # 1. Query length factor (0-0.2)
        length_score = self._analyze_length(query)
        complexity_score += length_score * 0.2
        if length_score > 0.5:
            reasons.append(f"Query is long ({len(query)} chars)")

        # 2. Pattern matching factor (0-0.3)
        pattern_score = self._analyze_patterns(query)
        complexity_score += pattern_score * 0.3
        if pattern_score > 0.5:
            reasons.append("Contains complex patterns (analysis, comparison, decision)")

        # 3. Tool requirements factor (0-0.2)
        tool_score, required_tools = self._analyze_tools(query)
        complexity_score += tool_score * 0.2
        if tool_score > 0.5:
            reasons.append(f"Requires multiple tools: {', '.join(required_tools)}")

        # 4. Ambiguity factor (0-0.15)
        ambiguity_score = self._analyze_ambiguity(query)
        complexity_score += ambiguity_score * 0.15
        if ambiguity_score > 0.5:
            reasons.append("Contains ambiguous language")

        # 5. Domain specificity factor (0-0.15)
        domain_score = self._analyze_domain(query)
        complexity_score += domain_score * 0.15
        if domain_score > 0.5:
            reasons.append("Domain-specific terminology detected")

        # Determine complexity level
        if complexity_score >= self.complex_threshold:
            level = ComplexityLevel.COMPLEX
            suggested_agents = 3  # Multiple agents for complex
            needs_voi = True
        elif complexity_score >= self.simple_threshold:
            level = ComplexityLevel.MODERATE
            suggested_agents = 2  # Two agents for moderate
            needs_voi = False
        else:
            level = ComplexityLevel.SIMPLE
            suggested_agents = 1  # Single agent for simple
            needs_voi = False

        # Confidence is how far we are from thresholds
        if level == ComplexityLevel.COMPLEX:
            confidence = (complexity_score - self.complex_threshold) / (
                1.0 - self.complex_threshold
            )
        elif level == ComplexityLevel.SIMPLE:
            confidence = 1.0 - (complexity_score / self.simple_threshold)
        else:  # MODERATE
            distance_from_simple = complexity_score - self.simple_threshold
            moderate_range = self.complex_threshold - self.simple_threshold
            confidence = 0.5 + (distance_from_simple / moderate_range - 0.5) * 0.5

        confidence = max(0.0, min(1.0, confidence))

        if not reasons:
            reasons.append("Query complexity determined by overall score")

        return ComplexityAnalysis(
            level=level,
            confidence=confidence,
            reasons=reasons,
            suggested_agents=suggested_agents,
            needs_voi=needs_voi,
        )

    def _analyze_length(self, query: str) -> float:
        """Analyze query length factor.

        Returns:
            Score from 0.0 (short) to 1.0 (long)
        """
        length = len(query)
        if length < 50:
            return 0.0
        elif length < 100:
            return 0.3
        elif length < 200:
            return 0.6
        elif length < 400:
            return 0.8
        else:
            return 1.0

    def _analyze_patterns(self, query: str) -> float:
        """Analyze query patterns.

        Returns:
            Score from 0.0 (simple) to 1.0 (complex)
        """
        query_lower = query.lower()

        # Check for simple patterns
        for pattern in self.SIMPLE_PATTERNS:
            if re.match(pattern, query_lower):
                return 0.0

        # Check for complex patterns
        complex_matches = 0
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, query_lower):
                complex_matches += 1

        if complex_matches == 0:
            return 0.2
        elif complex_matches == 1:
            return 0.5
        else:
            return 1.0

    def _analyze_tools(self, query: str) -> tuple[float, Set[str]]:
        """Analyze tool requirements.

        Returns:
            (score, set of required tools)
        """
        query_lower = query.lower()
        required_tools = set()

        for tool_name, keywords in self.TOOL_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                required_tools.add(tool_name)

        if not required_tools:
            return 0.0, set()
        elif len(required_tools) == 1:
            return 0.3, required_tools
        elif len(required_tools) == 2:
            return 0.6, required_tools
        else:
            return 1.0, required_tools

    def _analyze_ambiguity(self, query: str) -> float:
        """Analyze ambiguity in query.

        Returns:
            Score from 0.0 (certain) to 1.0 (ambiguous)
        """
        query_lower = query.lower()
        ambiguity_count = sum(1 for word in self.AMBIGUITY_KEYWORDS if word in query_lower)

        if ambiguity_count == 0:
            return 0.0
        elif ambiguity_count == 1:
            return 0.5
        else:
            return 1.0

    def _analyze_domain(self, query: str) -> float:
        """Analyze domain specificity.

        Returns:
            Score from 0.0 (general) to 1.0 (domain-specific)
        """
        # Domain-specific terms (technical jargon)
        domain_terms = {
            "machine learning",
            "neural network",
            "api",
            "database",
            "algorithm",
            "data structure",
            "design pattern",
            "microservice",
            "kubernetes",
            "docker",
            "ci/cd",
            "authentication",
            "authorization",
            "encryption",
        }

        query_lower = query.lower()
        domain_count = sum(1 for term in domain_terms if term in query_lower)

        if domain_count == 0:
            return 0.0
        elif domain_count <= 2:
            return 0.5
        else:
            return 1.0

    def should_use_bayesian(self, query: str) -> bool:
        """Quick check if query should use Bayesian orchestration.

        Args:
            query: User query string

        Returns:
            True if Bayesian orchestration is recommended
        """
        analysis = self.analyze(query)
        return analysis.level in {ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX}
