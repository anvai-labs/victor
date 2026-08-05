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

"""Search query routing for optimal tool selection.

This module analyzes search queries and routes them to the most
appropriate search or graph tool:
- Keyword search: Exact patterns, code identifiers, literal strings
- Semantic search: Conceptual queries, explanations, pattern finding
- Graph traversal: Caller/callee tracing and execution-path questions

Design Principles:
- Fast routing decisions (no LLM calls)
- Defaulting to keyword for precision
- Configurable thresholds
- Explainable decisions
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Type of search to perform."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"  # Use both and merge results


# Type alias for suggested search tools
SuggestedTool = Literal["code_search", "semantic_code_search"]


@dataclass
class SearchRoute:
    """Result of search routing decision.

    Attributes:
        search_type: The recommended search type
        confidence: Confidence in the routing decision (0.0-1.0)
        reason: Human-readable explanation for the decision
        transformed_query: Optionally transformed query for the search
        matched_patterns: List of patterns that influenced the decision
    """

    search_type: SearchType
    confidence: float
    reason: str
    transformed_query: Optional[str]
    matched_patterns: List[str]
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)


# Patterns that indicate KEYWORD search is better
# Format: (regex_pattern, weight, description, case_sensitive)
# Case-sensitive patterns (4th element True) are NOT compiled with IGNORECASE
KEYWORD_SIGNALS: List[Tuple[str, float, str, bool]] = [
    # Quoted strings are literal searches
    (r'"[^"]+"', 1.0, "quoted_string", False),
    (r"'[^']+'", 1.0, "single_quoted_string", False),
    # Code identifiers (case-sensitive for accuracy)
    (r"\bclass\s+[A-Z][a-zA-Z0-9_]*\b", 1.0, "class_name", True),
    (r"\bdef\s+[a-z_][a-zA-Z0-9_]*\b", 1.0, "function_def", True),
    (r"\bfunction\s+[a-z_][a-zA-Z0-9_]*\b", 0.9, "function_name", True),
    (r"\bimport\s+[a-z_][a-zA-Z0-9_.]*\b", 1.0, "import_statement", True),
    (r"\bfrom\s+[a-z_][a-zA-Z0-9_.]*\s+import\b", 1.0, "from_import", True),
    # Error patterns (case-sensitive - must be PascalCase)
    (r"\b[A-Z][a-zA-Z]*Error\b", 0.9, "error_class", True),
    (r"\b[A-Z][a-zA-Z]*Exception\b", 0.9, "exception_class", True),
    # Specific method/attribute access
    (r"\.\s*[a-z_][a-zA-Z0-9_]*\s*\(", 0.8, "method_call", True),
    (r"\b[a-z_][a-zA-Z0-9_]*\s*=", 0.7, "variable_assignment", True),
    # File patterns (case-insensitive)
    (r"\b\w+\.(py|js|ts|java|cpp|go|rs)\b", 0.8, "file_extension", False),
    # Decorators (case-sensitive)
    (r"@[a-z_][a-zA-Z0-9_]*", 0.9, "decorator", True),
    # Type hints
    (r":\s*(str|int|float|bool|list|dict|None)\b", 0.7, "type_hint", True),
    (r"->\s*(str|int|float|bool|list|dict|None)\b", 0.7, "return_type", True),
    # Exact identifiers (CamelCase or snake_case) - MUST be case-sensitive
    (r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b", 0.6, "camel_case", True),
    (r"\b[a-z]+_[a-z_]+\b", 0.5, "snake_case", True),
]

# Patterns that indicate SEMANTIC search is better
# All semantic patterns are case-insensitive (4th element False)
SEMANTIC_SIGNALS: List[Tuple[str, float, str, bool]] = [
    # Conceptual queries
    (r"\bhow\s+(does|do|is|are|can|to)\b", 1.0, "how_question", False),
    (r"\bwhy\s+(does|do|is|are)\b", 1.0, "why_question", False),
    (r"\bwhat\s+(is|are|does)\b", 0.9, "what_question", False),
    (r"\bexplain\b", 1.0, "explain", False),
    (r"\bdescribe\b", 0.9, "describe", False),
    (r"\bunderstand\b", 0.8, "understand", False),
    # Pattern/concept finding
    (r"\bpattern(s)?\b", 0.9, "patterns", False),
    (r"\bapproach(es)?\b", 0.8, "approaches", False),
    (r"\bstrateg(y|ies)\b", 0.8, "strategies", False),
    (r"\barchitecture\b", 0.9, "architecture", False),
    (r"\bdesign\b", 0.7, "design", False),
    (r"\bworkflow\b", 0.8, "workflow", False),
    # Relationships
    (r"\brelat(ed|ionship|es)\b", 0.9, "relationships", False),
    (r"\bdependen(cy|cies|t)\b", 0.8, "dependencies", False),
    (r"\bconnect(ed|ion|s)?\b", 0.7, "connections", False),
    (r"\binteract(s|ion|ions)?\b", 0.8, "interactions", False),
    # Quality/characteristics
    (r"\bbest\s+practic(e|es)\b", 0.9, "best_practices", False),
    (r"\bissue(s)?\b", 0.7, "issues", False),
    (r"\bproblem(s)?\b", 0.7, "problems", False),
    (r"\bimprove(ment|ments)?\b", 0.8, "improvements", False),
    (r"\boptimiz(e|ation)\b", 0.8, "optimization", False),
    # Abstract concepts
    (r"\bconcept(s)?\b", 0.9, "concepts", False),
    (r"\bpurpose\b", 0.8, "purpose", False),
    (r"\breason(s|ing)?\b", 0.7, "reasoning", False),
    (r"\bgoal(s)?\b", 0.7, "goals", False),
]

# Patterns that indicate bug similarity search is better than generic semantic search.
# These queries still map to code_search, but with mode="bugs" so providers can
# use graph-enriched bug similarity when available.
BUG_SIMILARITY_SIGNALS: List[Tuple[str, float, str, bool]] = [
    (
        r"\bsimilar\s+(bug|bugs|issue|issues|failure|failures|regression|regressions)\b",
        1.0,
        "similar_bug",
        False,
    ),
    (r"\b(regression|regressions)\b", 0.9, "regression", False),
    (
        r"\b(crash|crashes|crashed|panic|panics|panicked|segfault)\b",
        0.9,
        "crash",
        False,
    ),
    (r"\b(traceback|stack\s*trace)\b", 0.8, "stacktrace", False),
    (r"\b(bug|bugs|failing|failure|failures|broken)\b", 0.7, "bug_symptom", False),
]

ISSUE_LOCALIZATION_SIGNALS: List[Tuple[str, float, str, bool]] = [
    (
        r"\b(?:localize|locate)\s+(?:the\s+)?(?:issue|bug|regression|failure|problem|fix)\b",
        1.0,
        "localize_issue",
        False,
    ),
    (
        r"\b(?:which|what)\s+files?\s+(?:should\s+i\s+)?(?:edit|change|modify|fix)\b",
        1.0,
        "which_files_to_edit",
        False,
    ),
    (
        r"\bwhere\s+(?:should\s+i\s+)?(?:edit|change|modify|fix)\b",
        0.9,
        "where_to_edit",
        False,
    ),
    (
        r"\b(?:relevant|affected|suspicious)\s+files?\b",
        0.8,
        "relevant_files",
        False,
    ),
    (
        r"\bfiles?\s+to\s+(?:edit|change|modify|fix)\b",
        0.9,
        "files_to_edit",
        False,
    ),
]

CHANGE_IMPACT_SIGNALS: List[Tuple[str, float, str, bool]] = [
    (
        r"\bwhat\s+breaks\s+if\s+i\s+(?:change|modify|edit)\b",
        1.0,
        "what_breaks_if_i_change",
        False,
    ),
    (
        r"\bimpact\s+of\s+(?:changing|modifying|editing)\b",
        0.95,
        "impact_of_change",
        False,
    ),
    (
        r"\bblast\s+radius\b",
        0.95,
        "blast_radius",
        False,
    ),
    (
        r"\baffected\s+files?\b",
        0.85,
        "affected_files",
        False,
    ),
    (
        r"\b(?:what|which)\s+(?:breaks|is\s+affected)\b",
        0.8,
        "what_breaks",
        False,
    ),
]

GRAPH_ANALYTIC_SIGNALS: List[Tuple[str, str, str, bool]] = [
    (
        r"\b(?:show|find|get|list)\s+(?:the\s+)?(?:top[\s-]*\d+\s+)?(?:most\s+)?"
        r"(?:important|central)\s+(?:symbols?|functions?|methods?|modules?|nodes?)\b",
        "pagerank",
        "central_symbols",
        False,
    ),
    (
        r"\b(?:important|central|hub)\s+(?:symbols?|functions?|methods?|modules?|nodes?)\b",
        "pagerank",
        "important_symbols",
        False,
    ),
    (
        r"\bpagerank\b",
        "pagerank",
        "pagerank",
        False,
    ),
]

GRAPH_ARCHITECTURE_SIGNALS: List[Tuple[str, str, str, bool]] = [
    (
        r"\b(?:summari(?:ze|se)|overview|describe|explain|show|give\s+me|provide)\s+"
        r"(?:the\s+)?architecture\b",
        "module_pagerank",
        "architecture_summary",
        False,
    ),
    (
        r"\b(?:show|find|get|list)\s+(?:the\s+)?(?:top[\s-]*\d+\s+)?(?:most\s+)?"
        r"important\s+(?:modules?|components?|packages?|subsystems?)\b",
        "module_pagerank",
        "important_modules",
        False,
    ),
    (
        r"\b(?:show|find|get|list)\s+(?:the\s+)?(?:top[\s-]*\d+\s+)?(?:most\s+)?"
        r"(?:central|critical)\s+(?:modules?|components?|packages?|subsystems?)\b",
        "module_centrality",
        "central_modules",
        False,
    ),
    (
        r"\barchitecture\s+(?:hotspots?|summary)\b",
        "module_pagerank",
        "architecture_hotspots",
        False,
    ),
]


def _graph_symbol_fragment(*, quoted_group: str, bare_group: str) -> str:
    """Return a reusable regex fragment for graph symbol extraction."""
    return (
        rf"(?:[`'\"](?P<{quoted_group}>[^`'\"]+)[`'\"]|"
        rf"(?P<{bare_group}>[A-Za-z_][A-Za-z0-9_:.<>/-]*))"
    )


_GRAPH_SYMBOL_FRAGMENT = _graph_symbol_fragment(quoted_group="quoted", bare_group="bare")
_GRAPH_SOURCE_SYMBOL_FRAGMENT = _graph_symbol_fragment(
    quoted_group="source_quoted",
    bare_group="source_bare",
)
_GRAPH_TARGET_SYMBOL_FRAGMENT = _graph_symbol_fragment(
    quoted_group="target_quoted",
    bare_group="target_bare",
)
_GRAPH_FILE_FRAGMENT = r"(?P<file>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_]+)"

GRAPH_NAVIGATION_SIGNALS: List[Tuple[str, str, int, str, bool]] = [
    (
        rf"\b(?:show|get|find|list)\s+(?:the\s+)?(?:neighbors?|connections?|dependencies)\s+"
        rf"(?:of|for)\s+(?:the\s+)?(?:symbol\s+|function\s+|method\s+|class\s+|module\s+)?"
        rf"{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "neighbors",
        1,
        "neighbor_of",
        False,
    ),
    (
        rf"\b(?:what|which)\s+(?:symbols?|functions?|methods?|modules?|nodes?)?\s*"
        rf"(?:are\s+)?(?:connected|related)\s+to\s+(?:the\s+)?"
        rf"(?:symbol\s+|function\s+|method\s+|class\s+|module\s+)?"
        rf"{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "neighbors",
        1,
        "connected_to",
        False,
    ),
    (
        rf"\b(?:find|show|get|trace|list)\s+(?:the\s+)?"
        rf"(?:shortest\s+|dependency\s+|connection\s+)?path\s+between\s+"
        rf"{_GRAPH_SOURCE_SYMBOL_FRAGMENT}\s+and\s+{_GRAPH_TARGET_SYMBOL_FRAGMENT}"
        rf"(?=$|[\s?.!,])",
        "path",
        0,
        "path_between",
        False,
    ),
    (
        rf"\b(?:find|show|get|trace|list)\s+(?:the\s+)?"
        rf"(?:shortest\s+|dependency\s+|connection\s+)?path\s+from\s+"
        rf"{_GRAPH_SOURCE_SYMBOL_FRAGMENT}\s+to\s+{_GRAPH_TARGET_SYMBOL_FRAGMENT}"
        rf"(?=$|[\s?.!,])",
        "path",
        0,
        "path_from_to",
        False,
    ),
]

GRAPH_FILE_SIGNALS: List[Tuple[str, str, str, bool]] = [
    (
        rf"\b(?:show|find|get|list)\s+(?:the\s+)?(?:file\s+)?dependencies\s+(?:for|of)\s+"
        rf"{_GRAPH_FILE_FRAGMENT}(?=$|[\s?.!,])",
        "file_deps",
        "file_dependencies_for",
        False,
    ),
    (
        rf"\b(?:what|which)\s+files?\s+depend\s+on\s+{_GRAPH_FILE_FRAGMENT}(?=$|[\s?.!,])",
        "file_deps",
        "file_dependents",
        False,
    ),
    (
        rf"\bwhat\s+files?\s+does\s+{_GRAPH_FILE_FRAGMENT}\s+depend\s+on\b",
        "file_deps",
        "file_depends_on",
        False,
    ),
    (
        rf"\b(?:summari(?:ze|se)|overview|describe|explain|show)\s+(?:the\s+)?"
        rf"architecture\s+(?:of|around|for)\s+{_GRAPH_FILE_FRAGMENT}(?=$|[\s?.!,])",
        "file_deps",
        "file_architecture_summary",
        False,
    ),
]

GRAPH_DEPENDENCY_SIGNALS: List[Tuple[str, str, str, bool]] = [
    (
        rf"\b(?:what|which)\s+(?:modules?|packages?|components?|subsystems?)\s+depend\s+on\s+"
        rf"{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "in",
        "module_dependents",
        False,
    ),
    (
        rf"\b(?:what|which)\s+(?:modules?|packages?|components?|subsystems?)\s+does\s+"
        rf"{_GRAPH_SYMBOL_FRAGMENT}\s+depend\s+on\b",
        "out",
        "module_dependencies",
        False,
    ),
]

GRAPH_DEPENDENCY_TRAVERSAL_SIGNALS: List[Tuple[str, str, str, bool]] = [
    (
        rf"\b(?:show|find|get|trace|list)\s+(?:the\s+)?(?:downstream|outgoing)\s+"
        rf"dependencies\s+(?:of|for)\s+{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "out",
        "downstream_dependencies",
        False,
    ),
    (
        rf"\b(?:show|find|get|trace|list)\s+(?:the\s+)?(?:upstream|incoming)\s+"
        rf"dependencies\s+(?:of|for)\s+{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "in",
        "upstream_dependencies",
        False,
    ),
]

GRAPH_SCOPED_ARCHITECTURE_SIGNALS: List[Tuple[str, str, bool]] = [
    (
        rf"\b(?:summari(?:ze|se)|overview|describe|explain|show)\s+(?:the\s+)?"
        rf"architecture\s+(?:of|around|for)\s+{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "scoped_architecture_summary",
        False,
    ),
]

# Patterns that indicate a graph traversal is more appropriate than plain search.
# Format: (regex_pattern, mode, depth, description, case_sensitive)
GRAPH_TRAVERSAL_SIGNALS: List[Tuple[str, str, int, str, bool]] = [
    (
        rf"\b(?:who|what|which(?:\s+functions?)?)\s+call(?:s)?\s+"
        rf"{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "callers",
        2,
        "who_calls",
        False,
    ),
    (
        rf"\b(?:find|show|get|list)\s+(?:all\s+)?callers?\s+of\s+"
        rf"(?:the\s+)?(?:function\s+|method\s+)?{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "callers",
        2,
        "find_callers",
        False,
    ),
    (
        rf"\bwhat\s+(?:functions?\s+)?does\s+{_GRAPH_SYMBOL_FRAGMENT}\s+call\b",
        "callees",
        2,
        "what_does_call",
        False,
    ),
    (
        rf"\b(?:find|show|get|list)\s+(?:all\s+)?callees?\s+of\s+"
        rf"(?:the\s+)?(?:function\s+|method\s+)?{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "callees",
        2,
        "find_callees",
        False,
    ),
    (
        rf"\btrace\s+(?:the\s+)?execution(?:\s+path)?\s+(?:from|for|of)\s+"
        rf"{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "trace",
        3,
        "trace_execution",
        False,
    ),
    (
        rf"\b(?:show|get|find)\s+(?:the\s+)?execution(?:\s+path)?\s+(?:from|for|of)\s+"
        rf"{_GRAPH_SYMBOL_FRAGMENT}(?=$|[\s?.!,])",
        "trace",
        3,
        "execution_path",
        False,
    ),
]


class SearchRouter:
    """Routes search queries to appropriate search tools.

    This router analyzes query characteristics to determine whether
    keyword or semantic search would yield better results.

    Example:
        router = SearchRouter()
        route = router.route("class BaseTool")
        print(route.search_type)  # SearchType.KEYWORD

        route = router.route("how does error handling work")
        print(route.search_type)  # SearchType.SEMANTIC
    """

    def __init__(
        self,
        keyword_threshold: float = 0.5,
        semantic_threshold: float = 0.5,
        hybrid_threshold: float = 0.3,
        custom_signals: Optional[dict[str, List[Tuple[Any, ...]]]] = None,
        custom_routers: Optional[List[Callable[[str], Optional[SearchRoute]]]] = None,
    ):
        """Initialize the search router.

        Args:
            keyword_threshold: Min score for keyword routing
            semantic_threshold: Min score for semantic routing
            hybrid_threshold: Min scores to trigger hybrid search
            custom_signals: Additional routing signals
            custom_routers: Custom router functions to try first
        """
        self.keyword_threshold = keyword_threshold
        self.semantic_threshold = semantic_threshold
        self.hybrid_threshold = hybrid_threshold
        self.custom_routers = custom_routers or []

        # Compile patterns
        self._keyword_patterns: List[Tuple[re.Pattern, float, str]] = []
        self._semantic_patterns: List[Tuple[re.Pattern, float, str]] = []
        self._bug_patterns: List[Tuple[re.Pattern, float, str]] = []
        self._localization_patterns: List[Tuple[re.Pattern, float, str]] = []
        self._impact_patterns: List[Tuple[re.Pattern, float, str]] = []
        self._graph_architecture_patterns: List[Tuple[re.Pattern, str, str]] = []
        self._graph_analytic_patterns: List[Tuple[re.Pattern, str, str]] = []
        self._graph_file_patterns: List[Tuple[re.Pattern, str, str]] = []
        self._graph_dependency_patterns: List[Tuple[re.Pattern, str, str]] = []
        self._graph_dependency_traversal_patterns: List[Tuple[re.Pattern, str, str]] = []
        self._graph_scoped_architecture_patterns: List[Tuple[re.Pattern, str]] = []
        self._graph_navigation_patterns: List[Tuple[re.Pattern, str, int, str]] = []
        self._graph_patterns: List[Tuple[re.Pattern, str, int, str]] = []

        self._compile_patterns(KEYWORD_SIGNALS, self._keyword_patterns)
        self._compile_patterns(SEMANTIC_SIGNALS, self._semantic_patterns)
        self._compile_patterns(BUG_SIMILARITY_SIGNALS, self._bug_patterns)
        self._compile_patterns(ISSUE_LOCALIZATION_SIGNALS, self._localization_patterns)
        self._compile_patterns(CHANGE_IMPACT_SIGNALS, self._impact_patterns)
        self._compile_graph_analytic_patterns(
            GRAPH_ARCHITECTURE_SIGNALS, self._graph_architecture_patterns
        )
        self._compile_graph_analytic_patterns(GRAPH_ANALYTIC_SIGNALS, self._graph_analytic_patterns)
        self._compile_graph_analytic_patterns(GRAPH_FILE_SIGNALS, self._graph_file_patterns)
        self._compile_graph_analytic_patterns(
            GRAPH_DEPENDENCY_SIGNALS, self._graph_dependency_patterns
        )
        self._compile_graph_analytic_patterns(
            GRAPH_DEPENDENCY_TRAVERSAL_SIGNALS,
            self._graph_dependency_traversal_patterns,
        )
        self._compile_graph_scoped_patterns(
            GRAPH_SCOPED_ARCHITECTURE_SIGNALS,
            self._graph_scoped_architecture_patterns,
        )
        self._compile_graph_patterns(GRAPH_NAVIGATION_SIGNALS, self._graph_navigation_patterns)
        self._compile_graph_patterns(GRAPH_TRAVERSAL_SIGNALS, self._graph_patterns)

        # Add custom signals
        if custom_signals:
            if "keyword" in custom_signals:
                self._compile_patterns(custom_signals["keyword"], self._keyword_patterns)
            if "semantic" in custom_signals:
                self._compile_patterns(custom_signals["semantic"], self._semantic_patterns)
            if "bug" in custom_signals:
                self._compile_patterns(custom_signals["bug"], self._bug_patterns)
            if "localization" in custom_signals:
                self._compile_patterns(custom_signals["localization"], self._localization_patterns)
            if "impact" in custom_signals:
                self._compile_patterns(custom_signals["impact"], self._impact_patterns)
            if "graph_architecture" in custom_signals:
                self._compile_graph_analytic_patterns(
                    custom_signals["graph_architecture"],
                    self._graph_architecture_patterns,
                )
            if "graph_analytics" in custom_signals:
                self._compile_graph_analytic_patterns(
                    custom_signals["graph_analytics"],
                    self._graph_analytic_patterns,
                )
            if "graph_files" in custom_signals:
                self._compile_graph_analytic_patterns(
                    custom_signals["graph_files"],
                    self._graph_file_patterns,
                )
            if "graph_dependencies" in custom_signals:
                self._compile_graph_analytic_patterns(
                    custom_signals["graph_dependencies"],
                    self._graph_dependency_patterns,
                )
            if "graph_dependency_traversal" in custom_signals:
                self._compile_graph_analytic_patterns(
                    custom_signals["graph_dependency_traversal"],
                    self._graph_dependency_traversal_patterns,
                )
            if "graph_scoped_architecture" in custom_signals:
                self._compile_graph_scoped_patterns(
                    custom_signals["graph_scoped_architecture"],
                    self._graph_scoped_architecture_patterns,
                )
            if "graph_navigation" in custom_signals:
                self._compile_graph_patterns(
                    custom_signals["graph_navigation"],
                    self._graph_navigation_patterns,
                )
            if "graph" in custom_signals:
                self._compile_graph_patterns(custom_signals["graph"], self._graph_patterns)

    def _compile_patterns(
        self,
        signals: List[Tuple[str, float, str, bool]],
        target: List[Tuple[re.Pattern, float, str]],
    ) -> None:
        """Compile regex patterns.

        Args:
            signals: List of (pattern, weight, name, case_sensitive) tuples
            target: Target list to append compiled patterns
        """
        for pattern_str, weight, name, case_sensitive in signals:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled = re.compile(pattern_str, flags)
                target.append((compiled, weight, name))
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern_str}': {e}")

    def _compile_graph_patterns(
        self,
        signals: List[Tuple[str, str, int, str, bool]],
        target: List[Tuple[re.Pattern, str, int, str]],
    ) -> None:
        """Compile graph-routing regex patterns."""
        for pattern_str, mode, depth, name, case_sensitive in signals:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled = re.compile(pattern_str, flags)
                target.append((compiled, mode, depth, name))
            except re.error as e:
                logger.warning(f"Invalid graph pattern '{pattern_str}': {e}")

    def _compile_graph_analytic_patterns(
        self,
        signals: List[Tuple[str, str, str, bool]],
        target: List[Tuple[re.Pattern, str, str]],
    ) -> None:
        """Compile graph-analysis regex patterns."""
        for pattern_str, mode, name, case_sensitive in signals:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled = re.compile(pattern_str, flags)
                target.append((compiled, mode, name))
            except re.error as e:
                logger.warning(f"Invalid graph analytic pattern '{pattern_str}': {e}")

    def _compile_graph_scoped_patterns(
        self,
        signals: List[Tuple[str, str, bool]],
        target: List[Tuple[re.Pattern, str]],
    ) -> None:
        """Compile graph patterns that extract a single scoped symbol."""
        for pattern_str, name, case_sensitive in signals:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled = re.compile(pattern_str, flags)
                target.append((compiled, name))
            except re.error as e:
                logger.warning(f"Invalid graph scoped pattern '{pattern_str}': {e}")

    def route(self, query: str) -> SearchRoute:
        """Route a search query to appropriate search type.

        Args:
            query: The search query

        Returns:
            SearchRoute with recommended search type and metadata
        """
        # Try custom routers first
        for router in self.custom_routers:
            result = router(query)
            if result is not None:
                return result

        graph_route = self._route_graph_query(query)
        if graph_route is not None:
            return graph_route

        graph_file_route = self._route_graph_file_query(query)
        if graph_file_route is not None:
            return graph_file_route

        scoped_architecture_route = self._route_scoped_architecture_query(query)
        if scoped_architecture_route is not None:
            return scoped_architecture_route

        graph_architecture_route = self._route_graph_architecture_query(query)
        if graph_architecture_route is not None:
            return graph_architecture_route

        graph_analytic_route = self._route_graph_analytic_query(query)
        if graph_analytic_route is not None:
            return graph_analytic_route

        graph_dependency_route = self._route_graph_dependency_query(query)
        if graph_dependency_route is not None:
            return graph_dependency_route

        graph_dependency_traversal_route = self._route_graph_dependency_traversal_query(query)
        if graph_dependency_traversal_route is not None:
            return graph_dependency_traversal_route

        graph_navigation_route = self._route_graph_navigation_query(query)
        if graph_navigation_route is not None:
            return graph_navigation_route

        localization_route = self._route_issue_localization_query(query)
        if localization_route is not None:
            return localization_route

        impact_route = self._route_change_impact_query(query)
        if impact_route is not None:
            return impact_route

        bug_route = self._route_bug_similarity_query(query)
        if bug_route is not None:
            return bug_route

        # Check for quoted strings (always keyword)
        if self._has_quoted_string(query):
            return SearchRoute(
                search_type=SearchType.KEYWORD,
                confidence=1.0,
                reason="Query contains quoted literal string",
                transformed_query=self._extract_quoted_string(query),
                matched_patterns=["quoted_string"],
            )

        # Score both search types
        keyword_score, keyword_matches = self._score_patterns(query, self._keyword_patterns)
        semantic_score, semantic_matches = self._score_patterns(query, self._semantic_patterns)

        # Normalize scores
        max_score = max(keyword_score + semantic_score, 1.0)
        keyword_norm = keyword_score / max_score if max_score > 0 else 0
        semantic_norm = semantic_score / max_score if max_score > 0 else 0

        # Determine route
        if keyword_score >= self.hybrid_threshold and semantic_score >= self.hybrid_threshold:
            # Both signals present - hybrid search
            return SearchRoute(
                search_type=SearchType.HYBRID,
                confidence=min(keyword_norm, semantic_norm) + 0.3,
                reason=f"Query has both keyword ({keyword_score:.2f}) and semantic ({semantic_score:.2f}) signals",
                transformed_query=None,
                matched_patterns=keyword_matches + semantic_matches,
            )
        elif keyword_score > semantic_score and keyword_score >= self.keyword_threshold:
            return SearchRoute(
                search_type=SearchType.KEYWORD,
                confidence=keyword_norm,
                reason=f"Query matches keyword patterns (score: {keyword_score:.2f})",
                transformed_query=None,
                matched_patterns=keyword_matches,
            )
        elif semantic_score > keyword_score and semantic_score >= self.semantic_threshold:
            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=semantic_norm,
                reason=f"Query matches semantic patterns (score: {semantic_score:.2f})",
                transformed_query=None,
                matched_patterns=semantic_matches,
            )
        else:
            # Default to keyword for precision
            return SearchRoute(
                search_type=SearchType.KEYWORD,
                confidence=0.3,
                reason="No strong signals detected, defaulting to keyword search",
                transformed_query=None,
                matched_patterns=[],
            )

    def _route_graph_query(self, query: str) -> Optional[SearchRoute]:
        """Return a graph-tool route for caller/callee/trace style queries."""
        for pattern, mode, depth, name in self._graph_patterns:
            match = pattern.search(query)
            if not match:
                continue

            symbol = self._normalize_graph_symbol(
                match.groupdict().get("quoted") or match.groupdict().get("bare")
            )
            if not symbol:
                continue

            mode_reason = {
                "callers": "reverse call-graph traversal",
                "callees": "forward call-graph traversal",
                "trace": "execution-path tracing",
            }.get(mode, "graph traversal")

            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=0.95,
                reason=f"Query is a {mode_reason} question suited for graph traversal",
                transformed_query=symbol,
                matched_patterns=[name],
                tool_name="graph",
                tool_arguments={"mode": mode, "node": symbol, "depth": depth},
            )

        return None

    def _route_graph_analytic_query(self, query: str) -> Optional[SearchRoute]:
        """Return graph-tool routes for centrality / importance questions."""
        for pattern, mode, name in self._graph_analytic_patterns:
            if not pattern.search(query):
                continue

            if mode == "pagerank":
                top_k = self._extract_requested_top_k(query, default=5)
                return SearchRoute(
                    search_type=SearchType.SEMANTIC,
                    confidence=0.9,
                    reason="Query asks for important or central symbols suited for graph ranking",
                    transformed_query=None,
                    matched_patterns=[name],
                    tool_name="graph",
                    tool_arguments={"mode": "pagerank", "top_k": top_k},
                )

        return None

    def _route_graph_architecture_query(self, query: str) -> Optional[SearchRoute]:
        """Return graph-tool routes for module-level architecture questions."""
        for pattern, mode, name in self._graph_architecture_patterns:
            if not pattern.search(query):
                continue

            top_k = self._extract_requested_top_k(query, default=5)
            tool_arguments: Dict[str, Any] = {
                "mode": mode,
                "top_k": top_k,
                "only_runtime": True,
                "include_callsites": True,
                "max_callsites": 3,
            }
            if name == "architecture_summary":
                tool_arguments.update(
                    {
                        "structured": True,
                        "include_modules": True,
                    }
                )
            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=0.93,
                reason="Query asks for architecture-level hotspots suited for module graph ranking",
                transformed_query=None,
                matched_patterns=[name],
                tool_name="graph",
                tool_arguments=tool_arguments,
            )

        return None

    def _route_graph_file_query(self, query: str) -> Optional[SearchRoute]:
        """Return graph-tool routes for file dependency questions."""
        for pattern, mode, name in self._graph_file_patterns:
            match = pattern.search(query)
            if not match:
                continue

            file_path = self._normalize_graph_file(match.groupdict().get("file"))
            if not file_path:
                continue

            tool_arguments: Dict[str, Any] = {"mode": mode, "file": file_path}
            if name == "file_dependents":
                tool_arguments["direction"] = "in"
            elif name == "file_depends_on":
                tool_arguments["direction"] = "out"
            elif name == "file_architecture_summary":
                tool_arguments.update(
                    {
                        "direction": "both",
                        "structured": True,
                        "include_modules": True,
                        "include_symbols": True,
                        "include_calls": True,
                        "include_refs": True,
                    }
                )

            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=0.93,
                reason="Query asks for file-level dependency analysis suited for graph file dependencies",
                transformed_query=file_path,
                matched_patterns=[name],
                tool_name="graph",
                tool_arguments=tool_arguments,
            )

        return None

    def _route_scoped_architecture_query(self, query: str) -> Optional[SearchRoute]:
        """Return graph-tool routes for architecture questions scoped to one component."""
        for pattern, name in self._graph_scoped_architecture_patterns:
            match = pattern.search(query)
            if not match:
                continue

            symbol = self._extract_graph_symbol(match)
            if not symbol:
                continue

            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=0.94,
                reason="Query asks for a scoped architecture summary suited for structured graph neighborhood analysis",
                transformed_query=symbol,
                matched_patterns=[name],
                tool_name="graph",
                tool_arguments={
                    "mode": "neighbors",
                    "node": symbol,
                    "depth": self._extract_requested_depth(query, default=2),
                    "direction": "both",
                    "structured": True,
                    "include_modules": True,
                    "include_symbols": True,
                    "include_calls": True,
                    "include_refs": True,
                },
            )

        return None

    def _route_graph_dependency_query(self, query: str) -> Optional[SearchRoute]:
        """Return graph-tool routes for directional module dependency questions."""
        for pattern, direction, name in self._graph_dependency_patterns:
            match = pattern.search(query)
            if not match:
                continue

            symbol = self._extract_graph_symbol(match)
            if not symbol:
                continue

            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=0.92,
                reason="Query asks for directional module dependencies suited for graph neighborhood traversal",
                transformed_query=symbol,
                matched_patterns=[name],
                tool_name="graph",
                tool_arguments={
                    "mode": "neighbors",
                    "node": symbol,
                    "depth": 1,
                    "direction": direction,
                    "modules_only": True,
                },
            )

        return None

    def _route_graph_dependency_traversal_query(self, query: str) -> Optional[SearchRoute]:
        """Return graph-tool routes for deep upstream/downstream dependency traversal."""
        for pattern, direction, name in self._graph_dependency_traversal_patterns:
            match = pattern.search(query)
            if not match:
                continue

            symbol = self._extract_graph_symbol(match)
            if not symbol:
                continue

            return SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=0.93,
                reason="Query asks for deeper dependency traversal suited for graph neighborhood analysis",
                transformed_query=symbol,
                matched_patterns=[name],
                tool_name="graph",
                tool_arguments={
                    "mode": "neighbors",
                    "node": symbol,
                    "depth": self._extract_requested_depth(query, default=2),
                    "direction": direction,
                    "modules_only": True,
                },
            )

        return None

    def _route_graph_navigation_query(self, query: str) -> Optional[SearchRoute]:
        """Return graph-tool routes for neighbor and path questions."""
        for pattern, mode, depth, name in self._graph_navigation_patterns:
            match = pattern.search(query)
            if not match:
                continue

            if mode == "neighbors":
                symbol = self._extract_graph_symbol(match)
                if not symbol:
                    continue
                return SearchRoute(
                    search_type=SearchType.SEMANTIC,
                    confidence=0.92,
                    reason="Query asks for symbol relationships suited for graph neighborhood traversal",
                    transformed_query=symbol,
                    matched_patterns=[name],
                    tool_name="graph",
                    tool_arguments={
                        "mode": "neighbors",
                        "node": symbol,
                        "depth": depth,
                    },
                )

            if mode == "path":
                source = self._extract_graph_symbol(match, prefix="source_")
                target = self._extract_graph_symbol(match, prefix="target_")
                if not source or not target:
                    continue
                return SearchRoute(
                    search_type=SearchType.SEMANTIC,
                    confidence=0.94,
                    reason="Query asks for a dependency/connection path suited for graph path analysis",
                    transformed_query=None,
                    matched_patterns=[name],
                    tool_name="graph",
                    tool_arguments={"mode": "path", "source": source, "target": target},
                )

        return None

    def _route_bug_similarity_query(self, query: str) -> Optional[SearchRoute]:
        """Return a code_search bug-similarity route for bug/regression style queries."""
        if self._has_quoted_string(query):
            return None

        bug_score, bug_matches = self._score_patterns(query, self._bug_patterns)
        if bug_score < 0.7:
            return None

        keyword_score, _ = self._score_patterns(query, self._keyword_patterns)
        if keyword_score >= 1.0:
            return None

        return SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=min(1.0, 0.45 + bug_score / 2.0),
            reason="Query describes a bug/regression pattern suited for bug similarity search",
            transformed_query=None,
            matched_patterns=bug_matches,
            tool_name="code_search",
            tool_arguments={"mode": "bugs"},
        )

    def _route_issue_localization_query(self, query: str) -> Optional[SearchRoute]:
        """Return a code_search localization route for file-localization queries."""
        localization_score, localization_matches = self._score_patterns(
            query, self._localization_patterns
        )
        if localization_score < 0.8:
            return None

        return SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=min(1.0, 0.5 + localization_score / 2.0),
            reason="Query asks for issue/file localization suited for graph-guided localization",
            transformed_query=None,
            matched_patterns=localization_matches,
            tool_name="code_search",
            tool_arguments={"mode": "localize"},
        )

    def _route_change_impact_query(self, query: str) -> Optional[SearchRoute]:
        """Return a code_search impact-analysis route for blast-radius queries."""
        impact_score, impact_matches = self._score_patterns(query, self._impact_patterns)
        if impact_score < 0.8:
            return None

        return SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=min(1.0, 0.5 + impact_score / 2.0),
            reason="Query asks for blast-radius / change-impact analysis suited for graph expansion",
            transformed_query=None,
            matched_patterns=impact_matches,
            tool_name="code_search",
            tool_arguments={"mode": "impact"},
        )

    def _score_patterns(
        self,
        query: str,
        patterns: List[Tuple[re.Pattern, float, str]],
    ) -> Tuple[float, List[str]]:
        """Score patterns against query."""
        total_score = 0.0
        matched = []
        for pattern, weight, name in patterns:
            if pattern.search(query):
                total_score += weight
                matched.append(name)
        return total_score, matched

    def _has_quoted_string(self, query: str) -> bool:
        """Check if query contains quoted string."""
        return bool(re.search(r'"[^"]+"', query) or re.search(r"'[^']+'", query))

    def _extract_quoted_string(self, query: str) -> str:
        """Extract quoted string from query."""
        match = re.search(r'"([^"]+)"', query)
        if match:
            return match.group(1)
        match = re.search(r"'([^']+)'", query)
        if match:
            return match.group(1)
        return query

    def _normalize_graph_symbol(self, symbol: Optional[str]) -> Optional[str]:
        """Normalize a graph symbol extracted from a natural-language query."""
        if not symbol:
            return None

        normalized = symbol.strip().strip("`'\"").strip().rstrip("?.!,:;")
        normalized = normalized.strip("()[]{}")
        if not normalized:
            return None

        if normalized.lower() in {
            "this",
            "that",
            "it",
            "function",
            "method",
            "class",
            "module",
            "symbol",
            "file",
        }:
            return None

        return normalized

    def _extract_graph_symbol(self, match: re.Match[str], prefix: str = "") -> Optional[str]:
        """Extract and normalize a graph symbol from a regex match."""
        groups = match.groupdict()
        return self._normalize_graph_symbol(
            groups.get(f"{prefix}quoted") or groups.get(f"{prefix}bare")
        )

    def _normalize_graph_file(self, file_path: Optional[str]) -> Optional[str]:
        """Normalize a file path extracted from a graph-oriented query."""
        if not file_path:
            return None

        normalized = file_path.strip().strip("`'\"").strip().rstrip("?.!,:;")
        return normalized or None

    def _extract_requested_top_k(
        self,
        query: str,
        default: int = 5,
        *,
        minimum: int = 1,
        maximum: int = 20,
    ) -> int:
        """Extract a user-requested top-k value from a natural-language query."""
        patterns = [
            r"\btop[\s-]*(\d+)\b",
            r"\b(\d+)\s+(?:most|top)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if not match:
                continue
            try:
                requested = int(match.group(1))
            except (TypeError, ValueError):
                continue
            return max(minimum, min(maximum, requested))
        return default

    def _extract_requested_depth(
        self,
        query: str,
        default: int = 1,
        *,
        minimum: int = 1,
        maximum: int = 6,
    ) -> int:
        """Extract a requested traversal depth from a natural-language query."""
        patterns = [
            r"\bdepth\s+(\d+)\b",
            r"\bto\s+depth\s+(\d+)\b",
            r"\b(\d+)\s+hops?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if not match:
                continue
            try:
                requested = int(match.group(1))
            except (TypeError, ValueError):
                continue
            return max(minimum, min(maximum, requested))
        return default

    def suggest_tool(self, query: str) -> SuggestedTool:
        """Suggest the best search tool for a query.

        Args:
            query: The search query

        Returns:
            SuggestedTool Literal value
        """
        route = self.route(query)
        if route.tool_name:
            return route.tool_name  # type: ignore
        if route.search_type == SearchType.SEMANTIC:
            return "semantic_code_search"
        return "code_search"


def route_query(query: str) -> SearchRoute:
    """Convenience function to route a query.

    Args:
        query: Search query

    Returns:
        SearchRoute with recommendation
    """
    router = SearchRouter()
    return router.route(query)


def suggest_search_tool(query: str) -> SuggestedTool:
    """Convenience function to get recommended tool.

    Args:
        query: Search query

    Returns:
        SuggestedTool Literal value
    """
    router = SearchRouter()
    return router.suggest_tool(query)


def is_keyword_query(query: str) -> bool:
    """Quick check if query should use keyword search.

    Args:
        query: Search query

    Returns:
        True if keyword search is recommended
    """
    route = route_query(query)
    return route.search_type in (SearchType.KEYWORD, SearchType.HYBRID)


def is_semantic_query(query: str) -> bool:
    """Quick check if query should use semantic search.

    Args:
        query: Search query

    Returns:
        True if semantic search is recommended
    """
    route = route_query(query)
    return route.search_type in (SearchType.SEMANTIC, SearchType.HYBRID)
