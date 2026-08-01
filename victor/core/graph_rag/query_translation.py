# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Query Translation - Natural Language to Graph Query (PH3-001 to PH3-006).

This module implements the translation layer for converting natural language
queries into structured graph queries. It provides:

1. Query Template Schema - Define structure for graph query templates
2. Common Templates - Pre-built templates for common query patterns
3. Template Validation - Validate template structure and parameters
4. Template Registry - Register and discover query templates
5. Translation Interface - Interface for NL→Graph query translation
6. LLM-based Translator - Use LLM to translate NL to graph queries

Usage:
    from victor.core.graph_rag.query_translation import (
        QueryTemplate,
        QueryTranslator,
        translate_query,
    )

    # Translate natural language to graph query
    result = translate_query(
        "Find all functions that call parse_json",
        graph_store=store,
    )
"""

from __future__ import annotations

import abc
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.storage.graph.protocol import GraphStoreProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# Query Template Schema (PH3-001)
# =============================================================================


class QueryType(Enum):
    """Types of graph queries."""

    # Structural queries - navigate graph structure
    NEIGHBORS = "neighbors"  # Find neighbors of a node
    PATH = "path"  # Find path between nodes
    REACHABLE = "reachable"  # Find reachable nodes
    CONNECTED = "connected"  # Check connectivity

    # Semantic queries - search by meaning
    SIMILAR = "similar"  # Find similar nodes
    SEARCH = "search"  # Full-text search
    SEMANTIC_SEARCH = "semantic_search"  # Vector similarity search

    # Pattern queries - find specific patterns
    PATTERN = "pattern"  # Find graph patterns
    SUBGRAPH = "subgraph"  # Extract subgraph
    COMMUNITY = "community"  # Find communities

    # Impact queries - analyze dependencies
    IMPACT = "impact"  # Analyze impact of changes
    DEPENDENCIES = "dependencies"  # Find dependencies
    CALLERS = "callers"  # Find what calls a function
    CALLEES = "callees"  # Find what a function calls

    # Aggregation queries
    COUNT = "count"  # Count nodes/edges
    AGGREGATE = "aggregate"  # Aggregate values
    STATISTICS = "statistics"  # Graph statistics

    # Custom queries
    CUSTOM = "custom"  # Custom query logic


class MatchStrategy(Enum):
    """Strategy for matching query to template."""

    EXACT = "exact"  # Exact keyword match
    KEYWORD = "keyword"  # Keyword-based matching
    SEMANTIC = "semantic"  # Semantic similarity
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class QueryParameter:
    """Parameter definition for a query template.

    Attributes:
        name: Parameter name
        type: Parameter type (string, int, list, etc.)
        required: Whether parameter is required
        default: Default value if not provided
        description: Parameter description
        validation: Optional validation regex or function
    """

    name: str
    type: str  # "string", "int", "list", "bool", "node_id", "edge_type"
    required: bool = True
    default: Any = None
    description: str = ""
    validation: Optional[Union[str, Any]] = None

    def validate(self, value: Any) -> bool:
        """Validate a parameter value.

        Args:
            value: Value to validate

        Returns:
            True if valid
        """
        if value is None:
            return not self.required

        # Type validation
        if self.type == "string":
            if not isinstance(value, str):
                return False
        elif self.type == "int":
            if not isinstance(value, int):
                return False
        elif self.type == "bool":
            if not isinstance(value, bool):
                return False
        elif self.type == "list":
            if not isinstance(value, (list, tuple, set)):
                return False
        elif self.type == "node_id":
            if not isinstance(value, str):
                return False

        # Regex validation if provided
        if self.validation and isinstance(self.validation, str):
            if isinstance(value, str):
                return bool(re.match(self.validation, value))

        return True


@dataclass
class QueryExample:
    """Example query for a template.

    Attributes:
        natural_language: Natural language query example
        parameters: Expected parameter values
        description: What this query does
    """

    natural_language: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class QueryTemplate:
    """Template for a graph query pattern.

    Attributes:
        name: Template name (unique identifier)
        query_type: Type of query this template generates
        description: Template description
        patterns: Regex patterns to match natural language
        keywords: Keywords for matching
        parameters: Parameter definitions
        template_string: String template for generating the query
        examples: Example queries
        priority: Priority for template matching (higher = preferred)
        enabled: Whether template is active
        metadata: Additional metadata
    """

    name: str
    query_type: QueryType
    description: str
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    parameters: List[QueryParameter] = field(default_factory=list)
    template_string: str = ""
    examples: List[QueryExample] = field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        query: str,
        strategy: MatchStrategy = MatchStrategy.HYBRID,
    ) -> float:
        """Check if this template matches the given query.

        Args:
            query: Natural language query
            strategy: Matching strategy to use

        Returns:
            Match score (0-1, higher is better)
        """
        query_lower = query.lower().strip()

        if not self.enabled:
            return 0.0

        score = 0.0

        # Pattern matching
        if strategy in (
            MatchStrategy.EXACT,
            MatchStrategy.KEYWORD,
            MatchStrategy.HYBRID,
        ):
            for pattern in self.patterns:
                try:
                    if re.search(pattern, query_lower, re.IGNORECASE):
                        score += 0.5
                        break
                except re.error:
                    pass

        # Keyword matching
        if strategy in (MatchStrategy.KEYWORD, MatchStrategy.HYBRID):
            keyword_matches = sum(1 for kw in self.keywords if kw.lower() in query_lower)
            if keyword_matches > 0:
                keyword_score = min(keyword_matches / len(self.keywords), 1.0)
                score += keyword_score * 0.5

        return min(score, 1.0)

    def extract_parameters(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Extract parameters from natural language query.

        Args:
            query: Natural language query

        Returns:
            Extracted parameters with defaults applied
        """
        params = {}

        # Set defaults
        for param_def in self.parameters:
            if param_def.default is not None:
                params[param_def.name] = param_def.default

        # Extract using regex patterns
        # Common extraction patterns
        # Extract function/class/module names
        name_patterns = [
            # "neighbors of X", "path to Y", "callers of Z", "calls X"
            r"(?:neighbors?|path|callers?|callees?|impact|dependencies?)\s+(?:of|to|for|in)\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)",
            r"what\s+(?:function\s+)?(?:does\s+)?['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?\s+(?:calls?|invokes?|uses?)",
            # "calls X" at end
            r"(?:calls?|invokes?|uses?)\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?$",
            # "for X" at end (for impact analysis)
            r"(?:for|in)\s+['\"]?([a-zA-Z_][a-zA-Z0-9_\.]+)['\"]?$",
            # "function X", "class Y"
            r"(?:function|class|method|module|symbol)\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)",
            # "find function X"
            r"find\s+(?:the\s+)?(?:function|class|method)\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)",
            # Quoted names
            r"['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
            # Function calls: "name("
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            # "from X to Y" pattern
            r"from\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?\s+to\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?",
        ]

        for pattern in name_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                # Handle different numbers of groups
                if len(groups) == 2 and groups[0] and groups[1]:
                    # "from X to Y" pattern
                    if "source" not in params:
                        params["source"] = groups[0]
                    if "target" not in params:
                        params["target"] = groups[1]
                elif groups[0]:
                    extracted_name = groups[0]
                    # Map to appropriate parameter name
                    if (
                        "node_id" not in params
                        and "function" not in params
                        and "target" not in params
                        and "source" not in params
                    ):
                        # Check which parameter this should map to
                        param_names = [p.name for p in self.parameters]
                        if "node_id" in param_names:
                            params["node_id"] = extracted_name
                        elif "function" in param_names:
                            params["function"] = extracted_name
                        elif "target" in param_names:
                            params["target"] = extracted_name
                        elif "source" in param_names:
                            params["source"] = extracted_name
                        else:
                            params["name"] = extracted_name
                break

        # Extract file paths - be more specific to avoid false matches
        file_patterns = [
            r"(?:in\s+)?file\s+['\"]?([a-zA-Z0-9_./\\]+)",
            r"(?:in|from)\s+['\"]?([a-zA-Z0-9_./\\]+)\.py",
            r"(?:in|from)\s+['\"]?([a-zA-Z0-9_./\\]+)\.js",
        ]

        for pattern in file_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["file"] = match.group(1)
                break

        # Extract numbers (for hops, depth, limit, etc.)
        number_patterns = [
            (r"(\d+)\s+hops?", "max_hops"),
            (r"(\d+)\s+deep?", "depth"),
            (r"top\s+(\d+)", "limit"),
            (r"first\s+(\d+)", "limit"),
            (r"limit\s+(\d+)", "limit"),
        ]

        for num_pattern, param_name in number_patterns:
            match = re.search(num_pattern, query, re.IGNORECASE)
            if match:
                params[param_name] = int(match.group(1))

        return params

    def validate_parameters(
        self,
        params: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate parameters against template definition.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for param_def in self.parameters:
            # Check required parameters
            if param_def.required and param_def.name not in params:
                errors.append(f"Required parameter '{param_def.name}' is missing")
                continue

            # Validate parameter if present
            if param_def.name in params:
                if not param_def.validate(params[param_def.name]):
                    errors.append(
                        f"Parameter '{param_def.name}' has invalid value: {params[param_def.name]}"
                    )

        return len(errors) == 0, errors

    def render(
        self,
        params: Dict[str, Any],
    ) -> str:
        """Render the template with parameters.

        Args:
            params: Parameters to render with

        Returns:
            Rendered query string
        """
        try:
            return self.template_string.format(**params)
        except KeyError as e:
            raise ValueError(f"Missing parameter for template: {e}")


# =============================================================================
# Template Registry (PH3-004)
# =============================================================================


class TemplateRegistry:
    """Registry for query templates.

    Provides template registration, discovery, and matching.
    """

    def __init__(self) -> None:
        """Initialize the template registry."""
        self._templates: Dict[str, QueryTemplate] = {}
        self._by_type: Dict[QueryType, List[str]] = {}

    def register(
        self,
        template: QueryTemplate,
    ) -> None:
        """Register a query template.

        Args:
            template: Template to register

        Raises:
            ValueError: If template name already exists
        """
        if template.name in self._templates:
            raise ValueError(f"Template '{template.name}' already registered")

        self._templates[template.name] = template

        # Index by type
        if template.query_type not in self._by_type:
            self._by_type[template.query_type] = []
        self._by_type[template.query_type].append(template.name)

        logger.debug(f"Registered query template: {template.name}")

    def unregister(
        self,
        name: str,
    ) -> None:
        """Unregister a query template.

        Args:
            name: Template name to unregister
        """
        if name in self._templates:
            template = self._templates[name]
            if template.query_type in self._by_type:
                self._by_type[template.query_type].remove(name)
            del self._templates[name]
            logger.debug(f"Unregistered query template: {name}")

    def get(
        self,
        name: str,
    ) -> Optional[QueryTemplate]:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template or None if not found
        """
        return self._templates.get(name)

    def find_by_type(
        self,
        query_type: QueryType,
    ) -> List[QueryTemplate]:
        """Find all templates of a given type.

        Args:
            query_type: Query type to filter by

        Returns:
            List of matching templates
        """
        names = self._by_type.get(query_type, [])
        return [self._templates[name] for name in names if name in self._templates]

    def match(
        self,
        query: str,
        strategy: MatchStrategy = MatchStrategy.HYBRID,
    ) -> Optional[Tuple[QueryTemplate, float]]:
        """Find the best matching template for a query.

        Args:
            query: Natural language query
            strategy: Matching strategy

        Returns:
            Tuple of (template, score) or None if no match
        """
        best_match: Optional[Tuple[QueryTemplate, float]] = None
        best_score = 0.0

        for template in self._templates.values():
            if not template.enabled:
                continue

            score = template.matches(query, strategy)

            # Apply priority boost
            score += template.priority * 0.1

            if score > best_score:
                best_score = score
                best_match = (template, score)

        # Only return if score is above threshold
        if best_match and best_score > 0.3:
            return best_match

        return None

    def list_all(
        self,
        enabled_only: bool = True,
    ) -> List[QueryTemplate]:
        """List all registered templates.

        Args:
            enabled_only: Only return enabled templates

        Returns:
            List of templates
        """
        templates = list(self._templates.values())
        if enabled_only:
            templates = [t for t in templates if t.enabled]
        return sorted(templates, key=lambda t: -t.priority)


# Global template registry
_global_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """Get the global template registry.

    Returns:
        Global TemplateRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = TemplateRegistry()
        _register_default_templates(_global_registry)
    return _global_registry


def _register_default_templates(registry: TemplateRegistry) -> None:
    """Register default query templates (PH3-002).

    Args:
        registry: Registry to register templates with
    """
    # Template: Find neighbors
    registry.register(
        QueryTemplate(
            name="find_neighbors",
            query_type=QueryType.NEIGHBORS,
            description="Find neighbors of a node in the graph",
            patterns=[
                r"(?:find|get|show|list)\s+(?:the\s+)?neighbors?\s+(?:of\s+)?",
                r"show\s+(?:the\s+)?connections?\s+(?:to|for|of)",
            ],
            keywords=["neighbors", "connected", "connections", "links", "adjacent"],
            parameters=[
                QueryParameter("node_id", "node_id", required=True),
                QueryParameter("direction", "string", required=False, default="out"),
                QueryParameter("edge_types", "list", required=False),
                QueryParameter("max_depth", "int", required=False, default=1),
            ],
            template_string="neighbors(node_id={node_id}, direction={direction})",
            examples=[
                QueryExample(
                    natural_language="Find neighbors of parse_json",
                    parameters={"node_id": "parse_json", "direction": "out"},
                    description="Find all nodes that parse_json connects to",
                ),
                QueryExample(
                    natural_language="Show connections to main",
                    parameters={"node_id": "main", "direction": "in"},
                    description="Find all nodes that connect to main",
                ),
            ],
            priority=10,
        )
    )

    # Template: Find path
    registry.register(
        QueryTemplate(
            name="find_path",
            query_type=QueryType.PATH,
            description="Find a path between two nodes",
            patterns=[
                r"(?:find|get)\s+(?:a\s+)?path\s+(?:between|from\s+\w+\s+to)",
                r"(?:how\s+to\s+)?(?:get|go)\s+from\s+.+\s+to",
            ],
            keywords=["path", "route", "way", "trace", "between"],
            parameters=[
                QueryParameter("source", "node_id", required=True),
                QueryParameter("target", "node_id", required=True),
                QueryParameter("max_hops", "int", required=False, default=5),
            ],
            template_string="path(source={source}, target={target}, max_hops={max_hops})",
            examples=[
                QueryExample(
                    natural_language="Find path from main to process_data",
                    parameters={"source": "main", "target": "process_data"},
                    description="Find the execution path from main to process_data",
                ),
            ],
            priority=8,
        )
    )

    # Template: Impact analysis
    registry.register(
        QueryTemplate(
            name="impact_analysis",
            query_type=QueryType.IMPACT,
            description="Analyze the impact of changing a node",
            patterns=[
                r"what\s+(?:would\s+)?(?:be\s+)?the\s+impact\s+(?:of\s+)?(?:changing|modifying)",
                r"what\s+(?:functions|modules|code)\s+(?:would\s+)?be\s+affected\s+by",
                r"impact\s+analysis\s+(?:for|of)",
            ],
            keywords=["impact", "affected", "depends", "breaks", "changes"],
            parameters=[
                QueryParameter("target", "node_id", required=True),
                QueryParameter("direction", "string", required=False, default="forward"),
                QueryParameter("max_depth", "int", required=False, default=3),
            ],
            template_string="impact(target={target}, direction={direction}, depth={max_depth})",
            examples=[
                QueryExample(
                    natural_language="What would be affected by changing parse_json",
                    parameters={"target": "parse_json", "direction": "forward"},
                    description="Find all code that depends on parse_json",
                ),
            ],
            priority=10,
        )
    )

    # Template: Semantic search
    registry.register(
        QueryTemplate(
            name="semantic_search",
            query_type=QueryType.SEMANTIC_SEARCH,
            description="Search for nodes by semantic similarity",
            patterns=[
                r"(?:find|search|look\s+for)\s+(?:functions?|classes?|code?|symbols?)\s+(?:about|related\s+to|that)",
                r"search\s+for\s+['\"]?[a-z\s]+['\"]?",
            ],
            keywords=["search", "find", "look for", "about", "related"],
            parameters=[
                QueryParameter("query", "string", required=True),
                QueryParameter("limit", "int", required=False, default=10),
                QueryParameter("node_types", "list", required=False),
            ],
            template_string="semantic_search(query={query}, limit={limit})",
            examples=[
                QueryExample(
                    natural_language="Find functions related to parsing JSON",
                    parameters={"query": "parsing JSON", "limit": 10},
                    description="Search for functions that handle JSON parsing",
                ),
            ],
            priority=9,
        )
    )

    # Template: Find callers
    registry.register(
        QueryTemplate(
            name="find_callers",
            query_type=QueryType.CALLERS,
            description="Find functions that call a given function",
            patterns=[
                r"what\s+(?:functions?|code)?\s+(?:calls?|invokes?|uses?)\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)",
                r"(?:find|get|list)\s+(?:the\s+)?callers?\s+(?:of\s+)?",
            ],
            keywords=["callers", "calls", "invoked by", "used by"],
            parameters=[
                QueryParameter("function", "node_id", required=True),
                QueryParameter("transitive", "bool", required=False, default=False),
            ],
            template_string="callers(function={function}, transitive={transitive})",
            examples=[
                QueryExample(
                    natural_language="What calls process_data",
                    parameters={"function": "process_data", "transitive": False},
                    description="Find direct callers of process_data",
                ),
            ],
            priority=10,
        )
    )

    # Template: Find callees
    registry.register(
        QueryTemplate(
            name="find_callees",
            query_type=QueryType.CALLEES,
            description="Find functions that a given function calls",
            patterns=[
                r"what\s+(?:function\s+)?(?:does\s+)?['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?\s+(?:calls?|invokes?|uses?)",
                r"(?:find|get|list)\s+(?:the\s+)?(?:callees?|dependencies?)\s+(?:of\s+)?",
            ],
            keywords=["callees", "calls", "invokes", "uses", "depends on", "what does"],
            parameters=[
                QueryParameter("function", "node_id", required=True),
                QueryParameter("transitive", "bool", required=False, default=False),
                QueryParameter("max_depth", "int", required=False, default=2),
            ],
            template_string="callees(function={function}, transitive={transitive}, depth={max_depth})",
            examples=[
                QueryExample(
                    natural_language="What does main call",
                    parameters={"function": "main", "transitive": False},
                    description="Find functions directly called by main",
                ),
            ],
            priority=10,
        )
    )

    # Template: Find similar
    registry.register(
        QueryTemplate(
            name="find_similar",
            query_type=QueryType.SIMILAR,
            description="Find nodes similar to a given node",
            patterns=[
                r"(?:find|get)\s+(?:functions?|code)?\s+(?:similar\s+to|like)",
                r"what\s+(?:else\s+)?is\s+(?:similar\s+to|like)",
            ],
            keywords=["similar", "like", "analogous", "related"],
            parameters=[
                QueryParameter("node_id", "node_id", required=True),
                QueryParameter("limit", "int", required=False, default=5),
            ],
            template_string="similar(node_id={node_id}, limit={limit})",
            examples=[
                QueryExample(
                    natural_language="Find functions similar to parse_json",
                    parameters={"node_id": "parse_json", "limit": 5},
                    description="Find functions with similar structure or purpose",
                ),
            ],
            priority=7,
        )
    )

    # Template: Count nodes
    registry.register(
        QueryTemplate(
            name="count_nodes",
            query_type=QueryType.COUNT,
            description="Count nodes matching criteria",
            patterns=[
                r"(?:how\s+many\s+)?(?:count)\s+(?:functions?|classes?|nodes?|symbols?)",
                r"number\s+of\s+(?:functions?|classes?|nodes?)",
            ],
            keywords=["count", "how many", "number of"],
            parameters=[
                QueryParameter("node_type", "string", required=False),
                QueryParameter("file", "string", required=False),
            ],
            template_string="count(node_type={node_type}, file={file})",
            examples=[
                QueryExample(
                    natural_language="How many functions in utils.py",
                    parameters={"node_type": "function", "file": "utils.py"},
                    description="Count functions in a specific file",
                ),
            ],
            priority=6,
        )
    )


# =============================================================================
# Translation Interface (PH3-005)
# =============================================================================


@dataclass
class TranslationResult:
    """Result of query translation.

    Attributes:
        original_query: Original natural language query
        matched_template: Template that was matched (if any)
        graph_query: Generated graph query
        parameters: Extracted parameters
        confidence: Confidence score (0-1)
        fallback: Whether translation fell back to keyword search
        errors: Any errors that occurred
        metadata: Additional metadata
    """

    original_query: str
    matched_template: Optional[QueryTemplate] = None
    graph_query: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    fallback: bool = False
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_successful(self) -> bool:
        """Check if translation was successful.

        Successful means: no errors AND one of a matched template, an explicit
        keyword-search fallback, or a non-empty query from another translation
        path (e.g. LLM translation, which has no template).
        """
        return len(self.errors) == 0 and (
            self.fallback or self.matched_template is not None or bool(self.graph_query)
        )


class QueryTranslator(abc.ABC):
    """Abstract interface for query translation (PH3-005).

    Implementations translate natural language queries into graph queries.
    """

    @abc.abstractmethod
    async def translate(
        self,
        query: str,
        graph_store: "GraphStoreProtocol",
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        """Translate a natural language query to a graph query.

        Args:
            query: Natural language query
            graph_store: Graph store for context
            context: Additional translation context

        Returns:
            TranslationResult with the translated query
        """
        ...

    @abc.abstractmethod
    def supports_batch(self) -> bool:
        """Check if translator supports batch translation."""
        ...

    async def translate_batch(
        self,
        queries: List[str],
        graph_store: "GraphStoreProtocol",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TranslationResult]:
        """Translate multiple queries (optional).

        Args:
            queries: List of natural language queries
            graph_store: Graph store for context
            context: Additional translation context

        Returns:
            List of TranslationResult objects
        """
        # Default implementation: translate sequentially
        results = []
        for query in queries:
            result = await self.translate(query, graph_store, context)
            results.append(result)
        return results


# =============================================================================
# Template-based Translator
# =============================================================================


class TemplateBasedTranslator(QueryTranslator):
    """Translator that uses query templates (PH3-002, PH3-004).

    Matches natural language queries against registered templates
    and extracts parameters to generate structured graph queries.
    """

    def __init__(
        self,
        registry: Optional[TemplateRegistry] = None,
        match_strategy: MatchStrategy = MatchStrategy.HYBRID,
    ) -> None:
        """Initialize the template-based translator.

        Args:
            registry: Template registry (uses global if None)
            match_strategy: Strategy for matching templates
        """
        self._registry = registry or get_template_registry()
        self._match_strategy = match_strategy

    async def translate(
        self,
        query: str,
        graph_store: "GraphStoreProtocol",
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        """Translate using template matching.

        Args:
            query: Natural language query
            graph_store: Graph store for context
            context: Additional translation context

        Returns:
            TranslationResult with translated query
        """
        result = TranslationResult(original_query=query)

        # Try to match a template
        match = self._registry.match(query, self._match_strategy)

        if match is None:
            # Fallback to keyword search (PH3-008)
            result.fallback = True
            result.graph_query = f"search(query={repr(query)})"
            result.parameters = {"query": query}
            result.confidence = 0.1
            result.metadata["fallback_reason"] = "No template matched"
            return result

        template, score = match
        result.matched_template = template
        result.confidence = score

        # Extract parameters
        params = template.extract_parameters(query)
        result.parameters = params

        # Validate parameters
        is_valid, errors = template.validate_parameters(params)
        if not is_valid:
            result.errors = errors
            return result

        # Render template
        try:
            result.graph_query = template.render(params)
        except ValueError as e:
            result.errors = [str(e)]

        return result

    def supports_batch(self) -> bool:
        """Template-based translator supports batch translation."""
        return True


# =============================================================================
# LLM-based Translator (PH3-006)
# =============================================================================


# Async completion function: prompt in, raw model text out. Injected by the
# consumer (tool, vertical, service) — core stays provider-agnostic.
LLMCompletionFn = Callable[[str], Awaitable[str]]


class LLMBasedTranslator(QueryTranslator):
    """Translator that uses an injected LLM completion to translate queries (PH3-006).

    The LLM is supplied as an async ``prompt -> text`` callable so this module
    never imports provider machinery; without one (the default), every call
    routes through the template-based fallback. Any LLM failure — call error,
    unparseable response, unknown query type — also falls back rather than
    surfacing on the query path.
    """

    #: Query types the prompt advertises; anything else in a response is rejected.
    _VALID_QUERY_TYPES = frozenset(
        {"neighbors", "path", "impact", "semantic_search", "callers", "callees"}
    )

    def __init__(
        self,
        fallback_translator: Optional[QueryTranslator] = None,
        llm: Optional[LLMCompletionFn] = None,
    ) -> None:
        """Initialize the LLM-based translator.

        Args:
            fallback_translator: Fallback if LLM unavailable
            llm: Async completion function (prompt -> raw response text).
                None (the default) means template-only operation — the old
                ``_llm_available = True`` stub claimed availability it never had.
        """
        self._fallback = fallback_translator or TemplateBasedTranslator()
        self._llm = llm
        self._llm_available = llm is not None

    async def translate(
        self,
        query: str,
        graph_store: "GraphStoreProtocol",
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        """Translate using the injected LLM, falling back to templates.

        Args:
            query: Natural language query
            graph_store: Graph store for context
            context: Additional translation context

        Returns:
            TranslationResult with translated query
        """
        if not self._llm_available or self._llm is None:
            result = await self._fallback.translate(query, graph_store, context)
            result.metadata["llm_used"] = False
            return result

        try:
            stats = await graph_store.stats()
        except Exception:
            stats = {}

        prompt = self._build_translation_prompt(query, stats, context)
        try:
            raw = await self._llm(prompt)
            translated = self._parse_llm_response(raw)
        except Exception as e:
            logger.debug(f"LLM translation failed, using template fallback: {e}")
            return await self._fallback_with_reason(query, graph_store, context, str(e))

        if translated is None:
            return await self._fallback_with_reason(
                query, graph_store, context, "unparseable LLM response"
            )

        query_type, params = translated
        rendered_params = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        result = TranslationResult(
            original_query=query,
            graph_query=f"{query_type}({rendered_params})",
            parameters=params,
            confidence=0.9,
        )
        result.metadata["llm_used"] = True
        result.metadata["query_type"] = query_type
        return result

    async def _fallback_with_reason(
        self,
        query: str,
        graph_store: "GraphStoreProtocol",
        context: Optional[Dict[str, Any]],
        reason: str,
    ) -> TranslationResult:
        result = await self._fallback.translate(query, graph_store, context)
        result.metadata["llm_used"] = False
        result.metadata["llm_error"] = reason
        return result

    def _parse_llm_response(self, raw: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Extract ``(query_type, params)`` from a model response.

        Accepts bare JSON, fenced JSON, or JSON embedded in prose. Returns
        None for anything that doesn't yield a known query type with a dict of
        params — the caller falls back rather than guessing.
        """
        import json

        candidates = [raw.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1))
        embedded = re.search(r"\{.*\}", raw, re.DOTALL)
        if embedded:
            candidates.append(embedded.group(0))

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            query_type = data.get("type")
            params = data.get("params", {})
            if query_type in self._VALID_QUERY_TYPES and isinstance(params, dict):
                return str(query_type), params
        return None

    def _build_translation_prompt(
        self,
        query: str,
        graph_stats: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt for LLM translation.

        Args:
            query: Natural language query
            graph_stats: Graph statistics
            context: Additional context

        Returns:
            Prompt string
        """
        return f"""Translate this natural language query into a structured graph query.

Query: {query}

Available graph information:
- nodes: {graph_stats.get('nodes', 'unknown')}
- edges: {graph_stats.get('edges', 'unknown')}

Supported query types:
- neighbors: Find connected nodes
- path: Find path between nodes
- impact: Analyze impact of changes
- semantic_search: Search by meaning
- callers: Find what calls a function
- callees: Find what a function calls

Return JSON with: {{"type": "query_type", "params": {{"key": "value"}}}}
"""

    def supports_batch(self) -> bool:
        """LLM-based translator supports batch translation."""
        return True


# =============================================================================
# Public API
# =============================================================================


async def translate_query(
    query: str,
    graph_store: "GraphStoreProtocol",
    translator: Optional[QueryTranslator] = None,
    context: Optional[Dict[str, Any]] = None,
) -> TranslationResult:
    """Translate a natural language query to a graph query.

    This is the main entry point for query translation.

    Args:
        query: Natural language query
        graph_store: Graph store for context
        translator: Custom translator (uses default if None)
        context: Additional translation context

    Returns:
        TranslationResult with the translated query

    Example:
        result = await translate_query(
            "Find all functions that call parse_json",
            graph_store=store,
        )
        if result.is_successful():
            print(f"Graph query: {result.graph_query}")
    """
    if translator is None:
        # Use LLM-based with template fallback
        translator = LLMBasedTranslator()

    return await translator.translate(query, graph_store, context)


def register_template(template: QueryTemplate) -> None:
    """Register a query template.

    Args:
        template: Template to register
    """
    registry = get_template_registry()
    registry.register(template)


def list_templates(
    query_type: Optional[QueryType] = None,
    enabled_only: bool = True,
) -> List[QueryTemplate]:
    """List available query templates.

    Args:
        query_type: Optional filter by query type
        enabled_only: Only return enabled templates

    Returns:
        List of templates
    """
    registry = get_template_registry()

    if query_type:
        return registry.find_by_type(query_type)

    return registry.list_all(enabled_only=enabled_only)


__all__ = [
    # Schema
    "QueryType",
    "MatchStrategy",
    "QueryParameter",
    "QueryExample",
    "QueryTemplate",
    # Results
    "TranslationResult",
    # Registry
    "TemplateRegistry",
    "get_template_registry",
    # Translators
    "QueryTranslator",
    "TemplateBasedTranslator",
    "LLMBasedTranslator",
    # API
    "translate_query",
    "register_template",
    "list_templates",
]
