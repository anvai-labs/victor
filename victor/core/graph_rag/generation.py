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

"""Graph-aware prompt building - G-Generation stage of Graph RAG.

This module implements the third stage of the Graph RAG pipeline:
constructing prompts with graph context for LLM consumption.

Based on GraphRAG methodology:
1. Format subgraphs as structured text
2. Preserve hierarchy and relationships
3. Include edge type annotations
4. Truncate to fit context window
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.storage.graph.protocol import (
        GraphEdge,
        GraphNode,
        Subgraph,
    )

logger = logging.getLogger(__name__)


@dataclass
class FormattedContext:
    """Formatted graph context for inclusion in prompts.

    Attributes:
        text: Formatted text representation
        sections: Individual context sections
        metadata: Metadata about the formatting
        token_estimate: Estimated token count
    """

    text: str
    sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0


@dataclass
class ContextSection:
    """A section of context to include in the prompt.

    Attributes:
        title: Section title
        nodes: Nodes in this section
        edges: Edges in this section
        priority: Priority for inclusion (higher = more important)
    """

    title: str
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    priority: float = 0.5


class GraphAwarePromptBuilder:
    """Build prompts with graph context for LLM consumption.

    This class formats graph context into structured text that can be
    included in LLM prompts. It handles:

    - Hierarchical formatting (functions contain methods, etc.)
    - Edge type annotations (CALLS, REFERENCES, etc.)
    - File location information
    - Truncation for context window limits

    Attributes:
        config: Prompt building configuration
    """

    def __init__(self, config: Any) -> None:
        """Initialize the prompt builder.

        Args:
            config: PromptConfig instance
        """
        self.config = config

    def build_prompt(
        self,
        query: str,
        retrieval_result: Any,  # RetrievalResult
        max_tokens: int = 8000,
    ) -> str:
        """Build a prompt with graph context.

        Args:
            query: User's query
            retrieval_result: Result from MultiHopRetriever
            max_tokens: Maximum tokens for context

        Returns:
            Formatted prompt string
        """
        # Format the graph context
        context = self.format_context(retrieval_result, max_tokens)

        # Build the full prompt
        prompt = self._apply_template(query, context)

        return prompt

    def format_context(
        self,
        retrieval_result: Any,  # RetrievalResult
        max_tokens: int = 8000,
    ) -> FormattedContext:
        """Format retrieval result into structured context.

        Args:
            retrieval_result: Result from MultiHopRetriever
            max_tokens: Maximum tokens for context

        Returns:
            FormattedContext with structured text
        """
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        hop_distances = getattr(retrieval_result, "hop_distances", {})

        # Group nodes by type and file
        sections = self._organize_sections(nodes, edges, hop_distances)

        # Format each section
        formatted_sections: Dict[str, str] = {}
        total_text = ""

        for section_name, section in sections.items():
            formatted = self._format_section(section)
            formatted_sections[section_name] = formatted

            # Check token limit
            current_estimate = self._estimate_tokens(total_text + formatted)
            if current_estimate > max_tokens:
                break

            total_text += formatted + self.config.context_separator

        return FormattedContext(
            text=total_text.strip(),
            sections=formatted_sections,
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges),
                "section_count": len(formatted_sections),
            },
            token_estimate=self._estimate_tokens(total_text),
        )

    def _organize_sections(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        hop_distances: Dict[str, int],
    ) -> Dict[str, ContextSection]:
        """Organize nodes into context sections.

        Args:
            nodes: Graph nodes
            edges: Graph edges
            hop_distances: Hop distances for nodes

        Returns:
            Dictionary of section name to ContextSection
        """
        sections: Dict[str, ContextSection] = {}

        # Group by type
        type_groups: Dict[str, List[GraphNode]] = {}
        for node in nodes:
            node_type = node.type or "unknown"
            if node_type not in type_groups:
                type_groups[node_type] = []
            type_groups[node_type].append(node)

        # Create sections
        for node_type, type_nodes in type_groups.items():
            section_name = self._get_section_name(node_type)
            sections[section_name] = ContextSection(
                title=section_name,
                nodes=type_nodes,
                priority=self._get_section_priority(node_type, hop_distances, type_nodes),
            )

        # Add edges to relevant sections
        for edge in edges:
            edge_type = edge.type
            section_name = self._get_edge_section_name(edge_type)
            if section_name not in sections:
                sections[section_name] = ContextSection(title=section_name)
            sections[section_name].edges.append(edge)

        return sections

    def _get_section_name(self, node_type: str) -> str:
        """Get section name for a node type.

        Args:
            node_type: Node type string

        Returns:
            Section name
        """
        section_map = {
            "function": "Functions",
            "method": "Methods",
            "class": "Classes",
            "statement": "Statements",
            "variable": "Variables",
            "module": "Modules",
            "file": "Files",
            "interface": "Interfaces",
            "type_alias": "Type Aliases",
        }
        return section_map.get(node_type, f"{node_type.capitalize()}s")

    def _get_edge_section_name(self, edge_type: str) -> str:
        """Get section name for an edge type.

        Args:
            edge_type: Edge type string

        Returns:
            Section name
        """
        from victor.storage.graph.edge_types import EdgeType

        if EdgeType.is_cfg_edge(edge_type):
            return "Control Flow"
        if EdgeType.is_cdg_edge(edge_type):
            return "Control Dependencies"
        if EdgeType.is_ddg_edge(edge_type):
            return "Data Dependencies"
        if edge_type in {EdgeType.CALLS, "CALLS"}:
            return "Call Graph"
        if edge_type in {EdgeType.REFERENCES, "REFERENCES"}:
            return "References"

        return "Relationships"

    def _get_section_priority(
        self,
        node_type: str,
        hop_distances: Dict[str, int],
        nodes: List[GraphNode],
    ) -> float:
        """Calculate priority for a section.

        Args:
            node_type: Node type
            hop_distances: Hop distances
            nodes: Nodes in section

        Returns:
            Priority score (0-1)
        """
        # Base priority by type
        type_priority = {
            "function": 0.9,
            "method": 0.9,
            "class": 0.8,
            "interface": 0.7,
            "statement": 0.5,
        }
        base = type_priority.get(node_type, 0.5)

        # Boost if contains close nodes (low hop distance)
        if nodes:
            avg_distance = sum(hop_distances.get(n.node_id, 99) for n in nodes) / len(nodes)
            proximity_boost = max(0, 1 - avg_distance / 3)
            return base * 0.7 + proximity_boost * 0.3

        return base

    def _format_section(self, section: ContextSection) -> str:
        """Format a context section into text.

        Args:
            section: ContextSection to format

        Returns:
            Formatted text
        """
        lines = [f"## {section.title}\n"]

        # Format nodes
        if section.nodes:
            for node in self._sort_nodes(section.nodes):
                node_text = self._format_node(node)
                lines.append(node_text)

        # Format edges
        if section.edges and self.config.include_edge_types:
            lines.append("\n### Relationships\n")
            for edge in section.edges[:20]:  # Limit edges
                edge_text = self._format_edge(edge)
                lines.append(edge_text)

        return "\n".join(lines)

    def _sort_nodes(self, nodes: List[GraphNode]) -> List[GraphNode]:
        """Sort nodes for display.

        Args:
            nodes: Nodes to sort

        Returns:
            Sorted nodes
        """
        # Sort by file, then line number
        return sorted(
            nodes,
            key=lambda n: (n.file or "", n.line or 0),
        )

    def _format_node(self, node: GraphNode) -> str:
        """Format a single node for display.

        Args:
            node: GraphNode to format

        Returns:
            Formatted text
        """
        parts = []

        # File and location
        if self.config.include_file_paths:
            location = f"{node.file}"
            if node.line:
                location += f":{node.line}"
            parts.append(f"**{location}**")

        # Name and type
        name_part = f"{node.type}: `{node.name}`"
        if node.signature:
            name_part += f" — `{node.signature}`"
        parts.append(name_part)

        # Docstring
        if node.docstring:
            doc_preview = node.docstring[:100]
            if len(node.docstring) > 100:
                doc_preview += "..."
            parts.append(f"> {doc_preview}")

        # Indicators
        if self.config.hierarchy_indicators:
            indicators = []
            if node.visibility:
                indicators.append(node.visibility)
            if node.statement_type:
                indicators.append(node.statement_type)
            if indicators:
                parts.append(f"*[{', '.join(indicators)}]*")

        return " ".join(parts)

    def _format_edge(self, edge: GraphEdge) -> str:
        """Format an edge for display.

        Args:
            edge: GraphEdge to format

        Returns:
            Formatted text
        """
        return f"- `{edge.src[:12]}` → `{edge.dst[:12]}` ({edge.type})"

    def _apply_template(self, query: str, context: FormattedContext) -> str:
        """Apply the prompt template.

        Args:
            query: User's query
            context: Formatted context

        Returns:
            Complete prompt
        """
        template = self.config.get_format_template()

        # For hierarchical format, organize by categories
        if self.config.format_style == "hierarchical":
            direct_matches = self._extract_direct_matches(context)
            impact_analysis = self._extract_impact_analysis(context)
            related_symbols = self._extract_related_symbols(context)
            data_flow = self._extract_data_flow(context)
            control_flow = self._extract_control_flow(context)

            return template.format(
                direct_matches=direct_matches,
                impact_analysis=impact_analysis,
                related_symbols=related_symbols,
                data_flow=data_flow,
                control_flow=control_flow,
            )
        elif self.config.format_style == "flat":
            all_symbols = context.text
            return template.format(all_symbols=all_symbols)
        else:  # compact
            compact_context = self._make_compact(context)
            return template.format(compact_context=compact_context)

    # Sections whose names are derived from impact-propagating edge types (see
    # _get_section_name): callers (CALLS -> "Call Graph") plus control- and
    # data-dependence dependents (CDG/DDG). A change to a node propagates to all of
    # these, so all are regression-relevant — not just callers.
    _IMPACT_SECTION_NAMES = ("Call Graph", "Control Dependencies", "Data Dependencies")

    def _extract_impact_analysis(self, context: FormattedContext) -> str:
        """Extract impact/regression information driven by the graph edge vocabulary.

        Impact is selected from the edge-type-derived sections rather than ad hoc string
        matching: callers + control/data-dependence dependents capture how a change
        propagates, and any test/spec section captures the suites that exercise the node
        (matched case-insensitively so "UnitTests"/"regression_tests" are not missed).

        Args:
            context: Formatted context

        Returns:
            Formatted impact analysis section
        """
        lines = []
        for section_name, section_text in context.sections.items():
            lowered = section_name.lower()
            is_test_section = "test" in lowered or "spec" in lowered
            if section_name in self._IMPACT_SECTION_NAMES or is_test_section:
                lines.append(section_text)
        if lines:
            return "\n\n".join(lines)
        # Be explicit that an empty result means "nothing indexed/retrieved", NOT "safe".
        return (
            "No impacted callers, dependents, or tests were found in the graph context "
            "(none were indexed or retrieved — this is not a guarantee the change is safe)."
        )

    def _extract_direct_matches(self, context: FormattedContext) -> str:
        """Extract directly matching symbols.

        Args:
            context: Formatted context

        Returns:
            Formatted direct matches section
        """
        # Filter nodes with low hop distance (direct matches)
        lines = []
        for section_name, section_text in context.sections.items():
            if section_name in {"Functions", "Methods", "Classes"}:
                lines.append(section_text)
        return "\n\n".join(lines) if lines else "No direct matches found."

    def _extract_related_symbols(self, context: FormattedContext) -> str:
        """Extract related symbols.

        Args:
            context: Formatted context

        Returns:
            Formatted related symbols section
        """
        lines = []
        for section_name, section_text in context.sections.items():
            if (
                section_name
                not in {
                    "Functions",
                    "Methods",
                    "Classes",
                    "Control Flow",
                    "Data Dependencies",
                    "Call Graph",
                }
                and "Test" not in section_name
            ):
                lines.append(section_text)
        return "\n\n".join(lines) if lines else "No related symbols found."

    def _extract_data_flow(self, context: FormattedContext) -> str:
        """Extract data flow information.

        Args:
            context: Formatted context

        Returns:
            Formatted data flow section
        """
        if "Data Dependencies" in context.sections:
            return context.sections["Data Dependencies"]
        return "No data flow information available."

    def _extract_control_flow(self, context: FormattedContext) -> str:
        """Extract control flow information.

        Args:
            context: Formatted context

        Returns:
            Formatted control flow section
        """
        if "Control Flow" in context.sections:
            return context.sections["Control Flow"]
        if "Control Dependencies" in context.sections:
            return context.sections["Control Dependencies"]
        return "No control flow information available."

    def _make_compact(self, context: FormattedContext) -> str:
        """Create a compact representation of context.

        Args:
            context: Formatted context

        Returns:
            Compact context string
        """
        # Limit to top symbols
        lines = context.text.split("\n")[:50]
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4


__all__ = ["GraphAwarePromptBuilder", "FormattedContext", "ContextSection"]
