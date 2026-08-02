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

"""Workflow DAG Visualization - State of the Art Rendering.

Supports multiple best-in-class rendering backends:

1. **D2 (Terrastruct)** - Modern diagram scripting language with multiple
   layout engines (Dagre, ELK, TALA). Best for software architecture.
   Install: brew install d2 | go install oss.terrastruct.com/d2@latest

2. **Kroki** - Unified API supporting 25+ diagram types. Can use free
   cloud (kroki.io) or self-hosted. No local install needed.

3. **Graphviz** - Classic DOT language, excellent for DAGs.
   Install: brew install graphviz | apt install graphviz

4. **Mermaid** - Markdown-friendly, renders via mermaid-cli or browser.
   Install: npm install -g @mermaid-js/mermaid-cli

5. **PlantUML** - Enterprise-standard UML diagrams.
   Install: brew install plantuml | apt install plantuml

6. **ASCII** - Terminal-friendly, no dependencies.

7. **NetworkX + Matplotlib** - Python-native, interactive.
   Install: pip install matplotlib networkx

Backend Selection Priority (auto mode):
  D2 > Kroki > Graphviz > Mermaid > NetworkX > ASCII

Renders compiled workflows as visual DAGs in multiple formats:
- ASCII art (terminal-friendly)
- Mermaid (markdown, renders to PNG/SVG via mermaid-cli)
- PlantUML (renders to PNG/SVG via plantuml)
- DOT/Graphviz (renders to PNG/SVG via graphviz)
- SVG (direct rendering via matplotlib if available)

Example:
    from victor.workflows import load_workflow_from_file
    from victor.workflows.visualization import WorkflowVisualizer

    workflow = load_workflow_from_file("workflow.yaml")
    viz = WorkflowVisualizer(workflow)

    # Print ASCII art to terminal
    print(viz.to_ascii())

    # Generate Mermaid markdown
    mermaid = viz.to_mermaid()

    # Save as SVG (requires matplotlib)
    viz.to_svg("workflow.svg")

    # Save as PNG (requires graphviz)
    viz.to_png("workflow.png")
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import base64
import zlib
import shutil

if TYPE_CHECKING:
    from victor.workflows.definition import WorkflowDefinition

logger = logging.getLogger(__name__)


def _check_command(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def _detect_best_backend() -> "RenderBackend":
    """Detect the best available rendering backend."""
    # Priority order: D2 > matplotlib > Kroki > Graphviz
    if _check_command("d2"):
        return RenderBackend.D2
    try:
        import matplotlib
        import networkx

        return RenderBackend.MATPLOTLIB
    except ImportError:
        pass
    # Fallback to Kroki (cloud API)
    return RenderBackend.KROKI


class OutputFormat(str, Enum):
    """Supported output formats for DAG visualization."""

    ASCII = "ascii"
    MERMAID = "mermaid"
    PLANTUML = "plantuml"
    DOT = "dot"
    D2 = "d2"  # Modern diagram language
    SVG = "svg"
    PNG = "png"


class RenderBackend(str, Enum):
    """Available rendering backends."""

    AUTO = "auto"  # Auto-detect best available
    D2 = "d2"  # Terrastruct D2
    KROKI = "kroki"  # Kroki unified API
    GRAPHVIZ = "graphviz"  # Classic DOT
    MERMAID_CLI = "mermaid-cli"  # mermaid-cli (mmdc)
    PLANTUML = "plantuml"  # PlantUML
    MATPLOTLIB = "matplotlib"  # NetworkX + Matplotlib
    ASCII = "ascii"  # No dependencies


# Kroki configuration
KROKI_URL = "https://kroki.io"  # Free public instance


def _validated_kroki_request(url: str, accept: str) -> Request:
    """Build a Kroki HTTP request, asserting the URL is https on the configured
    Kroki host. This prevents the rendered URL from being redirected to an
    unexpected scheme/host (SSRF / local-file access) before it is fetched.
    """
    parsed = urlparse(url)
    expected = urlparse(KROKI_URL)
    if parsed.scheme != "https" or parsed.hostname != expected.hostname:
        raise ValueError(f"Refusing to fetch non-Kroki/insecure URL: {url!r}")
    return Request(url, headers={"Accept": accept})


@dataclass
class NodeStyle:
    """Visual style for a node type."""

    shape: str  # mermaid/dot shape
    color: str  # fill color
    icon: str  # ASCII icon
    border: str  # border style


# Node type styling
NODE_STYLES: Dict[str, NodeStyle] = {
    "AgentNode": NodeStyle(
        shape="([{}])",  # stadium shape in mermaid
        color="#E3F2FD",  # light blue
        icon="@",
        border="bold",
    ),
    "ComputeNode": NodeStyle(
        shape="[/{}\\]",  # parallelogram
        color="#E8F5E9",  # light green
        icon="#",
        border="normal",
    ),
    "ConditionNode": NodeStyle(
        shape="{{}}",  # diamond
        color="#FFF3E0",  # light orange
        icon="?",
        border="normal",
    ),
    "ParallelNode": NodeStyle(
        shape="[[{}]]",  # subroutine
        color="#F3E5F5",  # light purple
        icon="=",
        border="double",
    ),
    "TransformNode": NodeStyle(
        shape="[{}]",  # rectangle
        color="#ECEFF1",  # light gray
        icon=">",
        border="normal",
    ),
    "HITLNode": NodeStyle(
        shape="(({}))",  # circle
        color="#FFEBEE",  # light red
        icon="!",
        border="bold",
    ),
}

DEFAULT_STYLE = NodeStyle(
    shape="[{}]",
    color="#FFFFFF",
    icon="*",
    border="normal",
)


@dataclass
class DAGNode:
    """Represents a node in the DAG."""

    id: str
    name: str
    node_type: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGEdge:
    """Represents an edge in the DAG."""

    source: str
    target: str
    label: Optional[str] = None
    conditional: bool = False


class WorkflowVisualizer:
    """Visualizes workflow definitions as DAGs.

    Supports multiple output formats for different use cases:
    - ASCII: Quick terminal visualization
    - Mermaid: Markdown-embeddable, renders via mermaid-cli
    - PlantUML: Enterprise documentation
    - DOT: Graphviz integration
    - SVG/PNG: Direct image output
    """

    def __init__(self, workflow: "WorkflowDefinition"):
        """Initialize visualizer with a workflow.

        Args:
            workflow: The workflow definition to visualize
        """
        self.workflow = workflow
        self._nodes: List[DAGNode] = []
        self._edges: List[DAGEdge] = []
        self._build_graph()

    def _build_graph(self) -> None:
        """Build internal graph representation from workflow."""
        from victor.workflows.definition import (
            AgentNode,
            ComputeNode,
            ConditionNode,
            ParallelNode,
            TransformNode,
        )

        # Try to import HITLNode for enhanced metadata
        try:
            from victor.workflows.hitl import HITLNode

            has_hitl = True
        except ImportError:
            HITLNode = None
            has_hitl = False

        # Build nodes
        for node_id, node in self.workflow.nodes.items():
            node_type = type(node).__name__
            description = None
            metadata: Dict[str, Any] = {}

            if isinstance(node, AgentNode):
                description = (
                    node.goal[:50] + "..." if node.goal and len(node.goal) > 50 else node.goal
                )
                metadata["role"] = node.role
            elif isinstance(node, ComputeNode):
                if node.handler:
                    metadata["handler"] = node.handler
                if node.tools:
                    metadata["tools"] = node.tools[:3]  # First 3 tools
                # Extract execution target
                if hasattr(node, "execution_target") and node.execution_target:
                    metadata["exec"] = node.execution_target
                # Extract constraints for visualization
                if hasattr(node, "constraints") and node.constraints:
                    constraints = node.constraints
                    blocked = []
                    if hasattr(constraints, "llm_allowed") and not constraints.llm_allowed:
                        blocked.append("llm")
                    if hasattr(constraints, "network_allowed") and not constraints.network_allowed:
                        blocked.append("network")
                    if hasattr(constraints, "write_allowed") and not constraints.write_allowed:
                        blocked.append("write")
                    if blocked:
                        metadata["blocked"] = blocked
                if hasattr(node, "timeout") and node.timeout:
                    metadata["timeout"] = node.timeout
            elif isinstance(node, ParallelNode):
                metadata["parallel_nodes"] = node.parallel_nodes
                metadata["join_strategy"] = node.join_strategy
            elif has_hitl and isinstance(node, HITLNode):
                metadata["hitl_type"] = node.hitl_type.value
                metadata["timeout"] = node.timeout
            elif hasattr(node, "description"):
                desc = getattr(node, "description", None)
                if desc:
                    description = desc[:50] + "..." if len(desc) > 50 else desc

            self._nodes.append(
                DAGNode(
                    id=node_id,
                    name=node.name or node_id,
                    node_type=node_type,
                    description=description,
                    metadata=metadata,
                )
            )

        # Build edges
        for node_id, node in self.workflow.nodes.items():
            if isinstance(node, ConditionNode):
                # Conditional edges
                for branch, target in node.branches.items():
                    if target:
                        self._edges.append(
                            DAGEdge(
                                source=node_id,
                                target=target,
                                label=branch,
                                conditional=True,
                            )
                        )
            elif isinstance(node, ParallelNode):
                # Parallel fork edges (dashed to show parallel execution)
                for parallel_target in node.parallel_nodes:
                    if parallel_target in self.workflow.nodes:
                        self._edges.append(
                            DAGEdge(
                                source=node_id,
                                target=parallel_target,
                                label="fork",
                                conditional=False,
                            )
                        )
                # Also add next_nodes for join
                if node.next_nodes:
                    for target in node.next_nodes:
                        self._edges.append(
                            DAGEdge(
                                source=node_id,
                                target=target,
                                label="join",
                                conditional=False,
                            )
                        )
            elif hasattr(node, "next_nodes") and node.next_nodes:
                for target in node.next_nodes:
                    self._edges.append(DAGEdge(source=node_id, target=target))

    def to_ascii(self, max_width: int = 80) -> str:
        """Render workflow as ASCII art.

        Args:
            max_width: Maximum width of output

        Returns:
            ASCII art representation of the DAG
        """
        lines = []
        lines.append("=" * max_width)
        lines.append(f" WORKFLOW: {self.workflow.name}")
        if self.workflow.description:
            lines.append(f" {self.workflow.description[:max_width-2]}")
        lines.append("=" * max_width)
        lines.append("")

        # Build adjacency for level calculation
        adjacency: Dict[str, List[str]] = {n.id: [] for n in self._nodes}
        in_degree: Dict[str, int] = {n.id: 0 for n in self._nodes}

        for edge in self._edges:
            if edge.target in adjacency:
                adjacency[edge.source].append(edge.target)
                in_degree[edge.target] += 1

        # Topological levels
        levels: Dict[str, int] = {}
        queue = [n for n in in_degree if in_degree[n] == 0]
        if not queue and self.workflow.start_node:
            queue = [self.workflow.start_node]

        current_level = 0
        while queue:
            next_queue = []
            for node_id in queue:
                levels[node_id] = current_level
                for target in adjacency.get(node_id, []):
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        next_queue.append(target)
            queue = next_queue
            current_level += 1

        # Group nodes by level
        level_nodes: Dict[int, List[DAGNode]] = {}
        for node in self._nodes:
            lvl = levels.get(node.id, 0)
            if lvl not in level_nodes:
                level_nodes[lvl] = []
            level_nodes[lvl].append(node)

        # Render each level
        for lvl in sorted(level_nodes.keys()):
            nodes = level_nodes[lvl]

            # Node boxes
            for node in nodes:
                style = NODE_STYLES.get(node.node_type, DEFAULT_STYLE)
                icon = style.icon

                # Build node box
                label = f"[{icon}] {node.id}"
                if node.node_type != "TransformNode":
                    label += f" ({node.node_type.replace('Node', '')})"

                box_width = min(len(label) + 4, max_width - 4)
                indent = "  " * lvl

                lines.append(f"{indent}+{'-' * box_width}+")
                lines.append(f"{indent}| {label:<{box_width-2}} |")
                if node.description:
                    desc = node.description[: box_width - 4]
                    lines.append(f"{indent}| {desc:<{box_width-2}} |")
                lines.append(f"{indent}+{'-' * box_width}+")

            # Edges to next level
            for node in nodes:
                for edge in self._edges:
                    if edge.source == node.id:
                        indent = "  " * lvl
                        arrow = f"--[{edge.label}]-->" if edge.label else "--->"
                        lines.append(f"{indent}    {arrow} {edge.target}")
            lines.append("")

        # Legend
        lines.append("-" * max_width)
        lines.append(" Legend:")
        for node_type, style in NODE_STYLES.items():
            lines.append(f"   [{style.icon}] = {node_type.replace('Node', '')}")
        lines.append("-" * max_width)

        return "\n".join(lines)

    def to_mermaid(self, direction: str = "TD") -> str:
        """Render workflow as Mermaid diagram.

        Args:
            direction: Graph direction (TD=top-down, LR=left-right)

        Returns:
            Mermaid markdown string
        """
        lines = [f"graph {direction}"]

        # Add title as comment
        lines.append(f"    %% Workflow: {self.workflow.name}")
        if self.workflow.description:
            lines.append(f"    %% {self.workflow.description}")
        lines.append("")

        # Define nodes with shapes
        for node in self._nodes:
            style = NODE_STYLES.get(node.node_type, DEFAULT_STYLE)
            label = node.name or node.id

            # Mermaid shape syntax
            if node.node_type == "ConditionNode":
                shape = f"{{{{{label}}}}}"  # diamond
            elif node.node_type == "AgentNode":
                shape = f"([{label}])"  # stadium
            elif node.node_type == "ParallelNode":
                shape = f"[[{label}]]"  # subroutine
            elif node.node_type == "HITLNode":
                shape = f"(({label}))"  # circle
            elif node.node_type == "ComputeNode":
                shape = f"[/{label}\\]"  # parallelogram
            else:
                shape = f"[{label}]"  # rectangle

            lines.append(f"    {node.id}{shape}")

        lines.append("")

        # Define edges
        for edge in self._edges:
            if edge.label:
                lines.append(f"    {edge.source} -->|{edge.label}| {edge.target}")
            else:
                lines.append(f"    {edge.source} --> {edge.target}")

        lines.append("")

        # Add styling
        lines.append("    %% Styling")
        for node in self._nodes:
            style = NODE_STYLES.get(node.node_type, DEFAULT_STYLE)
            lines.append(f"    style {node.id} fill:{style.color}")

        return "\n".join(lines)

    def to_plantuml(self) -> str:
        """Render workflow as PlantUML activity diagram.

        Returns:
            PlantUML string
        """
        lines = ["@startuml"]
        lines.append(f"title {self.workflow.name}")
        if self.workflow.description:
            lines.append(f"note right: {self.workflow.description}")
        lines.append("")

        # Skin parameters for styling
        lines.append("skinparam activity {")
        lines.append("    BackgroundColor<<Agent>> #E3F2FD")
        lines.append("    BackgroundColor<<Compute>> #E8F5E9")
        lines.append("    BackgroundColor<<Condition>> #FFF3E0")
        lines.append("    BackgroundColor<<Parallel>> #F3E5F5")
        lines.append("    BackgroundColor<<Transform>> #ECEFF1")
        lines.append("    BackgroundColor<<HITL>> #FFEBEE")
        lines.append("}")
        lines.append("")

        lines.append("start")
        lines.append("")

        # Build adjacency for traversal
        visited: Set[str] = set()

        def render_node(node_id: str, indent: str = "") -> None:
            if node_id in visited:
                lines.append(f"{indent}:{node_id};")
                return
            visited.add(node_id)

            node = next((n for n in self._nodes if n.id == node_id), None)
            if not node:
                return

            stereotype = node.node_type.replace("Node", "")

            if node.node_type == "ConditionNode":
                lines.append(f"{indent}if ({node.name}) then")
                edges = [e for e in self._edges if e.source == node_id]
                for i, edge in enumerate(edges):
                    if i > 0:
                        lines.append(f"{indent}elseif ({edge.label or 'else'}) then")
                    else:
                        lines.append(f"{indent}  -[{edge.label or 'yes'}]->")
                    render_node(edge.target, indent + "  ")
                lines.append(f"{indent}endif")
            elif node.node_type == "ParallelNode":
                lines.append(f"{indent}fork")
                edges = [e for e in self._edges if e.source == node_id]
                for i, edge in enumerate(edges):
                    if i > 0:
                        lines.append(f"{indent}fork again")
                    render_node(edge.target, indent + "  ")
                lines.append(f"{indent}end fork")
            else:
                lines.append(f"{indent}:{node.name}; <<{stereotype}>>")
                edges = [e for e in self._edges if e.source == node_id]
                for edge in edges:
                    render_node(edge.target, indent)

        if self.workflow.start_node:
            render_node(self.workflow.start_node)

        lines.append("")
        lines.append("stop")
        lines.append("@enduml")

        return "\n".join(lines)

    def to_d2(self) -> str:
        """Render workflow as D2 (Terrastruct) format.

        D2 is a modern diagram scripting language with excellent
        layout engines (Dagre, ELK, TALA) for software architecture.

        Returns:
            D2 format string
        """
        lines = []

        # Title and metadata
        lines.append(f"# {self.workflow.name}")
        if self.workflow.description:
            lines.append(f"# {self.workflow.description}")
        lines.append("")

        # Direction
        lines.append("direction: down")
        lines.append("")

        # Define styles
        lines.append("classes: {")
        lines.append("  agent: {")
        lines.append('    style.fill: "#E3F2FD"')
        lines.append('    style.stroke: "#1976D2"')
        lines.append("    style.border-radius: 8")
        lines.append("  }")
        lines.append("  compute: {")
        lines.append('    style.fill: "#E8F5E9"')
        lines.append('    style.stroke: "#388E3C"')
        lines.append("  }")
        lines.append("  condition: {")
        lines.append('    style.fill: "#FFF3E0"')
        lines.append('    style.stroke: "#F57C00"')
        lines.append("    shape: diamond")
        lines.append("  }")
        lines.append("  parallel: {")
        lines.append('    style.fill: "#F3E5F5"')
        lines.append('    style.stroke: "#7B1FA2"')
        lines.append("    style.double-border: true")
        lines.append("  }")
        lines.append("  hitl: {")
        lines.append('    style.fill: "#FFEBEE"')
        lines.append('    style.stroke: "#D32F2F"')
        lines.append("    shape: oval")
        lines.append("  }")
        lines.append("  transform: {")
        lines.append('    style.fill: "#ECEFF1"')
        lines.append('    style.stroke: "#607D8B"')
        lines.append("  }")
        lines.append("}")
        lines.append("")

        # Define nodes
        for node in self._nodes:
            label = node.name or node.id
            node_class = node.node_type.replace("Node", "").lower()

            # Add node type indicator
            type_indicator = {
                "agent": "🤖",
                "compute": "⚙️",
                "condition": "❓",
                "parallel": "⫸",
                "hitl": "👤",
                "transform": "→",
            }.get(node_class, "")

            # Add metadata details for compute nodes
            details = []
            if node.metadata.get("exec"):
                details.append(f"exec: {node.metadata['exec']}")
            if node.metadata.get("handler"):
                details.append(f"handler: {node.metadata['handler']}")
            if node.metadata.get("blocked"):
                details.append(f"blocked: [{', '.join(node.metadata['blocked'])}]")
            if node.metadata.get("timeout"):
                details.append(f"timeout: {node.metadata['timeout']}s")
            if node.metadata.get("role"):
                details.append(f"role: {node.metadata['role']}")

            # Build multi-line label for D2
            if details:
                label = f"{type_indicator} {label}\\n({node_class})\\n{chr(10).join(details)}"
            else:
                label = f"{type_indicator} {label}\\n({node_class})"

            # D2 node definition with class
            lines.append(f'{node.id}: "{label}" {{')
            lines.append(f"  class: {node_class}")
            if node.description:
                # Escape quotes in description
                desc = node.description.replace('"', '\\"')
                lines.append(f'  tooltip: "{desc}"')
            lines.append("}")

        lines.append("")

        # Define edges
        for edge in self._edges:
            if edge.label:
                lines.append(f"{edge.source} -> {edge.target}: {edge.label}")
            else:
                lines.append(f"{edge.source} -> {edge.target}")

        return "\n".join(lines)

    def to_dot(self) -> str:
        """Render workflow as DOT/Graphviz format.

        Returns:
            DOT format string
        """
        import re

        def sanitize_id(s: str) -> str:
            """Sanitize string for DOT node ID."""
            # Replace non-alphanumeric with underscore
            return re.sub(r"[^a-zA-Z0-9_]", "_", s)

        def escape_label(s: Any) -> str:
            """Escape string for DOT label."""
            if not isinstance(s, str):
                s = str(s)
            return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        lines = ["digraph workflow {"]
        lines.append(f'    label="{escape_label(self.workflow.name)}"')
        lines.append("    labelloc=t")
        lines.append("    rankdir=TB")
        lines.append("    node [fontname=Helvetica]")
        lines.append("    edge [fontname=Helvetica, fontsize=10]")
        lines.append("")

        # Define nodes
        for node in self._nodes:
            style = NODE_STYLES.get(node.node_type, DEFAULT_STYLE)
            node_id = sanitize_id(node.id)

            # DOT shapes
            if node.node_type == "ConditionNode":
                shape = "diamond"
            elif node.node_type == "AgentNode":
                shape = "box"
            elif node.node_type == "ParallelNode":
                shape = "box"
            elif node.node_type == "HITLNode":
                shape = "ellipse"
            elif node.node_type == "ComputeNode":
                shape = "parallelogram"
            else:
                shape = "box"

            extra = ""
            if node.node_type == "AgentNode":
                extra = ', style="rounded,filled"'
            elif node.node_type == "ParallelNode":
                extra = ", peripheries=2, style=filled"
            else:
                extra = ", style=filled"

            # Build enhanced label with type and metadata
            label_lines = [escape_label(node.name or node.id)]
            label_lines.append(f"({node.node_type.replace('Node', '').lower()})")

            # Add metadata details
            if node.metadata.get("exec"):
                label_lines.append(f"exec: {node.metadata['exec']}")
            if node.metadata.get("handler"):
                label_lines.append(f"handler: {node.metadata['handler']}")
            if node.metadata.get("blocked"):
                label_lines.append(f"blocked: [{', '.join(node.metadata['blocked'])}]")
            if node.metadata.get("timeout"):
                label_lines.append(f"timeout: {node.metadata['timeout']}s")
            if node.metadata.get("role"):
                label_lines.append(f"role: {node.metadata['role']}")

            label = "\\n".join(label_lines)
            lines.append(
                f'    {node_id} [label="{label}", shape={shape}, '
                f'fillcolor="{style.color}"{extra}]'
            )

        lines.append("")

        # Define edges
        for edge in self._edges:
            src = sanitize_id(edge.source)
            tgt = sanitize_id(edge.target)
            attrs = []
            if edge.label:
                attrs.append(f'label="{escape_label(edge.label)}"')
            if edge.conditional:
                attrs.append("style=dashed")
            attr_str = ", ".join(attrs)
            lines.append(f"    {src} -> {tgt} [{attr_str}]")

        lines.append("}")

        return "\n".join(lines)

    def to_svg(
        self,
        output_path: Optional[str] = None,
        backend: RenderBackend = RenderBackend.AUTO,
    ) -> str:
        """Render workflow as SVG.

        Tries backends in order: D2 > Kroki > Graphviz > Matplotlib

        Args:
            output_path: Optional path to save SVG file
            backend: Rendering backend to use

        Returns:
            SVG string
        """
        if backend == RenderBackend.AUTO:
            backend = _detect_best_backend()

        if backend == RenderBackend.D2:
            return self._to_svg_d2(output_path)
        elif backend == RenderBackend.KROKI:
            return self._to_svg_kroki(output_path)
        elif backend == RenderBackend.GRAPHVIZ:
            return self._to_svg_graphviz(output_path)
        elif backend == RenderBackend.MATPLOTLIB:
            return self._to_svg_matplotlib(output_path)
        else:
            # Try in order
            for method in [
                self._to_svg_d2,
                self._to_svg_kroki,
                self._to_svg_graphviz,
                self._to_svg_matplotlib,
            ]:
                try:
                    return method(output_path)
                except (ImportError, FileNotFoundError, RuntimeError):
                    continue
            raise ImportError("No SVG rendering backend available")

    def _to_svg_d2(self, output_path: Optional[str] = None) -> str:
        """Render SVG using D2 (Terrastruct)."""
        if not _check_command("d2"):
            raise FileNotFoundError("D2 not installed. Install: brew install d2")

        d2_content = self.to_d2()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".d2", delete=False) as f:
            f.write(d2_content)
            d2_file = f.name

        svg_file = output_path or tempfile.mktemp(suffix=".svg")

        try:
            result = subprocess.run(
                ["d2", "--layout=dagre", d2_file, svg_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"D2 error: {result.stderr}")

            with open(svg_file, "r") as f:
                svg_content = f.read()

            return svg_content
        finally:
            Path(d2_file).unlink(missing_ok=True)
            if not output_path:
                Path(svg_file).unlink(missing_ok=True)

    def _to_svg_kroki(
        self,
        output_path: Optional[str] = None,
        diagram_type: str = "graphviz",
    ) -> str:
        """Render SVG using Kroki unified API.

        Uses the free public Kroki instance (kroki.io) or self-hosted.
        No local installation required - just needs network access.
        """
        # Get diagram source based on type
        if diagram_type == "graphviz":
            source = self.to_dot()
        elif diagram_type == "mermaid":
            source = self.to_mermaid()
        elif diagram_type == "plantuml":
            source = self.to_plantuml()
        elif diagram_type == "d2":
            source = self.to_d2()
        else:
            source = self.to_dot()
            diagram_type = "graphviz"

        # Encode for Kroki API
        compressed = zlib.compress(source.encode("utf-8"), 9)
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")

        # Request SVG from Kroki
        url = f"{KROKI_URL}/{diagram_type}/svg/{encoded}"

        try:
            request = _validated_kroki_request(url, "image/svg+xml")
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urlopen(request, timeout=30) as response:  # noqa: S310
                svg_content = response.read().decode("utf-8")

            if output_path:
                with open(output_path, "w") as f:
                    f.write(svg_content)

            return svg_content

        except URLError as e:
            raise RuntimeError(f"Kroki API error: {e}. Check network or use local backend.")

    def _to_svg_matplotlib(self, output_path: Optional[str] = None) -> str:
        """Render SVG using matplotlib and networkx."""
        import io

        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import networkx as nx

        # Build networkx graph
        G = nx.DiGraph()

        for node in self._nodes:
            style = NODE_STYLES.get(node.node_type, DEFAULT_STYLE)
            G.add_node(node.id, label=node.name, color=style.color, node_type=node.node_type)

        for edge in self._edges:
            G.add_edge(edge.source, edge.target, label=edge.label or "")

        # Layout
        try:
            pos = nx.planar_layout(G)
        except nx.NetworkXException:
            pos = nx.spring_layout(G, k=2, iterations=50)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_title(self.workflow.name, fontsize=14, fontweight="bold")

        # Draw nodes with colors
        node_colors = [G.nodes[n].get("color", "#FFFFFF") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="#666666", arrows=True, ax=ax)

        # Edge labels
        edge_labels = {(e.source, e.target): e.label for e in self._edges if e.label}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

        ax.axis("off")
        plt.tight_layout()

        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format="svg", bbox_inches="tight")
        buffer.seek(0)
        svg_content = buffer.getvalue().decode("utf-8")

        if output_path:
            with open(output_path, "w") as f:
                f.write(svg_content)

        plt.close(fig)
        return svg_content

    def _to_svg_graphviz(self, output_path: Optional[str] = None) -> str:
        """Render SVG using Graphviz (fallback)."""
        dot_content = self.to_dot()

        # Try to use graphviz
        try:
            result = subprocess.run(
                ["dot", "-Tsvg"],
                input=dot_content,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                svg_content = result.stdout
                if output_path:
                    with open(output_path, "w") as f:
                        f.write(svg_content)
                return svg_content
            else:
                raise RuntimeError(f"Graphviz error: {result.stderr}")
        except FileNotFoundError:
            raise ImportError(
                "SVG rendering requires matplotlib+networkx or graphviz. "
                "Install with: pip install matplotlib networkx  OR  brew install graphviz"
            )

    def to_png(
        self,
        output_path: str,
        backend: RenderBackend = RenderBackend.AUTO,
    ) -> None:
        """Render workflow as PNG.

        Args:
            output_path: Path to save PNG file
            backend: Rendering backend to use
        """
        if backend == RenderBackend.AUTO:
            backend = _detect_best_backend()

        if backend == RenderBackend.D2:
            self._to_png_d2(output_path)
        elif backend == RenderBackend.KROKI:
            self._to_png_kroki(output_path)
        elif backend == RenderBackend.GRAPHVIZ:
            self._to_png_graphviz(output_path)
        else:
            # Try in order
            for method in [self._to_png_d2, self._to_png_kroki, self._to_png_graphviz]:
                try:
                    method(output_path)
                    return
                except (ImportError, FileNotFoundError, RuntimeError):
                    continue
            raise ImportError("No PNG rendering backend available")

    def _to_png_d2(self, output_path: str) -> None:
        """Render PNG using D2."""
        if not _check_command("d2"):
            raise FileNotFoundError("D2 not installed")

        d2_content = self.to_d2()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".d2", delete=False) as f:
            f.write(d2_content)
            d2_file = f.name

        try:
            result = subprocess.run(
                ["d2", "--layout=dagre", d2_file, output_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"D2 error: {result.stderr}")
        finally:
            Path(d2_file).unlink(missing_ok=True)

    def _to_png_kroki(self, output_path: str, diagram_type: str = "graphviz") -> None:
        """Render PNG using Kroki API."""
        if diagram_type == "graphviz":
            source = self.to_dot()
        elif diagram_type == "d2":
            source = self.to_d2()
        else:
            source = self.to_dot()
            diagram_type = "graphviz"

        compressed = zlib.compress(source.encode("utf-8"), 9)
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")

        url = f"{KROKI_URL}/{diagram_type}/png/{encoded}"

        try:
            request = _validated_kroki_request(url, "image/png")
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urlopen(request, timeout=30) as response:  # noqa: S310
                png_content = response.read()

            with open(output_path, "wb") as f:
                f.write(png_content)

        except URLError as e:
            raise RuntimeError(f"Kroki API error: {e}")

    def _to_png_graphviz(self, output_path: str) -> None:
        """Render PNG using Graphviz."""
        if not _check_command("dot"):
            raise FileNotFoundError("Graphviz not installed")

        dot_content = self.to_dot()

        result = subprocess.run(
            ["dot", "-Tpng", "-o", output_path],
            input=dot_content,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Graphviz error: {result.stderr}")

    def render(
        self,
        format: OutputFormat = OutputFormat.ASCII,
        output_path: Optional[str] = None,
        backend: RenderBackend = RenderBackend.AUTO,
    ) -> str:
        """Render workflow in specified format.

        Args:
            format: Output format
            output_path: Optional path to save output
            backend: Rendering backend for SVG/PNG

        Returns:
            Rendered content as string
        """
        if format == OutputFormat.ASCII:
            content = self.to_ascii()
        elif format == OutputFormat.MERMAID:
            content = self.to_mermaid()
        elif format == OutputFormat.PLANTUML:
            content = self.to_plantuml()
        elif format == OutputFormat.DOT:
            content = self.to_dot()
        elif format == OutputFormat.D2:
            content = self.to_d2()
        elif format == OutputFormat.SVG:
            content = self.to_svg(output_path, backend)
            return content  # Already saved if path provided
        elif format == OutputFormat.PNG:
            if not output_path:
                raise ValueError("PNG format requires output_path")
            self.to_png(output_path, backend)
            return f"PNG saved to {output_path}"
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            with open(output_path, "w") as f:
                f.write(content)

        return content


def visualize_workflow(
    workflow: "WorkflowDefinition",
    format: OutputFormat = OutputFormat.ASCII,
    output_path: Optional[str] = None,
) -> str:
    """Convenience function to visualize a workflow.

    Args:
        workflow: The workflow to visualize
        format: Output format (ascii, mermaid, plantuml, dot, svg, png)
        output_path: Optional path to save output

    Returns:
        Rendered content as string
    """
    viz = WorkflowVisualizer(workflow)
    return viz.render(format, output_path)


def get_available_backends() -> Dict[str, bool]:
    """Check which rendering backends are available.

    Returns:
        Dictionary of backend name -> availability
    """
    return {
        "d2": _check_command("d2"),
        "graphviz": _check_command("dot"),
        "mermaid-cli": _check_command("mmdc"),
        "plantuml": _check_command("plantuml"),
        "matplotlib": _has_matplotlib(),
        "kroki": True,  # Always available (cloud API)
        "ascii": True,  # Always available (no deps)
    }


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib
        import networkx

        return True
    except ImportError:
        return False


def print_backends() -> None:
    """Print available rendering backends."""
    backends = get_available_backends()
    print("Available Rendering Backends:")
    print("-" * 40)
    for name, available in backends.items():
        status = "[OK]" if available else "[NOT INSTALLED]"
        print(f"  {name:15} {status}")


__all__ = [
    # Enums
    "OutputFormat",
    "RenderBackend",
    # Data classes
    "NodeStyle",
    "DAGNode",
    "DAGEdge",
    # Main visualizer
    "WorkflowVisualizer",
    # Convenience functions
    "visualize_workflow",
    "get_available_backends",
    "print_backends",
    # Constants
    "KROKI_URL",
]
